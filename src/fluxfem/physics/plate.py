from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .lumped import (
    ArrayBackend,
    MatrixBackend,
    MatrixFormat,
    _as_array_backend,
    _resolve_array_backend,
    _resolve_matrix_format,
    _sparse_from_coo,
)


PLATE_DOF_PER_NODE = 3


@dataclass(frozen=True)
class PlateSection:
    """Material and thickness properties for an isotropic Mindlin plate."""

    E: float
    nu: float
    thickness: float
    shear_correction: float = 5.0 / 6.0
    rho: float | None = None


def plate_node_dofs(
    nodes: Sequence[int] | np.ndarray,
    components: Sequence[int] | str = "wrxry",
) -> np.ndarray:
    """
    Return flattened 3-DOF plate indices for nodes.

    Component names are ``w``, ``rx``, and ``ry``.
    """
    nodes_arr = np.asarray(nodes, dtype=int).reshape(-1)
    comp_map = {"w": 0, "rx": 1, "ry": 2, "x": 1, "y": 2}
    if isinstance(components, str):
        text = components.lower().replace(",", " ").replace("_", " ")
        if " " in text:
            tokens = [tok for tok in text.split() if tok]
        else:
            tokens = []
            i = 0
            while i < len(text):
                if text[i] == "w":
                    tokens.append("w")
                    i += 1
                else:
                    tokens.append(text[i : i + 2])
                    i += 2
        comps = np.asarray([comp_map[tok] for tok in tokens], dtype=int)
    else:
        comps = np.asarray(list(components), dtype=int)
    if np.any(comps < 0) or np.any(comps >= PLATE_DOF_PER_NODE):
        raise ValueError("plate components must be in [0, 2].")
    return np.asarray([PLATE_DOF_PER_NODE * int(n) + int(c) for n in nodes_arr for c in comps], dtype=int)


def structured_plate_grid(
    *,
    nx: int,
    ny: int,
    length_x: float,
    length_y: float,
    origin: Sequence[float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a structured rectangular Q4 plate mesh in the x-y plane."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    if length_x <= 0.0 or length_y <= 0.0:
        raise ValueError("length_x and length_y must be positive.")
    origin_arr = np.asarray(origin, dtype=float)
    if origin_arr.shape != (2,):
        raise ValueError("origin must have shape (2,).")

    xs = origin_arr[0] + np.linspace(0.0, float(length_x), int(nx) + 1)
    ys = origin_arr[1] + np.linspace(0.0, float(length_y), int(ny) + 1)
    coords = np.asarray([[x, y] for y in ys for x in xs], dtype=float)

    conn: list[list[int]] = []
    stride = int(nx) + 1
    for j in range(int(ny)):
        for i in range(int(nx)):
            n0 = j * stride + i
            conn.append([n0, n0 + 1, n0 + stride + 1, n0 + stride])
    return coords, np.asarray(conn, dtype=int)


def plate_element_dofs(conn: np.ndarray) -> np.ndarray:
    """Return element-to-DOF connectivity for 4-node plate elements."""
    conn_arr = np.asarray(conn, dtype=int)
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")
    elem_dofs = np.empty((conn_arr.shape[0], 12), dtype=int)
    for e, nodes in enumerate(conn_arr):
        for a, node in enumerate(nodes):
            elem_dofs[e, 3 * a : 3 * a + 3] = PLATE_DOF_PER_NODE * int(node) + np.arange(3)
    return elem_dofs


def _q4_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )
    dN_dxi = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [+(1.0 - eta), -(1.0 + xi)],
            [+(1.0 + eta), +(1.0 + xi)],
            [-(1.0 + eta), +(1.0 - xi)],
        ],
        dtype=float,
    )
    return N, dN_dxi


def _q4_gradients(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    N, dN_dxi = _q4_shape(float(xi), float(eta))
    J = dN_dxi.T @ coords
    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError("plate element has non-positive Jacobian; check Q4 node ordering.")
    dN_dx = dN_dxi @ np.linalg.inv(J)
    return N, dN_dx, detJ


def _plate_constitutive(section: PlateSection) -> tuple[np.ndarray, np.ndarray]:
    E = float(section.E)
    nu = float(section.nu)
    t = float(section.thickness)
    kappa = float(section.shear_correction)
    if E <= 0.0:
        raise ValueError("E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must lie in (-1, 0.5).")
    if t <= 0.0:
        raise ValueError("thickness must be positive.")
    if kappa <= 0.0:
        raise ValueError("shear_correction must be positive.")

    Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float,
    )
    G = E / (2.0 * (1.0 + nu))
    Ds = kappa * G * t * np.eye(2, dtype=float)
    return Db, Ds


def _bending_B(dN_dx: np.ndarray) -> np.ndarray:
    B = np.zeros((3, 12), dtype=float)
    for a in range(4):
        c = 3 * a
        dNx, dNy = dN_dx[a]
        B[0, c + 1] = dNx
        B[1, c + 2] = dNy
        B[2, c + 1] = dNy
        B[2, c + 2] = dNx
    return B


def _shear_B(N: np.ndarray, dN_dx: np.ndarray) -> np.ndarray:
    B = np.zeros((2, 12), dtype=float)
    for a in range(4):
        c = 3 * a
        dNx, dNy = dN_dx[a]
        B[0, c + 0] = -dNx
        B[0, c + 1] = N[a]
        B[1, c + 0] = -dNy
        B[1, c + 2] = N[a]
    return B


def mindlin_plate_element_stiffness(coords: np.ndarray, section: PlateSection) -> np.ndarray:
    """
    12x12 Q4 Reissner-Mindlin plate stiffness.

    DOFs per node are ``[w, theta_x, theta_y]``. Bending uses 2x2 integration;
    transverse shear uses one-point reduced integration.
    """
    x = np.asarray(coords, dtype=float)
    if x.shape != (4, 2):
        raise ValueError("coords must have shape (4, 2).")
    Db, Ds = _plate_constitutive(section)
    K = np.zeros((12, 12), dtype=float)

    g = 1.0 / np.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            _N, dN_dx, detJ = _q4_gradients(x, xi, eta)
            Bb = _bending_B(dN_dx)
            K += Bb.T @ Db @ Bb * detJ

    N, dN_dx, detJ = _q4_gradients(x, 0.0, 0.0)
    Bs = _shear_B(N, dN_dx)
    K += Bs.T @ Ds @ Bs * detJ * 4.0
    return 0.5 * (K + K.T)


def mindlin_plate_element_uniform_load(coords: np.ndarray, load: float) -> np.ndarray:
    """12-vector of equivalent nodal loads for a uniform transverse load."""
    x = np.asarray(coords, dtype=float)
    if x.shape != (4, 2):
        raise ValueError("coords must have shape (4, 2).")
    f = np.zeros((12,), dtype=float)
    g = 1.0 / np.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            N, _dN_dx, detJ = _q4_gradients(x, xi, eta)
            for a in range(4):
                f[3 * a] += float(load) * N[a] * detJ
    return f


def assemble_mindlin_plate_stiffness(
    coords: np.ndarray,
    conn: np.ndarray,
    section: PlateSection,
    *,
    format: MatrixFormat | None = None,
    backend: MatrixBackend | None = None,
):
    """Assemble stiffness for Q4 Reissner-Mindlin plate elements."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError("coords must have shape (n_nodes, 2).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")

    elem_dofs = plate_element_dofs(conn_arr)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e, nodes in enumerate(conn_arr):
        Ke = mindlin_plate_element_stiffness(coords_arr[nodes], section)
        dofs = elem_dofs[e]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(Ke.reshape(-1).tolist())

    n_dofs = PLATE_DOF_PER_NODE * coords_arr.shape[0]
    return _sparse_from_coo(rows, cols, data, n_dofs, format=_resolve_matrix_format(format, backend))


def assemble_mindlin_plate_uniform_load(
    coords: np.ndarray,
    conn: np.ndarray,
    load: float,
    *,
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble equivalent nodal load vector for uniform transverse pressure/load."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError("coords must have shape (n_nodes, 2).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")

    out = np.zeros(PLATE_DOF_PER_NODE * coords_arr.shape[0], dtype=float)
    elem_dofs = plate_element_dofs(conn_arr)
    for e, nodes in enumerate(conn_arr):
        fe = mindlin_plate_element_uniform_load(coords_arr[nodes], load)
        np.add.at(out, elem_dofs[e], fe)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


def assemble_mindlin_plate_point_load(
    n_nodes: int,
    node: int,
    *,
    force: float = 0.0,
    moments: Sequence[float] | np.ndarray = (0.0, 0.0),
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble a dense nodal load vector for one plate node."""
    return assemble_mindlin_plate_point_loads(
        n_nodes,
        [node],
        forces=[force],
        moments=[moments],
        array_backend=_resolve_array_backend(array_backend, backend),
    )


def assemble_mindlin_plate_point_loads(
    n_nodes: int,
    nodes: Sequence[int] | np.ndarray,
    *,
    forces: Sequence[float] | np.ndarray | None = None,
    moments: Sequence[Sequence[float]] | np.ndarray | None = None,
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble dense nodal transverse force and rotation-moment loads."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    nodes_arr = np.asarray(nodes, dtype=int).reshape(-1)
    if nodes_arr.size == 0:
        raise ValueError("nodes must contain at least one node.")
    if np.any(nodes_arr < 0) or np.any(nodes_arr >= int(n_nodes)):
        raise ValueError("nodes contains an index outside n_nodes.")

    if forces is None:
        force_arr = np.zeros(nodes_arr.size, dtype=float)
    else:
        force_arr = np.asarray(forces, dtype=float).reshape(-1)
        if force_arr.size != nodes_arr.size:
            raise ValueError(f"forces must have shape ({nodes_arr.size},).")

    if moments is None:
        moment_arr = np.zeros((nodes_arr.size, 2), dtype=float)
    else:
        moment_arr = np.asarray(moments, dtype=float)
        if moment_arr.shape == (2,):
            if nodes_arr.size != 1:
                raise ValueError("moments with shape (2,) is only valid for one node.")
            moment_arr = moment_arr.reshape(1, 2)
        if moment_arr.shape != (nodes_arr.size, 2):
            raise ValueError(f"moments must have shape ({nodes_arr.size}, 2).")

    out = np.zeros(PLATE_DOF_PER_NODE * int(n_nodes), dtype=float)
    for node, force, moment_vec in zip(nodes_arr, force_arr, moment_arr, strict=True):
        dofs = PLATE_DOF_PER_NODE * int(node) + np.arange(3)
        values = np.array([force, moment_vec[0], moment_vec[1]], dtype=float)
        np.add.at(out, dofs, values)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


__all__ = [
    "PLATE_DOF_PER_NODE",
    "PlateSection",
    "assemble_mindlin_plate_point_load",
    "assemble_mindlin_plate_point_loads",
    "assemble_mindlin_plate_stiffness",
    "assemble_mindlin_plate_uniform_load",
    "mindlin_plate_element_stiffness",
    "mindlin_plate_element_uniform_load",
    "plate_element_dofs",
    "plate_node_dofs",
    "structured_plate_grid",
]

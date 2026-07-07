from __future__ import annotations

from dataclasses import dataclass
import os
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
SHELL_DOF_PER_NODE = 6


@dataclass(frozen=True)
class PlateSection:
    """Material and thickness properties for an isotropic Mindlin plate."""

    E: float
    nu: float
    thickness: float
    shear_correction: float = 5.0 / 6.0
    rho: float | None = None


@dataclass(frozen=True)
class ShellSection:
    """Material and thickness properties for a flat isotropic Reissner-Mindlin shell."""

    E: float
    nu: float
    thickness: float
    shear_correction: float = 5.0 / 6.0
    drilling_stiffness: float = 1.0e-8
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


def shell_node_dofs(
    nodes: Sequence[int] | np.ndarray,
    components: Sequence[int] | str = "uxuyuzrxryrz",
) -> np.ndarray:
    """
    Return flattened 6-DOF shell indices for nodes.

    Component names are ``ux``, ``uy``, ``uz``, ``rx``, ``ry``, and ``rz``.
    """
    nodes_arr = np.asarray(nodes, dtype=int).reshape(-1)
    comp_map = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}
    if isinstance(components, str):
        text = components.lower().replace(",", " ").replace("_", " ")
        if " " in text:
            tokens = [tok for tok in text.split() if tok]
        else:
            if len(text) % 2 != 0:
                raise ValueError("Shell component string must use two-letter tokens: ux, uy, uz, rx, ry, rz.")
            tokens = [text[i : i + 2] for i in range(0, len(text), 2)]
        comps = np.asarray([comp_map[tok] for tok in tokens], dtype=int)
    else:
        comps = np.asarray(list(components), dtype=int)
    if np.any(comps < 0) or np.any(comps >= SHELL_DOF_PER_NODE):
        raise ValueError("shell components must be in [0, 5].")
    return np.asarray([SHELL_DOF_PER_NODE * int(n) + int(c) for n in nodes_arr for c in comps], dtype=int)


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


def shell_element_dofs(conn: np.ndarray) -> np.ndarray:
    """Return element-to-DOF connectivity for 4-node flat shell elements."""
    conn_arr = np.asarray(conn, dtype=int)
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")
    elem_dofs = np.empty((conn_arr.shape[0], 24), dtype=int)
    for e, nodes in enumerate(conn_arr):
        for a, node in enumerate(nodes):
            elem_dofs[e, 6 * a : 6 * a + 6] = SHELL_DOF_PER_NODE * int(node) + np.arange(6)
    return elem_dofs


def shell_solid_translational_tie_dofs(
    shell_coords: np.ndarray,
    solid_coords: np.ndarray,
    *,
    shell_nodes: Sequence[int] | np.ndarray | None = None,
    solid_nodes: Sequence[int] | np.ndarray | None = None,
    tol: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match coincident shell and solid nodes and return translational tie DOFs.

    The returned tuple is ``(matched_shell_nodes, matched_solid_nodes,
    shell_dofs, solid_dofs)``. Shell translational components ``ux,uy,uz`` are
    tied to solid vector DOFs ``x,y,z`` with the same physical coordinates.
    """
    xs = _surface_coords3(shell_coords)
    xg = _surface_coords3(solid_coords)
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    s_nodes = np.arange(xs.shape[0], dtype=int) if shell_nodes is None else np.asarray(shell_nodes, dtype=int).reshape(-1)
    g_nodes = np.arange(xg.shape[0], dtype=int) if solid_nodes is None else np.asarray(solid_nodes, dtype=int).reshape(-1)
    if s_nodes.size == 0 or g_nodes.size == 0:
        raise ValueError("shell_nodes and solid_nodes must contain at least one node.")
    if np.any(s_nodes < 0) or np.any(s_nodes >= xs.shape[0]):
        raise ValueError("shell_nodes contains an index outside shell_coords.")
    if np.any(g_nodes < 0) or np.any(g_nodes >= xg.shape[0]):
        raise ValueError("solid_nodes contains an index outside solid_coords.")

    matched_shell: list[int] = []
    matched_solid: list[int] = []
    used: set[int] = set()
    for sn in s_nodes.tolist():
        distances = np.linalg.norm(xg[g_nodes] - xs[int(sn)], axis=1)
        local = int(np.argmin(distances))
        if float(distances[local]) > float(tol):
            raise ValueError(f"no coincident solid node found for shell node {sn}.")
        gn = int(g_nodes[local])
        if gn in used:
            raise ValueError("multiple shell nodes matched the same solid node.")
        used.add(gn)
        matched_shell.append(int(sn))
        matched_solid.append(gn)

    shell_match = np.asarray(matched_shell, dtype=int)
    solid_match = np.asarray(matched_solid, dtype=int)
    return (
        shell_match,
        solid_match,
        shell_node_dofs(shell_match, "uxuyuz"),
        np.asarray([3 * int(n) + c for n in solid_match for c in range(3)], dtype=int),
    )


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
    J = coords.T @ dN_dxi
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


def _shell_section_as_plate(section: ShellSection) -> PlateSection:
    return PlateSection(
        E=section.E,
        nu=section.nu,
        thickness=section.thickness,
        shear_correction=section.shear_correction,
        rho=section.rho,
    )


def _membrane_constitutive(section: ShellSection) -> np.ndarray:
    E = float(section.E)
    nu = float(section.nu)
    t = float(section.thickness)
    if E <= 0.0:
        raise ValueError("E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must lie in (-1, 0.5).")
    if t <= 0.0:
        raise ValueError("thickness must be positive.")
    return (E * t / (1.0 - nu**2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float,
    )


def _membrane_B(dN_dx: np.ndarray) -> np.ndarray:
    B = np.zeros((3, 8), dtype=float)
    for a in range(4):
        c = 2 * a
        dNx, dNy = dN_dx[a]
        B[0, c + 0] = dNx
        B[1, c + 1] = dNy
        B[2, c + 0] = dNy
        B[2, c + 1] = dNx
    return B


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


def flat_shell_element_stiffness(coords: np.ndarray, section: ShellSection) -> np.ndarray:
    """
    24x24 flat Q4 Reissner-Mindlin shell stiffness in the x-y plane.

    DOFs per node are ``[ux, uy, uz, theta_x, theta_y, theta_z]``. The membrane
    block uses plane-stress Q4 integration, the bending/shear block reuses the
    Mindlin plate stiffness, and drilling ``theta_z`` gets a small diagonal
    stabilization controlled by ``section.drilling_stiffness``.
    """
    x = np.asarray(coords, dtype=float)
    if x.shape != (4, 2):
        raise ValueError("coords must have shape (4, 2).")
    Dm = _membrane_constitutive(section)
    Km = np.zeros((8, 8), dtype=float)
    area = 0.0
    g = 1.0 / np.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            _N, dN_dx, detJ = _q4_gradients(x, xi, eta)
            Bm = _membrane_B(dN_dx)
            Km += Bm.T @ Dm @ Bm * detJ
            area += detJ

    Kp = mindlin_plate_element_stiffness(x, _shell_section_as_plate(section))
    K = np.zeros((24, 24), dtype=float)
    for a in range(4):
        for b in range(4):
            K[6 * a + 0, 6 * b + 0] += Km[2 * a + 0, 2 * b + 0]
            K[6 * a + 0, 6 * b + 1] += Km[2 * a + 0, 2 * b + 1]
            K[6 * a + 1, 6 * b + 0] += Km[2 * a + 1, 2 * b + 0]
            K[6 * a + 1, 6 * b + 1] += Km[2 * a + 1, 2 * b + 1]

    # Plate slope convention is [w, dw/dx, dw/dy]. Shell rotations are physical
    # local rotations [rx, ry, rz], hence dw/dx = -ry and dw/dy = rx.
    P = np.zeros((12, 24), dtype=float)
    for a in range(4):
        P[3 * a + 0, 6 * a + 2] = 1.0
        P[3 * a + 1, 6 * a + 4] = -1.0
        P[3 * a + 2, 6 * a + 3] = 1.0
    K += P.T @ Kp @ P

    drill = float(section.drilling_stiffness)
    if drill < 0.0:
        raise ValueError("drilling_stiffness must be nonnegative.")
    if drill > 0.0:
        scale = float(section.E) * float(section.thickness) * max(area, 0.0) * drill / 4.0
        for a in range(4):
            K[6 * a + 5, 6 * a + 5] += scale
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


def flat_shell_element_uniform_load(coords: np.ndarray, load: Sequence[float] | np.ndarray) -> np.ndarray:
    """24-vector of equivalent nodal loads for a uniform global shell load."""
    x = np.asarray(coords, dtype=float)
    if x.shape != (4, 2):
        raise ValueError("coords must have shape (4, 2).")
    q = np.asarray(load, dtype=float).reshape(-1)
    if q.shape != (3,):
        raise ValueError("load must have shape (3,) for [qx, qy, qz].")
    f = np.zeros((24,), dtype=float)
    g = 1.0 / np.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            N, _dN_dx, detJ = _q4_gradients(x, xi, eta)
            for a in range(4):
                f[6 * a + 0 : 6 * a + 3] += q * N[a] * detJ
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


def assemble_flat_shell_stiffness(
    coords: np.ndarray,
    conn: np.ndarray,
    section: ShellSection,
    *,
    format: MatrixFormat | None = None,
    backend: MatrixBackend | None = None,
):
    """Assemble stiffness for flat Q4 Reissner-Mindlin shell elements."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError("coords must have shape (n_nodes, 2).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")

    elem_dofs = shell_element_dofs(conn_arr)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e, nodes in enumerate(conn_arr):
        Ke = flat_shell_element_stiffness(coords_arr[nodes], section)
        dofs = elem_dofs[e]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(Ke.reshape(-1).tolist())

    n_dofs = SHELL_DOF_PER_NODE * coords_arr.shape[0]
    return _sparse_from_coo(rows, cols, data, n_dofs, format=_resolve_matrix_format(format, backend))


def assemble_shell_stiffness(
    coords: np.ndarray,
    conn: np.ndarray,
    section: ShellSection,
    *,
    format: MatrixFormat | None = None,
    backend: MatrixBackend | None = None,
):
    """Assemble stiffness for Q4 Reissner-Mindlin shell elements in 2D or 3D coordinates."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] not in (2, 3):
        raise ValueError("coords must have shape (n_nodes, 2) or (n_nodes, 3).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")

    elem_dofs = shell_element_dofs(conn_arr)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e, nodes in enumerate(conn_arr):
        Ke = shell_element_stiffness_global(coords_arr[nodes], section)
        dofs = elem_dofs[e]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(Ke.reshape(-1).tolist())

    n_dofs = SHELL_DOF_PER_NODE * coords_arr.shape[0]
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


def assemble_flat_shell_uniform_load(
    coords: np.ndarray,
    conn: np.ndarray,
    load: Sequence[float] | np.ndarray,
    *,
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble equivalent nodal load vector for a uniform global shell load."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError("coords must have shape (n_nodes, 2).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")

    out = np.zeros(SHELL_DOF_PER_NODE * coords_arr.shape[0], dtype=float)
    elem_dofs = shell_element_dofs(conn_arr)
    for e, nodes in enumerate(conn_arr):
        fe = flat_shell_element_uniform_load(coords_arr[nodes], load)
        np.add.at(out, elem_dofs[e], fe)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


def assemble_shell_uniform_load(
    coords: np.ndarray,
    conn: np.ndarray,
    load: Sequence[float] | np.ndarray,
    *,
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble equivalent nodal load vector for a uniform global shell load."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] not in (2, 3):
        raise ValueError("coords must have shape (n_nodes, 2) or (n_nodes, 3).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")

    out = np.zeros(SHELL_DOF_PER_NODE * coords_arr.shape[0], dtype=float)
    elem_dofs = shell_element_dofs(conn_arr)
    for e, nodes in enumerate(conn_arr):
        fe = shell_element_uniform_load_global(coords_arr[nodes], load)
        np.add.at(out, elem_dofs[e], fe)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


def assemble_flat_shell_point_load(
    n_nodes: int,
    node: int,
    *,
    force: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    moment: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble a dense nodal load vector for one flat shell node."""
    return assemble_flat_shell_point_loads(
        n_nodes,
        [node],
        forces=[force],
        moments=[moment],
        array_backend=_resolve_array_backend(array_backend, backend),
    )


def assemble_flat_shell_point_loads(
    n_nodes: int,
    nodes: Sequence[int] | np.ndarray,
    *,
    forces: Sequence[Sequence[float]] | np.ndarray | None = None,
    moments: Sequence[Sequence[float]] | np.ndarray | None = None,
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble dense nodal force/moment loads for flat shell nodes."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    nodes_arr = np.asarray(nodes, dtype=int).reshape(-1)
    if nodes_arr.size == 0:
        raise ValueError("nodes must contain at least one node.")
    if np.any(nodes_arr < 0) or np.any(nodes_arr >= int(n_nodes)):
        raise ValueError("nodes contains an index outside n_nodes.")

    if forces is None:
        force_arr = np.zeros((nodes_arr.size, 3), dtype=float)
    else:
        force_arr = np.asarray(forces, dtype=float)
        if force_arr.shape == (3,):
            if nodes_arr.size != 1:
                raise ValueError("forces with shape (3,) is only valid for one node.")
            force_arr = force_arr.reshape(1, 3)
        if force_arr.shape != (nodes_arr.size, 3):
            raise ValueError(f"forces must have shape ({nodes_arr.size}, 3).")

    if moments is None:
        moment_arr = np.zeros((nodes_arr.size, 3), dtype=float)
    else:
        moment_arr = np.asarray(moments, dtype=float)
        if moment_arr.shape == (3,):
            if nodes_arr.size != 1:
                raise ValueError("moments with shape (3,) is only valid for one node.")
            moment_arr = moment_arr.reshape(1, 3)
        if moment_arr.shape != (nodes_arr.size, 3):
            raise ValueError(f"moments must have shape ({nodes_arr.size}, 3).")

    out = np.zeros(SHELL_DOF_PER_NODE * int(n_nodes), dtype=float)
    for node, force_vec, moment_vec in zip(nodes_arr, force_arr, moment_arr, strict=True):
        dofs = SHELL_DOF_PER_NODE * int(node) + np.arange(6)
        values = np.concatenate([force_vec, moment_vec])
        np.add.at(out, dofs, values)
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


def _surface_coords3(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] not in (2, 3):
        raise ValueError("coords must have shape (n_nodes, 2) or (n_nodes, 3).")
    if c.shape[1] == 3:
        return c
    out = np.zeros((c.shape[0], 3), dtype=float)
    out[:, :2] = c
    return out


def shell_element_frame(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a local shell frame for one Q4 element.

    Returns ``(R, local_coords)`` where rows of ``R`` are local unit axes in
    global coordinates and ``local = R @ (global - origin)``.
    """
    x = _surface_coords3(coords)
    if x.shape != (4, 3):
        raise ValueError("coords must have shape (4, 2) or (4, 3).")
    ex_vec = x[1] - x[0]
    ex_norm = float(np.linalg.norm(ex_vec))
    if ex_norm <= 0.0:
        raise ValueError("shell element has coincident first and second nodes.")
    ex = ex_vec / ex_norm

    ey_seed = x[3] - x[0]
    ez = np.cross(ex, ey_seed)
    ez_norm = float(np.linalg.norm(ez))
    if ez_norm <= 1.0e-14:
        ey_seed = x[2] - x[0]
        ez = np.cross(ex, ey_seed)
        ez_norm = float(np.linalg.norm(ez))
    if ez_norm <= 1.0e-14:
        raise ValueError("shell element local frame is degenerate.")
    ez = ez / ez_norm
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)
    R = np.vstack([ex, ey, ez])
    local = (R @ (x - x[0]).T).T
    return R, local[:, :2]


def _shell_transform(R: np.ndarray) -> np.ndarray:
    T = np.zeros((24, 24), dtype=float)
    for a in range(4):
        c = 6 * a
        T[c : c + 3, c : c + 3] = R
        T[c + 3 : c + 6, c + 3 : c + 6] = R
    return T


def shell_element_stiffness_global(coords: np.ndarray, section: ShellSection) -> np.ndarray:
    """
    24x24 Q4 Reissner-Mindlin shell stiffness for 2D or 3D shell coordinates.

    For 3D coordinates, the element is treated as a planar Q4 shell in its local
    frame and transformed back to global translational/rotational DOFs.
    """
    R, local = shell_element_frame(coords)
    K_local = flat_shell_element_stiffness(local, section)
    T = _shell_transform(R)
    return 0.5 * (T.T @ K_local @ T + (T.T @ K_local @ T).T)


def shell_element_uniform_load_global(coords: np.ndarray, load: Sequence[float] | np.ndarray) -> np.ndarray:
    """24-vector of equivalent nodal loads for a uniform global shell load."""
    q_global = np.asarray(load, dtype=float).reshape(-1)
    if q_global.shape != (3,):
        raise ValueError("load must have shape (3,) for [qx, qy, qz].")
    R, local = shell_element_frame(coords)
    q_local = R @ q_global
    f_local = flat_shell_element_uniform_load(local, q_local)
    T = _shell_transform(R)
    return T.T @ f_local


def _surface_dataarray(name: str, data: np.ndarray, ncomp: int = 1) -> str:
    arr = np.asarray(data, dtype=float)
    tuples = arr.reshape(-1, ncomp)
    comp_attr = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
    lines = [f'<DataArray type="Float64" Name="{name}" format="ascii"{comp_attr}>']
    lines.extend(" ".join(f"{float(v):.16e}" for v in row) for row in tuples)
    lines.append("</DataArray>")
    return "\n".join(lines)


def write_q4_surface_vtu(
    coords: np.ndarray,
    conn: np.ndarray,
    filepath: str,
    *,
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a Q4 surface mesh VTU file for plate/shell visualization."""
    coords3 = _surface_coords3(coords)
    conn_arr = np.asarray(conn, dtype=np.int32)
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 4:
        raise ValueError("conn must have shape (n_elems, 4).")
    if np.any(conn_arr < 0) or np.any(conn_arr >= coords3.shape[0]):
        raise ValueError("conn contains an index outside coords.")

    n_cells = conn_arr.shape[0]
    offsets = np.cumsum(np.full(n_cells, 4, dtype=np.int32))
    types = np.full(n_cells, 9, dtype=np.int32)  # VTK_QUAD
    point_data = point_data or {}
    cell_data = cell_data or {}

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{coords3.shape[0]}" NumberOfCells="{n_cells}">',
        "      <PointData>",
    ]
    for name, arr in point_data.items():
        arr_np = np.asarray(arr)
        ncomp = 1 if arr_np.ndim == 1 else arr_np.shape[1]
        lines.append("        " + _surface_dataarray(name, arr_np, ncomp))
    lines.extend(["      </PointData>", "      <CellData>"])
    for name, arr in cell_data.items():
        arr_np = np.asarray(arr)
        ncomp = 1 if arr_np.ndim == 1 else arr_np.shape[1]
        lines.append("        " + _surface_dataarray(name, arr_np, ncomp))
    lines.extend(
        [
            "      </CellData>",
            "      <Points>",
            "        " + _surface_dataarray("Points", coords3, 3),
            "      </Points>",
            "      <Cells>",
            '        <DataArray type="Int32" Name="connectivity" format="ascii">'
            + " ".join(str(int(v)) for v in conn_arr.reshape(-1))
            + "</DataArray>",
            '        <DataArray type="Int32" Name="offsets" format="ascii">'
            + " ".join(str(int(v)) for v in offsets)
            + "</DataArray>",
            '        <DataArray type="UInt8" Name="types" format="ascii">'
            + " ".join(str(int(v)) for v in types)
            + "</DataArray>",
            "      </Cells>",
            "    </Piece>",
            "  </UnstructuredGrid>",
            "</VTKFile>",
        ]
    )
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


__all__ = [
    "PLATE_DOF_PER_NODE",
    "SHELL_DOF_PER_NODE",
    "PlateSection",
    "ShellSection",
    "assemble_flat_shell_point_load",
    "assemble_flat_shell_point_loads",
    "assemble_flat_shell_stiffness",
    "assemble_flat_shell_uniform_load",
    "assemble_shell_stiffness",
    "assemble_shell_uniform_load",
    "assemble_mindlin_plate_point_load",
    "assemble_mindlin_plate_point_loads",
    "assemble_mindlin_plate_stiffness",
    "assemble_mindlin_plate_uniform_load",
    "flat_shell_element_stiffness",
    "flat_shell_element_uniform_load",
    "mindlin_plate_element_stiffness",
    "mindlin_plate_element_uniform_load",
    "plate_element_dofs",
    "plate_node_dofs",
    "shell_element_dofs",
    "shell_element_frame",
    "shell_element_stiffness_global",
    "shell_element_uniform_load_global",
    "shell_node_dofs",
    "shell_solid_translational_tie_dofs",
    "structured_plate_grid",
    "write_q4_surface_vtu",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

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


TRUSS_DOF_PER_NODE = 3


@dataclass(frozen=True)
class TrussSection:
    """Material and section properties for a 3D two-node truss/bar element."""

    E: float
    A: float
    rho: float | None = None


def truss_node_dofs(
    nodes: Sequence[int] | np.ndarray,
    components: Sequence[int] | str = "xyz",
) -> np.ndarray:
    """Return flattened 3-DOF truss indices for nodes."""
    nodes_arr = np.asarray(nodes, dtype=int).reshape(-1)
    if isinstance(components, str):
        comp_map = {"x": 0, "y": 1, "z": 2, "u": 0, "v": 1, "w": 2}
        comps = np.asarray([comp_map[c.lower()] for c in components], dtype=int)
    else:
        comps = np.asarray(list(components), dtype=int)
    return np.asarray([TRUSS_DOF_PER_NODE * int(n) + int(c) for n in nodes_arr for c in comps], dtype=int)


def structured_truss_chain(
    *,
    n_elems: int,
    length: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build coordinates and 2-node connectivity for a straight truss/bar chain."""
    if n_elems <= 0:
        raise ValueError("n_elems must be positive.")
    origin_arr = np.asarray(origin, dtype=float)
    direction_arr = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction_arr))
    if norm == 0.0:
        raise ValueError("direction must be nonzero.")
    axis = direction_arr / norm
    s = np.linspace(0.0, float(length), int(n_elems) + 1)
    coords = origin_arr[None, :] + s[:, None] * axis[None, :]
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)]).astype(int)
    return coords, conn


def truss_element_dofs(conn: np.ndarray) -> np.ndarray:
    """Return element-to-DOF connectivity for 2-node truss elements."""
    conn_arr = np.asarray(conn, dtype=int)
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 2:
        raise ValueError("conn must have shape (n_elems, 2).")
    elem_dofs = np.empty((conn_arr.shape[0], 6), dtype=int)
    for e, (n0, n1) in enumerate(conn_arr):
        elem_dofs[e, :3] = TRUSS_DOF_PER_NODE * int(n0) + np.arange(3)
        elem_dofs[e, 3:] = TRUSS_DOF_PER_NODE * int(n1) + np.arange(3)
    return elem_dofs


def _axis_and_length(xi: Sequence[float], xj: Sequence[float]) -> tuple[np.ndarray, float]:
    dx = np.asarray(xj, dtype=float) - np.asarray(xi, dtype=float)
    length = float(np.linalg.norm(dx))
    if length <= 0.0:
        raise ValueError("Truss element length must be positive.")
    return dx / length, length


def truss_element_stiffness_global(
    xi: Sequence[float],
    xj: Sequence[float],
    section: TrussSection,
) -> np.ndarray:
    """6x6 global stiffness for a two-node 3D truss/bar element."""
    axis, length = _axis_and_length(xi, xj)
    nn = np.outer(axis, axis)
    k = float(section.E) * float(section.A) / length
    return k * np.block([[nn, -nn], [-nn, nn]])


def truss_element_mass_global(
    xi: Sequence[float],
    xj: Sequence[float],
    section: TrussSection,
    *,
    kind: Literal["consistent", "lumped"] = "consistent",
) -> np.ndarray:
    """6x6 translational mass matrix for a two-node 3D truss/bar element."""
    if section.rho is None:
        raise ValueError("section.rho is required for truss mass assembly.")
    _axis, length = _axis_and_length(xi, xj)
    total_mass = float(section.rho) * float(section.A) * length
    eye = np.eye(3)
    if kind == "consistent":
        return (total_mass / 6.0) * np.block([[2.0 * eye, eye], [eye, 2.0 * eye]])
    if kind == "lumped":
        return np.diag(np.full(6, total_mass / 2.0))
    raise ValueError("kind must be 'consistent' or 'lumped'.")


def truss_element_uniform_load_global(
    xi: Sequence[float],
    xj: Sequence[float],
    load: Sequence[float] | np.ndarray,
    *,
    frame: Literal["global", "local"] = "global",
) -> np.ndarray:
    """6-vector of equivalent nodal loads for a uniform truss/bar load."""
    axis, length = _axis_and_length(xi, xj)
    q = np.asarray(load, dtype=float).reshape(-1)
    if frame == "local":
        if q.size != 1:
            raise ValueError("local truss load must have one axial component.")
        q_global = axis * float(q[0])
    elif frame == "global":
        if q.size != 3:
            raise ValueError("global truss load must have three components.")
        q_global = q
    else:
        raise ValueError("frame must be 'global' or 'local'.")
    nodal = q_global * length / 2.0
    return np.concatenate([nodal, nodal])


def _assemble_truss_matrix(
    coords: np.ndarray,
    conn: np.ndarray,
    element_matrix,
    *,
    format: MatrixFormat,
):
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (n_nodes, 3).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 2:
        raise ValueError("conn must have shape (n_elems, 2).")

    elem_dofs = truss_element_dofs(conn_arr)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e, (n0, n1) in enumerate(conn_arr):
        me = element_matrix(coords_arr[n0], coords_arr[n1])
        dofs = elem_dofs[e]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(me.reshape(-1).tolist())

    n_dofs = TRUSS_DOF_PER_NODE * coords_arr.shape[0]
    return _sparse_from_coo(rows, cols, data, n_dofs, format=format)


def assemble_truss_stiffness(
    coords: np.ndarray,
    conn: np.ndarray,
    section: TrussSection,
    *,
    format: MatrixFormat | None = None,
    backend: MatrixBackend | None = None,
):
    """Assemble global stiffness for 3D two-node truss/bar elements."""
    return _assemble_truss_matrix(
        coords,
        conn,
        lambda xi, xj: truss_element_stiffness_global(xi, xj, section),
        format=_resolve_matrix_format(format, backend),
    )


def assemble_truss_mass(
    coords: np.ndarray,
    conn: np.ndarray,
    section: TrussSection,
    *,
    kind: Literal["consistent", "lumped"] = "consistent",
    format: MatrixFormat | None = None,
    backend: MatrixBackend | None = None,
):
    """Assemble global translational mass for 3D two-node truss/bar elements."""
    return _assemble_truss_matrix(
        coords,
        conn,
        lambda xi, xj: truss_element_mass_global(xi, xj, section, kind=kind),
        format=_resolve_matrix_format(format, backend),
    )


def assemble_truss_uniform_load(
    coords: np.ndarray,
    conn: np.ndarray,
    load: Sequence[float] | np.ndarray,
    *,
    frame: Literal["global", "local"] = "global",
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble equivalent nodal loads for a uniform truss/bar distributed force."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (n_nodes, 3).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 2:
        raise ValueError("conn must have shape (n_elems, 2).")

    out = np.zeros(TRUSS_DOF_PER_NODE * coords_arr.shape[0], dtype=float)
    elem_dofs = truss_element_dofs(conn_arr)
    for e, (n0, n1) in enumerate(conn_arr):
        fe = truss_element_uniform_load_global(coords_arr[n0], coords_arr[n1], load, frame=frame)
        np.add.at(out, elem_dofs[e], fe)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


def assemble_truss_point_load(
    n_nodes: int,
    node: int,
    *,
    force: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble a dense nodal load vector for one truss/bar node."""
    return assemble_truss_point_loads(
        n_nodes,
        [node],
        forces=[force],
        array_backend=_resolve_array_backend(array_backend, backend),
    )


def assemble_truss_point_loads(
    n_nodes: int,
    nodes: Sequence[int] | np.ndarray,
    *,
    forces: Sequence[Sequence[float]] | np.ndarray | None = None,
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble dense nodal force loads for truss/bar nodes."""
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

    out = np.zeros(TRUSS_DOF_PER_NODE * int(n_nodes), dtype=float)
    for node, force_vec in zip(nodes_arr, force_arr):
        dofs = TRUSS_DOF_PER_NODE * int(node) + np.arange(3)
        np.add.at(out, dofs, force_vec)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


__all__ = [
    "TRUSS_DOF_PER_NODE",
    "TrussSection",
    "assemble_truss_mass",
    "assemble_truss_point_load",
    "assemble_truss_point_loads",
    "assemble_truss_stiffness",
    "assemble_truss_uniform_load",
    "structured_truss_chain",
    "truss_element_dofs",
    "truss_element_mass_global",
    "truss_element_stiffness_global",
    "truss_element_uniform_load_global",
    "truss_node_dofs",
]

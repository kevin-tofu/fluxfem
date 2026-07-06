from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import jax.numpy as jnp

from ..solver.sparse import FluxSparseMatrix


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


def _assemble_truss_matrix(
    coords: np.ndarray,
    conn: np.ndarray,
    element_matrix,
) -> FluxSparseMatrix:
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
    return FluxSparseMatrix(
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(data, dtype=jnp.float64),
        n_dofs,
    ).coalesce()


def assemble_truss_stiffness(
    coords: np.ndarray,
    conn: np.ndarray,
    section: TrussSection,
) -> FluxSparseMatrix:
    """Assemble global stiffness for 3D two-node truss/bar elements."""
    return _assemble_truss_matrix(coords, conn, lambda xi, xj: truss_element_stiffness_global(xi, xj, section))


def assemble_truss_mass(
    coords: np.ndarray,
    conn: np.ndarray,
    section: TrussSection,
    *,
    kind: Literal["consistent", "lumped"] = "consistent",
) -> FluxSparseMatrix:
    """Assemble global translational mass for 3D two-node truss/bar elements."""
    return _assemble_truss_matrix(coords, conn, lambda xi, xj: truss_element_mass_global(xi, xj, section, kind=kind))


__all__ = [
    "TRUSS_DOF_PER_NODE",
    "TrussSection",
    "assemble_truss_mass",
    "assemble_truss_stiffness",
    "structured_truss_chain",
    "truss_element_dofs",
    "truss_element_mass_global",
    "truss_element_stiffness_global",
    "truss_node_dofs",
]

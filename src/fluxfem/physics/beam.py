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


BEAM_DOF_PER_NODE = 6


@dataclass(frozen=True)
class BeamSection:
    """Material and section properties for a 3D Euler-Bernoulli frame element."""

    E: float
    G: float
    A: float
    Iy: float
    Iz: float
    J: float
    rho: float | None = None


def beam_node_dofs(
    nodes: Sequence[int] | np.ndarray,
    components: Sequence[int] | str = "uxuyuzrxryrz",
) -> np.ndarray:
    """
    Return flattened 6-DOF beam indices for nodes.

    Component names are ux, uy, uz, rx, ry, rz. A compact string such as
    "uxuyuz" or "rxryrz" is accepted.
    """
    nodes_arr = np.asarray(nodes, dtype=int).reshape(-1)
    comp_map = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}
    if isinstance(components, str):
        text = components.lower().replace(",", " ").replace("_", " ")
        if " " in text:
            tokens = [tok for tok in text.split() if tok]
        else:
            if len(text) % 2 != 0:
                raise ValueError("Beam component string must use two-letter tokens: ux, uy, uz, rx, ry, rz.")
            tokens = [text[i : i + 2] for i in range(0, len(text), 2)]
        comps = np.asarray([comp_map[tok] for tok in tokens], dtype=int)
    else:
        comps = np.asarray(list(components), dtype=int)
    return np.asarray([BEAM_DOF_PER_NODE * int(n) + int(c) for n in nodes_arr for c in comps], dtype=int)


def structured_beam_chain(
    *,
    n_elems: int,
    length: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build coordinates and 2-node connectivity for a straight beam chain."""
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


def beam_element_dofs(conn: np.ndarray) -> np.ndarray:
    """Return element-to-DOF connectivity for 2-node beam elements."""
    conn_arr = np.asarray(conn, dtype=int)
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 2:
        raise ValueError("conn must have shape (n_elems, 2).")
    elem_dofs = np.empty((conn_arr.shape[0], 12), dtype=int)
    for e, (n0, n1) in enumerate(conn_arr):
        elem_dofs[e, :6] = BEAM_DOF_PER_NODE * int(n0) + np.arange(6)
        elem_dofs[e, 6:] = BEAM_DOF_PER_NODE * int(n1) + np.arange(6)
    return elem_dofs


def _beam_rotation_matrix(
    xi: np.ndarray,
    xj: np.ndarray,
    reference: Sequence[float] | None,
) -> tuple[np.ndarray, float]:
    dx = np.asarray(xj, dtype=float) - np.asarray(xi, dtype=float)
    length = float(np.linalg.norm(dx))
    if length <= 0.0:
        raise ValueError("Beam element length must be positive.")
    ex = dx / length

    ref = np.asarray((0.0, 1.0, 0.0) if reference is None else reference, dtype=float)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0.0:
        raise ValueError("reference vector must be nonzero.")
    ref = ref / ref_norm
    if abs(float(np.dot(ex, ref))) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(ex, ref))) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])

    ey = ref - np.dot(ref, ex) * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)
    return np.vstack([ex, ey, ez]), length


def beam_element_stiffness_local(length: float, section: BeamSection) -> np.ndarray:
    """12x12 local stiffness for a 3D Euler-Bernoulli frame element."""
    L = float(length)
    if L <= 0.0:
        raise ValueError("length must be positive.")
    E = float(section.E)
    G = float(section.G)
    A = float(section.A)
    Iy = float(section.Iy)
    Iz = float(section.Iz)
    J = float(section.J)

    k = np.zeros((12, 12), dtype=float)

    axial = E * A / L
    k[np.ix_([0, 6], [0, 6])] += axial * np.array([[1.0, -1.0], [-1.0, 1.0]])

    torsion = G * J / L
    k[np.ix_([3, 9], [3, 9])] += torsion * np.array([[1.0, -1.0], [-1.0, 1.0]])

    bz = E * Iz / (L**3)
    k[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] += bz * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
        ]
    )

    by = E * Iy / (L**3)
    k[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] += by * np.array(
        [
            [12.0, -6.0 * L, -12.0, -6.0 * L],
            [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L],
            [-12.0, 6.0 * L, 12.0, 6.0 * L],
            [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L],
        ]
    )

    return k


def beam_element_stiffness_global(
    xi: Sequence[float],
    xj: Sequence[float],
    section: BeamSection,
    *,
    reference: Sequence[float] | None = None,
) -> np.ndarray:
    """12x12 global stiffness for a two-node beam element."""
    R, length = _beam_rotation_matrix(np.asarray(xi, dtype=float), np.asarray(xj, dtype=float), reference)
    transform = np.zeros((12, 12), dtype=float)
    for start in (0, 3, 6, 9):
        transform[start : start + 3, start : start + 3] = R
    k_local = beam_element_stiffness_local(length, section)
    return transform.T @ k_local @ transform


def beam_element_mass_local(
    length: float,
    section: BeamSection,
    *,
    kind: Literal["consistent", "lumped"] = "consistent",
) -> np.ndarray:
    """12x12 local mass matrix for a 3D Euler-Bernoulli frame element."""
    if section.rho is None:
        raise ValueError("section.rho is required for beam mass assembly.")
    L = float(length)
    if L <= 0.0:
        raise ValueError("length must be positive.")
    rho = float(section.rho)
    A = float(section.A)
    J = float(section.J)

    if kind == "lumped":
        m = np.zeros((12, 12), dtype=float)
        nodal_mass = rho * A * L / 2.0
        nodal_rotary = rho * J * L / 2.0
        for start in (0, 6):
            m[start + 0, start + 0] = nodal_mass
            m[start + 1, start + 1] = nodal_mass
            m[start + 2, start + 2] = nodal_mass
            m[start + 3, start + 3] = nodal_rotary
        return m
    if kind != "consistent":
        raise ValueError("kind must be 'consistent' or 'lumped'.")

    m = np.zeros((12, 12), dtype=float)

    axial = rho * A * L / 6.0
    m[np.ix_([0, 6], [0, 6])] += axial * np.array([[2.0, 1.0], [1.0, 2.0]])

    torsion = rho * J * L / 6.0
    m[np.ix_([3, 9], [3, 9])] += torsion * np.array([[2.0, 1.0], [1.0, 2.0]])

    bending = rho * A * L / 420.0
    m[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] += bending * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L * L, 13.0 * L, -3.0 * L * L],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L * L, -22.0 * L, 4.0 * L * L],
        ]
    )
    m[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] += bending * np.array(
        [
            [156.0, -22.0 * L, 54.0, 13.0 * L],
            [-22.0 * L, 4.0 * L * L, -13.0 * L, -3.0 * L * L],
            [54.0, -13.0 * L, 156.0, 22.0 * L],
            [13.0 * L, -3.0 * L * L, 22.0 * L, 4.0 * L * L],
        ]
    )

    return m


def beam_element_mass_global(
    xi: Sequence[float],
    xj: Sequence[float],
    section: BeamSection,
    *,
    reference: Sequence[float] | None = None,
    kind: Literal["consistent", "lumped"] = "consistent",
) -> np.ndarray:
    """12x12 global mass matrix for a two-node beam element."""
    R, length = _beam_rotation_matrix(np.asarray(xi, dtype=float), np.asarray(xj, dtype=float), reference)
    transform = np.zeros((12, 12), dtype=float)
    for start in (0, 3, 6, 9):
        transform[start : start + 3, start : start + 3] = R
    m_local = beam_element_mass_local(length, section, kind=kind)
    return transform.T @ m_local @ transform


def beam_element_uniform_load_local(
    length: float,
    load: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    12-vector of equivalent nodal loads for a uniform local distributed force.

    ``load`` is ``[qx, qy, qz]`` in force per length in the element local frame.
    Positive qz follows the local z axis; for a beam along global x with default
    orientation this is global z.
    """
    L = float(length)
    if L <= 0.0:
        raise ValueError("length must be positive.")
    q = np.asarray(load, dtype=float).reshape(-1)
    if q.size != 3:
        raise ValueError("load must have three components [qx, qy, qz].")
    qx, qy, qz = (float(q[0]), float(q[1]), float(q[2]))

    f = np.zeros(12, dtype=float)
    f[[0, 6]] += qx * L / 2.0
    f[[1, 7]] += qy * L / 2.0
    f[[5, 11]] += np.array([qy * L * L / 12.0, -qy * L * L / 12.0])
    f[[2, 8]] += qz * L / 2.0
    f[[4, 10]] += np.array([-qz * L * L / 12.0, qz * L * L / 12.0])
    return f


def beam_element_uniform_load_global(
    xi: Sequence[float],
    xj: Sequence[float],
    load: Sequence[float] | np.ndarray,
    *,
    reference: Sequence[float] | None = None,
    frame: Literal["global", "local"] = "global",
) -> np.ndarray:
    """12-vector of equivalent nodal loads for a uniform beam load."""
    R, length = _beam_rotation_matrix(np.asarray(xi, dtype=float), np.asarray(xj, dtype=float), reference)
    q = np.asarray(load, dtype=float).reshape(-1)
    if q.size != 3:
        raise ValueError("load must have three components.")
    if frame == "global":
        q_local = R @ q
    elif frame == "local":
        q_local = q
    else:
        raise ValueError("frame must be 'global' or 'local'.")

    transform = np.zeros((12, 12), dtype=float)
    for start in (0, 3, 6, 9):
        transform[start : start + 3, start : start + 3] = R
    f_local = beam_element_uniform_load_local(length, q_local)
    return transform.T @ f_local


def _assemble_beam_matrix(
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

    elem_dofs = beam_element_dofs(conn_arr)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e, (n0, n1) in enumerate(conn_arr):
        ke = element_matrix(coords_arr[n0], coords_arr[n1])
        dofs = elem_dofs[e]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(ke.reshape(-1).tolist())

    n_dofs = BEAM_DOF_PER_NODE * coords_arr.shape[0]
    return _sparse_from_coo(rows, cols, data, n_dofs, format=format)


def assemble_beam_stiffness(
    coords: np.ndarray,
    conn: np.ndarray,
    section: BeamSection,
    *,
    reference: Sequence[float] | None = None,
    format: MatrixFormat | None = None,
    backend: MatrixBackend | None = None,
):
    """Assemble a sparse global stiffness matrix for 3D Euler-Bernoulli beam elements."""
    return _assemble_beam_matrix(
        coords,
        conn,
        lambda xi, xj: beam_element_stiffness_global(xi, xj, section, reference=reference),
        format=_resolve_matrix_format(format, backend),
    )


def assemble_beam_mass(
    coords: np.ndarray,
    conn: np.ndarray,
    section: BeamSection,
    *,
    reference: Sequence[float] | None = None,
    kind: Literal["consistent", "lumped"] = "consistent",
    format: MatrixFormat | None = None,
    backend: MatrixBackend | None = None,
):
    """Assemble a sparse global mass matrix for 3D Euler-Bernoulli beam elements."""
    return _assemble_beam_matrix(
        coords,
        conn,
        lambda xi, xj: beam_element_mass_global(xi, xj, section, reference=reference, kind=kind),
        format=_resolve_matrix_format(format, backend),
    )


def assemble_beam_uniform_load(
    coords: np.ndarray,
    conn: np.ndarray,
    load: Sequence[float] | np.ndarray,
    *,
    reference: Sequence[float] | None = None,
    frame: Literal["global", "local"] = "global",
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble equivalent nodal loads for a uniform distributed beam force."""
    coords_arr = np.asarray(coords, dtype=float)
    conn_arr = np.asarray(conn, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (n_nodes, 3).")
    if conn_arr.ndim != 2 or conn_arr.shape[1] != 2:
        raise ValueError("conn must have shape (n_elems, 2).")

    out = np.zeros(BEAM_DOF_PER_NODE * coords_arr.shape[0], dtype=float)
    elem_dofs = beam_element_dofs(conn_arr)
    for e, (n0, n1) in enumerate(conn_arr):
        fe = beam_element_uniform_load_global(
            coords_arr[n0],
            coords_arr[n1],
            load,
            reference=reference,
            frame=frame,
        )
        np.add.at(out, elem_dofs[e], fe)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


def assemble_beam_point_load(
    n_nodes: int,
    node: int,
    *,
    force: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    moment: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble a dense nodal load vector for one beam node."""
    return assemble_beam_point_loads(
        n_nodes,
        [node],
        forces=[force],
        moments=[moment],
        array_backend=_resolve_array_backend(array_backend, backend),
    )


def assemble_beam_point_loads(
    n_nodes: int,
    nodes: Sequence[int] | np.ndarray,
    *,
    forces: Sequence[Sequence[float]] | np.ndarray | None = None,
    moments: Sequence[Sequence[float]] | np.ndarray | None = None,
    array_backend: ArrayBackend | None = None,
    backend: ArrayBackend | None = None,
):
    """Assemble dense nodal force/moment loads for beam nodes."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    nodes_arr = np.asarray(nodes, dtype=int).reshape(-1)
    if nodes_arr.size == 0:
        raise ValueError("nodes must contain at least one node.")
    if np.any(nodes_arr < 0) or np.any(nodes_arr >= int(n_nodes)):
        raise ValueError("nodes contains an index outside n_nodes.")

    def normalize(values, name: str) -> np.ndarray:
        if values is None:
            return np.zeros((nodes_arr.size, 3), dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.shape == (3,):
            if nodes_arr.size != 1:
                raise ValueError(f"{name} with shape (3,) is only valid for one node.")
            return arr.reshape(1, 3)
        if arr.shape != (nodes_arr.size, 3):
            raise ValueError(f"{name} must have shape ({nodes_arr.size}, 3).")
        return arr

    force_arr = normalize(forces, "forces")
    moment_arr = normalize(moments, "moments")
    out = np.zeros(BEAM_DOF_PER_NODE * int(n_nodes), dtype=float)
    for node, force_vec, moment_vec in zip(nodes_arr, force_arr, moment_arr):
        dofs = BEAM_DOF_PER_NODE * int(node) + np.arange(6)
        values = np.concatenate([force_vec, moment_vec])
        np.add.at(out, dofs, values)
    return _as_array_backend(out, _resolve_array_backend(array_backend, backend))


__all__ = [
    "BEAM_DOF_PER_NODE",
    "BeamSection",
    "assemble_beam_mass",
    "assemble_beam_point_load",
    "assemble_beam_point_loads",
    "assemble_beam_stiffness",
    "assemble_beam_uniform_load",
    "beam_element_dofs",
    "beam_element_mass_global",
    "beam_element_mass_local",
    "beam_element_stiffness_global",
    "beam_element_stiffness_local",
    "beam_element_uniform_load_global",
    "beam_element_uniform_load_local",
    "beam_node_dofs",
    "structured_beam_chain",
]

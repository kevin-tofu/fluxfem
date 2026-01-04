from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .surface import SurfaceMesh
from ..core.forms import FieldPair


@dataclass(eq=False)
class _SurfaceBasis:
    dofs_per_node: int


@dataclass(eq=False)
class SurfaceMixedFormField:
    """Surface form field for mixed weak-form evaluation."""
    N: np.ndarray
    gradN: np.ndarray | None
    value_dim: int
    basis: _SurfaceBasis


@dataclass(eq=False)
class SurfaceMixedFormContext:
    """Surface mixed context for weak-form evaluation on supermesh."""
    fields: dict[str, FieldPair]
    x_q: np.ndarray
    w: np.ndarray
    detJ: np.ndarray
    normal: np.ndarray | None = None
    trial_fields: dict[str, SurfaceMixedFormField] | None = None
    test_fields: dict[str, SurfaceMixedFormField] | None = None
    unknown_fields: dict[str, SurfaceMixedFormField] | None = None


_DEBUG_SURFACE_GRADN = os.getenv("FLUXFEM_DEBUG_SURFACE_GRADN")
_DEBUG_SURFACE_GRADN_MAX = int(os.getenv("FLUXFEM_DEBUG_SURFACE_GRADN_MAX", "8")) if _DEBUG_SURFACE_GRADN else 0
_DEBUG_SURFACE_GRADN_COUNT = 0
_DEBUG_SURFACE_SOURCE_ONCE = False
_DEBUG_CONTACT_MAP_ONCE = False
_DEBUG_CONTACT_N_ONCE = False


@dataclass(eq=False)
class MortarMatrix:
    """COO storage for mortar coupling matrices (can be rectangular)."""
    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]


def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _facet_area_estimate(facet_nodes: np.ndarray, coords: np.ndarray) -> float:
    n = int(len(facet_nodes))
    if n == 3:
        pts = coords[facet_nodes]
        return _tri_area(pts[0], pts[1], pts[2])
    if n == 4:
        pts = coords[facet_nodes]
        return _tri_area(pts[0], pts[1], pts[2]) + _tri_area(pts[0], pts[2], pts[3])
    if n == 8:
        corner_nodes = facet_nodes[:4]
        pts = coords[corner_nodes]
        return _tri_area(pts[0], pts[1], pts[2]) + _tri_area(pts[0], pts[2], pts[3])
    if n == 9:
        corner_nodes = facet_nodes[[0, 2, 8, 6]]
        pts = coords[corner_nodes]
        return _tri_area(pts[0], pts[1], pts[2]) + _tri_area(pts[0], pts[2], pts[3])
    pts = coords[facet_nodes]
    area = 0.0
    p0 = pts[0]
    for i in range(1, len(pts) - 1):
        area += _tri_area(p0, pts[i], pts[i + 1])
    return float(area)


def _tri_centroid(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (a + b + c) / 3.0


def _tri_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return reference triangle quadrature points (r, s) and weights.
    Reference triangle is (0,0), (1,0), (0,1); weights integrate over area 1/2.
    """
    if order <= 0:
        return np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5])
    if order <= 2:
        pts = np.array(
            [
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=float,
        )
        weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float)
        return pts, weights
    if order <= 3:
        pts = np.array(
            [
                [1.0 / 3.0, 1.0 / 3.0],
                [0.2, 0.2],
                [0.6, 0.2],
                [0.2, 0.6],
            ],
            dtype=float,
        )
        weights = np.array(
            [-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0],
            dtype=float,
        )
        return pts, weights
    if order <= 4:
        a = 0.445948490915965
        b = 0.108103018168070
        c = 0.091576213509771
        d = 0.816847572980459
        pts = np.array(
            [
                [a, a],
                [a, b],
                [b, a],
                [c, c],
                [c, d],
                [d, c],
            ],
            dtype=float,
        )
        weights = np.array(
            [
                0.111690794839005,
                0.111690794839005,
                0.111690794839005,
                0.054975871827661,
                0.054975871827661,
                0.054975871827661,
            ],
            dtype=float,
        )
        return pts, weights
    if order <= 5:
        a = 0.470142064105115
        b = 0.059715871789770
        c = 0.101286507323456
        d = 0.797426985353087
        pts = np.array(
            [
                [1.0 / 3.0, 1.0 / 3.0],
                [a, a],
                [a, b],
                [b, a],
                [c, c],
                [c, d],
                [d, c],
            ],
            dtype=float,
        )
        weights = np.array(
            [
                0.225000000000000,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827,
            ],
            dtype=float,
        )
        return pts, weights
    raise NotImplementedError("triangle quadrature order > 5 is not implemented")


def _barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=float)


def _point_in_tri(lam: np.ndarray, *, tol: float) -> bool:
    return np.all(lam >= -tol) and np.all(lam <= 1.0 + tol)


def _plane_basis(pts: np.ndarray, *, tol: float):
    v1 = pts[1] - pts[0]
    v2 = pts[3] - pts[0] if pts.shape[0] > 3 else pts[2] - pts[0]
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < tol:
        return None, None
    n = n / n_norm
    t1 = v1 / np.linalg.norm(v1)
    v2_proj = v2 - np.dot(v2, t1) * t1
    v2_norm = np.linalg.norm(v2_proj)
    if v2_norm < tol:
        return None, None
    t2 = v2_proj / v2_norm
    return t1, t2


def _quad_shape_and_local(
    point: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, float, float]:
    pts = coords[facet_nodes]
    basis = _plane_basis(pts, tol=tol)
    if basis[0] is None:
        return np.zeros((4,), dtype=float), 0.0, 0.0
    t1, t2 = basis
    origin = pts[0]
    local = (pts - origin) @ np.stack([t1, t2], axis=1)
    p_local = (point - origin) @ np.stack([t1, t2], axis=1)
    x = local[:, 0]
    y = local[:, 1]
    xp = float(p_local[0])
    yp = float(p_local[1])

    xi = 0.0
    eta = 0.0
    for _ in range(12):
        n1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        n2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        n3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        n4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        x_m = n1 * x[0] + n2 * x[1] + n3 * x[2] + n4 * x[3]
        y_m = n1 * y[0] + n2 * y[1] + n3 * y[2] + n4 * y[3]
        rx = x_m - xp
        ry = y_m - yp
        if abs(rx) + abs(ry) < tol:
            break
        dndxi = np.array(
            [
                -0.25 * (1.0 - eta),
                0.25 * (1.0 - eta),
                0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            dtype=float,
        )
        dndeta = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                0.25 * (1.0 + xi),
                0.25 * (1.0 - xi),
            ],
            dtype=float,
        )
        j11 = float(np.dot(dndxi, x))
        j12 = float(np.dot(dndeta, x))
        j21 = float(np.dot(dndxi, y))
        j22 = float(np.dot(dndeta, y))
        det = j11 * j22 - j12 * j21
        if abs(det) < tol:
            return np.zeros((4,), dtype=float), xi, eta
        dxi = (-j22 * rx + j12 * ry) / det
        deta = (j21 * rx - j11 * ry) / det
        xi += dxi
        eta += deta

    return np.array([n1, n2, n3, n4], dtype=float), xi, eta


def _quad_shape_values(
    point: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    values, _xi, _eta = _quad_shape_and_local(point, facet_nodes, coords, tol=tol)
    return values


def _quad8_shape_values(xi: float, eta: float) -> np.ndarray:
    n1 = -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta)
    n2 = -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta)
    n3 = -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta)
    n4 = -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta)
    n5 = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    n6 = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    n7 = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    n8 = 0.5 * (1.0 - xi) * (1.0 - eta * eta)
    return np.array([n1, n2, n3, n4, n5, n6, n7, n8], dtype=float)


def _quad9_shape_values(xi: float, eta: float) -> np.ndarray:
    def q1(t):
        return 0.5 * t * (t - 1.0)

    def q2(t):
        return 1.0 - t * t

    def q3(t):
        return 0.5 * t * (t + 1.0)

    Nx = [q1(xi), q2(xi), q3(xi)]
    Ny = [q1(eta), q2(eta), q3(eta)]
    out = []
    for j in range(3):
        for i in range(3):
            out.append(Nx[i] * Ny[j])
    return np.array(out, dtype=float)


def _facet_shape_values(
    point: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    """
    Evaluate nodal shape values on a facet at a point.

    Tri: standard barycentric.
    Quad: split into (0,1,2) and (0,2,3) triangles, piecewise linear.
    """
    pts = coords[facet_nodes]
    n = len(facet_nodes)
    if n == 3:
        lam = _barycentric(point, pts[0], pts[1], pts[2])
        if lam is None:
            return np.zeros((3,), dtype=float)
        return lam
    if n == 6:
        lam = _barycentric(point, pts[0], pts[1], pts[2])
        if lam is None or np.any(lam < -tol):
            return np.zeros((6,), dtype=float)
        L1, L2, L3 = lam
        N1 = L1 * (2.0 * L1 - 1.0)
        N2 = L2 * (2.0 * L2 - 1.0)
        N3 = L3 * (2.0 * L3 - 1.0)
        N4 = 4.0 * L1 * L2
        N5 = 4.0 * L2 * L3
        N6 = 4.0 * L1 * L3
        return np.array([N1, N2, N3, N4, N5, N6], dtype=float)
    if n == 4:
        return _quad_shape_values(point, facet_nodes, coords, tol=tol)
    if n == 8:
        corner_nodes = facet_nodes[:4]
        values, xi, eta = _quad_shape_and_local(point, corner_nodes, coords, tol=tol)
        if np.allclose(values, 0.0):
            return np.zeros((8,), dtype=float)
        return _quad8_shape_values(xi, eta)
    if n == 9:
        corner_nodes = facet_nodes[[0, 2, 8, 6]]
        values, xi, eta = _quad_shape_and_local(point, corner_nodes, coords, tol=tol)
        if np.allclose(values, 0.0):
            return np.zeros((9,), dtype=float)
        return _quad9_shape_values(xi, eta)
    raise ValueError("facet must be a triangle or quad")


def _gather_u_local(u_field: np.ndarray, nodes: np.ndarray, value_dim: int) -> np.ndarray:
    if value_dim == 1:
        return u_field[nodes]
    idx = np.repeat(nodes * value_dim, value_dim) + np.tile(np.arange(value_dim), len(nodes))
    return u_field[idx]


def _global_dof_indices(nodes: np.ndarray, value_dim: int, offset: int) -> np.ndarray:
    if value_dim == 1:
        return offset + nodes
    idx = np.repeat(nodes * value_dim, value_dim) + np.tile(np.arange(value_dim), len(nodes))
    return offset + idx


def map_surface_facets_to_tet_elements(surface: SurfaceMesh, tet_conn: np.ndarray) -> np.ndarray:
    """
    Map surface triangle facets to parent tet elements by node matching (tet4/tet10).
    """
    face_patterns_corner = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    face_patterns_quad = [
        (0, 1, 2, 4, 5, 6),
        (0, 1, 3, 4, 8, 7),
        (0, 2, 3, 6, 9, 7),
        (1, 2, 3, 5, 9, 8),
    ]
    tet_conn = np.asarray(tet_conn, dtype=int)
    if tet_conn.shape[1] not in {4, 10}:
        raise NotImplementedError("Only tet4 and tet10 are supported.")
    mapping_corner: dict[tuple[int, ...], int] = {}
    mapping_quad: dict[tuple[int, ...], int] = {}
    for e_id, elem in enumerate(tet_conn):
        for pattern in face_patterns_corner:
            face_nodes = tuple(sorted(int(elem[i]) for i in pattern))
            mapping_corner.setdefault(face_nodes, e_id)
        if elem.shape[0] == 10:
            for pattern in face_patterns_quad:
                face_nodes = tuple(sorted(int(elem[i]) for i in pattern))
                mapping_quad.setdefault(face_nodes, e_id)
    facet_map = np.full((surface.conn.shape[0],), -1, dtype=int)
    for f_id, facet in enumerate(np.asarray(surface.conn, dtype=int)):
        key = tuple(sorted(int(n) for n in facet))
        if len(facet) == 3 and key in mapping_corner:
            facet_map[f_id] = mapping_corner[key]
        elif len(facet) == 6 and key in mapping_quad:
            facet_map[f_id] = mapping_quad[key]
        elif key in mapping_corner:
            facet_map[f_id] = mapping_corner[key]
    return facet_map


def map_surface_facets_to_hex_elements(surface: SurfaceMesh, hex_conn: np.ndarray) -> np.ndarray:
    """
    Map surface quad facets to parent hex elements by node matching (hex8/hex20/hex27).
    """
    hex_conn = np.asarray(hex_conn, dtype=int)
    if hex_conn.shape[1] not in {8, 20, 27}:
        raise NotImplementedError("Only hex8/hex20/hex27 are supported.")
    face_patterns_corner = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    face_patterns_corner27 = [
        (0, 2, 8, 6),
        (18, 20, 26, 24),
        (0, 2, 20, 18),
        (6, 8, 26, 24),
        (0, 6, 24, 18),
        (2, 8, 26, 20),
    ]
    face_patterns_quad = [
        (0, 1, 2, 3, 8, 9, 10, 11),
        (4, 5, 6, 7, 12, 13, 14, 15),
        (0, 1, 5, 4, 8, 17, 12, 16),
        (1, 2, 6, 5, 9, 18, 13, 17),
        (2, 3, 7, 6, 10, 19, 14, 18),
        (3, 0, 4, 7, 11, 16, 15, 19),
    ]
    face_patterns_quad9 = [
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        (18, 19, 20, 21, 22, 23, 24, 25, 26),
        (0, 1, 2, 9, 10, 11, 18, 19, 20),
        (6, 7, 8, 15, 16, 17, 24, 25, 26),
        (0, 3, 6, 9, 12, 15, 18, 21, 24),
        (2, 5, 8, 11, 14, 17, 20, 23, 26),
    ]
    mapping_corner: dict[tuple[int, ...], int] = {}
    mapping_quad: dict[tuple[int, ...], int] = {}
    for e_id, elem in enumerate(hex_conn):
        if elem.shape[0] == 27:
            corner_patterns = face_patterns_corner27
        else:
            corner_patterns = face_patterns_corner
        for pattern in corner_patterns:
            face_nodes = tuple(sorted(int(elem[i]) for i in pattern))
            mapping_corner.setdefault(face_nodes, e_id)
        if elem.shape[0] == 20:
            for pattern in face_patterns_quad:
                face_nodes = tuple(sorted(int(elem[i]) for i in pattern))
                mapping_quad.setdefault(face_nodes, e_id)
        if elem.shape[0] == 27:
            for pattern in face_patterns_quad9:
                face_nodes = tuple(sorted(int(elem[i]) for i in pattern))
                mapping_quad.setdefault(face_nodes, e_id)
    facet_map = np.full((surface.conn.shape[0],), -1, dtype=int)
    for f_id, facet in enumerate(np.asarray(surface.conn, dtype=int)):
        key = tuple(sorted(int(n) for n in facet))
        if len(facet) == 4 and key in mapping_corner:
            facet_map[f_id] = mapping_corner[key]
        elif len(facet) == 8 and key in mapping_quad:
            facet_map[f_id] = mapping_quad[key]
        elif len(facet) == 9 and key in mapping_quad:
            facet_map[f_id] = mapping_quad[key]
        elif key in mapping_corner:
            facet_map[f_id] = mapping_corner[key]
    return facet_map


def _tet_shape_values(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    corner_coords = elem_coords[:4]
    M = np.stack([corner_coords[:, 0], corner_coords[:, 1], corner_coords[:, 2], np.ones(4)], axis=1)
    rhs = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    try:
        lam = np.linalg.solve(M.T, rhs)
    except np.linalg.LinAlgError:
        return np.zeros((elem_coords.shape[0],), dtype=float)
    if np.any(lam < -tol):
        return np.zeros((elem_coords.shape[0],), dtype=float)
    if elem_coords.shape[0] == 4:
        return lam
    if elem_coords.shape[0] != 10:
        raise NotImplementedError("tet shape evaluation supports tet4/tet10 only")
    L1, L2, L3, L4 = lam
    N1 = L1 * (2.0 * L1 - 1.0)
    N2 = L2 * (2.0 * L2 - 1.0)
    N3 = L3 * (2.0 * L3 - 1.0)
    N4 = L4 * (2.0 * L4 - 1.0)
    N5 = 4.0 * L1 * L2
    N6 = 4.0 * L2 * L3
    N7 = 4.0 * L1 * L3
    N8 = 4.0 * L1 * L4
    N9 = 4.0 * L2 * L4
    N10 = 4.0 * L3 * L4
    return np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10], dtype=float)


def _tet_gradN(elem_coords: np.ndarray, *, point: np.ndarray | None = None, tol: float) -> np.ndarray:
    corner_coords = elem_coords[:4]
    M = np.stack([corner_coords[:, 0], corner_coords[:, 1], corner_coords[:, 2], np.ones(4)], axis=1)
    try:
        invM = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.zeros((elem_coords.shape[0], 3), dtype=float)
    dL = invM[:3, :].T
    if elem_coords.shape[0] == 4:
        return dL
    if elem_coords.shape[0] != 10:
        raise NotImplementedError("tet grad evaluation supports tet4/tet10 only")
    if point is None:
        raise ValueError("tet10 grad evaluation requires point")
    rhs = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    try:
        lam = np.linalg.solve(M.T, rhs)
    except np.linalg.LinAlgError:
        return np.zeros((10, 3), dtype=float)
    if np.any(lam < -tol):
        return np.zeros((10, 3), dtype=float)
    L1, L2, L3, L4 = lam
    dL1, dL2, dL3, dL4 = dL
    dN1 = (4.0 * L1 - 1.0) * dL1
    dN2 = (4.0 * L2 - 1.0) * dL2
    dN3 = (4.0 * L3 - 1.0) * dL3
    dN4 = (4.0 * L4 - 1.0) * dL4
    dN5 = 4.0 * (L2 * dL1 + L1 * dL2)
    dN6 = 4.0 * (L3 * dL2 + L2 * dL3)
    dN7 = 4.0 * (L3 * dL1 + L1 * dL3)
    dN8 = 4.0 * (L4 * dL1 + L1 * dL4)
    dN9 = 4.0 * (L4 * dL2 + L2 * dL4)
    dN10 = 4.0 * (L4 * dL3 + L3 * dL4)
    return np.vstack([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])


def _tet_gradN_at_points(
    points: np.ndarray,
    elem_coords: np.ndarray,
    *,
    local: np.ndarray | None = None,
    tol: float,
) -> np.ndarray:
    n_nodes = elem_coords.shape[0]
    if n_nodes == 4:
        grad = _tet_gradN(elem_coords, tol=tol)
        grad_q = np.repeat(grad[None, :, :], points.shape[0], axis=0)
    elif n_nodes == 10:
        grad_q = np.array([_tet_gradN(elem_coords, point=pt, tol=tol) for pt in points], dtype=float)
    elif n_nodes == 8:
        grad_q = np.array([_hex8_gradN(pt, elem_coords, tol=tol) for pt in points], dtype=float)
    elif n_nodes == 20:
        grad_q = np.array([_hex20_gradN(pt, elem_coords, tol=tol) for pt in points], dtype=float)
    elif n_nodes == 27:
        grad_q = np.array([_hex27_gradN(pt, elem_coords, tol=tol) for pt in points], dtype=float)
    else:
        raise NotImplementedError("volume grad evaluation supports tet4/tet10/hex8/hex20/hex27 only")
    if local is not None:
        grad_q = grad_q[:, local, :]
    return grad_q


def _hex8_shape_and_local(
    point: np.ndarray,
    elem_coords: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, float, float, float]:
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    xi = 0.0
    eta = 0.0
    zeta = 0.0
    for _ in range(12):
        n = 0.125 * (1.0 + xi * signs[:, 0]) * (1.0 + eta * signs[:, 1]) * (1.0 + zeta * signs[:, 2])
        x = n @ elem_coords
        r = x - point
        if np.linalg.norm(r) < tol:
            break
        dN_dxi = 0.125 * signs[:, 0] * (1.0 + eta * signs[:, 1]) * (1.0 + zeta * signs[:, 2])
        dN_deta = 0.125 * signs[:, 1] * (1.0 + xi * signs[:, 0]) * (1.0 + zeta * signs[:, 2])
        dN_dzeta = 0.125 * signs[:, 2] * (1.0 + xi * signs[:, 0]) * (1.0 + eta * signs[:, 1])
        J = np.stack(
            [
                dN_dxi @ elem_coords,
                dN_deta @ elem_coords,
                dN_dzeta @ elem_coords,
            ],
            axis=1,
        )
        try:
            delta = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return np.zeros((8,), dtype=float), 0.0, 0.0, 0.0
        xi -= float(delta[0])
        eta -= float(delta[1])
        zeta -= float(delta[2])
    if max(abs(xi), abs(eta), abs(zeta)) > 1.0 + tol:
        return np.zeros((8,), dtype=float), xi, eta, zeta
    n = 0.125 * (1.0 + xi * signs[:, 0]) * (1.0 + eta * signs[:, 1]) * (1.0 + zeta * signs[:, 2])
    return n, xi, eta, zeta


def _hex8_shape_values(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    n, _, _, _ = _hex8_shape_and_local(point, elem_coords, tol=tol)
    return n


def _hex8_gradN(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    n, xi, eta, zeta = _hex8_shape_and_local(point, elem_coords, tol=tol)
    if np.allclose(n, 0.0):
        return np.zeros((8, 3), dtype=float)
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    dN_dxi = 0.125 * signs[:, 0] * (1.0 + eta * signs[:, 1]) * (1.0 + zeta * signs[:, 2])
    dN_deta = 0.125 * signs[:, 1] * (1.0 + xi * signs[:, 0]) * (1.0 + zeta * signs[:, 2])
    dN_dzeta = 0.125 * signs[:, 2] * (1.0 + xi * signs[:, 0]) * (1.0 + eta * signs[:, 1])
    J = np.stack(
        [
            dN_dxi @ elem_coords,
            dN_deta @ elem_coords,
            dN_dzeta @ elem_coords,
        ],
        axis=1,
    )
    try:
        invJ = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return np.zeros((8, 3), dtype=float)
    dN_dxi_eta = np.stack([dN_dxi, dN_deta, dN_dzeta], axis=1)  # (8,3)
    return dN_dxi_eta @ invJ


def _hex20_shape_ref(xi: float, eta: float, zeta: float) -> np.ndarray:
    s = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
    term = xi * sx + eta * sy + zeta * sz - 2.0
    n_corner = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta) * term

    def edge_x(sy, sz):
        return 0.25 * (1.0 - xi * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta)

    def edge_y(sx, sz):
        return 0.25 * (1.0 - eta * eta) * (1.0 + sx * xi) * (1.0 + sz * zeta)

    def edge_z(sx, sy):
        return 0.25 * (1.0 - zeta * zeta) * (1.0 + sx * xi) * (1.0 + sy * eta)

    n_edges = [
        edge_x(-1, -1),
        edge_y(1, -1),
        edge_x(1, -1),
        edge_y(-1, -1),
        edge_x(-1, 1),
        edge_y(1, 1),
        edge_x(1, 1),
        edge_y(-1, 1),
        edge_z(-1, -1),
        edge_z(1, -1),
        edge_z(1, 1),
        edge_z(-1, 1),
    ]

    return np.concatenate([n_corner, np.array(n_edges, dtype=float)], axis=0)


def _hex20_grad_ref(xi: float, eta: float, zeta: float) -> np.ndarray:
    s = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
    term = xi * sx + eta * sy + zeta * sz - 2.0

    dN_dxi_corner = (sx / 8.0) * (1.0 + sy * eta) * (1.0 + sz * zeta) * (term + (1.0 + sx * xi))
    dN_deta_corner = (sy / 8.0) * (1.0 + sx * xi) * (1.0 + sz * zeta) * (term + (1.0 + sy * eta))
    dN_dzeta_corner = (sz / 8.0) * (1.0 + sx * xi) * (1.0 + sy * eta) * (term + (1.0 + sz * zeta))
    d_corner = np.stack([dN_dxi_corner, dN_deta_corner, dN_dzeta_corner], axis=1)

    def d_edge_x(sy_val, sz_val):
        dxi = -0.5 * xi * (1.0 + sy_val * eta) * (1.0 + sz_val * zeta)
        deta = 0.25 * (1.0 - xi * xi) * sy_val * (1.0 + sz_val * zeta)
        dzeta = 0.25 * (1.0 - xi * xi) * (1.0 + sy_val * eta) * sz_val
        return np.array([dxi, deta, dzeta], dtype=float)

    def d_edge_y(sx_val, sz_val):
        dxi = 0.25 * (1.0 - eta * eta) * sx_val * (1.0 + sz_val * zeta)
        deta = -0.5 * eta * (1.0 + sx_val * xi) * (1.0 + sz_val * zeta)
        dzeta = 0.25 * (1.0 - eta * eta) * (1.0 + sx_val * xi) * sz_val
        return np.array([dxi, deta, dzeta], dtype=float)

    def d_edge_z(sx_val, sy_val):
        dxi = 0.25 * (1.0 - zeta * zeta) * sx_val * (1.0 + sy_val * eta)
        deta = 0.25 * (1.0 - zeta * zeta) * (1.0 + sx_val * xi) * sy_val
        dzeta = -0.5 * zeta * (1.0 + sx_val * xi) * (1.0 + sy_val * eta)
        return np.array([dxi, deta, dzeta], dtype=float)

    d_list = [
        d_edge_x(-1, -1),
        d_edge_y(1, -1),
        d_edge_x(1, -1),
        d_edge_y(-1, -1),
        d_edge_x(-1, 1),
        d_edge_y(1, 1),
        d_edge_x(1, 1),
        d_edge_y(-1, 1),
        d_edge_z(-1, -1),
        d_edge_z(1, -1),
        d_edge_z(1, 1),
        d_edge_z(-1, 1),
    ]

    d_edges = np.stack(d_list, axis=0)
    return np.concatenate([d_corner, d_edges], axis=0)


def _hex27_shape_ref(xi: float, eta: float, zeta: float) -> np.ndarray:
    def q1(t):
        return 0.5 * t * (t - 1.0)

    def q2(t):
        return 1.0 - t * t

    def q3(t):
        return 0.5 * t * (t + 1.0)

    Nx = [q1(xi), q2(xi), q3(xi)]
    Ny = [q1(eta), q2(eta), q3(eta)]
    Nz = [q1(zeta), q2(zeta), q3(zeta)]
    out = []
    for k in range(3):
        for j in range(3):
            for i in range(3):
                out.append(Nx[i] * Ny[j] * Nz[k])
    return np.array(out, dtype=float)


def _hex27_grad_ref(xi: float, eta: float, zeta: float) -> np.ndarray:
    def q1(t):
        return 0.5 * t * (t - 1.0)

    def q2(t):
        return 1.0 - t * t

    def q3(t):
        return 0.5 * t * (t + 1.0)

    def dq1(t):
        return t - 0.5

    def dq2(t):
        return -2.0 * t

    def dq3(t):
        return t + 0.5

    Nx = [q1(xi), q2(xi), q3(xi)]
    Ny = [q1(eta), q2(eta), q3(eta)]
    Nz = [q1(zeta), q2(zeta), q3(zeta)]
    dNx = [dq1(xi), dq2(xi), dq3(xi)]
    dNy = [dq1(eta), dq2(eta), dq3(eta)]
    dNz = [dq1(zeta), dq2(zeta), dq3(zeta)]
    out = []
    for k in range(3):
        for j in range(3):
            for i in range(3):
                dxi = dNx[i] * Ny[j] * Nz[k]
                deta = Nx[i] * dNy[j] * Nz[k]
                dzeta = Nx[i] * Ny[j] * dNz[k]
                out.append([dxi, deta, dzeta])
    return np.array(out, dtype=float)


def _hex20_shape_and_local(
    point: np.ndarray,
    elem_coords: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, float, float, float]:
    n8, xi, eta, zeta = _hex8_shape_and_local(point, elem_coords[:8], tol=tol)
    if np.allclose(n8, 0.0):
        return np.zeros((20,), dtype=float), 0.0, 0.0, 0.0
    if max(abs(xi), abs(eta), abs(zeta)) > 1.0 + tol:
        return np.zeros((20,), dtype=float), xi, eta, zeta
    n = _hex20_shape_ref(xi, eta, zeta)
    return n, xi, eta, zeta


def _hex20_shape_values(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    n, _, _, _ = _hex20_shape_and_local(point, elem_coords, tol=tol)
    return n


def _hex20_gradN(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    n, xi, eta, zeta = _hex20_shape_and_local(point, elem_coords, tol=tol)
    if np.allclose(n, 0.0):
        return np.zeros((20, 3), dtype=float)
    dN = _hex20_grad_ref(xi, eta, zeta)
    J = dN.T @ elem_coords
    try:
        invJ = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return np.zeros((20, 3), dtype=float)
    return dN @ invJ


def _hex27_shape_and_local(
    point: np.ndarray,
    elem_coords: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, float, float, float]:
    corner_ids = np.array([0, 2, 8, 6, 18, 20, 26, 24], dtype=int)
    corner_coords = elem_coords[corner_ids]
    n8, xi, eta, zeta = _hex8_shape_and_local(point, corner_coords, tol=tol)
    if np.allclose(n8, 0.0):
        return np.zeros((27,), dtype=float), 0.0, 0.0, 0.0
    if max(abs(xi), abs(eta), abs(zeta)) > 1.0 + tol:
        return np.zeros((27,), dtype=float), xi, eta, zeta
    n = _hex27_shape_ref(xi, eta, zeta)
    return n, xi, eta, zeta


def _hex27_shape_values(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    n, _, _, _ = _hex27_shape_and_local(point, elem_coords, tol=tol)
    return n


def _hex27_gradN(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    n, xi, eta, zeta = _hex27_shape_and_local(point, elem_coords, tol=tol)
    if np.allclose(n, 0.0):
        return np.zeros((27, 3), dtype=float)
    dN = _hex27_grad_ref(xi, eta, zeta)
    J = dN.T @ elem_coords
    try:
        invJ = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return np.zeros((27, 3), dtype=float)
    return dN @ invJ


def _volume_shape_values_at_points(
    points: np.ndarray,
    elem_coords: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    n_nodes = elem_coords.shape[0]
    if n_nodes in {4, 10}:
        return np.array([_tet_shape_values(pt, elem_coords, tol=tol) for pt in points], dtype=float)
    if n_nodes == 20:
        return np.array([_hex20_shape_values(pt, elem_coords, tol=tol) for pt in points], dtype=float)
    if n_nodes == 8:
        return np.array([_hex8_shape_values(pt, elem_coords, tol=tol) for pt in points], dtype=float)
    if n_nodes == 27:
        return np.array([_hex27_shape_values(pt, elem_coords, tol=tol) for pt in points], dtype=float)
    raise NotImplementedError("volume shape evaluation supports tet4/tet10/hex8/hex20/hex27 only")


def _local_indices(elem_nodes: np.ndarray, facet_nodes: np.ndarray) -> np.ndarray:
    index = {int(n): i for i, n in enumerate(elem_nodes)}
    try:
        return np.array([index[int(n)] for n in facet_nodes], dtype=int)
    except KeyError as exc:
        raise ValueError("facet nodes are not part of the element connectivity") from exc


def _surface_gradN(
    point: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    global _DEBUG_SURFACE_GRADN_COUNT
    pts = coords[facet_nodes]
    n = len(facet_nodes)
    debug = bool(_DEBUG_SURFACE_GRADN) and _DEBUG_SURFACE_GRADN_COUNT < _DEBUG_SURFACE_GRADN_MAX
    if n == 3:
        dN = np.array(
            [
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        dX_dxi = dN[:, 0] @ pts
        dX_deta = dN[:, 1] @ pts
        dN_lin = dN
    elif n == 4:
        values, xi, eta = _quad_shape_and_local(point, facet_nodes, coords, tol=tol)
        dN_dxi = np.array(
            [
                -0.25 * (1.0 - eta),
                0.25 * (1.0 - eta),
                0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            dtype=float,
        )
        dN_deta = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                0.25 * (1.0 + xi),
                0.25 * (1.0 - xi),
            ],
            dtype=float,
        )
        dX_dxi = dN_dxi @ pts
        dX_deta = dN_deta @ pts
        dN = np.stack([dN_dxi, dN_deta], axis=1)
        dN_lin = None
        if debug:
            n_sum = float(values.sum())
            x_phys = values @ pts
            n_raw = np.cross(dX_dxi, dX_deta)
            j_surf = float(np.linalg.norm(n_raw))
            print(
                "[fluxfem][surface_gradN][quad4]",
                f"pt={np.array2string(point, precision=6)}",
                f"xi={xi:.6f}",
                f"eta={eta:.6f}",
                f"N_sum={n_sum:.6e}",
                f"dN_dxi_sum={float(dN_dxi.sum()):.6e}",
                f"dN_deta_sum={float(dN_deta.sum()):.6e}",
                f"x_phys={np.array2string(x_phys, precision=6)}",
                f"t1={np.array2string(dX_dxi, precision=6)}",
                f"t2={np.array2string(dX_deta, precision=6)}",
                f"J_surf={j_surf:.6e}",
            )
            _DEBUG_SURFACE_GRADN_COUNT += 1
    elif n == 6:
        lam = _barycentric(point, pts[0], pts[1], pts[2])
        if lam is None:
            return np.zeros((6, 3), dtype=float)
        dN_lin = np.array(
            [
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        dX_dxi = dN_lin[:, 0] @ pts[:3]
        dX_deta = dN_lin[:, 1] @ pts[:3]
        dN = dN_lin
    elif n == 8:
        corner_nodes = facet_nodes[:4]
        values, xi, eta = _quad_shape_and_local(point, corner_nodes, coords, tol=tol)
        if np.allclose(values, 0.0):
            return np.zeros((8, 3), dtype=float)
        dN_dxi_corner = np.array(
            [
                -0.25 * (1.0 - eta),
                0.25 * (1.0 - eta),
                0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            dtype=float,
        )
        dN_deta_corner = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                0.25 * (1.0 + xi),
                0.25 * (1.0 - xi),
            ],
            dtype=float,
        )
        dX_dxi = dN_dxi_corner @ pts[:4]
        dX_deta = dN_deta_corner @ pts[:4]
        dN1_dxi = -0.25 * (1.0 - eta) * ((1.0 - xi) - (1.0 + xi + eta))
        dN1_deta = -0.25 * (1.0 - xi) * ((1.0 - eta) - (1.0 + xi + eta))
        dN2_dxi = 0.25 * (1.0 - eta) * ((1.0 + xi) - (1.0 - xi + eta))
        dN2_deta = -0.25 * (1.0 + xi) * ((1.0 - eta) - (1.0 - xi + eta))
        dN3_dxi = 0.25 * (1.0 + eta) * ((1.0 + xi) - (1.0 - xi - eta))
        dN3_deta = 0.25 * (1.0 + xi) * ((1.0 + eta) - (1.0 - xi - eta))
        dN4_dxi = -0.25 * (1.0 + eta) * ((1.0 - xi) - (1.0 + xi - eta))
        dN4_deta = 0.25 * (1.0 - xi) * ((1.0 + eta) - (1.0 + xi - eta))
        dN5_dxi = -xi * (1.0 - eta)
        dN5_deta = -0.5 * (1.0 - xi * xi)
        dN6_dxi = 0.5 * (1.0 - eta * eta)
        dN6_deta = -(1.0 + xi) * eta
        dN7_dxi = -xi * (1.0 + eta)
        dN7_deta = 0.5 * (1.0 - xi * xi)
        dN8_dxi = -0.5 * (1.0 - eta * eta)
        dN8_deta = -(1.0 - xi) * eta
        dN = np.array(
            [
                [dN1_dxi, dN1_deta],
                [dN2_dxi, dN2_deta],
                [dN3_dxi, dN3_deta],
                [dN4_dxi, dN4_deta],
                [dN5_dxi, dN5_deta],
                [dN6_dxi, dN6_deta],
                [dN7_dxi, dN7_deta],
                [dN8_dxi, dN8_deta],
            ],
            dtype=float,
        )
        if debug:
            values8 = _quad8_shape_values(xi, eta)
            n_sum = float(values8.sum())
            x_phys = values8 @ pts
            n_raw = np.cross(dX_dxi, dX_deta)
            j_surf = float(np.linalg.norm(n_raw))
            print(
                "[fluxfem][surface_gradN][quad8]",
                f"pt={np.array2string(point, precision=6)}",
                f"xi={xi:.6f}",
                f"eta={eta:.6f}",
                f"N_sum={n_sum:.6e}",
                f"dN_dxi_sum={float(dN[:, 0].sum()):.6e}",
                f"dN_deta_sum={float(dN[:, 1].sum()):.6e}",
                f"x_phys={np.array2string(x_phys, precision=6)}",
                f"t1={np.array2string(dX_dxi, precision=6)}",
                f"t2={np.array2string(dX_deta, precision=6)}",
                f"J_surf={j_surf:.6e}",
            )
            _DEBUG_SURFACE_GRADN_COUNT += 1
    elif n == 9:
        corner_nodes = facet_nodes[[0, 2, 8, 6]]
        values, xi, eta = _quad_shape_and_local(point, corner_nodes, coords, tol=tol)
        if np.allclose(values, 0.0):
            return np.zeros((9, 3), dtype=float)
        dN_dxi_corner = np.array(
            [
                -0.25 * (1.0 - eta),
                0.25 * (1.0 - eta),
                0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            dtype=float,
        )
        dN_deta_corner = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                0.25 * (1.0 + xi),
                0.25 * (1.0 - xi),
            ],
            dtype=float,
        )
        dX_dxi = dN_dxi_corner @ pts[:4]
        dX_deta = dN_deta_corner @ pts[:4]

        def q1(t):
            return 0.5 * t * (t - 1.0)

        def q2(t):
            return 1.0 - t * t

        def q3(t):
            return 0.5 * t * (t + 1.0)

        def dq1(t):
            return t - 0.5

        def dq2(t):
            return -2.0 * t

        def dq3(t):
            return t + 0.5

        Nx = [q1(xi), q2(xi), q3(xi)]
        Ny = [q1(eta), q2(eta), q3(eta)]
        dNx = [dq1(xi), dq2(xi), dq3(xi)]
        dNy = [dq1(eta), dq2(eta), dq3(eta)]
        dN = []
        for j in range(3):
            for i in range(3):
                dN_dxi = dNx[i] * Ny[j]
                dN_deta = Nx[i] * dNy[j]
                dN.append([dN_dxi, dN_deta])
        dN = np.array(dN, dtype=float)
        if debug:
            values9 = _quad9_shape_values(xi, eta)
            n_sum = float(values9.sum())
            x_phys = values9 @ pts
            n_raw = np.cross(dX_dxi, dX_deta)
            j_surf = float(np.linalg.norm(n_raw))
            print(
                "[fluxfem][surface_gradN][quad9]",
                f"pt={np.array2string(point, precision=6)}",
                f"xi={xi:.6f}",
                f"eta={eta:.6f}",
                f"N_sum={n_sum:.6e}",
                f"dN_dxi_sum={float(dN[:, 0].sum()):.6e}",
                f"dN_deta_sum={float(dN[:, 1].sum()):.6e}",
                f"x_phys={np.array2string(x_phys, precision=6)}",
                f"t1={np.array2string(dX_dxi, precision=6)}",
                f"t2={np.array2string(dX_deta, precision=6)}",
                f"J_surf={j_surf:.6e}",
            )
            _DEBUG_SURFACE_GRADN_COUNT += 1
    else:
        raise ValueError("facet must be a triangle or quad")

    J = np.stack([dX_dxi, dX_deta], axis=1)  # (3, 2)
    JTJ = J.T @ J
    if abs(np.linalg.det(JTJ)) < tol:
        return np.zeros((n, 3), dtype=float)
    M = J @ np.linalg.inv(JTJ)  # (3, 2)
    gradN = (M @ dN.T).T  # (n, 3)
    if n == 6:
        L1, L2, L3 = lam
        g1, g2, g3 = gradN[:3]
        gradN = np.array(
            [
                (4.0 * L1 - 1.0) * g1,
                (4.0 * L2 - 1.0) * g2,
                (4.0 * L3 - 1.0) * g3,
                4.0 * (L1 * g2 + L2 * g1),
                4.0 * (L2 * g3 + L3 * g2),
                4.0 * (L1 * g3 + L3 * g1),
            ],
            dtype=float,
        )
    return gradN


def _iter_supermesh_tris(coords: np.ndarray, conn: np.ndarray):
    for tri in conn:
        a, b, c = coords[tri]
        yield tri, a, b, c


def assemble_mortar_matrices(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    *,
    tol: float = 1e-8,
) -> tuple[MortarMatrix, MortarMatrix]:
    """
    Assemble mortar coupling matrices M_aa and M_ab using centroid quadrature.
    """
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)

    rows_aa: list[int] = []
    cols_aa: list[int] = []
    data_aa: list[float] = []

    rows_ab: list[int] = []
    cols_ab: list[int] = []
    data_ab: list[float] = []

    for (tri, a, b, c), fa, fb in zip(
        _iter_supermesh_tris(supermesh_coords, supermesh_conn),
        source_facets_a,
        source_facets_b,
    ):
        centroid = _tri_centroid(a, b, c)
        weight = _tri_area(a, b, c)
        if weight <= tol:
            continue

        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        Na = _facet_shape_values(centroid, facet_a, coords_a, tol=tol)
        Nb = _facet_shape_values(centroid, facet_b, coords_b, tol=tol)

        for i, node_i in enumerate(facet_a):
            for j, node_j in enumerate(facet_a):
                rows_aa.append(int(node_i))
                cols_aa.append(int(node_j))
                data_aa.append(weight * float(Na[i]) * float(Na[j]))

        for i, node_i in enumerate(facet_a):
            for j, node_j in enumerate(facet_b):
                rows_ab.append(int(node_i))
                cols_ab.append(int(node_j))
                data_ab.append(weight * float(Na[i]) * float(Nb[j]))

    n_a = int(np.asarray(surface_a.coords).shape[0])
    n_b = int(np.asarray(surface_b.coords).shape[0])
    M_aa = MortarMatrix(
        rows=np.asarray(rows_aa, dtype=int),
        cols=np.asarray(cols_aa, dtype=int),
        data=np.asarray(data_aa, dtype=float),
        shape=(n_a, n_a),
    )
    M_ab = MortarMatrix(
        rows=np.asarray(rows_ab, dtype=int),
        cols=np.asarray(cols_ab, dtype=int),
        data=np.asarray(data_ab, dtype=float),
        shape=(n_a, n_b),
    )
    return M_aa, M_ab


def assemble_mixed_surface_residual(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    res_form,
    u_a: np.ndarray,
    u_b: np.ndarray,
    params,
    *,
    value_dim_a: int = 1,
    value_dim_b: int = 1,
    offset_a: int = 0,
    offset_b: int | None = None,
    field_a: str = "a",
    field_b: str = "b",
    elem_conn_a: np.ndarray | None = None,
    elem_conn_b: np.ndarray | None = None,
    facet_to_elem_a: np.ndarray | None = None,
    facet_to_elem_b: np.ndarray | None = None,
    normal_source: str = "master",
    normal_from: str | None = None,
    master_field: str | None = None,
    normal_sign: float = 1.0,
    grad_source: str = "volume",
    dof_source: str = "surface",
    quad_order: int = 0,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Assemble mixed surface residual over a supermesh (centroid quadrature).

    normal_source can be "master", "slave", "a", "b", or "avg"; use master_field
    to pick which field acts as the master when normal_source is "master"/"slave".
    dof_source="volume" assembles into element nodes (requires elem_conn_* mappings).
    """
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    n_a = int(coords_a.shape[0] * value_dim_a)
    n_b = int(coords_b.shape[0] * value_dim_b)
    if offset_b is None:
        offset_b = offset_a + n_a
    n_total = int(offset_b + n_b)
    R = np.zeros((n_total,), dtype=float)

    normals_a = None
    normals_b = None
    if hasattr(surface_a, "facet_normals"):
        normals_a = surface_a.facet_normals()
    if hasattr(surface_b, "facet_normals"):
        normals_b = surface_b.facet_normals()

    area_scale = float(os.getenv("FLUXFEM_SMALL_TRI_EPS_SCALE", "0.0"))
    skip_small_tri = os.getenv("FLUXFEM_SKIP_SMALL_TRI", "0") == "1" and area_scale > 0.0
    facet_area_a = None
    facet_area_b = None
    if area_scale > 0.0:
        facet_area_a = np.array([_facet_area_estimate(fa, coords_a) for fa in facets_a], dtype=float)
        facet_area_b = np.array([_facet_area_estimate(fb, coords_b) for fb in facets_b], dtype=float)

    includes_measure = getattr(res_form, "_includes_measure", {})

    use_elem_a = elem_conn_a is not None and facet_to_elem_a is not None
    use_elem_b = elem_conn_b is not None and facet_to_elem_b is not None

    if grad_source not in {"volume", "surface"}:
        raise ValueError("grad_source must be 'volume' or 'surface'")
    if dof_source not in {"surface", "volume"}:
        raise ValueError("dof_source must be 'surface' or 'volume'")
    if dof_source == "volume" and grad_source == "surface":
        raise ValueError("dof_source 'volume' requires grad_source 'volume'")
    global _DEBUG_SURFACE_SOURCE_ONCE
    if grad_source == "surface" and not _DEBUG_SURFACE_SOURCE_ONCE:
        print("[fluxfem] using surface gradN in mortar")
        _DEBUG_SURFACE_SOURCE_ONCE = True

    if normal_from is not None:
        if normal_from not in {"master", "slave"}:
            raise ValueError("normal_from must be 'master' or 'slave'")
        master_name = field_a if master_field is None else master_field
        if master_name not in {field_a, field_b}:
            raise ValueError("master_field must match field_a or field_b")
        if normal_from == "master":
            normal_source = "a" if master_name == field_a else "b"
        else:
            normal_source = "b" if master_name == field_a else "a"
    if normal_source not in {"a", "b", "avg", "master", "slave"}:
        raise ValueError("normal_source must be 'a', 'b', 'avg', 'master', or 'slave'")
    if normal_source == "master":
        normal_source = "a" if (master_field is None or master_field == field_a) else "b"
    if normal_source == "slave":
        normal_source = "b" if (master_field is None or master_field == field_a) else "a"

    for (tri, a, b, c), fa, fb in zip(
        _iter_supermesh_tris(supermesh_coords, supermesh_conn),
        source_facets_a,
        source_facets_b,
    ):
        area = _tri_area(a, b, c)
        if area <= tol:
            continue
        if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
            area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
            if area_ref > 0.0 and area < area_scale * area_ref:
                continue
        detJ = 2.0 * area
        if quad_order <= 0:
            quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
            quad_w = np.array([0.5], dtype=float)
        else:
            quad_pts, quad_w = _tri_quadrature(quad_order)

        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)

        gradNa = None
        gradNb = None
        nodes_a = facet_a
        nodes_b = facet_b

        Na = None
        Nb = None

        elem_nodes_a = None
        elem_coords_a = None
        if use_elem_a:
            elem_id = int(facet_to_elem_a[int(fa)])
            if elem_id < 0:
                raise ValueError("facet_to_elem_a has invalid mapping")
            elem_nodes_a = np.asarray(elem_conn_a[elem_id], dtype=int)
            elem_coords_a = coords_a[elem_nodes_a]
            if elem_coords_a.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")

        elem_nodes_b = None
        elem_coords_b = None
        if use_elem_b:
            elem_id = int(facet_to_elem_b[int(fb)])
            if elem_id < 0:
                raise ValueError("facet_to_elem_b has invalid mapping")
            elem_nodes_b = np.asarray(elem_conn_b[elem_id], dtype=int)
            elem_coords_b = coords_b[elem_nodes_b]
            if elem_coords_b.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")

        if grad_source == "surface":
            gradNa = np.array(
                [_surface_gradN(pt, facet_a, coords_a, tol=tol) for pt in x_q],
                dtype=float,
            )
            gradNb = np.array(
                [_surface_gradN(pt, facet_b, coords_b, tol=tol) for pt in x_q],
                dtype=float,
            )
        if use_elem_a and grad_source == "volume":
            local = _local_indices(elem_nodes_a, facet_a)
            gradNa = _tet_gradN_at_points(x_q, elem_coords_a, local=local, tol=tol)

        if use_elem_b and grad_source == "volume":
            local = _local_indices(elem_nodes_b, facet_b)
            gradNb = _tet_gradN_at_points(x_q, elem_coords_b, local=local, tol=tol)

        if dof_source == "volume":
            if not use_elem_a or elem_nodes_a is None or elem_coords_a is None:
                raise ValueError("dof_source 'volume' requires elem_conn_a and facet_to_elem_a")
            if not use_elem_b or elem_nodes_b is None or elem_coords_b is None:
                raise ValueError("dof_source 'volume' requires elem_conn_b and facet_to_elem_b")
            nodes_a = elem_nodes_a
            nodes_b = elem_nodes_b
            Na = _volume_shape_values_at_points(x_q, elem_coords_a, tol=tol)
            Nb = _volume_shape_values_at_points(x_q, elem_coords_b, tol=tol)
            if grad_source == "volume":
                gradNa = _tet_gradN_at_points(x_q, elem_coords_a, tol=tol)
                gradNb = _tet_gradN_at_points(x_q, elem_coords_b, tol=tol)
        else:
            Na = np.array([_facet_shape_values(pt, facet_a, coords_a, tol=tol) for pt in x_q], dtype=float)
            Nb = np.array([_facet_shape_values(pt, facet_b, coords_b, tol=tol) for pt in x_q], dtype=float)

        normal = None
        na = normals_a[int(fa)] if normals_a is not None else None
        nb = normals_b[int(fb)] if normals_b is not None else None
        if normal_source == "a":
            normal = na
        elif normal_source == "b":
            normal = nb
        else:
            if na is not None and nb is not None:
                avg = na + nb
                norm = np.linalg.norm(avg)
                normal = avg / norm if norm > tol else na
            else:
                normal = na if na is not None else nb
        if normal is not None:
            normal = normal_sign * normal

        field_a_obj = SurfaceMixedFormField(
            N=Na,
            gradN=gradNa,
            value_dim=value_dim_a,
            basis=_SurfaceBasis(dofs_per_node=value_dim_a),
        )
        field_b_obj = SurfaceMixedFormField(
            N=Nb,
            gradN=gradNb,
            value_dim=value_dim_b,
            basis=_SurfaceBasis(dofs_per_node=value_dim_b),
        )
        fields = {
            field_a: FieldPair(test=field_a_obj, trial=field_a_obj),
            field_b: FieldPair(test=field_b_obj, trial=field_b_obj),
        }
        normal_q = None if normal is None else np.repeat(normal[None, :], quad_pts.shape[0], axis=0)
        ctx = SurfaceMixedFormContext(
            fields=fields,
            x_q=x_q,
            w=quad_w,
            detJ=np.array([detJ], dtype=float),
            normal=normal_q,
            trial_fields={field_a: field_a_obj, field_b: field_b_obj},
            test_fields={field_a: field_a_obj, field_b: field_b_obj},
            unknown_fields={field_a: field_a_obj, field_b: field_b_obj},
        )

        u_elem = {
            field_a: _gather_u_local(u_a, nodes_a, value_dim_a),
            field_b: _gather_u_local(u_b, nodes_b, value_dim_b),
        }
        fe_q = res_form(ctx, u_elem, params)
        for name, facet, value_dim, offset in (
            (field_a, nodes_a, value_dim_a, offset_a),
            (field_b, nodes_b, value_dim_b, offset_b),
        ):
            fe_field = fe_q[name]
            if fe_field.ndim != 2 or fe_field.shape[0] != ctx.x_q.shape[0]:
                raise ValueError("surface residual must return shape (n_q, n_ldofs) per field")
            if includes_measure.get(name, False):
                fe = np.sum(np.asarray(fe_field), axis=0)
            else:
                wJ = ctx.w * ctx.detJ
                fe = np.einsum("qi,q->i", np.asarray(fe_field), wJ)
            dofs = _global_dof_indices(facet, value_dim, int(offset))
            R[dofs] += fe
    return R


def assemble_mixed_surface_jacobian(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    res_form,
    u_a: np.ndarray,
    u_b: np.ndarray,
    params,
    *,
    value_dim_a: int = 1,
    value_dim_b: int = 1,
    offset_a: int = 0,
    offset_b: int | None = None,
    field_a: str = "a",
    field_b: str = "b",
    elem_conn_a: np.ndarray | None = None,
    elem_conn_b: np.ndarray | None = None,
    facet_to_elem_a: np.ndarray | None = None,
    facet_to_elem_b: np.ndarray | None = None,
    normal_source: str = "master",
    normal_from: str | None = None,
    master_field: str | None = None,
    normal_sign: float = 1.0,
    grad_source: str = "volume",
    dof_source: str = "surface",
    quad_order: int = 0,
    tol: float = 1e-8,
    sparse: bool = False,
):
    """
    Assemble mixed surface Jacobian over a supermesh (centroid quadrature).

    normal_source can be "master", "slave", "a", "b", or "avg"; use master_field
    to pick which field acts as the master when normal_source is "master"/"slave".
    dof_source="volume" assembles into element nodes (requires elem_conn_* mappings).
    """
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    n_a = int(coords_a.shape[0] * value_dim_a)
    n_b = int(coords_b.shape[0] * value_dim_b)
    if offset_b is None:
        offset_b = offset_a + n_a
    n_total = int(offset_b + n_b)

    normals_a = None
    normals_b = None
    if hasattr(surface_a, "facet_normals"):
        normals_a = surface_a.facet_normals()
    if hasattr(surface_b, "facet_normals"):
        normals_b = surface_b.facet_normals()

    area_scale = float(os.getenv("FLUXFEM_SMALL_TRI_EPS_SCALE", "0.0"))
    skip_small_tri = os.getenv("FLUXFEM_SKIP_SMALL_TRI", "0") == "1" and area_scale > 0.0
    facet_area_a = None
    facet_area_b = None
    if area_scale > 0.0:
        facet_area_a = np.array([_facet_area_estimate(fa, coords_a) for fa in facets_a], dtype=float)
        facet_area_b = np.array([_facet_area_estimate(fb, coords_b) for fb in facets_b], dtype=float)

    includes_measure = getattr(res_form, "_includes_measure", {})

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    K_dense = np.zeros((n_total, n_total), dtype=float) if not sparse else None

    use_elem_a = elem_conn_a is not None and facet_to_elem_a is not None
    use_elem_b = elem_conn_b is not None and facet_to_elem_b is not None

    if grad_source not in {"volume", "surface"}:
        raise ValueError("grad_source must be 'volume' or 'surface'")
    if dof_source not in {"surface", "volume"}:
        raise ValueError("dof_source must be 'surface' or 'volume'")
    if dof_source == "volume" and grad_source == "surface":
        raise ValueError("dof_source 'volume' requires grad_source 'volume'")
    global _DEBUG_SURFACE_SOURCE_ONCE
    if grad_source == "surface" and not _DEBUG_SURFACE_SOURCE_ONCE:
        print("[fluxfem] using surface gradN in mortar")
        _DEBUG_SURFACE_SOURCE_ONCE = True
    diag_map = os.getenv("FLUXFEM_DIAG_CONTACT_MAP", "0") == "1"
    diag_n = os.getenv("FLUXFEM_DIAG_CONTACT_N", "0") == "1"

    if normal_from is not None:
        if normal_from not in {"master", "slave"}:
            raise ValueError("normal_from must be 'master' or 'slave'")
        master_name = field_a if master_field is None else master_field
        if master_name not in {field_a, field_b}:
            raise ValueError("master_field must match field_a or field_b")
        if normal_from == "master":
            normal_source = "a" if master_name == field_a else "b"
        else:
            normal_source = "b" if master_name == field_a else "a"
    if normal_source not in {"a", "b", "avg", "master", "slave"}:
        raise ValueError("normal_source must be 'a', 'b', 'avg', 'master', or 'slave'")
    if normal_source == "master":
        normal_source = "a" if (master_field is None or master_field == field_a) else "b"
    if normal_source == "slave":
        normal_source = "b" if (master_field is None or master_field == field_a) else "a"

    for (tri, a, b, c), fa, fb in zip(
        _iter_supermesh_tris(supermesh_coords, supermesh_conn),
        source_facets_a,
        source_facets_b,
    ):
        area = _tri_area(a, b, c)
        if area <= tol:
            continue
        if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
            area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
            if area_ref > 0.0 and area < area_scale * area_ref:
                continue
        detJ = 2.0 * area
        if quad_order <= 0:
            quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
            quad_w = np.array([0.5], dtype=float)
        else:
            quad_pts, quad_w = _tri_quadrature(quad_order)

        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)

        gradNa = None
        gradNb = None
        nodes_a = facet_a
        nodes_b = facet_b

        Na = None
        Nb = None

        elem_nodes_a = None
        elem_coords_a = None
        local_a = None
        if use_elem_a:
            elem_id = int(facet_to_elem_a[int(fa)])
            if elem_id < 0:
                raise ValueError("facet_to_elem_a has invalid mapping")
            elem_nodes_a = np.asarray(elem_conn_a[elem_id], dtype=int)
            elem_coords_a = coords_a[elem_nodes_a]
            if elem_coords_a.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")

        elem_nodes_b = None
        elem_coords_b = None
        local_b = None
        if use_elem_b:
            elem_id = int(facet_to_elem_b[int(fb)])
            if elem_id < 0:
                raise ValueError("facet_to_elem_b has invalid mapping")
            elem_nodes_b = np.asarray(elem_conn_b[elem_id], dtype=int)
            elem_coords_b = coords_b[elem_nodes_b]
            if elem_coords_b.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")

        if grad_source == "surface":
            gradNa = np.array(
                [_surface_gradN(pt, facet_a, coords_a, tol=tol) for pt in x_q],
                dtype=float,
            )
            gradNb = np.array(
                [_surface_gradN(pt, facet_b, coords_b, tol=tol) for pt in x_q],
                dtype=float,
            )
        if use_elem_a and grad_source == "volume":
            local_a = _local_indices(elem_nodes_a, facet_a)
            gradNa = _tet_gradN_at_points(x_q, elem_coords_a, local=local_a, tol=tol)

        if use_elem_b and grad_source == "volume":
            local_b = _local_indices(elem_nodes_b, facet_b)
            gradNb = _tet_gradN_at_points(x_q, elem_coords_b, local=local_b, tol=tol)

        if dof_source == "volume":
            if not use_elem_a or elem_nodes_a is None or elem_coords_a is None:
                raise ValueError("dof_source 'volume' requires elem_conn_a and facet_to_elem_a")
            if not use_elem_b or elem_nodes_b is None or elem_coords_b is None:
                raise ValueError("dof_source 'volume' requires elem_conn_b and facet_to_elem_b")
            nodes_a = elem_nodes_a
            nodes_b = elem_nodes_b
            Na = _volume_shape_values_at_points(x_q, elem_coords_a, tol=tol)
            Nb = _volume_shape_values_at_points(x_q, elem_coords_b, tol=tol)
            if grad_source == "volume":
                gradNa = _tet_gradN_at_points(x_q, elem_coords_a, tol=tol)
                gradNb = _tet_gradN_at_points(x_q, elem_coords_b, tol=tol)
        else:
            Na = np.array([_facet_shape_values(pt, facet_a, coords_a, tol=tol) for pt in x_q], dtype=float)
            Nb = np.array([_facet_shape_values(pt, facet_b, coords_b, tol=tol) for pt in x_q], dtype=float)

        global _DEBUG_CONTACT_MAP_ONCE
        if diag_map and not _DEBUG_CONTACT_MAP_ONCE:
            elem_id_a = int(facet_to_elem_a[int(fa)]) if use_elem_a else -1
            elem_id_b = int(facet_to_elem_b[int(fb)]) if use_elem_b else -1
            print("[fluxfem][diag][contact-map] first facet")
            print(f"  fa={int(fa)} fb={int(fb)} elem_a={elem_id_a} elem_b={elem_id_b}")
            print(f"  facet_nodes_a={facet_a.tolist()}")
            print(f"  facet_nodes_b={facet_b.tolist()}")
            print(f"  facet_coords_a={coords_a[facet_a].tolist()}")
            print(f"  facet_coords_b={coords_b[facet_b].tolist()}")
            if elem_nodes_a is not None:
                if local_a is None:
                    local_a = _local_indices(elem_nodes_a, facet_a)
                match_a = np.all(elem_nodes_a[local_a] == facet_a)
                print(f"  elem_nodes_a={elem_nodes_a.tolist()}")
                print(f"  local_indices_a={local_a.tolist()} match={bool(match_a)}")
            if elem_nodes_b is not None:
                if local_b is None:
                    local_b = _local_indices(elem_nodes_b, facet_b)
                match_b = np.all(elem_nodes_b[local_b] == facet_b)
                print(f"  elem_nodes_b={elem_nodes_b.tolist()}")
                print(f"  local_indices_b={local_b.tolist()} match={bool(match_b)}")
            _DEBUG_CONTACT_MAP_ONCE = True

        global _DEBUG_CONTACT_N_ONCE
        if diag_n and not _DEBUG_CONTACT_N_ONCE:
            dofs_a = _global_dof_indices(nodes_a, value_dim_a, int(offset_a))
            dofs_b = _global_dof_indices(nodes_b, value_dim_b, int(offset_b))
            samples = min(3, Na.shape[0])
            print("[fluxfem][diag][contact-n] first facet q-points")
            print(f"  nodes_a={nodes_a.tolist()} nodes_b={nodes_b.tolist()}")
            print(f"  dofs_a={dofs_a.tolist()} dofs_b={dofs_b.tolist()}")
            for qi in range(samples):
                print(f"  q{qi} x={x_q[qi].tolist()} Na={Na[qi].tolist()} Nb={Nb[qi].tolist()}")
            _DEBUG_CONTACT_N_ONCE = True

        normal = None
        na = normals_a[int(fa)] if normals_a is not None else None
        nb = normals_b[int(fb)] if normals_b is not None else None
        if normal_source == "a":
            normal = na
        elif normal_source == "b":
            normal = nb
        else:
            if na is not None and nb is not None:
                avg = na + nb
                norm = np.linalg.norm(avg)
                normal = avg / norm if norm > tol else na
            else:
                normal = na if na is not None else nb
        if normal is not None:
            normal = normal_sign * normal

        field_a_obj = SurfaceMixedFormField(
            N=Na,
            gradN=gradNa,
            value_dim=value_dim_a,
            basis=_SurfaceBasis(dofs_per_node=value_dim_a),
        )
        field_b_obj = SurfaceMixedFormField(
            N=Nb,
            gradN=gradNb,
            value_dim=value_dim_b,
            basis=_SurfaceBasis(dofs_per_node=value_dim_b),
        )
        fields = {
            field_a: FieldPair(test=field_a_obj, trial=field_a_obj),
            field_b: FieldPair(test=field_b_obj, trial=field_b_obj),
        }
        normal_q = None if normal is None else np.repeat(normal[None, :], quad_pts.shape[0], axis=0)
        ctx = SurfaceMixedFormContext(
            fields=fields,
            x_q=x_q,
            w=quad_w,
            detJ=np.array([detJ], dtype=float),
            normal=normal_q,
            trial_fields={field_a: field_a_obj, field_b: field_b_obj},
            test_fields={field_a: field_a_obj, field_b: field_b_obj},
            unknown_fields={field_a: field_a_obj, field_b: field_b_obj},
        )

        u_elem = {
            field_a: _gather_u_local(u_a, nodes_a, value_dim_a),
            field_b: _gather_u_local(u_b, nodes_b, value_dim_b),
        }
        u_local = np.concatenate([u_elem[field_a], u_elem[field_b]], axis=0)
        sizes = (u_elem[field_a].shape[0], u_elem[field_b].shape[0])
        slices = {
            field_a: slice(0, sizes[0]),
            field_b: slice(sizes[0], sizes[0] + sizes[1]),
        }

        def _res_local(u_vec):
            u_dict = {name: u_vec[slices[name]] for name in (field_a, field_b)}
            fe_q = res_form(ctx, u_dict, params)
            res_parts = []
            for name in (field_a, field_b):
                fe_field = fe_q[name]
                if includes_measure.get(name, False):
                    fe = jnp.sum(jnp.asarray(fe_field), axis=0)
                else:
                    wJ = jnp.asarray(ctx.w) * jnp.asarray(ctx.detJ)
                    fe = jnp.einsum("qi,q->i", jnp.asarray(fe_field), wJ)
                res_parts.append(fe)
            return jnp.concatenate(res_parts, axis=0)

        J_local = jax.jacrev(_res_local)(jnp.asarray(u_local))
        J_local_np = np.asarray(J_local)

        dofs_a = _global_dof_indices(nodes_a, value_dim_a, int(offset_a))
        dofs_b = _global_dof_indices(nodes_b, value_dim_b, int(offset_b))
        dofs = np.concatenate([dofs_a, dofs_b], axis=0)
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                val = float(J_local_np[i, j])
                if sparse:
                    rows.append(int(gi))
                    cols.append(int(gj))
                    data.append(val)
                else:
                    K_dense[int(gi), int(gj)] += val

    if sparse:
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), np.asarray(data, dtype=float), n_total
    assert K_dense is not None
    return K_dense

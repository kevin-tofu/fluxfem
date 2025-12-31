from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .surface import SurfaceMesh


@dataclass(eq=False)
class MortarMatrix:
    """COO storage for mortar coupling matrices (can be rectangular)."""
    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]


def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _tri_centroid(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (a + b + c) / 3.0


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
    if n == 4:
        tri0 = (0, 1, 2)
        lam = _barycentric(point, pts[tri0[0]], pts[tri0[1]], pts[tri0[2]])
        if lam is not None and _point_in_tri(lam, tol=tol):
            out = np.zeros((4,), dtype=float)
            out[list(tri0)] = lam
            return out
        tri1 = (0, 2, 3)
        lam = _barycentric(point, pts[tri1[0]], pts[tri1[1]], pts[tri1[2]])
        if lam is not None:
            out = np.zeros((4,), dtype=float)
            out[list(tri1)] = lam
            return out
        return np.zeros((4,), dtype=float)
    raise ValueError("facet must be a triangle or quad")


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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .surface import SurfaceMesh


@dataclass(eq=False)
class SurfaceSupermesh:
    """Intersection supermesh for two surface meshes."""
    coords: np.ndarray
    conn: np.ndarray
    source_facets_a: np.ndarray
    source_facets_b: np.ndarray


def _polygon_area_2d(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _line_intersection(p1, p2, p3, p4, *, tol: float):
    d1 = p2 - p1
    d2 = p4 - p3
    denom = _cross2(d1, d2)
    if abs(denom) < tol:
        return p2
    t = _cross2(p3 - p1, d2) / denom
    return p1 + t * d1


def _sutherland_hodgman(subject: list[np.ndarray], clip: list[np.ndarray], *, tol: float):
    if not subject:
        return []
    orient = np.sign(_polygon_area_2d(np.array(clip)))
    if orient == 0:
        return []

    def inside(pt, a, b):
        return orient * _cross2(b - a, pt - a) >= -tol

    output = subject
    for i in range(len(clip)):
        input_list = output
        if not input_list:
            break
        output = []
        cp1 = clip[i]
        cp2 = clip[(i + 1) % len(clip)]
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output.append(_line_intersection(s, e, cp1, cp2, tol=tol))
                output.append(e)
            elif inside(s, cp1, cp2):
                output.append(_line_intersection(s, e, cp1, cp2, tol=tol))
            s = e
    return output


def _plane_basis(normal: np.ndarray):
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    t1 = np.cross(n, ref)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2, n


def _facet_plane(pts: np.ndarray, *, tol: float):
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < tol:
        return None, None
    n = n / n_norm
    d = -float(np.dot(n, pts[0]))
    return n, d


def _coplanar(pts_a: np.ndarray, pts_b: np.ndarray, *, tol: float) -> bool:
    n, d = _facet_plane(pts_a, tol=tol)
    if n is None:
        return False
    n2, d2 = _facet_plane(pts_b, tol=tol)
    if n2 is None:
        return False
    if abs(abs(np.dot(n, n2)) - 1.0) > 1e-4:
        return False
    dist_a = np.abs(pts_a @ n + d)
    dist_b = np.abs(pts_b @ n + d)
    return np.max(dist_a) <= tol and np.max(dist_b) <= tol


def _project(points: np.ndarray, origin: np.ndarray, t1: np.ndarray, t2: np.ndarray):
    rel = points - origin[None, :]
    x = rel @ t1
    y = rel @ t2
    return np.stack([x, y], axis=1)


def _unique_points(points: Iterable[np.ndarray], *, tol: float):
    scale = 1.0 / tol
    mapping: dict[tuple[int, int, int], int] = {}
    coords: list[np.ndarray] = []
    indices: list[int] = []
    for p in points:
        key = tuple(np.round(p * scale).astype(int))
        idx = mapping.get(key)
        if idx is None:
            idx = len(coords)
            mapping[key] = idx
            coords.append(p)
        indices.append(idx)
    return np.asarray(coords, dtype=float), indices


def build_surface_supermesh(
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    *,
    tol: float = 1e-8,
) -> SurfaceSupermesh:
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)

    all_coords: list[np.ndarray] = []
    all_conn: list[tuple[int, int, int]] = []
    src_a: list[int] = []
    src_b: list[int] = []

    for ia, fa in enumerate(facets_a):
        pts_a = coords_a[fa]
        min_a = pts_a.min(axis=0)
        max_a = pts_a.max(axis=0)
        for ib, fb in enumerate(facets_b):
            pts_b = coords_b[fb]
            if np.any(pts_b.max(axis=0) < min_a - tol) or np.any(pts_b.min(axis=0) > max_a + tol):
                continue
            if not _coplanar(pts_a, pts_b, tol=tol):
                continue

            n, _d = _facet_plane(pts_a, tol=tol)
            t1, t2, _ = _plane_basis(n)
            origin = pts_a[0]

            poly_a = _project(pts_a, origin, t1, t2)
            poly_b = _project(pts_b, origin, t1, t2)

            inter = _sutherland_hodgman(
                [p.copy() for p in poly_a],
                [p.copy() for p in poly_b],
                tol=tol,
            )
            if len(inter) < 3:
                continue
            inter_np = np.asarray(inter)
            if abs(_polygon_area_2d(inter_np)) <= tol:
                continue

            inter_3d = origin[None, :] + inter_np[:, 0:1] * t1 + inter_np[:, 1:2] * t2
            coords_local, idx = _unique_points(inter_3d, tol=tol)
            base = len(all_coords)
            for p in coords_local:
                all_coords.append(p)
            for i in range(1, len(idx) - 1):
                all_conn.append((base + idx[0], base + idx[i], base + idx[i + 1]))
                src_a.append(ia)
                src_b.append(ib)

    if not all_conn:
        return SurfaceSupermesh(
            coords=np.zeros((0, 3), dtype=float),
            conn=np.zeros((0, 3), dtype=int),
            source_facets_a=np.zeros((0,), dtype=int),
            source_facets_b=np.zeros((0,), dtype=int),
        )

    coords = np.asarray(all_coords, dtype=float)
    conn = np.asarray(all_conn, dtype=int)
    return SurfaceSupermesh(
        coords=coords,
        conn=conn,
        source_facets_a=np.asarray(src_a, dtype=int),
        source_facets_b=np.asarray(src_b, dtype=int),
    )

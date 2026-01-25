from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np

from .surface import SurfaceMesh

_SUPERMESH_CACHE: dict[tuple, "SurfaceSupermesh"] = {}


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
    n = None
    for i in range(len(pts) - 2):
        v1 = pts[i + 1] - pts[i]
        v2 = pts[i + 2] - pts[i]
        n_candidate = np.cross(v1, v2)
        n_norm = np.linalg.norm(n_candidate)
        if n_norm > tol:
            n = n_candidate / n_norm
            d = -float(np.dot(n, pts[i]))
            return n, d
    return None, None


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


def _facet_polygon_coords(coords: np.ndarray, facet: np.ndarray) -> np.ndarray:
    n = int(len(facet))
    if n == 9:
        corner = [0, 2, 8, 6]
        return coords[facet][corner]
    return coords[facet]


def _triangle_min_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    def angle(a, b, c):
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.arccos(cosang))

    return min(angle(p1, p0, p2), angle(p0, p1, p2), angle(p0, p2, p1))


def _triangulate_polygon(indices: list[int], poly2d: np.ndarray) -> list[tuple[int, int, int]]:
    n = len(indices)
    if n < 3:
        return []
    if n == 3:
        return [(indices[0], indices[1], indices[2])]
    if n == 4:
        p = poly2d
        diag_pref = os.getenv("FLUXFEM_SUPERMESH_QUAD_DIAG", "alt").lower()
        if diag_pref == "alt":
            return [(indices[0], indices[1], indices[3]), (indices[1], indices[2], indices[3])]
        if diag_pref == "fan":
            return [(indices[0], indices[1], indices[2]), (indices[0], indices[2], indices[3])]
        min_a = min(
            _triangle_min_angle(p[0], p[1], p[2]),
            _triangle_min_angle(p[0], p[2], p[3]),
        )
        min_b = min(
            _triangle_min_angle(p[0], p[1], p[3]),
            _triangle_min_angle(p[1], p[2], p[3]),
        )
        if min_b > min_a:
            return [(indices[0], indices[1], indices[3]), (indices[1], indices[2], indices[3])]
        return [(indices[0], indices[1], indices[2]), (indices[0], indices[2], indices[3])]
    tris = []
    for i in range(1, n - 1):
        tris.append((indices[0], indices[i], indices[i + 1]))
    return tris


def build_surface_supermesh(
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    *,
    tol: float = 1e-8,
    cache_enabled: bool | None = None,
    cache_trace: bool | None = None,
) -> SurfaceSupermesh:
    from ..solver.bc import facet_normals
    import hashlib

    if cache_enabled is None:
        cache_enabled = os.getenv("FLUXFEM_SUPERMESH_CACHE", "0") not in ("0", "", "false", "False")
    if cache_trace is None:
        cache_trace = os.getenv("FLUXFEM_SUPERMESH_CACHE_TRACE", "0") not in ("0", "", "false", "False")

    def _array_sig(arr: np.ndarray) -> tuple:
        arr_c = np.ascontiguousarray(arr)
        h = hashlib.blake2b(arr_c.view(np.uint8), digest_size=8).hexdigest()
        return (arr_c.shape, str(arr_c.dtype), h)
    if cache_enabled:
        global _SUPERMESH_CACHE
        try:
            _SUPERMESH_CACHE
        except NameError:
            _SUPERMESH_CACHE = {}
        key = (
            _array_sig(np.asarray(surface_a.coords)),
            _array_sig(np.asarray(surface_a.conn)),
            _array_sig(np.asarray(surface_b.coords)),
            _array_sig(np.asarray(surface_b.conn)),
            float(tol),
        )
        cached = _SUPERMESH_CACHE.get(key)
        if cached is not None:
            if cache_trace:
                print(f"[supermesh] cache hit n_tris={int(cached.conn.shape[0])}", flush=True)
            return cached

    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    normals_a = facet_normals(surface_a, outward_from=np.mean(coords_a, axis=0), normalize=True)

    all_coords: list[np.ndarray] = []
    all_conn: list[tuple[int, int, int]] = []
    src_a: list[int] = []
    src_b: list[int] = []

    for ia, fa in enumerate(facets_a):
        pts_a = _facet_polygon_coords(coords_a, fa)
        min_a = pts_a.min(axis=0)
        max_a = pts_a.max(axis=0)
        for ib, fb in enumerate(facets_b):
            pts_b = _facet_polygon_coords(coords_b, fb)
            if np.any(pts_b.max(axis=0) < min_a - tol) or np.any(pts_b.min(axis=0) > max_a + tol):
                continue
            if not _coplanar(pts_a, pts_b, tol=tol):
                continue

            n, _d = _facet_plane(pts_a, tol=tol)
            if n is not None:
                n_ref = normals_a[int(ia)]
                if np.dot(n, n_ref) < 0.0:
                    n = -n
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
            center = np.mean(inter_np, axis=0)
            angles = np.arctan2(inter_np[:, 1] - center[1], inter_np[:, 0] - center[0])
            order = np.argsort(angles)
            inter_np = inter_np[order]

            inter_3d = origin[None, :] + inter_np[:, 0:1] * t1 + inter_np[:, 1:2] * t2
            coords_local, idx = _unique_points(inter_3d, tol=tol)
            base = len(all_coords)
            for p in coords_local:
                all_coords.append(p)
            tris = _triangulate_polygon(idx, inter_np)
            for a_idx, b_idx, c_idx in tris:
                a_id = base + a_idx
                b_id = base + b_idx
                c_id = base + c_idx
                if n is not None:
                    pa = all_coords[a_id]
                    pb = all_coords[b_id]
                    pc = all_coords[c_id]
                    n_tri = np.cross(pb - pa, pc - pa)
                    if np.dot(n_tri, n) < 0.0:
                        b_id, c_id = c_id, b_id
                all_conn.append((a_id, b_id, c_id))
                src_a.append(ia)
                src_b.append(ib)

    if not all_conn:
        sm = SurfaceSupermesh(
            coords=np.zeros((0, 3), dtype=float),
            conn=np.zeros((0, 3), dtype=int),
            source_facets_a=np.zeros((0,), dtype=int),
            source_facets_b=np.zeros((0,), dtype=int),
        )
        if cache_enabled:
            _SUPERMESH_CACHE[key] = sm
        return sm

    coords = np.asarray(all_coords, dtype=float)
    conn = np.asarray(all_conn, dtype=int)
    sm = SurfaceSupermesh(
        coords=coords,
        conn=conn,
        source_facets_a=np.asarray(src_a, dtype=int),
        source_facets_b=np.asarray(src_b, dtype=int),
    )
    if cache_enabled:
        _SUPERMESH_CACHE[key] = sm
        if cache_trace:
            print(f"[supermesh] cache store n_tris={int(sm.conn.shape[0])}", flush=True)
    return sm

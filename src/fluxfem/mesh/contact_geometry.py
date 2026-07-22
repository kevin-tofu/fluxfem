from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable, cast

import jax.numpy as jnp
import numpy as np

from .surface import SurfaceMesh


@dataclass(eq=False)
class _SupermeshTriangleQuadratureCache:
    detJ: np.ndarray
    x_q: np.ndarray
    quad_pts: np.ndarray
    quad_w: np.ndarray


_DEBUG_SURFACE_GRADN = os.getenv("FLUXFEM_DEBUG_SURFACE_GRADN")
_DEBUG_SURFACE_GRADN_MAX = int(os.getenv("FLUXFEM_DEBUG_SURFACE_GRADN_MAX", "8")) if _DEBUG_SURFACE_GRADN else 0
_DEBUG_SURFACE_GRADN_COUNT = 0
_DEBUG_CONTACT_PROJ_ONCE = False
_DEBUG_PROJ_QP_CACHE = None
_DEBUG_PROJ_QP_SOURCE = None
_DEBUG_PROJ_QP_DUMPED = False
_PROJ_DIAG_STATS: dict[str, Any] | None = None
_PROJ_DIAG_COUNT = 0
_PROJ_DIAG_CONTEXT: dict[str, int | str] = {}

def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Public wrapper for triangle area (used in contact diagnostics)."""
    return _tri_area(a, b, c)


def build_supermesh_triangle_quadrature_cache(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    *,
    quad_order: int,
    tol: float,
) -> _SupermeshTriangleQuadratureCache:
    if quad_order <= 0:
        quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        quad_w = np.array([0.5], dtype=float)
    else:
        quad_pts, quad_w = _tri_quadrature(quad_order)
    conn = np.asarray(supermesh_conn, dtype=int)
    coords = np.asarray(supermesh_coords, dtype=float)
    n_tri = int(conn.shape[0])
    n_q = int(quad_pts.shape[0])
    detJ = np.zeros((n_tri,), dtype=float)
    x_q = np.zeros((n_tri, n_q, 3), dtype=float)
    for i, tri in enumerate(conn):
        a, b, c = coords[tri]
        area = _tri_area(a, b, c)
        if area <= tol:
            continue
        detJ[i] = 2.0 * area
        x_q[i] = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)
    return _SupermeshTriangleQuadratureCache(
        detJ=detJ,
        x_q=x_q,
        quad_pts=quad_pts,
        quad_w=quad_w,
    )


def tri_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Public wrapper for triangle quadrature."""
    return _tri_quadrature(order)


def facet_triangles(coords: np.ndarray, facet_nodes: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Public wrapper for facet triangulation."""
    return _facet_triangles(coords, facet_nodes)


def facet_shape_values(point: np.ndarray, facet_nodes: np.ndarray, coords: np.ndarray, *, tol: float) -> np.ndarray:
    """Public wrapper for facet shape values at a point."""
    return _facet_shape_values(point, facet_nodes, coords, tol=tol)


def volume_shape_values_at_points(x_q: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    """Public wrapper for volume shape values at quadrature points."""
    return _volume_shape_values_at_points(x_q, elem_coords, tol=tol)


def quad_shape_and_local(
    point: np.ndarray,
    quad_nodes: np.ndarray,
    corner_coords: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, float, float]:
    """Public wrapper for quad shape values and local coordinates."""
    return _quad_shape_and_local(point, quad_nodes, corner_coords, tol=tol)


def quad9_shape_values(xi: float, eta: float) -> np.ndarray:
    """Public wrapper for quad9 shape values."""
    return _quad9_shape_values(xi, eta)


def hex27_gradN(point: np.ndarray, elem_coords: np.ndarray, *, tol: float) -> np.ndarray:
    """Public wrapper for hex27 gradN (diagnostics)."""
    return _hex27_gradN(point, elem_coords, tol=tol)


def _quad_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order <= 1:
        order = 2
    n = int(np.ceil((order + 1.0) / 2.0))
    x1d, w1d = np.polynomial.legendre.leggauss(n)
    X: np.ndarray
    Y: np.ndarray
    X, Y = np.meshgrid(x1d, x1d, indexing="xy")
    W = np.outer(w1d, w1d)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    w = W.ravel()
    return pts, w


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
        corner_nodes = facet_nodes[[2, 0, 6, 8]]
        pts = coords[corner_nodes]
        return _tri_area(pts[0], pts[1], pts[2]) + _tri_area(pts[0], pts[2], pts[3])
    pts = coords[facet_nodes]
    area = 0.0
    p0 = pts[0]
    for i in range(1, len(pts) - 1):
        area += _tri_area(p0, pts[i], pts[i + 1])
    return float(area)


def _facet_triangles(coords: np.ndarray, facet_nodes: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = int(len(facet_nodes))
    if n in {3, 6}:
        corner = facet_nodes[:3]
        pts = coords[corner]
        return [(pts[0], pts[1], pts[2])]
    if n == 4:
        corner = facet_nodes
    elif n == 8:
        corner = facet_nodes[:4]
    elif n == 9:
        corner = facet_nodes[[2, 0, 6, 8]]
    else:
        corner = facet_nodes
    pts = coords[corner]
    if len(pts) < 3:
        return []
    if len(pts) == 3:
        return [(pts[0], pts[1], pts[2])]
    tris = [(pts[0], pts[1], pts[2])]
    if len(pts) >= 4:
        tris.append((pts[0], pts[2], pts[3]))
    if len(pts) > 4:
        for i in range(2, len(pts) - 1):
            tris.append((pts[0], pts[i], pts[i + 1]))
    return tris




def _tri_centroid(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (a + b + c) / 3.0


def _tri3_shape_values_jax(
    point: jnp.ndarray,
    facet_nodes: np.ndarray,
    coords: jnp.ndarray,
) -> jnp.ndarray:
    pts = coords[facet_nodes]
    a = pts[0]
    b = pts[1]
    c = pts[2]
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = jnp.dot(v0, v0)
    d01 = jnp.dot(v0, v1)
    d11 = jnp.dot(v1, v1)
    d20 = jnp.dot(v2, v0)
    d21 = jnp.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    denom_safe = jnp.where(jnp.abs(denom) < 1e-14, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom_safe
    w = (d00 * d21 - d01 * d20) / denom_safe
    u = 1.0 - v - w
    lam = jnp.stack([u, v, w])
    return jnp.where(jnp.abs(denom) < 1e-14, jnp.zeros_like(lam), lam)


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
        weights *= 0.5
        return pts, weights
    raise NotImplementedError("triangle quadrature order > 5 is not implemented")


def _proj_diag_enabled() -> bool:
    return os.getenv("FLUXFEM_PROJ_DIAG", "0") == "1"


def _proj_diag_max() -> int:
    return int(os.getenv("FLUXFEM_PROJ_DIAG_MAX", "20"))


def _proj_diag_reset() -> None:
    global _PROJ_DIAG_STATS, _PROJ_DIAG_COUNT
    _PROJ_DIAG_STATS = {
        "total": 0,
        "fail": 0,
        "by_code": {},
    }
    _PROJ_DIAG_COUNT = 0


def _proj_diag_set_context(
    *,
    fa: int,
    fb: int,
    face_a: str,
    face_b: str,
    elem_a: int,
    elem_b: int,
) -> None:
    _PROJ_DIAG_CONTEXT.clear()
    _PROJ_DIAG_CONTEXT.update(
        {
            "fa": int(fa),
            "fb": int(fb),
            "face_a": face_a,
            "face_b": face_b,
            "elem_a": int(elem_a),
            "elem_b": int(elem_b),
        }
    )


def _proj_diag_attempt() -> None:
    if _PROJ_DIAG_STATS is None:
        return
    _PROJ_DIAG_STATS["total"] += 1


def _proj_diag_log(
    code: str,
    *,
    iters: int,
    res_norm: float,
    delta_norm: float | None,
    detJ: float | None,
    point: np.ndarray,
    local: np.ndarray,
    in_ref_domain: bool,
) -> None:
    global _PROJ_DIAG_COUNT
    if _PROJ_DIAG_STATS is None:
        return
    _PROJ_DIAG_STATS["fail"] += 1
    by_code = cast(dict[str, int], _PROJ_DIAG_STATS["by_code"])
    by_code[code] = by_code.get(code, 0) + 1
    if _PROJ_DIAG_COUNT >= _proj_diag_max():
        return
    _PROJ_DIAG_COUNT += 1
    ctx = " ".join(f"{k}={v}" for k, v in _PROJ_DIAG_CONTEXT.items()) if _PROJ_DIAG_CONTEXT else "ctx=unknown"
    det_str = "None" if detJ is None else f"{detJ:.6e}"
    delta_str = "None" if delta_norm is None else f"{delta_norm:.6e}"
    print(
        "[fluxfem][proj][fail]",
        f"code={code}",
        ctx,
        f"iters={iters}",
        f"res={res_norm:.6e}",
        f"delta={delta_str}",
        f"detJ={det_str}",
        f"in_ref={bool(in_ref_domain)}",
        f"point={point.tolist()}",
        f"local={local.tolist()}",
    )


def _proj_diag_report() -> None:
    if _PROJ_DIAG_STATS is None:
        return
    total = _PROJ_DIAG_STATS["total"]
    fail = _PROJ_DIAG_STATS["fail"]
    by_code = _PROJ_DIAG_STATS["by_code"]
    print("[fluxfem][proj][diag] total=", total, "fail=", fail, "by_code=", by_code)


def _facet_label(facet: np.ndarray) -> str:
    n = int(len(facet))
    if n == 3:
        return "tri3"
    if n == 4:
        return "quad4"
    if n == 6:
        return "tri6"
    if n == 8:
        return "quad8"
    if n == 9:
        return "quad9"
    return f"n{n}"


def _diag_quad_override(diag_force: bool, mode: str, path: str) -> tuple[np.ndarray, np.ndarray] | None:
    global _DEBUG_PROJ_QP_CACHE, _DEBUG_PROJ_QP_SOURCE
    if not diag_force or mode != "load" or not path:
        return None
    if _DEBUG_PROJ_QP_CACHE is None:
        data = np.load(path)
        _DEBUG_PROJ_QP_CACHE = (np.asarray(data["quad_pts"], dtype=float), np.asarray(data["quad_w"], dtype=float))
        _DEBUG_PROJ_QP_SOURCE = f"file:{path}"
    return _DEBUG_PROJ_QP_CACHE


def _diag_quad_dump(diag_force: bool, mode: str, path: str, quad_pts: np.ndarray, quad_w: np.ndarray) -> None:
    global _DEBUG_PROJ_QP_DUMPED
    if not diag_force or mode != "dump" or not path or _DEBUG_PROJ_QP_DUMPED:
        return
    np.savez(path, quad_pts=np.asarray(quad_pts, dtype=float), quad_w=np.asarray(quad_w, dtype=float))
    _DEBUG_PROJ_QP_DUMPED = True


def _volume_local_coords(point: np.ndarray, elem_coords: np.ndarray, *, tol: float):
    n_nodes = elem_coords.shape[0]
    if n_nodes in {4, 10}:
        corner_coords = elem_coords[:4]
        M = np.stack([corner_coords[:, 0], corner_coords[:, 1], corner_coords[:, 2], np.ones(4)], axis=1)
        rhs = np.array([point[0], point[1], point[2], 1.0], dtype=float)
        try:
            lam = np.linalg.solve(M.T, rhs)
        except np.linalg.LinAlgError:
            return None
        return lam
    if n_nodes == 8:
        _, xi, eta, zeta = _hex8_shape_and_local(point, elem_coords, tol=tol)
        return np.array([xi, eta, zeta], dtype=float)
    if n_nodes == 20:
        _, xi, eta, zeta = _hex20_shape_and_local(point, elem_coords, tol=tol)
        return np.array([xi, eta, zeta], dtype=float)
    if n_nodes == 27:
        _, xi, eta, zeta = _hex27_shape_and_local(point, elem_coords, tol=tol)
        return np.array([xi, eta, zeta], dtype=float)
    return None


def _diag_contact_projection(
    *,
    fa: int,
    fb: int,
    quad_pts: np.ndarray,
    quad_w: np.ndarray,
    x_q: np.ndarray,
    Na: np.ndarray,
    Nb: np.ndarray,
    nodes_a: np.ndarray,
    nodes_b: np.ndarray,
    dofs_a: np.ndarray,
    dofs_b: np.ndarray,
    elem_coords_a: np.ndarray | None,
    elem_coords_b: np.ndarray | None,
    na: np.ndarray | None,
    nb: np.ndarray | None,
    normal: np.ndarray | None,
    normal_source: str,
    normal_sign: float,
    detJ: float,
    diag_facet: int,
    diag_max_q: int,
    guard: bool,
    skip_nonfinite: bool,
    quad_source: str,
    tol: float,
) -> None:
    global _DEBUG_CONTACT_PROJ_ONCE
    if _DEBUG_CONTACT_PROJ_ONCE:
        return
    if diag_facet >= 0 and fa != diag_facet:
        return
    samples = min(diag_max_q, int(x_q.shape[0]))
    print("[fluxfem][diag][proj] first facet")
    print(f"  fa={fa} fb={fb} quad_source={quad_source}")
    print(f"  quad_pts={quad_pts.tolist()} quad_w={quad_w.tolist()}")
    print(f"  normal_source={normal_source} normal_sign={normal_sign}")
    print(f"  n_master={None if na is None else na.tolist()}")
    print(f"  n_slave={None if nb is None else nb.tolist()}")
    print(f"  n_used={None if normal is None else normal.tolist()}")
    if normal is not None and na is not None:
        print(f"  dot(n_used,n_master)={float(np.dot(normal, na)):.6e}")
    if normal is not None and nb is not None:
        print(f"  dot(n_used,n_slave)={float(np.dot(normal, nb)):.6e}")
    print(f"  detJ={float(detJ):.6e}")
    print(f"  nodes_a={nodes_a.tolist()} nodes_b={nodes_b.tolist()}")
    print(f"  dofs_a={dofs_a.tolist()} dofs_b={dofs_b.tolist()}")
    for qi in range(samples):
        nsum_a = float(np.sum(Na[qi]))
        nsum_b = float(np.sum(Nb[qi]))
        xq = x_q[qi]
        msg = f"  q{qi} x={xq.tolist()} sum(Na)={nsum_a:.6e} sum(Nb)={nsum_b:.6e}"
        if elem_coords_a is not None:
            xa = Na[qi] @ elem_coords_a
            msg += f" x_a={xa.tolist()} |x_a-x_q|={float(np.linalg.norm(xa - xq)):.6e}"
            local_a = _volume_local_coords(xq, elem_coords_a, tol=tol)
            if local_a is not None:
                msg += f" xi_a={local_a.tolist()}"
        if elem_coords_b is not None:
            xb = Nb[qi] @ elem_coords_b
            msg += f" x_b={xb.tolist()} |x_b-x_q|={float(np.linalg.norm(xb - xq)):.6e}"
            local_b = _volume_local_coords(xq, elem_coords_b, tol=tol)
            if local_b is not None:
                msg += f" xi_b={local_b.tolist()}"
        print(msg)
    _DEBUG_CONTACT_PROJ_ONCE = True


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
    return bool(np.all(lam >= -tol) and np.all(lam <= 1.0 + tol))


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
    if _proj_diag_enabled():
        _proj_diag_attempt()
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
    res_norm = 0.0
    detJ = None
    iters = 0
    for _ in range(12):
        iters += 1
        n1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        n2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        n3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        n4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        x_m = n1 * x[0] + n2 * x[1] + n3 * x[2] + n4 * x[3]
        y_m = n1 * y[0] + n2 * y[1] + n3 * y[2] + n4 * y[3]
        rx = x_m - xp
        ry = y_m - yp
        res_norm = float(np.hypot(rx, ry))
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
        detJ = float(det)
        if abs(det) < tol:
            if _proj_diag_enabled():
                _proj_diag_log(
                    "SINGULAR_H",
                    iters=iters,
                    res_norm=res_norm,
                    delta_norm=None,
                    detJ=detJ,
                    point=point,
                    local=np.array([xi, eta], dtype=float),
                    in_ref_domain=False,
                )
            return np.zeros((4,), dtype=float), xi, eta
        dxi = (-j22 * rx + j12 * ry) / det
        deta = (j21 * rx - j11 * ry) / det
        xi += dxi
        eta += deta
        if not np.isfinite(xi) or not np.isfinite(eta):
            if _proj_diag_enabled():
                _proj_diag_log(
                    "NAN_INF",
                    iters=iters,
                    res_norm=res_norm,
                    delta_norm=float(np.hypot(dxi, deta)),
                    detJ=detJ,
                    point=point,
                    local=np.array([xi, eta], dtype=float),
                    in_ref_domain=False,
                )
            return np.zeros((4,), dtype=float), 0.0, 0.0

    in_ref = max(abs(xi), abs(eta)) <= 1.0 + tol
    if _proj_diag_enabled() and (not in_ref or res_norm > tol):
        code = "OUTSIDE_DOMAIN" if not in_ref else "NEWTON_NO_CONVERGE"
        _proj_diag_log(
            code,
            iters=iters,
            res_norm=res_norm,
            delta_norm=None,
            detJ=detJ,
            point=point,
            local=np.array([xi, eta], dtype=float),
            in_ref_domain=in_ref,
        )

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


def _quad9_shape_grad_ref(xi: float, eta: float) -> np.ndarray:
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
    out = []
    for j in range(3):
        for i in range(3):
            out.append([dNx[i] * Ny[j], Nx[i] * dNy[j]])
    return np.array(out, dtype=float)


def _quad9_map_and_jacobian(pts: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    N = _quad9_shape_values(xi, eta)
    dN = _quad9_shape_grad_ref(xi, eta)
    x = N @ pts
    J = (dN.T @ pts).T  # (3,2)
    return x, J


def _project_point_to_quad9(
    point: np.ndarray,
    pts: np.ndarray,
    *,
    tol: float,
    max_iter: int = 15,
) -> tuple[float, float, bool, np.ndarray, np.ndarray, dict]:
    xi0 = 0.0
    eta0 = 0.0
    xi = xi0
    eta = eta0
    last_delta = np.array([np.nan, np.nan], dtype=float)
    last_r = np.array([np.nan, np.nan], dtype=float)
    last_det = np.nan
    status = "OK"
    for _ in range(max_iter):
        x, J = _quad9_map_and_jacobian(pts, xi, eta)
        JTJ = J.T @ J
        det = float(np.linalg.det(JTJ))
        last_det = det
        if abs(det) < tol:
            status = "SINGULAR_H"
            return xi, eta, False, x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, det, JTJ)
        r = J.T @ (x - point)
        last_r = r
        try:
            delta = -np.linalg.solve(JTJ, r)
        except np.linalg.LinAlgError:
            status = "SINGULAR_H"
            return xi, eta, False, x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, det, JTJ)
        if not np.all(np.isfinite(delta)):
            status = "NAN_INF"
            return xi, eta, False, x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, det, JTJ)
        last_delta = delta
        step = float(np.max(np.abs(delta)))
        if step > 1.0:
            delta = delta / step
        xi += float(delta[0])
        eta += float(delta[1])
        if float(np.linalg.norm(delta)) < tol and float(np.linalg.norm(r)) < tol:
            break
    x, J = _quad9_map_and_jacobian(pts, xi, eta)
    ok = abs(xi) <= 1.0 + tol and abs(eta) <= 1.0 + tol
    if not ok:
        status = "OUTSIDE_DOMAIN"
    if status == "OK" and (float(np.linalg.norm(last_delta)) >= tol or float(np.linalg.norm(last_r)) >= tol):
        status = "NEWTON_NO_CONVERGE"
    return xi, eta, ok and status == "OK", x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, last_det, J.T @ J)


def _tri6_shape_values(xi: float, eta: float) -> np.ndarray:
    L1 = 1.0 - xi - eta
    L2 = xi
    L3 = eta
    return np.array(
        [
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            4.0 * L1 * L2,
            4.0 * L2 * L3,
            4.0 * L1 * L3,
        ],
        dtype=float,
    )


def _tri6_shape_grad_ref(xi: float, eta: float) -> np.ndarray:
    L1 = 1.0 - xi - eta
    L2 = xi
    L3 = eta
    dN1 = np.array([-(4.0 * L1 - 1.0), -(4.0 * L1 - 1.0)], dtype=float)
    dN2 = np.array([4.0 * L2 - 1.0, 0.0], dtype=float)
    dN3 = np.array([0.0, 4.0 * L3 - 1.0], dtype=float)
    dN4 = np.array([4.0 * (L1 - L2), -4.0 * L2], dtype=float)
    dN5 = np.array([4.0 * L3, 4.0 * L2], dtype=float)
    dN6 = np.array([-4.0 * L3, 4.0 * (L1 - L3)], dtype=float)
    return np.array([dN1, dN2, dN3, dN4, dN5, dN6], dtype=float)


def _tri6_map_and_jacobian(pts: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    N = _tri6_shape_values(xi, eta)
    dN = _tri6_shape_grad_ref(xi, eta)
    x = N @ pts
    J = (dN.T @ pts).T  # (3,2)
    return x, J


def _projection_info(
    status: str,
    xi0: float,
    eta0: float,
    xi: float,
    eta: float,
    r: np.ndarray,
    delta: np.ndarray,
    det: float,
    JTJ: np.ndarray,
) -> dict:
    r_norm = float(np.linalg.norm(r)) if r.size else float("nan")
    d_norm = float(np.linalg.norm(delta)) if delta.size else float("nan")
    cond = float(np.linalg.cond(JTJ)) if JTJ.size and np.isfinite(JTJ).all() else float("nan")
    return {
        "status": status,
        "xi0": float(xi0),
        "eta0": float(eta0),
        "xi": float(xi),
        "eta": float(eta),
        "r_norm": r_norm,
        "d_norm": d_norm,
        "det": float(det),
        "cond": cond,
    }


def _project_point_to_tri6(
    point: np.ndarray,
    pts: np.ndarray,
    *,
    tol: float,
    max_iter: int = 15,
) -> tuple[float, float, bool, np.ndarray, np.ndarray, dict]:
    xi0 = 1.0 / 3.0
    eta0 = 1.0 / 3.0
    xi = xi0
    eta = eta0
    last_delta = np.array([np.nan, np.nan], dtype=float)
    last_r = np.array([np.nan, np.nan], dtype=float)
    last_det = np.nan
    status = "OK"
    for _ in range(max_iter):
        x, J = _tri6_map_and_jacobian(pts, xi, eta)
        JTJ = J.T @ J
        det = float(np.linalg.det(JTJ))
        last_det = det
        if abs(det) < tol:
            status = "SINGULAR_H"
            return xi, eta, False, x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, det, JTJ)
        r = J.T @ (x - point)
        last_r = r
        try:
            delta = -np.linalg.solve(JTJ, r)
        except np.linalg.LinAlgError:
            status = "SINGULAR_H"
            return xi, eta, False, x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, det, JTJ)
        if not np.all(np.isfinite(delta)):
            status = "NAN_INF"
            return xi, eta, False, x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, det, JTJ)
        last_delta = delta
        step = float(np.max(np.abs(delta)))
        if step > 1.0:
            delta = delta / step
        xi += float(delta[0])
        eta += float(delta[1])
        if float(np.linalg.norm(delta)) < tol and float(np.linalg.norm(r)) < tol:
            break
    x, J = _tri6_map_and_jacobian(pts, xi, eta)
    ok = xi >= -tol and eta >= -tol and (xi + eta) <= 1.0 + tol
    if not ok:
        status = "OUTSIDE_DOMAIN"
    if status == "OK" and (float(np.linalg.norm(last_delta)) >= tol or float(np.linalg.norm(last_r)) >= tol):
        status = "NEWTON_NO_CONVERGE"
    return xi, eta, ok and status == "OK", x, J, _projection_info(status, xi0, eta0, xi, eta, last_r, last_delta, last_det, J.T @ J)


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
        corner_nodes = facet_nodes[[2, 0, 6, 8]]
        values, xi, eta = _quad_shape_and_local(point, corner_nodes, coords, tol=tol)
        if np.allclose(values, 0.0):
            return np.zeros((9,), dtype=float)
        return _quad9_shape_values(xi, eta)
    raise ValueError("facet must be a triangle or quad")



def map_surface_facets_to_tet_elements(surface: SurfaceMesh, tet_conn: np.ndarray) -> np.ndarray:
    """
    Map surface triangle facets to parent tet elements by node matching (tet4/tet10).
    """
    face_patterns_corner: list[tuple[int, ...]] = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    face_patterns_quad: list[tuple[int, ...]] = [
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
            face_nodes: tuple[int, ...] = tuple(sorted(int(elem[i]) for i in pattern))
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
    face_patterns_corner: list[tuple[int, ...]] = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    face_patterns_corner27: list[tuple[int, ...]] = [
        (0, 2, 8, 6),
        (18, 20, 26, 24),
        (0, 2, 20, 18),
        (6, 8, 26, 24),
        (0, 6, 24, 18),
        (2, 8, 26, 20),
    ]
    face_patterns_quad: list[tuple[int, ...]] = [
        (0, 1, 2, 3, 8, 9, 10, 11),
        (4, 5, 6, 7, 12, 13, 14, 15),
        (0, 1, 5, 4, 8, 17, 12, 16),
        (1, 2, 6, 5, 9, 18, 13, 17),
        (2, 3, 7, 6, 10, 19, 14, 18),
        (3, 0, 4, 7, 11, 16, 15, 19),
    ]
    face_patterns_quad9: list[tuple[int, ...]] = [
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
            face_nodes: tuple[int, ...] = tuple(sorted(int(elem[i]) for i in pattern))
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
    if _proj_diag_enabled():
        _proj_diag_attempt()
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
    res_norm = 0.0
    detJ = None
    iters = 0
    for _ in range(12):
        iters += 1
        n = 0.125 * (1.0 + xi * signs[:, 0]) * (1.0 + eta * signs[:, 1]) * (1.0 + zeta * signs[:, 2])
        x = n @ elem_coords
        r = x - point
        res_norm = float(np.linalg.norm(r))
        if res_norm < tol:
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
        detJ = float(np.linalg.det(J))
        try:
            delta = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            if _proj_diag_enabled():
                _proj_diag_log(
                    "SINGULAR_H",
                    iters=iters,
                    res_norm=res_norm,
                    delta_norm=None,
                    detJ=detJ,
                    point=point,
                    local=np.array([xi, eta, zeta], dtype=float),
                    in_ref_domain=False,
                )
            return np.zeros((8,), dtype=float), 0.0, 0.0, 0.0
        delta_norm = float(np.linalg.norm(delta))
        xi -= float(delta[0])
        eta -= float(delta[1])
        zeta -= float(delta[2])
        if not np.isfinite(xi) or not np.isfinite(eta) or not np.isfinite(zeta):
            if _proj_diag_enabled():
                _proj_diag_log(
                    "NAN_INF",
                    iters=iters,
                    res_norm=res_norm,
                    delta_norm=delta_norm,
                    detJ=detJ,
                    point=point,
                    local=np.array([xi, eta, zeta], dtype=float),
                    in_ref_domain=False,
                )
            return np.zeros((8,), dtype=float), 0.0, 0.0, 0.0
    if max(abs(xi), abs(eta), abs(zeta)) > 1.0 + tol:
        if _proj_diag_enabled():
            _proj_diag_log(
                "OUTSIDE_DOMAIN",
                iters=iters,
                res_norm=res_norm,
                delta_norm=None,
                detJ=detJ,
                point=point,
                local=np.array([xi, eta, zeta], dtype=float),
                in_ref_domain=False,
            )
        return np.zeros((8,), dtype=float), xi, eta, zeta
    if _proj_diag_enabled() and res_norm > tol:
        _proj_diag_log(
            "NEWTON_NO_CONVERGE",
            iters=iters,
            res_norm=res_norm,
            delta_norm=None,
            detJ=detJ,
            point=point,
            local=np.array([xi, eta, zeta], dtype=float),
            in_ref_domain=True,
        )
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
        corner_nodes = facet_nodes[[2, 0, 6, 8]]
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
        corner_pts = coords[corner_nodes]
        dX_dxi = dN_dxi_corner @ corner_pts
        dX_deta = dN_deta_corner @ corner_pts

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




def _diag_quad_source(default: str = "override") -> str:
    return _DEBUG_PROJ_QP_SOURCE or default

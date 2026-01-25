from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from .surface import SurfaceMesh
from ..core.forms import FormFieldLike
if TYPE_CHECKING:
    from ..core.forms import FieldPair
    from ..core.weakform import Params as WeakParams


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
    fields: dict[str, "FieldPair"]
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
_DEBUG_PROJECTION_DIAG = os.getenv("FLUXFEM_PROJ_DIAG")
_DEBUG_PROJECTION_DIAG_MAX = int(os.getenv("FLUXFEM_PROJ_DIAG_MAX", "20")) if _DEBUG_PROJECTION_DIAG else 0
_DEBUG_CONTACT_PROJ_ONCE = False
_DEBUG_PROJ_QP_CACHE = None
_DEBUG_PROJ_QP_SOURCE = None
_DEBUG_PROJ_QP_DUMPED = False
_PROJ_DIAG_STATS: dict[str, Any] | None = None
_PROJ_DIAG_COUNT = 0
_PROJ_DIAG_CONTEXT: dict[str, int | str] = {}


def _mortar_dbg_enabled() -> bool:
    return os.getenv("FLUXFEM_MORTAR_DEBUG", "0") not in ("0", "", "false", "False")


def _mortar_dbg(msg: str) -> None:
    if _mortar_dbg_enabled():
        print(msg, flush=True)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in ("0", "", "false", "False")


@dataclass(eq=False)
class MortarMatrix:
    """COO storage for mortar coupling matrices (can be rectangular)."""
    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]


def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Public wrapper for triangle area (used in contact diagnostics)."""
    return _tri_area(a, b, c)


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
        corner_nodes = facet_nodes[[0, 2, 8, 6]]
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
        corner = facet_nodes[[0, 2, 8, 6]]
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


def _projection_surface_batches(
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    *,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    quad_order: int,
    grad_source: str,
    dof_source: str,
    normal_source: str,
    normal_sign: float,
    tol: float,
):
    if dof_source != "volume" or grad_source != "volume":
        return None, False

    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)

    if facets_a.shape[1] != facets_b.shape[1] or facets_a.shape[1] not in {6, 9}:
        return None, False
    if elem_conn_a is None or elem_conn_b is None or facet_to_elem_a is None or facet_to_elem_b is None:
        return None, False

    diag = bool(_DEBUG_PROJECTION_DIAG)
    diag_max = _DEBUG_PROJECTION_DIAG_MAX if diag else 0
    total_points = 0
    fail_points = 0
    fail_by_code: dict[str, int] = {}
    fail_samples: list[dict] = []

    def _record_failure(code: str, info: dict | None, *, face_type: str, fa: int, fb: int, elem_id_a: int, elem_id_b: int, xm):
        nonlocal fail_points
        fail_points += 1
        fail_by_code[code] = fail_by_code.get(code, 0) + 1
        if not diag or len(fail_samples) >= diag_max:
            return
        sample = {
            "code": code,
            "face_type": face_type,
            "fa": int(fa),
            "fb": int(fb),
            "elem_a": int(elem_id_a),
            "elem_b": int(elem_id_b),
            "xm": None if xm is None else np.array(xm, dtype=float),
        }
        if info:
            sample.update(info)
        fail_samples.append(sample)

    pairs = {(int(fa), int(fb)) for fa, fb in zip(source_facets_a, source_facets_b)}
    if facets_a.shape[1] == 9:
        quad_pts, quad_w = _quad_quadrature(quad_order if quad_order > 0 else 2)
        face_type = "quad9"
    else:
        quad_pts, quad_w = _tri_quadrature(quad_order if quad_order > 0 else 1)
        face_type = "tri6"
    batches = []
    fallback = False

    for fa, fb in pairs:
        facet_a = facets_a[fa]
        facet_b = facets_b[fb]
        pts_a = coords_a[facet_a]
        pts_b = coords_b[facet_b]

        elem_id_a = int(facet_to_elem_a[fa])
        elem_id_b = int(facet_to_elem_b[fb])
        if elem_id_a < 0 or elem_id_b < 0:
            return None, True
        elem_nodes_a = np.asarray(elem_conn_a[elem_id_a], dtype=int)
        elem_nodes_b = np.asarray(elem_conn_b[elem_id_b], dtype=int)
        elem_coords_a = coords_a[elem_nodes_a]
        elem_coords_b = coords_b[elem_nodes_b]

        x_m_list = []
        x_s_list = []
        detJ_list = []
        normal_list = []
        for (xi, eta), w in zip(quad_pts, quad_w):
            if facets_a.shape[1] == 9:
                x_m, Jm = _quad9_map_and_jacobian(pts_a, xi, eta)
                xi_s, eta_s, ok, x_s, Js, info = _project_point_to_quad9(x_m, pts_b, tol=tol)
            else:
                x_m, Jm = _tri6_map_and_jacobian(pts_a, xi, eta)
                xi_s, eta_s, ok, x_s, Js, info = _project_point_to_tri6(x_m, pts_b, tol=tol)
            total_points += 1
            n_raw = np.cross(Jm[:, 0], Jm[:, 1])
            j_surf = float(np.linalg.norm(n_raw))
            if j_surf <= tol:
                fallback = True
                _record_failure(
                    "DEGENERATE_MASTER",
                    None,
                    face_type=face_type,
                    fa=fa,
                    fb=fb,
                    elem_id_a=elem_id_a,
                    elem_id_b=elem_id_b,
                    xm=x_m,
                )
                continue
            if not ok:
                fallback = True
                _record_failure(
                    info.get("status", "PROJECTION_FAIL"),
                    info,
                    face_type=face_type,
                    fa=fa,
                    fb=fb,
                    elem_id_a=elem_id_a,
                    elem_id_b=elem_id_b,
                    xm=x_m,
                )
                continue
            n_m = n_raw / j_surf
            n_use = n_m
            if normal_source in {"b", "slave"}:
                n_raw_b = np.cross(Js[:, 0], Js[:, 1])
                n_norm_b = float(np.linalg.norm(n_raw_b))
                if n_norm_b <= tol:
                    fallback = True
                    _record_failure(
                        "DEGENERATE_SLAVE",
                        None,
                        face_type=face_type,
                        fa=fa,
                        fb=fb,
                        elem_id_a=elem_id_a,
                        elem_id_b=elem_id_b,
                        xm=x_m,
                    )
                    continue
                n_use = n_raw_b / n_norm_b
            elif normal_source == "avg":
                n_raw_b = np.cross(Js[:, 0], Js[:, 1])
                n_norm_b = float(np.linalg.norm(n_raw_b))
                if n_norm_b <= tol:
                    fallback = True
                    _record_failure(
                        "DEGENERATE_SLAVE",
                        None,
                        face_type=face_type,
                        fa=fa,
                        fb=fb,
                        elem_id_a=elem_id_a,
                        elem_id_b=elem_id_b,
                        xm=x_m,
                    )
                    continue
                n_b = n_raw_b / n_norm_b
                avg = n_m + n_b
                avg_norm = float(np.linalg.norm(avg))
                n_use = avg / avg_norm if avg_norm > tol else n_m
            x_m_list.append(x_m)
            x_s_list.append(x_s)
            detJ_list.append(float(w * j_surf))
            normal_list.append(n_use)

        if not x_m_list:
            continue
        x_m = np.array(x_m_list, dtype=float)
        x_s = np.array(x_s_list, dtype=float)
        weights = np.array(detJ_list, dtype=float)
        normals = normal_sign * np.array(normal_list, dtype=float)

        Na = _volume_shape_values_at_points(x_m, elem_coords_a, tol=tol)
        Nb = _volume_shape_values_at_points(x_s, elem_coords_b, tol=tol)
        gradNa = _tet_gradN_at_points(x_m, elem_coords_a, tol=tol)
        gradNb = _tet_gradN_at_points(x_s, elem_coords_b, tol=tol)

        batches.append(
            dict(
                x_q=x_m,
                w=weights,
                detJ=np.ones_like(weights),
                Na=Na,
                Nb=Nb,
                gradNa=gradNa,
                gradNb=gradNb,
                nodes_a=elem_nodes_a,
                nodes_b=elem_nodes_b,
                normal=normals,
            )
        )

    if diag and fail_points:
        print(
            "[fluxfem][proj][diag]",
            f"total={total_points}",
            f"fail={fail_points}",
            f"fallback={fallback}",
            f"face_type={face_type}",
            f"fail_by_code={fail_by_code}",
        )
        for i, sample in enumerate(fail_samples):
            xm = sample.get("xm")
            xm_str = np.array2string(xm, precision=6) if xm is not None else "None"
            print(
                "[fluxfem][proj][diag]",
                f"sample={i}",
                f"code={sample.get('code')}",
                f"face={sample.get('face_type')}",
                f"fa={sample.get('fa')}",
                f"fb={sample.get('fb')}",
                f"elem_a={sample.get('elem_a')}",
                f"elem_b={sample.get('elem_b')}",
                f"xm={xm_str}",
                f"xi0={sample.get('xi0', float('nan')):.6f}",
                f"eta0={sample.get('eta0', float('nan')):.6f}",
                f"xi={sample.get('xi', float('nan')):.6f}",
                f"eta={sample.get('eta', float('nan')):.6f}",
                f"r={sample.get('r_norm', float('nan')):.3e}",
                f"d={sample.get('d_norm', float('nan')):.3e}",
                f"det={sample.get('det', float('nan')):.3e}",
                f"cond={sample.get('cond', float('nan')):.3e}",
            )

    return batches, fallback


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
    from ..core.forms import FieldPair
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    n_a = int(coords_a.shape[0] * value_dim_a)
    n_b = int(coords_b.shape[0] * value_dim_b)
    if offset_b is None:
        offset_b = offset_a + n_a
    n_total = int(offset_b + n_b)
    R: np.ndarray = np.zeros((n_total,), dtype=float)

    trace = os.getenv("FLUXFEM_MORTAR_TRACE", "0") not in ("0", "", "false", "False")

    def _trace_time(msg: str, t0: float) -> None:
        if trace:
            print(f"{msg} dt={time.perf_counter() - t0:.3e}s", flush=True)

    t_norm = time.perf_counter()
    normals_a = None
    normals_b = None
    if hasattr(surface_a, "facet_normals"):
        normals_a = surface_a.facet_normals()
    if hasattr(surface_b, "facet_normals"):
        normals_b = surface_b.facet_normals()
    if trace:
        _trace_time("[CONTACT] normals_done", t_norm)

    area_scale = float(os.getenv("FLUXFEM_SMALL_TRI_EPS_SCALE", "0.0"))
    skip_small_tri = os.getenv("FLUXFEM_SKIP_SMALL_TRI", "0") == "1" and area_scale > 0.0
    facet_area_a = None
    facet_area_b = None
    if area_scale > 0.0:
        t_area = time.perf_counter()
        facet_area_a = np.array([_facet_area_estimate(fa, coords_a) for fa in facets_a], dtype=float)
        facet_area_b = np.array([_facet_area_estimate(fb, coords_b) for fb in facets_b], dtype=float)
        if trace:
            _trace_time("[CONTACT] facet_area_done", t_area)

    includes_measure = getattr(res_form, "_includes_measure", {})

    use_elem_a = elem_conn_a is not None and facet_to_elem_a is not None
    use_elem_b = elem_conn_b is not None and facet_to_elem_b is not None
    if use_elem_a:
        assert elem_conn_a is not None
        assert facet_to_elem_a is not None
    if use_elem_b:
        assert elem_conn_b is not None
        assert facet_to_elem_b is not None

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
    proj_diag = _proj_diag_enabled()
    if proj_diag:
        _proj_diag_reset()
    diag_force = os.getenv("FLUXFEM_PROJ_DIAG_FORCE", "0") == "1"
    diag_qp_mode = os.getenv("FLUXFEM_PROJ_DIAG_QP_MODE", "").strip().lower()
    diag_qp_path = os.getenv("FLUXFEM_PROJ_DIAG_QP_PATH", "").strip()
    diag_normal = os.getenv("FLUXFEM_PROJ_DIAG_NORMAL", "").strip().lower()
    diag_facet = int(os.getenv("FLUXFEM_PROJ_DIAG_FACET", "-1"))
    diag_max_q = int(os.getenv("FLUXFEM_PROJ_DIAG_MAX_Q", "3"))
    diag_abs_detj = os.getenv("FLUXFEM_PROJ_DIAG_ABS_DETJ", "1") == "1"

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
    if diag_force and diag_normal:
        normal_source = diag_normal
    if normal_source not in {"a", "b", "avg", "master", "slave"}:
        raise ValueError("normal_source must be 'a', 'b', 'avg', 'master', or 'slave'")
    if normal_source == "master":
        normal_source = "a" if (master_field is None or master_field == field_a) else "b"
    if normal_source == "slave":
        normal_source = "b" if (master_field is None or master_field == field_a) else "a"

    mortar_mode = os.getenv("FLUXFEM_MORTAR_MODE", "supermesh").lower()
    if mortar_mode == "projection":
        batches, fallback = _projection_surface_batches(
            source_facets_a,
            source_facets_b,
            surface_a,
            surface_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            quad_order=quad_order,
            grad_source=grad_source,
            dof_source=dof_source,
            normal_source=normal_source,
            normal_sign=normal_sign,
            tol=tol,
        )
        if batches is not None and not fallback:
            for batch in batches:
                Na = batch["Na"]
                Nb = batch["Nb"]
                gradNa = batch["gradNa"]
                gradNb = batch["gradNb"]
                nodes_a = batch["nodes_a"]
                nodes_b = batch["nodes_b"]
                normal_q = batch["normal"]

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
                    field_a: FieldPair(test=cast("FormFieldLike", field_a_obj), trial=cast("FormFieldLike", field_a_obj)),
                    field_b: FieldPair(test=cast("FormFieldLike", field_b_obj), trial=cast("FormFieldLike", field_b_obj)),
                }
                ctx = SurfaceMixedFormContext(
                    fields=fields,
                    x_q=batch["x_q"],
                    w=batch["w"],
                    detJ=batch["detJ"],
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
                        raise ValueError("mixed surface residual must return (n_q, n_ldofs)")
                    if includes_measure.get(name, False):
                        fe = jnp.sum(jnp.asarray(fe_field), axis=0)
                    else:
                        wJ = jnp.asarray(ctx.w) * jnp.asarray(ctx.detJ)
                        fe = jnp.einsum("qi,q->i", jnp.asarray(fe_field), wJ)
                    dofs = _global_dof_indices(facet, value_dim, int(offset))
                    R[dofs] += np.asarray(fe)
            return R

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
        if diag_force and diag_abs_detj:
            detJ = abs(detJ)
        if quad_order <= 0:
            quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
            quad_w = np.array([0.5], dtype=float)
        else:
            quad_pts, quad_w = _tri_quadrature(quad_order)
        quad_source = "fluxfem"
        quad_override = _diag_quad_override(diag_force, diag_qp_mode, diag_qp_path)
        if quad_override is not None:
            quad_pts, quad_w = quad_override
            quad_source = _DEBUG_PROJ_QP_SOURCE or "override"
        _diag_quad_dump(diag_force, diag_qp_mode, diag_qp_path, quad_pts, quad_w)

        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)

        gradNa = None
        gradNb = None
        nodes_a = facet_a
        nodes_b = facet_b

        Na = None
        Nb = None

        elem_id_a = -1
        elem_nodes_a = None
        elem_coords_a = None
        if use_elem_a:
            assert elem_conn_a is not None
            assert facet_to_elem_a is not None
            elem_id_a = int(facet_to_elem_a[int(fa)])
            if elem_id_a < 0:
                raise ValueError("facet_to_elem_a has invalid mapping")
            elem_nodes_a = np.asarray(elem_conn_a[elem_id_a], dtype=int)
            elem_coords_a = coords_a[elem_nodes_a]
            if elem_coords_a.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")

        elem_id_b = -1
        elem_nodes_b = None
        elem_coords_b = None
        if use_elem_b:
            assert elem_conn_b is not None
            assert facet_to_elem_b is not None
            elem_id_b = int(facet_to_elem_b[int(fb)])
            if elem_id_b < 0:
                raise ValueError("facet_to_elem_b has invalid mapping")
            elem_nodes_b = np.asarray(elem_conn_b[elem_id_b], dtype=int)
            elem_coords_b = coords_b[elem_nodes_b]
            if elem_coords_b.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")
        if proj_diag:
            _proj_diag_set_context(
                fa=int(fa),
                fb=int(fb),
                face_a=_facet_label(facet_a),
                face_b=_facet_label(facet_b),
                elem_a=elem_id_a,
                elem_b=elem_id_b,
            )

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
            assert elem_nodes_a is not None
            assert elem_coords_a is not None
            local = _local_indices(elem_nodes_a, facet_a)
            gradNa = _tet_gradN_at_points(x_q, elem_coords_a, local=local, tol=tol)

        if use_elem_b and grad_source == "volume":
            assert elem_nodes_b is not None
            assert elem_coords_b is not None
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
        if diag_force:
            dofs_a = _global_dof_indices(nodes_a, value_dim_a, int(offset_a))
            dofs_b = _global_dof_indices(nodes_b, value_dim_b, int(offset_b))
            _diag_contact_projection(
                fa=int(fa),
                fb=int(fb),
                quad_pts=quad_pts,
                quad_w=quad_w,
                x_q=x_q,
                Na=Na,
                Nb=Nb,
                nodes_a=nodes_a,
                nodes_b=nodes_b,
                dofs_a=dofs_a,
                dofs_b=dofs_b,
                elem_coords_a=elem_coords_a if dof_source == "volume" else None,
                elem_coords_b=elem_coords_b if dof_source == "volume" else None,
                na=na,
                nb=nb,
                normal=normal,
                normal_source=normal_source,
                normal_sign=normal_sign,
                detJ=detJ,
                diag_facet=diag_facet,
                diag_max_q=diag_max_q,
                quad_source=quad_source,
                tol=tol,
            )

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
            field_a: FieldPair(test=cast("FormFieldLike", field_a_obj), trial=cast("FormFieldLike", field_a_obj)),
            field_b: FieldPair(test=cast("FormFieldLike", field_b_obj), trial=cast("FormFieldLike", field_b_obj)),
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
    if proj_diag:
        _proj_diag_report()
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
    backend: str = "jax",
    batch_jac: bool | None = None,
    fd_eps: float = 1e-6,
    fd_mode: str = "central",
    fd_block_size: int = 1,
):
    """
    Assemble mixed surface Jacobian over a supermesh (centroid quadrature).

    normal_source can be "master", "slave", "a", "b", or "avg"; use master_field
    to pick which field acts as the master when normal_source is "master"/"slave".
    dof_source="volume" assembles into element nodes (requires elem_conn_* mappings).
    """
    source_facets_a = list(source_facets_a)
    source_facets_b = list(source_facets_b)
    from ..core.forms import FieldPair
    _mortar_dbg(
        f"[mortar] enter assemble_mixed_surface_jacobian quad_order={quad_order} backend={backend}"
    )
    trace = os.getenv("FLUXFEM_MORTAR_TRACE", "0") not in ("0", "", "false", "False")
    trace_max = int(os.getenv("FLUXFEM_MORTAR_TRACE_MAX", "5"))
    trace_every = int(os.getenv("FLUXFEM_MORTAR_TRACE_EVERY", "50"))
    trace_fd_max = int(os.getenv("FLUXFEM_MORTAR_TRACE_FD_MAX", "5"))
    def _trace(msg: str) -> None:
        if trace:
            print(msg, flush=True)
    def _trace_time(msg: str, t0: float) -> None:
        if trace:
            print(f"{msg}: {time.perf_counter() - t0:.6f}s", flush=True)
    t_prep = time.perf_counter()
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    n_a = int(coords_a.shape[0] * value_dim_a)
    n_b = int(coords_b.shape[0] * value_dim_b)
    if offset_b is None:
        offset_b = offset_a + n_a
    n_total = int(offset_b + n_b)
    if trace:
        _trace("[CONTACT] assemble_mixed_surface_jacobian ENTER")
        _trace(f"[CONTACT] shapes: coords_a={coords_a.shape} coords_b={coords_b.shape} supermesh={supermesh_conn.shape}")
        _trace(f"[CONTACT] dtypes: coords_a={coords_a.dtype} coords_b={coords_b.dtype} supermesh={supermesh_conn.dtype}")
        _trace(f"[CONTACT] finite: coords_a={np.isfinite(coords_a).all()} coords_b={np.isfinite(coords_b).all()}")
        _trace_time("[CONTACT] prep_done", t_prep)

    guard = os.getenv("FLUXFEM_CONTACT_GUARD", "0") == "1"
    detj_eps = float(os.getenv("FLUXFEM_CONTACT_DETJ_EPS", "0.0"))
    tri_timeout = float(os.getenv("FLUXFEM_CONTACT_TRI_TIMEOUT_S", "0.0"))
    skip_nonfinite = os.getenv("FLUXFEM_CONTACT_SKIP_NONFINITE", "1") == "1"
    if guard:
        if not (np.isfinite(coords_a).all() and np.isfinite(coords_b).all()):
            raise RuntimeError("[CONTACT] non-finite coords in contact surfaces")
        if not np.isfinite(supermesh_coords).all():
            raise RuntimeError("[CONTACT] non-finite supermesh coords")
        if supermesh_conn.size:
            min_idx = int(supermesh_conn.min())
            max_idx = int(supermesh_conn.max())
            if min_idx < 0 or max_idx >= supermesh_coords.shape[0]:
                raise RuntimeError(
                    f"[CONTACT] supermesh_conn index out of range: min={min_idx} max={max_idx} n={supermesh_coords.shape[0]}"
                )
        if len(supermesh_conn) != len(source_facets_a) or len(supermesh_conn) != len(source_facets_b):
            raise RuntimeError(
                "[CONTACT] supermesh_conn and source_facets lengths mismatch "
                f"conn={len(supermesh_conn)} fa={len(source_facets_a)} fb={len(source_facets_b)}"
            )

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
    K_dense: np.ndarray | None = np.zeros((n_total, n_total), dtype=float) if not sparse else None

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
    proj_diag = _proj_diag_enabled()
    if proj_diag:
        _proj_diag_reset()
    diag_force = os.getenv("FLUXFEM_PROJ_DIAG_FORCE", "0") == "1"
    diag_qp_mode = os.getenv("FLUXFEM_PROJ_DIAG_QP_MODE", "").strip().lower()
    diag_qp_path = os.getenv("FLUXFEM_PROJ_DIAG_QP_PATH", "").strip()
    diag_normal = os.getenv("FLUXFEM_PROJ_DIAG_NORMAL", "").strip().lower()
    diag_facet = int(os.getenv("FLUXFEM_PROJ_DIAG_FACET", "-1"))
    diag_max_q = int(os.getenv("FLUXFEM_PROJ_DIAG_MAX_Q", "3"))
    diag_abs_detj = os.getenv("FLUXFEM_PROJ_DIAG_ABS_DETJ", "1") == "1"
    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")
    if backend == "numpy":
        if fd_eps <= 0.0:
            raise ValueError("fd_eps must be positive for numpy backend")
        if fd_mode not in {"central", "forward"}:
            raise ValueError("fd_mode must be 'central' or 'forward' for numpy backend")
    if batch_jac is None:
        batch_jac = _env_flag("FLUXFEM_MORTAR_BATCH_JAC", True)

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
    if diag_force and diag_normal:
        normal_source = diag_normal
    if normal_source not in {"a", "b", "avg", "master", "slave"}:
        raise ValueError("normal_source must be 'a', 'b', 'avg', 'master', or 'slave'")
    if normal_source == "master":
        normal_source = "a" if (master_field is None or master_field == field_a) else "b"
    if normal_source == "slave":
        normal_source = "b" if (master_field is None or master_field == field_a) else "a"

    mortar_mode = os.getenv("FLUXFEM_MORTAR_MODE", "supermesh").lower()
    _mortar_dbg(f"[mortar] mode={mortar_mode}")
    if mortar_mode == "projection":
        batches, fallback = _projection_surface_batches(
            source_facets_a,
            source_facets_b,
            surface_a,
            surface_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            quad_order=quad_order,
            grad_source=grad_source,
            dof_source=dof_source,
            normal_source=normal_source,
            normal_sign=normal_sign,
            tol=tol,
        )
        if batches is not None and not fallback:
            for batch in batches:
                Na = batch["Na"]
                Nb = batch["Nb"]
                gradNa = batch["gradNa"]
                gradNb = batch["gradNb"]
                nodes_a = batch["nodes_a"]
                nodes_b = batch["nodes_b"]
                normal_q = batch["normal"]

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
                    field_a: FieldPair(test=cast("FormFieldLike", field_a_obj), trial=cast("FormFieldLike", field_a_obj)),
                    field_b: FieldPair(test=cast("FormFieldLike", field_b_obj), trial=cast("FormFieldLike", field_b_obj)),
                }
                ctx = SurfaceMixedFormContext(
                    fields=fields,
                    x_q=batch["x_q"],
                    w=batch["w"],
                    detJ=batch["detJ"],
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

                def _res_local_np(u_vec):
                    u_dict = {name: u_vec[slices[name]] for name in (field_a, field_b)}
                    fe_q = res_form(ctx, u_dict, params)
                    res_parts = []
                    for name in (field_a, field_b):
                        fe_field = fe_q[name]
                        if includes_measure.get(name, False):
                            fe = np.sum(np.asarray(fe_field), axis=0)
                        else:
                            wJ = np.asarray(ctx.w) * np.asarray(ctx.detJ)
                            fe = np.einsum("qi,q->i", np.asarray(fe_field), wJ)
                        res_parts.append(np.asarray(fe))
                    return np.concatenate(res_parts, axis=0)

                if backend == "jax":
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
                else:
                    n_ldofs = int(u_local.shape[0])
                    J_local_np = np.zeros((n_ldofs, n_ldofs), dtype=float)
                    u_base = np.asarray(u_local, dtype=float)
                    r0 = _res_local_np(u_base) if fd_mode == "forward" else None
                    for i in range(n_ldofs):
                        u_p = u_base.copy()
                        u_p[i] += fd_eps
                        r_p = _res_local_np(u_p)
                        if fd_mode == "central":
                            u_m = u_base.copy()
                            u_m[i] -= fd_eps
                            r_m = _res_local_np(u_m)
                            col = (r_p - r_m) / (2.0 * fd_eps)
                        else:
                            col = (r_p - r0) / fd_eps
                        J_local_np[:, i] = np.asarray(col, dtype=float)

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
                            assert K_dense is not None
                            K_dense[int(gi), int(gj)] += val
            if sparse:
                return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), np.asarray(data, dtype=float), n_total
            assert K_dense is not None
            return K_dense

    if (
        batch_jac
        and backend == "jax"
        and dof_source == "volume"
        and grad_source == "volume"
        and use_elem_a
        and use_elem_b
        and not proj_diag
        and not diag_force
    ):
        if trace:
            _trace("[CONTACT] batch_jac_enter")
        batch_items = []
        dofs_batch = []
        u_local_batch = []
        batch_rows: list[np.ndarray] = []
        batch_cols: list[np.ndarray] = []
        batch_data: list[np.ndarray] = []
        batch_size = int(os.getenv("FLUXFEM_MORTAR_BATCH_SIZE", "128"))
        if batch_size <= 0:
            batch_size = 0
        n_q = None
        n_nodes_a = None
        n_nodes_b = None
        n_a_local_const = None
        n_b_local_const = None
        batch_failed = False
        jit_batch = _env_flag("FLUXFEM_MORTAR_BATCH_JIT", False)

        def _make_jac_fun(n_a_local: int, n_b_local: int):
            def _res_local_batch(u_vec, Na, Nb, gradNa, gradNb, x_q, w, detJ, normal):
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
                    field_a: FieldPair(test=cast("FormFieldLike", field_a_obj), trial=cast("FormFieldLike", field_a_obj)),
                    field_b: FieldPair(test=cast("FormFieldLike", field_b_obj), trial=cast("FormFieldLike", field_b_obj)),
                }
                normal_q = jnp.repeat(normal[None, :], x_q.shape[0], axis=0)
                ctx = SurfaceMixedFormContext(
                    fields=fields,
                    x_q=x_q,
                    w=w,
                    detJ=detJ,
                    normal=normal_q,
                    trial_fields={field_a: field_a_obj, field_b: field_b_obj},
                    test_fields={field_a: field_a_obj, field_b: field_b_obj},
                    unknown_fields={field_a: field_a_obj, field_b: field_b_obj},
                )
                u_dict = {
                    field_a: u_vec[:n_a_local],
                    field_b: u_vec[n_a_local:],
                }
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

            if trace:
                _trace(f"[CONTACT] batch_jac_build n_a={n_a_local} n_b={n_b_local} jit={jit_batch}")
            jac_fun = jax.vmap(jax.jacrev(_res_local_batch))
            return jax.jit(jac_fun) if jit_batch else jac_fun

        jac_fun_cache: dict[tuple[int, int], Callable[..., jnp.ndarray]] = {}

        def _emit_batch(
            Na_b,
            Nb_b,
            gradNa_b,
            gradNb_b,
            x_q_b,
            w_b,
            detJ_b,
            normal_b,
            u_local_b,
            dofs_batch_np,
            n_a_local,
            n_b_local,
            batch_n,
        ) -> None:
            if trace:
                _trace(f"[CONTACT] batch_emit start n={int(Na_b.shape[0])}")
            if batch_size and batch_n < batch_size:
                pad = int(batch_size - batch_n)
                if trace:
                    _trace(f"[CONTACT] batch_pad n={batch_n} target={batch_size}")

                def _pad_batch(x, pad_value: float = 0.0):
                    pad_width = [(0, pad)] + [(0, 0)] * (x.ndim - 1)
                    return jnp.pad(jnp.asarray(x), pad_width, mode="constant", constant_values=pad_value)

                Na_b = _pad_batch(Na_b)
                Nb_b = _pad_batch(Nb_b)
                gradNa_b = _pad_batch(gradNa_b)
                gradNb_b = _pad_batch(gradNb_b)
                x_q_b = _pad_batch(x_q_b)
                w_b = _pad_batch(w_b)
                detJ_b = _pad_batch(detJ_b)
                normal_b = _pad_batch(normal_b)
                u_local_b = _pad_batch(u_local_b)
            key = (n_a_local, n_b_local)
            jac_fun = jac_fun_cache.get(key)
            if jac_fun is None:
                jac_fun = _make_jac_fun(n_a_local, n_b_local)
                jac_fun_cache[key] = jac_fun
            t_batch = time.perf_counter()
            J_b = jac_fun(u_local_b, Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b)
            J_b_np = np.asarray(J_b)[:batch_n]
            if trace:
                _trace_time("[CONTACT] batch_emit jac_done", t_batch)
            n_ldofs = dofs_batch_np.shape[1]
            rows = np.repeat(dofs_batch_np, n_ldofs, axis=1).reshape(-1)
            cols = np.tile(dofs_batch_np, (1, n_ldofs)).reshape(-1)
            data = J_b_np.reshape(-1)
            if sparse:
                batch_rows.append(rows)
                batch_cols.append(cols)
                batch_data.append(data)
            else:
                assert K_dense is not None
                K_dense[rows, cols] += data
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
            if diag_force and diag_abs_detj:
                detJ = abs(detJ)
            if quad_order <= 0:
                quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
                quad_w = np.array([0.5], dtype=float)
            else:
                quad_pts, quad_w = _tri_quadrature(quad_order)

            facet_a = facets_a[int(fa)]
            facet_b = facets_b[int(fb)]
            x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)

            assert facet_to_elem_a is not None
            assert elem_conn_a is not None
            elem_id_a = int(facet_to_elem_a[int(fa)])
            elem_nodes_a = np.asarray(elem_conn_a[elem_id_a], dtype=int)
            elem_coords_a = coords_a[elem_nodes_a]
            assert facet_to_elem_b is not None
            assert elem_conn_b is not None
            elem_id_b = int(facet_to_elem_b[int(fb)])
            elem_nodes_b = np.asarray(elem_conn_b[elem_id_b], dtype=int)
            elem_coords_b = coords_b[elem_nodes_b]

            Na = _volume_shape_values_at_points(x_q, elem_coords_a, tol=tol)
            Nb = _volume_shape_values_at_points(x_q, elem_coords_b, tol=tol)
            gradNa = _tet_gradN_at_points(x_q, elem_coords_a, tol=tol)
            gradNb = _tet_gradN_at_points(x_q, elem_coords_b, tol=tol)

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
            if normal is None:
                batch_failed = True
                break

            u_elem = {
                field_a: _gather_u_local(u_a, elem_nodes_a, value_dim_a),
                field_b: _gather_u_local(u_b, elem_nodes_b, value_dim_b),
            }
            u_local = np.concatenate([u_elem[field_a], u_elem[field_b]], axis=0)

            dofs_a = _global_dof_indices(elem_nodes_a, value_dim_a, int(offset_a))
            dofs_b = _global_dof_indices(elem_nodes_b, value_dim_b, int(offset_b))
            dofs = np.concatenate([dofs_a, dofs_b], axis=0)

            batch_items.append((Na, Nb, gradNa, gradNb, x_q, quad_w, detJ, normal))
            dofs_batch.append(dofs)
            u_local_batch.append(u_local)

            if n_q is None:
                n_q = Na.shape[0]
                n_nodes_a = Na.shape[1]
                n_nodes_b = Nb.shape[1]
                n_a_local_const = dofs_a.shape[0]
                n_b_local_const = dofs_b.shape[0]
            else:
                shape_mismatch = (
                    Na.shape[0] != n_q
                    or Nb.shape[0] != n_q
                    or Na.shape[1] != n_nodes_a
                    or Nb.shape[1] != n_nodes_b
                    or dofs_a.shape[0] != n_a_local_const
                    or dofs_b.shape[0] != n_b_local_const
                )
                if shape_mismatch:
                    if batch_items:
                        Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
                        Na_b = jnp.asarray(np.stack(Na_b, axis=0))
                        Nb_b = jnp.asarray(np.stack(Nb_b, axis=0))
                        gradNa_b = jnp.asarray(np.stack(gradNa_b, axis=0))
                        gradNb_b = jnp.asarray(np.stack(gradNb_b, axis=0))
                        x_q_b = jnp.asarray(np.stack(x_q_b, axis=0))
                        w_b = jnp.asarray(np.stack(w_b, axis=0))
                        detJ_b = jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1)
                        normal_b = jnp.asarray(np.stack(normal_b, axis=0))
                        u_local_b = jnp.asarray(np.stack(u_local_batch, axis=0))
                        dofs_batch_np = np.asarray(dofs_batch, dtype=int)
                        assert n_a_local_const is not None
                        assert n_b_local_const is not None
                        _emit_batch(
                            Na_b,
                            Nb_b,
                            gradNa_b,
                            gradNb_b,
                            x_q_b,
                            w_b,
                            detJ_b,
                            normal_b,
                            u_local_b,
                            dofs_batch_np,
                            int(n_a_local_const),
                            int(n_b_local_const),
                            int(Na_b.shape[0]),
                        )
                    batch_items = [(Na, Nb, gradNa, gradNb, x_q, quad_w, detJ, normal)]
                    dofs_batch = [dofs]
                    u_local_batch = [u_local]
                    n_q = Na.shape[0]
                    n_nodes_a = Na.shape[1]
                    n_nodes_b = Nb.shape[1]
                    n_a_local_const = dofs_a.shape[0]
                    n_b_local_const = dofs_b.shape[0]

            if batch_size and len(batch_items) >= batch_size:
                Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
                Na_b = jnp.asarray(np.stack(Na_b, axis=0))
                Nb_b = jnp.asarray(np.stack(Nb_b, axis=0))
                gradNa_b = jnp.asarray(np.stack(gradNa_b, axis=0))
                gradNb_b = jnp.asarray(np.stack(gradNb_b, axis=0))
                x_q_b = jnp.asarray(np.stack(x_q_b, axis=0))
                w_b = jnp.asarray(np.stack(w_b, axis=0))
                detJ_b = jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1)
                normal_b = jnp.asarray(np.stack(normal_b, axis=0))
                u_local_b = jnp.asarray(np.stack(u_local_batch, axis=0))
                dofs_batch_np = np.asarray(dofs_batch, dtype=int)
                assert n_a_local_const is not None
                assert n_b_local_const is not None
                _emit_batch(
                    Na_b,
                    Nb_b,
                    gradNa_b,
                    gradNb_b,
                    x_q_b,
                    w_b,
                    detJ_b,
                    normal_b,
                    u_local_b,
                    dofs_batch_np,
                    int(n_a_local_const),
                    int(n_b_local_const),
                    int(Na_b.shape[0]),
                )
                batch_items = []
                dofs_batch = []
                u_local_batch = []

        if not batch_failed and batch_items:
            Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
            Na_b = jnp.asarray(np.stack(Na_b, axis=0))
            Nb_b = jnp.asarray(np.stack(Nb_b, axis=0))
            gradNa_b = jnp.asarray(np.stack(gradNa_b, axis=0))
            gradNb_b = jnp.asarray(np.stack(gradNb_b, axis=0))
            x_q_b = jnp.asarray(np.stack(x_q_b, axis=0))
            w_b = jnp.asarray(np.stack(w_b, axis=0))
            detJ_b = jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1)
            normal_b = jnp.asarray(np.stack(normal_b, axis=0))
            u_local_b = jnp.asarray(np.stack(u_local_batch, axis=0))
            dofs_batch_np = np.asarray(dofs_batch, dtype=int)
            assert n_a_local_const is not None
            assert n_b_local_const is not None
            _emit_batch(
                Na_b,
                Nb_b,
                gradNa_b,
                gradNb_b,
                x_q_b,
                w_b,
                detJ_b,
                normal_b,
                u_local_b,
                dofs_batch_np,
                int(n_a_local_const),
                int(n_b_local_const),
                int(Na_b.shape[0]),
            )

        if not batch_failed and (batch_rows or (not sparse and K_dense is not None)):
            if sparse:
                if batch_rows:
                    rows_np = np.concatenate(batch_rows)
                    cols_np = np.concatenate(batch_cols)
                    data_np = np.concatenate(batch_data)
                else:
                    rows_np = np.zeros((0,), dtype=int)
                    cols_np = np.zeros((0,), dtype=int)
                    data_np = np.zeros((0,), dtype=float)
                return rows_np, cols_np, data_np, n_total
            assert K_dense is not None
            return K_dense

        if trace:
            _trace("[CONTACT] batch_jac_fallback")

    if trace:
        _trace("[CONTACT] supermesh_loop_enter")
    _mortar_dbg("[mortar] step: supermesh loop START")
    t_loop = time.perf_counter()
    for it, ((tri, a, b, c), fa, fb) in enumerate(
        zip(
            _iter_supermesh_tris(supermesh_coords, supermesh_conn),
            source_facets_a,
            source_facets_b,
        )
    ):
        log_tri = trace and (it < trace_max or it % trace_every == 0)
        t_tri0 = time.perf_counter()
        def _tri_check(stage: str) -> None:
            if tri_timeout > 0.0 and (time.perf_counter() - t_tri0) > tri_timeout:
                raise RuntimeError(f"[CONTACT] tri {it} timeout at {stage}")
        if log_tri:
            _trace(f"[CONTACT] tri {it} start fa={int(fa)} fb={int(fb)}")
        t_geom = time.perf_counter()
        area = _tri_area(a, b, c)
        if area <= tol:
            continue
        if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
            area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
            if area_ref > 0.0 and area < area_scale * area_ref:
                continue
        detJ = 2.0 * area
        if diag_force and diag_abs_detj:
            detJ = abs(detJ)
        if guard:
            if not np.isfinite(detJ):
                if log_tri:
                    _trace(f"[CONTACT] tri {it} detJ non-finite; skip")
                if skip_nonfinite:
                    continue
                raise RuntimeError(f"[CONTACT] tri {it} detJ non-finite")
            if detj_eps > 0.0 and abs(detJ) < detj_eps:
                if log_tri:
                    _trace(f"[CONTACT] tri {it} detJ too small {detJ:.3e}; skip")
                continue
        if quad_order <= 0:
            quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
            quad_w = np.array([0.5], dtype=float)
        else:
            quad_pts, quad_w = _tri_quadrature(quad_order)
        quad_source = "fluxfem"
        quad_override = _diag_quad_override(diag_force, diag_qp_mode, diag_qp_path)
        if quad_override is not None:
            quad_pts, quad_w = quad_override
            quad_source = _DEBUG_PROJ_QP_SOURCE or "override"
        _diag_quad_dump(diag_force, diag_qp_mode, diag_qp_path, quad_pts, quad_w)

        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)
        if guard and not np.isfinite(x_q).all():
            if log_tri:
                _trace(f"[CONTACT] tri {it} x_q non-finite; skip")
            if skip_nonfinite:
                continue
            raise RuntimeError(f"[CONTACT] tri {it} x_q non-finite")
        if log_tri:
            _trace_time(f"[CONTACT] tri {it} geom_done", t_geom)
        _tri_check("geom_done")

        gradNa = None
        gradNb = None
        nodes_a = facet_a
        nodes_b = facet_b

        Na = None
        Nb = None

        elem_id_a = -1
        elem_nodes_a = None
        elem_coords_a = None
        local_a = None
        if use_elem_a:
            assert elem_conn_a is not None
            assert facet_to_elem_a is not None
            elem_id_a = int(facet_to_elem_a[int(fa)])
            if elem_id_a < 0:
                raise ValueError("facet_to_elem_a has invalid mapping")
            elem_nodes_a = np.asarray(elem_conn_a[elem_id_a], dtype=int)
            elem_coords_a = coords_a[elem_nodes_a]
            if elem_coords_a.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")

        elem_id_b = -1
        elem_nodes_b = None
        elem_coords_b = None
        local_b = None
        if use_elem_b:
            assert elem_conn_b is not None
            assert facet_to_elem_b is not None
            elem_id_b = int(facet_to_elem_b[int(fb)])
            if elem_id_b < 0:
                raise ValueError("facet_to_elem_b has invalid mapping")
            elem_nodes_b = np.asarray(elem_conn_b[elem_id_b], dtype=int)
            elem_coords_b = coords_b[elem_nodes_b]
            if elem_coords_b.shape[0] not in {4, 8, 10, 20, 27}:
                raise NotImplementedError("surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only")
        if proj_diag:
            _proj_diag_set_context(
                fa=int(fa),
                fb=int(fb),
                face_a=_facet_label(facet_a),
                face_b=_facet_label(facet_b),
                elem_a=elem_id_a,
                elem_b=elem_id_b,
            )

        t_basis = time.perf_counter()
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
            assert elem_nodes_a is not None
            assert elem_coords_a is not None
            local_a = _local_indices(elem_nodes_a, facet_a)
            gradNa = _tet_gradN_at_points(x_q, elem_coords_a, local=local_a, tol=tol)

        if use_elem_b and grad_source == "volume":
            assert elem_nodes_b is not None
            assert elem_coords_b is not None
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
        if guard and (not np.isfinite(Na).all() or not np.isfinite(Nb).all()):
            if log_tri:
                _trace(f"[CONTACT] tri {it} N non-finite; skip")
            if skip_nonfinite:
                continue
            raise RuntimeError(f"[CONTACT] tri {it} N non-finite")
        if log_tri:
            _trace_time(f"[CONTACT] tri {it} basis_done", t_basis)
        _tri_check("basis_done")

        global _DEBUG_CONTACT_MAP_ONCE
        if diag_map and not _DEBUG_CONTACT_MAP_ONCE:
            if use_elem_a:
                assert facet_to_elem_a is not None
                elem_id_a = int(facet_to_elem_a[int(fa)])
            else:
                elem_id_a = -1
            if use_elem_b:
                assert facet_to_elem_b is not None
                elem_id_b = int(facet_to_elem_b[int(fb)])
            else:
                elem_id_b = -1
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
            field_a: FieldPair(test=cast("FormFieldLike", field_a_obj), trial=cast("FormFieldLike", field_a_obj)),
            field_b: FieldPair(test=cast("FormFieldLike", field_b_obj), trial=cast("FormFieldLike", field_b_obj)),
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

        def _res_local_np(u_vec):
            u_dict = {name: u_vec[slices[name]] for name in (field_a, field_b)}
            fe_q = res_form(ctx, u_dict, params)
            res_parts = []
            for name in (field_a, field_b):
                fe_field = fe_q[name]
                if includes_measure.get(name, False):
                    fe = np.sum(np.asarray(fe_field), axis=0)
                else:
                    wJ = np.asarray(ctx.w) * np.asarray(ctx.detJ)
                    fe = np.einsum("qi...,q->i...", np.asarray(fe_field), wJ)
                res_parts.append(np.asarray(fe))
            return np.concatenate(res_parts, axis=0)

        t_jac = time.perf_counter()
        if backend == "jax":
            J_local = jax.jacrev(_res_local)(jnp.asarray(u_local))
            J_local_np = np.asarray(J_local)
        else:
            n_ldofs = int(u_local.shape[0])
            J_local_np = np.zeros((n_ldofs, n_ldofs), dtype=float)
            u_base = np.asarray(u_local, dtype=float)
            if log_tri:
                _trace(f"[CONTACT] tri {it} fd_start n_ldofs={n_ldofs} fd_mode={fd_mode}")
            r0 = _res_local_np(u_base) if fd_mode == "forward" else None
            if log_tri and fd_mode == "forward":
                _trace(f"[CONTACT] tri {it} fd_r0_done")
            block = max(1, int(fd_block_size))
            if block <= 1:
                for i in range(n_ldofs):
                    log_fd = log_tri and i < trace_fd_max
                    if log_fd:
                        _trace(f"[CONTACT] tri {it} fd_col {i} start")
                    u_p = u_base.copy()
                    u_p[i] += fd_eps
                    t_rp = time.perf_counter()
                    r_p = _res_local_np(u_p)
                    if log_fd:
                        _trace_time(f"[CONTACT] tri {it} fd_col {i} r_p", t_rp)
                    if fd_mode == "central":
                        u_m = u_base.copy()
                        u_m[i] -= fd_eps
                        t_rm = time.perf_counter()
                        r_m = _res_local_np(u_m)
                        if log_fd:
                            _trace_time(f"[CONTACT] tri {it} fd_col {i} r_m", t_rm)
                        col = (r_p - r_m) / (2.0 * fd_eps)
                    else:
                        col = (r_p - r0) / fd_eps
                    J_local_np[:, i] = np.asarray(col, dtype=float)
                    if log_fd:
                        _trace(f"[CONTACT] tri {it} fd_col {i} done")
            else:
                for i0 in range(0, n_ldofs, block):
                    idxs = np.arange(i0, min(i0 + block, n_ldofs))
                    u_block = np.repeat(u_base[:, None], idxs.size, axis=1)
                    for bi, idx in enumerate(idxs):
                        u_block[idx, bi] += fd_eps
                    t_rp = time.perf_counter()
                    r_p = _res_local_np(u_block)
                    if log_tri and i0 < trace_fd_max:
                        _trace_time(f"[CONTACT] tri {it} fd_block r_p", t_rp)
                    if fd_mode == "central":
                        u_block_m = np.repeat(u_base[:, None], idxs.size, axis=1)
                        for bi, idx in enumerate(idxs):
                            u_block_m[idx, bi] -= fd_eps
                        t_rm = time.perf_counter()
                        r_m = _res_local_np(u_block_m)
                        if log_tri and i0 < trace_fd_max:
                            _trace_time(f"[CONTACT] tri {it} fd_block r_m", t_rm)
                        cols = (r_p - r_m) / (2.0 * fd_eps)
                    else:
                        assert r0 is not None
                        cols = (r_p - r0[:, None]) / fd_eps
                    J_local_np[:, idxs] = np.asarray(cols, dtype=float)
        if log_tri:
            _trace_time(f"[CONTACT] tri {it} jac_done", t_jac)
        _tri_check("jac_done")

        dofs_a = _global_dof_indices(nodes_a, value_dim_a, int(offset_a))
        dofs_b = _global_dof_indices(nodes_b, value_dim_b, int(offset_b))
        if diag_force:
            _diag_contact_projection(
                fa=int(fa),
                fb=int(fb),
                quad_pts=quad_pts,
                quad_w=quad_w,
                x_q=x_q,
                Na=Na,
                Nb=Nb,
                nodes_a=nodes_a,
                nodes_b=nodes_b,
                dofs_a=dofs_a,
                dofs_b=dofs_b,
                elem_coords_a=elem_coords_a if dof_source == "volume" else None,
                elem_coords_b=elem_coords_b if dof_source == "volume" else None,
                na=na,
                nb=nb,
                normal=normal,
                normal_source=normal_source,
                normal_sign=normal_sign,
                detJ=detJ,
                diag_facet=diag_facet,
                diag_max_q=diag_max_q,
                quad_source=quad_source,
                tol=tol,
            )
        t_scatter = time.perf_counter()
        dofs = np.concatenate([dofs_a, dofs_b], axis=0)
        if sparse:
            n_ldofs = int(dofs.shape[0])
            rows.extend(np.repeat(dofs, n_ldofs).tolist())
            cols.extend(np.tile(dofs, n_ldofs).tolist())
            data.extend(J_local_np.reshape(-1).tolist())
        else:
            assert K_dense is not None
            K_dense[np.ix_(dofs, dofs)] += J_local_np
        if log_tri:
            _trace_time(f"[CONTACT] tri {it} scatter_done", t_scatter)
        _tri_check("scatter_done")


    if proj_diag:
        _proj_diag_report()
    if sparse:
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), np.asarray(data, dtype=float), n_total
    assert K_dense is not None
    return K_dense


def assemble_onesided_bilinear(
    surface_slave: SurfaceMesh,
    u_hat_fn,
    params: "WeakParams",
    *,
    surface_master: SurfaceMesh | None = None,
    u_master: np.ndarray | None = None,
    value_dim: int = 3,
    elem_conn: np.ndarray | None = None,
    facet_to_elem: np.ndarray | None = None,
    elem_conn_master: np.ndarray | None = None,
    facet_to_elem_master: np.ndarray | None = None,
    grad_source: str = "volume",
    dof_source: str = "volume",
    quad_order: int = 2,
    normal_sign: float = 1.0,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble one-sided (slave-only) Nitsche matrices without supermesh.

    The master side is treated as prescribed displacement u_hat(x). Provide
    either u_hat_fn(x_q) or u_master with master element mappings to evaluate
    u_hat at slave quadrature points.

    Note: this implementation currently assumes volume-trace bases for both
    gradients and DOFs. Surface-only bases are not supported here yet.
    """
    from ..core.forms import FieldPair
    coords_s = np.asarray(surface_slave.coords, dtype=float)
    facets_s = np.asarray(surface_slave.conn, dtype=int)
    coords_m = np.asarray(surface_master.coords, dtype=float) if surface_master is not None else coords_s
    facets_m = np.asarray(surface_master.conn, dtype=int) if surface_master is not None else facets_s
    n_s = int(coords_s.shape[0] * value_dim)
    K: np.ndarray = np.zeros((n_s, n_s), dtype=float)
    f: np.ndarray = np.zeros((n_s,), dtype=float)

    normals_s = surface_slave.facet_normals() if hasattr(surface_slave, "facet_normals") else None
    use_elem = elem_conn is not None and facet_to_elem is not None
    use_master = u_master is not None

    if use_master:
        if surface_master is None:
            raise ValueError("surface_master is required when u_master is provided")
        if elem_conn_master is None or facet_to_elem_master is None:
            raise ValueError("elem_conn_master and facet_to_elem_master are required when u_master is provided")
    else:
        if u_hat_fn is None:
            raise ValueError("u_hat_fn or u_master must be provided")
        if surface_master is None:
            surface_master = surface_slave

    if grad_source != "volume" or dof_source != "volume":
        raise ValueError("one-sided Nitsche currently supports only volume/volume")

    from ..core.weakform import (
        Params,
        compile_mixed_surface_residual_numpy,
        param_ref,
        test_ref,
        unknown_ref,
    )
    import fluxfem.helpers_wf as h_wf

    u = unknown_ref("u")
    v = test_ref("u")
    p = param_ref()
    n = h_wf.normal()
    t_u = h_wf.traction(u, n, p)
    t_v = h_wf.traction(v, n, p)
    sym_term = h_wf.einsum("qia,qi->qa", t_v, u.val)
    sym_term_hat = h_wf.einsum("qia,qi->qa", t_v, p.u_hat)
    expr = (
        -h_wf.dot(v, t_u)
        - sym_term
        + (p.alpha * p.inv_h) * h_wf.dot(v, u.val)
        + sym_term_hat
        - (p.alpha * p.inv_h) * h_wf.dot(v, p.u_hat)
    ) * h_wf.ds()
    res_form = compile_mixed_surface_residual_numpy({"u": expr})
    includes_measure = res_form._includes_measure

    quad_pts, quad_w = _tri_quadrature(quad_order) if quad_order > 0 else (np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5]))

    for f_id, facet in enumerate(facets_s):
        triangles = _facet_triangles(coords_s, facet)
        if not triangles:
            continue
        area_f = _facet_area_estimate(facet, coords_s)
        if area_f <= tol:
            continue
        inv_h = 1.0 / max(np.sqrt(area_f), tol)

        elem_nodes = None
        elem_coords = None
        local = None
        if use_elem:
            assert facet_to_elem is not None
            assert elem_conn is not None
            elem_id = int(facet_to_elem[int(f_id)])
            if elem_id < 0:
                raise ValueError("facet_to_elem has invalid mapping")
            elem_nodes = np.asarray(elem_conn[elem_id], dtype=int)
            elem_coords = coords_s[elem_nodes]

        for a, b, c in triangles:
            area = _tri_area(a, b, c)
            if area <= tol:
                continue
            detJ = 2.0 * area
            x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)
            if use_master:
                if dof_source == "surface":
                    assert u_master is not None
                    facet_m = facets_m[int(f_id)]
                    u_master_local = _gather_u_local(u_master, facet_m, value_dim).reshape(-1, value_dim)
                    N_master = np.array(
                        [_facet_shape_values(pt, facet_m, coords_m, tol=tol) for pt in x_q],
                        dtype=float,
                    )
                    u_hat = N_master @ u_master_local
                else:
                    assert u_master is not None
                    assert facet_to_elem_master is not None
                    assert elem_conn_master is not None
                    elem_id_m = int(facet_to_elem_master[int(f_id)])
                    if elem_id_m < 0:
                        raise ValueError("facet_to_elem_master has invalid mapping")
                    elem_nodes_m = np.asarray(elem_conn_master[elem_id_m], dtype=int)
                    elem_coords_m = coords_m[elem_nodes_m]
                    u_master_local = _gather_u_local(u_master, elem_nodes_m, value_dim).reshape(-1, value_dim)
                    N_master = _volume_shape_values_at_points(x_q, elem_coords_m, tol=tol)
                    u_hat = N_master @ u_master_local
            else:
                u_hat = np.asarray(u_hat_fn(x_q), dtype=float)
                if u_hat.shape[0] != x_q.shape[0]:
                    raise ValueError("u_hat_fn must return shape (n_q, value_dim)")

            gradN = None
            nodes = facet
            N = None

            if grad_source == "surface":
                gradN = np.array(
                    [_surface_gradN(pt, facet, coords_s, tol=tol) for pt in x_q],
                    dtype=float,
                )
            if use_elem and grad_source == "volume":
                assert elem_nodes is not None
                assert elem_coords is not None
                local = _local_indices(elem_nodes, facet)
                gradN = _tet_gradN_at_points(x_q, elem_coords, local=local, tol=tol)

            if dof_source == "volume":
                if not use_elem or elem_nodes is None or elem_coords is None:
                    raise ValueError("dof_source 'volume' requires elem_conn and facet_to_elem")
                nodes = elem_nodes
                N = _volume_shape_values_at_points(x_q, elem_coords, tol=tol)
                if grad_source == "volume":
                    gradN = _tet_gradN_at_points(x_q, elem_coords, tol=tol)
            else:
                N = np.array([_facet_shape_values(pt, facet, coords_s, tol=tol) for pt in x_q], dtype=float)

            field = SurfaceMixedFormField(
                N=N,
                gradN=gradN,
                value_dim=value_dim,
                basis=_SurfaceBasis(dofs_per_node=value_dim),
            )
            fields = {"u": FieldPair(test=cast("FormFieldLike", field), trial=cast("FormFieldLike", field))}
            normal = normals_s[int(f_id)] if normals_s is not None else None
            if normal is not None:
                normal = normal_sign * normal
            normal_q = None if normal is None else np.repeat(normal[None, :], quad_pts.shape[0], axis=0)
            ctx = SurfaceMixedFormContext(
                fields=fields,
                x_q=x_q,
                w=quad_w,
                detJ=np.array([detJ], dtype=float),
                normal=normal_q,
                trial_fields={"u": field},
                test_fields={"u": field},
                unknown_fields={"u": field},
            )
            params_local = Params(
                lam=params.lam,
                mu=params.mu,
                alpha=params.alpha,
                inv_h=inv_h,
                u_hat=u_hat,
            )
            u_zero: np.ndarray = np.zeros((len(nodes) * value_dim,), dtype=float)
            u_dict = {"u": u_zero}
            sizes = (u_zero.shape[0],)
            slices = {"u": slice(0, sizes[0])}

            def _res_local_np_single(u_vec: np.ndarray) -> np.ndarray:
                u_local = {"u": u_vec[slices["u"]]}
                fe_q = res_form(ctx, u_local, params_local)["u"]
                if includes_measure.get("u", False):
                    return np.sum(np.asarray(fe_q), axis=0)
                wJ = np.asarray(ctx.w) * np.asarray(ctx.detJ)
                return np.einsum("qi,q->i", np.asarray(fe_q), wJ)

            def _res_local_np(u_vec: np.ndarray) -> np.ndarray:
                if u_vec.ndim == 1:
                    return _res_local_np_single(u_vec)
                out = np.empty((u_vec.shape[0], u_vec.shape[1]), dtype=float)
                for col in range(u_vec.shape[1]):
                    out[:, col] = _res_local_np_single(u_vec[:, col])
                return out

            f_local = _res_local_np(u_zero)
            n_ldofs = int(u_zero.shape[0])
            k_local: np.ndarray = np.zeros((n_ldofs, n_ldofs), dtype=float)
            block = max(1, int(os.getenv("FLUXFEM_ONESIDE_BLOCK_SIZE", "16")))
            for start in range(0, n_ldofs, block):
                idxs: np.ndarray = np.arange(start, min(n_ldofs, start + block), dtype=int)
                u_block: np.ndarray = np.zeros((n_ldofs, idxs.size), dtype=float)
                u_block[idxs, np.arange(idxs.size, dtype=int)] = 1.0
                r_block = _res_local_np(u_block)
                k_local[:, idxs] = r_block - f_local[:, None]

            dofs = _global_dof_indices(nodes, value_dim, 0)
            f[dofs] += f_local
            K[np.ix_(dofs, dofs)] += k_local

    return K, f


def assemble_contact_onesided_floor(
    surface_slave: SurfaceMesh,
    u: np.ndarray,
    *,
    n: np.ndarray | None = None,
    c: float,
    k: float,
    beta: float,
    value_dim: int = 3,
    elem_conn: np.ndarray | None = None,
    facet_to_elem: np.ndarray | None = None,
    quad_order: int = 2,
    normal_sign: float = 1.0,
    tol: float = 1e-8,
    return_metrics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Assemble one-sided contact penalty against a rigid plane g = n·x - c.

    Uses softplus for a smooth contact pressure:
        p(g) = k * softplus(-g; beta)
    with softplus(z; beta) = (1 / beta) * log(1 + exp(beta z)).

    Note: the resulting stiffness matrix can be nonsymmetric; avoid CG.
    """
    if elem_conn is None or facet_to_elem is None:
        raise ValueError("elem_conn and facet_to_elem are required")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    import jax
    import jax.numpy as jnp

    coords_s = np.asarray(surface_slave.coords, dtype=float)
    facets_s = np.asarray(surface_slave.conn, dtype=int)
    n_s = int(coords_s.shape[0] * value_dim)
    K: np.ndarray = np.zeros((n_s, n_s), dtype=float)
    f: np.ndarray = np.zeros((n_s,), dtype=float)

    normals_s = surface_slave.facet_normals() if hasattr(surface_slave, "facet_normals") else None
    if n is not None:
        n = np.asarray(n, dtype=float).reshape(-1)
        if n.shape[0] != 3:
            raise ValueError("n must be a 3-vector")
        n_norm = np.linalg.norm(n)
        if n_norm <= tol:
            raise ValueError("n must be non-zero")
        n = (n / n_norm) * float(normal_sign)
    elif normals_s is None:
        raise ValueError("surface normals are required when n is not provided")

    penetration = 0.0
    min_g = float("inf")
    quad_pts, quad_w = _tri_quadrature(quad_order) if quad_order > 0 else (np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5]))

    for f_id, facet in enumerate(facets_s):
        triangles = _facet_triangles(coords_s, facet)
        if not triangles:
            continue
        area_f = _facet_area_estimate(facet, coords_s)
        if area_f <= tol:
            continue

        elem_id = int(facet_to_elem[int(f_id)])
        if elem_id < 0:
            raise ValueError("facet_to_elem has invalid mapping")
        elem_nodes = np.asarray(elem_conn[elem_id], dtype=int)
        elem_coords = coords_s[elem_nodes]
        u_local = _gather_u_local(u, elem_nodes, value_dim).reshape(-1, value_dim)

        if n is not None:
            normal = n
        else:
            assert normals_s is not None
            normal = normal_sign * normals_s[int(f_id)]

        for a, b, c_tri in triangles:
            area = _tri_area(a, b, c_tri)
            if area <= tol:
                continue
            detJ = 2.0 * area
            x_q_ref = np.array([a + r * (b - a) + s * (c_tri - a) for r, s in quad_pts], dtype=float)
            N = _volume_shape_values_at_points(x_q_ref, elem_coords, tol=tol)

            normal_q = np.repeat(normal[None, :], quad_pts.shape[0], axis=0)

            u_q_np = N @ u_local
            x_q_cur = x_q_ref + u_q_np
            g_np = np.sum(normal_q * x_q_cur, axis=1) - float(c)
            min_g = min(min_g, float(np.min(g_np)))
            z_np = -float(beta) * g_np
            z_clip = np.minimum(z_np, 30.0)
            softplus_np = np.where(z_np > 30.0, z_np, np.log1p(np.exp(z_clip))) / float(beta)
            penetration += float(np.sum(softplus_np * quad_w) * detJ)

            def _res_local(u_vec):
                u_loc = u_vec.reshape(-1, value_dim)
                u_q = jnp.einsum("qi,ia->qa", jnp.asarray(N), u_loc)
                x_q_j = jnp.asarray(x_q_ref)
                n_q = jnp.asarray(normal_q)
                x_q_cur_j = x_q_j + u_q
                g = jnp.einsum("qa,qa->q", n_q, x_q_cur_j) - float(c)
                p = float(k) * jax.nn.softplus(-float(beta) * g) / float(beta)
                t = p[:, None] * n_q
                wJ = jnp.asarray(quad_w) * float(detJ)
                nodal = jnp.einsum("qi,qa,q->ia", jnp.asarray(N), t, wJ)
                return nodal.reshape(-1)

            u_vec0 = np.asarray(u_local.reshape(-1), dtype=float)
            f_local = np.asarray(_res_local(jnp.asarray(u_vec0)))
            k_local = np.asarray(jax.jacrev(_res_local)(jnp.asarray(u_vec0)))

            dofs = _global_dof_indices(elem_nodes, value_dim, 0)
            for i, gi in enumerate(dofs):
                f[int(gi)] += float(f_local[i])
                for j, gj in enumerate(dofs):
                    K[int(gi), int(gj)] += float(k_local[i, j])

    if return_metrics:
        if min_g == float("inf"):
            min_g = 0.0
        metrics = {
            "penetration": float(penetration),
            "min_g": float(min_g),
        }
        return K, f, metrics
    return K, f

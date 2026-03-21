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
    bindings: dict[str, "FieldPair"]
    x_q: np.ndarray
    w: np.ndarray
    detJ: np.ndarray
    normal: np.ndarray | None = None
    spaces: dict[str, "FieldPair"] | None = None


def _make_surface_field_pair(
    *,
    test_N: np.ndarray,
    test_gradN: np.ndarray | None,
    trial_N: np.ndarray,
    trial_gradN: np.ndarray | None,
    test_value_dim: int,
    trial_value_dim: int,
) -> "FieldPair":
    from ..core.forms import FieldPair

    test_field = SurfaceMixedFormField(
        N=test_N,
        gradN=test_gradN,
        value_dim=test_value_dim,
        basis=_SurfaceBasis(dofs_per_node=test_value_dim),
    )
    trial_field = SurfaceMixedFormField(
        N=trial_N,
        gradN=trial_gradN,
        value_dim=trial_value_dim,
        basis=_SurfaceBasis(dofs_per_node=trial_value_dim),
    )
    return FieldPair(
        test=cast("FormFieldLike", test_field),
        trial=cast("FormFieldLike", trial_field),
        unknown=cast("FormFieldLike", trial_field),
    )


@dataclass(eq=False)
class _SupermeshPairBasisData:
    elem_id_a: int
    elem_id_b: int
    Na: np.ndarray
    Nb: np.ndarray
    gradNa: np.ndarray
    gradNb: np.ndarray
    dofs_local_a: np.ndarray
    dofs_local_b: np.ndarray
    nodes_a: np.ndarray
    nodes_b: np.ndarray
    local_a: np.ndarray | None
    local_b: np.ndarray | None
    test_Na: np.ndarray | None = None
    test_Nb: np.ndarray | None = None
    trial_Na: np.ndarray | None = None
    trial_Nb: np.ndarray | None = None
    test_gradNa: np.ndarray | None = None
    test_gradNb: np.ndarray | None = None
    trial_gradNa: np.ndarray | None = None
    trial_gradNb: np.ndarray | None = None
    test_dofs_local_a: np.ndarray | None = None
    test_dofs_local_b: np.ndarray | None = None
    trial_dofs_local_a: np.ndarray | None = None
    trial_dofs_local_b: np.ndarray | None = None
    test_nodes_a: np.ndarray | None = None
    test_nodes_b: np.ndarray | None = None
    trial_nodes_a: np.ndarray | None = None
    trial_nodes_b: np.ndarray | None = None


def _merge_trial_pair_basis_data(
    base: _SupermeshPairBasisData,
    trial: _SupermeshPairBasisData,
) -> _SupermeshPairBasisData:
    base.trial_Na = trial.Na
    base.trial_Nb = trial.Nb
    base.trial_gradNa = trial.gradNa
    base.trial_gradNb = trial.gradNb
    base.trial_dofs_local_a = trial.dofs_local_a
    base.trial_dofs_local_b = trial.dofs_local_b
    base.trial_nodes_a = trial.nodes_a
    base.trial_nodes_b = trial.nodes_b
    return base


def _same_optional_int_array(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int))


@dataclass(eq=False)
class _JacobianTriangleGeometryData:
    detJ: float
    quad_pts: np.ndarray
    quad_w: np.ndarray
    quad_source: str
    facet_a: np.ndarray
    facet_b: np.ndarray
    x_q: np.ndarray


@dataclass(eq=False)
class _SupermeshTriangleQuadratureCache:
    detJ: np.ndarray
    x_q: np.ndarray
    quad_pts: np.ndarray
    quad_w: np.ndarray


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
_DIRECT_PAIR_NITSCHE_BATCH_FUN_CACHE: dict[bool, Callable[..., jnp.ndarray]] = {}


def _contact_interface_dbg_enabled() -> bool:
    return os.getenv("FLUXFEM_CONTACT_INTERFACE_DEBUG", "0") not in ("0", "", "false", "False")


def _contact_interface_dbg(msg: str) -> None:
    if _contact_interface_dbg_enabled():
        print(msg, flush=True)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in ("0", "", "false", "False")


@dataclass(eq=False)
class ContactCouplingMatrix:
    """COO storage for contact coupling matrices (can be rectangular)."""
    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]


def _is_jax_value(x: Any) -> bool:
    return isinstance(x, jax.core.Tracer)


def _uses_jax_geometry(*xs: Any) -> bool:
    for x in xs:
        if _is_jax_value(x):
            return True
    return False


def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _numpy_shape_matrix(N: np.ndarray, value_dim: int) -> np.ndarray:
    n_nodes = int(N.shape[0])
    out = np.zeros((int(value_dim), n_nodes * int(value_dim)), dtype=float)
    for a in range(n_nodes):
        col = int(value_dim) * a
        for i in range(int(value_dim)):
            out[i, col + i] = float(N[a])
    return out


def _numpy_sym_grad_matrix(gradN: np.ndarray, dofs_per_node: int = 3) -> np.ndarray:
    n_nodes = int(gradN.shape[0])
    n_dofs = int(dofs_per_node) * n_nodes
    B = np.zeros((6, n_dofs), dtype=float)
    for a in range(n_nodes):
        dNdx, dNdy, dNdz = float(gradN[a, 0]), float(gradN[a, 1]), float(gradN[a, 2])
        col = int(dofs_per_node) * a
        B[0, col + 0] = dNdx
        B[1, col + 1] = dNdy
        B[2, col + 2] = dNdz
        B[3, col + 0] = dNdy
        B[3, col + 1] = dNdx
        B[4, col + 1] = dNdz
        B[4, col + 2] = dNdy
        B[5, col + 0] = dNdz
        B[5, col + 2] = dNdx
    return B


def _numpy_isotropic_D(lam: float, mu: float) -> np.ndarray:
    return np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=float,
    )


def _numpy_voigt_traction_matrix(normal: np.ndarray) -> np.ndarray:
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    return np.array(
        [
            [nx, 0.0, 0.0, ny, 0.0, nz],
            [0.0, ny, 0.0, nx, nz, 0.0],
            [0.0, 0.0, nz, 0.0, ny, nx],
        ],
        dtype=float,
    )


def _jax_shape_matrix(N: jnp.ndarray, value_dim: int) -> jnp.ndarray:
    eye = jnp.eye(int(value_dim), dtype=N.dtype)
    return jnp.einsum("a,ij->iaj", N, eye).reshape(int(value_dim), int(N.shape[0]) * int(value_dim))


def _jax_sym_grad_matrix(gradN: jnp.ndarray, dofs_per_node: int = 3) -> jnp.ndarray:
    if int(dofs_per_node) != 3:
        raise NotImplementedError("JAX fast pair Nitsche kernel currently supports only dofs_per_node=3.")
    gx = gradN[:, 0]
    gy = gradN[:, 1]
    gz = gradN[:, 2]
    zeros = jnp.zeros_like(gx)
    rows = [
        jnp.stack([gx, zeros, zeros], axis=1),
        jnp.stack([zeros, gy, zeros], axis=1),
        jnp.stack([zeros, zeros, gz], axis=1),
        jnp.stack([gy, gx, zeros], axis=1),
        jnp.stack([zeros, gz, gy], axis=1),
        jnp.stack([gz, zeros, gx], axis=1),
    ]
    return jnp.stack(rows, axis=0).reshape(6, int(gradN.shape[0]) * int(dofs_per_node))


def _jax_isotropic_D(lam: Any, mu: Any, *, dtype: Any) -> jnp.ndarray:
    lam_j = jnp.asarray(lam, dtype=dtype)
    mu_j = jnp.asarray(mu, dtype=dtype)
    return jnp.array(
        [
            [lam_j + 2.0 * mu_j, lam_j, lam_j, 0.0, 0.0, 0.0],
            [lam_j, lam_j + 2.0 * mu_j, lam_j, 0.0, 0.0, 0.0],
            [lam_j, lam_j, lam_j + 2.0 * mu_j, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu_j, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu_j, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu_j],
        ],
        dtype=dtype,
    )


def _jax_voigt_traction_matrix(normal: jnp.ndarray) -> jnp.ndarray:
    nx, ny, nz = normal[0], normal[1], normal[2]
    zeros = jnp.asarray(0.0, dtype=normal.dtype)
    return jnp.array(
        [
            [nx, zeros, zeros, ny, zeros, nz],
            [zeros, ny, zeros, nx, nz, zeros],
            [zeros, zeros, nz, zeros, ny, nx],
        ],
        dtype=normal.dtype,
    )


def _fast_pair_nitsche_penalty_local_matrix(
    *,
    Na: np.ndarray,
    Nb: np.ndarray,
    gradNa: np.ndarray,
    gradNb: np.ndarray,
    normal_q: np.ndarray,
    w: np.ndarray,
    detJ: np.ndarray,
    alpha: float,
    inv_h: float,
    lam: float,
    mu: float,
    use_penalty: float,
    use_traction: float,
    value_dim_a: int,
    value_dim_b: int,
) -> np.ndarray:
    if int(value_dim_a) != 3 or int(value_dim_b) != 3:
        raise NotImplementedError("Fast pair Nitsche kernel currently supports only value_dim=3.")

    D = _numpy_isotropic_D(float(lam), float(mu))
    n_dofs_a = int(Na.shape[1] * value_dim_a)
    n_dofs_b = int(Nb.shape[1] * value_dim_b)
    Kaa = np.zeros((n_dofs_a, n_dofs_a), dtype=float)
    Kab = np.zeros((n_dofs_a, n_dofs_b), dtype=float)
    Kba = np.zeros((n_dofs_b, n_dofs_a), dtype=float)
    Kbb = np.zeros((n_dofs_b, n_dofs_b), dtype=float)

    wJ = np.asarray(w, dtype=float) * np.asarray(detJ, dtype=float)
    penalty_scale = float(use_penalty) * float(alpha * inv_h)
    traction_scale = float(use_traction)
    for q in range(int(Na.shape[0])):
        Nma = _numpy_shape_matrix(Na[q], value_dim_a)
        Nmb = _numpy_shape_matrix(Nb[q], value_dim_b)
        Ba = _numpy_sym_grad_matrix(gradNa[q], dofs_per_node=value_dim_a)
        Bb = _numpy_sym_grad_matrix(gradNb[q], dofs_per_node=value_dim_b)
        Pn = _numpy_voigt_traction_matrix(normal_q[q])
        Ta = Pn @ D @ Ba
        Tb = Pn @ D @ Bb
        s = float(wJ[q])

        # penalty
        Kaa += s * penalty_scale * (Nma.T @ Nma)
        Kab += -s * penalty_scale * (Nma.T @ Nmb)
        Kba += -s * penalty_scale * (Nmb.T @ Nma)
        Kbb += s * penalty_scale * (Nmb.T @ Nmb)

        # consistency and symmetry terms
        Kaa += traction_scale * s * (-0.5 * (Nma.T @ Ta) - 0.5 * (Ta.T @ Nma))
        Kab += traction_scale * s * (-0.5 * (Nma.T @ Tb) + 0.5 * (Ta.T @ Nmb))
        Kba += traction_scale * s * (0.5 * (Nmb.T @ Ta) - 0.5 * (Tb.T @ Nma))
        Kbb += traction_scale * s * (0.5 * (Nmb.T @ Tb) + 0.5 * (Tb.T @ Nmb))

    top = np.concatenate([Kaa, Kab], axis=1)
    bot = np.concatenate([Kba, Kbb], axis=1)
    return np.concatenate([top, bot], axis=0)


def _fast_pair_nitsche_penalty_local_matrix_jax(
    *,
    Na: jnp.ndarray,
    Nb: jnp.ndarray,
    gradNa: jnp.ndarray,
    gradNb: jnp.ndarray,
    normal_q: jnp.ndarray,
    w: jnp.ndarray,
    detJ: jnp.ndarray,
    alpha: float,
    inv_h: float,
    lam: float,
    mu: float,
    use_penalty: float,
    use_traction: float,
    value_dim_a: int,
    value_dim_b: int,
) -> jnp.ndarray:
    if int(value_dim_a) != 3 or int(value_dim_b) != 3:
        raise NotImplementedError("Fast pair Nitsche kernel currently supports only value_dim=3.")

    dtype = Na.dtype
    D = _jax_isotropic_D(lam, mu, dtype=dtype)
    n_dofs_a = int(Na.shape[1] * value_dim_a)
    n_dofs_b = int(Nb.shape[1] * value_dim_b)
    wJ = jnp.asarray(w, dtype=dtype) * jnp.asarray(detJ, dtype=dtype).reshape(-1)
    alpha_inv_h = jnp.asarray(use_penalty, dtype=dtype) * jnp.asarray(alpha, dtype=dtype) * jnp.asarray(inv_h, dtype=dtype)
    traction_scale = jnp.asarray(use_traction, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)

    def _q_local_matrix(Na_q, Nb_q, gradNa_q, gradNb_q, normal_qi, wJ_q):
        Nma = _jax_shape_matrix(Na_q, value_dim_a)
        Nmb = _jax_shape_matrix(Nb_q, value_dim_b)
        Ba = _jax_sym_grad_matrix(gradNa_q, dofs_per_node=value_dim_a)
        Bb = _jax_sym_grad_matrix(gradNb_q, dofs_per_node=value_dim_b)
        Pn = _jax_voigt_traction_matrix(normal_qi)
        Ta = Pn @ D @ Ba
        Tb = Pn @ D @ Bb
        Kaa = alpha_inv_h * (Nma.T @ Nma)
        Kab = -alpha_inv_h * (Nma.T @ Nmb)
        Kba = -alpha_inv_h * (Nmb.T @ Nma)
        Kbb = alpha_inv_h * (Nmb.T @ Nmb)

        Kaa = Kaa + traction_scale * (-half * (Nma.T @ Ta) - half * (Ta.T @ Nma))
        Kab = Kab + traction_scale * (-half * (Nma.T @ Tb) + half * (Ta.T @ Nmb))
        Kba = Kba + traction_scale * (half * (Nmb.T @ Ta) - half * (Tb.T @ Nma))
        Kbb = Kbb + traction_scale * (half * (Nmb.T @ Tb) + half * (Tb.T @ Nmb))

        top = jnp.concatenate([Kaa, Kab], axis=1)
        bot = jnp.concatenate([Kba, Kbb], axis=1)
        return wJ_q * jnp.concatenate([top, bot], axis=0)

    return jnp.sum(
        jax.vmap(_q_local_matrix)(Na, Nb, gradNa, gradNb, normal_q, wJ),
        axis=0,
    )


def _get_direct_pair_nitsche_batch_fun(*, jit: bool) -> Callable[..., jnp.ndarray]:
    cached = _DIRECT_PAIR_NITSCHE_BATCH_FUN_CACHE.get(bool(jit))
    if cached is not None:
        return cached

    def _local_matrix_batch(Na, Nb, gradNa, gradNb, w, detJ, normal, alpha, inv_h, lam, mu, use_penalty, use_traction):
        normal_q = jnp.repeat(normal[None, :], Na.shape[0], axis=0)
        return _fast_pair_nitsche_penalty_local_matrix_jax(
            Na=Na,
            Nb=Nb,
            gradNa=gradNa,
            gradNb=gradNb,
            normal_q=normal_q,
            w=w,
            detJ=detJ,
            alpha=alpha,
            inv_h=inv_h,
            lam=lam,
            mu=mu,
            use_penalty=use_penalty,
            use_traction=use_traction,
            value_dim_a=3,
            value_dim_b=3,
        )

    fun = jax.vmap(
        _local_matrix_batch,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None),
    )
    if jit:
        fun = jax.jit(fun)
    _DIRECT_PAIR_NITSCHE_BATCH_FUN_CACHE[bool(jit)] = fun
    return fun


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


def _facet_local_dofs(
    facet_id: int,
    *,
    n_facets: int,
    value_dim: int,
    facet_dofs: np.ndarray | None,
) -> np.ndarray:
    """
    Return local (field-relative) DOF indices for a facet-wise space.

    - When ``facet_dofs`` is provided, it must be shape ``(n_facets, n_ldofs_facet)``.
    - Otherwise, a default contiguous layout is used:
      ``facet_id * value_dim + [0..value_dim-1]``.
    """
    if facet_dofs is not None:
        arr = np.asarray(facet_dofs, dtype=int)
        if arr.ndim != 2:
            raise ValueError("facet_dofs must be a 2D array (n_facets, n_ldofs_facet).")
        if arr.shape[0] != int(n_facets):
            raise ValueError(
                f"facet_dofs first dimension mismatch: got {arr.shape[0]}, expected {int(n_facets)}."
            )
        return arr[int(facet_id)]
    start = int(facet_id) * int(value_dim)
    return start + np.arange(int(value_dim), dtype=int)


def _field_n_dofs(
    *,
    n_nodes: int,
    n_facets: int,
    value_dim: int,
    space_mode: str,
    facet_dofs: np.ndarray | None,
) -> int:
    if space_mode == "p0":
        if facet_dofs is not None:
            arr = np.asarray(facet_dofs, dtype=int)
            if arr.size == 0:
                return 0
            if np.any(arr < 0):
                raise ValueError("facet_dofs must be non-negative.")
            return int(arr.max()) + 1
        return int(n_facets) * int(value_dim)
    return int(n_nodes) * int(value_dim)


def _build_side_field_data_p0(
    *,
    n_q: int,
    facet_id: int,
    n_facets: int,
    value_dim: int,
    facet_nodes: np.ndarray,
    facet_dofs: np.ndarray | None,
    local: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    dofs_local = _facet_local_dofs(
        int(facet_id),
        n_facets=n_facets,
        value_dim=value_dim,
        facet_dofs=facet_dofs,
    )
    n_ldofs = int(dofs_local.shape[0])
    N = np.ones((n_q, n_ldofs), dtype=float)
    gradN = np.zeros((n_q, n_ldofs, 3), dtype=float)
    return N, gradN, dofs_local, facet_nodes, local


def _build_side_field_data_nodal(
    *,
    x_q: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    value_dim: int,
    dof_source: str,
    use_elem: bool,
    elem_nodes: np.ndarray | None,
    elem_coords: np.ndarray | None,
    tol: float,
    volume_dof_error: str,
    grad_source: str,
    gradN: np.ndarray | None,
    local: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    if dof_source == "volume":
        if not use_elem or elem_nodes is None or elem_coords is None:
            raise ValueError(volume_dof_error)
        N = _volume_shape_values_at_points(x_q, elem_coords, tol=tol)
        if grad_source == "volume":
            gradN = _tet_gradN_at_points(x_q, elem_coords, tol=tol)
        dofs_local = _global_dof_indices(elem_nodes, value_dim, 0)
        return N, gradN, dofs_local, elem_nodes, local

    N = np.array([_facet_shape_values(pt, facet_nodes, coords, tol=tol) for pt in x_q], dtype=float)
    dofs_local = _global_dof_indices(facet_nodes, value_dim, 0)
    return N, gradN, dofs_local, facet_nodes, local


def _build_side_grad_data(
    *,
    x_q: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    grad_source: str,
    use_elem: bool,
    elem_nodes: np.ndarray | None,
    elem_coords: np.ndarray | None,
    tol: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    gradN = None
    local = None
    if grad_source == "surface":
        gradN = np.array(
            [_surface_gradN(pt, facet_nodes, coords, tol=tol) for pt in x_q],
            dtype=float,
        )
    if use_elem and grad_source == "volume":
        assert elem_nodes is not None
        assert elem_coords is not None
        local = _local_indices(elem_nodes, facet_nodes)
        gradN = _tet_gradN_at_points(x_q, elem_coords, local=local, tol=tol)
    return gradN, local


def _prepare_supermesh_side_field_data(
    *,
    x_q: np.ndarray,
    facet_id: int,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    value_dim: int,
    n_facets: int,
    dof_source: str,
    grad_source: str,
    space_mode: str,
    use_elem: bool,
    elem_nodes: np.ndarray | None,
    elem_coords: np.ndarray | None,
    facet_dofs: np.ndarray | None,
    tol: float,
    volume_dof_error: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    gradN, local = _build_side_grad_data(
        x_q=x_q,
        facet_nodes=facet_nodes,
        coords=coords,
        grad_source=grad_source,
        use_elem=use_elem,
        elem_nodes=elem_nodes,
        elem_coords=elem_coords,
        tol=tol,
    )
    if space_mode == "p0":
        return _build_side_field_data_p0(
            n_q=int(x_q.shape[0]),
            facet_id=int(facet_id),
            n_facets=n_facets,
            value_dim=value_dim,
            facet_nodes=facet_nodes,
            facet_dofs=facet_dofs,
            local=local,
        )
    return _build_side_field_data_nodal(
        x_q=x_q,
        facet_nodes=facet_nodes,
        coords=coords,
        value_dim=value_dim,
        dof_source=dof_source,
        use_elem=use_elem,
        elem_nodes=elem_nodes,
        elem_coords=elem_coords,
        tol=tol,
        volume_dof_error=volume_dof_error,
        grad_source=grad_source,
        gradN=gradN,
        local=local,
    )


def _resolve_facet_element_context(
    *,
    use_elem: bool,
    facet_id: int,
    facet_to_elem: np.ndarray | None,
    elem_conn: np.ndarray | None,
    coords: np.ndarray,
    invalid_map_error: str,
    unsupported_error: str,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    if not use_elem:
        return -1, None, None
    assert facet_to_elem is not None
    assert elem_conn is not None
    elem_id = int(facet_to_elem[int(facet_id)])
    if elem_id < 0:
        raise ValueError(invalid_map_error)
    elem_nodes = np.asarray(elem_conn[elem_id], dtype=int)
    elem_coords = coords[elem_nodes]
    if elem_coords.shape[0] not in {4, 8, 10, 20, 27}:
        raise NotImplementedError(unsupported_error)
    return elem_id, elem_nodes, elem_coords


def _validate_contact_interface_space_setup(
    *,
    grad_source: str,
    dof_source: str,
    space_mode_a: str,
    space_mode_b: str,
) -> tuple[bool, bool]:
    if grad_source not in {"volume", "surface"}:
        raise ValueError("grad_source must be 'volume' or 'surface'")
    if dof_source not in {"surface", "volume"}:
        raise ValueError("dof_source must be 'surface' or 'volume'")
    if space_mode_a not in {"nodal", "p0"}:
        raise ValueError("space_mode_a must be 'nodal' or 'p0'")
    if space_mode_b not in {"nodal", "p0"}:
        raise ValueError("space_mode_b must be 'nodal' or 'p0'")
    if dof_source == "volume" and grad_source == "surface":
        raise ValueError("dof_source 'volume' requires grad_source 'volume'")
    return space_mode_a == "p0", space_mode_b == "p0"


def _resolve_normal_source_option(
    *,
    normal_source: str,
    normal_from: str | None,
    master_field: str | None,
    field_a: str,
    field_b: str,
    diag_force: bool,
    diag_normal: str,
) -> str:
    resolved = normal_source
    if normal_from is not None:
        if normal_from not in {"master", "slave"}:
            raise ValueError("normal_from must be 'master' or 'slave'")
        master_name = field_a if master_field is None else master_field
        if master_name not in {field_a, field_b}:
            raise ValueError("master_field must match field_a or field_b")
        if normal_from == "master":
            resolved = "a" if master_name == field_a else "b"
        else:
            resolved = "b" if master_name == field_a else "a"
    if diag_force and diag_normal:
        resolved = diag_normal
    if resolved not in {"a", "b", "avg", "master", "slave"}:
        raise ValueError("normal_source must be 'a', 'b', 'avg', 'master', or 'slave'")
    if resolved == "master":
        resolved = "a" if (master_field is None or master_field == field_a) else "b"
    if resolved == "slave":
        resolved = "b" if (master_field is None or master_field == field_a) else "a"
    return resolved


def _resolve_contact_normal(
    *,
    facet_id_a: int,
    facet_id_b: int,
    normals_a: np.ndarray | None,
    normals_b: np.ndarray | None,
    normal_source: str,
    normal_sign: float,
    tol: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    na = normals_a[int(facet_id_a)] if normals_a is not None else None
    nb = normals_b[int(facet_id_b)] if normals_b is not None else None
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
    return normal, na, nb


def _build_mixed_surface_context(
    *,
    field_a: str,
    field_b: str,
    test_space_key_a: str | None,
    test_space_key_b: str | None,
    unknown_space_key_a: str | None,
    unknown_space_key_b: str | None,
    test_Na: np.ndarray,
    test_Nb: np.ndarray,
    trial_Na: np.ndarray,
    trial_Nb: np.ndarray,
    test_gradNa: np.ndarray | None,
    test_gradNb: np.ndarray | None,
    trial_gradNa: np.ndarray | None,
    trial_gradNb: np.ndarray | None,
    test_value_dim_a: int,
    test_value_dim_b: int,
    trial_value_dim_a: int,
    trial_value_dim_b: int,
    x_q: np.ndarray,
    w: np.ndarray,
    detJ: np.ndarray,
    normal_q: np.ndarray | None,
) -> SurfaceMixedFormContext:
    fields = {
        field_a: _make_surface_field_pair(
            test_N=test_Na,
            test_gradN=test_gradNa,
            trial_N=trial_Na,
            trial_gradN=trial_gradNa,
            test_value_dim=test_value_dim_a,
            trial_value_dim=trial_value_dim_a,
        ),
        field_b: _make_surface_field_pair(
            test_N=test_Nb,
            test_gradN=test_gradNb,
            trial_N=trial_Nb,
            trial_gradN=trial_gradNb,
            test_value_dim=test_value_dim_b,
            trial_value_dim=trial_value_dim_b,
        ),
    }
    spaces = dict(fields)
    for key in (test_space_key_a, unknown_space_key_a):
        if key is not None:
            spaces[key] = fields[field_a]
    for key in (test_space_key_b, unknown_space_key_b):
        if key is not None:
            spaces[key] = fields[field_b]
    return SurfaceMixedFormContext(
        bindings=fields,
        x_q=x_q,
        w=w,
        detJ=detJ,
        normal=normal_q,
        spaces=spaces,
    )


def _surface_u_elem_with_space_aliases(
    *,
    field_a: str,
    field_b: str,
    unknown_space_key_a: str | None,
    unknown_space_key_b: str | None,
    u_local_a: np.ndarray,
    u_local_b: np.ndarray,
) -> dict[str, np.ndarray]:
    u_elem = {
        field_a: u_local_a,
        field_b: u_local_b,
    }
    if unknown_space_key_a is not None:
        u_elem[unknown_space_key_a] = u_elem[field_a]
    if unknown_space_key_b is not None:
        u_elem[unknown_space_key_b] = u_elem[field_b]
    return u_elem


def _surface_local_u_dict(
    *,
    u_vec: np.ndarray | jnp.ndarray,
    slices: dict[str, slice],
    field_a: str,
    field_b: str,
    ctx: SurfaceMixedFormContext,
) -> dict[str, np.ndarray | jnp.ndarray]:
    u_dict = {name: u_vec[slices[name]] for name in (field_a, field_b)}
    if ctx.spaces is not None:
        for key, pair in ctx.spaces.items():
            if key in u_dict:
                continue
            if pair is ctx.bindings[field_a]:
                u_dict[key] = u_dict[field_a]
            elif pair is ctx.bindings[field_b]:
                u_dict[key] = u_dict[field_b]
    return u_dict


def _mixed_surface_space_aliases(
    res_form: Callable[..., Any],
    *,
    field_a: str,
    field_b: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    test_space_keys = getattr(res_form, "_test_space_by_target", {})
    unknown_space_keys = getattr(res_form, "_unknown_space_by_target", {})
    legacy_space_keys = getattr(res_form, "_space_by_target", {})
    test_space_key_a = test_space_keys.get(field_a, legacy_space_keys.get(field_a))
    test_space_key_b = test_space_keys.get(field_b, legacy_space_keys.get(field_b))
    unknown_space_key_a = unknown_space_keys.get(field_a, legacy_space_keys.get(field_a))
    unknown_space_key_b = unknown_space_keys.get(field_b, legacy_space_keys.get(field_b))
    return test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b


def _reduce_surface_residual_jax(
    fe_field: Any,
    *,
    includes_measure: bool,
    w: np.ndarray,
    detJ: np.ndarray,
) -> jnp.ndarray:
    if includes_measure:
        return jnp.sum(jnp.asarray(fe_field), axis=0)
    wJ = jnp.asarray(w) * jnp.asarray(detJ)
    return jnp.einsum("qi,q->i", jnp.asarray(fe_field), wJ)


def _reduce_surface_residual_numpy(
    fe_field: Any,
    *,
    includes_measure: bool,
    w: np.ndarray,
    detJ: np.ndarray,
) -> np.ndarray:
    if includes_measure:
        return np.sum(np.asarray(fe_field), axis=0)
    wJ = np.asarray(w) * np.asarray(detJ)
    return np.einsum("qi...,q->i...", np.asarray(fe_field), wJ)


def _mixed_surface_local_residual_jax(
    *,
    u_vec: jnp.ndarray,
    slices: dict[str, slice],
    field_a: str,
    field_b: str,
    res_form: Callable[..., Any],
    ctx: SurfaceMixedFormContext,
    params: Any,
    includes_measure: dict[str, bool],
) -> jnp.ndarray:
    u_dict = _surface_local_u_dict(
        u_vec=u_vec,
        slices=slices,
        field_a=field_a,
        field_b=field_b,
        ctx=ctx,
    )
    fe_q = res_form(ctx, u_dict, params)
    res_parts = []
    for name in (field_a, field_b):
        fe_field = fe_q[name]
        fe = _reduce_surface_residual_jax(
            fe_field,
            includes_measure=bool(includes_measure.get(name, False)),
            w=ctx.w,
            detJ=ctx.detJ,
        )
        res_parts.append(fe)
    return jnp.concatenate(res_parts, axis=0)


def _mixed_surface_local_residual_numpy(
    *,
    u_vec: np.ndarray,
    slices: dict[str, slice],
    field_a: str,
    field_b: str,
    res_form: Callable[..., Any],
    ctx: SurfaceMixedFormContext,
    params: Any,
    includes_measure: dict[str, bool],
) -> np.ndarray:
    u_dict = _surface_local_u_dict(
        u_vec=u_vec,
        slices=slices,
        field_a=field_a,
        field_b=field_b,
        ctx=ctx,
    )
    fe_q = res_form(ctx, u_dict, params)
    res_parts = []
    for name in (field_a, field_b):
        fe_field = fe_q[name]
        fe = _reduce_surface_residual_numpy(
            fe_field,
            includes_measure=bool(includes_measure.get(name, False)),
            w=ctx.w,
            detJ=ctx.detJ,
        )
        res_parts.append(np.asarray(fe))
    return np.concatenate(res_parts, axis=0)


def _compute_mixed_surface_local_jacobian(
    *,
    u_local: np.ndarray,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    field_a: str,
    field_b: str,
    slices: dict[str, slice],
    res_form: Callable[..., Any],
    ctx: SurfaceMixedFormContext,
    params: Any,
    includes_measure: dict[str, bool],
) -> np.ndarray:
    if backend == "jax":
        def _res_local(u_vec):
            return _mixed_surface_local_residual_jax(
                u_vec=jnp.asarray(u_vec),
                slices=slices,
                field_a=field_a,
                field_b=field_b,
                res_form=res_form,
                ctx=ctx,
                params=params,
                includes_measure=includes_measure,
            )

        J_local = jax.jacrev(_res_local)(jnp.asarray(u_local))
        return np.asarray(J_local)
    if backend != "numpy":
        raise ValueError("backend must be 'jax' or 'numpy'")

    n_u = int(u_local.shape[0])
    J = np.zeros((n_u, n_u), dtype=float)
    for j in range(n_u):
        u_col = np.zeros((n_u,), dtype=float)
        u_col[j] = 1.0
        J[:, j] = _mixed_surface_local_residual_numpy(
            u_vec=u_col,
            slices=slices,
            field_a=field_a,
            field_b=field_b,
            res_form=res_form,
            ctx=ctx,
            params=params,
            includes_measure=includes_measure,
        )
    return J


def _accumulate_supermesh_residual_triangle(
    *,
    R: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    fa: int,
    fb: int,
    tol: float,
    skip_small_tri: bool,
    area_scale: float,
    facet_area_a: np.ndarray | None,
    facet_area_b: np.ndarray | None,
    diag_force: bool,
    diag_abs_detj: bool,
    quad_order: int,
    diag_qp_mode: str,
    diag_qp_path: str,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    pair_basis_builder: Callable[..., _SupermeshPairBasisData],
    trial_pair_basis_builder: Callable[..., _SupermeshPairBasisData],
    value_dim_a: int,
    value_dim_b: int,
    trial_value_dim_a: int,
    trial_value_dim_b: int,
    dof_source: str,
    grad_source: str,
    space_mode_a: str,
    space_mode_b: str,
    trial_space_mode_a: str,
    trial_space_mode_b: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    trial_facet_dofs_a: np.ndarray | None,
    trial_facet_dofs_b: np.ndarray | None,
    proj_diag: bool,
    normal_source: str,
    normal_sign: float,
    normals_a: np.ndarray | None,
    normals_b: np.ndarray | None,
    field_a: str,
    field_b: str,
    u_a: np.ndarray,
    u_b: np.ndarray,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    offset_a: int,
    offset_b: int,
    diag_facet: int,
    diag_max_q: int,
) -> None:
    area = _tri_area(a, b, c)
    if area <= tol:
        return
    if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
        area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
        if area_ref > 0.0 and area < area_scale * area_ref:
            return
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

    pair_data = pair_basis_builder(
        fa=int(fa),
        fb=int(fb),
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
        coords_a=coords_a,
        coords_b=coords_b,
        facets_a=facets_a,
        facets_b=facets_b,
        value_dim_a=value_dim_a,
        value_dim_b=value_dim_b,
        dof_source=dof_source,
        grad_source=grad_source,
        use_elem_a=use_elem_a,
        use_elem_b=use_elem_b,
        elem_conn_a=elem_conn_a,
        elem_conn_b=elem_conn_b,
        facet_to_elem_a=facet_to_elem_a,
        facet_to_elem_b=facet_to_elem_b,
        facet_dofs_a=facet_dofs_a,
        facet_dofs_b=facet_dofs_b,
        tol=tol,
    )
    if (
        int(trial_value_dim_a) != int(value_dim_a)
        or int(trial_value_dim_b) != int(value_dim_b)
        or str(trial_space_mode_a) != str(space_mode_a)
        or str(trial_space_mode_b) != str(space_mode_b)
        or not _same_optional_int_array(trial_facet_dofs_a, facet_dofs_a)
        or not _same_optional_int_array(trial_facet_dofs_b, facet_dofs_b)
    ):
        trial_pair = trial_pair_basis_builder(
            fa=int(fa),
            fb=int(fb),
            facet_a=facet_a,
            facet_b=facet_b,
            x_q=x_q,
            coords_a=coords_a,
            coords_b=coords_b,
            facets_a=facets_a,
            facets_b=facets_b,
            value_dim_a=trial_value_dim_a,
            value_dim_b=trial_value_dim_b,
            dof_source=dof_source,
            grad_source=grad_source,
            use_elem_a=use_elem_a,
            use_elem_b=use_elem_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            facet_dofs_a=trial_facet_dofs_a,
            facet_dofs_b=trial_facet_dofs_b,
            tol=tol,
        )
        pair_data = _merge_trial_pair_basis_data(pair_data, trial_pair)
    elem_id_a = pair_data.elem_id_a
    elem_id_b = pair_data.elem_id_b
    if proj_diag:
        _proj_diag_set_context(
            fa=int(fa),
            fb=int(fb),
            face_a=_facet_label(facet_a),
            face_b=_facet_label(facet_b),
            elem_a=elem_id_a,
            elem_b=elem_id_b,
        )

    test_Na = pair_data.test_Na if pair_data.test_Na is not None else pair_data.Na
    test_Nb = pair_data.test_Nb if pair_data.test_Nb is not None else pair_data.Nb
    trial_Na = pair_data.trial_Na if pair_data.trial_Na is not None else pair_data.Na
    trial_Nb = pair_data.trial_Nb if pair_data.trial_Nb is not None else pair_data.Nb
    test_gradNa = pair_data.test_gradNa if pair_data.test_gradNa is not None else pair_data.gradNa
    test_gradNb = pair_data.test_gradNb if pair_data.test_gradNb is not None else pair_data.gradNb
    trial_gradNa = pair_data.trial_gradNa if pair_data.trial_gradNa is not None else pair_data.gradNa
    trial_gradNb = pair_data.trial_gradNb if pair_data.trial_gradNb is not None else pair_data.gradNb
    test_dofs_local_a = pair_data.test_dofs_local_a if pair_data.test_dofs_local_a is not None else pair_data.dofs_local_a
    test_dofs_local_b = pair_data.test_dofs_local_b if pair_data.test_dofs_local_b is not None else pair_data.dofs_local_b
    trial_dofs_local_a = pair_data.trial_dofs_local_a if pair_data.trial_dofs_local_a is not None else pair_data.dofs_local_a
    trial_dofs_local_b = pair_data.trial_dofs_local_b if pair_data.trial_dofs_local_b is not None else pair_data.dofs_local_b
    nodes_a = pair_data.nodes_a
    nodes_b = pair_data.nodes_b

    normal, na, nb = _resolve_contact_normal(
        facet_id_a=int(fa),
        facet_id_b=int(fb),
        normals_a=normals_a,
        normals_b=normals_b,
        normal_source=normal_source,
        normal_sign=normal_sign,
        tol=tol,
    )
    if diag_force:
        dofs_a = int(offset_a) + np.asarray(test_dofs_local_a, dtype=int)
        dofs_b = int(offset_b) + np.asarray(test_dofs_local_b, dtype=int)
        _diag_contact_projection(
            fa=int(fa),
            fb=int(fb),
            quad_pts=quad_pts,
            quad_w=quad_w,
            x_q=x_q,
            Na=test_Na,
            Nb=test_Nb,
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            dofs_a=dofs_a,
            dofs_b=dofs_b,
            elem_coords_a=None,
            elem_coords_b=None,
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

    test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = _mixed_surface_space_aliases(
        res_form,
        field_a=field_a,
        field_b=field_b,
    )
    normal_q = None if normal is None else np.repeat(normal[None, :], quad_pts.shape[0], axis=0)
    ctx = _build_mixed_surface_context(
        field_a=field_a,
        field_b=field_b,
        test_space_key_a=test_space_key_a,
        test_space_key_b=test_space_key_b,
        unknown_space_key_a=unknown_space_key_a,
        unknown_space_key_b=unknown_space_key_b,
        test_Na=test_Na,
        test_Nb=test_Nb,
        trial_Na=trial_Na,
        trial_Nb=trial_Nb,
        test_gradNa=test_gradNa,
        test_gradNb=test_gradNb,
        trial_gradNa=trial_gradNa,
        trial_gradNb=trial_gradNb,
        test_value_dim_a=value_dim_a,
        test_value_dim_b=value_dim_b,
        trial_value_dim_a=value_dim_a,
        trial_value_dim_b=value_dim_b,
        x_q=x_q,
        w=quad_w,
        detJ=np.array([detJ], dtype=float),
        normal_q=normal_q,
    )

    u_elem = _surface_u_elem_with_space_aliases(
        field_a=field_a,
        field_b=field_b,
        unknown_space_key_a=unknown_space_key_a,
        unknown_space_key_b=unknown_space_key_b,
        u_local_a=np.asarray(u_a, dtype=float)[np.asarray(trial_dofs_local_a, dtype=int)],
        u_local_b=np.asarray(u_b, dtype=float)[np.asarray(trial_dofs_local_b, dtype=int)],
    )
    fe_q = res_form(ctx, u_elem, params)
    for name, dofs_local, offset in (
        (field_a, test_dofs_local_a, offset_a),
        (field_b, test_dofs_local_b, offset_b),
    ):
        fe_field = fe_q[name]
        if fe_field.ndim != 2 or fe_field.shape[0] != ctx.x_q.shape[0]:
            raise ValueError("surface residual must return shape (n_q, n_ldofs) per field")
        fe = _reduce_surface_residual_numpy(
            fe_field,
            includes_measure=bool(includes_measure.get(name, False)),
            w=ctx.w,
            detJ=ctx.detJ,
        )
        dofs = int(offset) + np.asarray(dofs_local, dtype=int)
        R[dofs] += fe


def _prepare_supermesh_jacobian_triangle_geometry(
    *,
    it: int,
    log_tri: bool,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    fa: int,
    fb: int,
    tol: float,
    skip_small_tri: bool,
    area_scale: float,
    facet_area_a: np.ndarray | None,
    facet_area_b: np.ndarray | None,
    diag_force: bool,
    diag_abs_detj: bool,
    guard: bool,
    skip_nonfinite: bool,
    detj_eps: float,
    quad_order: int,
    diag_qp_mode: str,
    diag_qp_path: str,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    tri_check: Callable[[str], None],
    trace_fn: Callable[[str], None],
    trace_time_fn: Callable[[str, float], None],
) -> _JacobianTriangleGeometryData | None:
    t_geom = time.perf_counter()
    area = _tri_area(a, b, c)
    if area <= tol:
        return None
    if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
        area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
        if area_ref > 0.0 and area < area_scale * area_ref:
            return None
    detJ = 2.0 * area
    if diag_force and diag_abs_detj:
        detJ = abs(detJ)
    if guard:
        if not np.isfinite(detJ):
            if log_tri:
                trace_fn(f"[CONTACT] tri {it} detJ non-finite; skip")
            if skip_nonfinite:
                return None
            raise RuntimeError(f"[CONTACT] tri {it} detJ non-finite")
        if detj_eps > 0.0 and abs(detJ) < detj_eps:
            if log_tri:
                trace_fn(f"[CONTACT] tri {it} detJ too small {detJ:.3e}; skip")
            return None
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
            trace_fn(f"[CONTACT] tri {it} x_q non-finite; skip")
        if skip_nonfinite:
            return None
        raise RuntimeError(f"[CONTACT] tri {it} x_q non-finite")
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} geom_done", t_geom)
    tri_check("geom_done")
    return _JacobianTriangleGeometryData(
        detJ=float(detJ),
        quad_pts=quad_pts,
        quad_w=quad_w,
        quad_source=quad_source,
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
    )


def _accumulate_supermesh_jacobian_triangle_core(
    *,
    it: int,
    log_tri: bool,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    quad_pts: np.ndarray,
    quad_w: np.ndarray,
    quad_source: str,
    detJ: float,
    tol: float,
    pair_basis_builder: Callable[..., _SupermeshPairBasisData],
    trial_pair_basis_builder: Callable[..., _SupermeshPairBasisData],
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    trial_value_dim_a: int,
    trial_value_dim_b: int,
    dof_source: str,
    grad_source: str,
    space_mode_a: str,
    space_mode_b: str,
    trial_space_mode_a: str,
    trial_space_mode_b: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    trial_facet_dofs_a: np.ndarray | None,
    trial_facet_dofs_b: np.ndarray | None,
    proj_diag: bool,
    diag_map: bool,
    diag_n: bool,
    diag_force: bool,
    diag_facet: int,
    diag_max_q: int,
    guard: bool,
    skip_nonfinite: bool,
    normal_source: str,
    normal_sign: float,
    normals_a: np.ndarray | None,
    normals_b: np.ndarray | None,
    field_a: str,
    field_b: str,
    u_a: np.ndarray,
    u_b: np.ndarray,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    offset_a: int,
    offset_b: int,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    sparse: bool,
    rows: list[int],
    cols: list[int],
    data: list[float],
    K_dense: np.ndarray | None,
    tri_check: Callable[[str], None],
    trace_fn: Callable[[str], None],
    trace_time_fn: Callable[[str, float], None],
) -> None:
    pair_data = pair_basis_builder(
        fa=int(fa),
        fb=int(fb),
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
        coords_a=coords_a,
        coords_b=coords_b,
        facets_a=facets_a,
        facets_b=facets_b,
        value_dim_a=value_dim_a,
        value_dim_b=value_dim_b,
        dof_source=dof_source,
        grad_source=grad_source,
        use_elem_a=use_elem_a,
        use_elem_b=use_elem_b,
        elem_conn_a=elem_conn_a,
        elem_conn_b=elem_conn_b,
        facet_to_elem_a=facet_to_elem_a,
        facet_to_elem_b=facet_to_elem_b,
        facet_dofs_a=facet_dofs_a,
        facet_dofs_b=facet_dofs_b,
        tol=tol,
    )
    if (
        int(trial_value_dim_a) != int(value_dim_a)
        or int(trial_value_dim_b) != int(value_dim_b)
        or str(trial_space_mode_a) != str(space_mode_a)
        or str(trial_space_mode_b) != str(space_mode_b)
        or not _same_optional_int_array(trial_facet_dofs_a, facet_dofs_a)
        or not _same_optional_int_array(trial_facet_dofs_b, facet_dofs_b)
    ):
        trial_pair = trial_pair_basis_builder(
            fa=int(fa),
            fb=int(fb),
            facet_a=facet_a,
            facet_b=facet_b,
            x_q=x_q,
            coords_a=coords_a,
            coords_b=coords_b,
            facets_a=facets_a,
            facets_b=facets_b,
            value_dim_a=trial_value_dim_a,
            value_dim_b=trial_value_dim_b,
            dof_source=dof_source,
            grad_source=grad_source,
            use_elem_a=use_elem_a,
            use_elem_b=use_elem_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            facet_dofs_a=trial_facet_dofs_a,
            facet_dofs_b=trial_facet_dofs_b,
            tol=tol,
        )
        pair_data = _merge_trial_pair_basis_data(pair_data, trial_pair)
    elem_id_a = pair_data.elem_id_a
    elem_id_b = pair_data.elem_id_b
    local_a = pair_data.local_a
    local_b = pair_data.local_b
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
    test_Na = pair_data.test_Na if pair_data.test_Na is not None else pair_data.Na
    test_Nb = pair_data.test_Nb if pair_data.test_Nb is not None else pair_data.Nb
    trial_Na = pair_data.trial_Na if pair_data.trial_Na is not None else pair_data.Na
    trial_Nb = pair_data.trial_Nb if pair_data.trial_Nb is not None else pair_data.Nb
    test_gradNa = pair_data.test_gradNa if pair_data.test_gradNa is not None else pair_data.gradNa
    test_gradNb = pair_data.test_gradNb if pair_data.test_gradNb is not None else pair_data.gradNb
    trial_gradNa = pair_data.trial_gradNa if pair_data.trial_gradNa is not None else pair_data.gradNa
    trial_gradNb = pair_data.trial_gradNb if pair_data.trial_gradNb is not None else pair_data.gradNb
    test_dofs_local_a = pair_data.test_dofs_local_a if pair_data.test_dofs_local_a is not None else pair_data.dofs_local_a
    test_dofs_local_b = pair_data.test_dofs_local_b if pair_data.test_dofs_local_b is not None else pair_data.dofs_local_b
    trial_dofs_local_a = pair_data.trial_dofs_local_a if pair_data.trial_dofs_local_a is not None else pair_data.dofs_local_a
    trial_dofs_local_b = pair_data.trial_dofs_local_b if pair_data.trial_dofs_local_b is not None else pair_data.dofs_local_b
    nodes_a = pair_data.nodes_a
    nodes_b = pair_data.nodes_b
    if guard and (not np.isfinite(test_Na).all() or not np.isfinite(test_Nb).all()):
        if log_tri:
            trace_fn(f"[CONTACT] tri {it} N non-finite; skip")
        if skip_nonfinite:
            return
        raise RuntimeError(f"[CONTACT] tri {it} N non-finite")
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} basis_done", t_basis)
    tri_check("basis_done")

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
        dofs_a = int(offset_a) + np.asarray(test_dofs_local_a, dtype=int)
        dofs_b = int(offset_b) + np.asarray(test_dofs_local_b, dtype=int)
        samples = min(3, Na.shape[0])
        print("[fluxfem][diag][contact-n] first facet q-points")
        print(f"  nodes_a={nodes_a.tolist()} nodes_b={nodes_b.tolist()}")
        print(f"  dofs_a={dofs_a.tolist()} dofs_b={dofs_b.tolist()}")
        for qi in range(samples):
            print(f"  q{qi} x={x_q[qi].tolist()} Na={test_Na[qi].tolist()} Nb={test_Nb[qi].tolist()}")
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

    normal_q = None if normal is None else np.repeat(normal[None, :], quad_pts.shape[0], axis=0)
    t_jac = time.perf_counter()
    formulation_tag = getattr(res_form, "_ff_contact_formulation", None)
    fastpath_tag = getattr(res_form, "_ff_contact_backend_fastpath", None)
    use_fast_pair_nitsche = (
        backend == "numpy"
        and formulation_tag == "pair_nitsche_penalty"
        and fastpath_tag == "numpy_local_kernel"
    )
    if use_fast_pair_nitsche:
        if normal_q is None:
            raise ValueError("pair_nitsche_penalty fast path requires surface normals.")
        J_local_np = _fast_pair_nitsche_penalty_local_matrix(
            Na=np.asarray(test_Na, dtype=float),
            Nb=np.asarray(test_Nb, dtype=float),
            gradNa=np.asarray(test_gradNa, dtype=float),
            gradNb=np.asarray(test_gradNb, dtype=float),
            normal_q=np.asarray(normal_q, dtype=float),
            w=np.asarray(quad_w, dtype=float),
            detJ=np.array([detJ], dtype=float),
            alpha=float(getattr(params, "alpha")),
            inv_h=float(getattr(params, "inv_h")),
            lam=float(getattr(params, "lam")),
            mu=float(getattr(params, "mu")),
            use_penalty=float(getattr(params, "use_penalty", 1.0)),
            use_traction=float(getattr(params, "use_traction", 1.0)),
            value_dim_a=int(value_dim_a),
            value_dim_b=int(value_dim_b),
        )
    else:
        test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = (
            _mixed_surface_space_aliases(
                res_form,
                field_a=field_a,
                field_b=field_b,
            )
        )
        ctx = _build_mixed_surface_context(
            field_a=field_a,
            field_b=field_b,
            test_space_key_a=test_space_key_a,
            test_space_key_b=test_space_key_b,
            unknown_space_key_a=unknown_space_key_a,
            unknown_space_key_b=unknown_space_key_b,
            test_Na=test_Na,
            test_Nb=test_Nb,
            trial_Na=trial_Na,
            trial_Nb=trial_Nb,
            test_gradNa=test_gradNa,
            test_gradNb=test_gradNb,
            trial_gradNa=trial_gradNa,
            trial_gradNb=trial_gradNb,
            test_value_dim_a=value_dim_a,
            test_value_dim_b=value_dim_b,
            trial_value_dim_a=value_dim_a,
            trial_value_dim_b=value_dim_b,
            x_q=x_q,
            w=quad_w,
            detJ=np.array([detJ], dtype=float),
            normal_q=normal_q,
        )
        u_elem = _surface_u_elem_with_space_aliases(
            field_a=field_a,
            field_b=field_b,
            unknown_space_key_a=unknown_space_key_a,
            unknown_space_key_b=unknown_space_key_b,
            u_local_a=np.asarray(u_a, dtype=float)[np.asarray(trial_dofs_local_a, dtype=int)],
            u_local_b=np.asarray(u_b, dtype=float)[np.asarray(trial_dofs_local_b, dtype=int)],
        )
        u_local = np.concatenate([u_elem[field_a], u_elem[field_b]], axis=0)
        sizes = (u_elem[field_a].shape[0], u_elem[field_b].shape[0])
        slices = {
            field_a: slice(0, sizes[0]),
            field_b: slice(sizes[0], sizes[0] + sizes[1]),
        }
        J_local_np = _compute_mixed_surface_local_jacobian(
            u_local=np.asarray(u_local, dtype=float),
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            field_a=field_a,
            field_b=field_b,
            slices=slices,
            res_form=res_form,
            ctx=ctx,
            params=params,
            includes_measure=includes_measure,
        )
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} jac_done", t_jac)
    tri_check("jac_done")

    row_dofs_a = int(offset_a) + np.asarray(test_dofs_local_a, dtype=int)
    row_dofs_b = int(offset_b) + np.asarray(test_dofs_local_b, dtype=int)
    col_dofs_a = int(offset_a) + np.asarray(trial_dofs_local_a, dtype=int)
    col_dofs_b = int(offset_b) + np.asarray(trial_dofs_local_b, dtype=int)
    if diag_force:
        _diag_contact_projection(
            fa=int(fa),
            fb=int(fb),
            quad_pts=quad_pts,
            quad_w=quad_w,
            x_q=x_q,
            Na=test_Na,
            Nb=test_Nb,
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            dofs_a=row_dofs_a,
            dofs_b=row_dofs_b,
            elem_coords_a=None,
            elem_coords_b=None,
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
    row_dofs = np.concatenate([row_dofs_a, row_dofs_b], axis=0)
    col_dofs = np.concatenate([col_dofs_a, col_dofs_b], axis=0)
    if sparse:
        n_row_ldofs = int(row_dofs.shape[0])
        n_col_ldofs = int(col_dofs.shape[0])
        rows.extend(np.repeat(row_dofs, n_col_ldofs).tolist())
        cols.extend(np.tile(col_dofs, n_row_ldofs).tolist())
        data.extend(J_local_np.reshape(-1).tolist())
    else:
        assert K_dense is not None
        K_dense[np.ix_(row_dofs, col_dofs)] += J_local_np
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} scatter_done", t_scatter)
    tri_check("scatter_done")


def _accumulate_projection_jacobian_batch(
    *,
    batch: dict[str, np.ndarray],
    field_a: str,
    field_b: str,
    value_dim_a: int,
    value_dim_b: int,
    u_a: np.ndarray,
    u_b: np.ndarray,
    offset_a: int,
    offset_b: int,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    sparse: bool,
    rows: list[int],
    cols: list[int],
    data: list[float],
    K_dense: np.ndarray | None,
) -> None:
    test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = _mixed_surface_space_aliases(
        res_form,
        field_a=field_a,
        field_b=field_b,
    )
    Na = batch["Na"]
    Nb = batch["Nb"]
    gradNa = batch["gradNa"]
    gradNb = batch["gradNb"]
    nodes_a = batch["nodes_a"]
    nodes_b = batch["nodes_b"]
    normal_q = batch["normal"]

    ctx = _build_mixed_surface_context(
        field_a=field_a,
        field_b=field_b,
        test_space_key_a=test_space_key_a,
        test_space_key_b=test_space_key_b,
        unknown_space_key_a=unknown_space_key_a,
        unknown_space_key_b=unknown_space_key_b,
        test_Na=Na,
        test_Nb=Nb,
        trial_Na=Na,
        trial_Nb=Nb,
        test_gradNa=gradNa,
        test_gradNb=gradNb,
        trial_gradNa=gradNa,
        trial_gradNb=gradNb,
        test_value_dim_a=value_dim_a,
        test_value_dim_b=value_dim_b,
        trial_value_dim_a=value_dim_a,
        trial_value_dim_b=value_dim_b,
        x_q=batch["x_q"],
        w=batch["w"],
        detJ=batch["detJ"],
        normal_q=normal_q,
    )

    u_elem = _surface_u_elem_with_space_aliases(
        field_a=field_a,
        field_b=field_b,
        unknown_space_key_a=unknown_space_key_a,
        unknown_space_key_b=unknown_space_key_b,
        u_local_a=_gather_u_local(u_a, nodes_a, value_dim_a),
        u_local_b=_gather_u_local(u_b, nodes_b, value_dim_b),
    )
    u_local = np.concatenate([u_elem[field_a], u_elem[field_b]], axis=0)
    sizes = (u_elem[field_a].shape[0], u_elem[field_b].shape[0])
    slices = {
        field_a: slice(0, sizes[0]),
        field_b: slice(sizes[0], sizes[0] + sizes[1]),
    }

    J_local_np = _compute_mixed_surface_local_jacobian(
        u_local=np.asarray(u_local, dtype=float),
        backend=backend,
        fd_eps=fd_eps,
        fd_mode=fd_mode,
        fd_block_size=fd_block_size,
        field_a=field_a,
        field_b=field_b,
        slices=slices,
        res_form=res_form,
        ctx=ctx,
        params=params,
        includes_measure=includes_measure,
    )

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


def _apply_projection_jacobian_batches(
    *,
    batches: list[dict[str, np.ndarray]],
    field_a: str,
    field_b: str,
    value_dim_a: int,
    value_dim_b: int,
    u_a: np.ndarray,
    u_b: np.ndarray,
    offset_a: int,
    offset_b: int,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    sparse: bool,
    rows: list[int],
    cols: list[int],
    data: list[float],
    K_dense: np.ndarray | None,
) -> None:
    u_a_np = np.asarray(u_a, dtype=float)
    u_b_np = np.asarray(u_b, dtype=float)
    for batch in batches:
        _accumulate_projection_jacobian_batch(
            batch=batch,
            field_a=field_a,
            field_b=field_b,
            value_dim_a=value_dim_a,
            value_dim_b=value_dim_b,
            u_a=u_a_np,
            u_b=u_b_np,
            offset_a=int(offset_a),
            offset_b=int(offset_b),
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            res_form=res_form,
            params=params,
            includes_measure=includes_measure,
            sparse=sparse,
            rows=rows,
            cols=cols,
            data=data,
            K_dense=K_dense,
        )


def _prepare_supermesh_pair_basis_data(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    space_mode_a: str,
    space_mode_b: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    elem_id_a, elem_nodes_a, elem_coords_a = _resolve_facet_element_context(
        use_elem=use_elem_a,
        facet_id=int(fa),
        facet_to_elem=facet_to_elem_a,
        elem_conn=elem_conn_a,
        coords=coords_a,
        invalid_map_error="facet_to_elem_a has invalid mapping",
        unsupported_error="surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only",
    )
    elem_id_b, elem_nodes_b, elem_coords_b = _resolve_facet_element_context(
        use_elem=use_elem_b,
        facet_id=int(fb),
        facet_to_elem=facet_to_elem_b,
        elem_conn=elem_conn_b,
        coords=coords_b,
        invalid_map_error="facet_to_elem_b has invalid mapping",
        unsupported_error="surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only",
    )
    Na, gradNa, dofs_local_a, nodes_a, local_a = _prepare_supermesh_side_field_data(
        x_q=x_q,
        facet_id=int(fa),
        facet_nodes=facet_a,
        coords=coords_a,
        value_dim=value_dim_a,
        n_facets=facets_a.shape[0],
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode=space_mode_a,
        use_elem=use_elem_a,
        elem_nodes=elem_nodes_a,
        elem_coords=elem_coords_a,
        facet_dofs=facet_dofs_a,
        tol=tol,
        volume_dof_error="dof_source 'volume' requires elem_conn_a and facet_to_elem_a",
    )
    Nb, gradNb, dofs_local_b, nodes_b, local_b = _prepare_supermesh_side_field_data(
        x_q=x_q,
        facet_id=int(fb),
        facet_nodes=facet_b,
        coords=coords_b,
        value_dim=value_dim_b,
        n_facets=facets_b.shape[0],
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode=space_mode_b,
        use_elem=use_elem_b,
        elem_nodes=elem_nodes_b,
        elem_coords=elem_coords_b,
        facet_dofs=facet_dofs_b,
        tol=tol,
        volume_dof_error="dof_source 'volume' requires elem_conn_b and facet_to_elem_b",
    )
    return _SupermeshPairBasisData(
        elem_id_a=elem_id_a,
        elem_id_b=elem_id_b,
        Na=Na,
        Nb=Nb,
        gradNa=gradNa,
        gradNb=gradNb,
        dofs_local_a=dofs_local_a,
        dofs_local_b=dofs_local_b,
        nodes_a=nodes_a,
        nodes_b=nodes_b,
        local_a=local_a,
        local_b=local_b,
        test_Na=Na,
        test_Nb=Nb,
        trial_Na=Na,
        trial_Nb=Nb,
        test_gradNa=gradNa,
        test_gradNb=gradNb,
        trial_gradNa=gradNa,
        trial_gradNb=gradNb,
        test_dofs_local_a=dofs_local_a,
        test_dofs_local_b=dofs_local_b,
        trial_dofs_local_a=dofs_local_a,
        trial_dofs_local_b=dofs_local_b,
        test_nodes_a=nodes_a,
        test_nodes_b=nodes_b,
        trial_nodes_a=nodes_a,
        trial_nodes_b=nodes_b,
    )


def _prepare_supermesh_pair_basis_data_nodal(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
        coords_a=coords_a,
        coords_b=coords_b,
        facets_a=facets_a,
        facets_b=facets_b,
        value_dim_a=value_dim_a,
        value_dim_b=value_dim_b,
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode_a="nodal",
        space_mode_b="nodal",
        use_elem_a=use_elem_a,
        use_elem_b=use_elem_b,
        elem_conn_a=elem_conn_a,
        elem_conn_b=elem_conn_b,
        facet_to_elem_a=facet_to_elem_a,
        facet_to_elem_b=facet_to_elem_b,
        facet_dofs_a=facet_dofs_a,
        facet_dofs_b=facet_dofs_b,
        tol=tol,
    )


def _prepare_supermesh_pair_basis_data_p0_a_nodal_b(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
        coords_a=coords_a,
        coords_b=coords_b,
        facets_a=facets_a,
        facets_b=facets_b,
        value_dim_a=value_dim_a,
        value_dim_b=value_dim_b,
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode_a="p0",
        space_mode_b="nodal",
        use_elem_a=use_elem_a,
        use_elem_b=use_elem_b,
        elem_conn_a=elem_conn_a,
        elem_conn_b=elem_conn_b,
        facet_to_elem_a=facet_to_elem_a,
        facet_to_elem_b=facet_to_elem_b,
        facet_dofs_a=facet_dofs_a,
        facet_dofs_b=facet_dofs_b,
        tol=tol,
    )


def _prepare_supermesh_pair_basis_data_nodal_a_p0_b(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
        coords_a=coords_a,
        coords_b=coords_b,
        facets_a=facets_a,
        facets_b=facets_b,
        value_dim_a=value_dim_a,
        value_dim_b=value_dim_b,
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode_a="nodal",
        space_mode_b="p0",
        use_elem_a=use_elem_a,
        use_elem_b=use_elem_b,
        elem_conn_a=elem_conn_a,
        elem_conn_b=elem_conn_b,
        facet_to_elem_a=facet_to_elem_a,
        facet_to_elem_b=facet_to_elem_b,
        facet_dofs_a=facet_dofs_a,
        facet_dofs_b=facet_dofs_b,
        tol=tol,
    )


def _prepare_supermesh_pair_basis_data_p0_p0(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
        coords_a=coords_a,
        coords_b=coords_b,
        facets_a=facets_a,
        facets_b=facets_b,
        value_dim_a=value_dim_a,
        value_dim_b=value_dim_b,
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode_a="p0",
        space_mode_b="p0",
        use_elem_a=use_elem_a,
        use_elem_b=use_elem_b,
        elem_conn_a=elem_conn_a,
        elem_conn_b=elem_conn_b,
        facet_to_elem_a=facet_to_elem_a,
        facet_to_elem_b=facet_to_elem_b,
        facet_dofs_a=facet_dofs_a,
        facet_dofs_b=facet_dofs_b,
        tol=tol,
    )


def _select_supermesh_pair_basis_builder(
    *,
    use_p0_a: bool,
    use_p0_b: bool,
) -> Callable[..., _SupermeshPairBasisData]:
    if not use_p0_a and not use_p0_b:
        return _prepare_supermesh_pair_basis_data_nodal
    if use_p0_a and use_p0_b:
        return _prepare_supermesh_pair_basis_data_p0_p0
    if use_p0_a:
        return _prepare_supermesh_pair_basis_data_p0_a_nodal_b
    return _prepare_supermesh_pair_basis_data_nodal_a_p0_b


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


def assemble_contact_coupling_matrices(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    *,
    tol: float = 1e-8,
    quad_order: int = 0,
) -> tuple[ContactCouplingMatrix, ContactCouplingMatrix]:
    """
    Assemble contact coupling matrices M_aa and M_ab on the supermesh.
    """
    if quad_order <= 0:
        quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        quad_w = np.array([0.5], dtype=float)
    else:
        quad_pts, quad_w = _tri_quadrature(int(quad_order))
    use_jax_geometry = _uses_jax_geometry(supermesh_coords, surface_a.coords, surface_b.coords)
    if use_jax_geometry:
        facets_a = np.asarray(surface_a.conn, dtype=int)
        facets_b = np.asarray(surface_b.conn, dtype=int)
        if facets_a.shape[1] != 3 or facets_b.shape[1] != 3:
            raise NotImplementedError(
                "JAX-traced contact coupling matrices are currently implemented only for tri3 facets."
            )
        coords_a_j = jnp.asarray(surface_a.coords)
        coords_b_j = jnp.asarray(surface_b.coords)
        supermesh_coords_j = jnp.asarray(supermesh_coords)

        rows_aa: list[int] = []
        cols_aa: list[int] = []
        data_aa: list[jnp.ndarray] = []
        rows_ab: list[int] = []
        cols_ab: list[int] = []
        data_ab: list[jnp.ndarray] = []

        for tri, fa, fb in zip(np.asarray(supermesh_conn, dtype=int), source_facets_a, source_facets_b):
            a = supermesh_coords_j[tri[0]]
            b = supermesh_coords_j[tri[1]]
            c = supermesh_coords_j[tri[2]]
            detJ = jnp.linalg.norm(jnp.cross(b - a, c - a))
            facet_a = facets_a[int(fa)]
            facet_b = facets_b[int(fb)]
            for (r, s), w_ref in zip(quad_pts, quad_w):
                x_q = a + float(r) * (b - a) + float(s) * (c - a)
                weight = detJ * float(w_ref)
                Na = _tri3_shape_values_jax(x_q, facet_a, coords_a_j)
                Nb = _tri3_shape_values_jax(x_q, facet_b, coords_b_j)

                for i, node_i in enumerate(facet_a):
                    for j, node_j in enumerate(facet_a):
                        rows_aa.append(int(node_i))
                        cols_aa.append(int(node_j))
                        data_aa.append(weight * Na[i] * Na[j])

                for i, node_i in enumerate(facet_a):
                    for j, node_j in enumerate(facet_b):
                        rows_ab.append(int(node_i))
                        cols_ab.append(int(node_j))
                        data_ab.append(weight * Na[i] * Nb[j])

        n_a = int(coords_a_j.shape[0])
        n_b = int(coords_b_j.shape[0])
        M_aa = ContactCouplingMatrix(
            rows=np.asarray(rows_aa, dtype=int),
            cols=np.asarray(cols_aa, dtype=int),
            data=jnp.stack(data_aa) if data_aa else jnp.zeros((0,), dtype=coords_a_j.dtype),
            shape=(n_a, n_a),
        )
        M_ab = ContactCouplingMatrix(
            rows=np.asarray(rows_ab, dtype=int),
            cols=np.asarray(cols_ab, dtype=int),
            data=jnp.stack(data_ab) if data_ab else jnp.zeros((0,), dtype=coords_b_j.dtype),
            shape=(n_a, n_b),
        )
        return M_aa, M_ab

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
        detJ = 2.0 * _tri_area(a, b, c)
        if detJ <= tol:
            continue

        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        for (r, s), w_ref in zip(quad_pts, quad_w):
            x_q = a + float(r) * (b - a) + float(s) * (c - a)
            weight = detJ * float(w_ref)
            Na = _facet_shape_values(x_q, facet_a, coords_a, tol=tol)
            Nb = _facet_shape_values(x_q, facet_b, coords_b, tol=tol)

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
    M_aa = ContactCouplingMatrix(
        rows=np.asarray(rows_aa, dtype=int),
        cols=np.asarray(cols_aa, dtype=int),
        data=np.asarray(data_aa, dtype=float),
        shape=(n_a, n_a),
    )
    M_ab = ContactCouplingMatrix(
        rows=np.asarray(rows_ab, dtype=int),
        cols=np.asarray(cols_ab, dtype=int),
        data=np.asarray(data_ab, dtype=float),
        shape=(n_a, n_b),
    )
    return M_aa, M_ab


def assemble_contact_interface_residual(
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
    trial_value_dim_a: int | None = None,
    trial_value_dim_b: int | None = None,
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
    space_mode_a: str = "nodal",
    space_mode_b: str = "nodal",
    trial_space_mode_a: str | None = None,
    trial_space_mode_b: str | None = None,
    facet_dofs_a: np.ndarray | None = None,
    facet_dofs_b: np.ndarray | None = None,
    trial_facet_dofs_a: np.ndarray | None = None,
    trial_facet_dofs_b: np.ndarray | None = None,
    quad_order: int = 0,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Assemble mixed surface residual over a supermesh (centroid quadrature).

    normal_source can be "master", "slave", "a", "b", or "avg"; use master_field
    to pick which field acts as the master when normal_source is "master"/"slave".
    dof_source="volume" assembles nodal fields into element nodes (requires elem_conn_* mappings).
    space_mode_* supports:
    - "nodal": existing nodal FE space behavior.
    - "p0": facet-wise constant space (one block of value_dim DOFs per facet, or
      custom mapping via facet_dofs_*).
    """
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    n_a = _field_n_dofs(
        n_nodes=int(coords_a.shape[0]),
        n_facets=int(facets_a.shape[0]),
        value_dim=int(value_dim_a),
        space_mode=space_mode_a,
        facet_dofs=facet_dofs_a,
    )
    n_b = _field_n_dofs(
        n_nodes=int(coords_b.shape[0]),
        n_facets=int(facets_b.shape[0]),
        value_dim=int(value_dim_b),
        space_mode=space_mode_b,
        facet_dofs=facet_dofs_b,
    )
    if offset_b is None:
        offset_b = offset_a + n_a
    if trial_value_dim_a is None:
        trial_value_dim_a = value_dim_a
    if trial_value_dim_b is None:
        trial_value_dim_b = value_dim_b
    if trial_space_mode_a is None:
        trial_space_mode_a = space_mode_a
    if trial_space_mode_b is None:
        trial_space_mode_b = space_mode_b
    if trial_facet_dofs_a is None:
        trial_facet_dofs_a = facet_dofs_a
    if trial_facet_dofs_b is None:
        trial_facet_dofs_b = facet_dofs_b
    distinct_trial_layout = (
        int(trial_value_dim_a) != int(value_dim_a)
        or int(trial_value_dim_b) != int(value_dim_b)
        or str(trial_space_mode_a) != str(space_mode_a)
        or str(trial_space_mode_b) != str(space_mode_b)
        or not _same_optional_int_array(trial_facet_dofs_a, facet_dofs_a)
        or not _same_optional_int_array(trial_facet_dofs_b, facet_dofs_b)
    )
    n_total = int(offset_b + n_b)
    R: np.ndarray = np.zeros((n_total,), dtype=float)
    u_a_np = np.asarray(u_a, dtype=float)
    u_b_np = np.asarray(u_b, dtype=float)

    trace = os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE", "0") not in ("0", "", "false", "False")

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
        if hasattr(surface_a, "facet_areas"):
            facet_area_a = np.asarray(surface_a.facet_areas(), dtype=float)
        else:
            facet_area_a = np.array([_facet_area_estimate(fa, coords_a) for fa in facets_a], dtype=float)
        if hasattr(surface_b, "facet_areas"):
            facet_area_b = np.asarray(surface_b.facet_areas(), dtype=float)
        else:
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

    use_p0_a, use_p0_b = _validate_contact_interface_space_setup(
        grad_source=grad_source,
        dof_source=dof_source,
        space_mode_a=space_mode_a,
        space_mode_b=space_mode_b,
    )
    global _DEBUG_SURFACE_SOURCE_ONCE
    if grad_source == "surface" and not _DEBUG_SURFACE_SOURCE_ONCE:
        print("[fluxfem] using surface gradN in contact interface")
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

    normal_source = _resolve_normal_source_option(
        normal_source=normal_source,
        normal_from=normal_from,
        master_field=master_field,
        field_a=field_a,
        field_b=field_b,
        diag_force=diag_force,
        diag_normal=diag_normal,
    )

    contact_interface_mode = os.getenv("FLUXFEM_CONTACT_INTERFACE_MODE", "supermesh").lower()
    if contact_interface_mode == "projection" and not (use_p0_a or use_p0_b) and not distinct_trial_layout:
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
            test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = (
                _mixed_surface_space_aliases(
                    res_form,
                    field_a=field_a,
                    field_b=field_b,
                )
            )
            for batch in batches:
                Na = batch["Na"]
                Nb = batch["Nb"]
                gradNa = batch["gradNa"]
                gradNb = batch["gradNb"]
                nodes_a = batch["nodes_a"]
                nodes_b = batch["nodes_b"]
                normal_q = batch["normal"]

                ctx = _build_mixed_surface_context(
                    field_a=field_a,
                    field_b=field_b,
                    test_space_key_a=test_space_key_a,
                    test_space_key_b=test_space_key_b,
                    unknown_space_key_a=unknown_space_key_a,
                    unknown_space_key_b=unknown_space_key_b,
                    test_Na=Na,
                    test_Nb=Nb,
                    trial_Na=Na,
                    trial_Nb=Nb,
                    test_gradNa=gradNa,
                    test_gradNb=gradNb,
                    trial_gradNa=gradNa,
                    trial_gradNb=gradNb,
                    test_value_dim_a=value_dim_a,
                    test_value_dim_b=value_dim_b,
                    trial_value_dim_a=value_dim_a,
                    trial_value_dim_b=value_dim_b,
                    x_q=batch["x_q"],
                    w=batch["w"],
                    detJ=batch["detJ"],
                    normal_q=normal_q,
                )
                u_elem = _surface_u_elem_with_space_aliases(
                    field_a=field_a,
                    field_b=field_b,
                    unknown_space_key_a=unknown_space_key_a,
                    unknown_space_key_b=unknown_space_key_b,
                    u_local_a=_gather_u_local(u_a_np, nodes_a, value_dim_a),
                    u_local_b=_gather_u_local(u_b_np, nodes_b, value_dim_b),
                )
                fe_q = res_form(ctx, u_elem, params)
                for name, facet, value_dim, offset in (
                    (field_a, nodes_a, value_dim_a, offset_a),
                    (field_b, nodes_b, value_dim_b, offset_b),
                ):
                    fe_field = fe_q[name]
                    if fe_field.ndim != 2 or fe_field.shape[0] != ctx.x_q.shape[0]:
                        raise ValueError("mixed surface residual must return (n_q, n_ldofs)")
                    fe = _reduce_surface_residual_jax(
                        fe_field,
                        includes_measure=bool(includes_measure.get(name, False)),
                        w=ctx.w,
                        detJ=ctx.detJ,
                    )
                    dofs = _global_dof_indices(facet, value_dim, int(offset))
                    R[dofs] += np.asarray(fe)
            return R

    pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=use_p0_a,
        use_p0_b=use_p0_b,
    )
    trial_use_p0_a = str(trial_space_mode_a) == "p0"
    trial_use_p0_b = str(trial_space_mode_b) == "p0"
    trial_pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=trial_use_p0_a,
        use_p0_b=trial_use_p0_b,
    )
    for (tri, a, b, c), fa, fb in zip(
        _iter_supermesh_tris(supermesh_coords, supermesh_conn),
        source_facets_a,
        source_facets_b,
    ):
        _accumulate_supermesh_residual_triangle(
            R=R,
            a=a,
            b=b,
            c=c,
            fa=int(fa),
            fb=int(fb),
            tol=tol,
            skip_small_tri=skip_small_tri,
            area_scale=area_scale,
            facet_area_a=facet_area_a,
            facet_area_b=facet_area_b,
            diag_force=diag_force,
            diag_abs_detj=diag_abs_detj,
            quad_order=quad_order,
            diag_qp_mode=diag_qp_mode,
            diag_qp_path=diag_qp_path,
            coords_a=coords_a,
            coords_b=coords_b,
            facets_a=facets_a,
            facets_b=facets_b,
            pair_basis_builder=pair_basis_builder,
            trial_pair_basis_builder=trial_pair_basis_builder,
            value_dim_a=value_dim_a,
            value_dim_b=value_dim_b,
            trial_value_dim_a=int(trial_value_dim_a),
            trial_value_dim_b=int(trial_value_dim_b),
            dof_source=dof_source,
            grad_source=grad_source,
            space_mode_a=space_mode_a,
            space_mode_b=space_mode_b,
            trial_space_mode_a=str(trial_space_mode_a),
            trial_space_mode_b=str(trial_space_mode_b),
            use_elem_a=use_elem_a,
            use_elem_b=use_elem_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            facet_dofs_a=facet_dofs_a,
            facet_dofs_b=facet_dofs_b,
            trial_facet_dofs_a=trial_facet_dofs_a,
            trial_facet_dofs_b=trial_facet_dofs_b,
            proj_diag=proj_diag,
            normal_source=normal_source,
            normal_sign=normal_sign,
            normals_a=normals_a,
            normals_b=normals_b,
            field_a=field_a,
            field_b=field_b,
            u_a=u_a_np,
            u_b=u_b_np,
            res_form=res_form,
            params=params,
            includes_measure=includes_measure,
            offset_a=int(offset_a),
            offset_b=int(offset_b),
            diag_facet=diag_facet,
            diag_max_q=diag_max_q,
        )
    if proj_diag:
        _proj_diag_report()
    return R


def assemble_contact_interface_jacobian(
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
    trial_value_dim_a: int | None = None,
    trial_value_dim_b: int | None = None,
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
    space_mode_a: str = "nodal",
    space_mode_b: str = "nodal",
    trial_space_mode_a: str | None = None,
    trial_space_mode_b: str | None = None,
    facet_dofs_a: np.ndarray | None = None,
    facet_dofs_b: np.ndarray | None = None,
    trial_facet_dofs_a: np.ndarray | None = None,
    trial_facet_dofs_b: np.ndarray | None = None,
    quad_order: int = 0,
    tol: float = 1e-8,
    sparse: bool = False,
    backend: str = "jax",
    batch_jac: bool | None = None,
    fd_eps: float = 1e-6,
    fd_mode: str = "central",
    fd_block_size: int = 1,
    supermesh_quad_cache: _SupermeshTriangleQuadratureCache | None = None,
):
    """
    Assemble mixed surface Jacobian over a supermesh (centroid quadrature).

    normal_source can be "master", "slave", "a", "b", or "avg"; use master_field
    to pick which field acts as the master when normal_source is "master"/"slave".
    dof_source="volume" assembles nodal fields into element nodes (requires elem_conn_* mappings).
    space_mode_* supports:
    - "nodal": existing nodal FE space behavior.
    - "p0": facet-wise constant space (one block of value_dim DOFs per facet, or
      custom mapping via facet_dofs_*).
    """
    source_facets_a = list(source_facets_a)
    source_facets_b = list(source_facets_b)
    from ..core.forms import FieldPair
    _contact_interface_dbg(
        f"[contact-interface] enter assemble_contact_interface_jacobian quad_order={quad_order} backend={backend}"
    )
    trace = os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE", "0") not in ("0", "", "false", "False")
    trace_max = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE_MAX", "5"))
    trace_every = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE_EVERY", "50"))
    trace_fd_max = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE_FD_MAX", "5"))
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
    n_a = _field_n_dofs(
        n_nodes=int(coords_a.shape[0]),
        n_facets=int(facets_a.shape[0]),
        value_dim=int(value_dim_a),
        space_mode=space_mode_a,
        facet_dofs=facet_dofs_a,
    )
    n_b = _field_n_dofs(
        n_nodes=int(coords_b.shape[0]),
        n_facets=int(facets_b.shape[0]),
        value_dim=int(value_dim_b),
        space_mode=space_mode_b,
        facet_dofs=facet_dofs_b,
    )
    if offset_b is None:
        offset_b = offset_a + n_a
    if trial_value_dim_a is None:
        trial_value_dim_a = value_dim_a
    if trial_value_dim_b is None:
        trial_value_dim_b = value_dim_b
    if trial_space_mode_a is None:
        trial_space_mode_a = space_mode_a
    if trial_space_mode_b is None:
        trial_space_mode_b = space_mode_b
    if trial_facet_dofs_a is None:
        trial_facet_dofs_a = facet_dofs_a
    if trial_facet_dofs_b is None:
        trial_facet_dofs_b = facet_dofs_b
    distinct_trial_layout = (
        int(trial_value_dim_a) != int(value_dim_a)
        or int(trial_value_dim_b) != int(value_dim_b)
        or str(trial_space_mode_a) != str(space_mode_a)
        or str(trial_space_mode_b) != str(space_mode_b)
        or not _same_optional_int_array(trial_facet_dofs_a, facet_dofs_a)
        or not _same_optional_int_array(trial_facet_dofs_b, facet_dofs_b)
    )
    n_total = int(offset_b + n_b)
    u_a_np = np.asarray(u_a, dtype=float)
    u_b_np = np.asarray(u_b, dtype=float)
    if trace:
        _trace("[CONTACT] assemble_contact_interface_jacobian ENTER")
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
        if hasattr(surface_a, "facet_areas"):
            facet_area_a = np.asarray(surface_a.facet_areas(), dtype=float)
        else:
            facet_area_a = np.array([_facet_area_estimate(fa, coords_a) for fa in facets_a], dtype=float)
        if hasattr(surface_b, "facet_areas"):
            facet_area_b = np.asarray(surface_b.facet_areas(), dtype=float)
        else:
            facet_area_b = np.array([_facet_area_estimate(fb, coords_b) for fb in facets_b], dtype=float)

    includes_measure = getattr(res_form, "_includes_measure", {})

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    K_dense: np.ndarray | None = np.zeros((n_total, n_total), dtype=float) if not sparse else None

    use_elem_a = elem_conn_a is not None and facet_to_elem_a is not None
    use_elem_b = elem_conn_b is not None and facet_to_elem_b is not None

    use_p0_a, use_p0_b = _validate_contact_interface_space_setup(
        grad_source=grad_source,
        dof_source=dof_source,
        space_mode_a=space_mode_a,
        space_mode_b=space_mode_b,
    )
    global _DEBUG_SURFACE_SOURCE_ONCE
    if grad_source == "surface" and not _DEBUG_SURFACE_SOURCE_ONCE:
        print("[fluxfem] using surface gradN in contact interface")
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
    if batch_jac is None:
        batch_jac = _env_flag("FLUXFEM_CONTACT_INTERFACE_BATCH_JAC", True)

    normal_source = _resolve_normal_source_option(
        normal_source=normal_source,
        normal_from=normal_from,
        master_field=master_field,
        field_a=field_a,
        field_b=field_b,
        diag_force=diag_force,
        diag_normal=diag_normal,
    )

    contact_interface_mode = os.getenv("FLUXFEM_CONTACT_INTERFACE_MODE", "supermesh").lower()
    _contact_interface_dbg(f"[contact-interface] mode={contact_interface_mode}")
    if contact_interface_mode == "projection" and not (use_p0_a or use_p0_b) and not distinct_trial_layout:
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
            _apply_projection_jacobian_batches(
                batches=batches,
                field_a=field_a,
                field_b=field_b,
                value_dim_a=value_dim_a,
                value_dim_b=value_dim_b,
                u_a=u_a_np,
                u_b=u_b_np,
                offset_a=int(offset_a),
                offset_b=int(offset_b),
                backend=backend,
                fd_eps=fd_eps,
                fd_mode=fd_mode,
                fd_block_size=fd_block_size,
                res_form=res_form,
                params=params,
                includes_measure=includes_measure,
                sparse=sparse,
                rows=rows,
                cols=cols,
                data=data,
                K_dense=K_dense,
            )
            if sparse:
                from ..solver import FluxSparseMatrix

                return FluxSparseMatrix(
                    np.asarray(rows, dtype=int),
                    np.asarray(cols, dtype=int),
                    np.asarray(data, dtype=float),
                    n_dofs=n_total,
                )
            assert K_dense is not None
            return K_dense

    batch_enabled = (
        batch_jac
        and backend == "jax"
        and dof_source == "volume"
        and grad_source == "volume"
        and use_elem_a
        and use_elem_b
        and not use_p0_a
        and not use_p0_b
        and not distinct_trial_layout
        and not proj_diag
        and not diag_force
    )
    if trace and batch_jac and not batch_enabled:
        reasons = []
        if backend != "jax":
            reasons.append(f"backend={backend}")
        if dof_source != "volume":
            reasons.append(f"dof_source={dof_source}")
        if grad_source != "volume":
            reasons.append(f"grad_source={grad_source}")
        if not use_elem_a:
            reasons.append("missing_elem_a")
        if not use_elem_b:
            reasons.append("missing_elem_b")
        if use_p0_a:
            reasons.append("space_mode_a=p0")
        if use_p0_b:
            reasons.append("space_mode_b=p0")
        if proj_diag:
            reasons.append("projection_diag")
        if diag_force:
            reasons.append("diag_force")
        _trace(f"[CONTACT] batch_jac_disabled {' '.join(reasons)}")

    if batch_enabled:
        if trace:
            _trace("[CONTACT] batch_jac_enter")
        batch_items = []
        dofs_batch = []
        u_local_batch = []
        batch_rows: list[np.ndarray] = []
        batch_cols: list[np.ndarray] = []
        batch_data: list[np.ndarray] = []
        batch_size = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_BATCH_SIZE", "128"))
        if batch_size <= 0:
            batch_size = 0
        n_q = None
        n_nodes_a = None
        n_nodes_b = None
        n_a_local_const = None
        n_b_local_const = None
        batch_failed = False
        jit_batch = _env_flag("FLUXFEM_CONTACT_INTERFACE_BATCH_JIT", False)
        direct_jit_batch = _env_flag("FLUXFEM_CONTACT_INTERFACE_DIRECT_BATCH_JIT", True)
        formulation_tag = getattr(res_form, "_ff_contact_formulation", None)
        fastpath_tag = getattr(res_form, "_ff_contact_backend_fastpath", None)
        has_pair_nitsche_params = all(hasattr(params, name) for name in ("alpha", "inv_h", "lam", "mu", "use_penalty", "use_traction"))
        use_direct_pair_nitsche_batch = (
            formulation_tag == "pair_nitsche_penalty"
            and fastpath_tag == "numpy_local_kernel"
            and has_pair_nitsche_params
            and int(value_dim_a) == 3
            and int(value_dim_b) == 3
        )

        def _make_jac_fun(n_a_local: int, n_b_local: int):
            test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = (
                _mixed_surface_space_aliases(
                    res_form,
                    field_a=field_a,
                    field_b=field_b,
                )
            )

            def _res_local_batch(u_vec, Na, Nb, gradNa, gradNb, x_q, w, detJ, normal):
                fields = {
                    field_a: _make_surface_field_pair(
                        test_N=Na,
                        test_gradN=gradNa,
                        trial_N=Na,
                        trial_gradN=gradNa,
                        test_value_dim=value_dim_a,
                        trial_value_dim=value_dim_a,
                    ),
                    field_b: _make_surface_field_pair(
                        test_N=Nb,
                        test_gradN=gradNb,
                        trial_N=Nb,
                        trial_gradN=gradNb,
                        test_value_dim=value_dim_b,
                        trial_value_dim=value_dim_b,
                    ),
                }
                spaces = dict(fields)
                for key in (test_space_key_a, unknown_space_key_a):
                    if key is not None:
                        spaces[key] = fields[field_a]
                for key in (test_space_key_b, unknown_space_key_b):
                    if key is not None:
                        spaces[key] = fields[field_b]
                normal_q = jnp.repeat(normal[None, :], x_q.shape[0], axis=0)
                ctx = SurfaceMixedFormContext(
                    bindings=fields,
                    x_q=x_q,
                    w=w,
                    detJ=detJ,
                    normal=normal_q,
                    spaces=spaces,
                )
                u_dict = {
                    field_a: u_vec[:n_a_local],
                    field_b: u_vec[n_a_local:],
                }
                if unknown_space_key_a is not None:
                    u_dict[unknown_space_key_a] = u_dict[field_a]
                if unknown_space_key_b is not None:
                    u_dict[unknown_space_key_b] = u_dict[field_b]
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
        direct_batch_fun = None

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
            if jit_batch and batch_size and batch_n < batch_size:
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
            t_batch = time.perf_counter()
            if use_direct_pair_nitsche_batch:
                nonlocal direct_batch_fun
                if direct_batch_fun is None:
                    if trace:
                        _trace(f"[CONTACT] batch_direct_build jit={direct_jit_batch}")
                    direct_batch_fun = _get_direct_pair_nitsche_batch_fun(jit=direct_jit_batch)
                J_b = direct_batch_fun(
                    Na_b,
                    Nb_b,
                    gradNa_b,
                    gradNb_b,
                    w_b,
                    detJ_b,
                    normal_b,
                    float(getattr(params, "alpha")),
                    float(getattr(params, "inv_h")),
                    float(getattr(params, "lam")),
                    float(getattr(params, "mu")),
                    float(getattr(params, "use_penalty", 1.0)),
                    float(getattr(params, "use_traction", 1.0)),
                )
            else:
                key = (n_a_local, n_b_local)
                jac_fun = jac_fun_cache.get(key)
                if jac_fun is None:
                    jac_fun = _make_jac_fun(n_a_local, n_b_local)
                    jac_fun_cache[key] = jac_fun
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
                # rows/cols contain repeated global DOF pairs across triangles in the batch.
                # Advanced indexing with += does not accumulate repeated indices reliably.
                np.add.at(K_dense, (rows, cols), data)
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
                if trace:
                    _trace(f"[CONTACT] batch_jac_abort normal_none tri={int(len(dofs_batch))}")
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
                    if trace:
                        _trace(
                            "[CONTACT] batch_jac_shape_mismatch "
                            f"nq={Na.shape[0]}/{n_q} "
                            f"na={Na.shape[1]}/{n_nodes_a} "
                            f"nb={Nb.shape[1]}/{n_nodes_b} "
                            f"da={dofs_a.shape[0]}/{n_a_local_const} "
                            f"db={dofs_b.shape[0]}/{n_b_local_const}"
                        )
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
                from ..solver import FluxSparseMatrix

                return FluxSparseMatrix(rows_np, cols_np, data_np, n_dofs=n_total)
            assert K_dense is not None
            return K_dense

        if trace:
            _trace("[CONTACT] batch_jac_fallback")

    pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=use_p0_a,
        use_p0_b=use_p0_b,
    )
    trial_use_p0_a = str(trial_space_mode_a) == "p0"
    trial_use_p0_b = str(trial_space_mode_b) == "p0"
    trial_pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=trial_use_p0_a,
        use_p0_b=trial_use_p0_b,
    )
    if trace:
        _trace("[CONTACT] supermesh_loop_enter")
    _contact_interface_dbg("[contact-interface] step: supermesh loop START")
    t_loop = time.perf_counter()
    use_cached_geom = supermesh_quad_cache is not None and not diag_force
    if use_cached_geom:
        assert supermesh_quad_cache is not None
        quad_pts_cached = np.asarray(supermesh_quad_cache.quad_pts, dtype=float)
        quad_w_cached = np.asarray(supermesh_quad_cache.quad_w, dtype=float)
        detJ_cached = np.asarray(supermesh_quad_cache.detJ, dtype=float)
        x_q_cached = np.asarray(supermesh_quad_cache.x_q, dtype=float)
        tri_iter = enumerate(zip(source_facets_a, source_facets_b))
    else:
        tri_iter = enumerate(
            zip(
                _iter_supermesh_tris(supermesh_coords, supermesh_conn),
                source_facets_a,
                source_facets_b,
            )
        )
    for it, tri_item in tri_iter:
        log_tri = trace and (it < trace_max or it % trace_every == 0)
        t_tri0 = time.perf_counter()
        def _tri_check(stage: str) -> None:
            if tri_timeout > 0.0 and (time.perf_counter() - t_tri0) > tri_timeout:
                raise RuntimeError(f"[CONTACT] tri {it} timeout at {stage}")
        if use_cached_geom:
            fa, fb = tri_item
            if log_tri:
                _trace(f"[CONTACT] tri {it} start fa={int(fa)} fb={int(fb)} [cached-geom]")
            detJ = float(detJ_cached[it])
            if detJ <= 2.0 * tol:
                continue
            if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
                area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
                if area_ref > 0.0 and 0.5 * detJ < area_scale * area_ref:
                    continue
            if guard:
                if not np.isfinite(detJ):
                    if skip_nonfinite:
                        continue
                    raise RuntimeError(f"[CONTACT] tri {it} detJ non-finite")
                if detj_eps > 0.0 and abs(detJ) < detj_eps:
                    if log_tri:
                        _trace(f"[CONTACT] tri {it} detJ too small {detJ:.3e}; skip")
                    continue
                if not np.isfinite(x_q_cached[it]).all():
                    if skip_nonfinite:
                        continue
                    raise RuntimeError(f"[CONTACT] tri {it} x_q non-finite")
            geom = _JacobianTriangleGeometryData(
                detJ=abs(detJ) if diag_abs_detj else detJ,
                quad_pts=quad_pts_cached,
                quad_w=quad_w_cached,
                quad_source="fluxfem-cache",
                facet_a=facets_a[int(fa)],
                facet_b=facets_b[int(fb)],
                x_q=x_q_cached[it],
            )
        else:
            (tri, a, b, c), fa, fb = tri_item
            if log_tri:
                _trace(f"[CONTACT] tri {it} start fa={int(fa)} fb={int(fb)}")
            geom = _prepare_supermesh_jacobian_triangle_geometry(
                it=it,
                log_tri=log_tri,
                a=a,
                b=b,
                c=c,
                fa=int(fa),
                fb=int(fb),
                tol=tol,
                skip_small_tri=skip_small_tri,
                area_scale=area_scale,
                facet_area_a=facet_area_a,
                facet_area_b=facet_area_b,
                diag_force=diag_force,
                diag_abs_detj=diag_abs_detj,
                guard=guard,
                skip_nonfinite=skip_nonfinite,
                detj_eps=detj_eps,
                quad_order=quad_order,
                diag_qp_mode=diag_qp_mode,
                diag_qp_path=diag_qp_path,
                facets_a=facets_a,
                facets_b=facets_b,
                tri_check=_tri_check,
                trace_fn=_trace,
                trace_time_fn=_trace_time,
            )
        if geom is None:
            continue

        _accumulate_supermesh_jacobian_triangle_core(
            it=it,
            log_tri=log_tri,
            fa=int(fa),
            fb=int(fb),
            facet_a=geom.facet_a,
            facet_b=geom.facet_b,
            x_q=geom.x_q,
            quad_pts=geom.quad_pts,
            quad_w=geom.quad_w,
            quad_source=geom.quad_source,
            detJ=geom.detJ,
            tol=tol,
            pair_basis_builder=pair_basis_builder,
            trial_pair_basis_builder=trial_pair_basis_builder,
            coords_a=coords_a,
            coords_b=coords_b,
            facets_a=facets_a,
            facets_b=facets_b,
            value_dim_a=value_dim_a,
            value_dim_b=value_dim_b,
            trial_value_dim_a=int(trial_value_dim_a),
            trial_value_dim_b=int(trial_value_dim_b),
            dof_source=dof_source,
            grad_source=grad_source,
            space_mode_a=space_mode_a,
            space_mode_b=space_mode_b,
            trial_space_mode_a=str(trial_space_mode_a),
            trial_space_mode_b=str(trial_space_mode_b),
            use_elem_a=use_elem_a,
            use_elem_b=use_elem_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            facet_dofs_a=facet_dofs_a,
            facet_dofs_b=facet_dofs_b,
            trial_facet_dofs_a=trial_facet_dofs_a,
            trial_facet_dofs_b=trial_facet_dofs_b,
            proj_diag=proj_diag,
            diag_map=diag_map,
            diag_n=diag_n,
            diag_force=diag_force,
            diag_facet=diag_facet,
            diag_max_q=diag_max_q,
            guard=guard,
            skip_nonfinite=skip_nonfinite,
            normal_source=normal_source,
            normal_sign=normal_sign,
            normals_a=normals_a,
            normals_b=normals_b,
            field_a=field_a,
            field_b=field_b,
            u_a=u_a_np,
            u_b=u_b_np,
            res_form=res_form,
            params=params,
            includes_measure=includes_measure,
            offset_a=int(offset_a),
            offset_b=int(offset_b),
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            sparse=sparse,
            rows=rows,
            cols=cols,
            data=data,
            K_dense=K_dense,
            tri_check=_tri_check,
            trace_fn=_trace,
            trace_time_fn=_trace_time,
        )


    if proj_diag:
        _proj_diag_report()
    if sparse:
        from ..solver import FluxSparseMatrix

        return FluxSparseMatrix(
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
            np.asarray(data, dtype=float),
            n_dofs=n_total,
        )
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

    disable_consistency = os.getenv("FF_ONESIDED_DISABLE_CONSISTENCY", "").strip().lower() in {"1", "true", "yes", "on"}
    disable_symmetry = os.getenv("FF_ONESIDED_DISABLE_SYMMETRY", "").strip().lower() in {"1", "true", "yes", "on"}
    disable_penalty = os.getenv("FF_ONESIDED_DISABLE_PENALTY", "").strip().lower() in {"1", "true", "yes", "on"}

    expr = 0.0
    if not disable_consistency:
        expr = expr - h_wf.dot(v, t_u)
    if not disable_symmetry:
        expr = expr - sym_term + sym_term_hat
    if not disable_penalty:
        expr = expr + (p.alpha * p.inv_h) * h_wf.dot(v, u.val) - (p.alpha * p.inv_h) * h_wf.dot(v, p.u_hat)
    expr = expr * h_wf.ds()
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
                bindings=fields,
                x_q=x_q,
                w=quad_w,
                detJ=np.array([detJ], dtype=float),
                normal=normal_q,
                spaces=fields,
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

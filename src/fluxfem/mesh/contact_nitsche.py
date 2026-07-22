from __future__ import annotations

from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp

_FF_CONTACT_FORMULATION_ATTR = "_ff_contact_formulation"
_FF_CONTACT_FASTPATH_ATTR = "_ff_contact_backend_fastpath"


_DIRECT_PAIR_NITSCHE_BATCH_FUN_CACHE: dict[bool, Callable[..., jnp.ndarray]] = {}


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


def _tag_pair_nitsche_penalty_bilinear(
    fn: Callable[..., Any],
    *,
    backend_fastpath: str = "numpy_local_kernel",
) -> Callable[..., Any]:
    setattr(fn, _FF_CONTACT_FORMULATION_ATTR, "pair_nitsche_penalty")
    if backend_fastpath is not None:
        setattr(fn, _FF_CONTACT_FASTPATH_ATTR, str(backend_fastpath))
    return fn


def make_pair_nitsche_supermesh_bilinear(
    *,
    backend_fastpath: str = "numpy_local_kernel",
) -> Callable[..., Any]:
    """Build the symmetric pair-Nitsche bilinear used on contact supermeshes."""
    import fluxfem.helpers_wf as h_wf
    from ..core.weakform import einsum as wf_einsum

    def _bilin(v1, v2, u1, u2, p):
        n = h_wf.normal()
        ju = u1.val - u2.val
        t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
        t_v1 = h_wf.traction(v1, n, p)
        t_v2 = h_wf.traction(v2, n, p)
        penalty = p.use_penalty * (p.alpha * p.inv_h) * (
            h_wf.dot(v1, ju) - h_wf.dot(v2, ju)
        )
        traction = p.use_traction * (-h_wf.dot(v1, t_u) + h_wf.dot(v2, t_u))
        traction -= p.use_traction * 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
        traction -= p.use_traction * 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
        return (penalty + traction) * h_wf.ds()

    return _tag_pair_nitsche_penalty_bilinear(
        _bilin,
        backend_fastpath=backend_fastpath,
    )


def params_with_pair_nitsche_defaults(
    params: Any,
    *,
    use_penalty: float | None,
    use_traction: float | None,
) -> Any:
    defaults = {
        "use_penalty": 1.0 if use_penalty is None else float(use_penalty),
        "use_traction": 1.0 if use_traction is None else float(use_traction),
    }
    data = dict(getattr(params, "_data", {}))
    if not data:
        data = dict(vars(params))
    changed = False
    for name, value in defaults.items():
        if name not in data or (name == "use_penalty" and use_penalty is not None) or (
            name == "use_traction" and use_traction is not None
        ):
            data[name] = value
            changed = True
    if not changed:
        return params
    from ..core.weakform import Params

    return Params(**data)


def assemble_pair_nitsche_supermesh_impl(
    contact,
    params: Any,
    *,
    contribution_cls: type,
    sparse: bool = False,
    normal_source: str = "master",
    use_penalty: float | None = None,
    use_traction: float | None = None,
    backend_fastpath: str = "numpy_local_kernel",
):
    """
    Assemble pair-Nitsche contact terms over a prepared contact supermesh.

    The contact object must provide ``assemble_bilinear_form``; prepared
    ``ContactSurfaceSpace`` and ``OneToManyContactSurfaceSpace`` objects do.
    """
    if not hasattr(contact, "assemble_bilinear_form"):
        raise TypeError("contact must provide assemble_bilinear_form() for pair-Nitsche supermesh assembly.")
    params_eff = params_with_pair_nitsche_defaults(
        params,
        use_penalty=use_penalty,
        use_traction=use_traction,
    )
    bilin = make_pair_nitsche_supermesh_bilinear(backend_fastpath=backend_fastpath)
    jacobian = contact.assemble_bilinear_form(
        bilin,
        params_eff,
        sparse=sparse,
        normal_source=normal_source,
    )
    diagnostics: dict[str, Any] = {}
    if hasattr(contact, "supermesh_conn"):
        diagnostics["supermesh_triangles"] = int(np.asarray(contact.supermesh_conn).shape[0])
    diagnostics["use_penalty"] = float(getattr(params_eff, "use_penalty", 1.0))
    diagnostics["use_traction"] = float(getattr(params_eff, "use_traction", 1.0))
    return contribution_cls(
        enforcement="nitsche",
        law="frictionless_tied",
        formulation="pair_nitsche_penalty",
        jacobian=jacobian,
        diagnostics=diagnostics,
    )

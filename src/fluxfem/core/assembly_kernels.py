from __future__ import annotations

from typing import Any, Literal, Mapping, cast

import jax
import jax.numpy as jnp

from .forms import FormContext


def _integrate_q_linear(integrand: jnp.ndarray, wJ: jnp.ndarray, *, includes_measure: bool) -> jnp.ndarray:
    if includes_measure:
        return jnp.einsum("qa->a", integrand)
    return jnp.einsum("qa,q->a", integrand, wJ)


def _integrate_q_bilinear(integrand: jnp.ndarray, wJ: jnp.ndarray, *, includes_measure: bool) -> jnp.ndarray:
    if includes_measure:
        return jnp.einsum("qab->ab", integrand)
    return jnp.einsum("qab,q->ab", integrand, wJ)


def _integrate_q_tree(integrand: Any, wJ: jnp.ndarray, *, includes_measure: bool) -> Any:
    if includes_measure:
        return jax.tree_util.tree_map(lambda x: jnp.einsum("qa->a", x), integrand)
    return jax.tree_util.tree_map(lambda x: jnp.einsum("qa,q->a", x, wJ), integrand)


def _integrate_q_named_fields(
    integrand: Mapping[str, jnp.ndarray],
    ctx: FormContext,
    includes_measure: Any,
) -> dict[str, jnp.ndarray]:
    out: dict[str, jnp.ndarray] = {}
    for name, val in integrand.items():
        use_measure = bool(isinstance(includes_measure, dict) and includes_measure.get(name, False))
        if use_measure:
            out[name] = jnp.einsum("qa->a", val)
        else:
            wJ = ctx.w * ctx.bindings[name].test.detJ
            out[name] = jnp.einsum("qa,q->a", val, wJ)
    return out


def make_element_bilinear_kernel(form, params, *, jit: bool = True):
    """Element kernel: (ctx) -> Ke."""

    def per_element(ctx: FormContext):
        integrand = form(ctx, params)
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_bilinear(
            integrand,
            wJ,
            includes_measure=bool(getattr(form, "_includes_measure", False)),
        )

    return jax.jit(per_element) if jit else per_element


def make_element_linear_kernel(form, params, *, jit: bool = True):
    """Element kernel: (ctx) -> fe."""

    def per_element(ctx: FormContext):
        integrand = form(ctx, params)
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(form, "_includes_measure", False)),
        )

    return jax.jit(per_element) if jit else per_element


def make_element_residual_kernel(res_form, params):
    """Jitted element residual kernel: (ctx, u_elem) -> fe."""

    def per_element(ctx: FormContext, u_elem: jnp.ndarray):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

    return jax.jit(per_element)


def make_element_jacobian_kernel(res_form, params):
    """Jitted element Jacobian kernel: (ctx, u_elem) -> Ke."""

    def fe_fun(u_elem, ctx: FormContext):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

    return jax.jit(jax.jacrev(fe_fun, argnums=0))


def element_residual(res_form, ctx: FormContext, u_elem, params):
    """
    Element residual vector r_e(u_e) = sum_q w_q * detJ_q * res_form(ctx, u_e, params).
    Returns shape (n_ldofs,) or pytree of same structure.
    """
    integrand = res_form(ctx, u_elem, params)
    includes_measure = getattr(res_form, "_includes_measure", False)
    if isinstance(integrand, jnp.ndarray):
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_linear(integrand, wJ, includes_measure=bool(includes_measure))
    if hasattr(ctx, "bindings") and ctx.bindings is not None:
        return _integrate_q_named_fields(cast(Mapping[str, jnp.ndarray], integrand), ctx, includes_measure)
    return _integrate_q_tree(
        integrand,
        ctx.w * ctx.test.detJ,
        includes_measure=bool(includes_measure),
    )


def element_jacobian(res_form, ctx: FormContext, u_elem, params):
    """
    Element Jacobian K_e = d r_e / d u_e (AD via jacfwd), shape (n_ldofs, n_ldofs).
    """

    def _r_elem(u_local):
        return element_residual(res_form, ctx, u_local, params)

    return jax.jacfwd(_r_elem)(u_elem)


def make_element_kernel(
    form,
    params,
    *,
    kind: Literal["bilinear", "linear", "residual", "jacobian"],
    jit: bool = True,
):
    """
    Unified entry point for element kernels.
    """
    kind = cast(Literal["bilinear", "linear", "residual", "jacobian"], kind.lower())
    if kind == "bilinear":
        return make_element_bilinear_kernel(form, params, jit=jit)
    if kind == "linear":
        return make_element_linear_kernel(form, params, jit=jit)
    if kind == "residual":
        return make_element_residual_kernel(form, params)
    if kind == "jacobian":
        return make_element_jacobian_kernel(form, params)
    raise ValueError(f"Unknown kernel kind: {kind}")

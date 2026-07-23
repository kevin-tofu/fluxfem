from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..core.forms import FormFieldLike

if TYPE_CHECKING:
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

    bindings: dict[str, "FieldPair"]
    x_q: np.ndarray
    w: np.ndarray
    detJ: np.ndarray
    normal: np.ndarray | None = None
    spaces: dict[str, "FieldPair"] | None = None


def make_surface_field_pair(
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


def build_mixed_surface_context(
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
        field_a: make_surface_field_pair(
            test_N=test_Na,
            test_gradN=test_gradNa,
            trial_N=trial_Na,
            trial_gradN=trial_gradNa,
            test_value_dim=test_value_dim_a,
            trial_value_dim=trial_value_dim_a,
        ),
        field_b: make_surface_field_pair(
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


def surface_u_elem_with_space_aliases(
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


def surface_local_u_dict(
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


def mixed_surface_space_aliases(
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


def reduce_surface_residual_jax(
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


def reduce_surface_residual_numpy(
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


def mixed_surface_local_residual_jax(
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
    u_dict = surface_local_u_dict(
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
        fe = reduce_surface_residual_jax(
            fe_field,
            includes_measure=bool(includes_measure.get(name, False)),
            w=ctx.w,
            detJ=ctx.detJ,
        )
        res_parts.append(fe)
    return jnp.concatenate(res_parts, axis=0)


def mixed_surface_local_residual_numpy(
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
    u_dict = surface_local_u_dict(
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
        fe = reduce_surface_residual_numpy(
            fe_field,
            includes_measure=bool(includes_measure.get(name, False)),
            w=ctx.w,
            detJ=ctx.detJ,
        )
        res_parts.append(np.asarray(fe))
    return np.concatenate(res_parts, axis=0)


def compute_mixed_surface_local_jacobian(
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
    _ = (fd_eps, fd_mode, fd_block_size)
    if backend == "jax":

        def _res_local(u_vec):
            return mixed_surface_local_residual_jax(
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
        J[:, j] = mixed_surface_local_residual_numpy(
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


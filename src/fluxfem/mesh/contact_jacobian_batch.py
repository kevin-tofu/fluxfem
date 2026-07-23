from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .contact_mixed_surface import (
    SurfaceMixedFormContext,
    make_surface_field_pair,
    mixed_surface_space_aliases,
)


@dataclass(frozen=True)
class ContactBatchJacobianGate:
    enabled: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContactBatchJacobianStack:
    Na: Any
    Nb: Any
    gradNa: Any
    gradNb: Any
    x_q: Any
    w: Any
    detJ: Any
    normal: Any
    u_local: Any
    dofs: np.ndarray
    n_a_local: int
    n_b_local: int
    batch_n: int


def contact_batch_jacobian_gate(
    *,
    requested: bool,
    backend: str,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    use_p0_a: bool,
    use_p0_b: bool,
    distinct_trial_layout: bool,
    proj_diag: bool,
    diag_force: bool,
) -> ContactBatchJacobianGate:
    reasons: list[str] = []
    if not requested:
        reasons.append("not_requested")
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
    if distinct_trial_layout:
        reasons.append("distinct_trial_layout")
    if proj_diag:
        reasons.append("projection_diag")
    if diag_force:
        reasons.append("diag_force")
    return ContactBatchJacobianGate(enabled=not reasons, reasons=tuple(reasons))


def stack_contact_batch_items(
    batch_items: Sequence[tuple[Any, Any, Any, Any, Any, Any, Any, Any]],
    dofs_batch: Sequence[np.ndarray],
    u_local_batch: Sequence[np.ndarray],
    *,
    n_a_local: int,
    n_b_local: int,
) -> ContactBatchJacobianStack:
    Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
    return ContactBatchJacobianStack(
        Na=jnp.asarray(np.stack(Na_b, axis=0)),
        Nb=jnp.asarray(np.stack(Nb_b, axis=0)),
        gradNa=jnp.asarray(np.stack(gradNa_b, axis=0)),
        gradNb=jnp.asarray(np.stack(gradNb_b, axis=0)),
        x_q=jnp.asarray(np.stack(x_q_b, axis=0)),
        w=jnp.asarray(np.stack(w_b, axis=0)),
        detJ=jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1),
        normal=jnp.asarray(np.stack(normal_b, axis=0)),
        u_local=jnp.asarray(np.stack(u_local_batch, axis=0)),
        dofs=np.asarray(dofs_batch, dtype=int),
        n_a_local=int(n_a_local),
        n_b_local=int(n_b_local),
        batch_n=int(len(batch_items)),
    )


def make_contact_batch_jacobian_function(
    *,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    field_a: str,
    field_b: str,
    value_dim_a: int,
    value_dim_b: int,
    n_a_local: int,
    n_b_local: int,
    jit: bool,
) -> Callable[..., Any]:
    test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = (
        mixed_surface_space_aliases(
            res_form,
            field_a=field_a,
            field_b=field_b,
        )
    )

    def _res_local_batch(u_vec, Na, Nb, gradNa, gradNb, x_q, w, detJ, normal):
        fields = {
            field_a: make_surface_field_pair(
                test_N=Na,
                test_gradN=gradNa,
                trial_N=Na,
                trial_gradN=gradNa,
                test_value_dim=value_dim_a,
                trial_value_dim=value_dim_a,
            ),
            field_b: make_surface_field_pair(
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

    jac_fun = jax.vmap(jax.jacrev(_res_local_batch))
    return jax.jit(jac_fun) if jit else jac_fun


def pad_contact_batch_for_jit(stack: ContactBatchJacobianStack, *, target_size: int) -> ContactBatchJacobianStack:
    pad = int(target_size) - int(stack.batch_n)
    if pad <= 0:
        return stack

    def _pad_batch(x, pad_value: float = 0.0):
        x_arr = jnp.asarray(x)
        pad_width = [(0, pad)] + [(0, 0)] * (x_arr.ndim - 1)
        return jnp.pad(x_arr, pad_width, mode="constant", constant_values=pad_value)

    return ContactBatchJacobianStack(
        Na=_pad_batch(stack.Na),
        Nb=_pad_batch(stack.Nb),
        gradNa=_pad_batch(stack.gradNa),
        gradNb=_pad_batch(stack.gradNb),
        x_q=_pad_batch(stack.x_q),
        w=_pad_batch(stack.w),
        detJ=_pad_batch(stack.detJ),
        normal=_pad_batch(stack.normal),
        u_local=_pad_batch(stack.u_local),
        dofs=stack.dofs,
        n_a_local=stack.n_a_local,
        n_b_local=stack.n_b_local,
        batch_n=stack.batch_n,
    )


def contact_batch_dof_pairs(dofs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_ldofs = int(dofs_batch.shape[1])
    rows = np.repeat(dofs_batch, n_ldofs, axis=1).reshape(-1)
    cols = np.tile(dofs_batch, (1, n_ldofs)).reshape(-1)
    return rows, cols

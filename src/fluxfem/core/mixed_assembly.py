from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp

from .dtypes import INDEX_DTYPE
from .assembly import element_residual, make_sparsity_pattern


def _coerce_mixed_u(space, u):
    if isinstance(u, Mapping):
        return space.pack_fields(u)
    return jnp.asarray(u)


def _split_elem_vec(field_names, elem_slices, u_elem_vec):
    return {name: u_elem_vec[elem_slices[name]] for name in field_names}


def _concat_residuals(field_names, res_dict):
    return jnp.concatenate([res_dict[name] for name in field_names], axis=0)


def make_element_mixed_residual_kernel(res_form, params, field_names, elem_slices):
    """Jitted element residual kernel for mixed systems."""

    def per_element(ctx, u_elem_vec):
        u_elem = _split_elem_vec(field_names, elem_slices, u_elem_vec)
        res_dict = element_residual(res_form, ctx, u_elem, params)
        return _concat_residuals(field_names, res_dict)

    return jax.jit(per_element)


def make_element_mixed_jacobian_kernel(res_form, params, field_names, elem_slices):
    """Jitted element Jacobian kernel for mixed systems."""
    res_kernel = make_element_mixed_residual_kernel(res_form, params, field_names, elem_slices)

    def fe_fun(u_elem_vec, ctx):
        return res_kernel(ctx, u_elem_vec)

    return jax.jit(jax.jacrev(fe_fun, argnums=0))


def assemble_mixed_residual_scatter(space, res_form, u, params, *, sparse: bool = False, kernel=None):
    """Assemble mixed residual using jitted element kernels + scatter_add."""
    u_vec = _coerce_mixed_u(space, u)
    ctxs = space.build_form_contexts()
    ker = kernel if kernel is not None else make_element_mixed_residual_kernel(
        res_form, params, space.field_names, space.elem_slices
    )

    u_elems = u_vec[space.elem_dofs]
    elem_res = jax.vmap(ker)(ctxs, u_elems)
    rows = space.elem_dofs.reshape(-1)
    data = elem_res.reshape(-1)

    if sparse:
        return rows, data, space.n_dofs

    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    F = jnp.zeros((space.n_dofs,), dtype=data.dtype)
    F = jax.lax.scatter_add(F, rows[:, None], data, sdn)
    return F


def assemble_mixed_jacobian_values(space, res_form, u, params, *, kernel=None):
    """Assemble numeric values for mixed Jacobian (pattern-free)."""
    u_vec = _coerce_mixed_u(space, u)
    ctxs = space.build_form_contexts()
    ker = kernel if kernel is not None else make_element_mixed_jacobian_kernel(
        res_form, params, space.field_names, space.elem_slices
    )

    u_elems = u_vec[space.elem_dofs]
    J_e_all = jax.vmap(ker)(u_elems, ctxs)
    return J_e_all.reshape(-1)


def assemble_mixed_jacobian_scatter(
    space,
    res_form,
    u,
    params,
    *,
    kernel=None,
    sparse: bool = True,
    return_flux_matrix: bool = False,
    pattern=None,
):
    """Assemble mixed Jacobian using jitted element kernels + scatter_add."""
    from ..solver import FluxSparseMatrix  # local import to avoid circular

    pat = pattern if pattern is not None else make_sparsity_pattern(space, with_idx=not sparse)
    data = assemble_mixed_jacobian_values(space, res_form, u, params, kernel=kernel)

    if sparse:
        if return_flux_matrix:
            return FluxSparseMatrix(pat, data)
        return pat.rows, pat.cols, data, pat.n_dofs

    idx = pat.idx
    if idx is None:
        idx = (pat.rows.astype(jnp.int64) * int(pat.n_dofs) + pat.cols.astype(jnp.int64)).astype(INDEX_DTYPE)

    n_entries = pat.n_dofs * pat.n_dofs
    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    K_flat = jnp.zeros(n_entries, dtype=data.dtype)
    K_flat = jax.lax.scatter_add(K_flat, idx[:, None], data, sdn)
    return K_flat.reshape(pat.n_dofs, pat.n_dofs)


def assemble_mixed_residual(space, res_form, u, params, *, sparse: bool = False):
    """Assemble the global mixed residual vector."""
    return assemble_mixed_residual_scatter(space, res_form, u, params, sparse=sparse)


def assemble_mixed_jacobian(
    space,
    res_form,
    u,
    params,
    *,
    sparse: bool = True,
    return_flux_matrix: bool = False,
    pattern=None,
):
    """Assemble the global mixed Jacobian."""
    return assemble_mixed_jacobian_scatter(
        space,
        res_form,
        u,
        params,
        sparse=sparse,
        return_flux_matrix=return_flux_matrix,
        pattern=pattern,
    )


__all__ = [
    "make_element_mixed_residual_kernel",
    "make_element_mixed_jacobian_kernel",
    "assemble_mixed_residual",
    "assemble_mixed_jacobian",
    "assemble_mixed_residual_scatter",
    "assemble_mixed_jacobian_scatter",
    "assemble_mixed_jacobian_values",
]

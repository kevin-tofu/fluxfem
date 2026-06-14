from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp

from .assembly import element_residual, make_sparsity_pattern, chunk_pad_stats, _maybe_trace_pad


def _coerce_mixed_u(space, u):
    if isinstance(u, Mapping):
        return space.pack_fields(u)
    return jnp.asarray(u)


def _coerce_mixed_role_u(space, u):
    if isinstance(u, Mapping):
        return space.pack_unknown_fields(u)
    return jnp.asarray(u)


def _ensure_mixed_role_volume_form(res_form, *, op: str) -> None:
    domain = getattr(res_form, "_ff_domain", None)
    if domain == "surface":
        raise NotImplementedError(
            f"MixedRoleFESpace {op} is currently volume-only; "
            "compiled mixed surface residuals/contact facet assembly are not supported yet."
        )


def _split_elem_vec(field_names, elem_slices, space_key_by_field, alias_space_keys_by_field, u_elem_vec):
    split = {name: u_elem_vec[elem_slices[name]] for name in field_names}
    for name in field_names:
        key = space_key_by_field.get(name, name)
        split.setdefault(key, split[name])
        for alias in alias_space_keys_by_field.get(name, ()):
            split.setdefault(alias, split[name])
    return split


def _concat_residuals(field_names, res_dict):
    return jnp.concatenate([res_dict[name] for name in field_names], axis=0)


def make_element_mixed_residual_kernel(
    res_form,
    params,
    field_names,
    elem_slices,
    space_key_by_field,
    alias_space_keys_by_field,
):
    """Jitted element residual kernel for mixed systems."""

    def per_element(ctx, u_elem_vec):
        u_elem = _split_elem_vec(
            field_names,
            elem_slices,
            space_key_by_field,
            alias_space_keys_by_field,
            u_elem_vec,
        )
        res_dict = element_residual(res_form, ctx, u_elem, params)
        return _concat_residuals(field_names, res_dict)

    return jax.jit(per_element)


def make_element_mixed_jacobian_kernel(
    res_form,
    params,
    field_names,
    elem_slices,
    space_key_by_field,
    alias_space_keys_by_field,
):
    """Jitted element Jacobian kernel for mixed systems."""
    res_kernel = make_element_mixed_residual_kernel(
        res_form,
        params,
        field_names,
        elem_slices,
        space_key_by_field,
        alias_space_keys_by_field,
    )

    def fe_fun(u_elem_vec, ctx):
        return res_kernel(ctx, u_elem_vec)

    return jax.jit(jax.jacrev(fe_fun, argnums=0))


def make_element_mixed_role_residual_kernel(
    res_form,
    params,
    field_names,
    unknown_elem_slices,
    role_space_keys_by_field,
):
    """Jitted element residual kernel for role-aware mixed systems."""

    def per_element(ctx, u_elem_vec):
        u_elem = {name: u_elem_vec[unknown_elem_slices[name]] for name in field_names}
        for name in field_names:
            for key in role_space_keys_by_field.get(name, (name,)):
                u_elem.setdefault(key, u_elem[name])
        res_dict = element_residual(res_form, ctx, u_elem, params)
        return _concat_residuals(field_names, res_dict)

    return jax.jit(per_element)


def make_element_mixed_role_jacobian_kernel(
    res_form,
    params,
    field_names,
    unknown_elem_slices,
    role_space_keys_by_field,
):
    """Jitted element Jacobian kernel for role-aware mixed systems."""
    res_kernel = make_element_mixed_role_residual_kernel(
        res_form,
        params,
        field_names,
        unknown_elem_slices,
        role_space_keys_by_field,
    )

    def fe_fun(u_elem_vec, ctx):
        return res_kernel(ctx, u_elem_vec)

    return jax.jit(jax.jacrev(fe_fun, argnums=0))


def assemble_mixed_residual_scatter(
    space,
    res_form,
    u,
    params,
    *,
    sparse: bool = False,
    kernel=None,
    n_chunks: int | None = None,
    pad_trace: bool = False,
):
    """Assemble mixed residual using jitted element kernels + scatter_add."""
    u_vec = _coerce_mixed_u(space, u)
    ctxs = space.build_form_contexts()
    ker = kernel if kernel is not None else make_element_mixed_residual_kernel(
        res_form,
        params,
        space.field_names,
        space.elem_slices,
        space.space_key_by_field,
        space.alias_space_keys_by_field,
    )

    u_elems = u_vec[space.elem_dofs]
    if n_chunks is None:
        elem_res = jax.vmap(ker)(ctxs, u_elems)
    else:
        n_elems = int(u_elems.shape[0])
        if n_chunks <= 0:
            raise ValueError("n_chunks must be a positive integer.")
        n_chunks = min(int(n_chunks), int(n_elems))
        chunk_size = (n_elems + n_chunks - 1) // n_chunks
        stats = chunk_pad_stats(n_elems, n_chunks)
        _maybe_trace_pad(stats, n_chunks=n_chunks, pad_trace=pad_trace)
        pad = (-n_elems) % chunk_size
        if pad:
            ctxs_pad = jax.tree_util.tree_map(
                lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
                ctxs,
            )
            u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
        else:
            ctxs_pad = ctxs
            u_elems_pad = u_elems

        n_pad = n_elems + pad
        n_chunks = n_pad // chunk_size

        def _slice_first_dim(x, start, size):
            start_idx = (start,) + (0,) * (x.ndim - 1)
            slice_sizes = (size,) + x.shape[1:]
            return jax.lax.dynamic_slice(x, start_idx, slice_sizes)

        def chunk_fn(i):
            start = i * chunk_size
            ctx_chunk = jax.tree_util.tree_map(
                lambda x: _slice_first_dim(x, start, chunk_size),
                ctxs_pad,
            )
            u_chunk = _slice_first_dim(u_elems_pad, start, chunk_size)
            res_chunk = jax.vmap(ker)(ctx_chunk, u_chunk)
            return res_chunk.reshape(-1)

        data_chunks = jax.vmap(chunk_fn)(jnp.arange(n_chunks))
        elem_res = data_chunks.reshape(-1)[: n_elems * int(space.n_ldofs)].reshape(n_elems, -1)
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


def assemble_mixed_jacobian_values(
    space, res_form, u, params, *, kernel=None, n_chunks: int | None = None, pad_trace: bool = False
):
    """Assemble numeric values for mixed Jacobian (pattern-free)."""
    u_vec = _coerce_mixed_u(space, u)
    ctxs = space.build_form_contexts()
    ker = kernel if kernel is not None else make_element_mixed_jacobian_kernel(
        res_form,
        params,
        space.field_names,
        space.elem_slices,
        space.space_key_by_field,
        space.alias_space_keys_by_field,
    )

    u_elems = u_vec[space.elem_dofs]
    if n_chunks is None:
        J_e_all = jax.vmap(ker)(u_elems, ctxs)
        return J_e_all.reshape(-1)

    n_elems = int(u_elems.shape[0])
    if n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer.")
    n_chunks = min(int(n_chunks), int(n_elems))
    chunk_size = (n_elems + n_chunks - 1) // n_chunks
    stats = chunk_pad_stats(n_elems, n_chunks)
    _maybe_trace_pad(stats, n_chunks=n_chunks, pad_trace=pad_trace)
    pad = (-n_elems) % chunk_size
    if pad:
        ctxs_pad = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
            ctxs,
        )
        u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
    else:
        ctxs_pad = ctxs
        u_elems_pad = u_elems

    n_pad = n_elems + pad
    n_chunks = n_pad // chunk_size
    m = int(space.n_ldofs)

    def _slice_first_dim(x, start, size):
        start_idx = (start,) + (0,) * (x.ndim - 1)
        slice_sizes = (size,) + x.shape[1:]
        return jax.lax.dynamic_slice(x, start_idx, slice_sizes)

    def chunk_fn(i):
        start = i * chunk_size
        ctx_chunk = jax.tree_util.tree_map(
            lambda x: _slice_first_dim(x, start, chunk_size),
            ctxs_pad,
        )
        u_chunk = _slice_first_dim(u_elems_pad, start, chunk_size)
        J_e = jax.vmap(ker)(u_chunk, ctx_chunk)
        return J_e.reshape(-1)

    data_chunks = jax.vmap(chunk_fn)(jnp.arange(n_chunks))
    return data_chunks.reshape(-1)[: n_elems * m * m]


def assemble_mixed_jacobian_scatter(
    space,
    res_form,
    u,
    params,
    *,
    kernel=None,
    pattern=None,
    n_chunks: int | None = None,
    pad_trace: bool = False,
):
    """Assemble mixed Jacobian using jitted element kernels + scatter_add."""
    from ..solver import FluxSparseMatrix  # local import to avoid circular

    pat = pattern if pattern is not None else make_sparsity_pattern(space, with_idx=True)
    data = assemble_mixed_jacobian_values(
        space, res_form, u, params, kernel=kernel, n_chunks=n_chunks, pad_trace=pad_trace
    )
    return FluxSparseMatrix(pat, data)


def assemble_mixed_residual(
    space,
    res_form,
    u,
    params,
    *,
    sparse: bool = False,
    pattern=None,
    n_chunks: int | None = None,
    pad_trace: bool = False,
):
    """Assemble the global mixed residual vector."""
    return assemble_mixed_residual_scatter(
        space, res_form, u, params, sparse=sparse, n_chunks=n_chunks, pad_trace=pad_trace
    )


def assemble_mixed_jacobian(
    space,
    res_form,
    u,
    params,
    *,
    pattern=None,
    n_chunks: int | None = None,
    pad_trace: bool = False,
):
    """Assemble the global mixed Jacobian."""
    return assemble_mixed_jacobian_scatter(
        space,
        res_form,
        u,
        params,
        pattern=pattern,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )


def assemble_mixed_role_residual(
    space,
    res_form,
    u,
    params,
    *,
    sparse: bool = False,
    kernel=None,
    n_chunks: int | None = None,
    pad_trace: bool = False,
):
    """Assemble residual for MixedRoleFESpace using distinct unknown/residual layouts."""
    _ensure_mixed_role_volume_form(res_form, op="assemble_residual")
    if n_chunks is not None:
        raise ValueError("MixedRoleFESpace assemble_residual currently supports only non-chunked assembly.")
    if pad_trace:
        raise ValueError("MixedRoleFESpace assemble_residual does not support pad_trace.")

    u_vec = _coerce_mixed_role_u(space, u)
    ctxs = space.build_form_contexts()
    role_space_keys_by_field = {
        name: tuple(
            key for key in (
                name,
                (space.test_space_key_by_field or {}).get(name, name),
                (space.trial_space_key_by_field or {}).get(name, name),
                (space.unknown_space_key_by_field or {}).get(name, name),
            ) if key is not None
        )
        for name in space.field_names
    }
    ker = kernel if kernel is not None else make_element_mixed_role_residual_kernel(
        res_form,
        params,
        space.field_names,
        space.unknown_elem_slices,
        role_space_keys_by_field,
    )
    u_elems = u_vec[space.unknown_elem_dofs]
    elem_res = jax.vmap(ker)(ctxs, u_elems)
    rows = space.residual_elem_dofs.reshape(-1)
    data = elem_res.reshape(-1)
    if sparse:
        return rows, data, space.n_residual_dofs

    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    F = jnp.zeros((space.n_residual_dofs,), dtype=data.dtype)
    F = jax.lax.scatter_add(F, rows[:, None], data, sdn)
    return F


def assemble_mixed_role_jacobian(
    space,
    res_form,
    u,
    params,
    *,
    kernel=None,
    n_chunks: int | None = None,
    pad_trace: bool = False,
):
    """Assemble Jacobian for MixedRoleFESpace using residual x unknown layouts."""
    from ..solver import FluxSparseOperator

    _ensure_mixed_role_volume_form(res_form, op="assemble_jacobian")
    if n_chunks is not None:
        raise ValueError("MixedRoleFESpace assemble_jacobian currently supports only non-chunked assembly.")
    if pad_trace:
        raise ValueError("MixedRoleFESpace assemble_jacobian does not support pad_trace.")

    u_vec = _coerce_mixed_role_u(space, u)
    ctxs = space.build_form_contexts()
    role_space_keys_by_field = {
        name: tuple(
            key for key in (
                name,
                (space.test_space_key_by_field or {}).get(name, name),
                (space.trial_space_key_by_field or {}).get(name, name),
                (space.unknown_space_key_by_field or {}).get(name, name),
            ) if key is not None
        )
        for name in space.field_names
    }
    ker = kernel if kernel is not None else make_element_mixed_role_jacobian_kernel(
        res_form,
        params,
        space.field_names,
        space.unknown_elem_slices,
        role_space_keys_by_field,
    )
    u_elems = u_vec[space.unknown_elem_dofs]
    J_e = jax.vmap(ker)(u_elems, ctxs)
    rows = jnp.repeat(space.residual_elem_dofs, int(space.n_unknown_ldofs), axis=1).reshape(-1)
    cols = jnp.tile(space.unknown_elem_dofs, (1, int(space.n_residual_ldofs))).reshape(-1)
    data = J_e.reshape(-1)
    return FluxSparseOperator(
        rows,
        cols,
        data,
        shape=(int(space.n_residual_dofs), int(space.n_unknown_dofs)),
    )


__all__ = [
    "make_element_mixed_residual_kernel",
    "make_element_mixed_jacobian_kernel",
    "assemble_mixed_residual",
    "assemble_mixed_jacobian",
    "assemble_mixed_role_residual",
    "assemble_mixed_role_jacobian",
    "assemble_mixed_residual_scatter",
    "assemble_mixed_jacobian_scatter",
    "assemble_mixed_jacobian_values",
]

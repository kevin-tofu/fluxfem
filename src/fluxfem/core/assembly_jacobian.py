from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .dtypes import INDEX_DTYPE
from .forms import FormContext


def _contains_jax_value(obj: Any) -> bool:
    if isinstance(obj, jax.Array) or isinstance(obj, jax.core.Tracer):
        return True
    if isinstance(obj, np.ndarray):
        return False
    if isinstance(obj, dict):
        return any(_contains_jax_value(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_jax_value(v) for v in obj)
    data = getattr(obj, "data", None)
    if data is not None and data is not obj and not isinstance(obj, np.ndarray):
        return _contains_jax_value(data)
    return False


def _infer_backend(*values: Any, default: str = "jax") -> str:
    return "jax" if any(_contains_jax_value(v) for v in values) else default


def _assemble_jacobian_fixed_chunk_tail(
    *,
    space,
    vmapped_kernel,
    u_elems: jnp.ndarray,
    chunk_size: int,
    include_x_q,
    lightweight_context,
    chunk_build_context,
):
    from . import assembly as _a

    n_elems = int(u_elems.shape[0])
    n_full = n_elems // int(chunk_size)
    tail = n_elems % int(chunk_size)
    m = int(space.n_ldofs)
    use_chunk_context, conn_pad, elem_ids, ctxs_pad = _a._prepare_chunk_context_source(
        space,
        n_pad=n_elems,
        pad=0,
        dep=None,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        chunk_build_context=chunk_build_context,
    )

    def chunk_values_fn(start: int):
        ctx_chunk = _a._chunk_context_from_source(
            space,
            start=start,
            chunk_size=chunk_size,
            use_chunk_context=use_chunk_context,
            conn_pad=conn_pad,
            elem_ids=elem_ids,
            ctxs_pad=ctxs_pad,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
        )
        u_chunk = _a._slice_first_dim(u_elems, start, chunk_size)
        return vmapped_kernel(u_chunk, ctx_chunk)

    first_ctx = _a._chunk_context_from_source(
        space,
        start=0,
        chunk_size=1,
        use_chunk_context=use_chunk_context,
        conn_pad=conn_pad,
        elem_ids=elem_ids,
        ctxs_pad=ctxs_pad,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
    )
    sample_J = vmapped_kernel(_a._slice_first_dim(u_elems, 0, 1), first_ctx)[0]
    data = _a._accumulate_chunk_matrix_data(
        n_chunks=n_full,
        chunk_size=chunk_size,
        n_pad=n_elems,
        m=m,
        dtype=sample_J.dtype,
        valid_mask=jnp.ones((n_elems,), dtype=bool),
        chunk_values_fn=chunk_values_fn,
    )
    if tail:
        ctx_tail = _a._chunk_context_from_source(
            space,
            start=n_full * chunk_size,
            chunk_size=tail,
            use_chunk_context=use_chunk_context,
            conn_pad=conn_pad,
            elem_ids=elem_ids,
            ctxs_pad=ctxs_pad,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
        )
        u_tail = _a._slice_first_dim(u_elems, n_full * chunk_size, tail)
        tail_vals = vmapped_kernel(u_tail, ctx_tail).reshape(tail * m * m)
        data = jax.lax.dynamic_update_slice(data, tail_vals, (n_full * chunk_size * m * m,))
    return data[: n_elems * m * m]


def assemble_jacobian_global(
    space,
    res_form,
    u: jnp.ndarray,
    params: Any,
):
    """
    Assemble Jacobian (dR/du) from element residual res_form.
    res_form(ctx, u_elem, params) -> (n_q, n_ldofs)
    """
    from . import assembly as _a
    from ..solver import FluxSparseMatrix

    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs

    elem_data = space.build_form_contexts()

    def fe_fun(u_elem, ctx: FormContext, elem_id):
        ctx_with_id = FormContext(ctx.test, ctx.trial, ctx.x_q, ctx.w, elem_id)
        integrand = res_form(ctx_with_id, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        fe = _a._integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )
        return fe

    jac_fun = jax.jacrev(fe_fun, argnums=0)

    u_elems = u[elem_dofs]
    elem_ids = jnp.arange(elem_dofs.shape[0], dtype=INDEX_DTYPE)
    J_e_all = jax.vmap(jac_fun)(u_elems, elem_data, elem_ids)

    pat = _a._get_pattern(space, with_idx=False)
    if pat is None:
        rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1)
        cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1)
    else:
        rows = pat.rows
        cols = pat.cols
    data = J_e_all.reshape(-1)
    return FluxSparseMatrix(rows, cols, data, n_dofs)


def assemble_jacobian_elementwise(
    space,
    res_form,
    u: jnp.ndarray,
    params: Any,
):
    """
    Assemble Jacobian with element kernels via vmap + scatter_add.
    Recompiles if n_dofs changes, but independent of element count.
    """
    from . import assembly as _a
    from ..solver import FluxSparseMatrix

    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs

    ctxs = space.build_form_contexts()

    def fe_fun(u_elem, ctx: FormContext):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return _a._integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

    jac_fun = jax.jacrev(fe_fun, argnums=0)
    u_elems = u[elem_dofs]
    J_e_all = jax.vmap(jac_fun)(u_elems, ctxs)

    pat = _a._get_pattern(space, with_idx=False)
    if pat is None:
        rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1)
        cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1)
    else:
        rows = pat.rows
        cols = pat.cols
    data = J_e_all.reshape(-1)
    return FluxSparseMatrix(rows, cols, data, n_dofs)


def assemble_jacobian_values(
    space,
    res_form,
    u: jnp.ndarray,
    params: Any,
    *,
    backend: str | None = None,
    kernel=None,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy=None,
):
    """
    Assemble only the numeric values for the Jacobian (pattern-free).
    """
    from . import assembly as _a
    backend = _infer_backend(u, params, kernel, default="jax") if backend is None else str(backend).lower()
    if backend != "jax":
        raise NotImplementedError("assemble_jacobian backend='numpy' is not implemented.")

    include_x_q_req: bool | None = None if policy is not None else False
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _a._resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=include_x_q_req,
        lightweight_context=None,
        chunk_build_context=None,
        pad_trace=pad_trace,
    )
    fixed_chunk_size, max_padded_elems, allow_tail_chunk = _a._resolve_bucket_policy(policy=policy)
    if fixed_chunk_size is not None and n_chunks is not None:
        raise ValueError("Use either n_chunks or fixed_chunk_size/max_padded_elems, not both.")
    ker = kernel if kernel is not None else _a.make_element_jacobian_kernel(res_form, params)
    vmapped_kernel = jax.vmap(ker)

    u_elems = u[space.elem_dofs]
    if n_chunks is None and fixed_chunk_size is None:
        def _eval(include_x_q_eff: bool):
            ctxs = space.build_form_contexts(include_x_q=include_x_q_eff, lightweight=lightweight_context)
            return vmapped_kernel(u_elems, ctxs)

        try:
            J_e_all = _eval(include_x_q)
        except Exception:
            if include_x_q:
                raise
            J_e_all = _eval(True)
        return J_e_all.reshape(-1)

    n_elems = int(u_elems.shape[0])
    if fixed_chunk_size is not None and max_padded_elems is None and allow_tail_chunk:
        def _eval_tail(include_x_q_eff: bool):
            return _assemble_jacobian_fixed_chunk_tail(
                space=space,
                vmapped_kernel=vmapped_kernel,
                u_elems=u_elems,
                chunk_size=int(fixed_chunk_size),
                include_x_q=include_x_q_eff,
                lightweight_context=lightweight_context,
                chunk_build_context=chunk_build_context,
            )

        try:
            return _eval_tail(include_x_q)
        except Exception:
            if include_x_q:
                raise
            return _eval_tail(True)
    n_chunks, chunk_size, pad, n_pad, valid_mask = _a._prepare_chunk_iteration(
        n_elems=int(n_elems),
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        fixed_chunk_size=fixed_chunk_size,
        max_padded_elems=max_padded_elems,
    )
    m = int(space.n_ldofs)
    if pad:
        u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
    else:
        u_elems_pad = u_elems

    def _init_chunk(include_x_q_eff: bool):
        use_chunk_context, conn_pad, elem_ids, ctxs_pad = _a._prepare_chunk_context_source(
            space,
            n_pad=int(n_pad),
            pad=int(pad),
            dep=None,
            include_x_q=include_x_q_eff,
            lightweight_context=lightweight_context,
            chunk_build_context=chunk_build_context,
        )

        def chunk_values_fn(start: int):
            ctx_chunk = _a._chunk_context_from_source(
                space,
                start=start,
                chunk_size=chunk_size,
                use_chunk_context=use_chunk_context,
                conn_pad=conn_pad,
                elem_ids=elem_ids,
                ctxs_pad=ctxs_pad,
                include_x_q=include_x_q_eff,
                lightweight_context=lightweight_context,
            )
            u_chunk = _a._slice_first_dim(u_elems_pad, start, chunk_size)
            return vmapped_kernel(u_chunk, ctx_chunk)

        sample_J = chunk_values_fn(0)[0]
        return sample_J, chunk_values_fn

    try:
        sample_J, chunk_values_fn = _init_chunk(include_x_q)
    except Exception:
        if include_x_q:
            raise
        sample_J, chunk_values_fn = _init_chunk(True)

    data = _a._accumulate_chunk_matrix_data(
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        n_pad=n_pad,
        m=m,
        dtype=sample_J.dtype,
        valid_mask=valid_mask,
        chunk_values_fn=chunk_values_fn,
    )
    return data[: n_elems * m * m]


def assemble_jacobian_scatter(
    space,
    res_form,
    u: jnp.ndarray,
    params: Any,
    *,
    backend: str | None = None,
    kernel=None,
    pattern=None,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy=None,
):
    """
    Assemble Jacobian using jitted element kernel + vmap + scatter_add.
    If a SparsityPattern is provided, rows/cols are reused without regeneration.
    """
    from . import assembly as _a
    from ..solver import FluxSparseMatrix
    backend = _infer_backend(u, params, kernel, default="jax") if backend is None else str(backend).lower()
    if backend != "jax":
        raise NotImplementedError("assemble_jacobian backend='numpy' is not implemented.")

    if pattern is not None:
        pat = pattern
    else:
        pat = _a._get_pattern(space, with_idx=True)
        if pat is None:
            pat = _a.make_sparsity_pattern(space, with_idx=True)
    data = assemble_jacobian_values(
        space,
        res_form,
        u,
        params,
        backend=backend,
        kernel=kernel,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        policy=policy,
    )
    return FluxSparseMatrix(pat, data)


def assemble_jacobian(
    space,
    res_form,
    u: jnp.ndarray,
    params: Any,
    *,
    backend: str | None = None,
    kernel=None,
    pattern=None,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy=None,
):
    """
    Assemble the global Jacobian (scatter-based).
    If kernel is provided: kernel(u_elem, ctx) -> (n_ldofs, n_ldofs).
    """
    return assemble_jacobian_scatter(
        space,
        res_form,
        u,
        params,
        backend=backend,
        kernel=kernel,
        pattern=pattern,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        policy=policy,
    )

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .forms import FormContext


def _assemble_residual_fixed_chunk_tail(
    *,
    space,
    ker,
    u_elems: jnp.ndarray,
    n_dofs: int,
    vector_accumulation: str,
    sparse: bool,
    chunk_size: int,
    include_x_q,
    lightweight_context,
    chunk_build_context,
) -> Any:
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
    valid_mask = jnp.ones((n_elems,), dtype=bool)

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
        return jax.vmap(ker)(ctx_chunk, u_chunk)

    if sparse:
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
        sample_res = jax.vmap(ker)(first_ctx, _a._slice_first_dim(u_elems, 0, 1))[0]
        data = _a._accumulate_chunk_vector_data(
            n_chunks=n_full,
            chunk_size=chunk_size,
            n_pad=n_elems,
            m=m,
            dtype=sample_res.dtype,
            valid_mask=valid_mask,
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
            tail_vals = jax.vmap(ker)(ctx_tail, u_tail).reshape(tail * m)
            data = jax.lax.dynamic_update_slice(data, tail_vals, (n_full * chunk_size * m,))
        rows = _a._get_elem_rows(space)
        return data[: n_elems * m], rows

    sample_ctx = _a._chunk_context_from_source(
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
    sample_res = jax.vmap(ker)(sample_ctx, _a._slice_first_dim(u_elems, 0, 1))[0]
    elem_dofs = space.elem_dofs
    if vector_accumulation == "scatter":
        F = _a._accumulate_chunk_vector_scatter(
            n_chunks=n_full,
            chunk_size=chunk_size,
            m=m,
            n_dofs=n_dofs,
            dtype=sample_res.dtype,
            valid_mask=valid_mask,
            elem_dofs_pad=elem_dofs,
            chunk_values_fn=chunk_values_fn,
        )
    else:
        F = _a._accumulate_chunk_vector_segment(
            n_chunks=n_full,
            chunk_size=chunk_size,
            m=m,
            n_dofs=n_dofs,
            dtype=sample_res.dtype,
            valid_mask=valid_mask,
            elem_dofs_pad=elem_dofs,
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
        tail_vals = jax.vmap(ker)(ctx_tail, u_tail)
        tail_rows = _a._slice_first_dim(elem_dofs, n_full * chunk_size, tail).reshape(-1)
        tail_data = tail_vals.reshape(-1)
        if vector_accumulation == "scatter":
            sdn = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            F = jax.lax.scatter_add(F, tail_rows[:, None], tail_data, sdn)
        else:
            F = F + jax.ops.segment_sum(tail_data, tail_rows, n_dofs)
    return F


def assemble_residual_global(
    space,
    form,
    u: jnp.ndarray,
    params: Any,
    *,
    sparse: bool = False,
):
    """
    Assemble residual vector that depends on u.
    form(ctx, u_elem, params) -> (n_q, n_ldofs)
    """
    from . import assembly as _a

    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs

    elem_data = space.build_form_contexts()

    def per_element(ctx: FormContext, conn: jnp.ndarray, elem_id: jnp.ndarray):
        u_elem = u[conn]
        ctx_with_id = FormContext(ctx.test, ctx.trial, ctx.x_q, ctx.w, elem_id)
        integrand = form(ctx_with_id, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        fe = _a._integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(form, "_includes_measure", False)),
        )
        return fe

    elem_ids = jnp.arange(elem_dofs.shape[0], dtype=_a.INDEX_DTYPE)
    F_e_all = jax.vmap(per_element)(elem_data, elem_dofs, elem_ids)

    rows = _a._get_elem_rows(space)
    data = F_e_all.reshape(-1)

    if sparse:
        return rows, data, n_dofs

    F = jax.ops.segment_sum(data, rows, n_dofs)
    return F


def assemble_residual_elementwise(
    space,
    res_form,
    u: jnp.ndarray,
    params: Any,
    *,
    sparse: bool = False,
):
    """
    Assemble residual using element kernels via vmap + scatter_add.
    Recompiles if n_dofs changes, but independent of element count.
    """
    from . import assembly as _a

    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    ctxs = space.build_form_contexts()

    def per_element(ctx: FormContext, u_elem: jnp.ndarray):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return _a._integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

    u_elems = u[elem_dofs]
    F_e_all = jax.vmap(per_element)(ctxs, u_elems)
    rows = _a._get_elem_rows(space)
    data = F_e_all.reshape(-1)

    if sparse:
        return rows, data, n_dofs

    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    F = jnp.zeros(n_dofs, dtype=data.dtype)
    F = jax.lax.scatter_add(F, rows[:, None], data, sdn)
    return F


def assemble_residual_scatter(
    space,
    res_form,
    u: jnp.ndarray,
    params: Any,
    *,
    backend: str = "jax",
    kernel=None,
    sparse: bool = False,
    vector_accumulation: str = "scatter",
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy=None,
):
    """
    Assemble residual using jitted element kernel + vmap + scatter_add.
    Avoids Python loops; good for JIT stability.
    """
    from . import assembly as _a
    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")

    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _a._resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=None,
        lightweight_context=None,
        chunk_build_context=None,
        pad_trace=pad_trace,
    )
    fixed_chunk_size, max_padded_elems, allow_tail_chunk = _a._resolve_bucket_policy(policy=policy)
    if fixed_chunk_size is not None and n_chunks is not None:
        raise ValueError("Use either n_chunks or fixed_chunk_size/max_padded_elems, not both.")
    if vector_accumulation not in ("segment", "scatter"):
        raise ValueError(
            f"vector_accumulation must be 'segment' or 'scatter' (got {vector_accumulation!r})"
        )
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    if jax.core.trace_ctx.is_top_level():
        if np.max(elem_dofs) >= n_dofs:
            raise ValueError("elem_dofs contains index outside n_dofs")
        if np.min(elem_dofs) < 0:
            raise ValueError("elem_dofs contains negative index")
    ker = kernel if kernel is not None else _a.make_element_residual_kernel(res_form, params)

    if backend == "numpy":
        if n_chunks is not None or fixed_chunk_size is not None:
            raise ValueError("backend='numpy' currently supports only non-chunked assembly.")
        elem_dofs_np = np.asarray(elem_dofs, dtype=int)
        u_np = np.asarray(u)
        ctxs = space.build_form_contexts(include_x_q=include_x_q, lightweight=lightweight_context)
        n_elems = int(elem_dofs_np.shape[0])
        data_parts: list[np.ndarray] = []
        for e in range(n_elems):
            ctx_e = jax.tree_util.tree_map(lambda x: x[e], ctxs)
            u_elem = u_np[elem_dofs_np[e]]
            fe = np.asarray(ker(ctx_e, u_elem), dtype=float).reshape(-1)
            data_parts.append(fe)
        data = np.concatenate(data_parts, axis=0) if data_parts else np.zeros((0,), dtype=float)
        rows = np.asarray(_a._get_elem_rows(space), dtype=int)
        if sparse:
            return rows, data, n_dofs
        F = np.zeros((int(n_dofs),), dtype=float)
        if data.size:
            np.add.at(F, rows, data)
        return F

    u_elems = u[elem_dofs]
    if n_chunks is None and fixed_chunk_size is None:
        ctxs = space.build_form_contexts(include_x_q=include_x_q, lightweight=lightweight_context)
        elem_res = jax.vmap(ker)(ctxs, u_elems)
        data = elem_res.reshape(-1)
    else:
        n_elems = int(u_elems.shape[0])
        if fixed_chunk_size is not None and max_padded_elems is None and allow_tail_chunk:
            out = _assemble_residual_fixed_chunk_tail(
                space=space,
                ker=ker,
                u_elems=u_elems,
                n_dofs=n_dofs,
                vector_accumulation=vector_accumulation,
                sparse=sparse,
                chunk_size=int(fixed_chunk_size),
                include_x_q=include_x_q,
                lightweight_context=lightweight_context,
                chunk_build_context=chunk_build_context,
            )
            if sparse:
                data, rows = out
                return rows, data, n_dofs
            return out
        n_chunks, chunk_size, pad, n_pad, valid_mask = _a._prepare_chunk_iteration(
            n_elems=int(n_elems),
            n_chunks=n_chunks,
            pad_trace=pad_trace,
            fixed_chunk_size=fixed_chunk_size,
            max_padded_elems=max_padded_elems,
        )
        if pad:
            u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
        else:
            u_elems_pad = u_elems

        use_chunk_context, conn_pad, elem_ids, ctxs_pad = _a._prepare_chunk_context_source(
            space,
            n_pad=int(n_pad),
            pad=int(pad),
            dep=None,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
            chunk_build_context=chunk_build_context,
        )

        m = int(space.n_ldofs)
        if sparse:
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
            sample_res = jax.vmap(ker)(first_ctx, _a._slice_first_dim(u_elems_pad, 0, 1))[0]

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
                u_chunk = _a._slice_first_dim(u_elems_pad, start, chunk_size)
                return jax.vmap(ker)(ctx_chunk, u_chunk)

            data = _a._accumulate_chunk_vector_data(
                n_chunks=n_chunks,
                chunk_size=chunk_size,
                n_pad=n_pad,
                m=m,
                dtype=sample_res.dtype,
                valid_mask=valid_mask,
                chunk_values_fn=chunk_values_fn,
            )
            data = data[: n_elems * m]
        else:
            if pad:
                elem_dofs_pad = jnp.concatenate([elem_dofs, jnp.repeat(elem_dofs[-1:], pad, axis=0)], axis=0)
            else:
                elem_dofs_pad = elem_dofs
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
            sample_res = jax.vmap(ker)(first_ctx, _a._slice_first_dim(u_elems_pad, 0, 1))[0]

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
                u_chunk = _a._slice_first_dim(u_elems_pad, start, chunk_size)
                return jax.vmap(ker)(ctx_chunk, u_chunk)

            if vector_accumulation == "scatter":
                F = _a._accumulate_chunk_vector_scatter(
                    n_chunks=n_chunks,
                    chunk_size=chunk_size,
                    m=m,
                    n_dofs=n_dofs,
                    dtype=sample_res.dtype,
                    valid_mask=valid_mask,
                    elem_dofs_pad=elem_dofs_pad,
                    chunk_values_fn=chunk_values_fn,
                )
            else:
                F = _a._accumulate_chunk_vector_segment(
                    n_chunks=n_chunks,
                    chunk_size=chunk_size,
                    m=m,
                    n_dofs=n_dofs,
                    dtype=sample_res.dtype,
                    valid_mask=valid_mask,
                    elem_dofs_pad=elem_dofs_pad,
                    chunk_values_fn=chunk_values_fn,
                )
            if jax.core.trace_ctx.is_top_level():
                if not bool(jax.block_until_ready(jnp.all(jnp.isfinite(F)))):
                    bad = int(jnp.count_nonzero(~jnp.isfinite(F)))
                    raise RuntimeError(f"[assemble_residual_scatter] residual vector nonfinite: {bad}")
            return F
    if jax.core.trace_ctx.is_top_level():
        if not bool(jax.block_until_ready(jnp.all(jnp.isfinite(data)))):
            bad = int(jnp.count_nonzero(~jnp.isfinite(data)))
            raise RuntimeError(f"[assemble_residual_scatter] residual data nonfinite: {bad}")

    rows = _a._get_elem_rows(space)

    if sparse:
        return rows, data, n_dofs

    if vector_accumulation == "scatter":
        sdn = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        F = jnp.zeros((n_dofs,), dtype=data.dtype)
        F = jax.lax.scatter_add(F, rows[:, None], data, sdn)
        return F
    F = jax.ops.segment_sum(data, rows, n_dofs)
    return F


def assemble_residual(
    space,
    form,
    u: jnp.ndarray,
    params: Any,
    *,
    backend: str = "jax",
    kernel=None,
    sparse: bool = False,
    vector_accumulation: str = "scatter",
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy=None,
):
    """
    Assemble the global residual vector (scatter-based).
    If kernel is provided: kernel(ctx, u_elem) -> (n_ldofs,).
    """
    return assemble_residual_scatter(
        space,
        form,
        u,
        params,
        backend=backend,
        kernel=kernel,
        sparse=sparse,
        vector_accumulation=vector_accumulation,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        policy=policy,
    )

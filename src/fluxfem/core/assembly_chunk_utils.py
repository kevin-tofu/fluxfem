from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from .dtypes import INDEX_DTYPE
from .forms import FormContext

Array = jnp.ndarray


def chunk_pad_stats(n_elems: int, n_chunks: Optional[int]) -> dict[str, int | float | None]:
    """
    Compute padding overhead for chunked assembly.
    Returns dict with chunk_size, pad, n_pad, and pad_ratio.
    """
    n_elems = int(n_elems)
    if n_chunks is None or n_elems <= 0:
        return {"chunk_size": None, "pad": 0, "n_pad": n_elems, "pad_ratio": 0.0}
    n_chunks = min(int(n_chunks), n_elems)
    chunk_size = (n_elems + n_chunks - 1) // n_chunks
    pad = (-n_elems) % chunk_size
    n_pad = n_elems + pad
    pad_ratio = float(pad) / float(n_elems) if n_elems else 0.0
    return {"chunk_size": int(chunk_size), "pad": int(pad), "n_pad": int(n_pad), "pad_ratio": pad_ratio}


def _maybe_trace_pad(
    stats: dict[str, int | float | None], *, n_chunks: Optional[int], pad_trace: bool
) -> None:
    if not pad_trace or not jax.core.trace_ctx.is_top_level():
        return
    if n_chunks is None:
        return
    print(
        "[pad]",
        f"n_chunks={int(n_chunks)}",
        f"chunk_size={stats['chunk_size']}",
        f"pad={stats['pad']}",
        f"pad_ratio={stats['pad_ratio']:.4f}",
        flush=True,
    )


def _slice_first_dim(x: Array, start: int, size: int) -> Array:
    start_idx = (start,) + (0,) * (x.ndim - 1)
    slice_sizes = (size,) + x.shape[1:]
    return jax.lax.dynamic_slice(x, start_idx, slice_sizes)


def _prepare_chunk_iteration(
    *,
    n_elems: int,
    n_chunks: int | None,
    pad_trace: bool,
) -> tuple[int, int, int, int, Array]:
    if n_chunks is None or n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer.")
    n_chunks_eff = min(int(n_chunks), int(n_elems))
    chunk_size = (int(n_elems) + n_chunks_eff - 1) // n_chunks_eff
    stats = chunk_pad_stats(n_elems, n_chunks_eff)
    _maybe_trace_pad(stats, n_chunks=n_chunks_eff, pad_trace=pad_trace)
    pad = (-int(n_elems)) % chunk_size
    n_pad = int(n_elems) + pad
    n_chunks_eff = n_pad // chunk_size
    valid_mask = jnp.arange(int(n_pad), dtype=INDEX_DTYPE) < int(n_elems)
    return int(n_chunks_eff), int(chunk_size), int(pad), int(n_pad), valid_mask


def _prepare_chunk_context_source(
    space,
    *,
    n_pad: int,
    pad: int,
    dep: jnp.ndarray | None,
    include_x_q: bool | None,
    lightweight_context: bool | None,
    chunk_build_context: bool,
    elem_data: FormContext | None = None,
) -> tuple[bool, Array | None, Array | None, FormContext | None]:
    use_chunk_context = bool(
        chunk_build_context
        and hasattr(space, "build_form_contexts_from_elem_coords")
        and hasattr(space, "mesh")
    )
    if use_chunk_context:
        conn = space.mesh.conn
        if pad:
            conn_pad = jnp.concatenate([conn, jnp.repeat(conn[-1:], pad, axis=0)], axis=0)
        else:
            conn_pad = conn
        elem_ids = jnp.arange(int(n_pad), dtype=INDEX_DTYPE)
        return True, conn_pad, elem_ids, None

    ctxs = elem_data if elem_data is not None else space.build_form_contexts(
        dep=dep,
        include_x_q=include_x_q,
        lightweight=lightweight_context,
    )
    if pad:
        ctxs_pad = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
            ctxs,
        )
    else:
        ctxs_pad = ctxs
    return False, None, None, ctxs_pad


def _chunk_context_from_source(
    space,
    *,
    start: int,
    chunk_size: int,
    use_chunk_context: bool,
    conn_pad: Array | None,
    elem_ids: Array | None,
    ctxs_pad: FormContext | None,
    include_x_q: bool | None,
    lightweight_context: bool | None,
) -> FormContext:
    if use_chunk_context:
        assert conn_pad is not None
        assert elem_ids is not None
        conn_chunk = _slice_first_dim(conn_pad, start, chunk_size)
        elem_coords_chunk = space.mesh.coords[conn_chunk]
        elem_id_chunk = _slice_first_dim(elem_ids, start, chunk_size)
        return space.build_form_contexts_from_elem_coords(
            elem_coords_chunk,
            include_x_q=include_x_q,
            lightweight=lightweight_context,
            elem_id=elem_id_chunk,
        )
    assert ctxs_pad is not None
    return jax.tree_util.tree_map(
        lambda x: _slice_first_dim(x, start, chunk_size),
        ctxs_pad,
    )

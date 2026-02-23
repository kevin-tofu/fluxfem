from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .assembly_chunk_utils import _slice_first_dim

Array = jnp.ndarray


def accumulate_chunk_vector_data(
    *,
    n_chunks: int,
    chunk_size: int,
    n_pad: int,
    m: int,
    dtype: jnp.dtype | np.dtype | type,
    valid_mask: Array,
    chunk_values_fn: Callable[[int], Array],
) -> Array:
    """Accumulate per-chunk vector blocks into a flat padded element-data buffer."""
    data = jnp.zeros((n_pad * m,), dtype=dtype)

    def loop_body(i, data_flat):
        start = i * chunk_size
        vec_chunk = chunk_values_fn(start)
        chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(vec_chunk.dtype)
        vec_chunk = vec_chunk * chunk_valid[:, None]
        return jax.lax.dynamic_update_slice(
            data_flat,
            vec_chunk.reshape(chunk_size * m),
            (start * m,),
        )

    return jax.lax.fori_loop(0, n_chunks, loop_body, data)


def accumulate_chunk_vector_scatter(
    *,
    n_chunks: int,
    chunk_size: int,
    m: int,
    n_dofs: int,
    dtype: jnp.dtype | np.dtype | type,
    valid_mask: Array,
    elem_dofs_pad: Array,
    chunk_values_fn: Callable[[int], Array],
) -> Array:
    """Accumulate per-chunk vectors directly into global DOF space via scatter_add."""
    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    F0 = jnp.zeros((n_dofs,), dtype=dtype)

    def loop_body(i, F_acc):
        start = i * chunk_size
        vec_chunk = chunk_values_fn(start)
        chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(vec_chunk.dtype)
        vec_chunk = vec_chunk * chunk_valid[:, None]
        data_chunk = vec_chunk.reshape(chunk_size * m)
        rows_chunk = _slice_first_dim(elem_dofs_pad, start, chunk_size).reshape(-1)
        return jax.lax.scatter_add(F_acc, rows_chunk[:, None], data_chunk, sdn)

    return jax.lax.fori_loop(0, n_chunks, loop_body, F0)


def accumulate_chunk_vector_segment(
    *,
    n_chunks: int,
    chunk_size: int,
    m: int,
    n_dofs: int,
    dtype: jnp.dtype | np.dtype | type,
    valid_mask: Array,
    elem_dofs_pad: Array,
    chunk_values_fn: Callable[[int], Array],
) -> Array:
    """Accumulate per-chunk vectors into global DOF space via per-chunk segment_sum."""
    F0 = jnp.zeros((n_dofs,), dtype=dtype)

    def loop_body(i, F_acc):
        start = i * chunk_size
        vec_chunk = chunk_values_fn(start)
        chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(vec_chunk.dtype)
        vec_chunk = vec_chunk * chunk_valid[:, None]
        data_chunk = vec_chunk.reshape(chunk_size * m)
        rows_chunk = _slice_first_dim(elem_dofs_pad, start, chunk_size).reshape(-1)
        return F_acc + jax.ops.segment_sum(data_chunk, rows_chunk, n_dofs)

    return jax.lax.fori_loop(0, n_chunks, loop_body, F0)

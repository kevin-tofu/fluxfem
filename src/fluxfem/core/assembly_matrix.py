from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .assembly_chunk_utils import _slice_first_dim

Array = jnp.ndarray


def accumulate_chunk_matrix_data(
    *,
    n_chunks: int,
    chunk_size: int,
    n_pad: int,
    m: int,
    dtype: jnp.dtype | np.dtype | type,
    valid_mask: Array,
    chunk_values_fn: Callable[[int], Array],
) -> Array:
    """Accumulate per-chunk matrix blocks into a flat padded element-data buffer."""
    data = jnp.zeros((n_pad * m * m,), dtype=dtype)

    def loop_body(i, data_flat):
        start = i * chunk_size
        mat_chunk = chunk_values_fn(start)
        chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(mat_chunk.dtype)
        mat_chunk = mat_chunk * chunk_valid[:, None, None]
        return jax.lax.dynamic_update_slice(
            data_flat,
            mat_chunk.reshape(chunk_size * m * m),
            (start * m * m,),
        )

    return jax.lax.fori_loop(0, n_chunks, loop_body, data)


def accumulate_chunk_matrix_and_vector_scatter(
    *,
    n_chunks: int,
    chunk_size: int,
    n_pad: int,
    m: int,
    n_dofs: int,
    matrix_dtype: jnp.dtype | np.dtype | type,
    vector_dtype: jnp.dtype | np.dtype | type,
    valid_mask: Array,
    elem_dofs_pad: Array,
    chunk_values_fn: Callable[[int], tuple[Array, Array]],
) -> tuple[Array, Array]:
    """Single-pass chunk accumulation for matrix data buffer + global RHS scatter."""
    K_data = jnp.zeros((n_pad * m * m,), dtype=matrix_dtype)
    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    F0 = jnp.zeros((n_dofs,), dtype=vector_dtype)

    def loop_body(i, carry):
        K_flat, F_acc = carry
        start = i * chunk_size
        mat_chunk, vec_chunk = chunk_values_fn(start)
        chunk_valid_mat = _slice_first_dim(valid_mask, start, chunk_size).astype(mat_chunk.dtype)
        mat_chunk = mat_chunk * chunk_valid_mat[:, None, None]
        chunk_valid_vec = _slice_first_dim(valid_mask, start, chunk_size).astype(vec_chunk.dtype)
        vec_chunk = vec_chunk * chunk_valid_vec[:, None]
        K_flat = jax.lax.dynamic_update_slice(
            K_flat,
            mat_chunk.reshape(chunk_size * m * m),
            (start * m * m,),
        )
        data_chunk = vec_chunk.reshape(chunk_size * m)
        rows_chunk = _slice_first_dim(elem_dofs_pad, start, chunk_size).reshape(-1)
        F_acc = jax.lax.scatter_add(F_acc, rows_chunk[:, None], data_chunk, sdn)
        return (K_flat, F_acc)

    return jax.lax.fori_loop(0, n_chunks, loop_body, (K_data, F0))


def accumulate_chunk_matrix_and_vector_segment(
    *,
    n_chunks: int,
    chunk_size: int,
    n_pad: int,
    m: int,
    n_dofs: int,
    matrix_dtype: jnp.dtype | np.dtype | type,
    vector_dtype: jnp.dtype | np.dtype | type,
    valid_mask: Array,
    elem_dofs_pad: Array,
    chunk_values_fn: Callable[[int], tuple[Array, Array]],
) -> tuple[Array, Array]:
    """Chunk accumulation with matrix buffering + per-chunk segment_sum for vector."""
    K_data = jnp.zeros((n_pad * m * m,), dtype=matrix_dtype)
    F_acc = jnp.zeros((n_dofs,), dtype=vector_dtype)

    def loop_body(i, carry):
        K_flat, F = carry
        start = i * chunk_size
        mat_chunk, vec_chunk = chunk_values_fn(start)
        chunk_valid_mat = _slice_first_dim(valid_mask, start, chunk_size).astype(mat_chunk.dtype)
        chunk_valid_vec = _slice_first_dim(valid_mask, start, chunk_size).astype(vec_chunk.dtype)
        mat_chunk = mat_chunk * chunk_valid_mat[:, None, None]
        vec_chunk = vec_chunk * chunk_valid_vec[:, None]
        K_flat = jax.lax.dynamic_update_slice(
            K_flat,
            mat_chunk.reshape(chunk_size * m * m),
            (start * m * m,),
        )
        data_chunk = vec_chunk.reshape(chunk_size * m)
        rows_chunk = _slice_first_dim(elem_dofs_pad, start, chunk_size).reshape(-1)
        F = F + jax.ops.segment_sum(data_chunk, rows_chunk, n_dofs)
        return (K_flat, F)

    return jax.lax.fori_loop(0, n_chunks, loop_body, (K_data, F_acc))

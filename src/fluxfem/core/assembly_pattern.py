from __future__ import annotations

import jax.numpy as jnp

from .dtypes import INDEX_DTYPE


def make_sparsity_pattern(space, *, with_idx: bool = True):
    """
    Build a SparsityPattern (rows/cols[/idx]) that is independent of the solution.
    NOTE: rows/cols ordering matches assemble_jacobian_values(...).reshape(-1)
    so that pattern and data are aligned 1:1. If you change the flattening/
    compression strategy, keep this ordering contract in sync.
    """
    from ..solver import SparsityPattern

    elem_dofs = jnp.asarray(space.elem_dofs, dtype=INDEX_DTYPE)
    n_dofs = int(space.n_dofs)
    n_ldofs = int(space.n_ldofs)

    rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1).astype(INDEX_DTYPE)
    cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1).astype(INDEX_DTYPE)

    key = rows.astype(INDEX_DTYPE) * jnp.asarray(n_dofs, dtype=INDEX_DTYPE) + cols.astype(INDEX_DTYPE)
    order = jnp.argsort(key).astype(INDEX_DTYPE)
    rows_sorted = rows[order]
    cols_sorted = cols[order]
    counts = jnp.bincount(rows_sorted, length=n_dofs).astype(INDEX_DTYPE)
    indptr_j = jnp.concatenate([jnp.array([0], dtype=INDEX_DTYPE), jnp.cumsum(counts)])
    indices_j = cols_sorted.astype(INDEX_DTYPE)
    perm = order

    if with_idx:
        idx = (rows.astype(INDEX_DTYPE) * jnp.asarray(n_dofs, dtype=INDEX_DTYPE) + cols.astype(INDEX_DTYPE)).astype(INDEX_DTYPE)
        return SparsityPattern(
            rows=rows,
            cols=cols,
            n_dofs=n_dofs,
            idx=idx,
            perm=perm,
            indptr=indptr_j,
            indices=indices_j,
        )
    return SparsityPattern(
        rows=rows,
        cols=cols,
        n_dofs=n_dofs,
        idx=None,
        perm=perm,
        indptr=indptr_j,
        indices=indices_j,
    )

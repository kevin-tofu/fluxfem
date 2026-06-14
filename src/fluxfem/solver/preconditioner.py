from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeAlias

import numpy as np
import jax.numpy as jnp

try:
    from jax.experimental import sparse as jsparse
except Exception:  # pragma: no cover
    jsparse = None

from .sparse import FluxSparseMatrix

ArrayLike: TypeAlias = jnp.ndarray
Preconditioner: TypeAlias = Callable[[jnp.ndarray], jnp.ndarray]


def _extract_block_sizes(
    n: int,
    *,
    dof_per_node: int | None,
    block_sizes: Sequence[int] | None,
    meta: dict | None,
) -> int:
    sizes = None
    if block_sizes is not None:
        sizes = np.asarray(block_sizes, dtype=int)
    elif dof_per_node is not None:
        if dof_per_node <= 0:
            raise ValueError("dof_per_node must be positive")
        if n % dof_per_node != 0:
            raise ValueError("dof_per_node must divide n_dofs")
        sizes = np.full(n // dof_per_node, dof_per_node, dtype=int)
    elif meta:
        if meta.get("dof_layout") not in (None, "blocked"):
            raise ValueError("block_jacobi requires dof_layout='blocked'")
        if "block_sizes" in meta:
            sizes = np.asarray(meta["block_sizes"], dtype=int)
        elif "dof_per_node" in meta:
            d = int(meta["dof_per_node"])
            if d <= 0 or n % d != 0:
                raise ValueError("meta['dof_per_node'] must divide n_dofs")
            sizes = np.full(n // d, d, dtype=int)
    if sizes is None or sizes.size == 0:
        raise ValueError("block_jacobi requires block_sizes or dof_per_node")
    if np.any(sizes <= 0):
        raise ValueError("block_sizes entries must be positive")
    if int(sizes.sum()) != n:
        raise ValueError("sum(block_sizes) must equal n_dofs")
    if not np.all(sizes == sizes[0]):
        raise ValueError("block_sizes must be uniform for block_jacobi")
    return int(sizes[0])


def make_block_jacobi_preconditioner(
    A: FluxSparseMatrix | "jsparse.BCOO",
    *,
    dof_per_node: int | None = None,
    block_sizes: Sequence[int] | None = None,
) -> Preconditioner:
    """
    Build block Jacobi preconditioner for blocked DOF layouts.

    Priority:
      block_sizes -> dof_per_node -> A.meta (block_sizes / dof_per_node) -> error
    """
    if jsparse is not None and isinstance(A, jsparse.BCOO):
        n = int(A.shape[0])
        block_size = _extract_block_sizes(
            n, dof_per_node=dof_per_node, block_sizes=block_sizes, meta=None
        )
        rows = jnp.asarray(A.indices[:, 0])
        cols = jnp.asarray(A.indices[:, 1])
        data = jnp.asarray(A.data)
    elif isinstance(A, FluxSparseMatrix):
        n = int(A.n_dofs)
        block_size = _extract_block_sizes(
            n, dof_per_node=dof_per_node, block_sizes=block_sizes, meta=A.meta
        )
        rows = jnp.asarray(A.pattern.rows)
        cols = jnp.asarray(A.pattern.cols)
        data = jnp.asarray(A.data)
    else:
        raise ValueError("block_jacobi requires FluxSparseMatrix or BCOO")

    if n % block_size != 0:
        raise ValueError("block_size must divide n_dofs")
    n_block = n // block_size
    block_rows = rows // block_size
    block_cols = cols // block_size
    lr = rows % block_size
    lc = cols % block_size
    mask = block_rows == block_cols
    block_rows = block_rows[mask]
    lr = lr[mask]
    lc = lc[mask]
    data = data[mask]
    blocks = jnp.zeros((n_block, block_size, block_size), dtype=data.dtype)
    blocks = blocks.at[block_rows, lr, lc].add(data)
    blocks = blocks + 1e-12 * jnp.eye(block_size)[None, :, :]
    inv_blocks = jnp.linalg.inv(blocks)

    def precon(r: jnp.ndarray) -> jnp.ndarray:
        rb = r.reshape((n_block, block_size))
        zb = jnp.einsum("bij,bj->bi", inv_blocks, rb)
        return zb.reshape((-1,))

    return precon

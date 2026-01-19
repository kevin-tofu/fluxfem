from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .block_system import split_block_matrix
from .sparse import FluxSparseMatrix


def diag(**blocks):
    return dict(blocks)


def _infer_sizes_from_diag(diag_blocks):
    sizes = {}
    for name, blk in diag_blocks.items():
        if isinstance(blk, FluxSparseMatrix):
            sizes[name] = int(blk.n_dofs)
        elif sp is not None and sp.issparse(blk):
            shape = blk.shape
            if shape[0] != shape[1]:
                raise ValueError(f"diag block {name} must be square, got {shape}")
            sizes[name] = int(shape[0])
        else:
            arr = np.asarray(blk)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(f"diag block {name} must be square, got {arr.shape}")
            sizes[name] = int(arr.shape[0])
    return sizes


def _add_blocks(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, FluxSparseMatrix):
        a = a.to_csr()
    if isinstance(b, FluxSparseMatrix):
        b = b.to_csr()
    if sp is not None and sp.issparse(a):
        if sp.issparse(b):
            return a + b
        return a + sp.csr_matrix(np.asarray(b))
    if sp is not None and sp.issparse(b):
        return sp.csr_matrix(np.asarray(a)) + b
    return np.asarray(a) + np.asarray(b)


def _transpose_block(block, rule: str):
    if isinstance(block, FluxSparseMatrix):
        if sp is None:
            raise ImportError("scipy is required to transpose FluxSparseMatrix blocks.")
        block = block.to_csr()
    if sp is not None and sp.issparse(block):
        out = block.T
    else:
        out = np.asarray(block).T
    if rule == "H":
        return out.conjugate()
    return out


def make(
    *,
    diag: Mapping[str, object] | Sequence[object],
    rel: Mapping[tuple[str, str], object] | None = None,
    add_contiguous: object | None = None,
    sizes: Mapping[str, int] | None = None,
    symmetric: bool = False,
    transpose_rule: str = "T",
):
    """
    Build a blocks dict from diagonal blocks, optional relations, and a full matrix.
    """
    if isinstance(diag, Mapping):
        diag_map = dict(diag)
    else:
        diag_seq = list(diag)
        if sizes is None:
            diag_map = dict(zip(range(len(diag_seq)), diag_seq))
        else:
            order = tuple(sizes.keys())
            if len(diag_seq) != len(order):
                raise ValueError("diag sequence length must match sizes")
            diag_map = dict(zip(order, diag_seq))

    if sizes is None:
        sizes = _infer_sizes_from_diag(diag_map)
    order = tuple(sizes.keys())

    if add_contiguous is None:
        blocks = {name: {} for name in order}
    else:
        blocks = split_block_matrix(add_contiguous, sizes=sizes)

    for name, blk in diag_map.items():
        if name not in sizes:
            raise KeyError(f"Unknown field '{name}' in diag")
        blocks.setdefault(name, {})
        blocks[name][name] = _add_blocks(blocks[name].get(name), blk)

    if transpose_rule not in {"T", "H", "none"}:
        raise ValueError("transpose_rule must be one of: T, H, none")

    if rel is not None:
        for (name_i, name_j), blk in rel.items():
            if name_i not in sizes or name_j not in sizes:
                raise KeyError(f"Unknown field in rel: {(name_i, name_j)}")
            blocks.setdefault(name_i, {})
            blocks[name_i][name_j] = _add_blocks(blocks[name_i].get(name_j), blk)
            if symmetric and name_i != name_j:
                if transpose_rule == "none":
                    blocks.setdefault(name_j, {})
                    blocks[name_j][name_i] = _add_blocks(blocks[name_j].get(name_i), blk)
                else:
                    blocks.setdefault(name_j, {})
                    blocks[name_j][name_i] = _add_blocks(
                        blocks[name_j].get(name_i),
                        _transpose_block(blk, transpose_rule),
                    )

    return blocks


__all__ = ["diag", "make"]

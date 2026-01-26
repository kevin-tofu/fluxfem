from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from typing import Any, Iterator, Mapping, Sequence, TypeAlias

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .block_system import split_block_matrix
from .sparse import FluxSparseMatrix

MatrixLike: TypeAlias = Any
FieldKey: TypeAlias = str | int
BlockMap: TypeAlias = dict[FieldKey, dict[FieldKey, MatrixLike]]


def diag(**blocks: MatrixLike) -> dict[str, MatrixLike]:
    return dict(blocks)


def _infer_sizes_from_diag(diag_blocks: Mapping[FieldKey, MatrixLike]) -> dict[FieldKey, int]:
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


def _infer_format(blocks: AbcMapping[FieldKey, AbcMapping[FieldKey, MatrixLike]], fmt: str) -> str:
    if fmt != "auto":
        return fmt
    for row in blocks.values():
        for blk in row.values():
            if isinstance(blk, FluxSparseMatrix):
                return "flux"
            if sp is not None and sp.issparse(blk):
                return "csr"
    return "dense"


def _add_blocks(a: MatrixLike | None, b: MatrixLike | None) -> MatrixLike | None:
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


def _transpose_block(block: MatrixLike, rule: str) -> MatrixLike:
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


class FluxBlockMatrix(AbcMapping[FieldKey, Mapping[FieldKey, MatrixLike]]):
    """
    Lazy block-matrix container that assembles on demand.
    """

    def __init__(
        self,
        blocks: BlockMap,
        *,
        sizes: Mapping[FieldKey, int],
        order: Sequence[FieldKey] | None = None,
        symmetric: bool = False,
        transpose_rule: str = "T",
    ) -> None:
        self._blocks = blocks
        self.sizes = {name: int(size) for name, size in sizes.items()}
        self.field_order = tuple(order) if order is not None else tuple(self.sizes.keys())
        self.symmetric = bool(symmetric)
        self.transpose_rule = transpose_rule
        for name in self.field_order:
            if name not in self.sizes:
                raise KeyError(f"Missing size for field '{name}'")

    def __getitem__(self, key: FieldKey) -> Mapping[FieldKey, MatrixLike]:
        return self._blocks[key]

    def __iter__(self) -> Iterator[FieldKey]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def blocks(self) -> BlockMap:
        return self._blocks

    def assemble(self, *, format: str = "flux") -> MatrixLike:
        if format not in {"auto", "flux", "csr", "dense"}:
            raise ValueError("format must be one of: auto, flux, csr, dense")
        use_format = _infer_format(self._blocks, format)

        offsets = {}
        offset = 0
        for name in self.field_order:
            size = int(self.sizes[name])
            offsets[name] = offset
            offset += size
        n_total = offset

        def _block_shape(name_i: FieldKey, name_j: FieldKey) -> tuple[int, int]:
            return (int(self.sizes[name_i]), int(self.sizes[name_j]))

        if use_format == "flux":
            rows_list = []
            cols_list = []
            data_list = []
            for name_i in self.field_order:
                row_blocks = self._blocks.get(name_i, {})
                for name_j in self.field_order:
                    blk = row_blocks.get(name_j)
                    if blk is None:
                        continue
                    shape = _block_shape(name_i, name_j)
                    if isinstance(blk, FluxSparseMatrix):
                        if shape[0] != shape[1] or int(blk.n_dofs) != shape[0]:
                            raise ValueError(f"Block {name_i},{name_j} has incompatible FluxSparseMatrix size")
                        r = np.asarray(blk.pattern.rows, dtype=np.int64)
                        c = np.asarray(blk.pattern.cols, dtype=np.int64)
                        d = np.asarray(blk.data)
                    elif sp is not None and sp.issparse(blk):
                        coo = blk.tocoo()
                        r = np.asarray(coo.row, dtype=np.int64)
                        c = np.asarray(coo.col, dtype=np.int64)
                        d = np.asarray(coo.data)
                        if coo.shape != shape:
                            raise ValueError(f"Block {name_i},{name_j} has shape {coo.shape}, expected {shape}")
                    else:
                        arr = np.asarray(blk)
                        if arr.shape != shape:
                            raise ValueError(f"Block {name_i},{name_j} has shape {arr.shape}, expected {shape}")
                        r, c = np.nonzero(arr)
                        d = arr[r, c]
                    if r.size:
                        rows_list.append(r + offsets[name_i])
                        cols_list.append(c + offsets[name_j])
                        data_list.append(d)
            rows = np.concatenate(rows_list) if rows_list else np.asarray([], dtype=np.int32)
            cols = np.concatenate(cols_list) if cols_list else np.asarray([], dtype=np.int32)
            data = np.concatenate(data_list) if data_list else np.asarray([], dtype=float)
            return FluxSparseMatrix(rows, cols, data, n_total)

        if use_format == "csr" and sp is None:
            raise ImportError("scipy is required for CSR block systems.")
        block_rows = []
        for name_i in self.field_order:
            row = []
            row_blocks = self._blocks.get(name_i, {})
            for name_j in self.field_order:
                blk = row_blocks.get(name_j)
                shape = _block_shape(name_i, name_j)
                if blk is None:
                    if use_format == "csr":
                        row.append(sp.csr_matrix(shape))
                    else:
                        row.append(np.zeros(shape, dtype=float))
                    continue
                if isinstance(blk, FluxSparseMatrix):
                    if sp is None:
                        raise ImportError("scipy is required to assemble sparse block systems.")
                    blk = blk.to_csr()
                if sp is not None and sp.issparse(blk):
                    blk = blk.tocsr()
                    if blk.shape != shape:
                        raise ValueError(f"Block {name_i},{name_j} has shape {blk.shape}, expected {shape}")
                    row.append(blk)
                else:
                    arr = np.asarray(blk)
                    if arr.shape != shape:
                        raise ValueError(f"Block {name_i},{name_j} has shape {arr.shape}, expected {shape}")
                    if use_format == "csr":
                        row.append(sp.csr_matrix(arr))
                    else:
                        row.append(arr)
            block_rows.append(row)
        if use_format == "csr":
            return sp.bmat(block_rows, format="csr")
        return np.block(block_rows)


def make(
    *,
    diag: Mapping[FieldKey, MatrixLike] | Sequence[MatrixLike],
    rel: Mapping[tuple[FieldKey, FieldKey], MatrixLike] | None = None,
    add_contiguous: MatrixLike | None = None,
    sizes: Mapping[FieldKey, int] | None = None,
    symmetric: bool = False,
    transpose_rule: str = "T",
) -> FluxBlockMatrix:
    """
    Build a lazy FluxBlockMatrix from diagonal blocks, optional relations, and a full matrix.
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

    return FluxBlockMatrix(
        blocks,
        sizes=sizes,
        order=order,
        symmetric=symmetric,
        transpose_rule=transpose_rule,
    )


__all__ = ["FluxBlockMatrix", "diag", "make"]

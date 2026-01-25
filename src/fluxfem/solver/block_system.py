from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .dirichlet import DirichletBC, free_dofs
from .sparse import FluxSparseMatrix

MatrixLike: TypeAlias = Any
FieldKey: TypeAlias = str | int
BlockMap: TypeAlias = Mapping[FieldKey, Mapping[FieldKey, MatrixLike]]


@dataclass(frozen=True)
class BlockSystem:
    K: MatrixLike
    F: np.ndarray
    free_dofs: np.ndarray
    dirichlet: DirichletBC
    field_order: tuple[FieldKey, ...]
    field_slices: dict[FieldKey, slice]

    def expand(self, u_free: np.ndarray) -> np.ndarray:
        return self.dirichlet.expand_solution(u_free, free=self.free_dofs, n_total=self.F.shape[0])

    def split(self, u_full: np.ndarray) -> dict[FieldKey, np.ndarray]:
        return {name: np.asarray(u_full)[self.field_slices[name]] for name in self.field_order}

    def join(self, fields: Mapping[FieldKey, np.ndarray]) -> np.ndarray:
        parts = []
        for name in self.field_order:
            if name not in fields:
                raise KeyError(f"Missing field '{name}' in join.")
            parts.append(np.asarray(fields[name]))
        return np.concatenate(parts, axis=0)


def _build_field_slices(
    order: Sequence[FieldKey], sizes: Mapping[FieldKey, int]
) -> tuple[dict[FieldKey, int], dict[FieldKey, slice], int]:
    offsets = {}
    slices = {}
    offset = 0
    for name in order:
        size = int(sizes[name])
        offsets[name] = offset
        slices[name] = slice(offset, offset + size)
        offset += size
    return offsets, slices, offset


def split_block_matrix(
    matrix: MatrixLike,
    *,
    sizes: Mapping[FieldKey, int],
    order: Sequence[FieldKey] | None = None,
) -> dict[FieldKey, dict[FieldKey, MatrixLike]]:
    """
    Split a block matrix into a dict-of-dicts by field order and sizes.
    """
    field_order = tuple(order) if order is not None else tuple(sizes.keys())
    for name in field_order:
        if name not in sizes:
            raise KeyError(f"Missing size for field '{name}'")
    offsets, _, n_total = _build_field_slices(field_order, sizes)

    if isinstance(matrix, FluxSparseMatrix):
        if sp is None:
            raise ImportError("scipy is required to split FluxSparseMatrix blocks.")
        mat = matrix.to_csr()
    elif sp is not None and sp.issparse(matrix):
        mat = matrix.tocsr()
    else:
        mat = np.asarray(matrix)

    if mat.shape != (n_total, n_total):
        raise ValueError(f"matrix has shape {mat.shape}, expected {(n_total, n_total)}")

    blocks: dict[FieldKey, dict[FieldKey, MatrixLike]] = {}
    for name_i in field_order:
        row = {}
        i0 = offsets[name_i]
        i1 = i0 + int(sizes[name_i])
        for name_j in field_order:
            j0 = offsets[name_j]
            j1 = j0 + int(sizes[name_j])
            row[name_j] = mat[i0:i1, j0:j1]
        blocks[name_i] = row
    return blocks


def _infer_format(blocks: BlockMap, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    for row in blocks.values():
        for blk in row.values():
            if isinstance(blk, FluxSparseMatrix):
                return "flux"
            if sp is not None and sp.issparse(blk):
                return "csr"
    return "dense"


def _infer_sizes_from_diag_seq(diag_seq: Sequence[MatrixLike]) -> dict[int, int]:
    sizes = {}
    for idx, blk in enumerate(diag_seq):
        if isinstance(blk, FluxSparseMatrix):
            sizes[idx] = int(blk.n_dofs)
        elif sp is not None and sp.issparse(blk):
            shape = blk.shape
            if shape[0] != shape[1]:
                raise ValueError(f"diag block {idx} must be square, got {shape}")
            sizes[idx] = int(shape[0])
        else:
            arr = np.asarray(blk)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(f"diag block {idx} must be square, got {arr.shape}")
            sizes[idx] = int(arr.shape[0])
    return sizes


def _coerce_rhs(
    rhs: MatrixLike | Sequence[MatrixLike] | Mapping[FieldKey, MatrixLike] | None,
    order: Sequence[FieldKey],
    sizes: Mapping[FieldKey, int],
) -> np.ndarray:
    if rhs is None:
        return np.zeros(sum(int(sizes[n]) for n in order), dtype=float)
    if isinstance(rhs, Mapping):
        parts = [np.asarray(rhs.get(name, np.zeros(int(sizes[name]), dtype=float))) for name in order]
        for name, part in zip(order, parts):
            if part.shape != (int(sizes[name]),):
                raise ValueError(f"rhs[{name}] has shape {part.shape}, expected {(int(sizes[name]),)}")
        return np.concatenate(parts, axis=0)
    if hasattr(rhs, "shape") and not isinstance(rhs, (list, tuple)):
        rhs_arr = np.asarray(rhs)
        if rhs_arr.shape != (sum(int(sizes[n]) for n in order),):
            raise ValueError("rhs vector has unexpected shape")
        return rhs_arr
    parts = list(rhs)
    if len(parts) != len(order):
        raise ValueError("rhs sequence length must match number of fields")
    parts = [np.asarray(p) for p in parts]
    for name, part in zip(order, parts):
        if part.shape != (int(sizes[name]),):
            raise ValueError(f"rhs for {name} has shape {part.shape}, expected {(int(sizes[name]),)}")
    return np.concatenate(parts, axis=0)


def _build_dirichlet_from_fields(
    fields: Mapping[FieldKey, object], offsets: Mapping[FieldKey, int], *, merge: str
) -> DirichletBC:
    if merge not in {"check_equal", "error", "first", "last"}:
        raise ValueError("merge must be one of: check_equal, error, first, last")
    dof_map: dict[int, float] = {}
    for name, spec in fields.items():
        if name not in offsets:
            raise KeyError(f"Unknown field '{name}' in constraints")
        offset = int(offsets[name])
        if isinstance(spec, DirichletBC):
            dofs = spec.dofs
            vals = spec.vals
        elif isinstance(spec, tuple) and len(spec) == 2:
            dofs, vals = spec
        else:
            dofs, vals = spec, None
        bc = DirichletBC(dofs, vals)
        g_dofs = np.asarray(bc.dofs, dtype=int) + offset
        g_vals = np.asarray(bc.vals, dtype=float)
        for d, v in zip(g_dofs, g_vals):
            if d in dof_map:
                if merge == "error":
                    raise ValueError(f"Duplicate Dirichlet DOF {d} in constraints")
                if merge == "check_equal":
                    if not np.isclose(dof_map[d], v):
                        raise ValueError(f"Conflicting Dirichlet value for DOF {d}")
                if merge == "first":
                    continue
            dof_map[d] = float(v)
    if not dof_map:
        return DirichletBC(np.array([], dtype=int), np.array([], dtype=float))
    dofs_sorted = np.array(sorted(dof_map.keys()), dtype=int)
    vals_sorted = np.array([dof_map[d] for d in dofs_sorted], dtype=float)
    return DirichletBC(dofs_sorted, vals_sorted)


def _build_dirichlet_from_sequence(
    seq: Sequence[object | None],
    order: Sequence[FieldKey],
    offsets: Mapping[FieldKey, int],
    *,
    merge: str,
) -> DirichletBC:
    if merge not in {"check_equal", "error", "first", "last"}:
        raise ValueError("merge must be one of: check_equal, error, first, last")
    if len(seq) != len(order):
        raise ValueError("constraints sequence length must match order")
    dof_map: dict[int, float] = {}
    for name, spec in zip(order, seq):
        if spec is None:
            continue
        offset = int(offsets[name])
        if isinstance(spec, DirichletBC):
            dofs = spec.dofs
            vals = spec.vals
        elif isinstance(spec, tuple) and len(spec) == 2:
            dofs, vals = spec
        else:
            dofs, vals = spec, None
        bc = DirichletBC(dofs, vals)
        g_dofs = np.asarray(bc.dofs, dtype=int) + offset
        g_vals = np.asarray(bc.vals, dtype=float)
        for d, v in zip(g_dofs, g_vals):
            if d in dof_map:
                if merge == "error":
                    raise ValueError(f"Duplicate Dirichlet DOF {d} in constraints")
                if merge == "check_equal":
                    if not np.isclose(dof_map[d], v):
                        raise ValueError(f"Conflicting Dirichlet value for DOF {d}")
                if merge == "first":
                    continue
            dof_map[d] = float(v)
    if not dof_map:
        return DirichletBC(np.array([], dtype=int), np.array([], dtype=float))
    dofs_sorted = np.array(sorted(dof_map.keys()), dtype=int)
    vals_sorted = np.array([dof_map[d] for d in dofs_sorted], dtype=float)
    return DirichletBC(dofs_sorted, vals_sorted)


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


def _blocks_from_diag_rel(
    *,
    diag: Mapping[FieldKey, MatrixLike] | Sequence[MatrixLike],
    sizes: Mapping[FieldKey, int],
    order: Sequence[FieldKey],
    rel: Mapping[tuple[FieldKey, FieldKey], MatrixLike] | None = None,
    add_contiguous: MatrixLike | None = None,
    symmetric: bool = False,
    transpose_rule: str = "T",
) -> BlockMap:
    if isinstance(diag, Mapping):
        diag_map = dict(diag)
    else:
        diag_seq = list(diag)
        if len(diag_seq) != len(order):
            raise ValueError("diag sequence length must match order")
        diag_map = dict(zip(order, diag_seq))

    if add_contiguous is None:
        blocks = {name: {} for name in order}
    else:
        blocks = split_block_matrix(add_contiguous, sizes=sizes, order=order)

    if transpose_rule not in {"T", "H", "none"}:
        raise ValueError("transpose_rule must be one of: T, H, none")

    for name, blk in diag_map.items():
        if name not in sizes:
            raise KeyError(f"Unknown field '{name}' in diag")
        blocks.setdefault(name, {})
        blocks[name][name] = _add_blocks(blocks[name].get(name), blk)

    if rel is not None:
        for (name_i, name_j), blk in rel.items():
            if name_i not in sizes or name_j not in sizes:
                raise KeyError(f"Unknown field in rel: {(name_i, name_j)}")
            blocks.setdefault(name_i, {})
            blocks[name_i][name_j] = _add_blocks(blocks[name_i].get(name_j), blk)
            if symmetric and name_i != name_j:
                blocks.setdefault(name_j, {})
                if transpose_rule == "none":
                    blocks[name_j][name_i] = _add_blocks(blocks[name_j].get(name_i), blk)
                else:
                    blocks[name_j][name_i] = _add_blocks(
                        blocks[name_j].get(name_i),
                        _transpose_block(blk, transpose_rule),
                    )

    return blocks


def build_block_system(
    *,
    diag: Mapping[FieldKey, MatrixLike] | Sequence[MatrixLike],
    sizes: Mapping[FieldKey, int] | None = None,
    rel: Mapping[tuple[FieldKey, FieldKey], MatrixLike] | None = None,
    add_contiguous: MatrixLike | None = None,
    rhs: Mapping[FieldKey, MatrixLike] | Sequence[MatrixLike] | np.ndarray | None = None,
    constraints: object | None = None,
    merge: str = "check_equal",
    format: str = "auto",
    symmetric: bool = False,
    transpose_rule: str = "T",
) -> BlockSystem:
    """
    Build a block system from diagonal blocks, optional relations, and constraints.

    format:
      - "auto": FluxSparseMatrix if any block is FluxSparseMatrix, CSR if any block is sparse, else dense
      - "flux": return FluxSparseMatrix
      - "csr": return scipy.sparse CSR
      - "dense": return numpy ndarray
    """
    if sizes is None:
        if isinstance(diag, Mapping):
            sizes = _infer_sizes_from_diag(diag)
            field_order = tuple(sizes.keys())
        else:
            sizes = _infer_sizes_from_diag_seq(diag)
            field_order = tuple(range(len(diag)))
    else:
        field_order = tuple(sizes.keys())
    offsets, field_slices, n_total = _build_field_slices(field_order, sizes)
    prefer_flux = False
    if format == "auto":
        if isinstance(add_contiguous, FluxSparseMatrix):
            prefer_flux = True
        if isinstance(diag, Mapping):
            prefer_flux = any(isinstance(blk, FluxSparseMatrix) for blk in diag.values())
        else:
            prefer_flux = any(isinstance(blk, FluxSparseMatrix) for blk in diag)
    blocks = _blocks_from_diag_rel(
        diag=diag,
        rel=rel,
        add_contiguous=add_contiguous,
        sizes=sizes,
        order=field_order,
        symmetric=symmetric,
        transpose_rule=transpose_rule,
    )
    use_format = "flux" if prefer_flux else _infer_format(blocks, format)

    def _block_shape(name_i, name_j):
        return (int(sizes[name_i]), int(sizes[name_j]))

    if use_format == "flux":
        rows_list = []
        cols_list = []
        data_list = []
        for name_i in field_order:
            row_blocks = blocks.get(name_i, {})
            for name_j in field_order:
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
        K = FluxSparseMatrix(rows, cols, data, n_total)
    else:
        if use_format == "csr" and sp is None:
            raise ImportError("scipy is required for CSR block systems.")
        block_rows = []
        for name_i in field_order:
            row = []
            row_blocks = blocks.get(name_i, {})
            for name_j in field_order:
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
            K = sp.bmat(block_rows, format="csr")
        else:
            K = np.block(block_rows)

    F = _coerce_rhs(rhs, field_order, sizes)

    if constraints is None:
        bc = DirichletBC(np.array([], dtype=int), np.array([], dtype=float))
        free = free_dofs(n_total, bc.dofs)
        return BlockSystem(K=K, F=F, free_dofs=free, dirichlet=bc, field_order=field_order, field_slices=field_slices)

    if isinstance(constraints, DirichletBC):
        bc = constraints
    elif isinstance(constraints, tuple) and len(constraints) == 2:
        bc = DirichletBC(constraints[0], constraints[1])
    elif isinstance(constraints, Mapping):
        bc = _build_dirichlet_from_fields(constraints, offsets, merge=merge)
    elif isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)):
        bc = _build_dirichlet_from_sequence(constraints, field_order, offsets, merge=merge)
    else:
        raise ValueError("constraints must be DirichletBC, (dofs, vals), or mapping")

    system = bc.condense_system(K, F)
    return BlockSystem(
        K=system.K,
        F=np.asarray(system.F),
        free_dofs=system.free_dofs,
        dirichlet=bc,
        field_order=field_order,
        field_slices=field_slices,
    )

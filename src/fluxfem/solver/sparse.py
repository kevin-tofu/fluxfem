from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TYPE_CHECKING, TypeAlias

import numpy as np
import jax
import jax.numpy as jnp

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


if TYPE_CHECKING:
    from jax import Array as JaxArray

    ArrayLike: TypeAlias = np.ndarray | JaxArray
else:
    ArrayLike: TypeAlias = np.ndarray
COOTuple: TypeAlias = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]


def coalesce_coo(
    rows: ArrayLike, cols: ArrayLike, data: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum duplicate COO entries by sorting (CPU-friendly).
    Returns (rows_u, cols_u, data_u) as NumPy arrays.
    """
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    d = np.asarray(data)
    if r.size == 0:
        return r, c, d
    order = np.lexsort((c, r))
    r_s = r[order]
    c_s = c[order]
    d_s = d[order]
    new_group = np.ones(r_s.size, dtype=bool)
    new_group[1:] = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1])
    starts = np.nonzero(new_group)[0]
    r_u = r_s[starts]
    c_u = c_s[starts]
    d_u = np.add.reduceat(d_s, starts)
    return r_u, c_u, d_u


def _normalize_flux_mats(mats: Sequence["FluxSparseMatrix"]) -> tuple["FluxSparseMatrix", ...]:
    if len(mats) == 1 and isinstance(mats[0], (list, tuple)):
        mats = tuple(mats[0])
    if not mats:
        raise ValueError("At least one FluxSparseMatrix is required.")
    return mats


def concat_flux(*mats: "FluxSparseMatrix", n_dofs: int | None = None) -> "FluxSparseMatrix":
    """
    Concatenate COO entries from multiple FluxSparseMatrix objects.
    All matrices must share the same n_dofs unless n_dofs is provided.
    """
    mats = _normalize_flux_mats(mats)
    if n_dofs is None:
        n_dofs = int(mats[0].n_dofs)
        for mat in mats[1:]:
            if int(mat.n_dofs) != n_dofs:
                raise ValueError("All matrices must share n_dofs for concat_flux.")
    rows_list = [np.asarray(mat.pattern.rows, dtype=np.int32) for mat in mats]
    cols_list = [np.asarray(mat.pattern.cols, dtype=np.int32) for mat in mats]
    data_list = [np.asarray(mat.data) for mat in mats]
    rows = np.concatenate(rows_list) if rows_list else np.asarray([], dtype=np.int32)
    cols = np.concatenate(cols_list) if cols_list else np.asarray([], dtype=np.int32)
    data = np.concatenate(data_list) if data_list else np.asarray([], dtype=float)
    return FluxSparseMatrix(rows, cols, data, int(n_dofs))


def block_diag_flux(*mats: "FluxSparseMatrix") -> "FluxSparseMatrix":
    """Block-diagonal concatenation for FluxSparseMatrix objects."""
    mats = _normalize_flux_mats(mats)
    rows_out = []
    cols_out = []
    data_out = []
    offset = 0
    for mat in mats:
        rows = np.asarray(mat.pattern.rows, dtype=np.int32)
        cols = np.asarray(mat.pattern.cols, dtype=np.int32)
        data = np.asarray(mat.data)
        if rows.size:
            rows_out.append(rows + offset)
            cols_out.append(cols + offset)
            data_out.append(data)
        offset += int(mat.n_dofs)
    rows = np.concatenate(rows_out) if rows_out else np.asarray([], dtype=np.int32)
    cols = np.concatenate(cols_out) if cols_out else np.asarray([], dtype=np.int32)
    data = np.concatenate(data_out) if data_out else np.asarray([], dtype=float)
    return FluxSparseMatrix(rows, cols, data, int(offset))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparsityPattern:
    """
    Jacobian sparsity pattern (rows/cols) that is independent of the solution.
    """

    rows: jnp.ndarray
    cols: jnp.ndarray
    n_dofs: int
    idx: jnp.ndarray | None = None
    diag_idx: jnp.ndarray | None = None
    perm: jnp.ndarray | None = None        # permutation mapping COO data -> CSR data
    indptr: jnp.ndarray | None = None      # CSR row pointer
    indices: jnp.ndarray | None = None     # CSR column indices

    def __post_init__(self):
        # Ensure n_dofs is always a Python int so JAX treats it as a static aux value.
        object.__setattr__(self, "n_dofs", int(self.n_dofs))

    def tree_flatten(self):
        children = (
            self.rows,
            self.cols,
            self.idx if self.idx is not None else jnp.array([], jnp.int32),
            self.diag_idx if self.diag_idx is not None else jnp.array([], jnp.int32),
            self.perm if self.perm is not None else jnp.array([], jnp.int32),
            self.indptr if self.indptr is not None else jnp.array([], jnp.int32),
            self.indices if self.indices is not None else jnp.array([], jnp.int32),
        )
        aux = {
            "n_dofs": self.n_dofs,
            "has_idx": self.idx is not None,
            "has_diag_idx": self.diag_idx is not None,
            "has_perm": self.perm is not None,
            "has_indptr": self.indptr is not None,
            "has_indices": self.indices is not None,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        rows, cols, idx, diag_idx, perm, indptr, indices = children
        idx_out = idx if aux["has_idx"] else None
        diag_out = diag_idx if aux["has_diag_idx"] else None
        perm_out = perm if aux["has_perm"] else None
        indptr_out = indptr if aux["has_indptr"] else None
        indices_out = indices if aux["has_indices"] else None
        return cls(
            rows=rows,
            cols=cols,
            n_dofs=aux["n_dofs"],
            idx=idx_out,
            diag_idx=diag_out,
            perm=perm_out,
            indptr=indptr_out,
            indices=indices_out,
        )


@jax.tree_util.register_pytree_node_class
class FluxSparseMatrix:
    """
    Sparse matrix wrapper (COO) with a fixed pattern and mutable values.
    - pattern stores rows/cols/n_dofs (optionally idx for dense scatter)
    - data stores the numeric values for the current nonlinear iterate
    """

    def __init__(
        self,
        rows_or_pattern: SparsityPattern | ArrayLike,
        cols: ArrayLike | None = None,
        data: ArrayLike | None = None,
        n_dofs: int | None = None,
        meta: dict | None = None,
    ):
        # New signature: FluxSparseMatrix(pattern, data)
        if isinstance(rows_or_pattern, SparsityPattern):
            pattern = rows_or_pattern
            values = cols if data is None else data
            values = jnp.asarray(values)
        else:
            # Legacy signature: FluxSparseMatrix(rows, cols, data, n_dofs)
            r_j = jnp.asarray(rows_or_pattern, dtype=jnp.int32)
            c_j = jnp.asarray(cols, dtype=jnp.int32)
            is_tracer = isinstance(rows_or_pattern, jax.core.Tracer) or isinstance(cols, jax.core.Tracer)
            diag_idx_j = None
            if not is_tracer:
                diag_idx_j = jnp.nonzero(r_j == c_j)[0].astype(jnp.int32)
            if n_dofs is None:
                if is_tracer:
                    raise ValueError("n_dofs must be provided when constructing FluxSparseMatrix under JIT.")
                n_dofs = int(np.asarray(cols).max()) + 1
            pattern = SparsityPattern(
                rows=r_j,
                cols=c_j,
                n_dofs=int(n_dofs) if n_dofs is not None else int(np.asarray(cols).max()) + 1,
                idx=None,
                diag_idx=diag_idx_j,
            )
            values = jnp.asarray(data)

        self.pattern = pattern
        self.rows = pattern.rows
        self.cols = pattern.cols
        self.n_dofs = int(pattern.n_dofs)
        self.data = values
        self.meta = dict(meta) if meta is not None else None

    @classmethod
    def from_bilinear(cls, coo_tuple: COOTuple) -> "FluxSparseMatrix":
        """Construct from assemble_bilinear_dense(..., sparse=True)."""
        rows, cols, data, n_dofs = coo_tuple
        return cls(rows, cols, data, n_dofs)

    @classmethod
    def from_linear(cls, coo_tuple: tuple[jnp.ndarray, jnp.ndarray, int]) -> "FluxSparseMatrix":
        """Construct from assemble_linear_form(..., sparse=True) (matrix interpretation only)."""
        rows, data, n_dofs = coo_tuple
        cols = jnp.zeros_like(rows)
        return cls(rows, cols, data, n_dofs)

    def with_data(self, data: ArrayLike) -> "FluxSparseMatrix":
        """Return a new FluxSparseMatrix sharing the same pattern with updated data."""
        return FluxSparseMatrix(self.pattern, data, meta=self.meta)

    def add_dense(self, dense: ArrayLike) -> "FluxSparseMatrix":
        """Return a new FluxSparseMatrix with dense entries added on the pattern."""
        dense_vals = jnp.asarray(dense)[self.pattern.rows, self.pattern.cols]
        return FluxSparseMatrix(self.pattern, self.data + dense_vals)

    def to_coo(self) -> COOTuple:
        return self.pattern.rows, self.pattern.cols, self.data, self.pattern.n_dofs

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    def coalesce(self) -> "FluxSparseMatrix":
        """Return a new FluxSparseMatrix with duplicate entries summed."""
        rows_u, cols_u, data_u = coalesce_coo(self.pattern.rows, self.pattern.cols, self.data)
        return FluxSparseMatrix(rows_u, cols_u, data_u, self.pattern.n_dofs)

    def to_csr(self):
        if sp is None:
            raise ImportError("scipy is required for to_csr()")
        if (
            self.pattern.indptr is not None
            and self.pattern.indices is not None
            and self.pattern.perm is not None
        ):
            indptr = np.array(self.pattern.indptr, dtype=np.int32, copy=True)
            indices = np.array(self.pattern.indices, dtype=np.int32, copy=True)
            data = np.array(self.data, copy=True)[np.asarray(self.pattern.perm, dtype=np.int32)]
            return sp.csr_matrix((data, indices, indptr), shape=(self.pattern.n_dofs, self.pattern.n_dofs))
        r = np.array(self.pattern.rows, dtype=np.int64, copy=True)
        c = np.array(self.pattern.cols, dtype=np.int64, copy=True)
        d = np.array(self.data, copy=True)
        return sp.csr_matrix((d, (r, c)), shape=(self.pattern.n_dofs, self.pattern.n_dofs))

    def to_dense(self) -> jnp.ndarray:
        # small debug helper
        dense = jnp.zeros((self.pattern.n_dofs, self.pattern.n_dofs), dtype=self.data.dtype)
        dense = dense.at[self.pattern.rows, self.pattern.cols].add(self.data)
        return dense

    def to_bcoo(self):
        """Construct jax.experimental.sparse.BCOO (requires jax.experimental.sparse)."""
        try:
            from jax.experimental import sparse as jsparse  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("jax.experimental.sparse is required for to_bcoo()") from exc
        idx = jnp.stack([self.pattern.rows, self.pattern.cols], axis=-1)
        return jsparse.BCOO((self.data, idx), shape=(self.pattern.n_dofs, self.pattern.n_dofs))

    def matvec(self, x: ArrayLike) -> jnp.ndarray:
        """Compute y = A x in JAX (iterative solvers)."""
        xj = jnp.asarray(x)
        contrib = self.data * xj[self.pattern.cols]
        # Use scatter_add to avoid tracing a dynamic int(x.max()) in jnp.bincount,
        # which triggers concretization errors under jit/while_loop.
        out = jnp.zeros(self.pattern.n_dofs, dtype=contrib.dtype)
        return out.at[self.pattern.rows].add(contrib)

    def as_cg_operator(
        self,
        *,
        matvec: str = "flux",
        preconditioner=None,
        solver: str = "cg",
        dof_per_node: int | None = None,
        block_sizes=None,
    ):
        from .cg import build_cg_operator

        return build_cg_operator(
            self,
            matvec=matvec,
            preconditioner=preconditioner,
            solver=solver,
            dof_per_node=dof_per_node,
            block_sizes=block_sizes,
        )

    def diag(self):
        """Diagonal entries aggregated for Jacobi preconditioning."""
        if self.pattern.diag_idx is not None:
            r = self.pattern.rows[self.pattern.diag_idx]
            d = self.data[self.pattern.diag_idx]
            return jax.ops.segment_sum(d, r, self.pattern.n_dofs)

        # Fallback for patterns without diag_idx (kept for backward compatibility).
        mask = self.pattern.rows == self.pattern.cols
        diag_contrib = jnp.where(mask, self.data, jnp.zeros_like(self.data))
        return jax.ops.segment_sum(diag_contrib, self.pattern.rows, self.pattern.n_dofs)

    def tree_flatten(self):
        return (self.pattern, self.data), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        pattern, data = children
        return cls(pattern, data)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


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

    def __init__(self, rows_or_pattern, cols=None, data=None, n_dofs: int | None = None):
        # New signature: FluxSparseMatrix(pattern, data)
        if isinstance(rows_or_pattern, SparsityPattern):
            pattern = rows_or_pattern
            values = cols if data is None else data
            values = jnp.asarray(values)
        else:
            # Legacy signature: FluxSparseMatrix(rows, cols, data, n_dofs)
            r_np = np.asarray(rows_or_pattern, dtype=np.int32)
            c_np = np.asarray(cols, dtype=np.int32)
            diag_idx_np = np.nonzero(r_np == c_np)[0].astype(np.int32)
            pattern = SparsityPattern(
                rows=jnp.asarray(r_np),
                cols=jnp.asarray(c_np),
                n_dofs=int(n_dofs) if n_dofs is not None else int(c_np.max()) + 1,
                idx=None,
                diag_idx=jnp.asarray(diag_idx_np),
            )
            values = jnp.asarray(data)

        self.pattern = pattern
        self.rows = pattern.rows
        self.cols = pattern.cols
        self.n_dofs = int(pattern.n_dofs)
        self.data = values

    @classmethod
    def from_bilinear(cls, coo_tuple):
        """Construct from assemble_bilinear_dense(..., sparse=True)."""
        rows, cols, data, n_dofs = coo_tuple
        return cls(rows, cols, data, n_dofs)

    @classmethod
    def from_linear(cls, coo_tuple):
        """Construct from assemble_linear_form(..., sparse=True) (matrix interpretation only)."""
        rows, data, n_dofs = coo_tuple
        cols = jnp.zeros_like(rows)
        return cls(rows, cols, data, n_dofs)

    def with_data(self, data):
        """Return a new FluxSparseMatrix sharing the same pattern with updated data."""
        return FluxSparseMatrix(self.pattern, data)

    def to_coo(self):
        return self.pattern.rows, self.pattern.cols, self.data, self.pattern.n_dofs

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

    def to_dense(self):
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

    def matvec(self, x):
        """Compute y = A x in JAX (iterative solvers)."""
        xj = jnp.asarray(x)
        contrib = self.data * xj[self.pattern.cols]
        # Use scatter_add to avoid tracing a dynamic int(x.max()) in jnp.bincount,
        # which triggers concretization errors under jit/while_loop.
        out = jnp.zeros(self.pattern.n_dofs, dtype=contrib.dtype)
        return out.at[self.pattern.rows].add(contrib)

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

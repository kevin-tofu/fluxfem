from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, TypeAlias

import numpy as np
import jax.numpy as jnp

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .sparse import FluxSparseMatrix, coalesce_coo

if TYPE_CHECKING:
    from jax import Array as JaxArray

    ArrayLike: TypeAlias = np.ndarray | JaxArray
else:
    ArrayLike: TypeAlias = np.ndarray
DirichletLike: TypeAlias = tuple[np.ndarray, np.ndarray]


def _normalize_dirichlet_values(dofs: ArrayLike, vals: ArrayLike | None) -> np.ndarray:
    if vals is None:
        return np.zeros(np.asarray(dofs).shape[0], dtype=float)
    arr = np.asarray(vals)
    if arr.ndim == 0:
        return np.full(np.asarray(dofs).shape[0], float(arr), dtype=float)
    return arr


def _normalize_dirichlet(dofs: ArrayLike, vals: ArrayLike | None) -> DirichletLike:
    dir_arr = np.asarray(dofs, dtype=int)
    return dir_arr, _normalize_dirichlet_values(dir_arr, vals)


@dataclass(frozen=True)
class CondensedSystem:
    K: Any
    F: Any
    free_dofs: np.ndarray
    dir_dofs: np.ndarray
    dir_vals: np.ndarray
    n_dofs: int

    def expand(self, u_free: ArrayLike, *, fill_dirichlet: bool = True) -> np.ndarray:
        u_full = np.zeros(self.n_dofs, dtype=np.asarray(u_free).dtype)
        u_full[self.free_dofs] = np.asarray(u_free)
        if fill_dirichlet and self.dir_dofs.size:
            u_full[self.dir_dofs] = np.asarray(self.dir_vals, dtype=u_full.dtype)
        return u_full




@dataclass(frozen=True)
class DirichletBC:
    """
    Dirichlet boundary condition container with helper methods.
    """
    dofs: np.ndarray
    vals: np.ndarray

    def __post_init__(self):
        dofs, vals = _normalize_dirichlet(self.dofs, self.vals)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "vals", vals)

    @classmethod
    def from_boundary_dofs(cls, mesh, predicate, *, values: ArrayLike | None = None, **kwargs) -> "DirichletBC":
        """
        Build from mesh.boundary_dofs_where predicate.

        kwargs are forwarded to mesh.boundary_dofs_where (e.g. components=..., dof_per_node=...).
        """
        dofs = mesh.boundary_dofs_where(predicate, **kwargs)
        vals = _normalize_dirichlet_values(dofs, values)
        return cls(dofs, vals)

    @classmethod
    def from_bbox(
        cls,
        mesh,
        *,
        mins: ArrayLike | None = None,
        maxs: ArrayLike | None = None,
        tol: float = 1e-8,
        values: ArrayLike | None = None,
        **kwargs,
    ) -> "DirichletBC":
        """
        Build from the mesh axis-aligned bounding box.

        mins/maxs default to mesh coordinate extrema. kwargs are forwarded to
        mesh.boundary_dofs_where (e.g. components=..., dof_per_node=...).
        """
        from ..mesh.predicate import bbox_predicate

        coords = np.asarray(mesh.coords)
        if mins is None:
            mins = coords.min(axis=0)
        if maxs is None:
            maxs = coords.max(axis=0)
        pred = bbox_predicate(mins, maxs, tol=tol)
        dofs = mesh.boundary_dofs_where(pred, **kwargs)
        vals = _normalize_dirichlet_values(dofs, values)
        return cls(dofs, vals)

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return self.dofs, self.vals

    def condense_system(self, A: Any, F: ArrayLike, *, check: bool = True) -> CondensedSystem:
        return condense_dirichlet_system(A, F, self.dofs, self.vals, check=check)

    def enforce_system(self, A: Any, F: ArrayLike):
        return enforce_dirichlet_system(A, F, self.dofs, self.vals)

    def condense_flux(self, A: FluxSparseMatrix, F: ArrayLike):
        """
        Condense for FluxSparseMatrix and return (K_free, F_free, free_dofs).
        """
        condensed = self.condense_system(A, F)
        free = condensed.free_dofs
        return restrict_flux_to_free(A, free), condensed.F, free

    def enforce_flux(self, A: FluxSparseMatrix, F: ArrayLike):
        return enforce_dirichlet_fluxsparse(A, F, self.dofs, self.vals)

    def split_matrix(self, A: Any, *, n_total: int | None = None):
        return split_dirichlet_matrix(A, self.dofs, n_total=n_total)

    def free_dofs(self, n_dofs: int) -> np.ndarray:
        return free_dofs(n_dofs, self.dofs)

    def expand_solution(
        self,
        u_free: ArrayLike,
        *,
        free: np.ndarray | None = None,
        n_total: int | None = None,
    ) -> np.ndarray:
        if free is None:
            if n_total is None:
                raise ValueError("n_total is required when free is not provided.")
            free = free_dofs(n_total, self.dofs)
        if n_total is None:
            max_free = int(np.max(free)) if len(free) else -1
            max_dir = int(np.max(self.dofs)) if len(self.dofs) else -1
            n_total = max(max_free, max_dir) + 1
        return expand_dirichlet_solution(u_free, free, self.dofs, self.vals, n_total=n_total)


def enforce_dirichlet_dense(K, F, dofs, vals):
    """Apply Dirichlet conditions directly to stiffness/load (dense)."""
    Kc = np.asarray(K, dtype=float).copy()
    Fc = np.asarray(F, dtype=float).copy()
    dofs, vals = _normalize_dirichlet(dofs, vals)
    if Fc.ndim == 2:
        Fc = Fc - (Kc[:, dofs] @ vals)[:, None]
    else:
        Fc = Fc - Kc[:, dofs] @ vals
    for d, v in zip(dofs, vals):
        Kc[d, :] = 0.0
        Kc[:, d] = 0.0
        Kc[d, d] = 1.0
        if Fc.ndim == 2:
            Fc[d, :] = v
        else:
            Fc[d] = v
    return Kc, Fc


def enforce_dirichlet_dense_jax(K, F, dofs, vals):
    """Apply Dirichlet conditions directly to stiffness/load (dense, JAX-friendly)."""
    import jax.numpy as jnp

    dofs, vals = _normalize_dirichlet(dofs, vals)
    dofs = jnp.asarray(dofs, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=F.dtype)
    if F.ndim == 2:
        F_mod = F - (K[:, dofs] @ vals)[:, None]
    else:
        F_mod = F - K[:, dofs] @ vals
    K_mod = K.at[:, dofs].set(0.0)
    K_mod = K_mod.at[dofs, :].set(0.0)
    K_mod = K_mod.at[dofs, dofs].set(1.0)
    if F.ndim == 2:
        F_mod = F_mod.at[dofs, :].set(vals[:, None])
    else:
        F_mod = F_mod.at[dofs].set(vals)
    return K_mod, F_mod


def enforce_dirichlet_system(A, F, dofs, vals):
    """
    Apply Dirichlet conditions directly to stiffness/load.
    Dispatches based on matrix type (FluxSparseMatrix, JAX dense, or numpy dense).
    """
    if isinstance(A, FluxSparseMatrix):
        return enforce_dirichlet_sparse(A, F, dofs, vals)
    try:
        import jax.numpy as jnp

        if isinstance(A, jnp.ndarray):
            return enforce_dirichlet_dense_jax(A, F, dofs, vals)
    except Exception:
        pass
    return enforce_dirichlet_dense(A, F, dofs, vals)


def split_dirichlet_matrix(A, dir_dofs, *, n_total: int | None = None):
    """
    Split a matrix into free-free and free-dirichlet blocks.

    Returns (free, dir_dofs, A_ff, A_fd).
    """
    dir_dofs, _ = _normalize_dirichlet(dir_dofs, None)
    if n_total is None:
        if hasattr(A, "n_dofs"):
            n_total = int(A.n_dofs)
        else:
            arr = np.asarray(A)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError("A must be square when n_total is not provided.")
            n_total = int(arr.shape[0])
    free = free_dofs(n_total, dir_dofs)

    if isinstance(A, FluxSparseMatrix):
        if sp is None:
            raise ImportError("scipy is required to split FluxSparseMatrix.")
        A = A.to_csr()

    if sp is not None and sp.issparse(A):
        return free, dir_dofs, A[free][:, free], A[free][:, dir_dofs]

    if isinstance(A, jnp.ndarray):
        free_j = jnp.asarray(free, dtype=jnp.int32)
        dir_j = jnp.asarray(dir_dofs, dtype=jnp.int32)
        return free, dir_dofs, A[jnp.ix_(free_j, free_j)], A[jnp.ix_(free_j, dir_j)]

    arr = np.asarray(A)
    return free, dir_dofs, arr[np.ix_(free, free)], arr[np.ix_(free, dir_dofs)]


def condense_dirichlet_system(A, F, dofs, vals, *, check: bool = True) -> CondensedSystem:
    """
    Condense Dirichlet DOFs and return a structured system.
    """
    dir_arr, dir_vals_arr = _normalize_dirichlet(dofs, vals)
    F_arr = np.asarray(F)
    if hasattr(A, "n_dofs"):
        n_total = int(A.n_dofs)
    else:
        A_np = np.asarray(A)
        if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
            raise ValueError("A must be square for Dirichlet condensation.")
        n_total = int(A_np.shape[0])

    if check:
        if dir_arr.size != dir_vals_arr.size:
            raise ValueError("dir_dofs and dir_vals must have the same length")
        if dir_arr.size:
            if np.min(dir_arr) < 0 or np.max(dir_arr) >= n_total:
                raise ValueError("dir_dofs out of bounds")
            if np.unique(dir_arr).size != dir_arr.size:
                raise ValueError("dir_dofs contains duplicates")

    mask = np.ones(n_total, dtype=bool)
    mask[dir_arr] = False
    free = np.nonzero(mask)[0]

    if isinstance(A, FluxSparseMatrix):
        K_csr = A.to_csr()
    elif sp is not None and sp.issparse(A):
        K_csr = A.tocsr()
    elif hasattr(A, "to_csr"):
        K_csr = A.to_csr()
    else:
        K_csr = np.asarray(A)

    K_ff = K_csr[free][:, free]
    F_free = F_arr[free]
    if dir_arr.size:
        K_fd = K_csr[free][:, dir_arr]
        if F_free.ndim == 2:
            F_free = F_free - (K_fd @ dir_vals_arr)[:, None]
        else:
            F_free = F_free - K_fd @ dir_vals_arr

    return CondensedSystem(
        K=K_ff,
        F=F_free,
        free_dofs=free,
        dir_dofs=dir_arr,
        dir_vals=dir_vals_arr,
        n_dofs=n_total,
    )




def enforce_dirichlet_sparse(A: FluxSparseMatrix, F, dofs, vals):
    """Apply Dirichlet conditions to FluxSparseMatrix + load (CSR)."""
    K_csr = A.to_csr().tolil()
    Fc = np.asarray(F, dtype=float).copy()
    dofs, vals = _normalize_dirichlet(dofs, vals)
    if Fc.ndim == 2:
        Fc = Fc - (K_csr[:, dofs] @ vals)[:, None]
    else:
        Fc = Fc - K_csr[:, dofs] @ vals
    for d, v in zip(dofs, vals):
        K_csr.rows[d] = [d]
        K_csr.data[d] = [1.0]
        K_csr[:, d] = 0.0
        K_csr[d, d] = 1.0
        if Fc.ndim == 2:
            Fc[d, :] = v
        else:
            Fc[d] = v
    return K_csr.tocsr(), Fc


def enforce_dirichlet_fluxsparse(A: FluxSparseMatrix, F, dofs, vals):
    """Alias for enforce_dirichlet_sparse for FluxSparseMatrix inputs."""
    return enforce_dirichlet_sparse(A, F, dofs, vals)


def condense_dirichlet_fluxsparse(A: FluxSparseMatrix, F, dofs, vals):
    """
    Condense Dirichlet DOFs for a FluxSparseMatrix.
    Returns: (K_ff, F_free, free_dofs, dir_dofs, dir_vals)
    """
    K_csr = A.to_csr()
    dir_arr, dir_vals_arr = _normalize_dirichlet(dofs, vals)
    mask = np.ones(K_csr.shape[0], dtype=bool)
    mask[dir_arr] = False
    free = np.nonzero(mask)[0]
    K_ff = K_csr[free][:, free]
    K_fd = K_csr[free][:, dir_arr] if dir_arr.size > 0 else None
    F_full = np.asarray(F, dtype=float)
    F_free = F_full[free]
    if K_fd is not None and dir_arr.size > 0:
        if F_free.ndim == 2:
            F_free = F_free - (K_fd @ dir_vals_arr)[:, None]
        else:
            F_free = F_free - K_fd @ dir_vals_arr
    return K_ff, F_free, free, dir_arr, dir_vals_arr


def condense_dirichlet_fluxsparse_coo(
    A: FluxSparseMatrix,
    F,
    dofs,
    vals,
    *,
    coalesce: bool = True,
):
    """
    Condense Dirichlet DOFs for a FluxSparseMatrix using COO filtering.
    Returns: (K_free, F_free, free_dofs, dir_dofs, dir_vals)
    """
    dir_arr, dir_vals_arr = _normalize_dirichlet(dofs, vals)
    n_total = int(A.n_dofs)
    mask = np.ones(n_total, dtype=bool)
    mask[dir_arr] = False
    free = np.nonzero(mask)[0]

    rows = np.asarray(A.pattern.rows, dtype=np.int64)
    cols = np.asarray(A.pattern.cols, dtype=np.int64)
    data = np.asarray(A.data)

    g2l = -np.ones(n_total, dtype=np.int32)
    g2l[free] = np.arange(free.size, dtype=np.int32)
    r2 = g2l[rows]
    c2 = g2l[cols]
    keep = (r2 >= 0) & (c2 >= 0)

    rows_f = r2[keep]
    cols_f = c2[keep]
    data_f = data[keep]
    if coalesce:
        rows_f, cols_f, data_f = coalesce_coo(rows_f, cols_f, data_f)

    K_free = FluxSparseMatrix(rows_f, cols_f, data_f, int(free.size))

    F_arr = np.asarray(F, dtype=float)
    F_free = F_arr[free]
    if dir_arr.size > 0 and not np.allclose(dir_vals_arr, 0.0):
        dir_full = np.zeros(n_total, dtype=F_arr.dtype)
        dir_full[dir_arr] = dir_vals_arr
        mask_fd = mask[rows] & (~mask[cols])
        if np.any(mask_fd):
            rows_fd = rows[mask_fd]
            cols_fd = cols[mask_fd]
            data_fd = data[mask_fd]
            contrib = data_fd * dir_full[cols_fd]
            delta = np.zeros(n_total, dtype=F_arr.dtype)
            np.add.at(delta, rows_fd, contrib)
            if F_free.ndim == 2:
                F_free = F_free - delta[free][:, None]
            else:
                F_free = F_free - delta[free]

    return K_free, jnp.asarray(F_free), free, dir_arr, dir_vals_arr


def free_dofs(n_dofs: int, dir_dofs) -> np.ndarray:
    """
    Return free DOF indices given total DOFs and Dirichlet DOFs.
    """
    dir_set = np.asarray(dir_dofs, dtype=int)
    mask = np.ones(int(n_dofs), dtype=bool)
    mask[dir_set] = False
    return np.nonzero(mask)[0]


def restrict_flux_to_free(K: FluxSparseMatrix, free: np.ndarray, *, coalesce: bool = True) -> FluxSparseMatrix:
    """
    Restrict a FluxSparseMatrix to free DOFs and return the condensed matrix.
    """
    free = np.asarray(free, dtype=np.int32)
    g2l = -np.ones(K.n_dofs, dtype=np.int32)
    g2l[free] = np.arange(free.size, dtype=np.int32)

    rows = np.asarray(K.pattern.rows)
    cols = np.asarray(K.pattern.cols)
    data = np.asarray(K.data)
    r2 = g2l[rows]
    c2 = g2l[cols]
    mask = (r2 >= 0) & (c2 >= 0)
    K_free = FluxSparseMatrix(
        jnp.asarray(r2[mask]),
        jnp.asarray(c2[mask]),
        jnp.asarray(data[mask]),
        int(free.size),
    )
    return K_free.coalesce() if coalesce else K_free


def condense_dirichlet_dense(K, F, dofs, vals):
    """
    Eliminate Dirichlet dofs for dense/CSR matrices and return condensed system.
    Returns: (K_cc, F_c, free_dofs, dir_dofs, dir_vals)
    """
    K_np = np.asarray(K, dtype=float)
    F_np = np.asarray(F, dtype=float)
    n = K_np.shape[0]

    dir_set, dir_vals = _normalize_dirichlet(dofs, vals)
    mask = np.ones(n, dtype=bool)
    mask[dir_set] = False
    free_dofs = np.nonzero(mask)[0]

    K_ff = K_np[np.ix_(free_dofs, free_dofs)]
    K_fd = K_np[np.ix_(free_dofs, dir_set)]
    F_f = F_np[free_dofs]
    if F_f.ndim == 2:
        F_f = F_f - (K_fd @ dir_vals)[:, None]
    else:
        F_f = F_f - K_fd @ dir_vals

    return K_ff, F_f, free_dofs, dir_set, dir_vals


def expand_dirichlet_solution(u_free, free_dofs, dir_dofs, dir_vals, n_total):
    """Expand condensed solution back to full vector."""
    dir_dofs, dir_vals = _normalize_dirichlet(dir_dofs, dir_vals)
    u_free_arr = np.asarray(u_free, dtype=float)
    if u_free_arr.ndim == 2:
        u = np.zeros((n_total, u_free_arr.shape[1]), dtype=float)
        u[free_dofs, :] = u_free_arr
        u[dir_dofs, :] = np.asarray(dir_vals, dtype=float)
    else:
        u = np.zeros(n_total, dtype=float)
        u[free_dofs] = u_free_arr
        u[dir_dofs] = np.asarray(dir_vals, dtype=float)
    return u

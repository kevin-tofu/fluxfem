from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .sparse import FluxSparseMatrix, coalesce_coo


def _normalize_dirichlet(dofs, vals):
    dir_arr = np.asarray(dofs, dtype=int)
    if vals is None:
        return dir_arr, np.zeros(dir_arr.shape[0], dtype=float)
    return dir_arr, np.asarray(vals, dtype=float)


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

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .sparse import FluxSparseMatrix


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


def free_dofs(n_dofs: int, dir_dofs) -> np.ndarray:
    """
    Return free DOF indices given total DOFs and Dirichlet DOFs.
    """
    dir_set = np.asarray(dir_dofs, dtype=int)
    mask = np.ones(int(n_dofs), dtype=bool)
    mask[dir_set] = False
    return np.nonzero(mask)[0]


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

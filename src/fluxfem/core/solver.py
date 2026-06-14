"""
Helper to bridge JAX-assembled matrices back to NumPy/SciPy and solve.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Any

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for spsolve utilities") from exc


def coo_to_csr(rows: Any, cols: Any, data: Any, n_dofs: int):
    """
    Convert COO triplets to SciPy CSR matrix.
    """
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    d = np.asarray(data)
    return sp.csr_matrix((d, (r, c)), shape=(n_dofs, n_dofs))




def spdirect_solve_cpu(K: Any, F: jnp.ndarray, *, use_jax: bool = False) -> np.ndarray:
    """
    Convert JAX arrays to NumPy/SciPy and solve K u = F with sparse solver.
    If use_jax=True, dispatch to JAX's experimental sparse spsolve.

    Parameters
    ----------
    K : jnp.ndarray
        Global stiffness matrix (n_dofs, n_dofs), dense or symmetric.
    F : jnp.ndarray
        Load vector (n_dofs,) or multiple RHS (n_dofs, n_rhs)

    Returns
    -------
    np.ndarray
        Solution vector u (n_dofs,) or (n_dofs, n_rhs)
    """
    if use_jax:
        try:
            return spdirect_solve_jax(K, F)
        except Exception:
            pass

    if hasattr(K, "to_csr"):
        K_csr = K.to_csr()
    elif isinstance(K, tuple) and len(K) == 4:
        K_csr = coo_to_csr(*K)
    elif sp.issparse(K):
        K_csr = K.tocsr()
    else:
        K_np = np.asarray(K)
        K_csr = sp.csr_matrix(K_np)

    F_np = np.asarray(F)
    u = spsolve(K_csr, F_np)
    return np.asarray(u)


def spdirect_solve_jax(K: Any, F: jnp.ndarray) -> np.ndarray:
    """
    Direct sparse solve in JAX via jax.experimental.sparse.linalg.spsolve.
    Accepts FluxSparseMatrix or jax.experimental.sparse.BCOO.
    """
    try:
        import jax
        if jax.default_backend() == "cpu":
            # JAX spsolve falls back to SciPy on CPU and can hit read-only buffers.
            return spdirect_solve_cpu(K, F, use_jax=False)
    except Exception:
        pass
    try:
        from jax.experimental.sparse.linalg import spsolve as jspsolve
        from jax.experimental import sparse as jsparse
    except Exception as exc:  # pragma: no cover
        raise ImportError("jax.experimental.sparse is required for spdirect_solve_jax") from exc

    if sp.issparse(K):
        data = jnp.asarray(K.data)
        indices = jnp.asarray(K.indices)
        indptr = jnp.asarray(K.indptr)
        F_arr = jnp.asarray(F)
        if F_arr.ndim == 1:
            return np.asarray(jspsolve(data, indices, indptr, F_arr))
        return np.asarray(jnp.stack([jspsolve(data, indices, indptr, F_arr[:, i]) for i in range(F_arr.shape[1])], axis=1))

    if isinstance(K, tuple) and len(K) == 4:
        rows, cols, data, n_dofs = K
        idx = jnp.stack([jnp.asarray(rows), jnp.asarray(cols)], axis=-1)
        bcoo = jsparse.BCOO((jnp.asarray(data), idx), shape=(int(n_dofs), int(n_dofs)))
    elif isinstance(K, jsparse.BCOO):
        bcoo = K
    elif hasattr(K, "to_bcoo"):
        bcoo = K.to_bcoo()
    else:
        raise TypeError("spdirect_solve_jax expects FluxSparseMatrix, BCOO, CSR, or COO tuple")

    bcsr = jsparse.BCSR.from_bcoo(bcoo)
    F_arr = jnp.asarray(F)
    if F_arr.ndim == 1:
        return np.asarray(jspsolve(bcsr.data, bcsr.indices, bcsr.indptr, F_arr))
    return np.asarray(jnp.stack([jspsolve(bcsr.data, bcsr.indices, bcsr.indptr, F_arr[:, i]) for i in range(F_arr.shape[1])], axis=1))

def spdirect_solve_gpu(K: Any, F: jnp.ndarray) -> np.ndarray:
    """
    GPU direct sparse solve via JAX experimental sparse solver.
    """
    return spdirect_solve_jax(K, F)

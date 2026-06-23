"""
Helper to bridge JAX-assembled matrices back to NumPy/SciPy and solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import jax.numpy as jnp

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


def _ensure_force_matrix_numpy(force_matrix: Any) -> np.ndarray:
    forces = np.asarray(force_matrix, dtype=float)
    if forces.ndim == 1:
        forces = forces.reshape(1, -1)
    if forces.ndim != 2:
        raise ValueError("force_matrix must be a 1-D force vector or a 2-D force matrix.")
    return forces


def pack_reduced_kkt_rhs(projected_force: Any, *, n_constraints: int = 0, hub_dofs: int = 0) -> np.ndarray:
    """Pack projected ROM forces into reduced KKT RHS rows.

    The structural reduced force occupies the leading columns.  Appended hub or
    constraint equations are homogeneous by default, so their RHS columns are
    zero-filled.
    """
    projected = np.asarray(projected_force, dtype=float)
    if projected.ndim == 1:
        projected = projected.reshape(1, -1)
    if projected.ndim != 2:
        raise ValueError("projected_force must be a 1-D vector or a 2-D matrix.")
    n_tail = int(hub_dofs) + int(n_constraints)
    if n_tail < 0:
        raise ValueError("hub_dofs + n_constraints must be non-negative.")
    if n_tail == 0:
        return projected
    zeros = np.zeros((projected.shape[0], n_tail), dtype=projected.dtype)
    return np.concatenate([projected, zeros], axis=1)


def project_reduced_rhs_cpu(
    force_matrix: Any,
    basis: Any,
    *,
    free_dofs: Any | None = None,
    n_constraints: int = 0,
    hub_dofs: int = 0,
) -> np.ndarray:
    """Project full-space force rows to reduced KKT RHS rows on CPU.

    Parameters
    ----------
    force_matrix
        Full force vector ``(n_full,)`` or force rows ``(n_cases, n_full)``.
    basis
        Reduction basis with shape ``(n_basis_rows, n_reduced)``.
    free_dofs
        Optional full-space DOF indices corresponding to the basis rows.  If
        omitted, ``force_matrix`` is assumed to already contain basis-row DOFs.
    """
    forces = _ensure_force_matrix_numpy(force_matrix)
    basis_np = np.asarray(basis, dtype=float)
    if basis_np.ndim != 2:
        raise ValueError("basis must be a 2-D array.")
    if free_dofs is None:
        force_free = forces
    else:
        free = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
        if free.size != basis_np.shape[0]:
            raise ValueError(f"free_dofs size {free.size} != basis rows {basis_np.shape[0]}.")
        force_free = forces[:, free]
    if force_free.shape[1] != basis_np.shape[0]:
        raise ValueError(f"force DOF count {force_free.shape[1]} != basis rows {basis_np.shape[0]}.")
    projected = force_free @ basis_np
    return pack_reduced_kkt_rhs(projected, n_constraints=n_constraints, hub_dofs=hub_dofs)


def make_reduced_rhs_projector_jax(
    basis: Any,
    *,
    free_dofs: Any | None = None,
    n_constraints: int = 0,
    hub_dofs: int = 0,
    jit: bool = True,
) -> Callable[[Any], Any]:
    """Create a JAX projector for ``f -> [Phi.T @ f, 0]``.

    The basis and optional free-DOF gather are captured on the active JAX device.
    With ``jit=True`` the returned callable is compiled on first use for the
    input rank/shape, which is the intended online path for repeated ROM RHS
    assembly on GPU.
    """
    import jax

    basis_j = jnp.asarray(basis, dtype=jnp.float64)
    if basis_j.ndim != 2:
        raise ValueError("basis must be a 2-D array.")
    free_j = None if free_dofs is None else jnp.asarray(np.asarray(free_dofs, dtype=np.int64), dtype=jnp.int32)
    if free_j is not None and int(free_j.size) != int(basis_j.shape[0]):
        raise ValueError(f"free_dofs size {int(free_j.size)} != basis rows {int(basis_j.shape[0])}.")
    n_tail = int(hub_dofs) + int(n_constraints)
    if n_tail < 0:
        raise ValueError("hub_dofs + n_constraints must be non-negative.")

    def project(force_matrix):
        forces = jnp.asarray(force_matrix, dtype=basis_j.dtype)
        if forces.ndim == 1:
            forces = forces[None, :]
        force_free = forces if free_j is None else forces[:, free_j]
        projected = force_free @ basis_j
        if n_tail == 0:
            return projected
        zeros = jnp.zeros((projected.shape[0], n_tail), dtype=projected.dtype)
        return jnp.concatenate([projected, zeros], axis=1)

    return jax.jit(project) if jit else project


def project_reduced_rhs_jax(
    force_matrix: Any,
    basis: Any,
    *,
    free_dofs: Any | None = None,
    n_constraints: int = 0,
    hub_dofs: int = 0,
    warmup: bool = True,
    return_device: bool = False,
) -> tuple[Any, dict[str, float | str]]:
    """Project ROM RHS with JAX and return timing metadata."""
    import jax

    projector = make_reduced_rhs_projector_jax(
        basis,
        free_dofs=free_dofs,
        n_constraints=n_constraints,
        hub_dofs=hub_dofs,
        jit=True,
    )
    warmup_seconds = 0.0
    if warmup:
        t_warm = time.perf_counter()
        rhs = projector(force_matrix)
        jax.block_until_ready(rhs)
        warmup_seconds = time.perf_counter() - t_warm
    t0 = time.perf_counter()
    rhs = projector(force_matrix)
    jax.block_until_ready(rhs)
    seconds = time.perf_counter() - t0
    timing = {
        "backend": f"jax-{jax.default_backend()}",
        "projection_seconds": float(seconds),
        "warmup_seconds": float(warmup_seconds),
    }
    return (rhs if return_device else np.asarray(rhs), timing)


@dataclass(frozen=True)
class ReducedDenseKktFactorization:
    """JAX LU factorization of a dense reduced KKT matrix."""

    lu: Any
    pivots: Any
    system_size: int
    backend: str


def factor_reduced_dense_kkt_jax(kkt: Any) -> ReducedDenseKktFactorization:
    """Factorize a dense reduced KKT matrix on the active JAX device."""
    import jax
    from jax.scipy.linalg import lu_factor

    kkt_j = jnp.asarray(kkt, dtype=jnp.float64)
    if kkt_j.ndim != 2 or kkt_j.shape[0] != kkt_j.shape[1]:
        raise ValueError("kkt must be a square 2-D matrix.")
    lu, pivots = lu_factor(kkt_j)
    jax.block_until_ready(lu)
    jax.block_until_ready(pivots)
    return ReducedDenseKktFactorization(
        lu=lu,
        pivots=pivots,
        system_size=int(kkt_j.shape[0]),
        backend=f"jax-{jax.default_backend()}",
    )


def solve_reduced_dense_batch_jax(kkt: Any, rhs_matrix: Any, basis: Any, *, n_reduced: int | None = None) -> np.ndarray:
    """Solve a dense reduced KKT system for multiple RHS with JAX.

    Parameters
    ----------
    kkt
        Dense reduced KKT matrix with shape ``(n_kkt, n_kkt)``.
    rhs_matrix
        RHS rows with shape ``(n_rhs, n_kkt)``.
    basis
        Lifting basis with shape ``(n_free, n_reduced)``.
    n_reduced
        Number of structural reduced coordinates.  Defaults to ``basis.shape[1]``.

    Returns
    -------
    np.ndarray
        Lifted free-DOF displacements with shape ``(n_rhs, n_free)``.
    """
    import jax

    kkt_j = jnp.asarray(kkt, dtype=jnp.float64)
    rhs_j = jnp.asarray(rhs_matrix, dtype=jnp.float64)
    basis_j = jnp.asarray(basis, dtype=jnp.float64)
    n_red = int(basis_j.shape[1] if n_reduced is None else n_reduced)
    q_all = jnp.linalg.solve(kkt_j, rhs_j.T)
    u_all = (basis_j @ q_all[:n_red, :]).T
    return np.asarray(jax.block_until_ready(u_all))


def solve_reduced_dense_batch_jax_factorized(
    factorization: ReducedDenseKktFactorization,
    rhs_matrix: Any,
    basis: Any,
    *,
    n_reduced: int | None = None,
    return_device: bool = False,
) -> Any:
    """Solve a dense reduced KKT batch using a reusable JAX LU factorization."""
    import jax
    from jax.scipy.linalg import lu_solve

    rhs_j = jnp.asarray(rhs_matrix, dtype=jnp.float64)
    if rhs_j.ndim == 1:
        rhs_j = rhs_j[None, :]
    if rhs_j.ndim != 2:
        raise ValueError("rhs_matrix must be a 1-D RHS vector or 2-D RHS row matrix.")
    if int(rhs_j.shape[1]) != int(factorization.system_size):
        raise ValueError(f"rhs width {int(rhs_j.shape[1])} != KKT size {int(factorization.system_size)}.")
    basis_j = jnp.asarray(basis, dtype=jnp.float64)
    n_red = int(basis_j.shape[1] if n_reduced is None else n_reduced)
    q_all = lu_solve((factorization.lu, factorization.pivots), rhs_j.T)
    u_all = (basis_j @ q_all[:n_red, :]).T
    jax.block_until_ready(u_all)
    return u_all if return_device else np.asarray(u_all)


def make_reduced_dense_factorized_solver_jax(
    kkt: Any,
    basis: Any,
    *,
    n_reduced: int | None = None,
    jit: bool = True,
) -> Callable[[Any], Any]:
    """Factorize a reduced dense KKT matrix and return a reusable JAX solver."""
    import jax
    from jax.scipy.linalg import lu_solve

    factorization = factor_reduced_dense_kkt_jax(kkt)
    basis_j = jnp.asarray(basis, dtype=jnp.float64)
    n_red = int(basis_j.shape[1] if n_reduced is None else n_reduced)

    def solve(rhs_matrix):
        rhs_j = jnp.asarray(rhs_matrix, dtype=jnp.float64)
        if rhs_j.ndim == 1:
            rhs_j = rhs_j[None, :]
        q_all = lu_solve((factorization.lu, factorization.pivots), rhs_j.T)
        return (basis_j @ q_all[:n_red, :]).T

    return jax.jit(solve) if jit else solve

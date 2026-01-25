from __future__ import annotations

from typing import Any, Callable, TypeAlias

import jax
import jax.numpy as jnp
import jax.scipy as jsp

try:
    from jax.experimental import sparse as jsparse
except Exception:  # pragma: no cover
    jsparse = None

from .sparse import FluxSparseMatrix
from dataclasses import dataclass

from .preconditioner import make_block_jacobi_preconditioner

ArrayLike: TypeAlias = jnp.ndarray
MatVec: TypeAlias = Callable[[jnp.ndarray], jnp.ndarray]
CGInfo: TypeAlias = dict[str, Any]


def _matvec_builder(A: Any) -> MatVec:
    if jsparse is not None and isinstance(A, jsparse.BCOO):
        return lambda x: A @ x
    if isinstance(A, FluxSparseMatrix):
        return A.matvec
    if hasattr(A, "matvec"):
        return A.matvec
    if callable(A):
        return lambda x: A(x)
    if isinstance(A, tuple) and len(A) == 4:
        return FluxSparseMatrix.from_bilinear(A).matvec

    def mv(x):
        return jnp.asarray(A) @ x

    return mv


def _coo_tuple_from_any(A: Any):
    if isinstance(A, FluxSparseMatrix):
        return A.to_coo()
    if isinstance(A, tuple) and len(A) == 4:
        return A
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:  # pragma: no cover
        sp = None
    if sp is not None and sp.issparse(A):
        coo = A.tocoo()
        return (
            jnp.asarray(coo.row, dtype=jnp.int32),
            jnp.asarray(coo.col, dtype=jnp.int32),
            jnp.asarray(coo.data),
            int(A.shape[0]),
        )
    return None


def _to_flux_matrix(A: Any) -> FluxSparseMatrix:
    if isinstance(A, FluxSparseMatrix):
        return A
    coo = _coo_tuple_from_any(A)
    if coo is None:
        raise ValueError("Unable to build FluxSparseMatrix from A")
    return FluxSparseMatrix.from_bilinear(coo)


def _to_bcoo_matrix(A: Any):
    if jsparse is None:
        raise ImportError("jax.experimental.sparse is required for BCOO matvec")
    if jsparse is not None and isinstance(A, jsparse.BCOO):
        return A
    coo = _coo_tuple_from_any(A)
    if coo is None:
        raise ValueError("Unable to build BCOO from A")
    rows, cols, data, n = coo
    idx = jnp.stack([rows, cols], axis=-1)
    return jsparse.BCOO((data, idx), shape=(n, n))


def _normalize_matvec_matrix(A: Any, matvec: str):
    if matvec == "flux":
        return _to_flux_matrix(A)
    if matvec == "bcoo":
        return _to_bcoo_matrix(A)
    if matvec == "dense":
        return jnp.asarray(A)
    if matvec == "auto":
        if jsparse is not None:
            try:
                return _to_bcoo_matrix(A)
            except Exception:
                return _to_flux_matrix(A)
        return _to_flux_matrix(A)
    raise ValueError(f"Unknown matvec backend: {matvec}")


@dataclass(frozen=True)
class CGOperator:
    """
    Lightweight CG operator wrapper with a consistent solve() entry point.
    """
    A: object
    preconditioner: object | None = None
    solver: str = "cg"

    def solve(
        self,
        b: jnp.ndarray,
        *,
        x0: jnp.ndarray | None = None,
        tol: float = 1e-8,
        maxiter: int | None = None,
    ):
        if self.solver == "cg":
            return cg_solve(
                self.A,
                b,
                x0=x0,
                tol=tol,
                maxiter=maxiter,
                preconditioner=self.preconditioner,
            )
        if self.solver == "cg_jax":
            return cg_solve_jax(
                self.A,
                b,
                x0=x0,
                tol=tol,
                maxiter=maxiter,
                preconditioner=self.preconditioner,
            )
        raise ValueError(f"Unknown CG solver: {self.solver}")


def build_cg_operator(
    A: Any,
    *,
    matvec: str = "flux",
    preconditioner: object | None = None,
    solver: str = "cg",
    dof_per_node: int | None = None,
    block_sizes: object | None = None,
) -> CGOperator:
    """
    Normalize CG inputs into a single operator interface.
    """
    A_mat = _normalize_matvec_matrix(A, matvec)
    precon = preconditioner
    if preconditioner == "block_jacobi":
        precon = make_block_jacobi_preconditioner(
            A_mat, dof_per_node=dof_per_node, block_sizes=block_sizes
        )
    return CGOperator(A=A_mat, preconditioner=precon, solver=solver)


def _diag_builder(A: Any, n: int) -> jnp.ndarray:
    """
    Build diagonal for a Jacobi preconditioner when available.
    """
    if jsparse is not None and isinstance(A, jsparse.BCOO):
        idx = A.indices
        data = A.data
        rows = idx[:, 0]
        cols = idx[:, 1]
        mask = rows == cols
        diag_contrib = jnp.where(mask, data, jnp.zeros_like(data))
        return jax.ops.segment_sum(diag_contrib, rows, n)
    if isinstance(A, FluxSparseMatrix):
        return A.diag()
    if hasattr(A, "diagonal"):
        return jnp.asarray(A.diagonal())
    if isinstance(A, tuple) and len(A) == 4:
        return _diag_builder(FluxSparseMatrix.from_bilinear(A), n)
    if callable(A):
        raise ValueError("Jacobi preconditioner requires access to matrix diagonal")
    arr = jnp.asarray(A)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return jnp.diag(arr)
    raise ValueError("Cannot build Jacobi preconditioner: diagonal unavailable")


def _cg_solve_single(
    A: Any,
    b: jnp.ndarray,
    *,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: object | None = None,
) -> tuple[jnp.ndarray, CGInfo]:
    """
    Conjugate gradient (Ax=b) in JAX.
    A: FluxSparseMatrix / (rows, cols, data, n) / dense array
    b: RHS (jnp or np)
    preconditioner: None | "jacobi" | callable(r) -> z
    returns: (x, info dict)
    """
    b = jnp.asarray(b)
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if maxiter is None:
        maxiter = max(10 * n, 1)

    mv = _matvec_builder(A)
    precon = None

    if preconditioner is None:
        pass
    elif preconditioner == "jacobi":
        diag = _diag_builder(A, n)
        inv_diag = jnp.where(diag != 0.0, 1.0 / diag, 0.0)

        def precon(r):
            return inv_diag * r

    elif preconditioner == "block_jacobi":
        precon = make_block_jacobi_preconditioner(A)

    elif callable(preconditioner):
        precon = preconditioner
    else:
        raise ValueError(f"Unknown preconditioner type: {preconditioner}")

    def body_fun(state):
        k, x, r, p, rz_old = state
        Ap = mv(p)
        alpha = rz_old / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = r_new if precon is None else precon(r_new)
        rz_new = jnp.dot(r_new, z_new)
        beta = rz_new / rz_old
        p_new = z_new + beta * p
        return k + 1, x_new, r_new, p_new, rz_new

    r0 = b - mv(x0)
    z0 = r0 if precon is None else precon(r0)
    rz0 = jnp.dot(r0, z0)
    init_state = (0, x0, r0, z0, rz0)

    def cond_fun(state):
        k, x, r, p, rz_old = state
        return jnp.logical_and(k < maxiter, jnp.dot(r, r) > tol * tol)

    k, x, r, p, rz_old = jax.lax.while_loop(cond_fun, body_fun, init_state)
    res_norm = jnp.sqrt(jnp.dot(r, r))
    info = {"iters": k, "residual_norm": res_norm}
    return x, info


def cg_solve(
    A: Any,
    b: jnp.ndarray,
    *,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: object | None = None,
) -> tuple[jnp.ndarray, CGInfo]:
    """
    Conjugate gradient (Ax=b) in JAX.
    Supports single RHS (n,) or multiple RHS (n, n_rhs).
    """
    b_arr = jnp.asarray(b)
    if b_arr.ndim == 1:
        return _cg_solve_single(
            A,
            b_arr,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            preconditioner=preconditioner,
        )

    if b_arr.ndim != 2:
        raise ValueError("cg_solve expects b with shape (n,) or (n, n_rhs).")

    xs = []
    infos = []
    for i in range(b_arr.shape[1]):
        x0_i = None if x0 is None else jnp.asarray(x0)[:, i]
        x_i, info_i = _cg_solve_single(
            A,
            b_arr[:, i],
            x0=x0_i,
            tol=tol,
            maxiter=maxiter,
            preconditioner=preconditioner,
        )
        xs.append(x_i)
        infos.append(info_i)
    x_out = jnp.stack(xs, axis=1)
    info = {
        "iters": [int(i.get("iters", 0)) for i in infos],
        "residual_norm": jnp.asarray([i.get("residual_norm", 0.0) for i in infos]),
    }
    return x_out, info


def _cg_solve_jax_single(
    A: Any,
    b: jnp.ndarray,
    *,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: object | None = None,
) -> tuple[jnp.ndarray, CGInfo]:
    """
    Conjugate gradient via jax.scipy.sparse.linalg.cg.
    A: FluxSparseMatrix / (rows, cols, data, n) / dense array / callable
    preconditioner: None | "jacobi" | callable(r) -> z
    returns: (x, info dict)
    """
    b = jnp.asarray(b)
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if maxiter is None:
        maxiter = max(10 * n, 1)

    mv = _matvec_builder(A)

    precon = None
    if preconditioner is None:
        pass
    elif preconditioner == "jacobi":
        diag = _diag_builder(A, n)
        inv_diag = jnp.where(diag != 0.0, 1.0 / diag, 0.0)

        def precon(r):
            return inv_diag * r

    elif callable(preconditioner):
        precon = preconditioner
    else:
        raise ValueError(f"Unknown preconditioner type: {preconditioner}")

    def M_fun(x):
        return precon(x) if precon is not None else x

    x, info_val = jsp.sparse.linalg.cg(
        mv,
        b,
        x0=x0,
        tol=tol,
        atol=0.0,
        maxiter=maxiter,
        M=M_fun if precon is not None else None,
    )
    res = b - mv(x)
    res_norm = jnp.sqrt(jnp.dot(res, res))

    if isinstance(info_val, (int, jnp.integer)):
        iters = int(info_val)
        converged = iters == 0
    elif info_val is None:
        # jax.scipy may return None on success; treat as converged with unknown iter count.
        iters = 0
        converged = True
    else:
        # Unknown type; fall back to "not sure" but keep running.
        iters = 0
        converged = True
    info = {"iters": iters, "residual_norm": res_norm, "converged": converged, "info": info_val}
    return x, info


def cg_solve_jax(
    A: Any,
    b: jnp.ndarray,
    *,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: object | None = None,
) -> tuple[jnp.ndarray, CGInfo]:
    """
    Conjugate gradient via jax.scipy.sparse.linalg.cg.
    Supports single RHS (n,) or multiple RHS (n, n_rhs).
    """
    b_arr = jnp.asarray(b)
    if b_arr.ndim == 1:
        return _cg_solve_jax_single(
            A,
            b_arr,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            preconditioner=preconditioner,
        )

    if b_arr.ndim != 2:
        raise ValueError("cg_solve_jax expects b with shape (n,) or (n, n_rhs).")

    xs = []
    infos = []
    for i in range(b_arr.shape[1]):
        x0_i = None if x0 is None else jnp.asarray(x0)[:, i]
        x_i, info_i = _cg_solve_jax_single(
            A,
            b_arr[:, i],
            x0=x0_i,
            tol=tol,
            maxiter=maxiter,
            preconditioner=preconditioner,
        )
        xs.append(x_i)
        infos.append(info_i)
    x_out = jnp.stack(xs, axis=1)
    info = {
        "iters": [int(i.get("iters", 0)) for i in infos],
        "residual_norm": jnp.asarray([i.get("residual_norm", 0.0) for i in infos]),
        "converged": [bool(i.get("converged", True)) for i in infos],
    }
    return x_out, info

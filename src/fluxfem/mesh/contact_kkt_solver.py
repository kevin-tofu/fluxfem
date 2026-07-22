from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ContactKKTSolveInfo:
    """Diagnostics for a linear contact KKT solve."""

    backend: str
    solver: str
    residual_norm: float
    relative_residual_norm: float
    n_primal: int | None = None
    primal_scaling_min: float | None = None
    primal_scaling_max: float | None = None
    dual_scaling_min: float | None = None
    dual_scaling_max: float | None = None
    matrix_row_norm_min: float | None = None
    matrix_row_norm_max: float | None = None
    matrix_col_norm_min: float | None = None
    matrix_col_norm_max: float | None = None
    scaled_residual_norm: float | None = None
    scaled_relative_residual_norm: float | None = None
    scaled_matrix_row_norm_min: float | None = None
    scaled_matrix_row_norm_max: float | None = None
    scaled_matrix_col_norm_min: float | None = None
    scaled_matrix_col_norm_max: float | None = None


@dataclass(frozen=True)
class ContactKKTSolveResult:
    """Solution and diagnostics for a linear contact KKT solve."""

    solution: Any
    info: ContactKKTSolveInfo


@dataclass(frozen=True)
class ContactKKTSolveConfig:
    """Linear solve configuration for ``solve_contact_kkt``."""

    backend: str = "numpy"
    diagonal_shift: float = 0.0
    allow_dense_fallback: bool = True
    numpy_solver: str = "direct"  # "direct" | "block_scaled"
    n_primal: int | None = None
    scaling_floor: float = 1e-30
    jax_solver: str = "gmres"
    jax_tol: float = 1e-8
    jax_atol: float = 0.0
    jax_restart: int = 20
    jax_maxiter: int | None = None
    # Dense inputs default to the direct solve path for more stable autodiff.
    jax_dense_mode: str = "direct_custom_vjp"  # "iterative" | "direct_custom_vjp"
    petsc_ksp_type: str = "gmres"
    petsc_pc_type: str = "none"
    petsc_preconditioner: str | None = "diag0"
    petsc_rtol: float | None = 1e-10
    petsc_atol: float | None = None
    petsc_max_it: int | None = None
    petsc_options: Mapping[str, Any] | None = None
    petsc_options_prefix: str | None = "contact_kkt_"

    def validate(self) -> "ContactKKTSolveConfig":
        backend = str(self.backend).lower()
        if backend not in {"numpy", "jax", "petsc4py"}:
            raise ValueError("backend must be 'numpy', 'petsc4py', or 'jax'.")
        if self.numpy_solver not in {"direct", "block_scaled"}:
            raise ValueError("numpy_solver must be 'direct' or 'block_scaled'.")
        if self.n_primal is not None and int(self.n_primal) <= 0:
            raise ValueError("n_primal must be positive when provided.")
        if float(self.scaling_floor) <= 0.0:
            raise ValueError("scaling_floor must be positive.")
        if self.jax_solver not in {"gmres", "spsolve"}:
            raise ValueError("jax_solver must be 'gmres' or 'spsolve'.")
        if self.jax_dense_mode not in {"iterative", "direct_custom_vjp"}:
            raise ValueError("jax_dense_mode must be 'iterative' or 'direct_custom_vjp'.")
        if int(self.jax_restart) <= 0:
            raise ValueError("jax_restart must be positive.")
        if self.jax_solver == "spsolve" and float(self.diagonal_shift) != 0.0:
            raise ValueError("jax_solver='spsolve' currently requires diagonal_shift == 0.")
        return self


def _is_jax_like(x: Any) -> bool:
    try:
        import jax
    except Exception:
        return False
    return isinstance(x, jax.Array) or isinstance(x, jax.core.Tracer)


def _contains_jax_value(obj: Any) -> bool:
    if _is_jax_like(obj):
        return True
    if isinstance(obj, np.ndarray):
        return False
    if isinstance(obj, Mapping):
        return any(_contains_jax_value(v) for v in obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return any(_contains_jax_value(v) for v in obj)
    data = getattr(obj, "data", None)
    if data is not None and not isinstance(obj, np.ndarray) and data is not obj and _contains_jax_value(data):
        return True
    return False


def _infer_backend(*values: Any, default: str) -> str:
    return "jax" if any(_contains_jax_value(v) for v in values) else default


def _resolve_kkt_solve_config(
    *,
    backend: str | None,
    diagonal_shift: float,
    config: ContactKKTSolveConfig | None,
    kkt_matrix: Any | None = None,
    rhs: Any | None = None,
) -> ContactKKTSolveConfig:
    if config is None:
        if backend is None:
            backend = _infer_backend(kkt_matrix, rhs, default="numpy")
        return ContactKKTSolveConfig(backend=backend, diagonal_shift=diagonal_shift).validate()
    return config.validate()


def _as_numpy_dense(kkt_matrix) -> np.ndarray:
    return np.asarray(kkt_matrix.to_dense(), dtype=float) if hasattr(kkt_matrix, "to_dense") else np.asarray(kkt_matrix, dtype=float)


def _as_numpy_csr(kkt_matrix):
    try:
        import scipy.sparse as sp
    except Exception:
        return None
    if hasattr(kkt_matrix, "to_csr"):
        return kkt_matrix.to_csr()
    if sp.issparse(kkt_matrix):
        return kkt_matrix.tocsr()
    return sp.csr_matrix(_as_numpy_dense(kkt_matrix))


def _apply_numpy_diagonal_shift(A, diagonal_shift: float):
    shift = float(diagonal_shift)
    if shift == 0.0:
        return A
    try:
        import scipy.sparse as sp
    except Exception:
        sp = None
    if sp is not None and sp.issparse(A):
        return A + shift * sp.eye(A.shape[0], format="csr")
    A_np = np.asarray(A, dtype=float)
    return A_np + shift * np.eye(A_np.shape[0], dtype=float)


def _norm_range(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float).reshape(-1)
    if vals.size == 0:
        return 0.0, 0.0
    return float(np.min(vals)), float(np.max(vals))


def _matrix_axis_norm_ranges(A) -> tuple[float, float, float, float]:
    try:
        import scipy.sparse as sp
    except Exception:
        sp = None
    if sp is not None and sp.issparse(A):
        A_csr = A.tocsr()
        row_norms = np.sqrt(np.asarray(A_csr.multiply(A_csr).sum(axis=1)).reshape(-1))
        col_norms = np.sqrt(np.asarray(A_csr.multiply(A_csr).sum(axis=0)).reshape(-1))
    else:
        A_np = np.asarray(A, dtype=float)
        row_norms = np.linalg.norm(A_np, axis=1) if A_np.ndim == 2 else np.zeros((0,), dtype=float)
        col_norms = np.linalg.norm(A_np, axis=0) if A_np.ndim == 2 else np.zeros((0,), dtype=float)
    row_min, row_max = _norm_range(row_norms)
    col_min, col_max = _norm_range(col_norms)
    return row_min, row_max, col_min, col_max


def _matrix_vector_product_numpy(A, x: np.ndarray) -> np.ndarray:
    return np.asarray(A @ np.asarray(x, dtype=float), dtype=float)


def _relative_norm(numer: np.ndarray, denom: np.ndarray) -> float:
    n = float(np.linalg.norm(np.asarray(numer, dtype=float)))
    d = float(np.linalg.norm(np.asarray(denom, dtype=float)))
    return n / max(d, 1.0)


def _as_jax_linear_op(kkt_matrix):
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse  # type: ignore

    is_fluxsparse = hasattr(kkt_matrix, "matvec") and hasattr(kkt_matrix, "n_dofs")
    is_bcoo = isinstance(kkt_matrix, jsparse.BCOO)
    if is_fluxsparse:
        return (lambda x: kkt_matrix.matvec(x)), True
    if is_bcoo:
        return (lambda x: kkt_matrix @ x), True
    A = jnp.asarray(kkt_matrix.to_dense()) if hasattr(kkt_matrix, "to_dense") else jnp.asarray(kkt_matrix)
    return (lambda x: A @ x), False


def _solve_kkt_petsc(kkt_matrix, rhs, cfg: ContactKKTSolveConfig):
    from ..solver.petsc import petsc_shell_solve

    A_petsc = _as_numpy_csr(kkt_matrix)
    if A_petsc is None:
        A_petsc = _as_numpy_dense(kkt_matrix)
    if float(cfg.diagonal_shift) != 0.0:
        try:
            import scipy.sparse as sp
        except Exception:
            sp = None
        if sp is not None and hasattr(A_petsc, "tocsr"):
            A_petsc = A_petsc.tocsr() + float(cfg.diagonal_shift) * sp.eye(A_petsc.shape[0], format="csr")
        else:
            A_np = np.asarray(A_petsc, dtype=float)
            A_petsc = A_np + float(cfg.diagonal_shift) * np.eye(A_np.shape[0], dtype=A_np.dtype)

    rhs_np = np.asarray(rhs, dtype=float)
    n = int(rhs_np.shape[0])
    return petsc_shell_solve(
        A_petsc,
        rhs_np,
        n_dofs=n,
        ksp_type=str(cfg.petsc_ksp_type),
        pc_type=str(cfg.petsc_pc_type),
        preconditioner=cfg.petsc_preconditioner,
        pmat=A_petsc,
        rtol=cfg.petsc_rtol,
        atol=cfg.petsc_atol,
        max_it=cfg.petsc_max_it if cfg.petsc_max_it is not None else max(10 * n, 200),
        options=None if cfg.petsc_options is None else dict(cfg.petsc_options),
        options_prefix=cfg.petsc_options_prefix,
    )


def _kkt_block_scaling(A, *, n_primal: int, scaling_floor: float):
    try:
        import scipy.sparse as sp
    except Exception:
        sp = None

    n = int(A.shape[0])
    n_u = int(n_primal)
    if n_u <= 0 or n_u >= n:
        raise ValueError("n_primal must split a KKT matrix into non-empty primal and dual blocks.")

    floor = float(scaling_floor)
    if sp is not None and sp.issparse(A):
        A_csr = A.tocsr()
        primal_diag = np.abs(A_csr.diagonal()[:n_u])
        d_u = 1.0 / np.sqrt(np.maximum(primal_diag, floor))
        B = A_csr[n_u:, :n_u]
        row_norms = np.sqrt(np.asarray(B.multiply(d_u).power(2).sum(axis=1)).reshape(-1))
        d_l = 1.0 / np.maximum(row_norms, floor)
        d = np.concatenate([d_u, d_l])
        D = sp.diags(d, format="csr")
        return D @ A_csr @ D, d

    A_np = np.asarray(A, dtype=float)
    primal_diag = np.abs(np.diag(A_np)[:n_u])
    d_u = 1.0 / np.sqrt(np.maximum(primal_diag, floor))
    B = A_np[n_u:, :n_u]
    row_norms = np.linalg.norm(B * d_u[None, :], axis=1)
    d_l = 1.0 / np.maximum(row_norms, floor)
    d = np.concatenate([d_u, d_l])
    return d[:, None] * A_np * d[None, :], d


def _solve_kkt_numpy_direct(A, rhs, cfg: ContactKKTSolveConfig):
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception:
        sp = None
        spla = None

    if sp is not None and spla is not None and sp.issparse(A):
        return np.asarray(spla.spsolve(A.tocsr(), np.asarray(rhs, dtype=float)))

    if not bool(cfg.allow_dense_fallback):
        raise ValueError("Dense fallback is disabled by ContactKKTSolveConfig.allow_dense_fallback.")
    return np.linalg.solve(np.asarray(A, dtype=float), np.asarray(rhs, dtype=float))


def _solve_kkt_numpy(kkt_matrix, rhs, cfg: ContactKKTSolveConfig):
    A_csr = _as_numpy_csr(kkt_matrix)
    A = A_csr if A_csr is not None else _as_numpy_dense(kkt_matrix)
    A = _apply_numpy_diagonal_shift(A, float(cfg.diagonal_shift))

    if cfg.numpy_solver == "block_scaled":
        if cfg.n_primal is None:
            raise ValueError("numpy_solver='block_scaled' requires n_primal.")
        A_scaled, d = _kkt_block_scaling(
            A,
            n_primal=int(cfg.n_primal),
            scaling_floor=float(cfg.scaling_floor),
        )
        y = _solve_kkt_numpy_direct(A_scaled, d * np.asarray(rhs, dtype=float), cfg)
        return d * np.asarray(y, dtype=float)

    return _solve_kkt_numpy_direct(A, rhs, cfg)


def _solve_kkt_numpy_with_info(kkt_matrix, rhs, cfg: ContactKKTSolveConfig) -> ContactKKTSolveResult:
    A_csr = _as_numpy_csr(kkt_matrix)
    A = A_csr if A_csr is not None else _as_numpy_dense(kkt_matrix)
    A = _apply_numpy_diagonal_shift(A, float(cfg.diagonal_shift))
    rhs_np = np.asarray(rhs, dtype=float)
    row_min, row_max, col_min, col_max = _matrix_axis_norm_ranges(A)

    if cfg.numpy_solver == "block_scaled":
        if cfg.n_primal is None:
            raise ValueError("numpy_solver='block_scaled' requires n_primal.")
        n_primal = int(cfg.n_primal)
        A_scaled, d = _kkt_block_scaling(
            A,
            n_primal=n_primal,
            scaling_floor=float(cfg.scaling_floor),
        )
        rhs_scaled = d * rhs_np
        y = _solve_kkt_numpy_direct(A_scaled, rhs_scaled, cfg)
        solution = d * np.asarray(y, dtype=float)
        residual = _matrix_vector_product_numpy(A, solution) - rhs_np
        scaled_residual = _matrix_vector_product_numpy(A_scaled, np.asarray(y, dtype=float)) - rhs_scaled
        srow_min, srow_max, scol_min, scol_max = _matrix_axis_norm_ranges(A_scaled)
        d_u = np.asarray(d[:n_primal], dtype=float)
        d_l = np.asarray(d[n_primal:], dtype=float)
        primal_min, primal_max = _norm_range(d_u)
        dual_min, dual_max = _norm_range(d_l)
        info = ContactKKTSolveInfo(
            backend="numpy",
            solver="block_scaled",
            residual_norm=float(np.linalg.norm(residual)),
            relative_residual_norm=_relative_norm(residual, rhs_np),
            n_primal=n_primal,
            primal_scaling_min=primal_min,
            primal_scaling_max=primal_max,
            dual_scaling_min=dual_min,
            dual_scaling_max=dual_max,
            matrix_row_norm_min=row_min,
            matrix_row_norm_max=row_max,
            matrix_col_norm_min=col_min,
            matrix_col_norm_max=col_max,
            scaled_residual_norm=float(np.linalg.norm(scaled_residual)),
            scaled_relative_residual_norm=_relative_norm(scaled_residual, rhs_scaled),
            scaled_matrix_row_norm_min=srow_min,
            scaled_matrix_row_norm_max=srow_max,
            scaled_matrix_col_norm_min=scol_min,
            scaled_matrix_col_norm_max=scol_max,
        )
        return ContactKKTSolveResult(solution=solution, info=info)

    solution = _solve_kkt_numpy_direct(A, rhs_np, cfg)
    residual = _matrix_vector_product_numpy(A, solution) - rhs_np
    info = ContactKKTSolveInfo(
        backend="numpy",
        solver="direct",
        residual_norm=float(np.linalg.norm(residual)),
        relative_residual_norm=_relative_norm(residual, rhs_np),
        matrix_row_norm_min=row_min,
        matrix_row_norm_max=row_max,
        matrix_col_norm_min=col_min,
        matrix_col_norm_max=col_max,
    )
    return ContactKKTSolveResult(solution=solution, info=info)


def _solve_kkt_jax(kkt_matrix, rhs, cfg: ContactKKTSolveConfig):
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax.experimental import sparse as jsparse  # type: ignore

    mv_base, is_sparse_like = _as_jax_linear_op(kkt_matrix)

    def _gmres_solve(mv, bvec):
        maxiter = cfg.jax_maxiter if cfg.jax_maxiter is not None else max(10 * int(bvec.shape[0]), 100)
        x, _ = jsp.sparse.linalg.gmres(
            mv,
            bvec,
            tol=float(cfg.jax_tol),
            atol=float(cfg.jax_atol),
            restart=int(cfg.jax_restart),
            maxiter=int(maxiter),
        )
        return x

    if cfg.jax_solver == "spsolve":
        from jax.experimental.sparse.linalg import spsolve as jspsolve

        if hasattr(kkt_matrix, "to_bcoo"):
            bcoo = kkt_matrix.to_bcoo()
        elif isinstance(kkt_matrix, jsparse.BCOO):
            bcoo = kkt_matrix
        else:
            raise ValueError("jax_solver='spsolve' requires sparse input (FluxSparseMatrix or BCOO).")

        bcsr = jsparse.BCSR.from_bcoo(bcoo)
        b = jnp.asarray(rhs)
        if b.ndim == 1:
            return jspsolve(bcsr.data, bcsr.indices, bcsr.indptr, b)
        if b.ndim == 2:
            return jnp.stack([jspsolve(bcsr.data, bcsr.indices, bcsr.indptr, b[:, i]) for i in range(b.shape[1])], axis=1)
        raise ValueError("rhs must be rank-1 or rank-2.")

    shift = jnp.asarray(cfg.diagonal_shift, dtype=jnp.asarray(rhs).dtype)
    mv = (lambda x: mv_base(x) + shift * x)
    b = jnp.asarray(rhs)
    if is_sparse_like or cfg.jax_dense_mode == "iterative":
        if b.ndim == 1:
            return _gmres_solve(mv, b)
        if b.ndim == 2:
            return jnp.stack([_gmres_solve(mv, b[:, i]) for i in range(b.shape[1])], axis=1)
        raise ValueError("rhs must be rank-1 or rank-2.")

    @jax.custom_vjp
    def _solve_jax(A, bvec):
        return jnp.linalg.solve(A, bvec)

    def _solve_jax_fwd(A, bvec):
        x = jnp.linalg.solve(A, bvec)
        return x, (A, x)

    def _solve_jax_bwd(res, g):
        A, x = res
        lam = jnp.linalg.solve(A.T, g)
        gA = -jnp.outer(lam, x)
        gb = lam
        return gA, gb

    _solve_jax.defvjp(_solve_jax_fwd, _solve_jax_bwd)
    if not bool(cfg.allow_dense_fallback):
        raise ValueError("Dense fallback is disabled by ContactKKTSolveConfig.allow_dense_fallback.")
    A = jnp.asarray(kkt_matrix.to_dense()) if hasattr(kkt_matrix, "to_dense") else jnp.asarray(kkt_matrix)
    A = A + jnp.asarray(cfg.diagonal_shift, dtype=A.dtype) * jnp.eye(A.shape[0], dtype=A.dtype)
    return _solve_jax(A, b)


def solve_contact_kkt(
    kkt_matrix,
    rhs,
    *,
    backend: str | None = None,
    diagonal_shift: float = 0.0,
    config: ContactKKTSolveConfig | None = None,
):
    """
    Solve KKT linear system ``KKT * x = rhs``.

    `config` is the preferred control surface. `backend=None` auto-selects from
    ``kkt_matrix``/``rhs`` when no explicit config is provided.
    """
    cfg = _resolve_kkt_solve_config(
        backend=backend,
        diagonal_shift=diagonal_shift,
        config=config,
        kkt_matrix=kkt_matrix,
        rhs=rhs,
    )
    if cfg.backend == "petsc4py":
        return _solve_kkt_petsc(kkt_matrix, rhs, cfg)
    if cfg.backend == "numpy":
        return _solve_kkt_numpy(kkt_matrix, rhs, cfg)
    return _solve_kkt_jax(kkt_matrix, rhs, cfg)


def solve_contact_kkt_with_info(
    kkt_matrix,
    rhs,
    *,
    backend: str | None = None,
    diagonal_shift: float = 0.0,
    config: ContactKKTSolveConfig | None = None,
) -> ContactKKTSolveResult:
    """
    Solve ``KKT * x = rhs`` and return solution plus residual/scaling diagnostics.

    The NumPy ``numpy_solver="block_scaled"`` path reports block scaling ranges
    and scaled-system residuals.  Other backends currently report unscaled
    residuals only.
    """
    cfg = _resolve_kkt_solve_config(
        backend=backend,
        diagonal_shift=diagonal_shift,
        config=config,
        kkt_matrix=kkt_matrix,
        rhs=rhs,
    )
    if cfg.backend == "numpy":
        return _solve_kkt_numpy_with_info(kkt_matrix, rhs, cfg)
    if cfg.backend == "petsc4py":
        solution = _solve_kkt_petsc(kkt_matrix, rhs, cfg)
        A_csr = _as_numpy_csr(kkt_matrix)
        A = A_csr if A_csr is not None else _as_numpy_dense(kkt_matrix)
        A = _apply_numpy_diagonal_shift(A, float(cfg.diagonal_shift))
        rhs_np = np.asarray(rhs, dtype=float)
        residual = _matrix_vector_product_numpy(A, np.asarray(solution, dtype=float)) - rhs_np
        row_min, row_max, col_min, col_max = _matrix_axis_norm_ranges(A)
        return ContactKKTSolveResult(
            solution=solution,
            info=ContactKKTSolveInfo(
                backend="petsc4py",
                solver=str(cfg.petsc_ksp_type),
                residual_norm=float(np.linalg.norm(residual)),
                relative_residual_norm=_relative_norm(residual, rhs_np),
                matrix_row_norm_min=row_min,
                matrix_row_norm_max=row_max,
                matrix_col_norm_min=col_min,
                matrix_col_norm_max=col_max,
            ),
        )

    solution = _solve_kkt_jax(kkt_matrix, rhs, cfg)
    try:
        import jax.numpy as jnp

        mv, _ = _as_jax_linear_op(kkt_matrix)
        rhs_j = jnp.asarray(rhs)
        residual_j = mv(solution) - rhs_j
        if float(cfg.diagonal_shift) != 0.0:
            residual_j = residual_j + float(cfg.diagonal_shift) * solution
        residual_norm = float(jnp.linalg.norm(residual_j))
        relative_residual_norm = residual_norm / max(float(jnp.linalg.norm(rhs_j)), 1.0)
    except Exception:
        residual_norm = float("nan")
        relative_residual_norm = float("nan")
    return ContactKKTSolveResult(
        solution=solution,
        info=ContactKKTSolveInfo(
            backend="jax",
            solver=str(cfg.jax_solver),
            residual_norm=residual_norm,
            relative_residual_norm=relative_residual_norm,
        ),
    )

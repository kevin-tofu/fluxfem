from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from ..core.assembly import (
    ResidualForm,
    assemble_jacobian_scatter,
    assemble_residual_scatter,
    make_element_jacobian_kernel,
    make_element_residual_kernel,
    make_sparsity_pattern,
)
from .craig_bampton import LinearConstraintSystem, RBE3Patch
from .dirichlet import DirichletBC, _normalize_dirichlet
from .result import SolverResult


@dataclass(frozen=True)
class NonlinearConstrainedSolveResult:
    """Result of a linearly constrained nonlinear solve.

    ``u`` and ``multipliers`` are the final displacement and Lagrange
    multipliers. When load stepping is used through
    ``NonlinearConstrainedProblem.solve(config=...)``, ``load_factors`` records
    the completed factors and ``step_infos`` stores the per-step solver
    diagnostics.
    """

    u: jnp.ndarray
    multipliers: jnp.ndarray
    info: SolverResult
    load_factors: tuple[float, ...] = ()
    step_infos: tuple[SolverResult, ...] = ()


@dataclass(frozen=True)
class _ConstrainedSolveControls:
    tol: float
    atol: float
    maxiter: int
    load_factors: tuple[float, ...]
    linear_solver: str
    linear_tol: float | None
    linear_maxiter: int | None
    linear_preconditioner: Any | None
    petsc_ksp_type: str
    petsc_pc_type: str
    petsc_rtol: float | None
    petsc_atol: float | None
    petsc_max_it: int | None
    petsc_options: dict[str, Any] | None
    petsc_use_pmat: bool


def _as_constraint_system(constraints: LinearConstraintSystem | None, n_dofs: int, dtype) -> LinearConstraintSystem:
    if constraints is None:
        return LinearConstraintSystem(jnp.zeros((0, int(n_dofs)), dtype=dtype))
    if constraints.n_dofs != int(n_dofs):
        raise ValueError("constraint matrix column count must match space.n_dofs.")
    return constraints


def _normalize_dirichlet_optional(dirichlet, n_dofs: int, dtype):
    if dirichlet is None:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), np.arange(int(n_dofs), dtype=int)
    if isinstance(dirichlet, DirichletBC):
        dirichlet = dirichlet.as_tuple()
    dofs, values = dirichlet
    dofs, values = _normalize_dirichlet(dofs, values)
    if values.ndim == 0:
        values = np.full(dofs.shape[0], float(values), dtype=float)
    dofs = np.asarray(dofs, dtype=int).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if dofs.size and (dofs.min() < 0 or dofs.max() >= int(n_dofs)):
        raise ValueError("dirichlet dofs contain an index outside space.n_dofs.")
    if np.unique(dofs).size != dofs.size:
        raise ValueError("dirichlet dofs must not contain duplicates.")
    mask = np.ones(int(n_dofs), dtype=bool)
    mask[dofs] = False
    return dofs, values.astype(np.asarray(dtype).dtype if not isinstance(dtype, type) else dtype), np.flatnonzero(mask).astype(int)


def _validate_load_factors(load_factors) -> tuple[float, ...]:
    schedule: list[float] = []
    prev = 0.0
    for value in load_factors:
        factor = float(value)
        if not np.isfinite(factor):
            raise ValueError("load factors must be finite.")
        if factor < 0.0 or factor > 1.0:
            raise ValueError("load factors must lie in [0, 1].")
        if factor < prev:
            raise ValueError("load factors must be monotonically nondecreasing.")
        schedule.append(factor)
        prev = factor
    if not schedule:
        raise ValueError("load schedule must contain at least one load factor.")
    return tuple(schedule)


def _solve_controls_from_config(
    config: Any | None,
    *,
    tol: float,
    atol: float,
    maxiter: int,
) -> _ConstrainedSolveControls:
    if config is None:
        return _ConstrainedSolveControls(
            float(tol),
            float(atol),
            int(maxiter),
            (1.0,),
            "spsolve",
            None,
            None,
            None,
            "gmres",
            "none",
            None,
            None,
            None,
            None,
            False,
        )

    unsupported: list[str] = []
    if bool(getattr(config, "line_search", False)):
        unsupported.append("line_search")
    linear_solver = "spsolve" if getattr(config, "linear_solver", "spsolve") is None else str(config.linear_solver)
    if linear_solver not in ("spsolve", "gmres", "petsc_shell"):
        unsupported.append("linear_solver")
    if unsupported:
        names = ", ".join(unsupported)
        raise NotImplementedError(f"NonlinearConstrainedProblem.solve(config=...) does not support: {names}.")

    load_factors = config.schedule() if hasattr(config, "schedule") else (1.0,)
    return _ConstrainedSolveControls(
        float(config.tol),
        float(config.atol),
        int(config.maxiter),
        _validate_load_factors(load_factors),
        linear_solver,
        getattr(config, "linear_tol", None),
        getattr(config, "linear_maxiter", None),
        getattr(config, "linear_preconditioner", None),
        str(getattr(config, "petsc_ksp_type", "gmres")),
        str(getattr(config, "petsc_pc_type", "none")),
        getattr(config, "petsc_rtol", None),
        getattr(config, "petsc_atol", None),
        getattr(config, "petsc_max_it", None),
        getattr(config, "petsc_options", None),
        bool(getattr(config, "petsc_use_pmat", False)),
    )


def _summarize_load_step_infos(step_infos: tuple[SolverResult, ...], *, tol: float, atol: float) -> SolverResult:
    last = step_infos[-1]
    converged = all(info.converged for info in step_infos)
    stop_reason = "load_steps_converged" if converged else "load_steps_failed"
    return SolverResult(
        converged=converged,
        iters=sum(int(info.iters) for info in step_infos),
        residual_norm=last.residual_norm,
        residual0=step_infos[0].residual0,
        rel_residual=last.rel_residual,
        line_search_steps=sum(int(info.line_search_steps) for info in step_infos),
        linear_iters=sum(int(info.linear_iters or 0) for info in step_infos),
        linear_converged=all(info.linear_converged is not False for info in step_infos),
        linear_residual=last.linear_residual,
        tol=tol,
        atol=atol,
        stopping_criterion=last.stopping_criterion,
        step_norm=last.step_norm,
        stop_reason=stop_reason,
        nan_detected=any(bool(info.nan_detected) for info in step_infos),
    )


def _solve_kkt_delta(
    sp,
    spla,
    j_ff,
    c_csr,
    rhs: np.ndarray,
    *,
    linear_solver: str,
    linear_tol: float | None,
    linear_maxiter: int | None,
    linear_preconditioner: Any | None,
    petsc_ksp_type: str,
    petsc_pc_type: str,
    petsc_rtol: float | None,
    petsc_atol: float | None,
    petsc_max_it: int | None,
    petsc_options: dict[str, Any] | None,
    petsc_use_pmat: bool,
) -> tuple[np.ndarray, int | None, bool | None, float | None]:
    n_free = int(j_ff.shape[0])
    n_constraints = int(c_csr.shape[0])
    n_total = n_free + n_constraints

    if linear_solver == "spsolve":
        zero = sp.csr_matrix((n_constraints, n_constraints), dtype=float)
        lhs = sp.bmat([[j_ff, c_csr.T], [c_csr, zero]], format="csr")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            delta = spla.spsolve(lhs, rhs)
        if caught or not np.all(np.isfinite(delta)):
            lhs_dense = lhs.toarray()
            if np.linalg.matrix_rank(lhs_dense) < lhs_dense.shape[0]:
                raise np.linalg.LinAlgError("constrained Newton KKT matrix is singular.")
            delta = np.linalg.solve(lhs_dense, rhs)
        residual = float(np.linalg.norm(lhs @ delta - rhs, ord=np.inf))
        return delta, None, True, residual

    def matvec(x):
        x_u = x[:n_free]
        x_lam = x[n_free:]
        return np.concatenate([j_ff @ x_u + c_csr.T @ x_lam, c_csr @ x_u])

    if linear_solver == "petsc_shell":
        from .petsc import petsc_shell_solve

        pmat = None
        if petsc_use_pmat:
            zero = sp.csr_matrix((n_constraints, n_constraints), dtype=float)
            pmat = sp.bmat([[j_ff, c_csr.T], [c_csr, zero]], format="csr")
        delta, info = petsc_shell_solve(
            matvec,
            rhs,
            n_dofs=n_total,
            ksp_type=petsc_ksp_type,
            pc_type=petsc_pc_type,
            preconditioner=linear_preconditioner,
            pmat=pmat,
            rtol=petsc_rtol if petsc_rtol is not None else linear_tol,
            atol=petsc_atol,
            max_it=petsc_max_it if petsc_max_it is not None else linear_maxiter,
            options=petsc_options,
            return_info=True,
        )
        residual = float(np.linalg.norm(matvec(delta) - rhs, ord=np.inf))
        converged = bool(info.get("converged", True)) and np.all(np.isfinite(delta))
        if not converged:
            reason = info.get("reason", "unknown")
            raise np.linalg.LinAlgError(
                f"constrained Newton KKT PETSc shell solve did not converge: reason={reason}, residual={residual:.3e}."
            )
        linear_residual = info.get("residual_norm", residual)
        return (
            np.asarray(delta, dtype=float),
            info.get("iters", None),
            converged,
            float(linear_residual) if linear_residual is not None else residual,
        )

    if linear_solver != "gmres":
        raise ValueError(f"Unknown constrained KKT linear solver: {linear_solver}")

    operator = spla.LinearOperator((n_total, n_total), matvec=matvec, dtype=float)
    iterations = 0
    last_pr_norm = None

    def callback(value):
        nonlocal iterations, last_pr_norm
        iterations += 1
        if np.isscalar(value):
            last_pr_norm = float(value)

    rtol = float(linear_tol) if linear_tol is not None else 1.0e-10
    kwargs = {"rtol": rtol, "atol": 0.0, "maxiter": linear_maxiter, "callback": callback}
    try:
        delta, info = spla.gmres(operator, rhs, callback_type="pr_norm", **kwargs)
    except TypeError:  # pragma: no cover - compatibility with older SciPy
        delta, info = spla.gmres(operator, rhs, tol=rtol, maxiter=linear_maxiter, callback=callback)
    residual = float(np.linalg.norm(matvec(delta) - rhs, ord=np.inf))
    converged = int(info) == 0 and np.all(np.isfinite(delta))
    if not converged:
        raise np.linalg.LinAlgError(
            f"constrained Newton KKT GMRES did not converge: info={info}, residual={residual:.3e}."
        )
    if iterations == 0 and last_pr_norm is None:
        iterations = 1
    return delta, iterations, True, residual


def solve_nonlinear_constrained_kkt(
    space,
    residual_form: ResidualForm[Any],
    u0,
    params: Any,
    *,
    constraints: LinearConstraintSystem | None = None,
    dirichlet: tuple[np.ndarray, np.ndarray] | DirichletBC | None = None,
    external_vector=None,
    lambda0=None,
    tol: float = 1e-8,
    atol: float = 1e-10,
    maxiter: int = 20,
    jacobian_pattern=None,
    assembly_policy: Any | None = None,
    linear_solver: str = "spsolve",
    linear_tol: float | None = None,
    linear_maxiter: int | None = None,
    linear_preconditioner: Any | None = None,
    petsc_ksp_type: str = "gmres",
    petsc_pc_type: str = "none",
    petsc_rtol: float | None = None,
    petsc_atol: float | None = None,
    petsc_max_it: int | None = None,
    petsc_options: dict[str, Any] | None = None,
    petsc_use_pmat: bool = False,
) -> NonlinearConstrainedSolveResult:
    """Solve ``R(u) - f + C.T @ lambda = 0`` and ``C u = rhs`` by Newton-KKT.

    The constraints are linear and are enforced exactly at the Newton level.
    Dirichlet DOFs are eliminated from the displacement unknowns.
    """
    linear_solver = str(linear_solver)
    if linear_solver not in ("spsolve", "gmres", "petsc_shell"):
        raise ValueError("linear_solver must be 'spsolve', 'gmres', or 'petsc_shell'.")

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for solve_nonlinear_constrained_kkt.") from exc

    n_dofs = int(space.n_dofs)
    u = jnp.asarray(u0)
    if u.shape != (n_dofs,):
        raise ValueError("u0 must have shape (space.n_dofs,).")
    dtype = u.dtype
    constraint_system = _as_constraint_system(constraints, n_dofs, dtype)
    c_full = np.asarray(constraint_system.matrix, dtype=float)
    rhs_c = np.asarray(constraint_system.rhs, dtype=float).reshape(-1)
    n_constraints = int(c_full.shape[0])
    lam = jnp.zeros((n_constraints,), dtype=dtype) if lambda0 is None else jnp.asarray(lambda0, dtype=dtype)
    if lam.shape != (n_constraints,):
        raise ValueError("lambda0 must have shape (n_constraints,).")

    fixed, fixed_values, free = _normalize_dirichlet_optional(dirichlet, n_dofs, dtype)
    fixed_j = jnp.asarray(fixed, dtype=jnp.int64)
    fixed_values_j = jnp.asarray(fixed_values, dtype=dtype)
    free_j = jnp.asarray(free, dtype=jnp.int64)
    c_free = c_full[:, free]
    c_fixed = c_full[:, fixed] if fixed.size else np.zeros((n_constraints, 0), dtype=float)
    fixed_offset = c_fixed @ fixed_values if fixed.size else np.zeros((n_constraints,), dtype=float)
    external = jnp.zeros((n_dofs,), dtype=dtype) if external_vector is None else jnp.asarray(external_vector, dtype=dtype)
    if external.shape != (n_dofs,):
        raise ValueError("external_vector must have shape (space.n_dofs,).")

    u_free = u[free_j]
    if fixed.size:
        u = u.at[fixed_j].set(fixed_values_j)

    pattern = jacobian_pattern if jacobian_pattern is not None else make_sparsity_pattern(space, with_idx=True)
    res_kernel = make_element_residual_kernel(residual_form, params)
    jac_kernel = make_element_jacobian_kernel(residual_form, params)

    def expand_full(u_free_vec):
        out = jnp.zeros((n_dofs,), dtype=dtype).at[free_j].set(u_free_vec)
        if fixed.size:
            out = out.at[fixed_j].set(fixed_values_j)
        return out

    def assemble_state(u_free_vec, lam_vec):
        u_full = expand_full(u_free_vec)
        residual = assemble_residual_scatter(
            space,
            residual_form,
            u_full,
            params,
            kernel=res_kernel,
            policy=assembly_policy,
        )
        residual = residual - external
        stationarity = residual[free_j] + jnp.asarray(c_free.T, dtype=dtype) @ lam_vec
        constraint = jnp.asarray(c_free, dtype=dtype) @ u_free_vec + jnp.asarray(fixed_offset - rhs_c, dtype=dtype)
        return u_full, stationarity, constraint

    u_full, stationarity, constraint = assemble_state(u_free, lam)
    residual_vec = jnp.concatenate([stationarity, constraint])
    residual0 = float(jax.block_until_ready(jnp.linalg.norm(residual_vec, ord=jnp.inf)))
    threshold = max(float(atol), float(tol) * residual0)
    if residual0 <= threshold:
        return NonlinearConstrainedSolveResult(
            u=u_full,
            multipliers=lam,
            info=SolverResult(
                converged=True,
                iters=0,
                residual_norm=residual0,
                residual0=residual0,
                rel_residual=0.0,
                tol=tol,
                atol=atol,
                stopping_criterion=threshold,
                stop_reason="initial_converged",
            ),
        )

    final_norm = residual0
    last_linear_iters = None
    last_linear_converged = None
    last_linear_residual = None
    for iteration in range(1, int(maxiter) + 1):
        jac = assemble_jacobian_scatter(
            space,
            residual_form,
            u_full,
            params,
            kernel=jac_kernel,
            pattern=pattern,
            policy=assembly_policy,
        )
        j_ff = jac.to_csr()[np.ix_(free, free)]
        c_csr = sp.csr_matrix(c_free)
        rhs = -np.asarray(residual_vec, dtype=float)
        delta, last_linear_iters, last_linear_converged, last_linear_residual = _solve_kkt_delta(
            sp,
            spla,
            j_ff,
            c_csr,
            rhs,
            linear_solver=linear_solver,
            linear_tol=linear_tol,
            linear_maxiter=linear_maxiter,
            linear_preconditioner=linear_preconditioner,
            petsc_ksp_type=petsc_ksp_type,
            petsc_pc_type=petsc_pc_type,
            petsc_rtol=petsc_rtol,
            petsc_atol=petsc_atol,
            petsc_max_it=petsc_max_it,
            petsc_options=petsc_options,
            petsc_use_pmat=petsc_use_pmat,
        )
        du = jnp.asarray(delta[: free.size], dtype=dtype)
        dlambda = jnp.asarray(delta[free.size :], dtype=dtype)
        u_free = u_free + du
        lam = lam + dlambda
        u_full, stationarity, constraint = assemble_state(u_free, lam)
        residual_vec = jnp.concatenate([stationarity, constraint])
        final_norm = float(jax.block_until_ready(jnp.linalg.norm(residual_vec, ord=jnp.inf)))
        if final_norm <= threshold:
            return NonlinearConstrainedSolveResult(
                u=u_full,
                multipliers=lam,
                info=SolverResult(
                    converged=True,
                    iters=iteration,
                    residual_norm=final_norm,
                    residual0=residual0,
                    rel_residual=final_norm / max(residual0, 1.0e-30),
                    tol=tol,
                    atol=atol,
                    stopping_criterion=threshold,
                    step_norm=float(np.linalg.norm(delta)),
                    linear_iters=last_linear_iters,
                    linear_converged=last_linear_converged,
                    linear_residual=last_linear_residual,
                    stop_reason="converged",
                ),
            )

    return NonlinearConstrainedSolveResult(
        u=u_full,
        multipliers=lam,
        info=SolverResult(
            converged=False,
            iters=int(maxiter),
            residual_norm=final_norm,
            residual0=residual0,
            rel_residual=final_norm / max(residual0, 1.0e-30),
            tol=tol,
            atol=atol,
            stopping_criterion=threshold,
            linear_iters=last_linear_iters,
            linear_converged=last_linear_converged,
            linear_residual=last_linear_residual,
            stop_reason="maxiter",
        ),
    )


@dataclass
class NonlinearConstrainedProblem:
    """Small facade for nonlinear FEM with linear MPC/RBE constraints.

    The facade is the user-facing entry point for exact linear constraints such
    as MPC and RBE3-style fixture averages. It assembles the combined constraint
    system and calls the lower-level KKT Newton solver.
    """

    space: Any
    residual_form: ResidualForm[Any]
    params: Any
    dirichlet: tuple[np.ndarray, np.ndarray] | DirichletBC | None = None
    external_vector: Any | None = None
    constraints: list[LinearConstraintSystem] = field(default_factory=list)
    dtype: Any = jnp.float64
    jacobian_pattern: Any | None = None
    assembly_policy: Any | None = None

    def add_constraint(self, constraint: LinearConstraintSystem) -> "NonlinearConstrainedProblem":
        if constraint.n_dofs != int(self.space.n_dofs):
            raise ValueError("constraint matrix column count must match space.n_dofs.")
        self.constraints.append(constraint)
        return self

    def add_rbe3_patch_constraint(self, patch: RBE3Patch, rhs=None) -> "NonlinearConstrainedProblem":
        matrix = patch.average_matrix(int(self.space.n_dofs))
        rhs_arr = jnp.zeros((patch.dim,), dtype=matrix.dtype) if rhs is None else jnp.asarray(rhs, dtype=matrix.dtype)
        self.constraints.append(LinearConstraintSystem(matrix, rhs_arr))
        return self

    def add_local_force(self, dofs, values) -> "NonlinearConstrainedProblem":
        dofs_arr = np.asarray(dofs, dtype=int).reshape(-1)
        values_arr = jnp.asarray(values, dtype=self.dtype).reshape(-1)
        if dofs_arr.shape != values_arr.shape:
            raise ValueError("dofs and values must have the same flattened shape.")
        if dofs_arr.size and (dofs_arr.min() < 0 or dofs_arr.max() >= int(self.space.n_dofs)):
            raise ValueError("force dofs contain an index outside space.n_dofs.")
        base = (
            jnp.zeros((int(self.space.n_dofs),), dtype=self.dtype)
            if self.external_vector is None
            else jnp.asarray(self.external_vector, dtype=self.dtype)
        )
        self.external_vector = base.at[jnp.asarray(dofs_arr, dtype=jnp.int64)].add(values_arr)
        return self

    def constraint_system(self) -> LinearConstraintSystem:
        if not self.constraints:
            return LinearConstraintSystem(jnp.zeros((0, int(self.space.n_dofs)), dtype=self.dtype))
        matrices = [jnp.asarray(c.matrix, dtype=self.dtype) for c in self.constraints]
        rhs = [jnp.asarray(c.rhs, dtype=self.dtype) for c in self.constraints]
        return LinearConstraintSystem(jnp.vstack(matrices), jnp.concatenate(rhs))

    def solve(
        self,
        u0=None,
        *,
        lambda0=None,
        config: Any | None = None,
        tol: float = 1e-8,
        atol: float = 1e-10,
        maxiter: int = 20,
    ) -> NonlinearConstrainedSolveResult:
        """Solve the constrained nonlinear problem.

        Parameters
        ----------
        u0:
            Optional full-DOF initial displacement. Fixed Dirichlet values are
            imposed before each KKT solve.
        lambda0:
            Optional initial Lagrange multipliers for the first load step.
            Subsequent load steps warm-start from the previous multipliers.
        config:
            Optional ``NewtonLoopConfig``. Supported fields are ``tol``,
            ``atol``, ``maxiter``, ``n_steps``, ``load_sequence``,
            ``linear_solver`` (``"spsolve"``, ``"gmres"``, or
            ``"petsc_shell"``), ``linear_tol``, ``linear_maxiter``,
            ``linear_preconditioner``, and the ``petsc_*`` controls. Load
            stepping scales only ``external_vector``; linear constraints and
            Dirichlet values stay exact at every step.
        tol, atol, maxiter:
            Backward-compatible controls used when ``config`` is not provided.

        Notes
        -----
        ``line_search`` is not implemented for the constrained KKT path yet and
        raises ``NotImplementedError`` when requested.
        """
        controls = _solve_controls_from_config(config, tol=tol, atol=atol, maxiter=maxiter)
        u_init = (
            jnp.zeros((int(self.space.n_dofs),), dtype=self.dtype)
            if u0 is None
            else jnp.asarray(u0, dtype=self.dtype)
        )
        constraints = self.constraint_system()
        base_external = (
            None
            if self.external_vector is None
            else jnp.asarray(self.external_vector, dtype=self.dtype)
        )
        u_step = u_init
        lambda_step = lambda0
        step_infos: list[SolverResult] = []
        last_result: NonlinearConstrainedSolveResult | None = None
        for load_factor in controls.load_factors:
            external = None if base_external is None else load_factor * base_external
            last_result = solve_nonlinear_constrained_kkt(
                self.space,
                self.residual_form,
                u_step,
                self.params,
                constraints=constraints,
                dirichlet=self.dirichlet,
                external_vector=external,
                lambda0=lambda_step,
                tol=controls.tol,
                atol=controls.atol,
                maxiter=controls.maxiter,
                jacobian_pattern=self.jacobian_pattern,
                assembly_policy=self.assembly_policy,
                linear_solver=controls.linear_solver,
                linear_tol=controls.linear_tol,
                linear_maxiter=controls.linear_maxiter,
                linear_preconditioner=controls.linear_preconditioner,
                petsc_ksp_type=controls.petsc_ksp_type,
                petsc_pc_type=controls.petsc_pc_type,
                petsc_rtol=controls.petsc_rtol,
                petsc_atol=controls.petsc_atol,
                petsc_max_it=controls.petsc_max_it,
                petsc_options=controls.petsc_options,
                petsc_use_pmat=controls.petsc_use_pmat,
            )
            step_infos.append(last_result.info)
            u_step = last_result.u
            lambda_step = last_result.multipliers
            if not last_result.info.converged:
                break

        if last_result is None:  # pragma: no cover - guarded by load schedule validation
            raise RuntimeError("load schedule produced no solve steps.")
        step_infos_tuple = tuple(step_infos)
        if len(controls.load_factors) == 1:
            info = last_result.info
        else:
            info = _summarize_load_step_infos(step_infos_tuple, tol=controls.tol, atol=controls.atol)
        return NonlinearConstrainedSolveResult(
            u=last_result.u,
            multipliers=last_result.multipliers,
            info=info,
            load_factors=controls.load_factors[: len(step_infos_tuple)],
            step_infos=step_infos_tuple,
        )


__all__ = [
    "NonlinearConstrainedProblem",
    "NonlinearConstrainedSolveResult",
    "solve_nonlinear_constrained_kkt",
]

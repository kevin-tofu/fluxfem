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
    """Result of a linearly constrained nonlinear solve."""

    u: jnp.ndarray
    multipliers: jnp.ndarray
    info: SolverResult


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


def _solve_controls_from_config(config: Any | None, *, tol: float, atol: float, maxiter: int) -> tuple[float, float, int]:
    if config is None:
        return float(tol), float(atol), int(maxiter)

    unsupported: list[str] = []
    if bool(getattr(config, "line_search", False)):
        unsupported.append("line_search")
    if getattr(config, "load_sequence", None) is not None:
        unsupported.append("load_sequence")
    if int(getattr(config, "n_steps", 1)) != 1:
        unsupported.append("n_steps")
    if getattr(config, "linear_solver", "spsolve") not in (None, "spsolve"):
        unsupported.append("linear_solver")
    if unsupported:
        names = ", ".join(unsupported)
        raise NotImplementedError(f"NonlinearConstrainedProblem.solve(config=...) does not support: {names}.")

    return float(config.tol), float(config.atol), int(config.maxiter)


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
) -> NonlinearConstrainedSolveResult:
    """Solve ``R(u) - f + C.T @ lambda = 0`` and ``C u = rhs`` by Newton-KKT.

    The constraints are linear and are enforced exactly at the Newton level.
    Dirichlet DOFs are eliminated from the displacement unknowns.
    """
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
        zero = sp.csr_matrix((n_constraints, n_constraints), dtype=float)
        lhs = sp.bmat([[j_ff, c_csr.T], [c_csr, zero]], format="csr")
        rhs = -np.asarray(residual_vec, dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            delta = spla.spsolve(lhs, rhs)
        if caught or not np.all(np.isfinite(delta)):
            lhs_dense = lhs.toarray()
            if np.linalg.matrix_rank(lhs_dense) < lhs_dense.shape[0]:
                raise np.linalg.LinAlgError("constrained Newton KKT matrix is singular.")
            delta = np.linalg.solve(lhs_dense, rhs)
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
            stop_reason="maxiter",
        ),
    )


@dataclass
class NonlinearConstrainedProblem:
    """Small facade for nonlinear FEM with linear MPC/RBE constraints."""

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
        tol, atol, maxiter = _solve_controls_from_config(config, tol=tol, atol=atol, maxiter=maxiter)
        u_init = (
            jnp.zeros((int(self.space.n_dofs),), dtype=self.dtype)
            if u0 is None
            else jnp.asarray(u0, dtype=self.dtype)
        )
        return solve_nonlinear_constrained_kkt(
            self.space,
            self.residual_form,
            u_init,
            self.params,
            constraints=self.constraint_system(),
            dirichlet=self.dirichlet,
            external_vector=self.external_vector,
            lambda0=lambda0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            jacobian_pattern=self.jacobian_pattern,
            assembly_policy=self.assembly_policy,
        )


__all__ = [
    "NonlinearConstrainedProblem",
    "NonlinearConstrainedSolveResult",
    "solve_nonlinear_constrained_kkt",
]

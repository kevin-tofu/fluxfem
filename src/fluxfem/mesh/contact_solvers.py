from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np
import numpy.typing as npt

from .contact_kkt_solver import ContactKKTSolveConfig, solve_contact_kkt


@dataclass(frozen=True)
class AugmentedLagrangianState:
    """State passed through a generic augmented-Lagrangian outer loop."""

    lambda_values: Any
    rho: float
    iteration: int = 0
    constraint: Any | None = None
    active_mask: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AugmentedLagrangianResult:
    """Result of a generic augmented-Lagrangian outer loop."""

    solution: Any
    state: AugmentedLagrangianState
    converged: bool
    iters: int
    constraint_norm: float
    lambda_update_norm: float
    info: Any | None = None


@dataclass(frozen=True)
class UnilateralContactActiveSetRecord:
    """One active-set iteration for a linearized unilateral contact KKT solve."""

    iteration: int
    active_count: int
    min_gap: float
    min_lambda: float
    changed: bool


@dataclass(frozen=True)
class UnilateralContactActiveSetResult:
    """Result of a linearized unilateral contact active-set KKT solve."""

    displacement: np.ndarray
    lambda_n: np.ndarray
    gap: np.ndarray
    active_mask: np.ndarray
    converged: bool
    iters: int
    records: tuple[UnilateralContactActiveSetRecord, ...]


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
    if isinstance(obj, (str, bytes)):
        return False
    if isinstance(obj, (list, tuple)):
        return any(_contains_jax_value(v) for v in obj)
    data = getattr(obj, "data", None)
    if data is not None and not isinstance(obj, np.ndarray) and data is not obj and _contains_jax_value(data):
        return True
    return False


def _al_backend_namespace(*values: Any):
    if any(_contains_jax_value(v) for v in values):
        import jax.numpy as jnp

        return jnp
    return np


def _al_asarray(xp, value: Any):
    return xp.asarray(value)


def _al_norm(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    return float(np.linalg.norm(arr.reshape(-1), ord=np.inf)) if arr.size else 0.0


def _al_constraint_from_operator(B: Any, *, offset: Any | None = None) -> Callable[[Any], Any]:
    def constraint(solution: Any) -> Any:
        xp = _al_backend_namespace(B, solution, offset)
        value = _al_asarray(xp, B) @ _al_asarray(xp, solution)
        if offset is not None:
            value = value - _al_asarray(xp, offset)
        return value

    return constraint


def _al_project_lambda(
    lambda_trial: Any,
    *,
    projection: str | Callable[[Any, Any, Any, AugmentedLagrangianState], Any] | None,
    constraint: Any,
    solution: Any,
    state: AugmentedLagrangianState,
) -> Any:
    if projection is None or str(projection).lower() in {"none", "identity"}:
        return lambda_trial
    if isinstance(projection, str):
        key = projection.lower()
        if key in {"nonnegative", "positive", "unilateral"}:
            xp = _al_backend_namespace(lambda_trial)
            return xp.maximum(_al_asarray(xp, lambda_trial), 0.0)
        raise ValueError("projection must be None, 'nonnegative', or a callable.")
    return projection(lambda_trial, constraint, solution, state)


def solve_augmented_lagrangian_outer_loop(
    solve_subproblem: Callable[[Any, AugmentedLagrangianState], Any],
    x0: Any,
    *,
    constraint_fn: Callable[[Any], Any] | None = None,
    operators: Any | None = None,
    B: Any | None = None,
    offset: Any | None = None,
    lambda0: Any | None = None,
    rho: float = 1.0,
    maxiter: int = 10,
    tol: float = 1e-8,
    atol: float = 0.0,
    lambda_tol: float | None = None,
    penalty_growth: float = 1.0,
    projection: str | Callable[[Any, Any, Any, AugmentedLagrangianState], Any] | None = None,
    update_fn: Callable[[Any, Any, Any, AugmentedLagrangianState], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AugmentedLagrangianResult:
    """
    Generic augmented-Lagrangian outer loop.

    ``solve_subproblem(x, state)`` solves the current inner problem using
    ``state.lambda_values`` and ``state.rho`` and returns either ``solution`` or
    ``(solution, info)``. The outer loop then evaluates ``constraint_fn(solution)``
    and updates the multiplier. If ``constraint_fn`` is omitted, pass
    ``operators`` or ``B`` to use ``B @ solution - offset``.
    """
    if float(rho) <= 0.0:
        raise ValueError("rho must be positive.")
    if int(maxiter) <= 0:
        raise ValueError("maxiter must be positive.")
    if constraint_fn is None:
        B_eff = B
        if B_eff is None and operators is not None:
            B_eff = operators.B
        if B_eff is None:
            raise ValueError("constraint_fn, operators, or B is required.")
        constraint_fn = _al_constraint_from_operator(B_eff, offset=offset)

    x_curr = x0
    g0 = constraint_fn(x_curr)
    xp = _al_backend_namespace(x_curr, g0, lambda0)
    lam_curr = xp.zeros_like(_al_asarray(xp, g0)) if lambda0 is None else _al_asarray(xp, lambda0)
    rho_curr = float(rho)
    state_curr = AugmentedLagrangianState(
        lambda_values=lam_curr,
        rho=rho_curr,
        iteration=0,
        constraint=g0,
        metadata=dict(metadata or {}),
    )
    info_curr: Any | None = None
    constraint_norm = _al_norm(g0)
    lambda_update_norm = float("inf")
    converged = False

    for outer in range(1, int(maxiter) + 1):
        result = solve_subproblem(x_curr, state_curr)
        if isinstance(result, tuple) and len(result) == 2:
            x_next, info_curr = result
        else:
            x_next = result
            info_curr = None
        g_next = constraint_fn(x_next)
        xp = _al_backend_namespace(x_next, g_next, lam_curr)
        lam_arr = _al_asarray(xp, lam_curr)
        g_arr = _al_asarray(xp, g_next)
        if update_fn is None:
            lam_trial = lam_arr + xp.asarray(rho_curr) * g_arr
        else:
            lam_trial = update_fn(lam_arr, g_arr, x_next, state_curr)
        state_for_projection = AugmentedLagrangianState(
            lambda_values=lam_arr,
            rho=rho_curr,
            iteration=outer,
            constraint=g_arr,
            metadata=dict(metadata or {}),
        )
        lam_next = _al_project_lambda(
            lam_trial,
            projection=projection,
            constraint=g_arr,
            solution=x_next,
            state=state_for_projection,
        )
        lambda_update = _al_asarray(xp, lam_next) - lam_arr
        constraint_norm = _al_norm(g_arr)
        lambda_update_norm = _al_norm(lambda_update)
        active_mask = None
        if isinstance(projection, str) and projection.lower() in {"nonnegative", "positive", "unilateral"}:
            active_mask = _al_asarray(xp, lam_next) > 0.0
        state_curr = AugmentedLagrangianState(
            lambda_values=lam_next,
            rho=rho_curr,
            iteration=outer,
            constraint=g_arr,
            active_mask=active_mask,
            metadata=dict(metadata or {}),
        )
        x_curr = x_next
        lam_curr = lam_next
        lambda_limit = float(tol if lambda_tol is None else lambda_tol)
        if constraint_norm <= max(float(atol), float(tol)) and lambda_update_norm <= max(float(atol), lambda_limit):
            converged = True
            break
        rho_curr *= float(penalty_growth)
        if rho_curr <= 0.0:
            raise ValueError("penalty_growth produced a non-positive rho.")
        if rho_curr != state_curr.rho:
            state_curr = replace(state_curr, rho=rho_curr)

    return AugmentedLagrangianResult(
        solution=x_curr,
        state=state_curr,
        converged=converged,
        iters=int(state_curr.iteration),
        constraint_norm=float(constraint_norm),
        lambda_update_norm=float(lambda_update_norm),
        info=info_curr,
    )


def solve_unilateral_contact_active_set_kkt(
    stiffness: npt.ArrayLike,
    force: npt.ArrayLike,
    gap_matrix: npt.ArrayLike,
    gap0: npt.ArrayLike,
    *,
    fixed_dofs: npt.ArrayLike | None = None,
    initial_active: npt.ArrayLike | None = None,
    maxiter: int = 30,
    gap_tol: float = 1e-10,
    lambda_tol: float = 1e-10,
    config: ContactKKTSolveConfig | None = None,
) -> UnilateralContactActiveSetResult:
    """
    Solve a linearized unilateral contact problem with an active-set KKT loop.

    The problem is

    ``K u = f + G.T lambda``, ``g = gap0 + G u >= 0``,
    ``lambda >= 0``, and ``g * lambda = 0``.

    Active constraints are solved as equalities ``G_active u = -gap0_active``.
    The active set is then updated by removing negative multipliers and adding
    violated gaps.
    """
    if int(maxiter) <= 0:
        raise ValueError("maxiter must be positive.")

    K = np.asarray(stiffness, dtype=float)
    f = np.asarray(force, dtype=float).reshape(-1)
    G = np.asarray(gap_matrix, dtype=float)
    g0 = np.asarray(gap0, dtype=float).reshape(-1)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("stiffness must be a square matrix.")
    n_dofs = int(K.shape[0])
    if f.shape != (n_dofs,):
        raise ValueError("force must have shape (n_dofs,).")
    if G.ndim != 2 or G.shape[1] != n_dofs:
        raise ValueError("gap_matrix must have shape (n_contacts, n_dofs).")
    n_contacts = int(G.shape[0])
    if g0.shape != (n_contacts,):
        raise ValueError("gap0 must have shape (n_contacts,).")

    fixed = np.asarray([], dtype=np.int32) if fixed_dofs is None else np.asarray(fixed_dofs, dtype=np.int32).reshape(-1)
    if fixed.size:
        if fixed.min() < 0 or fixed.max() >= n_dofs:
            raise ValueError("fixed_dofs contains an index outside the unknown range.")
        if np.unique(fixed).size != fixed.size:
            raise ValueError("fixed_dofs must not contain duplicates.")
    free_mask = np.ones((n_dofs,), dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)

    if initial_active is None:
        try:
            u_trial = np.zeros((n_dofs,), dtype=float)
            if free.size:
                u_trial[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
            active = (g0 + G @ u_trial) < -float(gap_tol)
        except np.linalg.LinAlgError:
            active = g0 < -float(gap_tol)
    else:
        active = np.asarray(initial_active, dtype=bool).reshape(-1)
        if active.shape != (n_contacts,):
            raise ValueError("initial_active must have shape (n_contacts,).")

    cfg = ContactKKTSolveConfig(backend="numpy") if config is None else config.validate()
    records: list[UnilateralContactActiveSetRecord] = []
    u = np.zeros((n_dofs,), dtype=float)
    lambda_full = np.zeros((n_contacts,), dtype=float)
    gap = g0.copy()
    converged = False

    for iteration in range(1, int(maxiter) + 1):
        active_ids = np.flatnonzero(active)
        n_active = int(active_ids.size)
        K_ff = K[np.ix_(free, free)]
        f_f = f[free]
        if n_active:
            G_af = G[np.ix_(active_ids, free)]
            lhs = np.block(
                [
                    [K_ff, -G_af.T],
                    [G_af, np.zeros((n_active, n_active), dtype=K.dtype)],
                ]
            )
            rhs = np.concatenate([f_f, -g0[active_ids]])
            sol = np.asarray(solve_contact_kkt(lhs, rhs, config=cfg), dtype=float)
            u = np.zeros((n_dofs,), dtype=float)
            u[free] = sol[: free.size]
            lambda_full = np.zeros((n_contacts,), dtype=float)
            lambda_full[active_ids] = sol[free.size :]
        else:
            u = np.zeros((n_dofs,), dtype=float)
            if free.size:
                u[free] = np.asarray(solve_contact_kkt(K_ff, f_f, config=cfg), dtype=float)
            lambda_full = np.zeros((n_contacts,), dtype=float)

        gap = g0 + G @ u
        remove = active & (lambda_full < -float(lambda_tol))
        add = (~active) & (gap < -float(gap_tol))
        next_active = (active & ~remove) | add
        changed = not np.array_equal(next_active, active)
        min_lambda = float(np.min(lambda_full[active])) if np.any(active) else 0.0
        records.append(
            UnilateralContactActiveSetRecord(
                iteration=iteration,
                active_count=int(np.count_nonzero(active)),
                min_gap=float(np.min(gap)) if gap.size else 0.0,
                min_lambda=min_lambda,
                changed=changed,
            )
        )
        active = next_active
        if not changed:
            converged = bool(np.all(gap >= -float(gap_tol)) and np.all(lambda_full[active] >= -float(lambda_tol)))
            break

    return UnilateralContactActiveSetResult(
        displacement=u,
        lambda_n=lambda_full,
        gap=gap,
        active_mask=active,
        converged=converged,
        iters=len(records),
        records=tuple(records),
    )

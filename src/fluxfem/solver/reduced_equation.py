from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Any

import jax
import jax.numpy as jnp
import numpy as np


Array = Any
ResidualFn = Callable[..., Array]


@dataclass(frozen=True)
class ReducedEquationField:
    """Named reduced field owned by a residual-first reduced problem."""

    name: str
    offset: int
    n_dofs: int
    basis: Any | None = None

    @property
    def stop(self) -> int:
        return self.offset + self.n_dofs

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.stop)

    def expand(self, q_field: Array) -> Array:
        if self.basis is None:
            return jnp.asarray(q_field)
        return self.basis.expand(q_field)


@dataclass(frozen=True)
class _FieldResidualBlock:
    field: str
    residual: ResidualFn


@dataclass(frozen=True)
class _CouplingResidualBlock:
    fields: tuple[str, ...]
    residual: ResidualFn


def _call_residual(fn: ResidualFn, args: tuple[Array, ...], params: Any) -> Array:
    if params is None:
        return fn(*args)
    return fn(*args, params)


def _constraint_residual(constraint: Any) -> ResidualFn:
    if hasattr(constraint, "residual"):
        residual = constraint.residual
    else:
        residual = constraint
    if not callable(residual):
        raise TypeError("constraint must be callable or expose a callable residual method.")
    return residual


def _field_keys(fields: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(fields, str):
        return (fields,)
    return tuple(str(field) for field in fields)


class ReducedEquationProblem:
    """Composable reduced residual problem with autodiff Jacobian support."""

    def __init__(
        self,
        fields: Mapping[str, ReducedEquationField],
        field_blocks: Sequence[_FieldResidualBlock],
        coupling_blocks: Sequence[_CouplingResidualBlock],
    ):
        self.fields = dict(fields)
        self.field_blocks = tuple(field_blocks)
        self.coupling_blocks = tuple(coupling_blocks)
        self.n_dofs = sum(field.n_dofs for field in self.fields.values())

    def field(self, name: str) -> ReducedEquationField:
        return self.fields[str(name)]

    def field_vector(self, q: Array, name: str) -> Array:
        field = self.field(name)
        return jnp.asarray(q)[field.slice]

    def split(self, q: Array) -> dict[str, Array]:
        return {name: self.field_vector(q, name) for name in self.fields}

    def expand_field(self, q: Array, name: str) -> Array:
        field = self.field(name)
        return field.expand(self.field_vector(q, name))

    def expand(self, q: Array) -> dict[str, Array]:
        return {name: self.expand_field(q, name) for name in self.fields}

    def zeros(self, *, dtype=None) -> Array:
        dtype_use = jnp.float64 if dtype is None else dtype
        return jnp.zeros((self.n_dofs,), dtype=dtype_use)

    def _scatter_add(self, total: Array, field_name: str, value: Array) -> Array:
        field = self.field(field_name)
        value_arr = jnp.asarray(value, dtype=total.dtype).reshape(-1)
        if value_arr.shape != (field.n_dofs,):
            raise ValueError(
                f"Residual for field '{field_name}' must have shape {(field.n_dofs,)}, "
                f"got {value_arr.shape}."
            )
        return total.at[field.slice].add(value_arr)

    def _add_coupling_output(self, total: Array, block: _CouplingResidualBlock, output: Array) -> Array:
        if isinstance(output, Mapping):
            for field_name, value in output.items():
                if str(field_name) not in block.fields:
                    raise ValueError(f"Coupling residual returned unknown field '{field_name}'.")
                total = self._scatter_add(total, str(field_name), value)
            return total

        if isinstance(output, (tuple, list)):
            if len(output) != len(block.fields):
                raise ValueError("Coupling residual tuple/list length must match the number of fields.")
            for field_name, value in zip(block.fields, output):
                total = self._scatter_add(total, field_name, value)
            return total

        flat = jnp.asarray(output, dtype=total.dtype).reshape(-1)
        expected = sum(self.field(field).n_dofs for field in block.fields)
        if flat.shape != (expected,):
            raise ValueError(f"Coupling residual vector must have shape {(expected,)}, got {flat.shape}.")
        cursor = 0
        for field_name in block.fields:
            n = self.field(field_name).n_dofs
            total = self._scatter_add(total, field_name, flat[cursor : cursor + n])
            cursor += n
        return total

    def residual(self, q: Array, params: Any = None) -> Array:
        q_arr = jnp.asarray(q)
        total = jnp.zeros((self.n_dofs,), dtype=q_arr.dtype)
        for block in self.field_blocks:
            q_field = self.field_vector(q_arr, block.field)
            value = _call_residual(block.residual, (q_field,), params)
            total = self._scatter_add(total, block.field, value)
        for block in self.coupling_blocks:
            args = tuple(self.field_vector(q_arr, field) for field in block.fields)
            output = _call_residual(block.residual, args, params)
            total = self._add_coupling_output(total, block, output)
        return total

    def jacobian(self, q: Array, params: Any = None) -> Array:
        return jax.jacrev(lambda x: self.residual(x, params))(q)

    def solve(
        self,
        q0: Array,
        params: Any = None,
        *,
        fixed_dofs: Array | None = None,
        fixed_values: Array | None = None,
        tol: float = 1e-8,
        atol: float = 0.0,
        maxiter: int = 20,
    ) -> tuple[Array, "ReducedEquationSolveInfo"]:
        return solve_reduced_equation(
            self,
            q0,
            params,
            fixed_dofs=fixed_dofs,
            fixed_values=fixed_values,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
        )


class ReducedEquationBuilder:
    """Build a global reduced residual from named reduced fields."""

    def __init__(self):
        self._fields: dict[str, ReducedEquationField] = {}
        self._field_blocks: list[_FieldResidualBlock] = []
        self._coupling_blocks: list[_CouplingResidualBlock] = []

    @property
    def fields(self) -> Mapping[str, ReducedEquationField]:
        return dict(self._fields)

    def _next_offset(self) -> int:
        return sum(field.n_dofs for field in self._fields.values())

    def register_field(self, name: str, *, n_dofs: int | None = None, basis: Any | None = None) -> ReducedEquationField:
        key = str(name)
        if key in self._fields:
            raise ValueError(f"Reduced field '{key}' is already registered.")
        if basis is not None:
            basis_n = int(basis.n_reduced) if hasattr(basis, "n_reduced") else int(basis.basis.shape[1])
            if n_dofs is not None and int(n_dofs) != basis_n:
                raise ValueError("n_dofs must match basis.n_reduced when basis is provided.")
            n = basis_n
        elif n_dofs is not None:
            n = int(n_dofs)
        else:
            raise ValueError("register_field requires n_dofs or basis.")
        if n <= 0:
            raise ValueError("n_dofs must be positive.")
        field = ReducedEquationField(name=key, offset=self._next_offset(), n_dofs=n, basis=basis)
        self._fields[key] = field
        return field

    register_reduced_field = register_field

    def add_field_residual(self, field: str, residual: ResidualFn) -> "ReducedEquationBuilder":
        key = str(field)
        if key not in self._fields:
            raise ValueError(f"Reduced field '{key}' is not registered.")
        self._field_blocks.append(_FieldResidualBlock(field=key, residual=residual))
        return self

    def add_coupling_residual(self, fields: Sequence[str], residual: ResidualFn) -> "ReducedEquationBuilder":
        keys = _field_keys(fields)
        if len(keys) < 2:
            raise ValueError("A coupling residual must reference at least two fields.")
        for key in keys:
            if key not in self._fields:
                raise ValueError(f"Reduced field '{key}' is not registered.")
        self._coupling_blocks.append(_CouplingResidualBlock(fields=keys, residual=residual))
        return self

    def add_constraint(
        self,
        fields: str | Sequence[str] | Any,
        constraint: Any | None = None,
    ) -> "ReducedEquationBuilder":
        """Register a user constraint residual on one or more reduced fields.

        ``constraint`` may be a callable or an object exposing
        ``residual(...)``.  If ``constraint`` is omitted, ``fields`` is treated
        as a constraint object and must expose a ``fields`` attribute.
        """
        if constraint is None:
            constraint = fields
            if not hasattr(constraint, "fields"):
                raise ValueError("A constraint object must expose fields when fields are not passed explicitly.")
            keys = _field_keys(constraint.fields)
        else:
            keys = _field_keys(fields)

        residual = _constraint_residual(constraint)
        if len(keys) == 1:
            return self.add_field_residual(keys[0], residual)
        return self.add_coupling_residual(keys, residual)

    def build(self) -> ReducedEquationProblem:
        if not self._fields:
            raise ValueError("At least one reduced field is required.")
        return ReducedEquationProblem(self._fields, self._field_blocks, self._coupling_blocks)


@dataclass(frozen=True)
class ReducedEquationSolveInfo:
    """Newton solve summary for a reduced equation problem."""

    converged: bool
    iters: int
    residual_norm: float
    residual0: float
    rel_residual: float
    stop_reason: str


def _normalized_fixed_dofs(n_dofs: int, fixed_dofs: Array | None) -> tuple[jnp.ndarray, jnp.ndarray]:
    if fixed_dofs is None:
        fixed = jnp.zeros((0,), dtype=jnp.int32)
    else:
        fixed = jnp.asarray(fixed_dofs, dtype=jnp.int32).reshape(-1)
    if fixed.size:
        fixed_np = np.asarray(fixed)
        if fixed_np.min() < 0 or fixed_np.max() >= n_dofs:
            raise ValueError("fixed_dofs contains an index outside the reduced problem.")
        fixed = jnp.asarray(np.unique(fixed_np).astype(np.int32))
    all_dofs = np.arange(n_dofs, dtype=np.int32)
    if fixed.size:
        mask = np.ones((n_dofs,), dtype=bool)
        mask[np.asarray(fixed, dtype=np.int32)] = False
        free = jnp.asarray(all_dofs[mask], dtype=jnp.int32)
    else:
        free = jnp.asarray(all_dofs, dtype=jnp.int32)
    return fixed, free


def _apply_fixed_values(q: Array, fixed_dofs: Array, fixed_values: Array | None) -> Array:
    q_arr = jnp.asarray(q)
    fixed = jnp.asarray(fixed_dofs, dtype=jnp.int32).reshape(-1)
    if fixed.size == 0:
        return q_arr
    if fixed_values is None:
        values = jnp.zeros((fixed.size,), dtype=q_arr.dtype)
    else:
        values = jnp.asarray(fixed_values, dtype=q_arr.dtype).reshape(-1)
        if values.size == 1 and fixed.size != 1:
            values = jnp.full((fixed.size,), values[0], dtype=q_arr.dtype)
        if values.shape != (fixed.size,):
            raise ValueError("fixed_values must be scalar or match fixed_dofs.")
    return q_arr.at[fixed].set(values)


def solve_reduced_equation(
    problem: ReducedEquationProblem,
    q0: Array,
    params: Any = None,
    *,
    fixed_dofs: Array | None = None,
    fixed_values: Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
) -> tuple[Array, ReducedEquationSolveInfo]:
    """Solve a reduced residual equation with dense Newton iterations."""
    q = jnp.asarray(q0)
    if q.shape != (problem.n_dofs,):
        raise ValueError(f"q0 must have shape {(problem.n_dofs,)}, got {q.shape}.")
    fixed, free = _normalized_fixed_dofs(problem.n_dofs, fixed_dofs)
    q = _apply_fixed_values(q, fixed, fixed_values)

    def free_residual(q_current):
        return problem.residual(q_current, params)[free]

    residual = free_residual(q)
    residual0 = float(jnp.linalg.norm(residual, ord=jnp.inf))
    threshold = max(float(atol), float(tol) * residual0)
    if residual0 <= threshold:
        return q, ReducedEquationSolveInfo(True, 0, residual0, residual0, 1.0, "initial_converged")

    final_norm = residual0
    for iteration in range(1, int(maxiter) + 1):
        jacobian = problem.jacobian(q, params)
        jac_free = jacobian[jnp.ix_(free, free)]
        delta_free = jnp.linalg.solve(jac_free, -residual)
        q = q.at[free].add(delta_free)
        q = _apply_fixed_values(q, fixed, fixed_values)
        residual = free_residual(q)
        final_norm = float(jnp.linalg.norm(residual, ord=jnp.inf))
        if final_norm <= threshold:
            rel = final_norm / max(residual0, 1.0e-30)
            return q, ReducedEquationSolveInfo(True, iteration, final_norm, residual0, rel, "converged")

    rel = final_norm / max(residual0, 1.0e-30)
    return q, ReducedEquationSolveInfo(False, int(maxiter), final_norm, residual0, rel, "maxiter")


def solve_reduced_equation_active(
    q0: Array,
    initial_state: Any,
    problem_from_state: Callable[[Any], ReducedEquationProblem],
    update_state: Callable[[Array], Any],
    params: Any = None,
    *,
    fixed_dofs: Array | None = None,
    fixed_values: Array | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    state_changed: Callable[[Any, Any], bool] | None = None,
    max_active_updates: int = 8,
) -> tuple[Array, Any]:
    """Solve a reduced equation with an outer active/contact-state loop."""
    from .craig_bampton import active_contact_fixed_point_solve

    q_initial = jnp.asarray(q0)
    n_dofs = int(q_initial.size)

    def residual_from_state(state: Any) -> Callable[[Array], Array]:
        problem = problem_from_state(state)
        if problem.n_dofs != n_dofs:
            raise ValueError(f"problem_from_state returned {problem.n_dofs} DOFs, expected {n_dofs}.")
        return lambda q: problem.residual(q, params)

    def solve_fn(residual_fn: Callable[[Array], Array], q_init: Array) -> tuple[Array, ReducedEquationSolveInfo]:
        class _ResidualProblem:
            def residual(self, q: Array, _params: Any = None) -> Array:
                return residual_fn(q)

            def jacobian(self, q: Array, _params: Any = None) -> Array:
                return jax.jacrev(residual_fn)(q)

        _ResidualProblem.n_dofs = n_dofs
        return solve_reduced_equation(
            _ResidualProblem(),
            q_init,
            fixed_dofs=fixed_dofs,
            fixed_values=fixed_values,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
        )

    return active_contact_fixed_point_solve(
        q_initial,
        initial_state,
        residual_from_state,
        solve_fn,
        update_state,
        state_changed=state_changed,
        max_active_updates=max_active_updates,
    )


def _checked_square_matrix(matrix: Array, n_dofs: int, name: str) -> Array:
    arr = jnp.asarray(matrix)
    if arr.shape != (n_dofs, n_dofs):
        raise ValueError(f"{name} must have shape {(n_dofs, n_dofs)}, got {arr.shape}.")
    return arr


def _external_force_next(external_force: Array | Callable[[float], Array], state: Any, config: Any) -> Array:
    if callable(external_force):
        return external_force(float(state.t) + float(config.dt))
    return external_force


def make_reduced_equation_newmark_residual(
    problem: ReducedEquationProblem,
    mass: Array,
    damping: Array | None,
    external_force: Array | Callable[[float], Array],
    state: Any,
    config: Any,
    params: Any = None,
) -> Callable[[Array], Array]:
    """Build the implicit Newmark residual for a reduced equation problem."""
    from .craig_bampton import newmark_kinematics

    mass_arr = _checked_square_matrix(mass, problem.n_dofs, "mass")
    damping_arr = None if damping is None else _checked_square_matrix(damping, problem.n_dofs, "damping")
    force = jnp.asarray(_external_force_next(external_force, state, config))
    if force.shape != (problem.n_dofs,):
        raise ValueError(f"external_force must have shape {(problem.n_dofs,)}, got {force.shape}.")

    def _residual(q_next: Array) -> Array:
        qd_next, qdd_next = newmark_kinematics(q_next, state, config)
        residual = mass_arr @ qdd_next + problem.residual(q_next, params) - force
        if damping_arr is not None:
            residual = residual + damping_arr @ qd_next
        return residual

    return _residual


def reduced_equation_newmark_step(
    problem: ReducedEquationProblem,
    mass: Array,
    damping: Array | None,
    external_force: Array | Callable[[float], Array],
    state: Any,
    config: Any,
    params: Any = None,
    *,
    q_initial: Array | None = None,
    fixed_dofs: Array | None = None,
    fixed_values: Array | None = None,
) -> tuple[Any, ReducedEquationSolveInfo]:
    """Solve one implicit Newmark step using ``problem.residual`` as internal force."""
    from .craig_bampton import NewmarkState, newmark_kinematics

    q = jnp.asarray(state.q)
    if q.shape != (problem.n_dofs,):
        raise ValueError(f"state.q must have shape {(problem.n_dofs,)}, got {q.shape}.")
    dt = float(config.dt)
    beta = float(config.beta)
    q_pred = q + dt * jnp.asarray(state.qd) + dt**2 * (0.5 - beta) * jnp.asarray(state.qdd)
    q0 = jnp.asarray(q_initial) if q_initial is not None else q_pred
    residual_fn = make_reduced_equation_newmark_residual(problem, mass, damping, external_force, state, config, params)

    class _EffectiveProblem:
        n_dofs = problem.n_dofs

        def residual(self, q_next: Array, _params: Any = None) -> Array:
            return residual_fn(q_next)

        def jacobian(self, q_next: Array, _params: Any = None) -> Array:
            return jax.jacrev(residual_fn)(q_next)

    q_next, info = solve_reduced_equation(
        _EffectiveProblem(),
        q0,
        fixed_dofs=fixed_dofs,
        fixed_values=fixed_values,
        tol=float(config.tol),
        atol=float(config.atol),
        maxiter=int(config.maxiter),
    )
    qd_next, qdd_next = newmark_kinematics(q_next, state, config)
    next_state = NewmarkState(q=q_next, qd=qd_next, qdd=qdd_next, t=float(state.t) + dt)
    return next_state, info


def reduced_equation_active_newmark_step(
    problem_from_state: Callable[[Any], ReducedEquationProblem],
    mass: Array,
    damping: Array | None,
    external_force: Array | Callable[[float], Array],
    state: Any,
    config: Any,
    initial_state: Any,
    update_state: Callable[[Array], Any],
    params: Any = None,
    *,
    state_changed: Callable[[Any, Any], bool] | None = None,
    max_active_updates: int = 8,
    q_initial: Array | None = None,
) -> tuple[Any, Any]:
    """Solve one reduced Newmark step with an outer active/contact-state loop."""
    from .craig_bampton import active_contact_newmark_step

    n_dofs = int(jnp.asarray(state.q).size)
    force = jnp.asarray(_external_force_next(external_force, state, config))
    if force.shape != (n_dofs,):
        raise ValueError(f"external_force must have shape {(n_dofs,)}, got {force.shape}.")

    def internal_force_from_state(active_state: Any) -> Callable[[Array], Array]:
        problem = problem_from_state(active_state)
        if problem.n_dofs != n_dofs:
            raise ValueError(f"problem_from_state returned {problem.n_dofs} DOFs, expected {n_dofs}.")
        return lambda q: problem.residual(q, params)

    return active_contact_newmark_step(
        mass,
        damping,
        internal_force_from_state,
        force,
        state,
        config,
        initial_state,
        update_state,
        state_changed=state_changed,
        max_active_updates=max_active_updates,
        q_initial=q_initial,
    )


__all__ = [
    "ReducedEquationBuilder",
    "ReducedEquationField",
    "ReducedEquationProblem",
    "ReducedEquationSolveInfo",
    "make_reduced_equation_newmark_residual",
    "reduced_equation_active_newmark_step",
    "reduced_equation_newmark_step",
    "solve_reduced_equation_active",
    "solve_reduced_equation",
]

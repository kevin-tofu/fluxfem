from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Any

import jax
import jax.numpy as jnp


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
        keys = tuple(str(field) for field in fields)
        if len(keys) < 2:
            raise ValueError("A coupling residual must reference at least two fields.")
        for key in keys:
            if key not in self._fields:
                raise ValueError(f"Reduced field '{key}' is not registered.")
        self._coupling_blocks.append(_CouplingResidualBlock(fields=keys, residual=residual))
        return self

    def build(self) -> ReducedEquationProblem:
        if not self._fields:
            raise ValueError("At least one reduced field is required.")
        return ReducedEquationProblem(self._fields, self._field_blocks, self._coupling_blocks)


__all__ = [
    "ReducedEquationBuilder",
    "ReducedEquationField",
    "ReducedEquationProblem",
]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING, TypeAlias, TypeVar, cast

import numpy as np
import jax.numpy as jnp

from .dtypes import INDEX_DTYPE
from .forms import MixedFormContext, FieldPair
from .weakform import MixedWeakForm, compile_mixed_residual, make_mixed_residuals
from ..solver.dirichlet import DirichletBC, free_dofs
from ..solver.sparse import FluxSparseMatrix
from .space import FESpaceClosure

P = TypeVar("P")

if TYPE_CHECKING:
    from .assembly import JacobianReturn, LinearReturn

MixedResidualForm: TypeAlias = Callable[
    [MixedFormContext, Mapping[str, jnp.ndarray], P],
    Mapping[str, jnp.ndarray],
]


@dataclass(eq=False)
class MixedFESpace:
    """
    Mixed FE space composed of multiple scalar/vector spaces.

    Field DOFs are concatenated in field order:
      [field0 dofs | field1 dofs | ...]
    """
    fields: dict[str, FESpaceClosure]
    field_order: Sequence[str] | None = None
    field_names: tuple[str, ...] = field(init=False)
    field_offsets: dict[str, int] = field(init=False)
    field_slices: dict[str, slice] = field(init=False)
    elem_slices: dict[str, slice] = field(init=False)
    elem_dofs_by_field: dict[str, jnp.ndarray] = field(init=False)
    elem_dofs: jnp.ndarray = field(init=False)
    n_dofs: int = field(init=False)
    n_ldofs: int = field(init=False)

    def __post_init__(self):
        if not self.fields:
            raise ValueError("MixedFESpace requires at least one field.")

        if self.field_order is None:
            self.field_names = tuple(self.fields.keys())
        else:
            self.field_names = tuple(self.field_order)
            missing = set(self.fields.keys()) - set(self.field_names)
            extra = set(self.field_names) - set(self.fields.keys())
            if missing or extra:
                raise ValueError(f"field_order mismatch: missing={missing}, extra={extra}")

        ref_space = self.fields[self.field_names[0]]
        ref_mesh = ref_space.mesh
        ref_basis = ref_space.basis
        n_elems = int(ref_space.elem_dofs.shape[0])

        offsets: dict[str, int] = {}
        slices: dict[str, slice] = {}
        elem_slices: dict[str, slice] = {}
        elem_dofs_by_field: dict[str, jnp.ndarray] = {}
        elem_dofs_list = []

        dof_offset = 0
        ldof_offset = 0
        for name in self.field_names:
            space = self.fields[name]
            if space.mesh is not ref_mesh:
                raise ValueError("All mixed fields must share the same mesh object.")
            if space.basis.__class__ is not ref_basis.__class__:
                raise ValueError("All mixed fields must share the same basis type.")
            if int(space.elem_dofs.shape[0]) != n_elems:
                raise ValueError("All mixed fields must have the same element count.")

            n_dofs = int(space.n_dofs)
            n_ldofs = int(space.n_ldofs)
            offsets[name] = dof_offset
            slices[name] = slice(dof_offset, dof_offset + n_dofs)
            elem_slices[name] = slice(ldof_offset, ldof_offset + n_ldofs)

            elem_dofs = jnp.asarray(space.elem_dofs, dtype=INDEX_DTYPE) + dof_offset
            elem_dofs_by_field[name] = elem_dofs
            elem_dofs_list.append(elem_dofs)

            dof_offset += n_dofs
            ldof_offset += n_ldofs

        self.field_offsets = offsets
        self.field_slices = slices
        self.elem_slices = elem_slices
        self.elem_dofs_by_field = elem_dofs_by_field
        self.elem_dofs = jnp.concatenate(elem_dofs_list, axis=1)
        self.n_dofs = dof_offset
        self.n_ldofs = ldof_offset

    def pack_fields(self, fields: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """Concatenate per-field vectors into a single mixed vector."""
        parts = []
        for name in self.field_names:
            if name not in fields:
                raise KeyError(f"Missing field '{name}' in pack_fields.")
            parts.append(jnp.asarray(fields[name]))
        return jnp.concatenate(parts, axis=0)

    def unpack_fields(self, u: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Split a mixed vector into per-field vectors."""
        u = jnp.asarray(u)
        return {name: u[self.field_slices[name]] for name in self.field_names}

    def split_element_vector(self, u_elem: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Split an element-local mixed vector into per-field element vectors."""
        return {name: u_elem[self.elem_slices[name]] for name in self.field_names}

    def build_form_contexts(self, dep: jnp.ndarray | None = None) -> MixedFormContext:
        ctxs_by_field = {name: sp.build_form_contexts(dep) for name, sp in self.fields.items()}
        ref_ctx = ctxs_by_field[self.field_names[0]]

        fields = {
            name: FieldPair(test=ctx.test, trial=ctx.trial, unknown=None)
            for name, ctx in ctxs_by_field.items()
        }
        trial_fields = {name: ctx.trial for name, ctx in ctxs_by_field.items()}
        test_fields = {name: ctx.test for name, ctx in ctxs_by_field.items()}
        unknown_fields = {name: ctx.trial for name, ctx in ctxs_by_field.items()}

        return MixedFormContext(
            fields=fields,
            x_q=ref_ctx.x_q,
            w=ref_ctx.w,
            elem_id=ref_ctx.elem_id,
            trial_fields=trial_fields,
            test_fields=test_fields,
            unknown_fields=unknown_fields,
        )

    def get_sparsity_pattern(self, *, with_idx: bool = True):
        from .assembly import make_sparsity_pattern
        return make_sparsity_pattern(cast(Any, self), with_idx=with_idx)

    def assemble_residual(
        self,
        res_form: MixedResidualForm[P],
        u: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray] | jnp.ndarray,
        params: P,
        **kwargs,
    ) -> "LinearReturn":
        from .mixed_assembly import assemble_mixed_residual
        return assemble_mixed_residual(self, res_form, u, params, **kwargs)

    def assemble_jacobian(
        self,
        res_form: MixedResidualForm[P],
        u: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray] | jnp.ndarray,
        params: P,
        **kwargs,
    ) -> "JacobianReturn":
        from .mixed_assembly import assemble_mixed_jacobian
        return assemble_mixed_jacobian(self, res_form, u, params, **kwargs)

    def make_dirichlet(self, *, merge: str = "check_equal", **fields):
        """
        Build mixed Dirichlet BCs from per-field constraints.

        Usage:
          bc = mixed.make_dirichlet(u=DirichletBC(...), T=(dofs, vals))
        """
        if merge not in {"check_equal", "error", "first", "last"}:
            raise ValueError("merge must be one of: check_equal, error, first, last")

        dof_map: dict[int, float] = {}
        for name, spec in fields.items():
            if name not in self.field_offsets:
                raise KeyError(f"Unknown mixed field: {name}")
            offset = int(self.field_offsets[name])
            if isinstance(spec, DirichletBC):
                dofs = spec.dofs
                vals = spec.vals
            elif isinstance(spec, tuple) and len(spec) == 2:
                dofs, vals = spec
            else:
                dofs, vals = spec, None
            bc = DirichletBC(dofs, vals)
            g_dofs = np.asarray(bc.dofs, dtype=int) + offset
            g_vals = np.asarray(bc.vals, dtype=float)
            for d, v in zip(g_dofs, g_vals):
                if d in dof_map:
                    if merge == "error":
                        raise ValueError(f"Duplicate Dirichlet DOF {d} in mixed BCs")
                    if merge == "check_equal":
                        if not np.isclose(dof_map[d], v):
                            raise ValueError(f"Conflicting Dirichlet value for DOF {d}")
                    if merge == "first":
                        continue
                dof_map[d] = float(v)

        if not dof_map:
            return MixedDirichletBC(np.array([], dtype=int), np.array([], dtype=float))
        dofs_sorted = np.array(sorted(dof_map.keys()), dtype=int)
        vals_sorted = np.array([dof_map[d] for d in dofs_sorted], dtype=float)
        return MixedDirichletBC(dofs_sorted, vals_sorted)

    def build_block_system(
        self,
        *,
        diag: Mapping[str, object] | Sequence[object],
        rel: Mapping[tuple[str, str], object] | None = None,
        add_contiguous: object | None = None,
        rhs: Mapping[str, object] | Sequence[object] | np.ndarray | None = None,
        constraints=None,
        merge: str = "check_equal",
        format: str = "auto",
        symmetric: bool = False,
        transpose_rule: str = "T",
    ):
        """
        Build a mixed block system and apply optional constraints.
        """
        from ..solver.block_system import build_block_system as _build_block_system

        sizes = {name: int(self.fields[name].n_dofs) for name in self.field_names}

        if isinstance(constraints, MixedDirichletBC):
            constraints = constraints.as_dirichlet_bc()

        system = _build_block_system(
            diag=diag,
            rel=rel,
            add_contiguous=add_contiguous,
            rhs=rhs,
            constraints=constraints,
            merge=merge,
            sizes=sizes,
            format=format,
            symmetric=symmetric,
            transpose_rule=transpose_rule,
        )
        bc = MixedDirichletBC(system.dirichlet.dofs, system.dirichlet.vals)
        return MixedBlockSystem(self, system.K, system.F, free_dofs=system.free_dofs, dirichlet=bc)


@dataclass(eq=False)
class MixedProblem:
    """
    Lightweight wrapper for mixed residual assembly with cached compilation.
    """
    space: MixedFESpace
    residuals: dict[str, Callable] | MixedWeakForm
    params: object | None = None
    pattern: object | None = None
    n_chunks: int | None = None
    pad_trace: bool = False
    _compiled: Callable[..., Any] = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.residuals, MixedWeakForm):
            self._compiled = self.residuals.get_compiled()
        else:
            res = make_mixed_residuals(self.residuals)
            self._compiled = compile_mixed_residual(res)

    def _merge_kwargs(self, kwargs):
        merged = dict(kwargs)
        if self.pattern is not None and "pattern" not in merged:
            merged["pattern"] = self.pattern
        if self.n_chunks is not None and "n_chunks" not in merged:
            merged["n_chunks"] = self.n_chunks
        if self.pad_trace and "pad_trace" not in merged:
            merged["pad_trace"] = True
        return merged

    def _wrap_params(self, params):
        if callable(params):
            def _wrapped(ctx, u_elem, _params):
                return self._compiled(ctx, u_elem, params(ctx))

            _wrapped._includes_measure = getattr(self._compiled, "_includes_measure", False)  # type: ignore[attr-defined]
            return _wrapped, None
        return self._compiled, params

    def assemble_residual(
        self,
        u: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray] | jnp.ndarray,
        *,
        params: P | None = None,
        **kwargs,
    ) -> "LinearReturn":
        use_params = self.params if params is None else params
        res_form, use_params = self._wrap_params(use_params)
        return self.space.assemble_residual(
            res_form, u, use_params, **self._merge_kwargs(kwargs)
        )

    def assemble_jacobian(
        self,
        u: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray] | jnp.ndarray,
        *,
        params: P | None = None,
        **kwargs,
    ) -> "JacobianReturn":
        use_params = self.params if params is None else params
        res_form, use_params = self._wrap_params(use_params)
        return self.space.assemble_jacobian(
            res_form, u, use_params, **self._merge_kwargs(kwargs)
        )

    def with_params(self, params):
        return MixedProblem(
            self.space,
            self.residuals,
            params=params,
            pattern=self.pattern,
            n_chunks=self.n_chunks,
            pad_trace=self.pad_trace,
        )

    def solve(
        self,
        K,
        F,
        *,
        dirichlet=None,
        dirichlet_mode: str = "condense",
        solver=None,
        n_total: int | None = None,
    ):
        """
        Solve a mixed linear system with optional Dirichlet conditions.
        """
        from ..solver import LinearSolver

        if solver is None:
            solver = LinearSolver()
        if isinstance(dirichlet, MixedDirichletBC):
            dirichlet = dirichlet.as_dirichlet_bc()
        return solver.solve(K, F, dirichlet=dirichlet, dirichlet_mode=dirichlet_mode, n_total=n_total)

@dataclass(frozen=True)
class MixedDirichletBC:
    """
    Mixed-system Dirichlet BCs in global mixed DOF numbering.
    """
    dir_dofs: np.ndarray
    dir_vals: np.ndarray

    def as_dirichlet_bc(self) -> DirichletBC:
        return DirichletBC(self.dir_dofs, self.dir_vals)

    def condense_system(self, A, F, *, check: bool = True):
        return self.as_dirichlet_bc().condense_system(A, F, check=check)

    def free_dofs(self, n_dofs: int) -> np.ndarray:
        return free_dofs(n_dofs, self.dir_dofs)

    def expand_solution(self, u_free, *, free=None, n_total: int | None = None):
        return self.as_dirichlet_bc().expand_solution(u_free, free=free, n_total=n_total)


@dataclass(frozen=True)
class MixedBlockSystem:
    mixed: MixedFESpace
    K: object
    F: object
    free_dofs: np.ndarray
    dirichlet: MixedDirichletBC

    def expand(self, u_free):
        return self.dirichlet.expand_solution(u_free, free=self.free_dofs, n_total=self.mixed.n_dofs)

    def split(self, u_full: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self.mixed.unpack_fields(u_full)

    def join(self, fields: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return self.mixed.pack_fields(fields)


__all__ = ["MixedFESpace", "MixedProblem", "MixedDirichletBC", "MixedBlockSystem"]

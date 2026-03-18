from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING, TypeAlias, TypeVar, cast
import warnings

import numpy as np
import jax.numpy as jnp

from .dtypes import INDEX_DTYPE
from .forms import MixedFormContext, FieldPair
from .weakform import MixedWeakForm, compile_mixed_residual, make_mixed_residuals
from ..solver.dirichlet import DirichletBC, free_dofs
from ..solver.sparse import FluxSparseMatrix, FluxSparseOperator
from .space import (
    FESpaceClosure,
    NamedSpace,
    LinearSpaces,
    BilinearSpaces,
    ResidualSpaces,
    JacobianSpaces,
)

P = TypeVar("P")

if TYPE_CHECKING:
    from .assembly import JacobianReturn, LinearReturn

MixedResidualForm: TypeAlias = Callable[
    [MixedFormContext, Mapping[str, jnp.ndarray], P],
    Mapping[str, jnp.ndarray],
]


MixedFieldSpec: TypeAlias = NamedSpace | LinearSpaces | BilinearSpaces | ResidualSpaces | JacobianSpaces


def _compile_mixed_residual_like(
    res_form: Any,
) -> MixedResidualForm:
    if isinstance(res_form, MixedWeakForm):
        return cast(MixedResidualForm, res_form.get_compiled())
    get_compiled = getattr(res_form, "get_compiled", None)
    if callable(get_compiled):
        return cast(MixedResidualForm, get_compiled())
    return cast(MixedResidualForm, res_form)


def _normalize_mixed_field_spec(field_name: str, spec: MixedFieldSpec) -> tuple[FESpaceClosure, str, tuple[str, ...]]:
    if isinstance(spec, NamedSpace):
        return spec.space, spec.name, ()
    if isinstance(spec, LinearSpaces):
        return spec.test.space, spec.test.name, ()
    if isinstance(spec, ResidualSpaces):
        if spec.test.space is not spec.unknown.space:
            raise ValueError(
                f"MixedSpaces field '{field_name}' currently requires ResidualSpaces "
                "to use the same FE space for test and unknown because mixed assembly "
                "still packs one unknown vector and one residual block per field."
            )
        aliases = tuple(name for name in (spec.test.name, spec.unknown.name) if name != spec.unknown.name)
        return spec.unknown.space, spec.unknown.name, aliases
    if isinstance(spec, BilinearSpaces):
        if spec.test.space is not spec.trial.space:
            raise ValueError(
                f"MixedSpaces field '{field_name}' currently requires BilinearSpaces "
                "to use the same FE space for test and trial because mixed assembly "
                "still assumes one square block per field."
            )
        aliases = tuple(name for name in (spec.test.name, spec.trial.name) if name != spec.trial.name)
        return spec.trial.space, spec.trial.name, aliases
    if isinstance(spec, JacobianSpaces):
        if spec.test.space is not spec.trial.space:
            raise ValueError(
                f"MixedSpaces field '{field_name}' currently requires JacobianSpaces "
                "to use the same FE space for test and trial because mixed assembly "
                "still assumes one square block per field."
            )
        aliases = tuple(name for name in (spec.test.name, spec.trial.name) if name != spec.trial.name)
        return spec.trial.space, spec.trial.name, aliases
    raise TypeError(
        f"MixedSpaces field '{field_name}' must be a NamedSpace, LinearSpaces, "
        "BilinearSpaces, ResidualSpaces, or JacobianSpaces instance."
    )


def _normalize_mixed_role_field_spec(
    field_name: str,
    spec: MixedFieldSpec,
) -> tuple[NamedSpace, NamedSpace, NamedSpace]:
    if isinstance(spec, NamedSpace):
        return spec, spec, spec
    if isinstance(spec, LinearSpaces):
        return spec.test, spec.test, spec.test
    if isinstance(spec, ResidualSpaces):
        return spec.test, spec.unknown, spec.unknown
    if isinstance(spec, BilinearSpaces):
        return spec.test, spec.trial, spec.trial
    if isinstance(spec, JacobianSpaces):
        return spec.test, spec.trial, spec.trial
    raise TypeError(
        f"MixedRoleSpaces field '{field_name}' must be a NamedSpace, LinearSpaces, "
        "BilinearSpaces, ResidualSpaces, or JacobianSpaces instance."
    )


@dataclass(frozen=True)
class MixedSpaces:
    """Public spec that maps mixed field names to named FE space-role specs."""

    fields: Mapping[str, MixedFieldSpec]

    def __post_init__(self):
        normalized = {str(name): spec for name, spec in self.fields.items()}
        if not normalized:
            raise ValueError("MixedSpaces requires at least one field.")
        for name, spec in normalized.items():
            _normalize_mixed_field_spec(name, spec)
        object.__setattr__(self, "fields", normalized)

    def to_fe_space(self) -> "MixedFESpace":
        fields: dict[str, FESpaceClosure] = {}
        field_to_space_key: dict[str, str] = {}
        field_alias_space_keys: dict[str, tuple[str, ...]] = {}
        for name, spec in self.fields.items():
            space, primary_key, aliases = _normalize_mixed_field_spec(name, spec)
            fields[name] = space
            field_to_space_key[name] = primary_key
            field_alias_space_keys[name] = aliases
        return MixedFESpace(
            fields,
            field_to_space_key=field_to_space_key,
            field_alias_space_keys=field_alias_space_keys,
        )


@dataclass(frozen=True)
class MixedRoleSpaces:
    """
    Experimental mixed spec that preserves distinct role spaces per field.

    Unlike MixedSpaces, this does not collapse each field to a single FE space.
    It is currently a layout/context prototype and is not yet wired into the
    standard mixed assembly helpers.
    """

    fields: Mapping[str, MixedFieldSpec]

    def __post_init__(self):
        normalized = {str(name): spec for name, spec in self.fields.items()}
        if not normalized:
            raise ValueError("MixedRoleSpaces requires at least one field.")
        for name, spec in normalized.items():
            _normalize_mixed_role_field_spec(name, spec)
        object.__setattr__(self, "fields", normalized)

    def to_fe_space(self) -> "MixedRoleFESpace":
        warnings.warn(
            "MixedRoleSpaces is experimental. It currently supports only volume "
            "residual/Jacobian assembly and may change without notice.",
            UserWarning,
            stacklevel=2,
        )
        test_fields: dict[str, FESpaceClosure] = {}
        trial_fields: dict[str, FESpaceClosure] = {}
        unknown_fields: dict[str, FESpaceClosure] = {}
        test_keys: dict[str, str] = {}
        trial_keys: dict[str, str] = {}
        unknown_keys: dict[str, str] = {}
        for name, spec in self.fields.items():
            test_ns, trial_ns, unknown_ns = _normalize_mixed_role_field_spec(name, spec)
            test_fields[name] = test_ns.space
            trial_fields[name] = trial_ns.space
            unknown_fields[name] = unknown_ns.space
            test_keys[name] = test_ns.name
            trial_keys[name] = trial_ns.name
            unknown_keys[name] = unknown_ns.name
        return MixedRoleFESpace(
            test_fields=test_fields,
            trial_fields=trial_fields,
            unknown_fields=unknown_fields,
            test_space_key_by_field=test_keys,
            trial_space_key_by_field=trial_keys,
            unknown_space_key_by_field=unknown_keys,
        )


@dataclass(eq=False)
class MixedRoleFESpace:
    """
    Experimental mixed FE layout with explicit per-role spaces per field.

    This keeps separate unknown and residual layouts so field-internal distinct
    test/unknown or test/trial spaces can be represented without mutating the
    current MixedFESpace invariants.
    """

    test_fields: dict[str, FESpaceClosure]
    trial_fields: dict[str, FESpaceClosure]
    unknown_fields: dict[str, FESpaceClosure]
    field_order: Sequence[str] | None = None
    test_space_key_by_field: Mapping[str, str] | None = None
    trial_space_key_by_field: Mapping[str, str] | None = None
    unknown_space_key_by_field: Mapping[str, str] | None = None
    field_names: tuple[str, ...] = field(init=False)
    residual_field_slices: dict[str, slice] = field(init=False)
    unknown_field_slices: dict[str, slice] = field(init=False)
    residual_elem_slices: dict[str, slice] = field(init=False)
    unknown_elem_slices: dict[str, slice] = field(init=False)
    residual_elem_dofs: jnp.ndarray = field(init=False)
    unknown_elem_dofs: jnp.ndarray = field(init=False)
    n_residual_dofs: int = field(init=False)
    n_unknown_dofs: int = field(init=False)
    n_residual_ldofs: int = field(init=False)
    n_unknown_ldofs: int = field(init=False)

    def __post_init__(self):
        if not self.test_fields or not self.trial_fields or not self.unknown_fields:
            raise ValueError("MixedRoleFESpace requires non-empty role field mappings.")
        field_sets = (set(self.test_fields), set(self.trial_fields), set(self.unknown_fields))
        if len({frozenset(s) for s in field_sets}) != 1:
            raise ValueError("MixedRoleFESpace role field mappings must share the same field names.")
        if self.field_order is None:
            self.field_names = tuple(self.unknown_fields.keys())
        else:
            self.field_names = tuple(self.field_order)
        ref_space = self.unknown_fields[self.field_names[0]]
        ref_mesh = ref_space.mesh
        n_elems = int(ref_space.elem_dofs.shape[0])

        residual_field_slices: dict[str, slice] = {}
        unknown_field_slices: dict[str, slice] = {}
        residual_elem_slices: dict[str, slice] = {}
        unknown_elem_slices: dict[str, slice] = {}
        residual_elem_dofs_list = []
        unknown_elem_dofs_list = []
        residual_offset = 0
        unknown_offset = 0
        residual_ldof_offset = 0
        unknown_ldof_offset = 0

        for name in self.field_names:
            test_space = self.test_fields[name]
            trial_space = self.trial_fields[name]
            unknown_space = self.unknown_fields[name]
            for role_space in (test_space, trial_space, unknown_space):
                if role_space.mesh is not ref_mesh:
                    raise ValueError("All mixed role spaces must share the same mesh object.")
                if int(role_space.elem_dofs.shape[0]) != n_elems:
                    raise ValueError("All mixed role spaces must have the same element count.")

            n_residual = int(test_space.n_dofs)
            n_unknown = int(unknown_space.n_dofs)
            n_residual_ldofs = int(test_space.n_ldofs)
            n_unknown_ldofs = int(unknown_space.n_ldofs)

            residual_field_slices[name] = slice(residual_offset, residual_offset + n_residual)
            unknown_field_slices[name] = slice(unknown_offset, unknown_offset + n_unknown)
            residual_elem_slices[name] = slice(
                residual_ldof_offset, residual_ldof_offset + n_residual_ldofs
            )
            unknown_elem_slices[name] = slice(
                unknown_ldof_offset, unknown_ldof_offset + n_unknown_ldofs
            )
            residual_elem_dofs_list.append(
                jnp.asarray(test_space.elem_dofs, dtype=INDEX_DTYPE) + residual_offset
            )
            unknown_elem_dofs_list.append(
                jnp.asarray(unknown_space.elem_dofs, dtype=INDEX_DTYPE) + unknown_offset
            )

            residual_offset += n_residual
            unknown_offset += n_unknown
            residual_ldof_offset += n_residual_ldofs
            unknown_ldof_offset += n_unknown_ldofs

        self.residual_field_slices = residual_field_slices
        self.unknown_field_slices = unknown_field_slices
        self.residual_elem_slices = residual_elem_slices
        self.unknown_elem_slices = unknown_elem_slices
        self.residual_elem_dofs = jnp.concatenate(residual_elem_dofs_list, axis=1)
        self.unknown_elem_dofs = jnp.concatenate(unknown_elem_dofs_list, axis=1)
        self.n_residual_dofs = residual_offset
        self.n_unknown_dofs = unknown_offset
        self.n_residual_ldofs = residual_ldof_offset
        self.n_unknown_ldofs = unknown_ldof_offset

    def pack_unknown_fields(self, fields: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([jnp.asarray(fields[name]) for name in self.field_names], axis=0)

    def unpack_unknown_fields(self, u: jnp.ndarray) -> dict[str, jnp.ndarray]:
        u = jnp.asarray(u)
        return {name: u[self.unknown_field_slices[name]] for name in self.field_names}

    def split_unknown_element_vector(self, u_elem: jnp.ndarray) -> dict[str, jnp.ndarray]:
        u_elem = jnp.asarray(u_elem)
        out = {name: u_elem[self.unknown_elem_slices[name]] for name in self.field_names}
        for name in self.field_names:
            if self.unknown_space_key_by_field is not None:
                out.setdefault(self.unknown_space_key_by_field.get(name, name), out[name])
        return out

    def build_form_contexts(self, dep: jnp.ndarray | None = None) -> MixedFormContext:
        test_ctxs = {name: sp.build_form_contexts(dep) for name, sp in self.test_fields.items()}
        trial_ctxs = {name: sp.build_form_contexts(dep) for name, sp in self.trial_fields.items()}
        unknown_ctxs = {name: sp.build_form_contexts(dep) for name, sp in self.unknown_fields.items()}
        ref_ctx = next(iter(unknown_ctxs.values()))

        bindings = {}
        spaces = {}
        for name in self.field_names:
            pair = FieldPair(
                test=test_ctxs[name].test,
                trial=trial_ctxs[name].trial,
                unknown=unknown_ctxs[name].trial,
            )
            bindings[name] = pair
            if self.test_space_key_by_field is not None:
                spaces[self.test_space_key_by_field.get(name, name)] = pair
            if self.trial_space_key_by_field is not None:
                spaces[self.trial_space_key_by_field.get(name, name)] = pair
            if self.unknown_space_key_by_field is not None:
                spaces[self.unknown_space_key_by_field.get(name, name)] = pair

        return MixedFormContext(
            bindings=bindings,
            x_q=ref_ctx.x_q,
            w=ref_ctx.w,
            elem_id=ref_ctx.elem_id,
            spaces=spaces,
        )

    def assemble_residual(
        self,
        res_form: MixedResidualForm[P],
        u: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray] | jnp.ndarray,
        params: P,
        **kwargs,
    ):
        from .mixed_assembly import assemble_mixed_role_residual
        return assemble_mixed_role_residual(self, _compile_mixed_residual_like(res_form), u, params, **kwargs)

    def assemble_jacobian(
        self,
        res_form: MixedResidualForm[P],
        u: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray] | jnp.ndarray,
        params: P,
        **kwargs,
    ):
        from .mixed_assembly import assemble_mixed_role_jacobian
        for removed in ("sparse", "return_flux_matrix", "matrix_accumulation"):
            if removed in kwargs:
                raise ValueError(
                    f"{removed} is no longer supported for assemble_jacobian; "
                    "assemble_jacobian now returns FluxSparseOperator/FluxSparseMatrix "
                    "(use .to_dense() when needed)."
                )
        return assemble_mixed_role_jacobian(self, _compile_mixed_residual_like(res_form), u, params, **kwargs)

    def build_block_system(self, *args, **kwargs):
        raise NotImplementedError(
            "MixedRoleFESpace build_block_system is not supported yet. "
            "The current block-system helpers assume one square unknown block per field, "
            "while MixedRoleFESpace uses separate residual and unknown layouts."
        )

    def build_role_block_system(
        self,
        K,
        R,
        *,
        unknown_dirichlet: DirichletBC | MixedDirichletBC | None = None,
    ) -> "MixedRoleBlockSystem":
        return MixedRoleBlockSystem(self, K, R, unknown_dirichlet=unknown_dirichlet)


@dataclass(eq=False)
class MixedFESpace:
    """
    Mixed FE space composed of multiple scalar/vector spaces.

    Field DOFs are concatenated in field order:
      [field0 dofs | field1 dofs | ...]
    """
    fields: dict[str, FESpaceClosure]
    field_order: Sequence[str] | None = None
    field_to_space_key: Mapping[str, str] | None = None
    field_alias_space_keys: Mapping[str, Sequence[str]] | None = None
    field_names: tuple[str, ...] = field(init=False)
    space_key_by_field: dict[str, str] = field(init=False)
    alias_space_keys_by_field: dict[str, tuple[str, ...]] = field(init=False)
    fields_by_space_key: dict[str, str] = field(init=False)
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

        if self.field_to_space_key is None:
            space_key_by_field = {name: name for name in self.field_names}
        else:
            space_key_by_field = {
                name: self.field_to_space_key.get(name, name) for name in self.field_names
            }
            extra_space_keys = set(self.field_to_space_key.keys()) - set(self.field_names)
            if extra_space_keys:
                raise ValueError(f"field_to_space_key has unknown fields: {extra_space_keys}")

        alias_space_keys_by_field = {
            name: tuple(str(key) for key in self.field_alias_space_keys.get(name, ()))
            if self.field_alias_space_keys is not None
            else ()
            for name in self.field_names
        }
        if self.field_alias_space_keys is not None:
            extra_alias_keys = set(self.field_alias_space_keys.keys()) - set(self.field_names)
            if extra_alias_keys:
                raise ValueError(f"field_alias_space_keys has unknown fields: {extra_alias_keys}")

        fields_by_space_key: dict[str, str] = {}
        for name in self.field_names:
            keys = (str(space_key_by_field[name]),) + tuple(alias_space_keys_by_field[name])
            for key in keys:
                if key in fields_by_space_key:
                    prev = fields_by_space_key[key]
                    if self.fields[prev] is not self.fields[name]:
                        raise ValueError(
                            f"space key '{key}' is assigned to multiple distinct fields "
                            f"({prev!r}, {name!r}); use unique space keys."
                        )
                else:
                    fields_by_space_key[key] = name

        self.space_key_by_field = dict(space_key_by_field)
        self.alias_space_keys_by_field = alias_space_keys_by_field
        self.fields_by_space_key = fields_by_space_key

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
        split = {name: u_elem[self.elem_slices[name]] for name in self.field_names}
        for name in self.field_names:
            split.setdefault(self.space_key_by_field[name], split[name])
            for key in self.alias_space_keys_by_field.get(name, ()):
                split.setdefault(key, split[name])
        return split

    def build_form_contexts(self, dep: jnp.ndarray | None = None) -> MixedFormContext:
        ctxs_by_field = {name: sp.build_form_contexts(dep) for name, sp in self.fields.items()}
        ref_ctx = ctxs_by_field[self.field_names[0]]

        fields = {
            name: FieldPair(test=ctx.test, trial=ctx.trial, unknown=None)
            for name, ctx in ctxs_by_field.items()
        }
        spaces = {
            self.space_key_by_field[name]: fields[name]
            for name in self.field_names
        }
        for name in self.field_names:
            for key in self.alias_space_keys_by_field.get(name, ()):
                spaces[key] = fields[name]

        return MixedFormContext(
            bindings=fields,
            x_q=ref_ctx.x_q,
            w=ref_ctx.w,
            elem_id=ref_ctx.elem_id,
            spaces=spaces,
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
        return assemble_mixed_residual(self, _compile_mixed_residual_like(res_form), u, params, **kwargs)

    def assemble_jacobian(
        self,
        res_form: MixedResidualForm[P],
        u: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray] | jnp.ndarray,
        params: P,
        **kwargs,
    ) -> "JacobianReturn":
        from .mixed_assembly import assemble_mixed_jacobian
        for removed in ("sparse", "return_flux_matrix", "matrix_accumulation"):
            if removed in kwargs:
                raise ValueError(
                    f"{removed} is no longer supported for assemble_jacobian; "
                    "assemble_jacobian now returns FluxSparseMatrix (use .to_dense() when needed)."
                )
        return assemble_mixed_jacobian(self, _compile_mixed_residual_like(res_form), u, params, **kwargs)

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


@dataclass(frozen=True)
class MixedRoleBlockSystem:
    mixed: MixedRoleFESpace
    K: object
    R: object
    unknown_dirichlet: DirichletBC | MixedDirichletBC | None = None

    def split_unknown(self, u_full: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self.mixed.unpack_unknown_fields(u_full)

    def join_unknown(self, fields: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return self.mixed.pack_unknown_fields(fields)

    def split_residual(self, r_full: jnp.ndarray) -> dict[str, jnp.ndarray]:
        r_full = jnp.asarray(r_full)
        return {
            name: r_full[self.mixed.residual_field_slices[name]]
            for name in self.mixed.field_names
        }

    def join_residual(self, fields: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate(
            [jnp.asarray(fields[name]) for name in self.mixed.field_names],
            axis=0,
        )

    @property
    def free_unknown_dofs(self) -> np.ndarray:
        if self.unknown_dirichlet is None:
            return np.arange(self.mixed.n_unknown_dofs, dtype=int)
        bc = (
            self.unknown_dirichlet.as_dirichlet_bc()
            if isinstance(self.unknown_dirichlet, MixedDirichletBC)
            else self.unknown_dirichlet
        )
        return free_dofs(self.mixed.n_unknown_dofs, bc.dofs)

    def condense_unknown(self, unknown_dirichlet: DirichletBC | MixedDirichletBC | None = None) -> "MixedRoleCondensedSystem":
        bc_in = self.unknown_dirichlet if unknown_dirichlet is None else unknown_dirichlet
        if bc_in is None:
            bc = DirichletBC(np.array([], dtype=int), np.array([], dtype=float))
        else:
            bc = bc_in.as_dirichlet_bc() if isinstance(bc_in, MixedDirichletBC) else bc_in

        free = free_dofs(self.mixed.n_unknown_dofs, bc.dofs)
        K_free, R_free = _condense_mixed_role_unknown(self.K, self.R, free, bc.dofs, bc.vals)
        return MixedRoleCondensedSystem(
            mixed=self.mixed,
            K=K_free,
            R=R_free,
            free_unknown_dofs=free,
            dirichlet=bc,
        )


@dataclass(frozen=True)
class MixedRoleCondensedSystem:
    mixed: MixedRoleFESpace
    K: object
    R: object
    free_unknown_dofs: np.ndarray
    dirichlet: DirichletBC

    def expand_unknown(self, u_free):
        return self.dirichlet.expand_solution(
            u_free,
            free=self.free_unknown_dofs,
            n_total=self.mixed.n_unknown_dofs,
        )


def build_mixed_role_block_system(
    mixed: MixedRoleFESpace,
    *,
    blocks: Mapping[tuple[str, str], object] | Mapping[str, Mapping[str, object]],
    residual: Mapping[str, object] | Sequence[object] | np.ndarray | None = None,
    unknown_dirichlet: DirichletBC | MixedDirichletBC | None = None,
    format: str = "auto",
) -> MixedRoleBlockSystem:
    """
    Build a role-aware block container with explicit residual-row and unknown-col layouts.
    """
    if format not in {"auto", "flux", "csr", "dense"}:
        raise ValueError("format must be 'auto', 'flux', 'csr', or 'dense'.")

    row_offsets = {
        name: int(mixed.residual_field_slices[name].start)
        for name in mixed.field_names
    }
    col_offsets = {
        name: int(mixed.unknown_field_slices[name].start)
        for name in mixed.field_names
    }
    row_sizes = {
        name: int(mixed.residual_field_slices[name].stop - mixed.residual_field_slices[name].start)
        for name in mixed.field_names
    }
    col_sizes = {
        name: int(mixed.unknown_field_slices[name].stop - mixed.unknown_field_slices[name].start)
        for name in mixed.field_names
    }

    if all(isinstance(k, tuple) and len(k) == 2 for k in blocks):
        pair_blocks = dict(cast(Mapping[tuple[str, str], object], blocks))
    else:
        pair_blocks = {}
        nested = cast(Mapping[str, Mapping[str, object]], blocks)
        for row_name, row in nested.items():
            for col_name, blk in row.items():
                pair_blocks[(row_name, col_name)] = blk

    prefer_flux = format == "flux"
    if format == "auto":
        prefer_flux = any(
            isinstance(blk, (FluxSparseOperator, FluxSparseMatrix))
            for blk in pair_blocks.values()
        )

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    for (row_name, col_name), blk in pair_blocks.items():
        if row_name not in row_sizes:
            raise KeyError(f"Unknown residual field '{row_name}' in blocks.")
        if col_name not in col_sizes:
            raise KeyError(f"Unknown unknown field '{col_name}' in blocks.")
        shape = (row_sizes[row_name], col_sizes[col_name])

        if isinstance(blk, FluxSparseOperator):
            if tuple(map(int, blk.shape)) != shape:
                raise ValueError(f"Block {(row_name, col_name)} has shape {blk.shape}, expected {shape}")
            r = np.asarray(blk.rows, dtype=np.int64)
            c = np.asarray(blk.cols, dtype=np.int64)
            d = np.asarray(blk.data)
        elif isinstance(blk, FluxSparseMatrix):
            if shape[0] != shape[1] or int(blk.n_dofs) != shape[0]:
                raise ValueError(
                    f"Block {(row_name, col_name)} has incompatible FluxSparseMatrix size for expected shape {shape}"
                )
            r = np.asarray(blk.pattern.rows, dtype=np.int64)
            c = np.asarray(blk.pattern.cols, dtype=np.int64)
            d = np.asarray(blk.data)
        else:
            arr = np.asarray(blk)
            if arr.shape != shape:
                raise ValueError(f"Block {(row_name, col_name)} has shape {arr.shape}, expected {shape}")
            r, c = np.nonzero(arr)
            d = arr[r, c]

        if r.size:
            rows_list.append(r + row_offsets[row_name])
            cols_list.append(c + col_offsets[col_name])
            data_list.append(d)

    rows = np.concatenate(rows_list) if rows_list else np.asarray([], dtype=np.int64)
    cols = np.concatenate(cols_list) if cols_list else np.asarray([], dtype=np.int64)
    data = np.concatenate(data_list) if data_list else np.asarray([], dtype=float)

    shape = (int(mixed.n_residual_dofs), int(mixed.n_unknown_dofs))
    if prefer_flux:
        K = FluxSparseOperator(rows, cols, data, shape=shape)
    else:
        use_dense = format == "dense"
        if format == "csr":
            use_dense = False
        if not use_dense:
            try:
                import scipy.sparse as sp
            except Exception as exc:  # pragma: no cover
                if format == "csr":
                    raise ImportError("scipy is required for CSR mixed role block systems.") from exc
                use_dense = True
            else:
                K = sp.csr_matrix((data, (rows, cols)), shape=shape)
        if use_dense:
            K = np.zeros(shape, dtype=data.dtype if data.size else float)
            if data.size:
                K[rows, cols] += data

    if residual is None:
        R = np.zeros((mixed.n_residual_dofs,), dtype=float)
    elif isinstance(residual, Mapping):
        R = np.concatenate(
            [
                np.asarray(residual.get(name, np.zeros(row_sizes[name], dtype=float)))
                for name in mixed.field_names
            ],
            axis=0,
        )
    elif hasattr(residual, "shape") and not isinstance(residual, (list, tuple)):
        R = np.asarray(residual)
        if R.shape != (mixed.n_residual_dofs,):
            raise ValueError(
                f"residual has shape {R.shape}, expected {(mixed.n_residual_dofs,)}"
            )
    else:
        parts = [np.asarray(p) for p in cast(Sequence[object], residual)]
        if len(parts) != len(mixed.field_names):
            raise ValueError("residual sequence length must match the number of mixed fields")
        for name, part in zip(mixed.field_names, parts):
            if part.shape != (row_sizes[name],):
                raise ValueError(
                    f"residual for field '{name}' has shape {part.shape}, expected {(row_sizes[name],)}"
                )
        R = np.concatenate(parts, axis=0)

    return MixedRoleBlockSystem(mixed, K, R, unknown_dirichlet=unknown_dirichlet)


def _condense_mixed_role_unknown(K, R, free: np.ndarray, dir_dofs: np.ndarray, dir_vals: np.ndarray):
    R_arr = np.asarray(R)
    if R_arr.shape != (R_arr.shape[0],):
        raise ValueError("MixedRoleBlockSystem residual must be a rank-1 vector.")

    if isinstance(K, FluxSparseOperator):
        rows = np.asarray(K.rows, dtype=np.int64)
        cols = np.asarray(K.cols, dtype=np.int64)
        data = np.asarray(K.data)
        if dir_dofs.size:
            dir_pos = {int(d): i for i, d in enumerate(np.asarray(dir_dofs, dtype=int))}
            dir_mask = np.array([int(c) in dir_pos for c in cols], dtype=bool)
            if np.any(dir_mask):
                vals = np.array([dir_vals[dir_pos[int(c)]] for c in cols[dir_mask]], dtype=data.dtype)
                contrib = np.zeros(K.shape[0], dtype=data.dtype)
                np.add.at(contrib, rows[dir_mask], data[dir_mask] * vals)
                R_arr = R_arr - contrib
        free_pos = {int(d): i for i, d in enumerate(np.asarray(free, dtype=int))}
        free_mask = np.array([int(c) in free_pos for c in cols], dtype=bool)
        cols_free = np.array([free_pos[int(c)] for c in cols[free_mask]], dtype=np.int64)
        K_free = FluxSparseOperator(rows[free_mask], cols_free, data[free_mask], shape=(K.shape[0], int(free.size)))
        return K_free, R_arr

    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover
        sp = None

    if sp is not None and sp.issparse(K):
        K_fd = K[:, dir_dofs] if dir_dofs.size else sp.csr_matrix((K.shape[0], 0), dtype=K.dtype)
        if dir_dofs.size:
            R_arr = R_arr - np.asarray(K_fd @ np.asarray(dir_vals), dtype=R_arr.dtype)
        return K[:, free], R_arr

    K_arr = np.asarray(K)
    if K_arr.ndim != 2:
        raise ValueError("MixedRoleBlockSystem matrix must be rank-2.")
    if K_arr.shape[1] != int(free.size + dir_dofs.size):
        # only sanity-check total cols when the partition covers the full unknown layout
        total = int(np.max(np.concatenate([free, dir_dofs])) + 1) if (free.size or dir_dofs.size) else 0
        if K_arr.shape[1] != total:
            raise ValueError("MixedRoleBlockSystem matrix column count does not match unknown layout.")
    if dir_dofs.size:
        R_arr = R_arr - K_arr[:, dir_dofs] @ np.asarray(dir_vals)
    return K_arr[:, free], R_arr


__all__ = [
    "MixedFESpace",
    "MixedSpaces",
    "MixedRoleFESpace",
    "MixedRoleSpaces",
    "build_mixed_role_block_system",
    "MixedProblem",
    "MixedDirichletBC",
    "MixedBlockSystem",
    "MixedRoleBlockSystem",
    "MixedRoleCondensedSystem",
]

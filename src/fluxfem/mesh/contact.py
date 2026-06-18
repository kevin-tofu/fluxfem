from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING, TypeAlias, Union, cast
import warnings

import numpy as np
import numpy.typing as npt

try:
    from .._runtime_warn import warn_float32_assembly_once
except Exception:  # pragma: no cover
    _WARNED_FLOAT32_CONTACT_ASSEMBLY = False

    def warn_float32_assembly_once(*, context: str = "assembly") -> None:
        global _WARNED_FLOAT32_CONTACT_ASSEMBLY
        if _WARNED_FLOAT32_CONTACT_ASSEMBLY:
            return
        try:
            import jax
        except Exception:
            return
        if bool(jax.config.read("jax_enable_x64")):
            return
        _WARNED_FLOAT32_CONTACT_ASSEMBLY = True
        warnings.warn(
            "Running in float32 mode (x64 disabled). "
            f"{context} can suffer from residual/conditioning degradation; "
            "use x64 for reliable diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
from .contact_interface import (
    assemble_contact_interface_jacobian as _assemble_contact_interface_jacobian,
    assemble_contact_interface_residual as _assemble_contact_interface_residual,
    assemble_onesided_bilinear,
    assemble_contact_onesided_floor,
    assemble_contact_coupling_matrices as _assemble_contact_coupling_matrices,
    volume_shape_values_at_points as _volume_shape_values_at_points,
    map_surface_facets_to_tet_elements,
    map_surface_facets_to_hex_elements,
    build_supermesh_triangle_quadrature_cache,
    _facet_shape_values,
    _tri_centroid,
    _tri_area,
)
from .supermesh import build_surface_supermesh
from .surface import SurfaceMesh
from .base import BaseMesh

if TYPE_CHECKING:
    from .contact_interface import ContactCouplingMatrix
    from ..core.weakform import Params as WeakParams
    from .contact_interface import SurfaceMixedFormContext
    from ..solver import FluxSparseMatrix, FluxSparseOperator


def _warn_contact_legacy_name(old: str, new: str) -> None:
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


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


def _infer_contact_backend(*values: Any, default: str) -> str:
    return "jax" if any(_contains_jax_value(v) for v in values) else default

ContactJacobianReturn: TypeAlias = Union[np.ndarray, "FluxSparseMatrix", "FluxSparseOperator"]
MixedSurfaceResidualForm: TypeAlias = Callable[
    ["SurfaceMixedFormContext", Mapping[str, npt.ArrayLike], Any],
    Mapping[str, npt.ArrayLike],
]
SurfaceHatFn: TypeAlias = Callable[[np.ndarray], npt.ArrayLike]


class ContactBilinear(Protocol):
    """Structural protocol for contact interface bilinear DSL callables."""

    def __call__(self, v1: Any, v2: Any, u1: Any, u2: Any, p: Any) -> Any: ...


class RoleCompiledContactBilinear(Protocol):
    """Contact bilinear compiled to role slots, but not yet bound to a concrete contact."""

    fn: ContactBilinear
    _ff_kind: str
    _ff_domain: str


ContactBilinearLike: TypeAlias = Union[ContactBilinear, MixedSurfaceResidualForm, RoleCompiledContactBilinear]


def _is_compiled_contact_bilinear(obj: Any) -> bool:
    return callable(obj) and hasattr(obj, "_includes_measure")


def _is_role_compiled_contact_bilinear(obj: Any) -> bool:
    return (
        getattr(obj, "_ff_kind", None) == "bilinear"
        and getattr(obj, "_ff_domain", None) == "contact"
        and callable(getattr(obj, "fn", None))
    )


def _ensure_role_compiled_contact_bilinear(bilin: ContactBilinearLike) -> ContactBilinearLike:
    """Normalize raw contact bilinear DSL callables to role-slot compiled contact forms."""
    if _is_compiled_contact_bilinear(bilin) or _is_role_compiled_contact_bilinear(bilin):
        return bilin
    from ..core.weakform import BilinearForm

    return cast(ContactBilinearLike, BilinearForm.contact(cast(ContactBilinear, bilin)).get_compiled())


def _compile_contact_bilinear(
    bilin: ContactBilinearLike,
    *,
    field_master: str = "a",
    field_slave: str = "b",
    backend: str | None = None,
) -> MixedSurfaceResidualForm:
    """Compile a contact bilinear callable into a reusable mixed-surface residual form."""
    backend = _infer_contact_backend(bilin, default="jax") if backend is None else str(backend).lower()
    normalized = _ensure_role_compiled_contact_bilinear(bilin)
    if _is_compiled_contact_bilinear(normalized):
        return cast(MixedSurfaceResidualForm, normalized)
    source = normalized.fn if _is_role_compiled_contact_bilinear(normalized) else normalized
    role_test_spaces = getattr(normalized, "_ff_contact_test_space_by_role", {})
    role_unknown_spaces = getattr(normalized, "_ff_contact_unknown_space_by_role", {})
    test_space_by_target = {
        str(field_master): role_test_spaces.get("a", str(field_master)),
        str(field_slave): role_test_spaces.get("b", str(field_slave)),
    }
    unknown_space_by_target = {
        str(field_master): role_unknown_spaces.get("a", test_space_by_target[str(field_master)]),
        str(field_slave): role_unknown_spaces.get("b", test_space_by_target[str(field_slave)]),
    }

    resolved_formulation = getattr(source, _FF_CONTACT_FORMULATION_ATTR, None)
    resolved_fastpath = getattr(source, _FF_CONTACT_FASTPATH_ATTR, None)
    if resolved_formulation == "pair_nitsche_penalty" and resolved_fastpath is None:
        resolved_fastpath = "numpy_local_kernel"

    if (
        backend == "numpy"
        and resolved_formulation == "pair_nitsche_penalty"
        and resolved_fastpath == "numpy_local_kernel"
    ):
        # The tagged NumPy fast path only needs metadata; keep a stub so the
        # Jacobian path can branch before evaluating any residual callback.
        def _fastpath_stub(_ctx, _u_elem, _params):
            raise RuntimeError("Tagged NumPy contact fast-path stub should not be evaluated.")

        res_form = _fastpath_stub
        setattr(res_form, "_includes_measure", {str(field_master): True, str(field_slave): True})
        setattr(res_form, "_space_by_target", {})
        setattr(res_form, "_test_space_by_target", dict(test_space_by_target))
        setattr(res_form, "_unknown_space_by_target", dict(unknown_space_by_target))
    else:
        from ..core.weakform import (
            compile_mixed_surface_residual,
            compile_mixed_surface_residual_numpy,
            unknown_ref,
            test_ref,
            param_ref,
            zero_ref,
        )

        v1 = test_ref(str(field_master), space=test_space_by_target[str(field_master)])
        v2 = test_ref(str(field_slave), space=test_space_by_target[str(field_slave)])
        u1 = unknown_ref(str(field_master), space=unknown_space_by_target[str(field_master)])
        u2 = unknown_ref(str(field_slave), space=unknown_space_by_target[str(field_slave)])
        z1 = zero_ref(str(field_master), space=test_space_by_target[str(field_master)])
        z2 = zero_ref(str(field_slave), space=test_space_by_target[str(field_slave)])
        p = param_ref()

        expr_a = cast(ContactBilinear, source)(v1, z2, u1, u2, p)
        expr_b = cast(ContactBilinear, source)(z1, v2, u1, u2, p)
        if backend == "jax":
            res_form = compile_mixed_surface_residual({str(field_master): expr_a, str(field_slave): expr_b})
        elif backend == "numpy":
            res_form = compile_mixed_surface_residual_numpy({str(field_master): expr_a, str(field_slave): expr_b})
        else:
            raise ValueError("backend must be 'jax' or 'numpy'")

    setattr(res_form, "_test_space_by_target", dict(test_space_by_target))
    setattr(res_form, "_unknown_space_by_target", dict(unknown_space_by_target))
    if test_space_by_target == unknown_space_by_target:
        setattr(res_form, "_space_by_target", dict(test_space_by_target))
    if resolved_formulation is not None:
        setattr(res_form, _FF_CONTACT_FORMULATION_ATTR, resolved_formulation)
    if resolved_fastpath is not None:
        setattr(res_form, _FF_CONTACT_FASTPATH_ATTR, resolved_fastpath)
    return res_form

_CONTACT_SETUP_CACHE: dict[tuple, "ContactSurfaceSpace"] = {}

_FF_CONTACT_FORMULATION_ATTR = "_ff_contact_formulation"
_FF_CONTACT_FASTPATH_ATTR = "_ff_contact_backend_fastpath"


def _tag_contact_bilinear(
    fn: ContactBilinear,
    *,
    formulation: str,
    backend_fastpath: str | None = None,
) -> ContactBilinear:
    """Attach lightweight internal metadata to a contact bilinear callable."""
    setattr(fn, _FF_CONTACT_FORMULATION_ATTR, str(formulation))
    if backend_fastpath is not None:
        setattr(fn, _FF_CONTACT_FASTPATH_ATTR, str(backend_fastpath))
    return fn


def _tag_contact_residual_form(
    res_form: Callable[..., Any],
    *,
    formulation: str,
    backend_fastpath: str | None = None,
) -> Callable[..., Any]:
    """Attach lightweight internal metadata to a compiled contact residual form."""
    setattr(res_form, _FF_CONTACT_FORMULATION_ATTR, str(formulation))
    if backend_fastpath is not None:
        setattr(res_form, _FF_CONTACT_FASTPATH_ATTR, str(backend_fastpath))
    return res_form


def make_tagged_pair_nitsche_penalty_bilinear(
    fn: ContactBilinear,
    *,
    backend_fastpath: str = "numpy_local_kernel",
) -> ContactBilinear:
    """Internal helper for comparison/debug code that needs the pair-Nitsche fast path."""
    return _tag_contact_bilinear(
        fn,
        formulation="pair_nitsche_penalty",
        backend_fastpath=backend_fastpath,
    )


def compile_tagged_pair_nitsche_penalty_residual(
    residuals: Mapping[str, Callable[..., Any]],
    *,
    backend: str | None = None,
    backend_fastpath: str = "numpy_local_kernel",
) -> Callable[..., Any]:
    """Internal helper for comparison/debug code that needs a tagged pair-Nitsche residual."""
    from ..core.weakform import (
        compile_mixed_surface_residual,
        compile_mixed_surface_residual_numpy,
    )

    backend = _infer_contact_backend(residuals, default="jax") if backend is None else str(backend).lower()
    if backend == "jax":
        res_form = compile_mixed_surface_residual(residuals)
    elif backend == "numpy":
        res_form = compile_mixed_surface_residual_numpy(residuals)
    else:
        raise ValueError("backend must be 'jax' or 'numpy'")
    return _tag_contact_residual_form(
        res_form,
        formulation="pair_nitsche_penalty",
        backend_fastpath=backend_fastpath,
    )


@dataclass(frozen=True)
class ContactOperators:
    """Container for assembled contact operators."""

    enforcement: str
    law: str | None = None
    formulation: str | None = None
    coupling_aa: Any | None = None
    coupling_ab: Any | None = None
    B_a: Any | None = None
    B_b: Any | None = None
    B: Any | None = None
    Kuu: Any | None = None
    residual: Any | None = None
    jacobian: Any | None = None
    facet_conn_master: np.ndarray | None = None
    rho: float | None = None
    multiplier: Any | None = None


@dataclass(frozen=True)
class ContactState:
    """Lightweight state snapshot for state-explicit contact workflows."""

    interface_kind: str
    geometry: str = "reference"
    iteration: int = 0
    active_set: str | None = None
    field_summary: Mapping[str, Any] = field(default_factory=dict)
    gap_n: Any | None = None
    active_mask: Any | None = None
    lambda_n: Any | None = None
    penalty_param: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PenaltyContactContribution(ContactOperators):
    """Explicit penalty-family contact contribution."""


@dataclass(frozen=True)
class MultiplierContactContribution(ContactOperators):
    """Explicit multiplier-family contact contribution."""


@dataclass(frozen=True)
class ContactSolveResult:
    """Result of a contact solve with explicit solver/contact state."""

    state: Any
    contact_state: ContactState
    converged: bool
    iters: int
    residual_norm: float


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


def _summarize_contact_field_state(state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None) -> dict[str, Any]:
    def _shape_summary(value: Any) -> tuple[int, ...]:
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = np.asarray(value).shape
        return tuple(int(x) for x in shape)

    if state is None:
        return {}
    if isinstance(state, Mapping):
        summary: dict[str, Any] = {}
        for key, value in state.items():
            summary[str(key)] = _shape_summary(value)
        return summary
    if isinstance(state, Sequence) and not hasattr(state, "shape"):
        summary = {}
        for i, value in enumerate(state):
            summary[f"arg{i}"] = _shape_summary(value)
        return summary
    return {"arg0": _shape_summary(state)}


@dataclass(frozen=True)
class ContactSpaces:
    """Public spec that binds contact roles to contact sides."""

    master: "ContactSide"
    slave: "ContactSide"
    field_master: str = "a"
    field_slave: str = "b"

    def __post_init__(self) -> None:
        if not str(self.field_master):
            raise ValueError("ContactSpaces.field_master must be non-empty.")
        if not str(self.field_slave):
            raise ValueError("ContactSpaces.field_slave must be non-empty.")

    def to_contact_surface_space(
        self,
        *,
        quad_order: int = 0,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> "ContactSurfaceSpace":
        _warn_contact_legacy_name("ContactSpaces.to_contact_surface_space()", "ContactPairSpec.prepare()")
        return ContactSurfaceSpace.from_sides(
            self.master,
            self.slave,
            field_master=str(self.field_master),
            field_slave=str(self.field_slave),
            quad_order=int(quad_order),
            tol=float(tol),
            backend=backend,
            batch_jac=batch_jac,
        )

    def prepare(
        self,
        *,
        quad_order: int = 0,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> "ContactSurfaceSpace":
        """Public alias for heavy contact-interface setup."""
        return ContactSurfaceSpace.from_sides(
            self.master,
            self.slave,
            field_master=str(self.field_master),
            field_slave=str(self.field_slave),
            quad_order=int(quad_order),
            tol=float(tol),
            backend=backend,
            batch_jac=batch_jac,
        )


@dataclass(frozen=True)
class ContactGroupSpaces:
    """Public spec that binds one-master/many-slave contact roles."""

    master: "ContactSide"
    slaves: Sequence["ContactSide"]
    field_master: str = "master"
    field_slave: str = "slave"

    def __post_init__(self) -> None:
        if len(self.slaves) == 0:
            raise ValueError("ContactGroupSpaces.slaves must contain at least one ContactSide.")
        if not str(self.field_master):
            raise ValueError("ContactGroupSpaces.field_master must be non-empty.")
        if not str(self.field_slave):
            raise ValueError("ContactGroupSpaces.field_slave must be non-empty.")

    def to_contact_surface_space(
        self,
        *,
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        _warn_contact_legacy_name("ContactGroupSpaces.to_contact_surface_space()", "ContactGroupSpec.prepare()")
        return OneToManyContactSurfaceSpace.from_sides(
            self.master,
            list(self.slaves),
            field_master=str(self.field_master),
            field_slave=str(self.field_slave),
            quad_order=int(quad_order),
            space_mode_master=str(space_mode_master),
            space_mode_slave=str(space_mode_slave),
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            normal_sign=normal_sign,
            tol=float(tol),
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    def prepare(
        self,
        *,
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        """Public alias for heavy one-to-many contact-interface setup."""
        return OneToManyContactSurfaceSpace.from_sides(
            self.master,
            list(self.slaves),
            field_master=str(self.field_master),
            field_slave=str(self.field_slave),
            quad_order=int(quad_order),
            space_mode_master=str(space_mode_master),
            space_mode_slave=str(space_mode_slave),
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            normal_sign=normal_sign,
            tol=float(tol),
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )


@dataclass(frozen=True)
class OneSidedContactSpaces:
    """Public spec that binds one-sided contact roles to a contact side."""

    side: "ContactSide"
    surface_master: SurfaceMesh | None = None
    elem_conn_master: np.ndarray | None = None
    facet_to_elem_master: np.ndarray | None = None

    def to_contact_surface_space(
        self,
        *,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
    ) -> "OneSidedContactSurfaceSpace":
        _warn_contact_legacy_name("OneSidedContactSpaces.to_contact_surface_space()", "OneSidedContactSpec.prepare()")
        return OneSidedContactSurfaceSpace.from_side(
            self.side,
            surface_master=self.surface_master,
            elem_conn_master=self.elem_conn_master,
            facet_to_elem_master=self.facet_to_elem_master,
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
        )

    def prepare(
        self,
        *,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
    ) -> "OneSidedContactSurfaceSpace":
        """Public alias for heavy one-sided contact-interface setup."""
        return OneSidedContactSurfaceSpace.from_side(
            self.side,
            surface_master=self.surface_master,
            elem_conn_master=self.elem_conn_master,
            facet_to_elem_master=self.facet_to_elem_master,
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
        )


@dataclass(frozen=True)
class ContactMultiplierSpace:
    """Discrete LM-space description used by constraint-family contact assembly."""

    family: str = "dual_nodal"  # "dual_nodal" | "nodal" | "coarse_p1" | "p0" | "p0_active" | "p0_supermesh"
    side: str = "master"  # For p0-like families, current implementation supports only "master".
    value_dim: int = 1
    facet_conn: np.ndarray | None = None
    coarse_rank: int | None = None
    coarse_projection: np.ndarray | None = None
    coarse_mode: str | None = None
    coarse_energy_tol: float | None = None
    coarse_rtol: float | None = None
    coarse_max_rank: int | None = None
    coarse_patch_ids: np.ndarray | None = None
    coarse_basis: np.ndarray | None = None

    def __post_init__(self) -> None:
        fam = str(self.family).lower()
        if fam not in {"nodal", "dual_nodal", "coarse_p1", "p0", "p0_active", "p0_supermesh"}:
            raise ValueError(
                "ContactMultiplierSpace.family must be 'nodal', 'dual_nodal', 'coarse_p1', "
                "'p0', 'p0_active', or 'p0_supermesh'."
            )
        side = str(self.side).lower()
        if side not in {"master", "slave"}:
            raise ValueError("ContactMultiplierSpace.side must be 'master' or 'slave'.")
        if fam == "coarse_p1" and side != "master":
            raise NotImplementedError(
                "coarse_p1 multipliers currently support only side='master' "
                "(coarse basis is defined in the master-side nodal space)."
            )
        if int(self.value_dim) <= 0:
            raise ValueError("ContactMultiplierSpace.value_dim must be positive.")
        if self.coarse_rank is not None and int(self.coarse_rank) <= 0:
            raise ValueError("ContactMultiplierSpace.coarse_rank must be positive when provided.")
        if self.coarse_projection is not None and np.asarray(self.coarse_projection).ndim != 2:
            raise ValueError("ContactMultiplierSpace.coarse_projection must be a 2D matrix.")
        if self.coarse_mode is not None and str(self.coarse_mode).lower() not in {"qr", "svd", "auto"}:
            raise ValueError("ContactMultiplierSpace.coarse_mode must be 'qr', 'svd', or 'auto'.")
        if self.coarse_energy_tol is not None and not (0.0 < float(self.coarse_energy_tol) <= 1.0):
            raise ValueError("ContactMultiplierSpace.coarse_energy_tol must be in (0, 1].")
        if self.coarse_rtol is not None and float(self.coarse_rtol) < 0.0:
            raise ValueError("ContactMultiplierSpace.coarse_rtol must be non-negative.")
        if self.coarse_max_rank is not None and int(self.coarse_max_rank) <= 0:
            raise ValueError("ContactMultiplierSpace.coarse_max_rank must be positive when provided.")
        if self.coarse_patch_ids is not None:
            patch_ids = np.asarray(self.coarse_patch_ids, dtype=int).reshape(-1)
            if patch_ids.size == 0:
                raise ValueError("ContactMultiplierSpace.coarse_patch_ids must be non-empty when provided.")
            if np.any(patch_ids < 0):
                raise ValueError("ContactMultiplierSpace.coarse_patch_ids must not contain negative ids.")
            if fam not in {"p0", "p0_active", "p0_supermesh"}:
                raise ValueError("ContactMultiplierSpace.coarse_patch_ids are supported only for p0-like families.")
        if self.coarse_basis is not None:
            basis = np.asarray(self.coarse_basis, dtype=float)
            if basis.ndim != 2:
                raise ValueError("ContactMultiplierSpace.coarse_basis must be a 2D matrix.")
            if basis.shape[0] <= 0 or basis.shape[1] <= 0:
                raise ValueError("ContactMultiplierSpace.coarse_basis must be non-empty.")
            if fam != "coarse_p1":
                raise ValueError("ContactMultiplierSpace.coarse_basis is supported only for family='coarse_p1'.")
        if fam == "coarse_p1" and self.coarse_basis is None:
            raise ValueError("ContactMultiplierSpace.coarse_basis is required when family='coarse_p1'.")

    @classmethod
    def from_contact(
        cls,
        contact,
        *,
        family: str = "dual_nodal",
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        coarse_rank: int | None = None,
        coarse_projection: np.ndarray | None = None,
        coarse_mode: str | None = None,
        coarse_energy_tol: float | None = None,
        coarse_rtol: float | None = None,
        coarse_max_rank: int | None = None,
        coarse_patch_ids: np.ndarray | None = None,
        coarse_basis: np.ndarray | None = None,
    ) -> "ContactMultiplierSpace":
        fc = None if facet_conn is None else np.asarray(facet_conn, dtype=int)
        if str(family).lower() in {"p0", "p0_active", "p0_supermesh"} and fc is None:
            fc = _infer_contact_side_facets(contact, side=str(side))
        return cls(
            family=str(family).lower(),
            side=str(side).lower(),
            value_dim=int(value_dim),
            facet_conn=fc,
            coarse_rank=None if coarse_rank is None else int(coarse_rank),
            coarse_projection=None if coarse_projection is None else np.asarray(coarse_projection, dtype=float),
            coarse_mode=None if coarse_mode is None else str(coarse_mode).lower(),
            coarse_energy_tol=None if coarse_energy_tol is None else float(coarse_energy_tol),
            coarse_rtol=None if coarse_rtol is None else float(coarse_rtol),
            coarse_max_rank=None if coarse_max_rank is None else int(coarse_max_rank),
            coarse_patch_ids=None if coarse_patch_ids is None else np.asarray(coarse_patch_ids, dtype=int),
            coarse_basis=None if coarse_basis is None else np.asarray(coarse_basis, dtype=float),
        )

    @classmethod
    def dual_mortar(
        cls,
        *,
        side: str = "master",
        value_dim: int = 1,
    ) -> "ContactMultiplierSpace":
        return cls(family="dual_nodal", side=side, value_dim=int(value_dim))

    @classmethod
    def nodal_mortar(
        cls,
        *,
        side: str = "master",
        value_dim: int = 1,
    ) -> "ContactMultiplierSpace":
        return cls(family="nodal", side=side, value_dim=int(value_dim))

    @classmethod
    def coarse_dual_mortar(
        cls,
        *,
        mode: str = "auto",
        rank: int | None = None,
        energy_tol: float = 0.999,
        rtol: float = 1e-10,
        max_rank: int | None = None,
        projection: np.ndarray | None = None,
        side: str = "master",
        value_dim: int = 1,
    ) -> "ContactMultiplierSpace":
        coarse_mode = "qr" if rank is not None and str(mode).lower() == "auto" else str(mode).lower()
        return cls(
            family="dual_nodal",
            side=side,
            value_dim=int(value_dim),
            coarse_rank=None if rank is None else int(rank),
            coarse_projection=None if projection is None else np.asarray(projection, dtype=float),
            coarse_mode=coarse_mode,
            coarse_energy_tol=float(energy_tol),
            coarse_rtol=float(rtol),
            coarse_max_rank=None if max_rank is None else int(max_rank),
        )

    @classmethod
    def p0_mortar(
        cls,
        contact=None,
        *,
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        family: str = "p0",
    ) -> "ContactMultiplierSpace":
        if contact is None and facet_conn is None:
            raise ValueError("p0_mortar requires contact or facet_conn.")
        return cls.from_contact(
            contact,
            family=family,
            side=side,
            value_dim=value_dim,
            facet_conn=facet_conn,
        )

    @classmethod
    def coarse_p0_mortar(
        cls,
        contact=None,
        *,
        patch_ids: np.ndarray,
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        family: str = "p0",
    ) -> "ContactMultiplierSpace":
        """Facet-integrated coarse P0 mortar grouped by patch ids."""

        if contact is None and facet_conn is None:
            raise ValueError("coarse_p0_mortar requires contact or facet_conn.")
        return cls.from_contact(
            contact,
            family=family,
            side=side,
            value_dim=value_dim,
            facet_conn=facet_conn,
            coarse_patch_ids=np.asarray(patch_ids, dtype=int),
        )

    @classmethod
    def coarse_p1_mortar(
        cls,
        *,
        basis: np.ndarray,
        side: str = "master",
        value_dim: int = 1,
    ) -> "ContactMultiplierSpace":
        """Integrated coarse P1 mortar from coarse master-side nodal basis rows.

        ``basis`` has shape ``(n_coarse_nodes, n_master_nodes)``.  Each row is a
        coarse multiplier shape function represented in the fine master nodal
        basis, and the assembled rows are ``basis @ M_aa`` and ``basis @ M_ab``.
        """

        return cls(
            family="coarse_p1",
            side=side,
            value_dim=int(value_dim),
            coarse_basis=np.asarray(basis, dtype=float),
        )


def _contact_sparse_to_coo(jacobian: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if hasattr(jacobian, "to_coo"):
        rows, cols, data, shape_or_n_dofs = jacobian.to_coo()
        if isinstance(shape_or_n_dofs, tuple):
            if int(shape_or_n_dofs[0]) != int(shape_or_n_dofs[1]):
                raise ValueError("Rectangular contact operators are not supported in this path.")
            n_dofs = int(shape_or_n_dofs[0])
        else:
            n_dofs = int(shape_or_n_dofs)
        return (
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
            np.asarray(data, dtype=float),
            n_dofs,
        )
    rows, cols, data, n_dofs = jacobian
    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(data, dtype=float),
        int(n_dofs),
    )


@dataclass(frozen=True)
class ContactKKTSolveConfig:
    """Linear solve configuration for ``solve_contact_kkt``."""

    backend: str = "numpy"
    diagonal_shift: float = 0.0
    allow_dense_fallback: bool = True
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
        if self.jax_solver not in {"gmres", "spsolve"}:
            raise ValueError("jax_solver must be 'gmres' or 'spsolve'.")
        if self.jax_dense_mode not in {"iterative", "direct_custom_vjp"}:
            raise ValueError("jax_dense_mode must be 'iterative' or 'direct_custom_vjp'.")
        if int(self.jax_restart) <= 0:
            raise ValueError("jax_restart must be positive.")
        if self.jax_solver == "spsolve" and float(self.diagonal_shift) != 0.0:
            raise ValueError("jax_solver='spsolve' currently requires diagonal_shift == 0.")
        return self


@dataclass(frozen=True)
class EmbeddingMap:
    """Sparse mapping ``u_slave = W * u_master``."""

    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]
    mode: str = "nodal"
    meta: Mapping[str, Any] | None = None


def build_nodal_embedding_map(master_coords: np.ndarray, slave_coords: np.ndarray) -> EmbeddingMap:
    """
    Build nearest-neighbor nodal embedding map from slave nodes to master nodes.
    """
    xm = np.asarray(master_coords, dtype=float)
    xs = np.asarray(slave_coords, dtype=float)
    if xm.ndim != 2 or xs.ndim != 2:
        raise ValueError("master_coords and slave_coords must be rank-2 arrays.")
    if xm.shape[1] != xs.shape[1]:
        raise ValueError("master/slave coordinates must share spatial dimension.")
    if xm.shape[0] == 0 or xs.shape[0] == 0:
        return EmbeddingMap(
            rows=np.zeros((0,), dtype=int),
            cols=np.zeros((0,), dtype=int),
            data=np.zeros((0,), dtype=float),
            shape=(int(xs.shape[0]), int(xm.shape[0])),
            mode="nodal",
            meta={"mapped_count": 0, "unmapped_count": int(xs.shape[0])},
        )

    # Brute-force nearest master node per slave node.
    diffs = xs[:, None, :] - xm[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    nearest = np.argmin(d2, axis=1).astype(int)
    rows = np.arange(xs.shape[0], dtype=int)
    cols = nearest
    data = np.ones((xs.shape[0],), dtype=float)
    return EmbeddingMap(
        rows=rows,
        cols=cols,
        data=data,
        shape=(int(xs.shape[0]), int(xm.shape[0])),
        mode="nodal",
        meta={"mapped_count": int(xs.shape[0]), "unmapped_count": 0},
    )


def build_barycentric_embedding_map(
    master_coords: np.ndarray,
    master_conn: np.ndarray,
    slave_coords: np.ndarray,
    *,
    tol: float = 1e-8,
    allow_unmapped: str | bool = "error",
    return_unmapped_ids: bool = False,
) -> EmbeddingMap | tuple[EmbeddingMap, np.ndarray]:
    """
    Build barycentric/isoparametric embedding map from slave points to master element nodes.

    Notes:
    - Uses broad-phase AABB filtering and deterministic tie-break.
    - If multiple master elements pass inside checks (e.g. point on element boundary),
      the smallest candidate element id is selected.
    - Supports element types handled by ``volume_shape_values_at_points``.
    """
    xm = np.asarray(master_coords, dtype=float)
    conn = np.asarray(master_conn, dtype=int)
    xs = np.asarray(slave_coords, dtype=float)
    if xm.ndim != 2 or xs.ndim != 2:
        raise ValueError("master_coords and slave_coords must be rank-2 arrays.")
    if conn.ndim != 2:
        raise ValueError("master_conn must be rank-2 array.")
    if xm.shape[1] != xs.shape[1]:
        raise ValueError("master/slave coordinates must share spatial dimension.")
    if isinstance(allow_unmapped, bool):
        warnings.warn(
            "Boolean allow_unmapped is deprecated; use 'error' or 'skip'.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "skip" if allow_unmapped else "error"
    else:
        mode = str(allow_unmapped).lower()
    if mode not in {"error", "skip"}:
        raise ValueError("allow_unmapped must be 'error' or 'skip' (bool is accepted for compatibility).")

    if xs.shape[0] == 0:
        emb = EmbeddingMap(
            rows=np.zeros((0,), dtype=int),
            cols=np.zeros((0,), dtype=int),
            data=np.zeros((0,), dtype=float),
            shape=(int(xs.shape[0]), int(xm.shape[0])),
            mode="barycentric",
            meta={"mapped_count": 0, "unmapped_count": 0},
        )
        if return_unmapped_ids:
            return emb, np.zeros((0,), dtype=int)
        return emb
    if conn.shape[0] == 0:
        if mode == "error":
            raise ValueError("Failed to map slave points: master_conn has no elements.")
        emb = EmbeddingMap(
            rows=np.zeros((0,), dtype=int),
            cols=np.zeros((0,), dtype=int),
            data=np.zeros((0,), dtype=float),
            shape=(int(xs.shape[0]), int(xm.shape[0])),
            mode="barycentric",
            meta={"mapped_count": 0, "unmapped_count": int(xs.shape[0])},
        )
        if return_unmapped_ids:
            return emb, np.arange(xs.shape[0], dtype=int)
        return emb

    rows_l: list[int] = []
    cols_l: list[int] = []
    data_l: list[float] = []
    unmapped_l: list[int] = []

    # Broad-phase acceleration: precompute master element AABBs.
    elem_coords_all = xm[conn]  # (n_elem, n_loc, dim)
    elem_mins = np.min(elem_coords_all, axis=1)
    elem_maxs = np.max(elem_coords_all, axis=1)
    tol_eff = float(tol)

    for i_s, p in enumerate(xs):
        found = False
        in_min = p[None, :] >= (elem_mins - tol_eff)
        in_max = p[None, :] <= (elem_maxs + tol_eff)
        candidates = np.nonzero(np.all(in_min & in_max, axis=1))[0]
        if candidates.size:
            candidates = np.sort(candidates, kind="stable")
        for e_id in candidates:
            elem_nodes = conn[int(e_id)]
            elem_nodes_i = np.asarray(elem_nodes, dtype=int)
            elem_coords = xm[elem_nodes_i]
            try:
                N = np.asarray(_volume_shape_values_at_points(p[None, :], elem_coords, tol=tol_eff)[0], dtype=float)
            except Exception:
                continue
            if np.any(~np.isfinite(N)):
                continue
            # Robust inside check for small Newton / floating-point errors.
            if np.min(N) < -tol_eff or np.max(N) > 1.0 + tol_eff:
                continue
            if abs(float(np.sum(N)) - 1.0) > 10.0 * tol_eff:
                continue

            for j_local, w in enumerate(N):
                if abs(float(w)) <= tol_eff:
                    continue
                rows_l.append(int(i_s))
                cols_l.append(int(elem_nodes_i[j_local]))
                data_l.append(float(w))
            found = True
            break
        if not found:
            unmapped_l.append(int(i_s))
            if mode == "error":
                raise ValueError(f"Failed to map slave point index {i_s} to any master element (tol={tol}).")

    if rows_l:
        rows = np.asarray(rows_l, dtype=int)
        cols = np.asarray(cols_l, dtype=int)
        data = np.asarray(data_l, dtype=float)
    else:
        rows = np.zeros((0,), dtype=int)
        cols = np.zeros((0,), dtype=int)
        data = np.zeros((0,), dtype=float)
    mapped_ids = np.unique(rows) if rows.size else np.zeros((0,), dtype=int)
    unmapped_ids_np = np.asarray(unmapped_l, dtype=int)
    emb = EmbeddingMap(
        rows=rows,
        cols=cols,
        data=data,
        shape=(int(xs.shape[0]), int(xm.shape[0])),
        mode="barycentric",
        meta={
            "mapped_count": int(mapped_ids.shape[0]),
            "unmapped_count": int(unmapped_ids_np.shape[0]),
        },
    )
    if return_unmapped_ids:
        return emb, unmapped_ids_np
    return emb


def build_barycentric_embedding_map_from_meshes(
    master_mesh: BaseMesh,
    slave_mesh: BaseMesh,
    *,
    slave_facet_selector: Callable[[BaseMesh], np.ndarray] | None = None,
    slave_node_selector: Callable[[BaseMesh], np.ndarray] | None = None,
    master_element_selector: Callable[[BaseMesh], np.ndarray] | None = None,
    tol: float = 1e-8,
    allow_unmapped: str | bool = "error",
    return_unmapped_ids: bool = False,
) -> EmbeddingMap | tuple[EmbeddingMap, np.ndarray]:
    """
    Build barycentric embedding map directly from mesh objects and selectors.

    Typical usage is to select slave boundary facets (e.g., plane) and embed those
    slave nodes into the master volume.
    """
    if slave_facet_selector is not None and slave_node_selector is not None:
        raise ValueError("Provide only one of slave_facet_selector or slave_node_selector.")

    x_master = np.asarray(master_mesh.coords, dtype=float)
    conn_master = np.asarray(master_mesh.conn, dtype=int)
    x_slave = np.asarray(slave_mesh.coords, dtype=float)
    n_slave_total = int(x_slave.shape[0])
    n_master_total = int(x_master.shape[0])

    if master_element_selector is not None:
        master_elem_ids = np.asarray(master_element_selector(master_mesh), dtype=int)
        conn_embed = conn_master[master_elem_ids]
    else:
        conn_embed = conn_master

    if slave_node_selector is not None:
        slave_node_ids = np.asarray(slave_node_selector(slave_mesh), dtype=int).reshape(-1)
    elif slave_facet_selector is not None:
        facets = np.asarray(slave_facet_selector(slave_mesh), dtype=int)
        slave_node_ids = np.unique(facets.reshape(-1)) if facets.size else np.zeros((0,), dtype=int)
    else:
        slave_node_ids = np.arange(n_slave_total, dtype=int)

    x_slave_sel = x_slave[slave_node_ids] if slave_node_ids.size else np.zeros((0, x_slave.shape[1]), dtype=float)
    out_local = build_barycentric_embedding_map(
        x_master,
        conn_embed,
        x_slave_sel,
        tol=tol,
        allow_unmapped=allow_unmapped,
        return_unmapped_ids=return_unmapped_ids,
    )
    if return_unmapped_ids:
        emb_local, unmapped_local = out_local
    else:
        emb_local = out_local
        unmapped_local = np.zeros((0,), dtype=int)
    rows_global = slave_node_ids[np.asarray(emb_local.rows, dtype=int)] if emb_local.rows.size else np.zeros((0,), dtype=int)
    emb_global = EmbeddingMap(
        rows=np.asarray(rows_global, dtype=int),
        cols=np.asarray(emb_local.cols, dtype=int),
        data=np.asarray(emb_local.data, dtype=float),
        shape=(n_slave_total, n_master_total),
        mode="barycentric",
        meta={
            "mapped_count": int(np.unique(rows_global).shape[0]) if np.asarray(rows_global).size else 0,
            "unmapped_count": int(np.asarray(unmapped_local).shape[0]),
            "slave_selection": "node_selector"
            if slave_node_selector is not None
            else ("facet_selector" if slave_facet_selector is not None else "all_nodes"),
            "master_selection": "element_selector" if master_element_selector is not None else "all_elements",
        },
    )
    if return_unmapped_ids:
        return emb_global, slave_node_ids[np.asarray(unmapped_local, dtype=int)]
    return emb_global


def assemble_embedding_constraint_matrix(
    embedding: EmbeddingMap,
    *,
    n_master_nodes: int,
    n_slave_nodes: int,
    value_dim: int = 1,
    backend: str | None = None,
):
    """
    Assemble ``C`` for equality constraints ``W*u_master - u_slave = 0``.

    Returns matrix with shape ``(n_slave_nodes*value_dim, (n_master_nodes+n_slave_nodes)*value_dim)``.
    """
    backend = _infer_contact_backend(embedding, default="numpy") if backend is None else str(backend).lower()
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    n_m = int(n_master_nodes)
    n_s = int(n_slave_nodes)
    vd = int(value_dim)
    if vd <= 0:
        raise ValueError("value_dim must be positive.")
    if int(embedding.shape[0]) > n_s or int(embedding.shape[1]) != n_m:
        raise ValueError("embedding.shape must satisfy (<= n_slave_nodes, n_master_nodes).")

    emb_rows = np.asarray(embedding.rows, dtype=int)
    emb_cols = np.asarray(embedding.cols, dtype=int)
    emb_data = np.asarray(embedding.data, dtype=float)
    if emb_rows.size != emb_cols.size or emb_rows.size != emb_data.size:
        raise ValueError("embedding rows/cols/data must have same length.")
    if emb_rows.size == 0:
        n_rows = 0
        row_ids = np.zeros((0,), dtype=int)
    else:
        if np.min(emb_rows) < 0 or np.max(emb_rows) >= n_s:
            raise ValueError("embedding row ids must be within [0, n_slave_nodes).")
        if np.min(emb_cols) < 0 or np.max(emb_cols) >= n_m:
            raise ValueError("embedding col ids must be within [0, n_master_nodes).")
        row_ids = np.unique(emb_rows)
        n_rows = int(row_ids.shape[0]) * vd
    row_pos = {int(r): i for i, r in enumerate(row_ids.tolist())}
    n_cols = (n_m + n_s) * vd
    if backend == "jax":
        import jax.numpy as jnp

        C = jnp.zeros((n_rows, n_cols), dtype=float)
        for r_s, c_m, w in zip(emb_rows, emb_cols, emb_data):
            for d in range(vd):
                row = int(row_pos[int(r_s)]) * vd + d
                col_m = int(c_m) * vd + d
                col_s = n_m * vd + int(r_s) * vd + d
                C = C.at[row, col_m].add(float(w))
                C = C.at[row, col_s].add(-1.0)
        return C

    C = np.zeros((n_rows, n_cols), dtype=float)
    for r_s, c_m, w in zip(emb_rows, emb_cols, emb_data):
        for d in range(vd):
            row = int(row_pos[int(r_s)]) * vd + d
            col_m = int(c_m) * vd + d
            col_s = n_m * vd + int(r_s) * vd + d
            C[row, col_m] += float(w)
            C[row, col_s] += -1.0
    return C


def assemble_rbe2_constraint_matrix(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    backend: str | None = None,
):
    """
    Assemble 3D RBE2-style rigid kinematic constraints.

    Unknown ordering:
      q = [u_ref(3), omega_ref(3), u_slave_0(3), ..., u_slave_{n-1}(3)]

    Constraint for each slave node i:
      u_slave_i - u_ref - (omega_ref x (x_i - x_ref)) = 0
    """
    backend = "numpy" if backend is None else str(backend).lower()
    if backend != "numpy":
        raise ValueError("RBE2 constraint assembly currently supports backend='numpy' only.")
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape[0] != 3:
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")

    n_s = int(x_s.shape[0])
    n_rows = 3 * n_s
    n_cols = 6 + 3 * n_s

    C = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(n_s):
        rx, ry, rz = (x_s[i] - x_ref).tolist()
        r0 = 3 * i
        c_slave = 6 + 3 * i
        C[r0 + 0, 0] = -1.0
        C[r0 + 1, 1] = -1.0
        C[r0 + 2, 2] = -1.0
        C[r0 + 0, 4] = -rz
        C[r0 + 0, 5] = +ry
        C[r0 + 1, 3] = +rz
        C[r0 + 1, 5] = -rx
        C[r0 + 2, 3] = -ry
        C[r0 + 2, 4] = +rx
        C[r0 + 0, c_slave + 0] = +1.0
        C[r0 + 1, c_slave + 1] = +1.0
        C[r0 + 2, c_slave + 2] = +1.0
    return C


def assemble_rbe3_constraint_matrix(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    normalize_weights: bool = True,
    backend: str | None = None,
):
    """
    Assemble a weighted 3D RBE3-style interpolation constraint.

    Unknown ordering:
      q = [u_ref(3), omega_ref(3), u_slave_0(3), ..., u_slave_{n-1}(3)]

    The constraints are formed from weighted rigid-body reconstruction in normal-
    equation form:

      (sum_i w_i B_i^T B_i) q_ref - sum_i w_i B_i^T u_i = 0

    where ``B_i = [I, -[r_i]_x]`` and ``r_i = x_i - x_ref``.

    This yields a 6 x (6 + 3*n_slave) matrix. Repeated use of this helper allows
    multiple user-defined RBE3 couplings to be added to one system.
    """
    backend = "numpy" if backend is None else str(backend).lower()
    if backend != "numpy":
        raise ValueError("RBE3 constraint assembly currently supports backend='numpy' only.")
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape[0] != 3:
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")

    n_s = int(x_s.shape[0])
    if n_s == 0:
        raise ValueError("slave_coords must contain at least one node.")

    if weights is None:
        w = np.ones((n_s,), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != (n_s,):
            raise ValueError("weights must have shape (n_slave,).")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite.")
    if normalize_weights:
        w_sum = float(np.sum(w))
        if abs(w_sum) <= 1e-15:
            raise ValueError("weights sum must be non-zero when normalize_weights=True.")
        w = w / w_sum

    def _bmat(point: np.ndarray) -> np.ndarray:
        rx, ry, rz = (point - x_ref).tolist()
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0, rz, -ry],
                [0.0, 1.0, 0.0, -rz, 0.0, rx],
                [0.0, 0.0, 1.0, ry, -rx, 0.0],
            ],
            dtype=float,
        )

    M = np.zeros((6, 6), dtype=float)
    slave_blocks = []
    for wi, xi in zip(w.tolist(), x_s):
        Bi = _bmat(xi)
        M += float(wi) * (Bi.T @ Bi)
        slave_blocks.append(-float(wi) * Bi.T)

    n_cols = 6 + 3 * n_s
    C = np.zeros((6, n_cols), dtype=float)
    C[:, :6] = M
    for i, blk in enumerate(slave_blocks):
        c0 = 6 + 3 * i
        C[:, c0 : c0 + 3] = blk
    return C


def build_rbe3_weights(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    method: str = "equal",
    surface: SurfaceMesh | None = None,
    power: float = 2.0,
    eps: float = 1e-12,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build convenience weights for RBE3-style interpolation.

    Supported methods
    -----------------
    ``equal``:
        Uniform node weights.
    ``distance``:
        Inverse-distance^power weights from the remote point.
    ``facet_area``:
        Lump each facet area equally to its nodes, then normalize per node.
        Requires ``surface`` whose node numbering matches ``slave_coords`` order.
    """
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape != (3,):
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")
    n_s = int(x_s.shape[0])
    if n_s == 0:
        raise ValueError("slave_coords must contain at least one node.")

    method_key = str(method).lower()
    if method_key == "equal":
        w = np.ones((n_s,), dtype=float)
    elif method_key == "distance":
        d = np.linalg.norm(x_s - x_ref[None, :], axis=1)
        w = 1.0 / np.maximum(d, float(eps)) ** float(power)
    elif method_key == "facet_area":
        if surface is None:
            raise ValueError("surface is required for method='facet_area'.")
        facets = np.asarray(surface.conn, dtype=int)
        areas = np.asarray(surface.facet_areas(), dtype=float)
        if facets.shape[0] != areas.shape[0]:
            raise ValueError("surface facet count and facet areas mismatch.")
        w = np.zeros((n_s,), dtype=float)
        for nodes, area in zip(facets, areas):
            if np.any(nodes < 0) or np.any(nodes >= n_s):
                raise ValueError("surface facets must index slave_coords in local node numbering.")
            share = float(area) / float(len(nodes))
            for node in nodes:
                w[int(node)] += share
    else:
        raise ValueError("method must be one of: equal, distance, facet_area.")

    if normalize:
        s = float(np.sum(w))
        if abs(s) <= float(eps):
            raise ValueError("weight sum is zero; cannot normalize.")
        w = w / s
    return w


def build_rbe3_remote_resultant(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    surface: SurfaceMesh,
    load: npt.ArrayLike | None = None,
    pressure: float | npt.ArrayLike | None = None,
    outward_from: npt.ArrayLike | None = None,
) -> np.ndarray:
    """
    Build the equivalent remote-point resultant for an RBE3-supported surface.

    The returned 6-vector is ordered as ``[force(3), moment(3)]`` and is
    compatible with a 6-DOF remote-point field ordered as
    ``[u_ref(3), omega_ref(3)]``.

    Exactly one of ``load`` or ``pressure`` must be provided:

    - ``load``: constant vector load per unit area with shape ``(3,)`` or
      ``(n_facets, 3)``
    - ``pressure``: scalar normal traction with shape ``()`` or ``(n_facets,)``
    """
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape != (3,):
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")
    if surface is None:
        raise ValueError("surface is required.")
    if (load is None) == (pressure is None):
        raise ValueError("Specify exactly one of load or pressure.")

    n_s = int(x_s.shape[0])
    facets = np.asarray(surface.conn, dtype=int)
    if np.any(facets < 0) or np.any(facets >= n_s):
        raise ValueError("surface facets must index slave_coords in local node numbering.")

    if load is not None:
        nodal_load = surface.assemble_load(load, dim=3, n_total_nodes=n_s)
    else:
        from ..solver.bc import assemble_surface_traction

        nodal_load = assemble_surface_traction(
            surface,
            pressure,
            dim=3,
            n_total_nodes=n_s,
            outward_from=outward_from,
        )

    nodal_load = np.asarray(nodal_load, dtype=float).reshape(n_s, 3)
    force = np.sum(nodal_load, axis=0)
    arm = x_s - x_ref[None, :]
    moment = np.sum(np.cross(arm, nodal_load), axis=0)
    return np.concatenate([force, moment], axis=0)


def assemble_contact_interface_residual(*args, **kwargs):
    """Assemble residual on a contact interface supermesh."""
    return _assemble_contact_interface_residual(*args, **kwargs)


def assemble_contact_interface_jacobian(*args, **kwargs):
    """Assemble Jacobian on a contact interface supermesh."""
    return _assemble_contact_interface_jacobian(*args, **kwargs)


def assemble_contact_coupling_matrices(*args, **kwargs):
    """Assemble coupling matrices for contact interface constraints."""
    return _assemble_contact_coupling_matrices(*args, **kwargs)


def _coo_to_dense(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, shape: tuple[int, int], *, backend: str):
    if backend == "jax":
        import jax.numpy as jnp

        out = jnp.zeros(shape, dtype=jnp.asarray(data).dtype if np.asarray(data).size else float)
        if len(rows) == 0:
            return out
        return out.at[np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)].add(np.asarray(data))
    out = np.zeros(shape, dtype=float)
    for r, c, v in zip(np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), np.asarray(data, dtype=float)):
        out[int(r), int(c)] += float(v)
    return out


def _p0_reduction_matrix_from_facets(facet_conn: np.ndarray, n_nodes: int):
    facets = np.asarray(facet_conn, dtype=int)
    S = np.zeros((int(facets.shape[0]), int(n_nodes)), dtype=float)
    for f, nodes in enumerate(facets):
        S[int(f), np.asarray(nodes, dtype=int)] = 1.0
    return S


def _p0_patch_group_matrix(patch_ids: np.ndarray, n_rows: int) -> np.ndarray:
    patches = np.asarray(patch_ids, dtype=int).reshape(-1)
    if int(patches.size) != int(n_rows):
        raise ValueError("coarse_patch_ids must have one entry per fine P0 multiplier row.")
    if np.any(patches < 0):
        raise ValueError("coarse_patch_ids must not contain negative ids.")
    unique = np.unique(patches)
    row_of_patch = {int(patch): i for i, patch in enumerate(unique.tolist())}
    P = np.zeros((int(unique.size), int(n_rows)), dtype=float)
    for fine_row, patch in enumerate(patches.tolist()):
        P[row_of_patch[int(patch)], int(fine_row)] = 1.0
    return P


def _apply_integrated_coarse_p0_groups(B_a, B_b, patch_ids: np.ndarray | None, *, backend: str):
    if patch_ids is None:
        return B_a, B_b
    P_np = _p0_patch_group_matrix(patch_ids, int(B_a.shape[0]))
    if backend == "jax":
        import jax.numpy as jnp

        P = jnp.asarray(P_np)
    else:
        P = P_np
    return P @ B_a, P @ B_b


def coarse_p1_basis_from_node_groups(
    n_fine_nodes: int,
    groups,
    *,
    weights=None,
    normalize: bool = True,
) -> np.ndarray:
    """Build coarse P1 rows from groups of fine master-side nodes.

    Each group defines one coarse multiplier shape function represented in the
    fine nodal basis.  With the default ``normalize=True``, every row sums to
    one, giving a simple partition-style averaging basis.
    """

    n_nodes = int(n_fine_nodes)
    if n_nodes <= 0:
        raise ValueError("n_fine_nodes must be positive.")
    rows = []
    weight_rows = None if weights is None else list(weights)
    group_rows = list(groups)
    if not group_rows:
        raise ValueError("groups must contain at least one node group.")
    if weight_rows is not None and len(weight_rows) != len(group_rows):
        raise ValueError("weights must have the same number of rows as groups.")
    for row_id, group in enumerate(group_rows):
        nodes = np.asarray(group, dtype=int).reshape(-1)
        if nodes.size == 0:
            raise ValueError("each node group must be non-empty.")
        if np.any(nodes < 0) or np.any(nodes >= n_nodes):
            raise ValueError("node group contains an out-of-range node id.")
        if weight_rows is None:
            values = np.ones((int(nodes.size),), dtype=float)
        else:
            values = np.asarray(weight_rows[row_id], dtype=float).reshape(-1)
            if int(values.size) != int(nodes.size):
                raise ValueError("each weight row must match the corresponding node group size.")
        if normalize:
            total = float(np.sum(values))
            if abs(total) <= np.finfo(float).eps:
                raise ValueError("cannot normalize a coarse P1 basis row with zero weight sum.")
            values = values / total
        row = np.zeros((n_nodes,), dtype=float)
        for node, value in zip(nodes.tolist(), values.tolist()):
            row[int(node)] += float(value)
        rows.append(row)
    return np.vstack(rows)


def coarse_p1_basis_from_surface_grid(
    surface,
    *,
    shape: tuple[int, int],
    axes: tuple[int, int] = (0, 1),
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    clamp: bool = True,
) -> np.ndarray:
    """Build coarse P1 rows by bilinear interpolation on a surface coordinate grid.

    The returned matrix has shape ``(shape[0] * shape[1], n_surface_nodes)``.
    It is intended for planar or nearly planar surfaces where two coordinate
    axes provide a reasonable parameterization.
    """

    coords = np.asarray(getattr(surface, "coords", surface), dtype=float)
    if coords.ndim != 2:
        raise ValueError("surface must provide coords with shape (n_nodes, dim).")
    n_nodes = int(coords.shape[0])
    if n_nodes <= 0:
        raise ValueError("surface must contain at least one node.")
    ax0, ax1 = (int(axes[0]), int(axes[1]))
    if ax0 == ax1:
        raise ValueError("axes must contain two distinct coordinate axes.")
    if ax0 < 0 or ax1 < 0 or ax0 >= int(coords.shape[1]) or ax1 >= int(coords.shape[1]):
        raise ValueError("axes are out of range for surface coordinates.")
    nu, nv = int(shape[0]), int(shape[1])
    if nu < 2 or nv < 2:
        raise ValueError("shape must be at least (2, 2) for P1 grid basis.")
    uv = coords[:, [ax0, ax1]]
    if bounds is None:
        umin, vmin = np.min(uv, axis=0)
        umax, vmax = np.max(uv, axis=0)
    else:
        (umin, umax), (vmin, vmax) = bounds
        umin, umax, vmin, vmax = float(umin), float(umax), float(vmin), float(vmax)
    if not (umax > umin and vmax > vmin):
        raise ValueError("surface grid bounds must have positive extent.")

    u = (uv[:, 0] - umin) / (umax - umin) * (nu - 1)
    v = (uv[:, 1] - vmin) / (vmax - vmin) * (nv - 1)
    if clamp:
        u = np.clip(u, 0.0, float(nu - 1))
        v = np.clip(v, 0.0, float(nv - 1))
    elif np.any((u < 0.0) | (u > nu - 1) | (v < 0.0) | (v > nv - 1)):
        raise ValueError("surface node lies outside the requested grid bounds.")

    iu0 = np.floor(u).astype(int)
    iv0 = np.floor(v).astype(int)
    iu0 = np.clip(iu0, 0, nu - 2)
    iv0 = np.clip(iv0, 0, nv - 2)
    du = u - iu0
    dv = v - iv0

    basis = np.zeros((nu * nv, n_nodes), dtype=float)
    for node_id in range(n_nodes):
        i = int(iu0[node_id])
        j = int(iv0[node_id])
        weights = (
            ((1.0 - du[node_id]) * (1.0 - dv[node_id]), i, j),
            (du[node_id] * (1.0 - dv[node_id]), i + 1, j),
            ((1.0 - du[node_id]) * dv[node_id], i, j + 1),
            (du[node_id] * dv[node_id], i + 1, j + 1),
        )
        for value, ii, jj in weights:
            basis[int(jj) * nu + int(ii), node_id] += float(value)
    return basis


def _expand_scalar_constraint_dense(B_scalar, *, value_dim: int, backend: str):
    vd = int(value_dim)
    if vd <= 1:
        return B_scalar
    B_np = np.asarray(B_scalar, dtype=float)
    out = np.zeros((vd * B_np.shape[0], vd * B_np.shape[1]), dtype=B_np.dtype)
    for comp in range(vd):
        out[comp::vd, comp::vd] = B_np
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(out)
    return out


def _expand_scalar_constraint_coo(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    value_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    vd = int(value_dim)
    if vd <= 1:
        return rows, cols, data, int(n_rows), int(n_cols)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    data = np.asarray(data, dtype=float)
    rows_exp = np.concatenate([vd * rows + comp for comp in range(vd)], axis=0)
    cols_exp = np.concatenate([vd * cols + comp for comp in range(vd)], axis=0)
    data_exp = np.concatenate([data for _ in range(vd)], axis=0)
    return rows_exp, cols_exp, data_exp, vd * int(n_rows), vd * int(n_cols)


def _infer_contact_side_facets(contact, *, side: str) -> np.ndarray | None:
    side_norm = str(side).lower()
    if side_norm not in {"master", "slave"}:
        raise ValueError("side must be 'master' or 'slave'.")

    if hasattr(contact, "surface_master") and hasattr(contact, "surface_slave"):
        surf = contact.surface_master if side_norm == "master" else contact.surface_slave
        return np.asarray(surf.conn, dtype=int)

    if hasattr(contact, "contacts") and len(getattr(contact, "contacts")) > 0:
        if side_norm == "slave":
            return None
        first = contact.contacts[0]
        if hasattr(first, "surface_master"):
            return np.asarray(first.surface_master.conn, dtype=int)
    return None


def _resolve_multiplier_spec(
    contact,
    *,
    multiplier: ContactMultiplierSpace | None,
    facet_conn_master: np.ndarray | None,
) -> tuple[str, np.ndarray | None, ContactMultiplierSpace]:
    if multiplier is not None and not isinstance(multiplier, ContactMultiplierSpace):
        raise TypeError("multiplier must be a ContactMultiplierSpace.")
    if multiplier is None:
        multiplier = ContactMultiplierSpace.from_contact(contact, family="dual_nodal", side="master")
    fam = str(multiplier.family).lower()
    if fam in {"p0", "p0_active", "p0_supermesh"} and str(multiplier.side).lower() != "master":
        raise NotImplementedError(
            "p0-like multipliers currently support only side='master' "
            "(current implementation limitation)."
        )
    facet = multiplier.facet_conn
    if facet is None and fam == "p0":
        facet = _infer_contact_side_facets(contact, side=str(multiplier.side))
    if facet is None:
        facet = facet_conn_master
    if fam == "dual_nodal" and str(multiplier.side).lower() != "master":
        raise NotImplementedError(
            "dual_nodal multipliers currently support only side='master' "
            "(requires the master-side nodal mass block)."
        )
    if fam == "coarse_p1" and str(multiplier.side).lower() != "master":
        raise NotImplementedError(
            "coarse_p1 multipliers currently support only side='master' "
            "(coarse basis is defined in the master-side nodal space)."
        )
    if fam not in {"nodal", "dual_nodal", "coarse_p1", "p0", "p0_active", "p0_supermesh"}:
        raise ValueError(
            "multiplier.family must be 'nodal', 'dual_nodal', 'coarse_p1', "
            "'p0', 'p0_active', or 'p0_supermesh'"
        )
    if fam in {"p0", "p0_active"} and facet is None:
        raise ValueError(f"facet_conn_master is required when multiplier.family='{fam}'.")
    facet_arr = None if facet is None else np.asarray(facet, dtype=int)
    resolved_multiplier = ContactMultiplierSpace(
        family=fam,
        side=str(multiplier.side).lower(),
        value_dim=int(multiplier.value_dim),
        facet_conn=facet_arr,
        coarse_rank=None if multiplier.coarse_rank is None else int(multiplier.coarse_rank),
        coarse_projection=(
            None
            if multiplier.coarse_projection is None
            else np.asarray(multiplier.coarse_projection, dtype=float)
        ),
        coarse_mode=multiplier.coarse_mode,
        coarse_energy_tol=multiplier.coarse_energy_tol,
        coarse_rtol=multiplier.coarse_rtol,
        coarse_max_rank=multiplier.coarse_max_rank,
        coarse_patch_ids=(
            None
            if multiplier.coarse_patch_ids is None
            else np.asarray(multiplier.coarse_patch_ids, dtype=int)
        ),
        coarse_basis=(
            None
            if multiplier.coarse_basis is None
            else np.asarray(multiplier.coarse_basis, dtype=float)
        ),
    )
    return fam, facet_arr, resolved_multiplier


def _coalesce_int_coo(rows: np.ndarray, cols: np.ndarray, data: np.ndarray):
    from ..solver.sparse import coalesce_coo

    r, c, d = coalesce_coo(rows, cols, data)
    return np.asarray(r, dtype=int), np.asarray(c, dtype=int), np.asarray(d, dtype=float)


def _dual_nodal_blocks_from_dense(M_aa, M_ab, *, backend: str):
    """Build master-side dual nodal mortar blocks.

    Full-rank nodal mass blocks use the exact inverse. Rank-deficient blocks use
    the Moore-Penrose pseudoinverse, which gives the least-squares dual map and
    keeps inactive/degenerate rows from making the public API unusable.
    """
    if int(M_aa.shape[0]) != int(M_aa.shape[1]):
        raise ValueError("dual_nodal requires a square master-side nodal coupling block.")
    if int(M_ab.shape[0]) != int(M_aa.shape[0]):
        raise ValueError("dual_nodal requires compatible master/slave coupling row counts.")
    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np
    B_a = xp.eye(int(M_aa.shape[0]), dtype=M_aa.dtype)
    B_b = xp.linalg.pinv(M_aa) @ M_ab
    return B_a, B_b


def _dense_to_coo_entries(mat: np.ndarray, *, tol: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(mat, dtype=float)
    if tol > 0.0:
        rows, cols = np.nonzero(np.abs(arr) > float(tol))
    else:
        rows, cols = np.nonzero(arr)
    return rows.astype(int), cols.astype(int), arr[rows, cols].astype(float)


def _coarse_row_projection_from_rank(B, rank: int, *, backend: str):
    if int(rank) <= 0:
        raise ValueError("coarse_rank must be positive.")
    max_rank = min(int(B.shape[0]), int(B.shape[1]))
    if int(rank) > max_rank:
        raise ValueError("coarse_rank cannot exceed min(B.shape).")
    if backend == "jax":
        import jax.numpy as jnp

        q, _ = jnp.linalg.qr(B, mode="reduced")
        return q[:, : int(rank)].T
    q, _ = np.linalg.qr(np.asarray(B), mode="reduced")
    return q[:, : int(rank)].T


def _coarse_row_projection_from_svd(
    B,
    *,
    energy_tol: float,
    rtol: float,
    max_rank: int | None,
    backend: str,
):
    if backend == "jax":
        import jax.numpy as jnp

        u, s, _ = jnp.linalg.svd(B, full_matrices=False)
        s_np = np.asarray(s, dtype=float)
        xp = jnp
    else:
        u_np, s_np, _ = np.linalg.svd(np.asarray(B, dtype=float), full_matrices=False)
        u = u_np
        xp = np
    if s_np.size == 0:
        raise ValueError("Cannot build a coarse mortar projection from an empty B matrix.")
    total = float(np.sum(s_np**2))
    if total <= 0.0:
        rank_energy = 1
    else:
        cumulative = np.cumsum(s_np**2) / total
        rank_energy = int(np.searchsorted(cumulative, float(energy_tol), side="left") + 1)
    threshold = float(rtol) * float(s_np[0]) if s_np.size else 0.0
    rank_numeric = int(np.count_nonzero(s_np > threshold)) if threshold > 0.0 else int(s_np.size)
    rank = max(1, min(rank_energy, rank_numeric if rank_numeric > 0 else 1))
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    rank = min(rank, int(u.shape[1]))
    return xp.asarray(u[:, :rank]).T


def _apply_coarse_mortar_projection(B_a, B_b, multiplier: ContactMultiplierSpace, *, backend: str):
    projection = multiplier.coarse_projection
    rank = multiplier.coarse_rank
    mode = None if multiplier.coarse_mode is None else str(multiplier.coarse_mode).lower()
    if projection is None and rank is None and mode is None:
        return B_a, B_b
    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np
    B = xp.concatenate([B_a, -B_b], axis=1)
    if projection is not None:
        P = xp.asarray(projection)
        if int(P.shape[1]) != int(B.shape[0]):
            raise ValueError("coarse_projection must have shape (n_coarse, n_multiplier_rows).")
    elif mode in {"svd", "auto"} and rank is None:
        P = _coarse_row_projection_from_svd(
            B,
            energy_tol=0.999 if multiplier.coarse_energy_tol is None else float(multiplier.coarse_energy_tol),
            rtol=1e-10 if multiplier.coarse_rtol is None else float(multiplier.coarse_rtol),
            max_rank=multiplier.coarse_max_rank,
            backend=backend,
        )
    else:
        rank_eff = int(rank) if rank is not None else int(multiplier.coarse_max_rank or min(B.shape))
        P = _coarse_row_projection_from_rank(B, rank_eff, backend=backend)
    B_coarse = P @ B
    n_a = int(B_a.shape[1])
    return B_coarse[:, :n_a], -B_coarse[:, n_a:]


def _kkt_coo_from_coupling(
    coupling_aa,
    coupling_ab,
    *,
    rho: float,
    multiplier_space: str,
    facet_conn_master: np.ndarray | None,
    multiplier_value_dim: int = 1,
    coarse_rank: int | None = None,
    coarse_projection: np.ndarray | None = None,
    coarse_mode: str | None = None,
    coarse_energy_tol: float | None = None,
    coarse_rtol: float | None = None,
    coarse_max_rank: int | None = None,
    coarse_patch_ids: np.ndarray | None = None,
    coarse_basis: np.ndarray | None = None,
):
    if multiplier_space == "p0_supermesh":
        raise NotImplementedError(
            "multiplier_space='p0_supermesh' requires direct B/Kuu assembly from contact operators."
        )
    rows_aa, cols_aa, data_aa = _coalesce_int_coo(coupling_aa.rows, coupling_aa.cols, coupling_aa.data)
    rows_ab, cols_ab, data_ab = _coalesce_int_coo(coupling_ab.rows, coupling_ab.cols, coupling_ab.data)
    n_a = int(coupling_aa.shape[0])
    n_b = int(coupling_ab.shape[1])
    n_u = n_a + n_b

    if multiplier_space == "nodal":
        n_l = n_a
        b_rows = np.concatenate([rows_aa, rows_ab])
        b_cols = np.concatenate([cols_aa, n_a + cols_ab])
        b_data = np.concatenate([data_aa, -data_ab])
    elif multiplier_space == "dual_nodal":
        M_aa = _coo_to_dense(rows_aa, cols_aa, data_aa, coupling_aa.shape, backend="numpy")
        M_ab = _coo_to_dense(rows_ab, cols_ab, data_ab, coupling_ab.shape, backend="numpy")
        B_a, B_b = _dual_nodal_blocks_from_dense(M_aa, M_ab, backend="numpy")
        rows_a, cols_a, data_a = _dense_to_coo_entries(B_a)
        rows_b, cols_b, data_b = _dense_to_coo_entries(B_b)
        n_l = int(B_a.shape[0])
        b_rows = np.concatenate([rows_a, rows_b])
        b_cols = np.concatenate([cols_a, n_a + cols_b])
        b_data = np.concatenate([data_a, -data_b])
    elif multiplier_space == "coarse_p1":
        if coarse_basis is None:
            raise ValueError("coarse_basis is required when multiplier_space='coarse_p1'.")
        M_aa = _coo_to_dense(rows_aa, cols_aa, data_aa, coupling_aa.shape, backend="numpy")
        M_ab = _coo_to_dense(rows_ab, cols_ab, data_ab, coupling_ab.shape, backend="numpy")
        C = np.asarray(coarse_basis, dtype=float)
        if C.ndim != 2 or int(C.shape[1]) != int(M_aa.shape[0]):
            raise ValueError("coarse_basis must have shape (n_coarse_nodes, n_master_nodes).")
        B_a = C @ M_aa
        B_b = C @ M_ab
        rows_a, cols_a, data_a = _dense_to_coo_entries(B_a)
        rows_b, cols_b, data_b = _dense_to_coo_entries(B_b)
        n_l = int(B_a.shape[0])
        b_rows = np.concatenate([rows_a, rows_b])
        b_cols = np.concatenate([cols_a, n_a + cols_b])
        b_data = np.concatenate([data_a, -data_b])
    elif multiplier_space == "p0":
        if facet_conn_master is None:
            raise ValueError("facet_conn_master is required when multiplier_space='p0'.")
        facets = np.asarray(facet_conn_master, dtype=int)
        n_l = int(facets.shape[0])
        row_map: dict[int, list[int]] = {i: [] for i in range(n_a)}
        for k, r in enumerate(rows_aa):
            row_map[int(r)].append(int(k))
        row_map_ab: dict[int, list[int]] = {i: [] for i in range(n_a)}
        for k, r in enumerate(rows_ab):
            row_map_ab[int(r)].append(int(k))
        b_rows_l: list[int] = []
        b_cols_l: list[int] = []
        b_data_l: list[float] = []
        for lf, nodes in enumerate(facets):
            acc: dict[int, float] = {}
            for n in np.asarray(nodes, dtype=int):
                for k in row_map.get(int(n), []):
                    c = int(cols_aa[k])
                    acc[c] = acc.get(c, 0.0) + float(data_aa[k])
                for k in row_map_ab.get(int(n), []):
                    c = n_a + int(cols_ab[k])
                    acc[c] = acc.get(c, 0.0) - float(data_ab[k])
            for c, v in acc.items():
                b_rows_l.append(int(lf))
                b_cols_l.append(int(c))
                b_data_l.append(float(v))
        if b_rows_l:
            b_rows = np.asarray(b_rows_l, dtype=int)
            b_cols = np.asarray(b_cols_l, dtype=int)
            b_data = np.asarray(b_data_l, dtype=float)
            b_rows, b_cols, b_data = _coalesce_int_coo(b_rows, b_cols, b_data)
        else:
            b_rows = np.zeros((0,), dtype=int)
            b_cols = np.zeros((0,), dtype=int)
            b_data = np.zeros((0,), dtype=float)
    else:
        raise ValueError("multiplier_space must be 'nodal', 'dual_nodal', 'coarse_p1', or 'p0'")
    if coarse_patch_ids is not None:
        if multiplier_space != "p0":
            raise ValueError("coarse_patch_ids are supported only for p0 multiplier_space in sparse KKT assembly.")
        B_dense = np.zeros((n_l, n_u), dtype=float)
        B_dense[b_rows, b_cols] += b_data
        P = _p0_patch_group_matrix(coarse_patch_ids, int(n_l))
        B_dense = P @ B_dense
        b_rows, b_cols, b_data = _dense_to_coo_entries(B_dense)
        n_l = int(B_dense.shape[0])
    b_rows, b_cols, b_data, n_l, n_u = _expand_scalar_constraint_coo(
        b_rows,
        b_cols,
        b_data,
        n_rows=n_l,
        n_cols=n_u,
        value_dim=int(multiplier_value_dim),
    )
    if coarse_rank is not None or coarse_projection is not None or coarse_mode is not None:
        B_dense = np.zeros((n_l, n_u), dtype=float)
        B_dense[b_rows, b_cols] += b_data
        coarse_multiplier = ContactMultiplierSpace(
            family="nodal",
            value_dim=1,
            coarse_rank=coarse_rank,
            coarse_projection=coarse_projection,
            coarse_mode=coarse_mode,
            coarse_energy_tol=coarse_energy_tol,
            coarse_rtol=coarse_rtol,
            coarse_max_rank=coarse_max_rank,
            coarse_patch_ids=None,
            coarse_basis=None,
        )
        n_a_expanded = int(n_a) * int(multiplier_value_dim)
        B_a_dense = B_dense[:, :n_a_expanded]
        B_b_dense = -B_dense[:, n_a_expanded:]
        B_a_dense, B_b_dense = _apply_coarse_mortar_projection(
            B_a_dense,
            B_b_dense,
            coarse_multiplier,
            backend="numpy",
        )
        B_dense = np.concatenate([B_a_dense, -B_b_dense], axis=1)
        b_rows, b_cols, b_data = _dense_to_coo_entries(B_dense)
        n_l = int(B_dense.shape[0])
        n_u = int(B_dense.shape[1])

    # Build Kuu = rho * B^T B from row-wise products.
    by_row: dict[int, list[int]] = {}
    for k, r in enumerate(b_rows):
        by_row.setdefault(int(r), []).append(int(k))
    kuu_acc: dict[tuple[int, int], float] = {}
    if float(rho) != 0.0:
        rr = float(rho)
        for ids in by_row.values():
            for i in ids:
                ci = int(b_cols[i])
                vi = float(b_data[i])
                for j in ids:
                    cj = int(b_cols[j])
                    vj = float(b_data[j])
                    key = (ci, cj)
                    kuu_acc[key] = kuu_acc.get(key, 0.0) + rr * vi * vj

    kuu_rows = np.fromiter((k[0] for k in kuu_acc.keys()), dtype=int, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=int)
    kuu_cols = np.fromiter((k[1] for k in kuu_acc.keys()), dtype=int, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=int)
    kuu_data = np.fromiter((v for v in kuu_acc.values()), dtype=float, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=float)

    # KKT COO assembly:
    # [Kuu  B^T]
    # [ B    0 ]
    k_rows = []
    k_cols = []
    k_data = []
    if kuu_rows.size:
        k_rows.append(kuu_rows)
        k_cols.append(kuu_cols)
        k_data.append(kuu_data)
    if b_rows.size:
        # B^T block (top-right)
        k_rows.append(b_cols)
        k_cols.append(n_u + b_rows)
        k_data.append(b_data)
        # B block (bottom-left)
        k_rows.append(n_u + b_rows)
        k_cols.append(b_cols)
        k_data.append(b_data)
    if k_rows:
        rows = np.concatenate(k_rows)
        cols = np.concatenate(k_cols)
        data = np.concatenate(k_data)
        rows, cols, data = _coalesce_int_coo(rows, cols, data)
    else:
        rows = np.zeros((0,), dtype=int)
        cols = np.zeros((0,), dtype=int)
        data = np.zeros((0,), dtype=float)
    n_total = int(n_u + n_l)
    return rows, cols, data, n_total


def _assemble_supermesh_triangle_p0_blocks(
    contact,
    *,
    backend: str,
    value_dim: int,
    coarse_patch_ids: np.ndarray | None = None,
):
    if not all(
        hasattr(contact, name)
        for name in (
            "supermesh_coords",
            "supermesh_conn",
            "source_facets_master",
            "source_facets_slave",
            "surface_master",
            "surface_slave",
            "tol",
        )
    ):
        raise TypeError("contact must expose supermesh geometry for multiplier.family='p0_supermesh'.")

    supermesh_coords = np.asarray(contact.supermesh_coords, dtype=float)
    supermesh_conn = np.asarray(contact.supermesh_conn, dtype=int)
    source_facets_master = np.asarray(contact.source_facets_master, dtype=int)
    source_facets_slave = np.asarray(contact.source_facets_slave, dtype=int)
    facet_conn_master = np.asarray(contact.surface_master.conn, dtype=int)
    facet_conn_slave = np.asarray(contact.surface_slave.conn, dtype=int)
    coords_master = np.asarray(contact.surface_master.coords, dtype=float)
    coords_slave = np.asarray(contact.surface_slave.coords, dtype=float)
    tol = float(contact.tol)

    n_tri = int(supermesh_conn.shape[0])
    n_master_dofs = int(contact.surface_master.n_nodes)
    n_slave_dofs = int(contact.surface_slave.n_nodes)
    B_a = np.zeros((n_tri, n_master_dofs), dtype=float)
    B_b = np.zeros((n_tri, n_slave_dofs), dtype=float)

    for tri_id, (tri, fa, fb) in enumerate(zip(supermesh_conn, source_facets_master, source_facets_slave)):
        a = supermesh_coords[int(tri[0])]
        b = supermesh_coords[int(tri[1])]
        c = supermesh_coords[int(tri[2])]
        centroid = _tri_centroid(a, b, c)
        area = _tri_area(a, b, c)

        facet_master = facet_conn_master[int(fa)]
        facet_slave = facet_conn_slave[int(fb)]
        N_master = _facet_shape_values(centroid, facet_master, coords_master, tol=tol)
        N_slave = _facet_shape_values(centroid, facet_slave, coords_slave, tol=tol)
        B_a[tri_id, facet_master] += area * N_master
        B_b[tri_id, facet_slave] += area * N_slave

    B_a, B_b = _apply_integrated_coarse_p0_groups(B_a, B_b, coarse_patch_ids, backend=backend)
    B_a = _expand_scalar_constraint_dense(B_a, value_dim=int(value_dim), backend=backend)
    B_b = _expand_scalar_constraint_dense(B_b, value_dim=int(value_dim), backend=backend)
    return B_a, B_b


def _assemble_active_master_facet_p0_blocks(
    contact,
    *,
    backend: str,
    value_dim: int,
    coarse_patch_ids: np.ndarray | None = None,
):
    if not all(
        hasattr(contact, name)
        for name in (
            "supermesh_coords",
            "supermesh_conn",
            "source_facets_master",
            "source_facets_slave",
            "surface_master",
            "surface_slave",
            "tol",
        )
    ):
        raise TypeError("contact must expose supermesh geometry for multiplier.family='p0_active'.")

    supermesh_coords = np.asarray(contact.supermesh_coords, dtype=float)
    supermesh_conn = np.asarray(contact.supermesh_conn, dtype=int)
    source_facets_master = np.asarray(contact.source_facets_master, dtype=int)
    source_facets_slave = np.asarray(contact.source_facets_slave, dtype=int)
    active_facets = np.unique(source_facets_master)
    facet_row = {int(f): i for i, f in enumerate(active_facets.tolist())}

    facet_conn_master_all = np.asarray(contact.surface_master.conn, dtype=int)
    facet_conn_slave_all = np.asarray(contact.surface_slave.conn, dtype=int)
    coords_master = np.asarray(contact.surface_master.coords, dtype=float)
    coords_slave = np.asarray(contact.surface_slave.coords, dtype=float)
    tol = float(contact.tol)

    B_a = np.zeros((int(active_facets.shape[0]), int(contact.surface_master.n_nodes)), dtype=float)
    B_b = np.zeros((int(active_facets.shape[0]), int(contact.surface_slave.n_nodes)), dtype=float)

    for tri, fa, fb in zip(supermesh_conn, source_facets_master, source_facets_slave):
        row = facet_row[int(fa)]
        a = supermesh_coords[int(tri[0])]
        b = supermesh_coords[int(tri[1])]
        c = supermesh_coords[int(tri[2])]
        centroid = _tri_centroid(a, b, c)
        area = _tri_area(a, b, c)

        facet_master = facet_conn_master_all[int(fa)]
        facet_slave = facet_conn_slave_all[int(fb)]
        N_master = _facet_shape_values(centroid, facet_master, coords_master, tol=tol)
        N_slave = _facet_shape_values(centroid, facet_slave, coords_slave, tol=tol)
        B_a[row, facet_master] += area * N_master
        B_b[row, facet_slave] += area * N_slave

    B_a, B_b = _apply_integrated_coarse_p0_groups(B_a, B_b, coarse_patch_ids, backend=backend)
    B_a = _expand_scalar_constraint_dense(B_a, value_dim=int(value_dim), backend=backend)
    B_b = _expand_scalar_constraint_dense(B_b, value_dim=int(value_dim), backend=backend)
    return B_a, B_b, facet_conn_master_all[active_facets]


def assemble_contact_constraint_operators(
    contact,
    *,
    law: str | None = None,
    formulation: str | None = None,
    rho: float = 0.0,
    multiplier: ContactMultiplierSpace | None = None,
    backend: str | None = None,
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    warn_float32_assembly_once(context="contact constraint assembly")
    """Assemble constraint-family operators (coupling/B/Kuu, optionally residual/jacobian metadata)."""
    backend = _infer_contact_backend(state, u, params, res_form, weak_form, rho, default="numpy") if backend is None else str(backend).lower()
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    if state is not None and u is not None and state is not u:
        raise ValueError("state and u are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form
    u_eff = state if state is not None else u
    has_eval_inputs = (res_form_eff is not None) or (u_eff is not None) or (params is not None)
    if has_eval_inputs and (res_form_eff is None or u_eff is None or params is None):
        raise ValueError(
            "weak_form/state/params (or res_form/u/params) must be provided together for constraint residual/jacobian evaluation."
        )
    f_arg = None if formulation is None else str(formulation).lower()
    if f_arg is not None and f_arg in {"penalty", "penalty_consistent", "nitsche"}:
        raise ValueError(
            "Constraint operators are multiplier-family only. Use a multiplier/augmented_lagrangian formulation."
        )
    resolved = "mortar"
    law_resolved = str(law) if law is not None else "one_sided_normal_frictionless"
    formulation_resolved = str(formulation) if formulation is not None else "multiplier"
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    if has_eval_inputs and backend != "jax":
        raise NotImplementedError(
            "weak-form contact residual/jacobian evaluation requires backend='jax'. "
            "backend='numpy' remains available for coupling/KKT assembly only."
        )

    if not hasattr(contact, "assemble_contact_coupling_matrices"):
        raise TypeError("contact must provide assemble_contact_coupling_matrices() for constraint operators.")
    coupling_aa, coupling_ab = contact.assemble_contact_coupling_matrices()

    mult_space, facet_conn_master, multiplier_resolved = _resolve_multiplier_spec(
        contact,
        multiplier=multiplier,
        facet_conn_master=None,
    )

    M_aa = _coo_to_dense(coupling_aa.rows, coupling_aa.cols, coupling_aa.data, coupling_aa.shape, backend=backend)
    M_ab = _coo_to_dense(coupling_ab.rows, coupling_ab.cols, coupling_ab.data, coupling_ab.shape, backend=backend)

    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np

    if mult_space == "nodal":
        B_a = M_aa
        B_b = M_ab
    elif mult_space == "dual_nodal":
        B_a, B_b = _dual_nodal_blocks_from_dense(M_aa, M_ab, backend=backend)
    elif mult_space == "coarse_p1":
        C = xp.asarray(multiplier_resolved.coarse_basis)
        if int(C.shape[1]) != int(M_aa.shape[0]):
            raise ValueError("coarse_basis must have shape (n_coarse_nodes, n_master_nodes).")
        B_a = C @ M_aa
        B_b = C @ M_ab
        B_a = _expand_scalar_constraint_dense(
            B_a,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
        B_b = _expand_scalar_constraint_dense(
            B_b,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
    elif mult_space == "p0":
        n_master_nodes = int(coupling_aa.shape[0])
        S_np = _p0_reduction_matrix_from_facets(facet_conn_master, n_master_nodes)
        S = xp.asarray(S_np)
        B_a = S @ M_aa
        B_b = S @ M_ab
        B_a, B_b = _apply_integrated_coarse_p0_groups(
            B_a,
            B_b,
            multiplier_resolved.coarse_patch_ids,
            backend=backend,
        )
        B_a = _expand_scalar_constraint_dense(
            B_a,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
        B_b = _expand_scalar_constraint_dense(
            B_b,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
    elif mult_space == "p0_active":
        B_a, B_b, facet_conn_master = _assemble_active_master_facet_p0_blocks(
            contact,
            backend=backend,
            value_dim=int(multiplier_resolved.value_dim),
            coarse_patch_ids=multiplier_resolved.coarse_patch_ids,
        )
    elif mult_space == "p0_supermesh":
        B_a, B_b = _assemble_supermesh_triangle_p0_blocks(
            contact,
            backend=backend,
            value_dim=int(multiplier_resolved.value_dim),
            coarse_patch_ids=multiplier_resolved.coarse_patch_ids,
        )
        if backend == "jax":
            B_a = xp.asarray(B_a)
            B_b = xp.asarray(B_b)
    else:
        raise ValueError(
            "multiplier.family must be 'nodal', 'dual_nodal', 'coarse_p1', "
            "'p0', 'p0_active', or 'p0_supermesh'."
        )

    B_a, B_b = _apply_coarse_mortar_projection(B_a, B_b, multiplier_resolved, backend=backend)
    B = xp.concatenate([B_a, -B_b], axis=1)
    Kuu = xp.asarray(rho) * (B.T @ B)
    residual = None
    jacobian = None
    if has_eval_inputs:
        if not hasattr(contact, "assemble_residual") or not hasattr(contact, "assemble_jacobian"):
            raise TypeError("contact must provide assemble_residual() and assemble_jacobian() for weak-form evaluation.")
        residual = contact.assemble_residual(res_form_eff, u_eff, params, normal_source=normal_source)
        jacobian = contact.assemble_jacobian(
            res_form_eff,
            u_eff,
            params,
            normal_source=normal_source,
            sparse=sparse,
            backend=backend,
            batch_jac=batch_jac,
        )
    return MultiplierContactContribution(
        enforcement=resolved,
        law=law_resolved,
        formulation=formulation_resolved,
        coupling_aa=coupling_aa,
        coupling_ab=coupling_ab,
        B_a=B_a,
        B_b=B_b,
        B=B,
        Kuu=Kuu,
        residual=residual,
        jacobian=jacobian,
        facet_conn_master=facet_conn_master,
        rho=rho,
        multiplier=multiplier_resolved,
    )


def _resolve_contact_operator_enforcement(
    *,
    enforcement: str | None = None,
    method: str | None = None,
    formulation: str | None = None,
    multiplier: ContactMultiplierSpace | None = None,
) -> str:
    if enforcement is not None and method is not None and str(enforcement).lower() != str(method).lower():
        raise ValueError("enforcement and method are aliases; provide only one effective value.")
    value = enforcement if enforcement is not None else method
    if value is None and formulation is not None:
        formulation_key = str(formulation).lower()
        if formulation_key in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
            value = "constraint"
        elif formulation_key in {"penalty", "penalty_consistent", "nitsche"}:
            value = "penalty"
    if value is None:
        value = "constraint" if multiplier is not None else "penalty"
    value_key = str(value).lower()
    if value_key in {"penalty", "nitsche", "penalty_family", "penalty-family"}:
        return "penalty"
    if value_key in {"constraint", "mortar", "multiplier", "constraint_family", "constraint-family", "augmented_lagrangian"}:
        return "constraint"
    raise ValueError("enforcement must resolve to either 'penalty' or 'constraint'.")


def assemble_contact_penalty_operators(
    contact,
    *,
    law: str | None = None,
    formulation: str | None = None,
    backend: str | None = None,
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    warn_float32_assembly_once(context="contact penalty assembly")
    """Assemble penalty-family operators (residual/jacobian)."""
    f_arg = None if formulation is None else str(formulation).lower()
    if f_arg is not None and f_arg in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
        raise ValueError(
            "Penalty operators are penalty-family only. Use penalty/penalty_consistent formulation."
        )
    resolved = "nitsche"
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    if state is not None and u is not None and state is not u:
        raise ValueError("state and u are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form
    u_eff = state if state is not None else u

    law_resolved = str(law) if law is not None else "one_sided_normal_frictionless"
    formulation_resolved = str(formulation) if formulation is not None else "penalty_consistent"
    backend = _infer_contact_backend(contact, res_form_eff, u_eff, params, default="jax") if backend is None else str(backend).lower()
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    if backend != "jax":
        raise NotImplementedError(
            "Penalty-family weak-form Jacobian assembly requires backend='jax'. "
            "backend='numpy' for contact Jacobians has been removed."
        )
    if res_form_eff is None or u_eff is None or params is None:
        raise ValueError("weak_form/state/params (or res_form/u/params) are required for penalty operators.")
    if not hasattr(contact, "assemble_residual") or not hasattr(contact, "assemble_jacobian"):
        raise TypeError("contact must provide assemble_residual() and assemble_jacobian() for penalty operators.")
    residual = contact.assemble_residual(res_form_eff, u_eff, params, normal_source=normal_source)
    jacobian = contact.assemble_jacobian(
        res_form_eff,
        u_eff,
        params,
        normal_source=normal_source,
        sparse=sparse,
        backend=backend,
        batch_jac=batch_jac,
    )
    return PenaltyContactContribution(
        enforcement=resolved,
        law=law_resolved,
        formulation=formulation_resolved,
        residual=residual,
        jacobian=jacobian,
    )


def assemble_contact_kkt(
    coupling_aa,
    coupling_ab,
    *,
    rho: float = 0.0,
    multiplier: ContactMultiplierSpace | None = None,
    facet_conn_master: np.ndarray | None = None,
    backend: str | None = None,
    format: str = "fluxsparse",
    return_blocks: bool = False,
):
    warn_float32_assembly_once(context="contact KKT assembly")
    """
    Assemble contact KKT block from coupling matrices.

    KKT is assembled as:
      B = [B_a, -B_b]
      Kuu = rho * (B^T B)
      KKT = [[Kuu, B^T], [B, 0]]

    multiplier:
    - ``family="nodal"``: lambda lives on interface nodal basis (B_a=M_aa, B_b=M_ab)
    - ``family="coarse_p1"``: lambda lives on user-supplied coarse P1 rows (B_*=C M_*)
    - ``family="dual_nodal"``: master-side dual nodal basis (B_a=I, B_b=pinv(M_aa) M_ab)
    - ``family="p0"``: lambda is facet-wise constant on master side (B_* = S * M_*)
    - ``family="p0_active"``/``family="p0_supermesh"``: use ``assemble_contact_constraint_operators`` and pass ``ops`` to the builder
    """
    backend = _infer_contact_backend(coupling_aa, coupling_ab, rho, multiplier, default="numpy") if backend is None else str(backend).lower()
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    multiplier_eff = ContactMultiplierSpace() if multiplier is None else multiplier
    mult_space, facet_conn_master, _ = _resolve_multiplier_spec(
        None,
        multiplier=multiplier_eff,
        facet_conn_master=facet_conn_master,
    )
    if mult_space in {"p0_active", "p0_supermesh"}:
        raise NotImplementedError(
            "assemble_contact_kkt(..., multiplier.family in {'p0_active', 'p0_supermesh'}) is not supported; "
            "use assemble_contact_constraint_operators(...) and CoupledSystemBuilder.add_contact_mortar(...)."
        )
    if format not in {"dense", "fluxsparse", "bcoo"}:
        raise ValueError("format must be 'dense', 'fluxsparse', or 'bcoo'")
    if return_blocks and format != "dense":
        raise ValueError("return_blocks=True is supported only with format='dense'.")

    if format != "dense":
        import jax

        if isinstance(rho, jax.core.Tracer):
            raise ValueError("format='fluxsparse'/'bcoo' currently requires rho to be a concrete scalar.")
        rows, cols, data, n_total = _kkt_coo_from_coupling(
            coupling_aa,
            coupling_ab,
            rho=float(rho),
            multiplier_space=mult_space,
            facet_conn_master=facet_conn_master,
            multiplier_value_dim=int(getattr(multiplier_eff, "value_dim", 1)),
            coarse_rank=getattr(multiplier_eff, "coarse_rank", None),
            coarse_projection=getattr(multiplier_eff, "coarse_projection", None),
            coarse_mode=getattr(multiplier_eff, "coarse_mode", None),
            coarse_energy_tol=getattr(multiplier_eff, "coarse_energy_tol", None),
            coarse_rtol=getattr(multiplier_eff, "coarse_rtol", None),
            coarse_max_rank=getattr(multiplier_eff, "coarse_max_rank", None),
            coarse_patch_ids=getattr(multiplier_eff, "coarse_patch_ids", None),
            coarse_basis=getattr(multiplier_eff, "coarse_basis", None),
        )
        if format == "fluxsparse":
            from ..solver import FluxSparseMatrix

            return FluxSparseMatrix(rows, cols, data, n_dofs=n_total)
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        idx = jnp.stack([jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(cols, dtype=jnp.int32)], axis=-1)
        return jsparse.BCOO((jnp.asarray(data), idx), shape=(n_total, n_total))

    M_aa = _coo_to_dense(coupling_aa.rows, coupling_aa.cols, coupling_aa.data, coupling_aa.shape, backend=backend)
    M_ab = _coo_to_dense(coupling_ab.rows, coupling_ab.cols, coupling_ab.data, coupling_ab.shape, backend=backend)

    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np

    if mult_space == "nodal":
        B_a = M_aa
        B_b = M_ab
    elif mult_space == "dual_nodal":
        B_a, B_b = _dual_nodal_blocks_from_dense(M_aa, M_ab, backend=backend)
    elif mult_space == "coarse_p1":
        C = xp.asarray(multiplier_eff.coarse_basis)
        if int(C.shape[1]) != int(M_aa.shape[0]):
            raise ValueError("coarse_basis must have shape (n_coarse_nodes, n_master_nodes).")
        B_a = C @ M_aa
        B_b = C @ M_ab
    else:
        n_master_nodes = int(coupling_aa.shape[0])
        S_np = _p0_reduction_matrix_from_facets(facet_conn_master, n_master_nodes)
        S = xp.asarray(S_np)
        B_a = S @ M_aa
        B_b = S @ M_ab
        B_a, B_b = _apply_integrated_coarse_p0_groups(
            B_a,
            B_b,
            getattr(multiplier_eff, "coarse_patch_ids", None),
            backend=backend,
        )
    B_a = _expand_scalar_constraint_dense(
        B_a,
        value_dim=int(getattr(multiplier_eff, "value_dim", 1)),
        backend=backend,
    )
    B_b = _expand_scalar_constraint_dense(
        B_b,
        value_dim=int(getattr(multiplier_eff, "value_dim", 1)),
        backend=backend,
    )
    B_a, B_b = _apply_coarse_mortar_projection(B_a, B_b, multiplier_eff, backend=backend)

    B = xp.concatenate([B_a, -B_b], axis=1)
    Kuu = xp.asarray(rho) * (B.T @ B)
    n_lambda = int(B.shape[0])
    Zll = xp.zeros((n_lambda, n_lambda), dtype=Kuu.dtype)
    KKT = xp.block([[Kuu, B.T], [B, Zll]])

    if return_blocks:
        return KKT, B_a, B_b
    return KKT


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
            backend = _infer_contact_backend(kkt_matrix, rhs, default="numpy")
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


def _solve_kkt_numpy(kkt_matrix, rhs, cfg: ContactKKTSolveConfig):
    A_csr = _as_numpy_csr(kkt_matrix)
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception:
        sp = None
        spla = None

    if A_csr is not None and spla is not None:
        if float(cfg.diagonal_shift) != 0.0:
            A_csr = A_csr + float(cfg.diagonal_shift) * sp.eye(A_csr.shape[0], format="csr")
        return np.asarray(spla.spsolve(A_csr, np.asarray(rhs, dtype=float)))

    if not bool(cfg.allow_dense_fallback):
        raise ValueError("Dense fallback is disabled by ContactKKTSolveConfig.allow_dense_fallback.")
    A = _as_numpy_dense(kkt_matrix)
    if float(cfg.diagonal_shift) != 0.0:
        A = A + float(cfg.diagonal_shift) * np.eye(A.shape[0], dtype=A.dtype)
    return np.linalg.solve(A, np.asarray(rhs, dtype=float))


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



def _surface_node_normals(surface: SurfaceMesh, *, normal_sign: float = 1.0, tol: float = 1e-12) -> np.ndarray | None:
    if not hasattr(surface, "facet_normals"):
        return None
    facet_normals = np.asarray(surface.facet_normals(), dtype=float)
    facets = np.asarray(surface.conn, dtype=int)
    n_nodes = int(np.asarray(surface.coords).shape[0])
    if facet_normals.ndim != 2 or facet_normals.shape[0] != facets.shape[0]:
        return None
    node_normals = np.zeros((n_nodes, facet_normals.shape[1]), dtype=float)
    counts = np.zeros((n_nodes,), dtype=float)
    for f_id, facet in enumerate(facets):
        normal = float(normal_sign) * facet_normals[int(f_id)]
        for node in np.asarray(facet, dtype=int):
            node_normals[int(node)] += normal
            counts[int(node)] += 1.0
    valid = counts > 0.0
    if not np.any(valid):
        return None
    node_normals[valid] /= counts[valid, None]
    norms = np.linalg.norm(node_normals, axis=1)
    good = norms > float(tol)
    node_normals[good] /= norms[good, None]
    return node_normals


def _onesided_gap_diagnostics(
    contact: OneSidedContactSurfaceSpace,
    state_sol: Mapping[str, Any] | Sequence[Any] | Any,
    *,
    u_hat_fn: SurfaceHatFn | None,
    state_field: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if u_hat_fn is None:
        return None, None
    if isinstance(state_sol, Mapping):
        if state_field is not None:
            u_state = state_sol.get(state_field)
        elif len(state_sol) == 1:
            u_state = next(iter(state_sol.values()))
        else:
            return None, None
    elif isinstance(state_sol, Sequence) and not hasattr(state_sol, "shape"):
        if len(state_sol) != 1:
            return None, None
        u_state = state_sol[0]
    else:
        u_state = state_sol
    if u_state is None:
        return None, None

    coords = np.asarray(contact.surface_slave.coords, dtype=float)
    value_dim = int(contact.value_dim)
    u_nodes = np.asarray(u_state, dtype=float).reshape(-1, value_dim)
    u_hat = np.asarray(u_hat_fn(coords), dtype=float)
    if u_hat.shape != u_nodes.shape:
        return None, None
    node_normals = _surface_node_normals(contact.surface_slave, normal_sign=float(contact.normal_sign))
    if node_normals is None or node_normals.shape != u_nodes.shape:
        return None, None
    gap_n = np.einsum("ni,ni->n", u_nodes - u_hat, node_normals)
    active_mask = gap_n < 0.0
    return gap_n, active_mask


def _params_with_updates(params: "WeakParams", **updates: Any) -> "WeakParams":
    data = dict(getattr(params, "_data", {}))
    if not data:
        data = dict(vars(params))
    data.update(updates)
    from ..core.weakform import Params
    return Params(**data)


def _make_al_u_hat_fn(
    contact: OneSidedContactSurfaceSpace,
    base_u_hat_fn: SurfaceHatFn,
    lambda_n: np.ndarray,
    *,
    alpha: float,
) -> SurfaceHatFn:
    coords = np.asarray(contact.surface_slave.coords, dtype=float)
    node_normals = _surface_node_normals(contact.surface_slave, normal_sign=float(contact.normal_sign))
    if node_normals is None:
        raise ValueError("surface normals are required for augmented-Lagrangian one-sided updates")
    corr_nodes = (np.asarray(lambda_n, dtype=float).reshape(-1, 1) / max(float(alpha), 1e-30)) * node_normals

    def _u_hat_eff(x_q: np.ndarray) -> np.ndarray:
        x_q = np.asarray(x_q, dtype=float)
        base = np.asarray(base_u_hat_fn(x_q), dtype=float)
        diffs = x_q[:, None, :] - coords[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        exact = d2 <= 1e-24
        weights = 1.0 / np.maximum(d2, 1e-24)
        weights /= np.sum(weights, axis=1, keepdims=True)
        corr = weights @ corr_nodes
        if np.any(exact):
            row_ids = np.nonzero(np.any(exact, axis=1))[0]
            for row in row_ids:
                corr[row] = corr_nodes[int(np.argmax(exact[row]))]
        return base - corr

    return _u_hat_eff

def update_contact_state_penalty(
    *,
    state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | npt.ArrayLike | None,
    contact_state: ContactState | None = None,
    geometry: str = "current",
    gap_n: npt.ArrayLike | None = None,
    active_mask: npt.ArrayLike | None = None,
    lambda_n: npt.ArrayLike | None = None,
    penalty_param: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContactState:
    """Update numeric penalty-contact diagnostics in a state-explicit form."""
    base = ContactState(interface_kind="penalty", geometry="reference") if contact_state is None else contact_state
    merged_metadata = dict(base.metadata)
    if metadata is not None:
        merged_metadata.update(dict(metadata))
    gap_np = None if gap_n is None else np.asarray(gap_n)
    active_mask_np = None if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active_mask_np is None and gap_np is not None:
        active_mask_np = np.asarray(gap_np < 0.0, dtype=bool)
    lambda_np = None if lambda_n is None else np.asarray(lambda_n)
    resolved_penalty = penalty_param if penalty_param is not None else base.penalty_param
    active_set = base.active_set
    if active_mask_np is not None:
        active_set = "active" if bool(np.any(active_mask_np)) else "inactive"
    return replace(
        base,
        geometry=str(geometry),
        iteration=int(base.iteration) + 1,
        active_set=active_set,
        field_summary=_summarize_contact_field_state(state),
        gap_n=gap_np,
        active_mask=active_mask_np,
        lambda_n=lambda_np,
        penalty_param=resolved_penalty,
        metadata=merged_metadata,
    )


def solve_contact_penalty_jax(
    contact,
    *,
    weak_form: MixedSurfaceResidualForm | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    state0: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
    params: "WeakParams",
    normal_source: str = "master",
    u_hat_fn: SurfaceHatFn | None = None,
    u_master: npt.ArrayLike | None = None,
    state_field: str | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    diagonal_shift: float = 0.0,
    contact_state: ContactState | None = None,
    geometry: str = "current",
    metadata: Mapping[str, Any] | None = None,
    state_updater: Callable[..., ContactState] | None = None,
) -> ContactSolveResult:
    """Solve a penalty-contact residual with a JAX-friendly dense Newton loop."""
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form

    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from ..solver.newton_jax import newton_solve_jax

    state_vec0, unravel_state = ravel_pytree(state0)

    def _primary_state_entry(u_state):
        if isinstance(u_state, Mapping):
            if state_field is not None:
                if state_field not in u_state:
                    raise KeyError(f"state_field {state_field!r} was not found in state.")
                return state_field, u_state[state_field]
            if len(u_state) != 1:
                raise ValueError("One-sided penalty solve requires a single-state mapping or explicit state_field.")
            key = next(iter(u_state))
            return str(key), u_state[key]
        if isinstance(u_state, Sequence) and not hasattr(u_state, "shape"):
            if len(u_state) != 1:
                raise ValueError("One-sided penalty solve requires a single state vector.")
            return "arg0", u_state[0]
        return "arg0", u_state

    is_onesided = isinstance(contact, OneSidedContactSurfaceSpace)
    if not is_onesided and res_form_eff is None:
        raise ValueError("weak_form or res_form is required.")
    if is_onesided and u_hat_fn is None:
        raise ValueError("u_hat_fn is required when contact is OneSidedContactSurfaceSpace.")

    u_master_arr = None if u_master is None else np.asarray(u_master)

    def _assemble_ops(u_vec):
        u_state = unravel_state(u_vec)
        if is_onesided:
            _field_name, u_local = _primary_state_entry(u_state)
            K, f = contact.assemble_bilinear(u_hat_fn, params, u_master=u_master_arr)
            K_jax = jnp.asarray(K)
            f_jax = jnp.asarray(f)
            u_local_jax = jnp.ravel(jnp.asarray(u_local))
            return PenaltyContactContribution(
                enforcement="nitsche",
                law="one_sided_normal_frictionless",
                formulation="penalty_consistent",
                residual=K_jax @ u_local_jax + f_jax,
                jacobian=K_jax,
            )
        return assemble_contact_penalty_operators(
            contact,
            weak_form=res_form_eff,
            state=u_state,
            params=params,
            backend="jax",
            normal_source=normal_source,
            sparse=False,
        )

    def residual_fn(u_vec, _params):
        _ = _params
        return jnp.ravel(jnp.asarray(_assemble_ops(u_vec).residual))

    def jacobian_fn(u_vec, _params):
        _ = _params
        J = _assemble_ops(u_vec).jacobian
        if hasattr(J, "to_dense"):
            J = J.to_dense()
        return jnp.asarray(J)

    u_sol_vec, info = newton_solve_jax(
        residual_fn,
        jacobian_fn,
        jnp.asarray(state_vec0),
        params,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        diagonal_shift=diagonal_shift,
    )
    state_sol = unravel_state(u_sol_vec)
    updater = update_contact_state_penalty if state_updater is None else state_updater
    gap_n = None
    active_mask = None
    if is_onesided:
        gap_n, active_mask = _onesided_gap_diagnostics(
            contact,
            state_sol,
            u_hat_fn=u_hat_fn,
            state_field=state_field,
        )
    contact_state_sol = updater(
        state=state_sol,
        contact_state=contact_state,
        geometry=geometry,
        gap_n=gap_n,
        active_mask=active_mask,
        penalty_param=float(getattr(params, "alpha", 0.0)) if hasattr(params, "alpha") else None,
        metadata=metadata,
    )
    return ContactSolveResult(
        state=state_sol,
        contact_state=contact_state_sol,
        converged=info.converged,
        iters=info.iters,
        residual_norm=info.residual_norm,
    )

def solve_contact_al_jax(
    contact,
    *,
    state0: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
    params: "WeakParams",
    u_hat_fn: SurfaceHatFn,
    state_field: str | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    outer_maxiter: int = 3,
    gap_tol: float = 1e-6,
    penalty_growth: float = 2.0,
    diagonal_shift: float = 0.0,
    contact_state: ContactState | None = None,
    geometry: str = "current",
    metadata: Mapping[str, Any] | None = None,
) -> ContactSolveResult:
    """Minimal one-sided augmented-Lagrangian outer loop built on penalty Newton solves."""
    if not isinstance(contact, OneSidedContactSurfaceSpace):
        raise TypeError("solve_contact_al_jax currently supports only OneSidedContactSurfaceSpace.")
    alpha = float(getattr(params, "alpha", 0.0))
    if alpha <= 0.0:
        raise ValueError("params.alpha must be positive for solve_contact_al_jax.")
    n_nodes = int(contact.surface_slave.n_nodes)
    lambda_n = (
        np.zeros((n_nodes,), dtype=float)
        if contact_state is None or contact_state.lambda_n is None
        else np.asarray(contact_state.lambda_n, dtype=float).reshape(-1)
    )
    state_curr = state0
    contact_state_curr = contact_state
    inner_result: ContactSolveResult | None = None
    converged = False

    for outer in range(int(outer_maxiter)):
        params_eff = _params_with_updates(params, alpha=alpha)
        u_hat_eff = _make_al_u_hat_fn(contact, u_hat_fn, lambda_n, alpha=alpha)
        inner_result = solve_contact_penalty_jax(
            contact,
            state0=state_curr,
            params=params_eff,
            u_hat_fn=u_hat_eff,
            state_field=state_field,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            diagonal_shift=diagonal_shift,
            contact_state=contact_state_curr,
            geometry=geometry,
            metadata=metadata,
        )
        gap_n = None if inner_result.contact_state.gap_n is None else np.asarray(inner_result.contact_state.gap_n, dtype=float)
        if gap_n is None:
            raise RuntimeError("solve_contact_al_jax requires one-sided gap diagnostics.")
        lambda_n = np.maximum(0.0, lambda_n - alpha * gap_n)
        active_mask = gap_n < 0.0
        contact_state_curr = update_contact_state_penalty(
            state=inner_result.state,
            contact_state=inner_result.contact_state,
            geometry=geometry,
            gap_n=gap_n,
            active_mask=active_mask,
            lambda_n=lambda_n,
            penalty_param=alpha,
            metadata={**dict(metadata or {}), "al_outer_iter": outer + 1},
        )
        state_curr = inner_result.state
        penetration = float(np.max(np.maximum(-gap_n, 0.0))) if gap_n.size else 0.0
        if penetration <= float(gap_tol):
            converged = True
            break
        alpha *= float(penalty_growth)

    if inner_result is None:
        raise RuntimeError("solve_contact_al_jax executed zero outer iterations.")
    return ContactSolveResult(
        state=inner_result.state,
        contact_state=contact_state_curr,
        converged=np.asarray(converged),
        iters=np.asarray(int(contact_state_curr.iteration)),
        residual_norm=inner_result.residual_norm,
    )


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
    operators: ContactOperators | None = None,
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


@dataclass(frozen=True)
class ContactSide:
    surface: SurfaceMesh
    elem_conn: np.ndarray | None
    value_dim: int
    space: object | None = None

    @classmethod
    def from_facets(
        cls,
        mesh: BaseMesh,
        facets: np.ndarray,
        space=None,
        *,
        value_dim: int | None = None,
        mode: str = "touching",
    ):
        side = mesh.surface_with_elem_conn_from_facets(facets, mode=mode)
        if value_dim is None:
            if space is None:
                raise ValueError("space or value_dim is required for ContactSide.from_facets")
            value_dim = int(getattr(space, "value_dim", 1))
        return cls(surface=side.surface, elem_conn=side.elem_conn, value_dim=int(value_dim), space=space)

    @classmethod
    def from_surfaces(
        cls,
        surface: SurfaceMesh,
        *,
        elem_conn: np.ndarray | None = None,
        value_dim: int = 1,
        space: object | None = None,
    ):
        return cls(surface=surface, elem_conn=elem_conn, value_dim=int(value_dim), space=space)


def _facet_map_for_elem_conn(surface: SurfaceMesh, elem_conn: np.ndarray | None) -> np.ndarray:
    if elem_conn is None:
        raise ValueError("elem_conn is required to build facet_to_elem mapping.")
    if elem_conn.shape[1] in {4, 10}:
        return map_surface_facets_to_tet_elements(surface, elem_conn)
    if elem_conn.shape[1] in {8, 20, 27}:
        return map_surface_facets_to_hex_elements(surface, elem_conn)
    raise NotImplementedError("elem_conn must be tet4/tet10/hex8/hex20/hex27")


def facet_gap_values(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
) -> tuple[np.ndarray, float]:
    """
    Compute per-facet gap values for a one-sided contact plane.

    Returns (g_f, min_g_all) where g_f is reduced per facet and min_g_all is
    the global minimum node gap.
    """
    coords_np = np.asarray(coords, dtype=float)
    if value_dim is None:
        value_dim = int(coords_np.shape[1])
    u_nodes = np.asarray(u, dtype=float).reshape(-1, value_dim)
    x_cur = coords_np + u_nodes
    g_all = np.dot(x_cur, np.asarray(n, dtype=float)) - float(c)
    min_g_all = float(np.min(g_all)) if g_all.size else 0.0
    if facets is None or len(facets) == 0:
        return np.zeros((0,), dtype=float), min_g_all
    if reduce == "min":
        g_f = np.array([np.min(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    elif reduce == "mean":
        g_f = np.array([np.mean(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    else:
        raise ValueError("reduce must be 'min' or 'mean'")
    return g_f, min_g_all


def active_contact_facets(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
    threshold: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return active facet indices and global minimum gap for one-sided contact."""
    g_f, min_g_all = facet_gap_values(
        coords,
        facets,
        u,
        n,
        c,
        value_dim=value_dim,
        reduce=reduce,
    )
    active_ids = np.nonzero(g_f < threshold)[0]
    return active_ids, min_g_all


@dataclass(frozen=True)
class OneSidedContact:
    side: ContactSide
    n: np.ndarray | None
    c: float
    k: float
    beta: float
    quad_order: int = 2
    normal_sign: float = 1.0
    tol: float = 1e-8
    facet_map: np.ndarray | None = None

    @classmethod
    def from_side(
        cls,
        side: ContactSide,
        *,
        n: np.ndarray | None,
        c: float,
        k: float,
        beta: float,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
        facet_map: np.ndarray | None = None,
    ) -> "OneSidedContact":
        if facet_map is None:
            facet_map = _facet_map_for_elem_conn(side.surface, side.elem_conn)
        return cls(
            side=side,
            n=n,
            c=float(c),
            k=float(k),
            beta=float(beta),
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
            facet_map=facet_map,
        )

    def assemble(self, u, *, return_metrics: bool = False):
        return assemble_contact_onesided_floor(
            self.side.surface,
            np.asarray(u, dtype=float),
            n=None if self.n is None else np.asarray(self.n, dtype=float),
            c=self.c,
            k=self.k,
            beta=self.beta,
            value_dim=self.side.value_dim,
            elem_conn=np.asarray(self.side.elem_conn) if self.side.elem_conn is not None else None,
            facet_to_elem=self.facet_map,
            quad_order=self.quad_order,
            normal_sign=self.normal_sign,
            tol=self.tol,
            return_metrics=return_metrics,
        )


@dataclass(eq=False)
class OneSidedContactSurfaceSpace:
    """Surface wrapper for one-sided (Dirichlet) contact assembly."""

    surface_slave: SurfaceMesh
    elem_conn_slave: np.ndarray
    facet_to_elem_slave: np.ndarray
    value_dim: int = 1
    quad_order: int = 2
    normal_sign: float = 1.0
    tol: float = 1e-8
    surface_master: SurfaceMesh | None = None
    elem_conn_master: np.ndarray | None = None
    facet_to_elem_master: np.ndarray | None = None

    @classmethod
    def from_side(
        cls,
        side: ContactSide,
        *,
        surface_master: SurfaceMesh | None = None,
        elem_conn_master: np.ndarray | None = None,
        facet_to_elem_master: np.ndarray | None = None,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
    ) -> "OneSidedContactSurfaceSpace":
        if side.elem_conn is None:
            raise ValueError("side.elem_conn is required for one-sided assembly")
        facet_map_slave = _facet_map_for_elem_conn(side.surface, side.elem_conn)
        facet_map_master = facet_to_elem_master
        if surface_master is not None and elem_conn_master is not None and facet_map_master is None:
            facet_map_master = _facet_map_for_elem_conn(surface_master, elem_conn_master)
        return cls(
            surface_slave=side.surface,
            elem_conn_slave=np.asarray(side.elem_conn, dtype=int),
            facet_to_elem_slave=np.asarray(facet_map_slave, dtype=int),
            value_dim=int(side.value_dim),
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
            surface_master=surface_master,
            elem_conn_master=None if elem_conn_master is None else np.asarray(elem_conn_master, dtype=int),
            facet_to_elem_master=None if facet_map_master is None else np.asarray(facet_map_master, dtype=int),
        )

    @classmethod
    def from_facets(
        cls,
        mesh: BaseMesh,
        facets: np.ndarray,
        space=None,
        *,
        surface_master: SurfaceMesh | None = None,
        elem_conn_master: np.ndarray | None = None,
        facet_to_elem_master: np.ndarray | None = None,
        value_dim: int | None = None,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
        mode: str = "touching",
    ) -> "OneSidedContactSurfaceSpace":
        side = ContactSide.from_facets(mesh, facets, space, value_dim=value_dim, mode=mode)
        return cls.from_side(
            side,
            surface_master=surface_master,
            elem_conn_master=elem_conn_master,
            facet_to_elem_master=facet_to_elem_master,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
        )

    def initialize_state(self, *, metadata: Mapping[str, Any] | None = None) -> ContactState:
        return ContactState(
            interface_kind="one_sided",
            geometry="reference",
            iteration=0,
            active_set=None,
            field_summary={"slave": (int(self.elem_conn_slave.max()) + 1, int(self.value_dim))},
            metadata=dict(metadata or {}),
        )

    def update_state(
        self,
        *,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        contact_state: ContactState | None = None,
        geometry: str = "current",
        active_set: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContactState:
        base = self.initialize_state() if contact_state is None else contact_state
        merged_metadata = dict(base.metadata)
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        return replace(
            base,
            geometry=str(geometry),
            iteration=int(base.iteration) + 1,
            active_set=active_set if active_set is not None else base.active_set,
            field_summary=_summarize_contact_field_state(state),
            metadata=merged_metadata,
        )

    def assemble_bilinear(
        self,
        u_hat_fn: SurfaceHatFn | None,
        params: "WeakParams",
        *,
        u_master: np.ndarray | None = None,
        grad_source: str = "volume",
        dof_source: str = "volume",
        quad_order: int | None = None,
        normal_sign: float | None = None,
        tol: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return assemble_onesided_bilinear(
            self.surface_slave,
            u_hat_fn,
            params,
            surface_master=self.surface_master,
            u_master=u_master,
            value_dim=self.value_dim,
            elem_conn=self.elem_conn_slave,
            facet_to_elem=self.facet_to_elem_slave,
            elem_conn_master=self.elem_conn_master,
            facet_to_elem_master=self.facet_to_elem_master,
            grad_source=grad_source,
            dof_source=dof_source,
            quad_order=self.quad_order if quad_order is None else int(quad_order),
            normal_sign=self.normal_sign if normal_sign is None else float(normal_sign),
            tol=self.tol if tol is None else float(tol),
        )


@dataclass(eq=False)
class ContactSurfaceSpace:
    """Surface interface wrapper for contact assembly on a supermesh."""

    surface_master: SurfaceMesh
    surface_slave: SurfaceMesh
    supermesh_coords: np.ndarray
    supermesh_conn: np.ndarray
    source_facets_master: np.ndarray
    source_facets_slave: np.ndarray
    elem_conn_master: np.ndarray | None
    elem_conn_slave: np.ndarray | None
    facet_to_elem_master: np.ndarray | None
    facet_to_elem_slave: np.ndarray | None
    field_master: str = "a"
    field_slave: str = "b"
    value_dim_master: int = 1
    value_dim_slave: int = 1
    space_mode_master: str = "nodal"
    space_mode_slave: str = "nodal"
    facet_dofs_master: np.ndarray | None = None
    facet_dofs_slave: np.ndarray | None = None
    trial_value_dim_master: int | None = None
    trial_value_dim_slave: int | None = None
    trial_space_mode_master: str | None = None
    trial_space_mode_slave: str | None = None
    trial_facet_dofs_master: np.ndarray | None = None
    trial_facet_dofs_slave: np.ndarray | None = None
    quad_order: int = 0
    normal_sign: float | None = None
    tol: float = 1e-8
    backend: str = "jax"
    batch_jac: bool | None = None
    supermesh_quad_cache: Any | None = None
    _compiled_bilinear_cache: dict[tuple[int, str], MixedSurfaceResidualForm] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_surfaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slave: SurfaceMesh,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        trial_value_dim_master: int | None = None,
        trial_value_dim_slave: int | None = None,
        trial_space_mode_master: str | None = None,
        trial_space_mode_slave: str | None = None,
        trial_facet_dofs_master: np.ndarray | None = None,
        trial_facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        import hashlib
        import os
        backend = "jax" if backend is None else str(backend).lower()

        if setup_cache_enabled is None:
            setup_cache_enabled = os.getenv("FLUXFEM_CONTACT_SETUP_CACHE", "0") not in ("0", "", "false", "False")
        if setup_cache_trace is None:
            setup_cache_trace = os.getenv("FLUXFEM_CONTACT_SETUP_CACHE_TRACE", "0") not in ("0", "", "false", "False")

        def _array_sig(arr: np.ndarray) -> tuple:
            arr_c = np.ascontiguousarray(arr)
            h = hashlib.blake2b(arr_c.view(np.uint8), digest_size=8).hexdigest()
            return (arr_c.shape, str(arr_c.dtype), h)

        if setup_cache_enabled:
            global _CONTACT_SETUP_CACHE
            try:
                _CONTACT_SETUP_CACHE
            except NameError:
                _CONTACT_SETUP_CACHE = {}
            key = (
                _array_sig(np.asarray(surface_master.coords)),
                _array_sig(np.asarray(surface_master.conn)),
                _array_sig(np.asarray(surface_slave.coords)),
                _array_sig(np.asarray(surface_slave.conn)),
                None if elem_conn_master is None else _array_sig(np.asarray(elem_conn_master)),
                None if elem_conn_slave is None else _array_sig(np.asarray(elem_conn_slave)),
                field_master,
                field_slave,
                int(value_dim_master),
                int(value_dim_slave),
                str(space_mode_master),
                str(space_mode_slave),
                None if facet_dofs_master is None else _array_sig(np.asarray(facet_dofs_master)),
                None if facet_dofs_slave is None else _array_sig(np.asarray(facet_dofs_slave)),
                None if trial_value_dim_master is None else int(trial_value_dim_master),
                None if trial_value_dim_slave is None else int(trial_value_dim_slave),
                None if trial_space_mode_master is None else str(trial_space_mode_master),
                None if trial_space_mode_slave is None else str(trial_space_mode_slave),
                None if trial_facet_dofs_master is None else _array_sig(np.asarray(trial_facet_dofs_master)),
                None if trial_facet_dofs_slave is None else _array_sig(np.asarray(trial_facet_dofs_slave)),
                int(quad_order),
                float(normal_sign) if normal_sign is not None else None,
                float(tol),
                backend,
                bool(batch_jac) if batch_jac is not None else None,
            )
            cached = _CONTACT_SETUP_CACHE.get(key)
            if cached is not None:
                if setup_cache_trace:
                    print(
                        f"[contact] setup cache hit n_tris={int(cached.supermesh_conn.shape[0])}",
                        flush=True,
                    )
                return cached

        sm = build_surface_supermesh(surface_master, surface_slave, tol=tol)
        facet_map_master = None
        facet_map_slave = None
        if elem_conn_master is not None:
            if elem_conn_master.shape[1] in {4, 10}:
                facet_map_master = map_surface_facets_to_tet_elements(surface_master, elem_conn_master)
            elif elem_conn_master.shape[1] in {8, 20, 27}:
                facet_map_master = map_surface_facets_to_hex_elements(surface_master, elem_conn_master)
            else:
                raise NotImplementedError("elem_conn_master must be tet4/tet10/hex8/hex20/hex27")
        if elem_conn_slave is not None:
            if elem_conn_slave.shape[1] in {4, 10}:
                facet_map_slave = map_surface_facets_to_tet_elements(surface_slave, elem_conn_slave)
            elif elem_conn_slave.shape[1] in {8, 20, 27}:
                facet_map_slave = map_surface_facets_to_hex_elements(surface_slave, elem_conn_slave)
            else:
                raise NotImplementedError("elem_conn_slave must be tet4/tet10/hex8/hex20/hex27")
        obj = cls(
            surface_master=surface_master,
            surface_slave=surface_slave,
            supermesh_coords=sm.coords,
            supermesh_conn=sm.conn,
            source_facets_master=sm.source_facets_a,
            source_facets_slave=sm.source_facets_b,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            facet_to_elem_master=facet_map_master,
            facet_to_elem_slave=facet_map_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=None if facet_dofs_master is None else np.asarray(facet_dofs_master, dtype=int),
            facet_dofs_slave=None if facet_dofs_slave is None else np.asarray(facet_dofs_slave, dtype=int),
            trial_value_dim_master=None if trial_value_dim_master is None else int(trial_value_dim_master),
            trial_value_dim_slave=None if trial_value_dim_slave is None else int(trial_value_dim_slave),
            trial_space_mode_master=None if trial_space_mode_master is None else str(trial_space_mode_master),
            trial_space_mode_slave=None if trial_space_mode_slave is None else str(trial_space_mode_slave),
            trial_facet_dofs_master=None if trial_facet_dofs_master is None else np.asarray(trial_facet_dofs_master, dtype=int),
            trial_facet_dofs_slave=None if trial_facet_dofs_slave is None else np.asarray(trial_facet_dofs_slave, dtype=int),
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            supermesh_quad_cache=build_supermesh_triangle_quadrature_cache(
                sm.coords,
                sm.conn,
                quad_order=int(quad_order),
                tol=float(tol),
            ),
        )
        if setup_cache_enabled:
            _CONTACT_SETUP_CACHE[key] = obj
            if setup_cache_trace:
                print(
                    f"[contact] setup cache store n_tris={int(obj.supermesh_conn.shape[0])}",
                    flush=True,
                )
        return obj

    @classmethod
    def from_facets(
        cls,
        coords: np.ndarray,
        facets: np.ndarray,
        *,
        elem_conn: np.ndarray | None = None,
        value_dim: int = 1,
        space_mode: str = "nodal",
        facet_dofs: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        surface = SurfaceMesh.from_facets(coords, facets)
        return cls.from_surfaces(
            surface,
            surface,
            elem_conn_master=elem_conn,
            elem_conn_slave=elem_conn,
            value_dim_master=value_dim,
            value_dim_slave=value_dim,
            space_mode_master=space_mode,
            space_mode_slave=space_mode,
            facet_dofs_master=facet_dofs,
            facet_dofs_slave=facet_dofs,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod
    def from_surfaces_and_spaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slave: SurfaceMesh,
        space_master,
        space_slave,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int | None = None,
        value_dim_slave: int | None = None,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> "ContactSurfaceSpace":
        if value_dim_master is None:
            value_dim_master = int(getattr(space_master, "value_dim", 1))
        if value_dim_slave is None:
            value_dim_slave = int(getattr(space_slave, "value_dim", 1))
        return cls.from_surfaces(
            surface_master,
            surface_slave,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
        )

    @classmethod
    def from_sides(
        cls,
        master: ContactSide,
        slave: ContactSide,
        *,
        field_master: str = "a",
        field_slave: str = "b",
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        return cls.from_surfaces(
            master.surface,
            slave.surface,
            elem_conn_master=master.elem_conn,
            elem_conn_slave=slave.elem_conn,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=master.value_dim,
            value_dim_slave=slave.value_dim,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod  # type: ignore[no-redef]
    def from_facets(
        cls,
        coords_master: np.ndarray,
        facets_master: np.ndarray,
        coords_slave: np.ndarray,
        facets_slave: np.ndarray,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        surface_master = SurfaceMesh.from_facets(coords_master, facets_master)
        surface_slave = SurfaceMesh.from_facets(coords_slave, facets_slave)
        return cls.from_surfaces(
            surface_master,
            surface_slave,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    def _split_fields(self, u: Mapping[str, np.ndarray] | Sequence[np.ndarray]):
        if isinstance(u, Mapping):
            return u[self.field_master], u[self.field_slave]
        if len(u) != 2:
            raise ValueError("u must be a mapping or a length-2 sequence")
        return u[0], u[1]

    def _auto_normal_sign(self) -> float:
        if not hasattr(self.surface_master, "facet_normals"):
            return 1.0
        normals = self.surface_master.facet_normals()
        coords = np.asarray(self.surface_master.coords)
        coords_slave = np.asarray(self.surface_slave.coords)
        facets_m = np.asarray(self.surface_master.conn, dtype=int)
        facets_s = np.asarray(self.surface_slave.conn, dtype=int)
        dots = []
        for fa, fb in zip(self.source_facets_master, self.source_facets_slave):
            n = normals[int(fa)]
            cm = np.mean(coords[facets_m[int(fa)]], axis=0)
            cs = np.mean(coords_slave[facets_s[int(fb)]], axis=0)
            dots.append(float(np.dot(n, cs - cm)))
        if not dots:
            return 1.0
        return 1.0 if np.sum(dots) >= 0.0 else -1.0

    def _resolve_backend(self, backend: str | None) -> str:
        use_backend = self.backend if backend is None else backend
        if use_backend not in {"jax", "numpy"}:
            raise ValueError("backend must be 'jax' or 'numpy'")
        return use_backend

    def _trial_layout(self, *, side: str) -> tuple[int, str, np.ndarray | None]:
        if side == "master":
            value_dim = int(self.trial_value_dim_master or self.value_dim_master)
            space_mode = str(self.trial_space_mode_master or self.space_mode_master)
            facet_dofs = self.trial_facet_dofs_master if self.trial_facet_dofs_master is not None else self.facet_dofs_master
            return value_dim, space_mode, facet_dofs
        if side == "slave":
            value_dim = int(self.trial_value_dim_slave or self.value_dim_slave)
            space_mode = str(self.trial_space_mode_slave or self.space_mode_slave)
            facet_dofs = self.trial_facet_dofs_slave if self.trial_facet_dofs_slave is not None else self.facet_dofs_slave
            return value_dim, space_mode, facet_dofs
        raise ValueError("side must be 'master' or 'slave'")

    def _validate_square_trial_layout(self) -> None:
        test_master = _contact_space_side_n_dofs(self, side="master", role="test")
        test_slave = _contact_space_side_n_dofs(self, side="slave", role="test")
        trial_master = _contact_space_side_n_dofs(self, side="master", role="trial")
        trial_slave = _contact_space_side_n_dofs(self, side="slave", role="trial")
        if test_master != trial_master or test_slave != trial_slave:
            raise NotImplementedError(
                "Distinct contact trial layouts currently require the same total DOF counts as the test layouts. "
                "Rectangular contact operators are not enabled yet."
            )

    def initialize_state(self, *, metadata: Mapping[str, Any] | None = None) -> ContactState:
        return ContactState(
            interface_kind="pair",
            geometry="reference",
            iteration=0,
            active_set=None,
            field_summary={
                self.field_master: (_contact_space_side_n_dofs(self, side="master", role="trial"),),
                self.field_slave: (_contact_space_side_n_dofs(self, side="slave", role="trial"),),
            },
            metadata=dict(metadata or {}),
        )

    def update_state(
        self,
        *,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        contact_state: ContactState | None = None,
        geometry: str = "current",
        active_set: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContactState:
        base = self.initialize_state() if contact_state is None else contact_state
        merged_metadata = dict(base.metadata)
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        return replace(
            base,
            geometry=str(geometry),
            iteration=int(base.iteration) + 1,
            active_set=active_set if active_set is not None else base.active_set,
            field_summary=_summarize_contact_field_state(state),
            metadata=merged_metadata,
        )

    def assemble_contact_coupling_matrices(self) -> tuple["ContactCouplingMatrix", "ContactCouplingMatrix"]:
        """Return (M_aa, M_ab) coupling matrices on this contact interface."""
        return _assemble_contact_coupling_matrices(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            tol=self.tol,
            quad_order=self.quad_order,
        )

    def assemble_contact_kkt(
        self,
        *,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        format: str = "fluxsparse",
        return_blocks: bool = False,
    ):
        m_aa, m_ab = self.assemble_contact_coupling_matrices()
        return assemble_contact_kkt(
            m_aa,
            m_ab,
            rho=rho,
            multiplier=multiplier,
            facet_conn_master=np.asarray(self.surface_master.conn, dtype=int),
            backend=backend,
            format=format,
            return_blocks=return_blocks,
        )

    def assemble_contact_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_constraint_operators(
            self,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Legacy alias for assemble_multiplier()."""
        _warn_contact_legacy_name("PreparedContactInterface.assemble_constraint_operators()", "PreparedContactInterface.assemble_multiplier()")
        return self.assemble_contact_constraint_operators(
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_multiplier(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Preferred public alias for assemble_contact_constraint_operators()."""
        return self.assemble_contact_constraint_operators(
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_penalty_operators(
            self,
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Legacy alias for assemble_penalty()."""
        _warn_contact_legacy_name("PreparedContactInterface.assemble_penalty_operators()", "PreparedContactInterface.assemble_penalty()")
        return self.assemble_contact_penalty_operators(
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_penalty(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Preferred public alias for assemble_contact_penalty_operators()."""
        return self.assemble_contact_penalty_operators(
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_operators(
        self,
        *,
        enforcement: str | None = None,
        method: str | None = None,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Unified public alias that routes to penalty or constraint assembly."""
        return assemble_contact_operators(
            self,
            enforcement=enforcement,
            method=method,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_residual(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
        params: "WeakParams",
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
    ) -> np.ndarray:
        self._validate_square_trial_layout()
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        trial_value_dim_master, trial_space_mode_master, trial_facet_dofs_master = self._trial_layout(side="master")
        trial_value_dim_slave, trial_space_mode_slave, trial_facet_dofs_slave = self._trial_layout(side="slave")
        return _assemble_contact_interface_residual(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            res_form,
            u_master,
            u_slave,
            params,
            value_dim_a=self.value_dim_master,
            value_dim_b=self.value_dim_slave,
            trial_value_dim_a=trial_value_dim_master,
            trial_value_dim_b=trial_value_dim_slave,
            space_mode_a=self.space_mode_master,
            space_mode_b=self.space_mode_slave,
            trial_space_mode_a=trial_space_mode_master,
            trial_space_mode_b=trial_space_mode_slave,
            facet_dofs_a=self.facet_dofs_master,
            facet_dofs_b=self.facet_dofs_slave,
            trial_facet_dofs_a=trial_facet_dofs_master,
            trial_facet_dofs_b=trial_facet_dofs_slave,
            field_a=self.field_master,
            field_b=self.field_slave,
            elem_conn_a=self.elem_conn_master,
            elem_conn_b=self.elem_conn_slave,
            facet_to_elem_a=self.facet_to_elem_master,
            facet_to_elem_b=self.facet_to_elem_slave,
            normal_source=normal_source,
            normal_from="master",
            master_field=self.field_master,
            normal_sign=normal_sign,
            grad_source="volume",
            dof_source="volume",
            quad_order=self.quad_order,
            tol=self.tol,
        )

    def assemble_jacobian(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
        params: "WeakParams",
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
        sparse: bool = True,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> ContactJacobianReturn:
        self._validate_square_trial_layout()
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        use_backend = self._resolve_backend(backend)
        use_batch_jac = self.batch_jac if batch_jac is None else batch_jac
        trial_value_dim_master, trial_space_mode_master, trial_facet_dofs_master = self._trial_layout(side="master")
        trial_value_dim_slave, trial_space_mode_slave, trial_facet_dofs_slave = self._trial_layout(side="slave")
        return _assemble_contact_interface_jacobian(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            res_form,
            u_master,
            u_slave,
            params,
            value_dim_a=self.value_dim_master,
            value_dim_b=self.value_dim_slave,
            trial_value_dim_a=trial_value_dim_master,
            trial_value_dim_b=trial_value_dim_slave,
            space_mode_a=self.space_mode_master,
            space_mode_b=self.space_mode_slave,
            trial_space_mode_a=trial_space_mode_master,
            trial_space_mode_b=trial_space_mode_slave,
            facet_dofs_a=self.facet_dofs_master,
            facet_dofs_b=self.facet_dofs_slave,
            trial_facet_dofs_a=trial_facet_dofs_master,
            trial_facet_dofs_b=trial_facet_dofs_slave,
            field_a=self.field_master,
            field_b=self.field_slave,
            elem_conn_a=self.elem_conn_master,
            elem_conn_b=self.elem_conn_slave,
            facet_to_elem_a=self.facet_to_elem_master,
            facet_to_elem_b=self.facet_to_elem_slave,
            normal_source=normal_source,
            normal_from="master",
            master_field=self.field_master,
            normal_sign=normal_sign,
            grad_source="volume",
            dof_source="volume",
            quad_order=self.quad_order,
            tol=self.tol,
            sparse=sparse,
            backend=use_backend,
            batch_jac=use_batch_jac,
            supermesh_quad_cache=self.supermesh_quad_cache,
        )

    def compile_bilinear(
        self,
        bilin: ContactBilinearLike,
        *,
        backend: str | None = None,
        use_cache: bool = True,
    ) -> MixedSurfaceResidualForm:
        """Compile a contact bilinear callable to a reusable mixed-surface residual form."""
        if _is_compiled_contact_bilinear(bilin):
            return cast(MixedSurfaceResidualForm, bilin)
        use_backend = self._resolve_backend(backend)
        cache_key = (id(bilin), use_backend)
        if use_cache:
            cached = self._compiled_bilinear_cache.get(cache_key)
            if cached is not None:
                return cached
        res_form = _compile_contact_bilinear(
            bilin,
            field_master=self.field_master,
            field_slave=self.field_slave,
            backend=use_backend,
        )
        if use_cache:
            self._compiled_bilinear_cache[cache_key] = res_form
        return res_form

    def assemble_bilinear(
        self,
        bilin: ContactBilinearLike,
        u_master: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | npt.ArrayLike,
        u_slave: npt.ArrayLike | None = None,
        params: "WeakParams" | None = None,
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        """
        Assemble a mixed surface bilinear form with signature (v1, v2, u1, u2, params).

        Notes:
        - v1/v2/u1/u2 are symbolic field refs; use .val/.grad/.sym_grad in the expression.
        - The bilinear must be linear in v1 and v2 and include ds() in its expression.
        - When building dot products, prefer dot(v1, ...) and dot(v2, ...) to keep shapes consistent.
        - Normal orientation, grad_source, and dof_source are fixed internally for simplicity.
        - u_master/u_slave can be passed as a single mapping/length-2 sequence; in that case,
          pass params as the next positional arg or a keyword.
        """
        def _is_field_pair(obj) -> bool:
            if isinstance(obj, Mapping):
                return True
            return isinstance(obj, Sequence) and not hasattr(obj, "shape")

        if params is None:
            if u_slave is None:
                raise TypeError("params is required")
            if _is_field_pair(u_master):
                params = u_slave
                u_master, u_slave = self._split_fields(u_master)
            else:
                raise TypeError("params is required")
        elif u_slave is None:
            u_master, u_slave = self._split_fields(u_master)

        use_backend = self._resolve_backend(None)
        res_form = self.compile_bilinear(bilin, backend=use_backend)
        return self.assemble_jacobian(
            res_form,
            {self.field_master: u_master, self.field_slave: u_slave},
            params,
            normal_sign=None,
            normal_source=normal_source,
            sparse=sparse,
            backend=use_backend,
        )

    def assemble_bilinear_form(
        self,
        bilin: ContactBilinearLike,
        params: "WeakParams",
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        """Assemble an interface bilinear form without requiring a state vector."""
        n_master = _contact_space_side_n_dofs(self, side="master", role="trial")
        n_slave = _contact_space_side_n_dofs(self, side="slave", role="trial")
        u_master = np.zeros((n_master,), dtype=float)
        u_slave = np.zeros((n_slave,), dtype=float)
        return self.assemble_bilinear(
            bilin,
            u_master,
            u_slave,
            params,
            sparse=sparse,
            normal_source=normal_source,
        )


def _field_n_dofs(
    *,
    n_nodes: int,
    n_facets: int,
    value_dim: int,
    space_mode: str,
    facet_dofs: np.ndarray | None,
) -> int:
    if space_mode == "p0":
        if facet_dofs is not None:
            arr = np.asarray(facet_dofs, dtype=int)
            if arr.size == 0:
                return 0
            if np.any(arr < 0):
                raise ValueError("facet_dofs must be non-negative.")
            return int(arr.max()) + 1
        return int(n_facets) * int(value_dim)
    return int(n_nodes) * int(value_dim)


def _contact_space_side_n_dofs(space: "ContactSurfaceSpace", *, side: str, role: str = "test") -> int:
    if role not in {"test", "trial"}:
        raise ValueError("role must be 'test' or 'trial'")
    if side == "master":
        if role == "trial":
            value_dim, space_mode, facet_dofs = space._trial_layout(side="master")
        else:
            value_dim, space_mode, facet_dofs = int(space.value_dim_master), space.space_mode_master, space.facet_dofs_master
        return _field_n_dofs(
            n_nodes=int(np.asarray(space.surface_master.coords).shape[0]),
            n_facets=int(np.asarray(space.surface_master.conn).shape[0]),
            value_dim=int(value_dim),
            space_mode=space_mode,
            facet_dofs=facet_dofs,
        )
    if side == "slave":
        if role == "trial":
            value_dim, space_mode, facet_dofs = space._trial_layout(side="slave")
        else:
            value_dim, space_mode, facet_dofs = int(space.value_dim_slave), space.space_mode_slave, space.facet_dofs_slave
        return _field_n_dofs(
            n_nodes=int(np.asarray(space.surface_slave.coords).shape[0]),
            n_facets=int(np.asarray(space.surface_slave.conn).shape[0]),
            value_dim=int(value_dim),
            space_mode=space_mode,
            facet_dofs=facet_dofs,
        )
    raise ValueError("side must be 'master' or 'slave'")


@dataclass(eq=False)
class OneToManyContactSurfaceSpace:
    """One-master/multi-slave wrapper built from pairwise ContactSurfaceSpace objects."""

    contacts: tuple[ContactSurfaceSpace, ...]
    field_master: str = "master"
    field_slave: str = "slave"
    _compiled_bilinear_cache: dict[tuple[int, str], MixedSurfaceResidualForm] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_meshes(
        cls,
        master_mesh: BaseMesh,
        slave_meshes: Sequence[BaseMesh],
        *,
        master_facets: np.ndarray | None = None,
        slave_facets_list: Sequence[np.ndarray] | None = None,
        master_facet_selector: Callable[[BaseMesh], np.ndarray] | None = None,
        slave_facet_selectors: Sequence[Callable[[BaseMesh], np.ndarray] | None] | Callable[[BaseMesh], np.ndarray] | None = None,
        master_space: object | None = None,
        slave_spaces: Sequence[object | None] | object | None = None,
        value_dim_master: int | None = None,
        value_dim_slaves: Sequence[int | None] | int | None = None,
        mode_master: str = "touching",
        mode_slave: str = "touching",
        field_master: str = "master",
        field_slave: str = "slave",
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        if len(slave_meshes) == 0:
            raise ValueError("slave_meshes must contain at least one mesh.")
        n_slaves = len(slave_meshes)

        if master_facets is None:
            if master_facet_selector is None:
                raise ValueError("Provide either master_facets or master_facet_selector.")
            master_facets = np.asarray(master_facet_selector(master_mesh), dtype=int)
        else:
            master_facets = np.asarray(master_facets, dtype=int)

        if slave_facets_list is None:
            if slave_facet_selectors is None:
                raise ValueError("Provide either slave_facets_list or slave_facet_selectors.")
            if callable(slave_facet_selectors):
                slave_facets_list = [np.asarray(slave_facet_selectors(mesh), dtype=int) for mesh in slave_meshes]
            else:
                if len(slave_facet_selectors) != n_slaves:
                    raise ValueError("slave_facet_selectors length must match slave_meshes length.")
                out_facets: list[np.ndarray] = []
                for mesh, sel in zip(slave_meshes, slave_facet_selectors):
                    if sel is None:
                        raise ValueError("slave_facet_selectors contains None; provide a selector for each slave.")
                    out_facets.append(np.asarray(sel(mesh), dtype=int))
                slave_facets_list = out_facets
        else:
            if len(slave_facets_list) != n_slaves:
                raise ValueError("slave_facets_list length must match slave_meshes length.")
            slave_facets_list = [np.asarray(facets, dtype=int) for facets in slave_facets_list]

        if slave_spaces is None:
            slave_spaces = [None] * n_slaves
        elif isinstance(slave_spaces, Sequence) and not isinstance(slave_spaces, (str, bytes)):
            if len(slave_spaces) != n_slaves:
                raise ValueError("slave_spaces length must match slave_meshes length.")
            slave_spaces = list(slave_spaces)
        else:
            slave_spaces = [slave_spaces] * n_slaves

        if value_dim_slaves is None:
            value_dim_slaves = [None] * n_slaves
        elif isinstance(value_dim_slaves, Sequence) and not isinstance(value_dim_slaves, (str, bytes)):
            if len(value_dim_slaves) != n_slaves:
                raise ValueError("value_dim_slaves length must match slave_meshes length.")
            value_dim_slaves = list(value_dim_slaves)
        else:
            value_dim_slaves = [int(value_dim_slaves)] * n_slaves

        master_side = ContactSide.from_facets(
            master_mesh,
            master_facets,
            master_space,
            value_dim=value_dim_master,
            mode=mode_master,
        )
        slave_sides = [
            ContactSide.from_facets(
                mesh,
                np.asarray(facets, dtype=int),
                space,
                value_dim=value_dim,
                mode=mode_slave,
            )
            for mesh, facets, space, value_dim in zip(slave_meshes, slave_facets_list, slave_spaces, value_dim_slaves)
        ]
        return cls.from_sides(
            master_side,
            slave_sides,
            field_master=field_master,
            field_slave=field_slave,
            quad_order=quad_order,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod
    def from_sides(
        cls,
        master: ContactSide,
        slaves: Sequence[ContactSide],
        *,
        field_master: str = "master",
        field_slave: str = "slave",
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        if len(slaves) == 0:
            raise ValueError("slaves must contain at least one ContactSide.")
        contacts = tuple(
            ContactSurfaceSpace.from_sides(
                master,
                slave,
                field_master=field_master,
                field_slave=field_slave,
                quad_order=quad_order,
                space_mode_master=space_mode_master,
                space_mode_slave=space_mode_slave,
                facet_dofs_master=facet_dofs_master,
                facet_dofs_slave=facet_dofs_slave,
                normal_sign=normal_sign,
                tol=tol,
                backend=backend,
                batch_jac=batch_jac,
                setup_cache_enabled=setup_cache_enabled,
                setup_cache_trace=setup_cache_trace,
            )
            for slave in slaves
        )
        return cls(contacts=contacts, field_master=field_master, field_slave=field_slave)

    @classmethod
    def from_surfaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slaves: Sequence[SurfaceMesh],
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slaves: Sequence[np.ndarray | None] | None = None,
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        field_master: str = "master",
        field_slave: str = "slave",
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        if len(surface_slaves) == 0:
            raise ValueError("surface_slaves must contain at least one surface.")
        if elem_conn_slaves is None:
            elem_conn_slaves = [None] * len(surface_slaves)
        if len(elem_conn_slaves) != len(surface_slaves):
            raise ValueError("elem_conn_slaves length must match surface_slaves length.")
        contacts = tuple(
            ContactSurfaceSpace.from_surfaces(
                surface_master,
                surface_slave,
                elem_conn_master=elem_conn_master,
                elem_conn_slave=elem_conn_slave,
                field_master=field_master,
                field_slave=field_slave,
                value_dim_master=value_dim_master,
                value_dim_slave=value_dim_slave,
                space_mode_master=space_mode_master,
                space_mode_slave=space_mode_slave,
                facet_dofs_master=facet_dofs_master,
                facet_dofs_slave=facet_dofs_slave,
                quad_order=quad_order,
                normal_sign=normal_sign,
                tol=tol,
                backend=backend,
                batch_jac=batch_jac,
                setup_cache_enabled=setup_cache_enabled,
                setup_cache_trace=setup_cache_trace,
            )
            for surface_slave, elem_conn_slave in zip(surface_slaves, elem_conn_slaves)
        )
        return cls(contacts=contacts, field_master=field_master, field_slave=field_slave)

    def _split_fields(
        self, u: Mapping[str, npt.ArrayLike] | Sequence[Any]
    ) -> tuple[npt.ArrayLike, list[npt.ArrayLike]]:
        if isinstance(u, Mapping):
            if self.field_master not in u:
                raise KeyError(f"u mapping must contain master field '{self.field_master}'.")
            if "slaves" not in u:
                raise KeyError("u mapping must contain key 'slaves' with per-slave states.")
            u_master = u[self.field_master]
            u_slaves = list(u["slaves"])
        else:
            if len(u) != 2:
                raise ValueError("u must be a mapping or a sequence like (u_master, u_slaves).")
            u_master = u[0]
            u_slaves = list(u[1])
        if len(u_slaves) != len(self.contacts):
            raise ValueError(
                f"u_slaves length mismatch: got {len(u_slaves)}, expected {len(self.contacts)}."
            )
        return u_master, u_slaves

    def _dof_layout(self) -> tuple[int, list[int], int]:
        if len(self.contacts) == 0:
            return 0, [], 0
        n_master = _contact_space_side_n_dofs(self.contacts[0], side="master")
        slave_sizes = [_contact_space_side_n_dofs(contact, side="slave") for contact in self.contacts]
        total = int(n_master + sum(slave_sizes))
        return n_master, slave_sizes, total

    def initialize_state(self, *, metadata: Mapping[str, Any] | None = None) -> ContactState:
        n_master, slave_sizes, _ = self._dof_layout()
        return ContactState(
            interface_kind="one_to_many",
            geometry="reference",
            iteration=0,
            active_set=None,
            field_summary={
                self.field_master: (n_master,),
                self.field_slave: tuple(int(n) for n in slave_sizes),
            },
            metadata=dict(metadata or {}),
        )

    def update_state(
        self,
        *,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        contact_state: ContactState | None = None,
        geometry: str = "current",
        active_set: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContactState:
        base = self.initialize_state() if contact_state is None else contact_state
        merged_metadata = dict(base.metadata)
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        return replace(
            base,
            geometry=str(geometry),
            iteration=int(base.iteration) + 1,
            active_set=active_set if active_set is not None else base.active_set,
            field_summary=_summarize_contact_field_state(state),
            metadata=merged_metadata,
        )

    def _resolve_backend(self, backend: str | None) -> str:
        if backend is not None:
            return str(backend)
        if len(self.contacts) == 0:
            return "jax"
        return str(self.contacts[0].backend)

    def compile_bilinear(
        self,
        bilin: ContactBilinearLike,
        *,
        backend: str | None = None,
        use_cache: bool = True,
    ) -> MixedSurfaceResidualForm:
        """Compile a one-to-many contact bilinear once and reuse it across all pair contacts."""
        use_backend = self._resolve_backend(backend)
        if _is_compiled_contact_bilinear(bilin):
            return cast(MixedSurfaceResidualForm, bilin)
        cache_key = (id(bilin), use_backend)
        if use_cache:
            cached = self._compiled_bilinear_cache.get(cache_key)
            if cached is not None:
                return cached
        res_form = _compile_contact_bilinear(
            bilin,
            field_master=self.field_master,
            field_slave=self.field_slave,
            backend=use_backend,
        )
        if use_cache:
            self._compiled_bilinear_cache[cache_key] = res_form
        return res_form

    @staticmethod
    def _scatter_pair_indices(local_idx: np.ndarray, *, n_master: int, slave_offset: int) -> np.ndarray:
        idx = np.asarray(local_idx, dtype=int)
        out = np.empty_like(idx)
        master_mask = idx < int(n_master)
        out[master_mask] = idx[master_mask]
        out[~master_mask] = int(n_master) + int(slave_offset) + (idx[~master_mask] - int(n_master))
        return out

    def assemble_residual(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any],
        params: "WeakParams",
        *,
        normal_source: str = "master",
    ) -> np.ndarray:
        u_master, u_slaves = self._split_fields(u)
        n_master, slave_sizes, n_total = self._dof_layout()
        R = np.zeros((n_total,), dtype=float)
        slave_offset = 0
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            r_local = np.asarray(
                contact.assemble_residual(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                ),
                dtype=float,
            )
            if r_local.shape[0] != n_master + n_slave:
                raise ValueError("Pair residual size mismatch while assembling one-to-many residual.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            R[idx] += r_local
            slave_offset += n_slave
        return R

    def assemble_jacobian(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any],
        params: "WeakParams",
        *,
        normal_source: str = "master",
        sparse: bool = True,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> ContactJacobianReturn:
        u_master, u_slaves = self._split_fields(u)
        n_master, slave_sizes, n_total = self._dof_layout()
        slave_offset = 0
        if sparse:
            from ..solver import FluxSparseMatrix

            rows_all: list[np.ndarray] = []
            cols_all: list[np.ndarray] = []
            data_all: list[np.ndarray] = []
            for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
                j_local = contact.assemble_jacobian(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                    sparse=True,
                    backend=backend,
                    batch_jac=batch_jac,
                )
                rows, cols, data, n_pair = _contact_sparse_to_coo(j_local)
                if n_pair != n_master + n_slave:
                    raise ValueError("Pair Jacobian size mismatch while assembling sparse one-to-many Jacobian.")
                rows_all.append(
                    self._scatter_pair_indices(rows, n_master=n_master, slave_offset=slave_offset)
                )
                cols_all.append(
                    self._scatter_pair_indices(cols, n_master=n_master, slave_offset=slave_offset)
                )
                data_all.append(data)
                slave_offset += n_slave
            if rows_all:
                rows_out = np.concatenate(rows_all)
                cols_out = np.concatenate(cols_all)
                data_out = np.concatenate(data_all)
            else:
                rows_out = np.zeros((0,), dtype=int)
                cols_out = np.zeros((0,), dtype=int)
                data_out = np.zeros((0,), dtype=float)
            return FluxSparseMatrix(rows_out, cols_out, data_out, n_dofs=n_total)

        K = np.zeros((n_total, n_total), dtype=float)
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            j_local = np.asarray(
                contact.assemble_jacobian(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                    sparse=False,
                    backend=backend,
                    batch_jac=batch_jac,
                ),
                dtype=float,
            )
            if j_local.shape != (n_master + n_slave, n_master + n_slave):
                raise ValueError("Pair Jacobian shape mismatch while assembling dense one-to-many Jacobian.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            K[np.ix_(idx, idx)] += j_local
            slave_offset += n_slave
        return K

    def assemble_bilinear(
        self,
        bilin: ContactBilinearLike,
        u_master: Mapping[str, npt.ArrayLike] | Sequence[Any] | npt.ArrayLike,
        u_slaves: Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        if params is None:
            if u_slaves is None:
                raise TypeError("params is required")
            if isinstance(u_master, Mapping) or (isinstance(u_master, Sequence) and not hasattr(u_master, "shape")):
                params = u_slaves  # type: ignore[assignment]
                u_master, u_slaves = self._split_fields(u_master)  # type: ignore[arg-type]
            else:
                raise TypeError("params is required")
        elif u_slaves is None:
            u_master, u_slaves = self._split_fields(u_master)  # type: ignore[arg-type]
        assert params is not None
        assert u_slaves is not None
        res_form = self.compile_bilinear(bilin)

        n_master, slave_sizes, n_total = self._dof_layout()
        slave_offset = 0
        if sparse:
            from ..solver import FluxSparseMatrix

            rows_all: list[np.ndarray] = []
            cols_all: list[np.ndarray] = []
            data_all: list[np.ndarray] = []
            for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
                j_local = contact.assemble_bilinear(
                    res_form,
                    u_master,
                    u_slave,
                    params,
                    sparse=True,
                    normal_source=normal_source,
                )
                rows, cols, data, n_pair = _contact_sparse_to_coo(j_local)
                if n_pair != n_master + n_slave:
                    raise ValueError("Pair Jacobian size mismatch while assembling sparse one-to-many bilinear.")
                rows_all.append(
                    self._scatter_pair_indices(rows, n_master=n_master, slave_offset=slave_offset)
                )
                cols_all.append(
                    self._scatter_pair_indices(cols, n_master=n_master, slave_offset=slave_offset)
                )
                data_all.append(data)
                slave_offset += n_slave
            if rows_all:
                rows_out = np.concatenate(rows_all)
                cols_out = np.concatenate(cols_all)
                data_out = np.concatenate(data_all)
            else:
                rows_out = np.zeros((0,), dtype=int)
                cols_out = np.zeros((0,), dtype=int)
                data_out = np.zeros((0,), dtype=float)
            return FluxSparseMatrix(rows_out, cols_out, data_out, n_dofs=n_total)

        K = np.zeros((n_total, n_total), dtype=float)
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            j_local = np.asarray(
                contact.assemble_bilinear(
                    res_form,
                    u_master,
                    u_slave,
                    params,
                    sparse=False,
                    normal_source=normal_source,
                ),
                dtype=float,
            )
            if j_local.shape != (n_master + n_slave, n_master + n_slave):
                raise ValueError("Pair Jacobian shape mismatch while assembling dense one-to-many bilinear.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            K[np.ix_(idx, idx)] += j_local
            slave_offset += n_slave
        return K

    def assemble_bilinear_form(
        self,
        bilin: ContactBilinearLike,
        params: "WeakParams",
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        """Assemble a one-to-many interface bilinear form without requiring states."""
        n_master, slave_sizes, _ = self._dof_layout()
        u_master = np.zeros((n_master,), dtype=float)
        u_slaves = [np.zeros((n_slave,), dtype=float) for n_slave in slave_sizes]
        return self.assemble_bilinear(
            bilin,
            u_master,
            u_slaves,
            params,
            sparse=sparse,
            normal_source=normal_source,
        )

    def assemble_contact_coupling_matrices(self):
        from .contact_interface import ContactCouplingMatrix

        n_master, slave_sizes, _ = self._dof_layout()
        n_slaves_total = int(sum(slave_sizes))
        rows_mm: list[np.ndarray] = []
        cols_mm: list[np.ndarray] = []
        data_mm: list[np.ndarray] = []
        rows_ms: list[np.ndarray] = []
        cols_ms: list[np.ndarray] = []
        data_ms: list[np.ndarray] = []

        slave_offset = 0
        for contact, n_slave in zip(self.contacts, slave_sizes):
            m_mm, m_ms_local = contact.assemble_contact_coupling_matrices()
            rows_mm.append(np.asarray(m_mm.rows, dtype=int))
            cols_mm.append(np.asarray(m_mm.cols, dtype=int))
            data_mm.append(np.asarray(m_mm.data, dtype=float))
            rows_ms.append(np.asarray(m_ms_local.rows, dtype=int))
            cols_ms.append(np.asarray(m_ms_local.cols, dtype=int) + slave_offset)
            data_ms.append(np.asarray(m_ms_local.data, dtype=float))
            if m_mm.shape != (n_master, n_master):
                raise ValueError("Pair M_aa shape mismatch while assembling one-to-many coupling matrices.")
            if m_ms_local.shape != (n_master, n_slave):
                raise ValueError("Pair M_ab shape mismatch while assembling one-to-many coupling matrices.")
            slave_offset += n_slave

        mm = ContactCouplingMatrix(
            rows=np.concatenate(rows_mm) if rows_mm else np.zeros((0,), dtype=int),
            cols=np.concatenate(cols_mm) if cols_mm else np.zeros((0,), dtype=int),
            data=np.concatenate(data_mm) if data_mm else np.zeros((0,), dtype=float),
            shape=(n_master, n_master),
        )
        ms = ContactCouplingMatrix(
            rows=np.concatenate(rows_ms) if rows_ms else np.zeros((0,), dtype=int),
            cols=np.concatenate(cols_ms) if cols_ms else np.zeros((0,), dtype=int),
            data=np.concatenate(data_ms) if data_ms else np.zeros((0,), dtype=float),
            shape=(n_master, n_slaves_total),
        )
        return mm, ms

    def assemble_contact_kkt(
        self,
        *,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        format: str = "fluxsparse",
        return_blocks: bool = False,
    ):
        m_aa, m_ab = self.assemble_contact_coupling_matrices()
        master_facets = np.asarray(self.contacts[0].surface_master.conn, dtype=int)
        return assemble_contact_kkt(
            m_aa,
            m_ab,
            rho=rho,
            multiplier=multiplier,
            facet_conn_master=master_facets,
            backend=backend,
            format=format,
            return_blocks=return_blocks,
        )

    def assemble_contact_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_constraint_operators(
            self,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Alias for assemble_contact_constraint_operators()."""
        return self.assemble_contact_constraint_operators(
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_penalty_operators(
            self,
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Alias for assemble_contact_penalty_operators()."""
        return self.assemble_contact_penalty_operators(
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_operators(
        self,
        *,
        enforcement: str | None = None,
        method: str | None = None,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Unified public alias that routes to penalty or constraint assembly."""
        return assemble_contact_operators(
            self,
            enforcement=enforcement,
            method=method,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )


def assemble_contact_operators(
    contact,
    *,
    enforcement: str | None = None,
    method: str | None = None,
    law: str | None = None,
    formulation: str | None = None,
    rho: float = 0.0,
    multiplier: ContactMultiplierSpace | None = None,
    backend: str | None = None,
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    """Unified public contact assembly entry that routes to penalty or constraint operators."""
    resolved = _resolve_contact_operator_enforcement(
        enforcement=enforcement,
        method=method,
        formulation=formulation,
        multiplier=multiplier,
    )
    if resolved == "penalty":
        use_backend = "jax" if backend is None else backend
        return assemble_contact_penalty_operators(
            contact,
            law=law,
            formulation=formulation,
            backend=use_backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )
    use_backend = "numpy" if backend is None else backend
    return assemble_contact_constraint_operators(
        contact,
        law=law,
        formulation=formulation,
        rho=rho,
        multiplier=multiplier,
        backend=use_backend,
        weak_form=weak_form,
        state=state,
        res_form=res_form,
        u=u,
        params=params,
        normal_source=normal_source,
        sparse=sparse,
        batch_jac=batch_jac,
    )


def assemble_multiplier(contact, **kwargs):
    """Public alias for assemble_contact_constraint_operators()."""
    return assemble_contact_constraint_operators(contact, **kwargs)


def assemble_penalty(contact, **kwargs):
    """Public alias for assemble_contact_penalty_operators()."""
    return assemble_contact_penalty_operators(contact, **kwargs)


__all__ = [
    "ContactSideSpec",
    "ContactSide",
    "OneSidedContact",
    "PreparedOneSidedContactInterface",
    "OneSidedContactSurfaceSpace",
    "PreparedContactInterface",
    "ContactSurfaceSpace",
    "PreparedOneToManyContactInterface",
    "OneToManyContactSurfaceSpace",
    "ContactOperators",
    "MultiplierContactContribution",
    "PenaltyContactContribution",
    "ContactState",
    "AugmentedLagrangianState",
    "AugmentedLagrangianResult",
    "MultiplierSpec",
    "ContactMultiplierSpace",
    "coarse_p1_basis_from_node_groups",
    "coarse_p1_basis_from_surface_grid",
    "ContactPairSpec",
    "ContactGroupSpec",
    "OneSidedContactSpec",
    "ContactKKTSolveConfig",
    "EmbeddingMap",
    "build_nodal_embedding_map",
    "build_barycentric_embedding_map",
    "build_barycentric_embedding_map_from_meshes",
    "assemble_embedding_constraint_matrix",
    "assemble_rbe2_constraint_matrix",
    "assemble_rbe3_constraint_matrix",
    "build_rbe3_weights",
    "assemble_contact_constraint_operators",
    "assemble_multiplier",
    "assemble_contact_operators",
    "assemble_contact_penalty_operators",
    "assemble_penalty",
    "assemble_contact_interface_residual",
    "assemble_contact_interface_jacobian",
    "assemble_contact_coupling_matrices",
    "assemble_contact_kkt",
    "solve_contact_kkt",
    "solve_augmented_lagrangian_outer_loop",
    "facet_gap_values",
    "active_contact_facets",
]

# Phase-1 public naming aliases. These remain thin wrappers over the existing
# contact implementation until the state-explicit redesign is introduced.
ContactSideSpec = ContactSide
PreparedContactInterface = ContactSurfaceSpace
PreparedOneToManyContactInterface = OneToManyContactSurfaceSpace
PreparedOneSidedContactInterface = OneSidedContactSurfaceSpace
ContactPairSpec = ContactSpaces
ContactGroupSpec = ContactGroupSpaces
OneSidedContactSpec = OneSidedContactSpaces
MultiplierSpec = ContactMultiplierSpace

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence, TypeAlias, Union, cast

import numpy as np
import numpy.typing as npt

from .contact_diagnostics import (
    ContactConstraintDiagnostics,
    ContactConstraintQualityReport,
    assess_contact_constraint_quality,
    contact_constraint_matrix_diagnostics,
)
from .mortar_problem import dense_contact_operator_matrix as _dense_contact_operator_matrix

if TYPE_CHECKING:
    from .contact_interface import SurfaceMixedFormContext
    from ..solver import FluxSparseMatrix, FluxSparseOperator


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
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _flatten_contact_state_vector(state: Any, *, backend: str | None = None):
    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np
    if isinstance(state, Mapping):
        parts = [xp.ravel(xp.asarray(value)) for value in state.values()]
        if not parts:
            return xp.zeros((0,), dtype=float)
        return xp.concatenate(parts)
    if isinstance(state, Sequence) and not hasattr(state, "shape"):
        parts = [xp.ravel(xp.asarray(value)) for value in state]
        if not parts:
            return xp.zeros((0,), dtype=float)
        return xp.concatenate(parts)
    return xp.ravel(xp.asarray(state))


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

    def contact_energy(self, state: Any, *, matrix: Any | None = None, backend: str | None = None):
        """
        Return ``0.5 * u.T @ K_contact @ u`` for assembled penalty/Nitsche terms.

        This is a diagnostic quantity; for nonlinear contact it is the quadratic
        energy of the assembled tangent at the supplied state.
        """
        use_backend = (
            backend
            if backend is not None
            else _infer_contact_backend(state, matrix, self.jacobian, default="numpy")
        )
        K = _dense_contact_operator_matrix(
            self.jacobian if matrix is None else matrix,
            backend=use_backend,
        )
        if K is None:
            raise ValueError("contact_energy requires an assembled jacobian or explicit matrix.")
        u = _flatten_contact_state_vector(state, backend=use_backend)
        return 0.5 * (u @ (K @ u))

    def penalty_energy(self, state: Any, *, matrix: Any | None = None, backend: str | None = None):
        """Alias for ``contact_energy`` used by penalty/Nitsche diagnostics."""
        return self.contact_energy(state, matrix=matrix, backend=backend)


@dataclass(frozen=True)
class MultiplierContactContribution(ContactOperators):
    """Explicit multiplier-family contact contribution."""

    def constraint_diagnostics(
        self,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-14,
        max_singular_values: int | None = 20,
    ) -> ContactConstraintDiagnostics:
        """Return row norm, rank, and singular-value diagnostics for ``B``."""
        if self.B is None:
            raise ValueError("constraint_diagnostics requires an assembled B matrix.")
        return contact_constraint_matrix_diagnostics(
            self.B,
            rtol=rtol,
            atol=atol,
            max_singular_values=max_singular_values,
        )

    def constraint_quality(
        self,
        **kwargs,
    ) -> ContactConstraintQualityReport:
        """Evaluate opt-in quality thresholds for the assembled ``B`` matrix."""
        if self.B is None:
            raise ValueError("constraint_quality requires an assembled B matrix.")
        return assess_contact_constraint_quality(self.B, **kwargs)

    def constraint_residual(self, state: Any, *, backend: str | None = None):
        """Return the mortar compatibility residual ``B @ u``."""
        use_backend = (
            backend
            if backend is not None
            else _infer_contact_backend(state, self.B, default="numpy")
        )
        B = _dense_contact_operator_matrix(self.B, backend=use_backend)
        if B is None:
            raise ValueError("constraint_residual requires an assembled B matrix.")
        u = _flatten_contact_state_vector(state, backend=use_backend)
        return B @ u

    def constraint_residual_norm(self, state: Any, *, backend: str | None = None):
        """Return ``||B @ u||_2`` for mortar compatibility diagnostics."""
        use_backend = (
            backend
            if backend is not None
            else _infer_contact_backend(state, self.B, default="numpy")
        )
        r = self.constraint_residual(state, backend=use_backend)
        if use_backend == "jax":
            import jax.numpy as jnp

            return jnp.linalg.norm(r)
        return float(np.linalg.norm(np.asarray(r, dtype=float)))

    def augmentation_energy(self, state: Any, *, rho: float | None = None, backend: str | None = None):
        """Return ``0.5 * rho * ||B @ u||_2**2`` for augmented mortar diagnostics."""
        rho_eff = self.rho if rho is None else rho
        if rho_eff is None:
            raise ValueError("augmentation_energy requires rho or self.rho.")
        use_backend = (
            backend
            if backend is not None
            else _infer_contact_backend(state, self.B, rho_eff, default="numpy")
        )
        r = self.constraint_residual(state, backend=use_backend)
        if use_backend == "jax":
            import jax.numpy as jnp

            return 0.5 * jnp.asarray(rho_eff) * (r @ r)
        r_np = np.asarray(r, dtype=float)
        return float(0.5 * float(rho_eff) * (r_np @ r_np))


@dataclass(frozen=True)
class ContactSolveResult:
    """Result of a contact solve with explicit solver/contact state."""

    state: Any
    contact_state: ContactState
    converged: bool
    iters: int
    residual_norm: float


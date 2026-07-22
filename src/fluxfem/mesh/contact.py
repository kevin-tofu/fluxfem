from __future__ import annotations

from typing import Any, Mapping, Sequence, TYPE_CHECKING
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
    assemble_contact_coupling_matrices as _assemble_contact_coupling_matrices,
)
from .mortar_problem import (
    MortarContactProblemPair,
    MortarContactProblem,
    assemble_mortar_contact_problem,
)
from .mortar_multiplier import (
    ContactMultiplierSpace,
    MultiplierSpec,
)
from .contact_api import (
    ContactSide,
    ContactSideSpec,
    ContactSpaces,
    ContactPairSpec,
    ContactGroupSpaces,
    ContactGroupSpec,
    OneSidedContactSpaces,
    OneSidedContactSpec,
)
from .contact_diagnostics import (
    ContactConstraintDiagnostics,
    ContactConstraintQualityIssue,
    ContactConstraintQualityReport,
    contact_constraint_matrix_diagnostics,
    assess_contact_constraint_quality,
)
from .contact_forms import (
    ContactBilinear,
    ContactBilinearLike,
    ContactJacobianReturn,
    ContactOperators,
    ContactSolveResult,
    ContactState,
    MixedSurfaceResidualForm,
    MultiplierContactContribution,
    PenaltyContactContribution,
    SurfaceHatFn,
    _compile_contact_bilinear,
    _infer_contact_backend,
    _is_compiled_contact_bilinear,
    compile_tagged_pair_nitsche_penalty_residual,
    make_tagged_pair_nitsche_penalty_bilinear,
)
from .contact_constraint_assembly import (
    assemble_contact_constraint_operators,
    assemble_contact_kkt,
    coarse_p1_basis_from_node_groups,
    coarse_p1_basis_from_surface_grid,
)
from .contact_penalty_assembly import (
    assemble_contact_penalty_operators,
    assemble_pair_nitsche_supermesh,
    solve_contact_al_jax,
    solve_contact_penalty_jax,
    update_contact_state_penalty,
)
from .contact_kkt_solver import (
    ContactKKTSolveConfig,
    ContactKKTSolveInfo,
    ContactKKTSolveResult,
    solve_contact_kkt,
    solve_contact_kkt_with_info,
)
from .contact_solvers import (
    AugmentedLagrangianState,
    AugmentedLagrangianResult,
    UnilateralContactActiveSetRecord,
    UnilateralContactActiveSetResult,
    solve_augmented_lagrangian_outer_loop,
    solve_unilateral_contact_active_set_kkt,
)
from .contact_embedding import (
    EmbeddingMap,
    build_nodal_embedding_map,
    build_barycentric_embedding_map,
    build_barycentric_embedding_map_from_meshes,
    assemble_embedding_constraint_matrix,
    assemble_fixed_rigid_hub_constraint_matrix,
    assemble_rigid_hub_constraint_matrix,
    assemble_rbe2_constraint_matrix,
    assemble_rbe3_constraint_matrix,
    build_rbe3_weights,
    build_rbe3_remote_resultant,
)
from .contact_nitsche import make_pair_nitsche_supermesh_bilinear
from .contact_surface_helpers import (
    OneSidedContact,
    active_contact_facets,
    facet_gap_values,
)
from .contact_surface_space import ContactSurfaceSpace, OneSidedContactSurfaceSpace, OneToManyContactSurfaceSpace

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


def assemble_contact_interface_residual(*args, **kwargs):
    """Assemble residual on a contact interface supermesh."""
    return _assemble_contact_interface_residual(*args, **kwargs)


def assemble_contact_interface_jacobian(*args, **kwargs):
    """Assemble Jacobian on a contact interface supermesh."""
    return _assemble_contact_interface_jacobian(*args, **kwargs)


def assemble_contact_coupling_matrices(*args, **kwargs):
    """Assemble coupling matrices for contact interface constraints."""
    return _assemble_contact_coupling_matrices(*args, **kwargs)


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
        formulation_key = None if formulation is None else str(formulation).lower().replace("-", "_")
        has_explicit_weak_form = any(value is not None for value in (weak_form, res_form, state, u))
        if formulation_key in {"pair_nitsche_penalty", "pair_nitsche", "nitsche_supermesh"} and not has_explicit_weak_form:
            if params is None:
                raise ValueError("params is required for formulation='pair_nitsche_penalty'.")
            return assemble_pair_nitsche_supermesh(
                contact,
                params,
                sparse=sparse,
                normal_source=normal_source,
            )
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
    "ContactConstraintDiagnostics",
    "ContactConstraintQualityIssue",
    "ContactConstraintQualityReport",
    "MortarContactProblemPair",
    "MortarContactProblem",
    "MultiplierContactContribution",
    "PenaltyContactContribution",
    "ContactState",
    "ContactKKTSolveInfo",
    "ContactKKTSolveResult",
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
    "UnilateralContactActiveSetRecord",
    "UnilateralContactActiveSetResult",
    "EmbeddingMap",
    "build_nodal_embedding_map",
    "build_barycentric_embedding_map",
    "build_barycentric_embedding_map_from_meshes",
    "assemble_embedding_constraint_matrix",
    "assemble_fixed_rigid_hub_constraint_matrix",
    "assemble_rigid_hub_constraint_matrix",
    "assemble_rbe2_constraint_matrix",
    "assemble_rbe3_constraint_matrix",
    "build_rbe3_weights",
    "make_pair_nitsche_supermesh_bilinear",
    "assemble_pair_nitsche_supermesh",
    "assemble_mortar_contact_problem",
    "assemble_contact_constraint_operators",
    "contact_constraint_matrix_diagnostics",
    "assess_contact_constraint_quality",
    "assemble_multiplier",
    "assemble_contact_operators",
    "assemble_contact_penalty_operators",
    "assemble_penalty",
    "assemble_contact_interface_residual",
    "assemble_contact_interface_jacobian",
    "assemble_contact_coupling_matrices",
    "assemble_contact_kkt",
    "solve_contact_kkt",
    "solve_contact_kkt_with_info",
    "solve_unilateral_contact_active_set_kkt",
    "solve_augmented_lagrangian_outer_loop",
    "facet_gap_values",
    "active_contact_facets",
]

# Phase-1 public naming aliases. These remain thin wrappers over the existing
# contact implementation until the state-explicit redesign is introduced.
PreparedContactInterface = ContactSurfaceSpace
PreparedOneToManyContactInterface = OneToManyContactSurfaceSpace
PreparedOneSidedContactInterface = OneSidedContactSurfaceSpace

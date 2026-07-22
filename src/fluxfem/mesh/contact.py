from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING
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
from .contact_nitsche import (
    assemble_pair_nitsche_supermesh_impl as _assemble_pair_nitsche_supermesh_impl,
    make_pair_nitsche_supermesh_bilinear,
)
from .contact_surface_helpers import (
    OneSidedContact,
    active_contact_facets,
    contact_space_side_n_dofs as _contact_space_side_n_dofs,
    facet_gap_values,
    onesided_gap_diagnostics as _onesided_gap_diagnostics,
    summarize_contact_field_state as _summarize_contact_field_state,
    surface_node_normals as _surface_node_normals,
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


def assemble_pair_nitsche_supermesh(
    contact,
    params: "WeakParams",
    *,
    sparse: bool = False,
    normal_source: str = "master",
    use_penalty: float | None = None,
    use_traction: float | None = None,
    backend_fastpath: str = "numpy_local_kernel",
) -> PenaltyContactContribution:
    """Assemble pair-Nitsche contact terms over a prepared contact supermesh."""
    return _assemble_pair_nitsche_supermesh_impl(
        contact,
        params,
        contribution_cls=PenaltyContactContribution,
        sparse=sparse,
        normal_source=normal_source,
        use_penalty=use_penalty,
        use_traction=use_traction,
        backend_fastpath=backend_fastpath,
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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

def _default_dtype():
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


DTYPE = _default_dtype()


_VOIGT_INNER_WEIGHTS = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=DTYPE)


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Return Lamé parameters (lambda, mu) from Young's modulus and Poisson ratio."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lam), float(mu)


def isotropic_3d_D(E: float, nu: float) -> jnp.ndarray:
    """Return 3D isotropic linear elasticity constitutive matrix in Voigt form."""
    lam, mu = lame_parameters(E, nu)

    D = jnp.array(
        [
            [lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=_default_dtype(),
    )
    return D


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class J2Plasticity:
    """Small-strain isotropic J2 material with linear isotropic hardening."""

    E: float
    nu: float
    yield_stress: float
    hardening_modulus: float = 0.0

    def tree_flatten(self):
        return (), (float(self.E), float(self.nu), float(self.yield_stress), float(self.hardening_modulus))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class J2PlasticityState:
    """Material-point history for small-strain J2 plasticity.

    ``plastic_strain`` uses the same 6-component Voigt convention as
    ``isotropic_3d_D``: ``[xx, yy, zz, xy, yz, zx]`` with engineering shear
    strains in the last three components.
    """

    plastic_strain: jnp.ndarray
    equivalent_plastic_strain: jnp.ndarray

    def tree_flatten(self):
        return (self.plastic_strain, self.equivalent_plastic_strain), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        plastic_strain, equivalent_plastic_strain = children
        return cls(plastic_strain=plastic_strain, equivalent_plastic_strain=equivalent_plastic_strain)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class J2PlasticityQuadratureState:
    """Element/quadrature history arrays for small-strain J2 plasticity.

    The array shapes are ``(n_elems, n_q, 6)`` for plastic strain and
    ``(n_elems, n_q)`` for equivalent plastic strain.
    """

    plastic_strain: jnp.ndarray
    equivalent_plastic_strain: jnp.ndarray

    def tree_flatten(self):
        return (self.plastic_strain, self.equivalent_plastic_strain), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        plastic_strain, equivalent_plastic_strain = children
        return cls(plastic_strain=plastic_strain, equivalent_plastic_strain=equivalent_plastic_strain)


@dataclass(frozen=True)
class J2LoadStepResult:
    """One J2 load-step result with committed/trial state diagnostics."""

    load_factor: float
    converged: bool
    committed: bool
    u: Any
    state: J2PlasticityQuadratureState
    trial_state: J2PlasticityQuadratureState
    info: Any
    max_equivalent_plastic_strain: float
    exception: str | None = None


def make_j2_plasticity_state(*, dtype=None) -> J2PlasticityState:
    """Return a zero-history J2 material-point state."""
    dtype = _default_dtype() if dtype is None else dtype
    return J2PlasticityState(
        plastic_strain=jnp.zeros((6,), dtype=dtype),
        equivalent_plastic_strain=jnp.asarray(0.0, dtype=dtype),
    )


def make_j2_quadrature_state(
    space_or_n_elems,
    n_q: int | None = None,
    *,
    dtype=None,
) -> J2PlasticityQuadratureState:
    """Return zero J2 history arrays for every element quadrature point.

    ``space_or_n_elems`` may be an FE space with ``elem_dofs`` and ``basis`` or
    an integer element count. When an integer is passed, ``n_q`` is required.
    """
    dtype = _default_dtype() if dtype is None else dtype
    if hasattr(space_or_n_elems, "elem_dofs") and hasattr(space_or_n_elems, "basis"):
        n_elems = int(space_or_n_elems.elem_dofs.shape[0])
        n_q_val = int(space_or_n_elems.basis.shape_functions().shape[0])
    else:
        if n_q is None:
            raise ValueError("n_q is required when space_or_n_elems is an integer.")
        n_elems = int(space_or_n_elems)
        n_q_val = int(n_q)
    return J2PlasticityQuadratureState(
        plastic_strain=jnp.zeros((n_elems, n_q_val, 6), dtype=dtype),
        equivalent_plastic_strain=jnp.zeros((n_elems, n_q_val), dtype=dtype),
    )


def j2_element_state(state: J2PlasticityQuadratureState, elem_id) -> J2PlasticityState:
    """Extract one element's quadrature history as a material-point pytree batch."""
    return J2PlasticityState(
        plastic_strain=state.plastic_strain[elem_id],
        equivalent_plastic_strain=state.equivalent_plastic_strain[elem_id],
    )


def voigt_trace(strain_or_stress: jnp.ndarray) -> jnp.ndarray:
    """Trace of a 6-component symmetric tensor in Voigt form."""
    x = jnp.asarray(strain_or_stress)
    return x[..., 0] + x[..., 1] + x[..., 2]


def voigt_deviator(stress: jnp.ndarray) -> jnp.ndarray:
    """Deviatoric part of a 6-component stress vector."""
    s = jnp.asarray(stress)
    mean = voigt_trace(s) / 3.0
    shift = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=s.dtype) * mean[..., None]
    return s - shift


def voigt_tensor_inner(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Tensor inner product for 6-component Voigt vectors.

    Shear stress components are physical tensor components, so the inner
    product weights the last three components by two.
    """
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    weights = jnp.asarray(_VOIGT_INNER_WEIGHTS, dtype=jnp.result_type(a_arr, b_arr))
    return jnp.sum(weights * a_arr * b_arr, axis=-1)


def von_mises_stress_voigt(stress: jnp.ndarray) -> jnp.ndarray:
    """Von Mises equivalent stress from a 6-component Voigt stress vector."""
    dev = voigt_deviator(stress)
    return jnp.sqrt(jnp.maximum(1.5 * voigt_tensor_inner(dev, dev), 0.0))


def voigt_stress_to_tensor(stress: jnp.ndarray) -> jnp.ndarray:
    """Convert stress ``[xx, yy, zz, xy, yz, zx]`` to a symmetric tensor."""
    s = jnp.asarray(stress)
    return jnp.stack(
        [
            jnp.stack([s[..., 0], s[..., 3], s[..., 5]], axis=-1),
            jnp.stack([s[..., 3], s[..., 1], s[..., 4]], axis=-1),
            jnp.stack([s[..., 5], s[..., 4], s[..., 2]], axis=-1),
        ],
        axis=-2,
    )


def small_strain_voigt_from_grad(grad_u: jnp.ndarray) -> jnp.ndarray:
    """Small strain in Voigt form from displacement gradient ``du_i/dx_j``."""
    g = jnp.asarray(grad_u)
    return jnp.stack(
        [
            g[..., 0, 0],
            g[..., 1, 1],
            g[..., 2, 2],
            g[..., 0, 1] + g[..., 1, 0],
            g[..., 1, 2] + g[..., 2, 1],
            g[..., 2, 0] + g[..., 0, 2],
        ],
        axis=-1,
    )


def small_strain_voigt(ctx, u_elem: jnp.ndarray) -> jnp.ndarray:
    """Small-strain Voigt vector at element quadrature points."""
    return small_strain_voigt_from_grad(ctx.trial.grad(u_elem))


def j2_yield_function(stress: jnp.ndarray, state: J2PlasticityState, material: J2Plasticity) -> jnp.ndarray:
    """Yield function ``sigma_eq - (sigma_y + H p)``."""
    return von_mises_stress_voigt(stress) - (
        jnp.asarray(material.yield_stress, dtype=jnp.asarray(stress).dtype)
        + jnp.asarray(material.hardening_modulus, dtype=jnp.asarray(stress).dtype) * state.equivalent_plastic_strain
    )


def j2_return_mapping(strain: jnp.ndarray, state: J2PlasticityState, material: J2Plasticity) -> tuple[jnp.ndarray, J2PlasticityState]:
    """Update one material point by radial return mapping.

    Parameters
    ----------
    strain:
        Total small strain in 6-component Voigt form with engineering shear.
    state:
        Previous converged plastic state.
    material:
        Isotropic J2 material with linear isotropic hardening.

    Returns
    -------
    stress, next_state
        Updated Cauchy stress in Voigt form and updated material history.
    """
    eps = jnp.asarray(strain)
    eps_p = jnp.asarray(state.plastic_strain, dtype=eps.dtype)
    p_n = jnp.asarray(state.equivalent_plastic_strain, dtype=eps.dtype)
    D = isotropic_3d_D(material.E, material.nu).astype(eps.dtype)
    _lam, mu = lame_parameters(material.E, material.nu)
    mu_arr = jnp.asarray(mu, dtype=eps.dtype)
    H = jnp.asarray(material.hardening_modulus, dtype=eps.dtype)

    trial_stress = D @ (eps - eps_p)
    s_trial = voigt_deviator(trial_stress)
    sigma_eq_trial_sq = jnp.maximum(1.5 * voigt_tensor_inner(s_trial, s_trial), 0.0)
    yield_current = jnp.asarray(material.yield_stress, dtype=eps.dtype) + H * p_n
    plastic = sigma_eq_trial_sq > yield_current * yield_current

    def elastic_update(_):
        return trial_stress, J2PlasticityState(plastic_strain=eps_p, equivalent_plastic_strain=p_n)

    def plastic_update(_):
        sigma_eq_trial = jnp.sqrt(sigma_eq_trial_sq)
        f_trial = sigma_eq_trial - yield_current
        safe_sigma_eq = jnp.maximum(sigma_eq_trial, jnp.asarray(1.0e-30, dtype=eps.dtype))
        dgamma = f_trial / (3.0 * mu_arr + H)
        scale = 1.0 - (3.0 * mu_arr * dgamma / safe_sigma_eq)
        s_next = scale * s_trial
        hydro = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=eps.dtype) * (voigt_trace(trial_stress) / 3.0)
        stress_next = hydro + s_next

        flow = jnp.concatenate(
            [
                1.5 * s_trial[:3] / safe_sigma_eq,
                3.0 * s_trial[3:] / safe_sigma_eq,
            ]
        )
        eps_p_next = eps_p + dgamma * flow
        p_next = p_n + dgamma
        next_state = J2PlasticityState(plastic_strain=eps_p_next, equivalent_plastic_strain=p_next)
        return stress_next, next_state

    return jax.lax.cond(plastic, plastic_update, elastic_update, operand=None)


def j2_update_element_quadrature_state(
    ctx,
    u_elem: jnp.ndarray,
    element_state: J2PlasticityState,
    material: J2Plasticity,
) -> tuple[jnp.ndarray, J2PlasticityState]:
    """Return quadrature stresses and updated state for one element."""
    strain = small_strain_voigt(ctx, u_elem)
    return jax.vmap(j2_return_mapping, in_axes=(0, 0, None))(strain, element_state, material)


def j2_plasticity_residual_form(ctx, u_elem: jnp.ndarray, params) -> jnp.ndarray:
    """Small-strain J2 internal-force residual using frozen quadrature state.

    ``params`` may be a mapping with ``material`` and ``state`` keys or an
    object with matching attributes. The state is read at ``ctx.elem_id``.
    """
    if isinstance(params, Mapping):
        material = params["material"]
        state = params["state"]
    else:
        material = params.material
        state = params.state
    stress, _next_state = j2_update_element_quadrature_state(
        ctx,
        u_elem,
        j2_element_state(state, ctx.elem_id),
        material,
    )
    sigma = voigt_stress_to_tensor(stress)
    elem_res = jnp.einsum("qaj,qij->qai", ctx.trial.gradN, sigma)
    return elem_res.reshape(elem_res.shape[0], -1)


j2_plasticity_residual_form._ff_kind = "residual"
j2_plasticity_residual_form._ff_domain = "volume"


def update_j2_quadrature_state(
    space,
    u: jnp.ndarray,
    state: J2PlasticityQuadratureState,
    material: J2Plasticity,
) -> J2PlasticityQuadratureState:
    """Update all element quadrature states after a converged displacement."""
    ctxs = space.build_form_contexts()
    u_elems = jnp.asarray(u)[space.elem_dofs]

    def per_element(ctx, u_elem, eps_p, p):
        _stress, next_state = j2_update_element_quadrature_state(
            ctx,
            u_elem,
            J2PlasticityState(plastic_strain=eps_p, equivalent_plastic_strain=p),
            material,
        )
        return next_state.plastic_strain, next_state.equivalent_plastic_strain

    plastic_strain, equivalent_plastic_strain = jax.vmap(per_element)(
        ctxs,
        u_elems,
        state.plastic_strain,
        state.equivalent_plastic_strain,
    )
    return J2PlasticityQuadratureState(
        plastic_strain=plastic_strain,
        equivalent_plastic_strain=equivalent_plastic_strain,
    )


def evaluate_j2_quadrature_strain(space, u: jnp.ndarray) -> jnp.ndarray:
    """Evaluate small-strain Voigt vectors at all element quadrature points."""
    ctxs = space.build_form_contexts()
    u_elems = jnp.asarray(u)[space.elem_dofs]
    return jax.vmap(small_strain_voigt)(ctxs, u_elems)


def evaluate_j2_quadrature_stress(
    space,
    u: jnp.ndarray,
    state: J2PlasticityQuadratureState,
    material: J2Plasticity,
) -> jnp.ndarray:
    """Evaluate J2 stresses from a frozen committed quadrature state."""
    ctxs = space.build_form_contexts()
    u_elems = jnp.asarray(u)[space.elem_dofs]

    def per_element(ctx, u_elem, eps_p, p):
        stress, _next_state = j2_update_element_quadrature_state(
            ctx,
            u_elem,
            J2PlasticityState(plastic_strain=eps_p, equivalent_plastic_strain=p),
            material,
        )
        return stress

    return jax.vmap(per_element)(
        ctxs,
        u_elems,
        state.plastic_strain,
        state.equivalent_plastic_strain,
    )


def make_j2_cell_data(
    space,
    u: jnp.ndarray,
    state: J2PlasticityQuadratureState,
    material: J2Plasticity,
) -> dict[str, np.ndarray]:
    """Return element-averaged J2 fields suitable for VTU cell data."""
    stress = evaluate_j2_quadrature_stress(space, u, state, material)
    p_eq = jnp.asarray(state.equivalent_plastic_strain)
    sigma_vm = von_mises_stress_voigt(stress)
    return {
        "j2_p_eq": np.asarray(jnp.mean(p_eq, axis=1), dtype=np.float64),
        "j2_sigma_vm": np.asarray(jnp.mean(sigma_vm, axis=1), dtype=np.float64),
        "j2_sigma_xx": np.asarray(jnp.mean(stress[..., 0], axis=1), dtype=np.float64),
        "j2_sigma_yy": np.asarray(jnp.mean(stress[..., 1], axis=1), dtype=np.float64),
        "j2_sigma_zz": np.asarray(jnp.mean(stress[..., 2], axis=1), dtype=np.float64),
        "j2_stress_voigt": np.asarray(jnp.mean(stress, axis=1), dtype=np.float64),
    }


def make_j2_point_and_cell_data(
    mesh,
    space,
    u: jnp.ndarray,
    state: J2PlasticityQuadratureState,
    material: J2Plasticity,
    *,
    compute_j: bool = False,
    deformed_scale: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return displacement point data and J2 element-averaged cell data."""
    from ..postprocess import make_point_data_displacement

    point_data = make_point_data_displacement(
        mesh,
        space,
        u,
        compute_j=compute_j,
        deformed_scale=deformed_scale,
    )
    cell_data = make_j2_cell_data(space, u, state, material)
    return point_data, cell_data


def write_j2_vtu(
    mesh,
    space,
    u: jnp.ndarray,
    state: J2PlasticityQuadratureState,
    material: J2Plasticity,
    filepath: str,
    *,
    compute_j: bool = False,
    deformed_scale: float = 1.0,
) -> None:
    """Write VTU with displacement point data and J2 cell data."""
    from ...tools.visualizer import write_vtu

    point_data, cell_data = make_j2_point_and_cell_data(
        mesh,
        space,
        u,
        state,
        material,
        compute_j=compute_j,
        deformed_scale=deformed_scale,
    )
    write_vtu(mesh, filepath, point_data=point_data, cell_data=cell_data)


def _j2_normalize_dirichlet(dirichlet):
    if dirichlet is None:
        return None
    from ...solver.dirichlet import DirichletBC, _normalize_dirichlet

    if isinstance(dirichlet, DirichletBC):
        dirichlet = dirichlet.as_tuple()
    dofs, vals = _normalize_dirichlet(dirichlet[0], dirichlet[1])
    if vals.ndim == 0:
        vals = np.full(dofs.shape[0], float(vals), dtype=float)
    return dofs, vals


def _j2_schedule(load_sequence: Sequence[float] | None, n_steps: int) -> list[float]:
    if load_sequence is not None:
        raw = list(load_sequence)
    else:
        raw = list(np.linspace(0.0, 1.0, max(1, int(n_steps)) + 1, endpoint=True)[1:])
    schedule: list[float] = []
    prev = 0.0
    for value in raw:
        lf = float(value)
        if not np.isfinite(lf):
            raise ValueError("load factors must be finite.")
        if lf < 0.0 or lf > 1.0:
            raise ValueError("load factors must be in [0, 1].")
        if lf < prev:
            raise ValueError("load factors must be monotone nondecreasing.")
        schedule.append(lf)
        prev = lf
    return schedule


def solve_j2_plasticity_load_steps(
    space,
    material: J2Plasticity,
    *,
    initial_state: J2PlasticityQuadratureState | None = None,
    u0: jnp.ndarray | None = None,
    dirichlet=None,
    base_external_vector=None,
    load_sequence: Sequence[float] | None = None,
    n_steps: int = 1,
    tol: float = 1.0e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    linear_solver: str = "spsolve",
    line_search: bool = False,
    assembly_policy=None,
    jacobian_pattern=None,
    commit_on_converged: bool = True,
    stop_on_failure: bool = True,
) -> tuple[jnp.ndarray, J2PlasticityQuadratureState, list[J2LoadStepResult]]:
    """Solve small-strain J2 load steps with explicit state commit.

    The committed quadrature state is held fixed inside each Newton solve and
    updated only after a converged step. Dirichlet values and external loads are
    scaled by the load factor, so the helper supports basic force or prescribed
    displacement stepping.
    """
    from ...solver.result import SolverResult
    from ...solver.newton import newton_solve

    dtype = jnp.asarray(u0).dtype if u0 is not None else _default_dtype()
    state = initial_state if initial_state is not None else make_j2_quadrature_state(space, dtype=dtype)
    u = jnp.zeros((space.n_dofs,), dtype=dtype) if u0 is None else jnp.asarray(u0, dtype=dtype)
    dirichlet_final = _j2_normalize_dirichlet(dirichlet)
    schedule = _j2_schedule(load_sequence, n_steps)
    history: list[J2LoadStepResult] = []

    if dirichlet_final is None:
        final_dofs = np.array([], dtype=int)
        final_vals = np.array([], dtype=float)
    else:
        final_dofs, final_vals = dirichlet_final

    free_mask = np.ones(int(space.n_dofs), dtype=bool)
    free_mask[final_dofs] = False
    has_free_dofs = bool(np.any(free_mask))

    for lf in schedule:
        step_dofs = final_dofs
        step_vals = lf * final_vals
        if step_dofs.size:
            u = u.at[jnp.asarray(step_dofs, dtype=jnp.int64)].set(jnp.asarray(step_vals, dtype=u.dtype))

        params = {"material": material, "state": state}
        external = None
        if base_external_vector is not None:
            external = jnp.asarray(lf * jnp.asarray(base_external_vector, dtype=u.dtype), dtype=u.dtype)
        exception = None
        if has_free_dofs:
            try:
                u_step, info = newton_solve(
                    space,
                    j2_plasticity_residual_form,
                    u,
                    params,
                    tol=tol,
                    atol=atol,
                    maxiter=maxiter,
                    linear_solver=linear_solver,
                    dirichlet=(step_dofs, step_vals) if step_dofs.size else None,
                    line_search=line_search,
                    external_vector=external,
                    jacobian_pattern=jacobian_pattern,
                    assembly_policy=assembly_policy,
                )
                u = jnp.asarray(u_step, dtype=u.dtype)
            except Exception as exc:  # pragma: no cover - defensive path
                info = SolverResult(converged=False, iters=0, stop_reason="exception")
                exception = repr(exc)
        else:
            info = SolverResult(
                converged=True,
                iters=0,
                residual_norm=0.0,
                residual0=0.0,
                rel_residual=0.0,
                stop_reason="all_dirichlet",
            )

        converged = bool(getattr(info, "converged", False))
        trial_state = update_j2_quadrature_state(space, u, state, material)
        committed = bool(converged and commit_on_converged)
        if committed:
            state = trial_state
        max_p = float(jax.block_until_ready(jnp.max(trial_state.equivalent_plastic_strain)))
        history.append(
            J2LoadStepResult(
                load_factor=float(lf),
                converged=converged,
                committed=committed,
                u=u,
                state=state,
                trial_state=trial_state,
                info=info,
                max_equivalent_plastic_strain=max_p,
                exception=exception,
            )
        )
        if not converged and stop_on_failure:
            break

    return u, state, history


__all__ = [
    "J2LoadStepResult",
    "J2Plasticity",
    "J2PlasticityQuadratureState",
    "J2PlasticityState",
    "evaluate_j2_quadrature_strain",
    "evaluate_j2_quadrature_stress",
    "j2_element_state",
    "j2_plasticity_residual_form",
    "j2_return_mapping",
    "j2_update_element_quadrature_state",
    "j2_yield_function",
    "lame_parameters",
    "isotropic_3d_D",
    "make_j2_cell_data",
    "make_j2_point_and_cell_data",
    "make_j2_plasticity_state",
    "make_j2_quadrature_state",
    "small_strain_voigt",
    "small_strain_voigt_from_grad",
    "solve_j2_plasticity_load_steps",
    "update_j2_quadrature_state",
    "voigt_deviator",
    "voigt_stress_to_tensor",
    "voigt_tensor_inner",
    "voigt_trace",
    "von_mises_stress_voigt",
    "write_j2_vtu",
]

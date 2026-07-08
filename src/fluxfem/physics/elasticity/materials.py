from dataclasses import dataclass

import jax
import jax.numpy as jnp

DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


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
        dtype=DTYPE,
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


def make_j2_plasticity_state(*, dtype=DTYPE) -> J2PlasticityState:
    """Return a zero-history J2 material-point state."""
    return J2PlasticityState(
        plastic_strain=jnp.zeros((6,), dtype=dtype),
        equivalent_plastic_strain=jnp.asarray(0.0, dtype=dtype),
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
    sigma_eq_trial = von_mises_stress_voigt(trial_stress)
    f_trial = sigma_eq_trial - (jnp.asarray(material.yield_stress, dtype=eps.dtype) + H * p_n)
    plastic = f_trial > 0.0

    safe_sigma_eq = jnp.maximum(sigma_eq_trial, jnp.asarray(1.0e-30, dtype=eps.dtype))
    dgamma = jnp.where(plastic, f_trial / (3.0 * mu_arr + H), jnp.asarray(0.0, dtype=eps.dtype))
    scale = jnp.where(plastic, 1.0 - (3.0 * mu_arr * dgamma / safe_sigma_eq), 1.0)
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


__all__ = [
    "J2Plasticity",
    "J2PlasticityState",
    "j2_return_mapping",
    "j2_yield_function",
    "lame_parameters",
    "isotropic_3d_D",
    "make_j2_plasticity_state",
    "voigt_deviator",
    "voigt_tensor_inner",
    "voigt_trace",
    "von_mises_stress_voigt",
]

import jax
import jax.numpy as jnp

DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


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


__all__ = ["lame_parameters", "isotropic_3d_D"]

from __future__ import annotations

from typing import TypeAlias

import jax.numpy as jnp

ArrayLike: TypeAlias = jnp.ndarray


def _sym(A: ArrayLike) -> ArrayLike:
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def principal_stresses(S: ArrayLike) -> ArrayLike:
    """
    Return principal stresses (eigvals) for symmetric 3x3 stress tensor.
    Supports batching over leading dimensions.
    """
    S_sym = _sym(S)
    return jnp.linalg.eigvalsh(S_sym)


def principal_sum(S: ArrayLike) -> ArrayLike:
    """Sum of principal stresses (trace)."""
    return jnp.trace(S, axis1=-2, axis2=-1)


def max_shear_stress(S: ArrayLike) -> ArrayLike:
    """
    Maximum shear stress = (sigma_max - sigma_min) / 2.
    """
    vals = principal_stresses(S)
    return 0.5 * (vals[..., -1] - vals[..., 0])


def von_mises_stress(S: ArrayLike) -> ArrayLike:
    """
    von Mises equivalent stress: sqrt(3/2 * dev(S):dev(S)).
    """
    tr = jnp.trace(S, axis1=-2, axis2=-1)[..., None, None]
    dev = S - tr / 3.0
    return jnp.sqrt(1.5 * jnp.sum(dev * dev, axis=(-2, -1)))


__all__ = [
    "principal_stresses",
    "principal_sum",
    "max_shear_stress",
    "von_mises_stress",
]

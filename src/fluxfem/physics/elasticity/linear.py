from typing import TypeAlias

import jax.numpy as jnp

from ...core.assembly import LinearReturn
from ...core.forms import FormContext, vector_load_form
from ...core.space import FESpace
from ...physics.operators import sym_grad

ArrayLike: TypeAlias = jnp.ndarray

# from ...mechanics.kinematics import build_B_matrices


# def linear_elasticity_form(ctx: FormContext, D: jnp.ndarray) -> jnp.ndarray:
#     """3D linear elasticity bilinear form B^T D B."""
#     grad_v = ctx.test.grad
#     grad_u = ctx.trial.grad
#     B = build_B_matrices(grad_u)               # (n_q, 6, 24)
#     BT = jnp.swapaxes(build_B_matrices(grad_v), 1, 2)  # (n_q, 24, 6)
#     BDB = jnp.einsum("qik,kl,qlm->qim", BT, D, B)  # (n_q, 24, 24)
#     return BDB


def linear_elasticity_form(ctx: FormContext, D: jnp.ndarray) -> jnp.ndarray:
    """
    Linear-elasticity bilinear form in Voigt notation.

    Returns the per-quadrature integrand for Bv^T D Bu, where B is the
    symmetric-gradient operator for the test/trial fields.
    """
    Bu = sym_grad(ctx.trial)                 # (n_q, 6, ndofs_e)
    Bv = Bu if ctx.test is ctx.trial else sym_grad(ctx.test)
    return jnp.einsum("qik,kl,qlm->qim", jnp.swapaxes(Bv, 1, 2), D, Bu)


linear_elasticity_form._ff_kind = "bilinear"
linear_elasticity_form._ff_domain = "volume"


def vector_body_force_form(ctx: FormContext, load_vec: jnp.ndarray) -> jnp.ndarray:
    """Linear form for 3D vector body force f (constant in space)."""
    return vector_load_form(ctx.test, load_vec)


vector_body_force_form._ff_kind = "linear"
vector_body_force_form._ff_domain = "volume"

def assemble_constant_body_force(
    space: FESpace,
    gravity_vec: ArrayLike,
    density: float,
    *,
    sparse: bool = False,
) -> LinearReturn:
    """
    Convenience: assemble body force from density * gravity vector.
    gravity_vec: length-3 array-like (direction and magnitude of g)
    density: scalar density (consistent with unit system)
    """
    from ...core.assembly import assemble_linear_form
    g = jnp.asarray(gravity_vec)
    f_vec = density * g
    return assemble_linear_form(space, vector_body_force_form, params=f_vec, sparse=sparse)


# Backward compatibility alias
constant_body_force_vector_form = vector_body_force_form


__all__ = [
    "linear_elasticity_form",
    "vector_body_force_form",
    "constant_body_force_vector_form",
    "assemble_constant_body_force",
]

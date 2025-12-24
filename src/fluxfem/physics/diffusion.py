import jax.numpy as jnp

from ..core.forms import FormContext


def diffusion_form(ctx: FormContext, kappa: float) -> jnp.ndarray:
    """
    Scalar diffusion bilinear form: kappa * grad_v · grad_u.

    Returns the per-quadrature integrand for a standard Laplacian term.
    """
    grad_v = ctx.test.gradN
    grad_u = ctx.trial.gradN
    G = jnp.einsum("qia,qja->qij", grad_v, grad_u)  # ∇v_i · ∇u_j
    return kappa * G


__all__ = ["diffusion_form"]

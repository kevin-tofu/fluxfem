import jax

from ..core.assembly import assemble_residual, assemble_jacobian
from ..core.space import FESpace


def make_jitted_residual(space: FESpace, res_form, params, *, sparse: bool = False):
    """
    Create a jitted residual assembler: u -> R(u).
    params and space are closed over.
    """
    space_jax = space
    params_jax = params

    @jax.jit
    def residual(u):
        return assemble_residual(space_jax, res_form, u, params_jax, sparse=sparse)

    return residual


def make_jitted_jacobian(
    space: FESpace,
    res_form,
    params,
    *,
    sparse: bool = False,
    return_flux_matrix: bool = False,
):
    """
    Create a jitted Jacobian assembler: u -> J(u).
    params and space are closed over.
    """
    space_jax = space
    params_jax = params

    @jax.jit
    def jacobian(u):
        return assemble_jacobian(
            space_jax,
            res_form,
            u,
            params_jax,
            sparse=sparse,
            return_flux_matrix=return_flux_matrix,
        )

    return jacobian


__all__ = ["make_jitted_residual", "make_jitted_jacobian"]

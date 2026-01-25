from __future__ import annotations

import jax

from typing import Callable, TypeVar

from ..core.assembly import JacobianReturn, LinearReturn, ResidualForm, assemble_jacobian, assemble_residual
from ..core.space import FESpace

P = TypeVar("P")


def make_jitted_residual(
    space: FESpace,
    res_form: ResidualForm[P],
    params: P,
    *,
    sparse: bool = False,
) -> Callable[[jax.Array], LinearReturn]:
    """
    Create a jitted residual assembler: u -> R(u).
    params and space are closed over.
    """
    space_jax = space
    params_jax = params

    @jax.jit
    def residual(u: jax.Array) -> LinearReturn:
        return assemble_residual(space_jax, res_form, u, params_jax, sparse=sparse)

    return residual


def make_jitted_jacobian(
    space: FESpace,
    res_form: ResidualForm[P],
    params: P,
    *,
    sparse: bool = False,
    return_flux_matrix: bool = False,
) -> Callable[[jax.Array], JacobianReturn]:
    """
    Create a jitted Jacobian assembler: u -> J(u).
    params and space are closed over.
    """
    space_jax = space
    params_jax = params

    @jax.jit
    def jacobian(u: jax.Array) -> JacobianReturn:
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

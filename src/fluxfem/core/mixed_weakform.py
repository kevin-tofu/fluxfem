"""
Mixed weak-form helpers that keep core assembly untouched.

This module provides a small convenience API to build and assemble mixed
weak forms using the existing MixedWeakForm compiler.
"""

from __future__ import annotations

from typing import Callable, Mapping

import jax.numpy as jnp

from .mixed_space import MixedFESpace
from .weakform import MixedWeakForm

MixedResiduals = Mapping[str, Callable]

class MixedResidualForm:
    """Wrapper for mixed residuals to mirror the single-field ResidualForm API."""

    def __init__(self, residuals: MixedResiduals):
        self.residuals = dict(residuals)

    def get_compiled(self):
        return compile_mixed_weak_form(self.residuals)


def compile_mixed_weak_form(residuals: MixedResiduals):
    """
    Compile mixed weak-form residuals into an element kernel.

    The residuals must return Expr and include dOmega().
    """
    return MixedWeakForm(residuals=dict(residuals)).get_compiled()


def _wrap_params(res_form, params):
    if callable(params):
        def _wrapped(ctx, u_elem, _params):
            return res_form(ctx, u_elem, params(ctx))

        _wrapped._includes_measure = getattr(res_form, "_includes_measure", False)  # type: ignore[attr-defined]
        return _wrapped, None
    return res_form, params


def assemble_mixed_residual_wf(
    space: MixedFESpace,
    residuals: MixedResiduals | MixedWeakForm | MixedResidualForm | Callable,
    u: jnp.ndarray | Mapping[str, jnp.ndarray],
    params,
    **kwargs,
):
    """
    Assemble mixed residual from weak-form definitions.
    """
    if isinstance(residuals, MixedWeakForm):
        res_form = residuals.get_compiled()
    elif isinstance(residuals, MixedResidualForm):
        res_form = residuals.get_compiled()
    elif isinstance(residuals, Mapping):
        res_form = compile_mixed_weak_form(residuals)
    else:
        res_form = residuals
    res_form, params = _wrap_params(res_form, params)
    return space.assemble_residual(res_form, u, params, **kwargs)


def assemble_mixed_jacobian_wf(
    space: MixedFESpace,
    residuals: MixedResiduals | MixedWeakForm | MixedResidualForm | Callable,
    u: jnp.ndarray | Mapping[str, jnp.ndarray],
    params,
    **kwargs,
):
    """
    Assemble mixed Jacobian from weak-form definitions.
    """
    if isinstance(residuals, MixedWeakForm):
        res_form = residuals.get_compiled()
    elif isinstance(residuals, MixedResidualForm):
        res_form = residuals.get_compiled()
    elif isinstance(residuals, Mapping):
        res_form = compile_mixed_weak_form(residuals)
    else:
        res_form = residuals
    res_form, params = _wrap_params(res_form, params)
    return space.assemble_jacobian(res_form, u, params, **kwargs)


__all__ = [
    "MixedResidualForm",
    "compile_mixed_weak_form",
    "assemble_mixed_residual_wf",
    "assemble_mixed_jacobian_wf",
]

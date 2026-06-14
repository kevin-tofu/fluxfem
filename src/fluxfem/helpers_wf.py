"""WeakForm/Expr helpers (symbolic operators)."""
from __future__ import annotations

from .core.weakform import (
    grad,
    sym_grad,
    dot,
    sdot,
    ddot,
    inner,
    action,
    gaction,
    outer,
    I,
    det,
    inv,
    transpose,
    transpose_last2,
    matmul,
    matmul_std,
    log,
    einsum,
    normal,
    ds,
    dOmega,
    ParamRef,
)


def _voigt_A() -> tuple[tuple[tuple[float, ...], ...], ...]:
    return (
        ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.5)),
        ((0.0, 0.0, 0.0, 0.5, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.5, 0.0)),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    )


def _identity3() -> tuple[tuple[float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def voigt_to_tensor(sym_grad_u, p=None):
    """
    Convert Voigt-form symmetric gradient to a 3x3 tensor.

    Uses the standard Voigt mapping with 1/2 on shear terms.
    """
    A = _voigt_A()
    if p is not None and not isinstance(p, ParamRef):
        A = getattr(p, "A", A)
    return einsum("ijk,qk...->qij...", A, sym_grad_u)


def linear_stress(sym_grad_u, p):
    """Linear elastic stress from symmetric gradient in Voigt notation."""
    I = _identity3()
    if not isinstance(p, ParamRef):
        I = getattr(p, "I", I)
    eps = voigt_to_tensor(sym_grad_u, p)
    tr = einsum("ij,qij...->q...", I, eps)
    return p.lam * einsum("q...,ij->qij...", tr, I) + 2.0 * p.mu * eps


def traction(field, n, p):
    """Traction vector for a field using linear elastic stress."""
    stress = linear_stress(sym_grad(field), p)
    return einsum("qij...,qj->qi...", stress, n)

__all__ = [
    "grad",
    "sym_grad",
    "dot",
    "sdot",
    "ddot",
    "inner",
    "action",
    "gaction",
    "outer",
    "I",
    "det",
    "inv",
    "transpose",
    "transpose_last2",
    "matmul",
    "matmul_std",
    "log",
    "voigt_to_tensor",
    "linear_stress",
    "traction",
    "normal",
    "ds",
    "dOmega",
]

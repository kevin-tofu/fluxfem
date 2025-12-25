"""Tensor helpers (array operators)."""
from __future__ import annotations

from .physics.operators import dot, ddot, sym_grad, transpose_last2

__all__ = [
    "dot",
    "ddot",
    "sym_grad",
    "transpose_last2",
]

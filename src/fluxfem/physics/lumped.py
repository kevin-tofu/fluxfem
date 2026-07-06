from __future__ import annotations

from typing import Sequence

import numpy as np
import jax.numpy as jnp

from ..solver.sparse import FluxSparseMatrix


def _as_dofs(dofs: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(dofs, dtype=int).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one DOF.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return arr


def _coerce_square(value, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.eye(n, dtype=float) * float(arr)
    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError(f"{name} vector length must match selected DOFs.")
        return np.diag(arr)
    if arr.shape != (n, n):
        raise ValueError(f"{name} matrix shape must be ({n}, {n}).")
    return arr


def _connector_matrix(
    *,
    n_dofs: int,
    dofs: Sequence[int] | np.ndarray,
    value,
    other_dofs: Sequence[int] | np.ndarray | None,
    value_name: str,
) -> FluxSparseMatrix:
    if n_dofs <= 0:
        raise ValueError("n_dofs must be positive.")
    a = _as_dofs(dofs, "dofs")
    if np.any(a >= n_dofs):
        raise ValueError("dofs contains an index outside n_dofs.")

    k = _coerce_square(value, int(a.size), value_name)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_block(row_dofs: np.ndarray, col_dofs: np.ndarray, block: np.ndarray) -> None:
        rr, cc = np.meshgrid(row_dofs, col_dofs, indexing="ij")
        rows.extend(rr.reshape(-1).tolist())
        cols.extend(cc.reshape(-1).tolist())
        data.extend(block.reshape(-1).tolist())

    if other_dofs is None:
        add_block(a, a, k)
    else:
        b = _as_dofs(other_dofs, "other_dofs")
        if b.size != a.size:
            raise ValueError("other_dofs must have the same size as dofs.")
        if np.any(b >= n_dofs):
            raise ValueError("other_dofs contains an index outside n_dofs.")
        add_block(a, a, k)
        add_block(a, b, -k)
        add_block(b, a, -k)
        add_block(b, b, k)

    return FluxSparseMatrix(
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(data, dtype=jnp.float64),
        int(n_dofs),
    ).coalesce()


def assemble_dof_spring(
    n_dofs: int,
    dofs: Sequence[int] | np.ndarray,
    stiffness,
    *,
    other_dofs: Sequence[int] | np.ndarray | None = None,
) -> FluxSparseMatrix:
    """
    Assemble a linear spring matrix on selected DOFs.

    With ``other_dofs=None`` this creates a spring-to-ground contribution
    ``u[dofs]^T K u[dofs]``. With ``other_dofs`` it creates a connector
    contribution based on the relative displacement ``u[dofs] - u[other_dofs]``.
    ``stiffness`` may be a scalar, a vector of diagonal values, or a square matrix.
    """
    return _connector_matrix(
        n_dofs=n_dofs,
        dofs=dofs,
        value=stiffness,
        other_dofs=other_dofs,
        value_name="stiffness",
    )


def assemble_dof_dashpot(
    n_dofs: int,
    dofs: Sequence[int] | np.ndarray,
    damping,
    *,
    other_dofs: Sequence[int] | np.ndarray | None = None,
) -> FluxSparseMatrix:
    """
    Assemble a viscous dashpot matrix on selected DOFs.

    The matrix has the same topology as ``assemble_dof_spring`` but is intended
    for the damping matrix ``C`` in ``M u_ddot + C u_dot + K u = f``.
    """
    return _connector_matrix(
        n_dofs=n_dofs,
        dofs=dofs,
        value=damping,
        other_dofs=other_dofs,
        value_name="damping",
    )


__all__ = [
    "assemble_dof_dashpot",
    "assemble_dof_spring",
]

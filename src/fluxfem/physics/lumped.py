from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import jax.numpy as jnp

from ..solver.sparse import FluxSparseMatrix

MatrixBackend = Literal["jax", "scipy", "numpy"]
ArrayBackend = Literal["numpy", "jax"]


def _as_array_backend(array: np.ndarray, backend: ArrayBackend):
    if backend == "numpy":
        return array
    if backend == "jax":
        return jnp.asarray(array)
    raise ValueError("backend must be 'numpy' or 'jax'.")


def _sparse_from_coo(
    rows: Sequence[int],
    cols: Sequence[int],
    data: Sequence[float],
    n_dofs: int,
    *,
    backend: MatrixBackend,
):
    if backend == "jax":
        return FluxSparseMatrix(
            jnp.asarray(rows, dtype=jnp.int32),
            jnp.asarray(cols, dtype=jnp.int32),
            jnp.asarray(data, dtype=jnp.float64),
            int(n_dofs),
        ).coalesce()
    if backend == "scipy":
        try:
            import scipy.sparse as sp
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for backend='scipy'.") from exc
        return sp.csr_matrix((np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))), shape=(int(n_dofs), int(n_dofs)))
    if backend == "numpy":
        out = np.zeros((int(n_dofs), int(n_dofs)), dtype=float)
        np.add.at(out, (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)), np.asarray(data, dtype=float))
        return out
    raise ValueError("backend must be 'jax', 'scipy', or 'numpy'.")


def _as_dense_matrix(matrix, name: str) -> np.ndarray:
    if isinstance(matrix, FluxSparseMatrix):
        arr = np.asarray(matrix.to_dense(), dtype=float)
    else:
        arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return arr


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
    backend: MatrixBackend,
):
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

    return _sparse_from_coo(rows, cols, data, int(n_dofs), backend=backend)


def assemble_dof_spring(
    n_dofs: int,
    dofs: Sequence[int] | np.ndarray,
    stiffness,
    *,
    other_dofs: Sequence[int] | np.ndarray | None = None,
    backend: MatrixBackend = "jax",
):
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
        backend=backend,
    )


def assemble_dof_dashpot(
    n_dofs: int,
    dofs: Sequence[int] | np.ndarray,
    damping,
    *,
    other_dofs: Sequence[int] | np.ndarray | None = None,
    backend: MatrixBackend = "jax",
):
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
        backend=backend,
    )


def assemble_nodal_load(
    n_dofs: int,
    dofs: Sequence[int] | np.ndarray,
    values,
    *,
    backend: ArrayBackend = "numpy",
):
    """
    Assemble a dense nodal load vector on selected DOFs.

    ``values`` may be a scalar, a vector matching ``dofs``, or a column-like
    array. Duplicate DOFs are accumulated.
    """
    if n_dofs <= 0:
        raise ValueError("n_dofs must be positive.")
    selected = _as_dofs(dofs, "dofs")
    if np.any(selected >= n_dofs):
        raise ValueError("dofs contains an index outside n_dofs.")

    vals = np.asarray(values, dtype=float)
    if vals.ndim == 0:
        vals = np.full(selected.size, float(vals), dtype=float)
    else:
        vals = vals.reshape(-1)
        if vals.size != selected.size:
            raise ValueError("values length must match selected DOFs.")

    out = np.zeros(int(n_dofs), dtype=float)
    np.add.at(out, selected, vals)
    return _as_array_backend(out, backend)


def assemble_rayleigh_damping(
    mass,
    stiffness,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    backend: ArrayBackend = "numpy",
):
    """Return Rayleigh damping matrix ``C = alpha * M + beta * K``."""
    M = _as_dense_matrix(mass, "mass")
    K = _as_dense_matrix(stiffness, "stiffness")
    if M.shape != K.shape:
        raise ValueError("mass and stiffness must have the same shape.")
    return _as_array_backend(float(alpha) * M + float(beta) * K, backend)


def rayleigh_damping_ratio(omega, *, alpha: float, beta: float):
    """Evaluate ``zeta(omega) = 0.5 * (alpha / omega + beta * omega)``."""
    omega_arr = np.asarray(omega, dtype=float)
    if np.any(omega_arr <= 0.0):
        raise ValueError("omega must be positive.")
    return 0.5 * (float(alpha) / omega_arr + float(beta) * omega_arr)


def rayleigh_coefficients_from_modal_damping(
    omega1: float,
    zeta1: float,
    omega2: float,
    zeta2: float | None = None,
) -> tuple[float, float]:
    """
    Compute Rayleigh coefficients from damping ratios at two frequencies.

    If ``zeta2`` is omitted, the same damping ratio is used at ``omega2``.
    """
    w1 = float(omega1)
    w2 = float(omega2)
    z1 = float(zeta1)
    z2 = z1 if zeta2 is None else float(zeta2)
    if w1 <= 0.0 or w2 <= 0.0:
        raise ValueError("omega1 and omega2 must be positive.")
    if w1 == w2:
        raise ValueError("omega1 and omega2 must be distinct.")
    if z1 < 0.0 or z2 < 0.0:
        raise ValueError("damping ratios must be nonnegative.")

    system = np.array([[1.0 / w1, w1], [1.0 / w2, w2]], dtype=float)
    rhs = 2.0 * np.array([z1, z2], dtype=float)
    alpha, beta = np.linalg.solve(system, rhs)
    return float(alpha), float(beta)


__all__ = [
    "assemble_nodal_load",
    "assemble_rayleigh_damping",
    "assemble_dof_dashpot",
    "assemble_dof_spring",
    "rayleigh_coefficients_from_modal_damping",
    "rayleigh_damping_ratio",
]

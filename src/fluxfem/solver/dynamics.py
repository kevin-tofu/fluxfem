from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from .dirichlet import DirichletBC, split_dirichlet_matrix
from .solver import LinearSolver
from .sparse import FluxSparseMatrix

if TYPE_CHECKING:
    from jax import Array as JaxArray

    ArrayLike = np.ndarray | JaxArray
else:
    ArrayLike = np.ndarray

DirichletLike = DirichletBC | tuple[np.ndarray, np.ndarray]
ForceLike = ArrayLike | Callable[[float], ArrayLike]


@dataclass(frozen=True)
class NewmarkResult:
    """Time-history container for linear second-order dynamics."""

    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray


def _as_dense(matrix: FluxSparseMatrix | ArrayLike) -> np.ndarray:
    if isinstance(matrix, FluxSparseMatrix):
        return np.asarray(matrix.to_dense(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _parse_dirichlet(dirichlet: DirichletLike | None, n_dofs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dirichlet is None:
        return np.arange(n_dofs, dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    if isinstance(dirichlet, DirichletBC):
        dir_dofs, dir_vals = dirichlet.as_tuple()
    else:
        dir_dofs, dir_vals = dirichlet
    dir_dofs = np.asarray(dir_dofs, dtype=int)
    dir_vals = np.asarray(dir_vals, dtype=float)
    mask = np.ones(n_dofs, dtype=bool)
    mask[dir_dofs] = False
    free = np.nonzero(mask)[0]
    return free, dir_dofs, dir_vals


def _force_eval(force: ForceLike | None, time: float, n_dofs: int) -> np.ndarray:
    if force is None:
        return np.zeros(n_dofs, dtype=float)
    if callable(force):
        return np.asarray(force(time), dtype=float)
    arr = np.asarray(force, dtype=float)
    if arr.ndim == 1:
        return arr
    raise ValueError("force must be 1D array, callable(time)->1D array, or None.")


def newmark_solve_linear(
    M: FluxSparseMatrix | ArrayLike,
    C: FluxSparseMatrix | ArrayLike,
    K: FluxSparseMatrix | ArrayLike,
    *,
    u0: ArrayLike,
    v0: ArrayLike,
    dt: float,
    n_steps: int,
    force: ForceLike | None = None,
    dirichlet: DirichletLike | None = None,
    beta: float = 0.25,
    gamma: float = 0.5,
    linear_method: str = "spsolve",
) -> NewmarkResult:
    """
    Solve M u_ddot + C u_dot + K u = f(t) by Newmark-beta.

    Notes:
    - Dirichlet values are treated as time-invariant prescribed displacements.
    - Matrices are internally handled as dense arrays in this first implementation.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if beta <= 0.0:
        raise ValueError("beta must be > 0.")

    M_d = _as_dense(M)
    C_d = _as_dense(C)
    K_d = _as_dense(K)
    if M_d.shape[0] != M_d.shape[1]:
        raise ValueError("M must be square.")
    n_dofs = int(M_d.shape[0])

    u0 = np.asarray(u0, dtype=float).reshape(-1)
    v0 = np.asarray(v0, dtype=float).reshape(-1)
    if u0.shape[0] != n_dofs or v0.shape[0] != n_dofs:
        raise ValueError("u0 and v0 must match matrix size.")

    free, dir_dofs, dir_vals = _parse_dirichlet(dirichlet, n_dofs)

    M_ff = M_d[np.ix_(free, free)]
    C_ff = C_d[np.ix_(free, free)]
    K_ff = K_d[np.ix_(free, free)]

    if dir_dofs.size:
        _free_idx, _dir_idx, _M_ff_chk, M_fd = split_dirichlet_matrix(M_d, dir_dofs, n_total=n_dofs)
        _free_idx, _dir_idx, _C_ff_chk, C_fd = split_dirichlet_matrix(C_d, dir_dofs, n_total=n_dofs)
        _free_idx, _dir_idx, _K_ff_chk, K_fd = split_dirichlet_matrix(K_d, dir_dofs, n_total=n_dofs)
        if len(_free_idx) != len(free):
            raise RuntimeError("free DOF mismatch during Dirichlet split.")
    else:
        M_fd = np.zeros((free.size, 0), dtype=float)
        C_fd = np.zeros((free.size, 0), dtype=float)
        K_fd = np.zeros((free.size, 0), dtype=float)

    solver = LinearSolver(method=linear_method)

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)

    times = np.linspace(0.0, dt * n_steps, n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n_dofs), dtype=float)
    v_hist = np.zeros((n_steps + 1, n_dofs), dtype=float)
    a_hist = np.zeros((n_steps + 1, n_dofs), dtype=float)

    u_hist[0, :] = u0
    v_hist[0, :] = v0
    if dir_dofs.size:
        u_hist[:, dir_dofs] = dir_vals[None, :]
        u_hist[0, dir_dofs] = dir_vals
        v_hist[:, dir_dofs] = 0.0
        a_hist[:, dir_dofs] = 0.0

    f0 = _force_eval(force, times[0], n_dofs)
    rhs0 = f0[free] - K_fd @ dir_vals
    a_free0, _ = solver.solve(M_ff, rhs0 - C_ff @ v0[free] - K_ff @ u_hist[0, free])
    a_hist[0, free] = a_free0

    A_eff = K_ff + a0 * M_ff + a1 * C_ff

    for k in range(n_steps):
        t_np1 = times[k + 1]

        u_n = u_hist[k, free]
        v_n = v_hist[k, free]
        a_n = a_hist[k, free]

        u_pred = u_n + dt * v_n + dt * dt * (0.5 - beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        f_np1 = _force_eval(force, t_np1, n_dofs)
        rhs = f_np1[free] - K_fd @ dir_vals
        rhs = rhs + a0 * (M_ff @ u_pred) + C_ff @ (a1 * u_pred - v_pred)

        u_np1, _ = solver.solve(A_eff, rhs)
        a_np1 = a0 * (u_np1 - u_pred)
        v_np1 = v_pred + gamma * dt * a_np1

        u_hist[k + 1, free] = u_np1
        v_hist[k + 1, free] = v_np1
        a_hist[k + 1, free] = a_np1

    return NewmarkResult(t=times, u=u_hist, v=v_hist, a=a_hist)

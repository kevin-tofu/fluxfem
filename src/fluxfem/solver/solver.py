from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .cg import cg_solve, cg_solve_jax
from .newton import newton_solve
from ..core.solver import spdirect_solve_cpu, spdirect_solve_gpu
from .dirichlet import (
    condense_dirichlet_dense,
    condense_dirichlet_fluxsparse,
    expand_dirichlet_solution,
    enforce_dirichlet_dense,
    enforce_dirichlet_sparse,
)
from .sparse import FluxSparseMatrix
from ..core.space import FESpace


class LinearSolver:
    """
    Lightweight wrapper for solving linear systems with optional Dirichlet BCs.

    Supports dense arrays or FluxSparseMatrix and can either condense or enforce
    Dirichlet conditions before solving with the chosen backend.
    """

    def __init__(self, method: str = "spsolve", tol: float = 1e-8, maxiter: int = 200):
        self.method = method
        self.tol = tol
        self.maxiter = maxiter

    def _solve_free(self, A, b):
        if self.method == "cg":
            x, info = cg_solve_jax(A, b, tol=self.tol, maxiter=self.maxiter)
            return np.asarray(x), {"iters": info.get("iters"), "converged": info.get("converged", True)}
        elif self.method == "cg_custom":
            x, info = cg_solve(A, b, tol=self.tol, maxiter=self.maxiter)
            return np.asarray(x), {"iters": info.get("iters"), "converged": info.get("converged", True)}
        elif self.method == "spsolve":
            x = spdirect_solve_cpu(A, b)
            return np.asarray(x), {"iters": 1, "converged": True}
        elif self.method == "spsolve_jax":
            x = spdirect_solve_cpu(A, b, use_jax=True)
            return np.asarray(x), {"iters": 1, "converged": True}
        elif self.method == "spdirect_solve_gpu":
            x = spdirect_solve_gpu(A, b)
            return np.asarray(x), {"iters": 1, "converged": True}
        else:
            raise ValueError(f"Unknown linear method: {self.method}")

    def solve(
        self,
        A,
        b,
        *,
        dirichlet=None,
        dirichlet_mode: str = "condense",
        n_total: int | None = None,
    ):
        if dirichlet is None:
            return self._solve_free(A, b)

        if dirichlet_mode not in ("condense", "enforce"):
            raise ValueError("dirichlet_mode must be 'condense' or 'enforce'.")

        dir_dofs, dir_vals = dirichlet
        if dirichlet_mode == "enforce":
            if isinstance(A, FluxSparseMatrix):
                A_bc, b_bc = enforce_dirichlet_sparse(A, b, dir_dofs, dir_vals)
            else:
                A_bc, b_bc = enforce_dirichlet_dense(A, b, dir_dofs, dir_vals)
            return self._solve_free(A_bc, b_bc)

        if isinstance(A, FluxSparseMatrix):
            K_ff, F_free, free, dir_arr, dir_vals_arr = condense_dirichlet_fluxsparse(A, b, dir_dofs, dir_vals)
        else:
            K_ff, F_free, free, dir_arr, dir_vals_arr = condense_dirichlet_dense(A, b, dir_dofs, dir_vals)
        u_free, info = self._solve_free(K_ff, F_free)
        if n_total is not None:
            n_total_use = int(n_total)
        elif isinstance(A, FluxSparseMatrix):
            n_total_use = int(A.n_dofs)
        else:
            n_total_use = int(getattr(A, "shape", [0])[0])
        u_full = expand_dirichlet_solution(u_free, free, dir_arr, dir_vals_arr, n_total=n_total_use)
        return u_full, info


class NonlinearSolver:
    """
    Backward-compatible Newton-based nonlinear solver.

    This is a thin wrapper around ``newton_solve`` kept for legacy code paths.
    Prefer ``NonlinearAnalysis`` + ``NewtonSolveRunner`` for new workflows.
    """

    def __init__(
        self,
        space: FESpace,
        res_form,
        params,
        *,
        tol: float = 1e-8,
        maxiter: int = 20,
        linear_method: str = "spsolve",
        line_search: bool = False,
        max_ls: int = 10,
        ls_c: float = 1e-4,
        linear_tol: float | None = None,
        dirichlet=None,
        external_vector=None,
        linear_maxiter: int | None = None,
        jacobian_pattern=None,
    ):
        self.space = space
        self.res_form = res_form
        self.params = params
        self.tol = tol
        self.maxiter = maxiter
        self.linear_method = linear_method
        self.line_search = line_search
        self.max_ls = max_ls
        self.ls_c = ls_c
        self.linear_tol = linear_tol
        self.dirichlet = dirichlet
        self.external_vector = external_vector
        self.linear_maxiter = linear_maxiter
        self.jacobian_pattern = jacobian_pattern

    def solve(self, u0):
        return newton_solve(
            self.space,
            self.res_form,
            jnp.asarray(u0),
            self.params,
            tol=self.tol,
            maxiter=self.maxiter,
            linear_solver=self.linear_method,
            line_search=self.line_search,
            max_ls=self.max_ls,
            ls_c=self.ls_c,
            linear_tol=self.linear_tol,
            dirichlet=self.dirichlet,
            external_vector=self.external_vector,
            linear_maxiter=self.linear_maxiter,
            jacobian_pattern=self.jacobian_pattern,
        )

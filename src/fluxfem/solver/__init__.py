from .sparse import SparsityPattern, FluxSparseMatrix
from .dirichlet import (
    enforce_dirichlet_dense,
    enforce_dirichlet_sparse,
    free_dofs,
    condense_dirichlet_fluxsparse,
    condense_dirichlet_dense,
    expand_dirichlet_solution,
)
from .cg import cg_solve, cg_solve_jax
from .newton import newton_solve
from .solve_runner import (
    NonlinearAnalysis,
    NewtonLoopConfig,
    LoadStepResult,
    NewtonSolveRunner,
    solve_nonlinear,
    LinearAnalysis,
    LinearSolveConfig,
    LinearStepResult,
    LinearSolveRunner,
)
from .solver import LinearSolver, NonlinearSolver

__all__ = [
    "SparsityPattern",
    "FluxSparseMatrix",
    "enforce_dirichlet_dense",
    "enforce_dirichlet_sparse",
    "free_dofs",
    "condense_dirichlet_fluxsparse",
    "condense_dirichlet_dense",
    "expand_dirichlet_solution",
    "cg_solve",
    "cg_solve_jax",
    "newton_solve",
    "LinearAnalysis",
    "LinearSolveConfig",
    "LinearStepResult",
    "NonlinearAnalysis",
    "NewtonLoopConfig",
    "LoadStepResult",
    "NewtonSolveRunner",
    "solve_nonlinear",
    "LinearSolver",
    "NonlinearSolver",
]

from .sparse import (
    SparsityPattern,
    FluxSparseMatrix,
    coalesce_coo,
    concat_flux,
    block_diag_flux,
)
from .dirichlet import (
    enforce_dirichlet_dense,
    enforce_dirichlet_sparse,
    free_dofs,
    restrict_flux_to_free,
    condense_dirichlet_fluxsparse,
    condense_dirichlet_fluxsparse_coo,
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
from .petsc import petsc_solve, petsc_is_available

__all__ = [
    "SparsityPattern",
    "FluxSparseMatrix",
    "coalesce_coo",
    "concat_flux",
    "block_diag_flux",
    "enforce_dirichlet_dense",
    "enforce_dirichlet_sparse",
    "free_dofs",
    "restrict_flux_to_free",
    "condense_dirichlet_fluxsparse",
    "condense_dirichlet_fluxsparse_coo",
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
    "petsc_solve",
    "petsc_is_available",
]

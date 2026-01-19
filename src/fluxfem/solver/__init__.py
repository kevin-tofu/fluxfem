from .sparse import (
    SparsityPattern,
    FluxSparseMatrix,
    coalesce_coo,
    concat_flux,
    block_diag_flux,
)
from .dirichlet import (
    DirichletBC,
    enforce_dirichlet_dense,
    enforce_dirichlet_sparse,
    free_dofs,
    restrict_flux_to_free,
    condense_dirichlet_system,
    condense_dirichlet_fluxsparse,
    condense_dirichlet_fluxsparse_coo,
    condense_dirichlet_dense,
    expand_dirichlet_solution,
)
from .cg import cg_solve, cg_solve_jax, build_cg_operator, CGOperator
from .preconditioner import make_block_jacobi_preconditioner
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
from .petsc import petsc_solve, petsc_shell_solve, petsc_is_available

__all__ = [
    "SparsityPattern",
    "FluxSparseMatrix",
    "coalesce_coo",
    "concat_flux",
    "block_diag_flux",
    "DirichletBC",
    "enforce_dirichlet_dense",
    "enforce_dirichlet_sparse",
    "free_dofs",
    "restrict_flux_to_free",
    "condense_dirichlet_system",
    "condense_dirichlet_fluxsparse",
    "condense_dirichlet_fluxsparse_coo",
    "condense_dirichlet_dense",
    "expand_dirichlet_solution",
    "cg_solve",
    "cg_solve_jax",
    "build_cg_operator",
    "CGOperator",
    "make_block_jacobi_preconditioner",
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
    "petsc_shell_solve",
    "petsc_is_available",
]

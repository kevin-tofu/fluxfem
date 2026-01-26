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
    enforce_dirichlet_dense_jax,
    enforce_dirichlet_fluxsparse,
    enforce_dirichlet_sparse,
    free_dofs,
    split_dirichlet_matrix,
    restrict_flux_to_free,
    condense_dirichlet_system,
    enforce_dirichlet_system,
    condense_dirichlet_fluxsparse,
    condense_dirichlet_fluxsparse_coo,
    condense_dirichlet_dense,
    expand_dirichlet_solution,
)
from .cg import cg_solve, cg_solve_jax, build_cg_operator, CGOperator
from .preconditioner import make_block_jacobi_preconditioner
from .block_system import build_block_system, split_block_matrix, BlockSystem
from .block_matrix import FluxBlockMatrix, diag as block_diag, make as make_block_matrix
from .newton import newton_solve
from .result import SolverResult
from .history import NewtonIterRecord
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
    "enforce_dirichlet_dense_jax",
    "enforce_dirichlet_fluxsparse",
    "enforce_dirichlet_sparse",
    "split_dirichlet_matrix",
    "enforce_dirichlet_system",
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
    "build_block_system",
    "split_block_matrix",
    "BlockSystem",
    "FluxBlockMatrix",
    "block_diag",
    "make_block_matrix",
    "newton_solve",
    "SolverResult",
    "NewtonIterRecord",
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

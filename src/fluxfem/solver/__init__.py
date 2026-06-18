from .sparse import (
    SparsityPattern,
    FluxSparseMatrix,
    FluxSparseOperator,
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
from .newton_jax import newton_solve_jax, NewtonJaxResult
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
from .dynamics import NewmarkResult, newmark_solve_linear
from .coupled_system import CoupledSystem, CoupledSystemBuilder, DirichletSpec, ConstraintSpec
from .coupled_system_numpy import NumpyCoupledSystem, NumpyCoupledSystemBuilder
from .reduced_equation import (
    ReducedEquationBuilder,
    ReducedEquationField,
    ReducedEquationProblem,
    ReducedEquationSolveInfo,
    make_reduced_equation_newmark_residual,
    reduced_equation_newmark_step,
    solve_reduced_equation_active,
    solve_reduced_equation,
)
from .craig_bampton import (
    ActiveContactIterationRecord,
    ActiveContactNewmarkStepInfo,
    ActiveContactSolveInfo,
    ContactSearchManagerLike,
    CraigBamptonBasis,
    FrictionManagerLike,
    LinearConstraintSystem,
    NewmarkConfig,
    NewmarkState,
    NewmarkStepInfo,
    ProjectedReducedOperator,
    RBE3Patch,
    RBE3RemoteFixture,
    ReducedContactPair,
    ReducedContactPairAdapter,
    ReducedContactPairDofs,
    ReducedCoupledSystem,
    ReducedCoupledSystemBuilder,
    ReducedContactDynamics,
    ReducedLinearConstraintSystem,
    ReferencePointFixture,
    active_contact_fixed_point_solve,
    active_contact_newmark_step,
    assemble_reference_fixture_preload,
    complement_dofs,
    fixed_interface_modes,
    integrate_newmark,
    linear_constraint_system_from_reference_fixtures,
    make_craig_bampton_basis,
    make_newmark_effective_residual,
    newmark_kinematics,
    newmark_step,
    remote_reference_direction,
    remote_reference_size,
    retained_dofs_from_node_sets,
    rbe3_remote_reference_rank,
    reduced_jacobian_from_full,
    reduced_residual_from_full,
    solve_linear_constraint_kkt,
    solve_constraint_modes,
    validate_rbe3_remote_reference_rank,
    vector_dofs_from_nodes,
)

JAXCoupledSystem = CoupledSystem
JAXCoupledSystemBuilder = CoupledSystemBuilder


def make_coupled_system(K_u, F_u, *, backend: str | None = None):
    """Create a coupled system. ``backend=None`` auto-selects from the inputs."""
    return CoupledSystem.create(K_u, F_u, backend=backend)


def make_coupled_system_builder(K_u, F_u, *, backend: str | None = None):
    """Create a coupled-system builder. ``backend=None`` auto-selects from the inputs."""
    return CoupledSystemBuilder.create(K_u, F_u, backend=backend)


def make_jax_coupled_system(K_u, F_u):
    return CoupledSystem.from_structural(K_u, F_u)


def make_jax_coupled_system_builder(K_u, F_u):
    return CoupledSystemBuilder.from_structural(K_u, F_u)


def make_numpy_coupled_system(K_u, F_u):
    return NumpyCoupledSystem.from_structural(K_u, F_u)


def make_numpy_coupled_system_builder(K_u, F_u):
    return NumpyCoupledSystemBuilder.from_structural(K_u, F_u)

__all__ = [
    "SparsityPattern",
    "FluxSparseMatrix",
    "FluxSparseOperator",
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
    "newton_solve_jax",
    "NewtonJaxResult",
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
    "NewmarkResult",
    "newmark_solve_linear",
    "petsc_solve",
    "petsc_shell_solve",
    "petsc_is_available",
    "CoupledSystem",
    "CoupledSystemBuilder",
    "NumpyCoupledSystem",
    "NumpyCoupledSystemBuilder",
    "JAXCoupledSystem",
    "JAXCoupledSystemBuilder",
    "make_coupled_system",
    "make_coupled_system_builder",
    "make_jax_coupled_system",
    "make_jax_coupled_system_builder",
    "make_numpy_coupled_system",
    "make_numpy_coupled_system_builder",
    "DirichletSpec",
    "ConstraintSpec",
    "ActiveContactIterationRecord",
    "ActiveContactNewmarkStepInfo",
    "ActiveContactSolveInfo",
    "ContactSearchManagerLike",
    "CraigBamptonBasis",
    "FrictionManagerLike",
    "LinearConstraintSystem",
    "NewmarkConfig",
    "NewmarkState",
    "NewmarkStepInfo",
    "ProjectedReducedOperator",
    "RBE3Patch",
    "RBE3RemoteFixture",
    "ReducedContactPair",
    "ReducedContactPairAdapter",
    "ReducedContactPairDofs",
    "ReducedCoupledSystem",
    "ReducedCoupledSystemBuilder",
    "ReducedEquationBuilder",
    "ReducedEquationField",
    "ReducedEquationProblem",
    "ReducedEquationSolveInfo",
    "make_reduced_equation_newmark_residual",
    "reduced_equation_newmark_step",
    "solve_reduced_equation_active",
    "solve_reduced_equation",
    "ReducedContactDynamics",
    "ReducedLinearConstraintSystem",
    "ReferencePointFixture",
    "active_contact_fixed_point_solve",
    "active_contact_newmark_step",
    "assemble_reference_fixture_preload",
    "complement_dofs",
    "fixed_interface_modes",
    "integrate_newmark",
    "linear_constraint_system_from_reference_fixtures",
    "make_craig_bampton_basis",
    "make_newmark_effective_residual",
    "newmark_kinematics",
    "newmark_step",
    "remote_reference_direction",
    "remote_reference_size",
    "retained_dofs_from_node_sets",
    "rbe3_remote_reference_rank",
    "reduced_jacobian_from_full",
    "reduced_residual_from_full",
    "solve_linear_constraint_kkt",
    "solve_constraint_modes",
    "validate_rbe3_remote_reference_rank",
    "vector_dofs_from_nodes",
]

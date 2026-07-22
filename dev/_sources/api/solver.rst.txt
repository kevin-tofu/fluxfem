Solver
======

Solvers
-------

.. autoclass:: fluxfem.solver.LinearSolver
.. autoclass:: fluxfem.solver.NonlinearSolver

.. autoclass:: fluxfem.solver.LinearAnalysis
.. autoclass:: fluxfem.solver.NonlinearAnalysis
.. autoclass:: fluxfem.solver.LinearSolveRunner
.. autoclass:: fluxfem.solver.NewtonSolveRunner

.. autoclass:: fluxfem.solver.LinearSolveConfig
.. autoclass:: fluxfem.solver.NewtonLoopConfig

Coupled Systems
---------------

.. autoclass:: fluxfem.solver.CoupledSystem
.. autoclass:: fluxfem.solver.CoupledSystemBuilder
.. autoclass:: fluxfem.solver.DirichletSpec
.. autoclass:: fluxfem.solver.ConstraintSpec

Craig-Bampton ROM
-----------------

.. autoclass:: fluxfem.solver.CraigBamptonBasis
.. autoclass:: fluxfem.solver.ProjectedReducedOperator
.. autofunction:: fluxfem.solver.make_craig_bampton_basis
.. autofunction:: fluxfem.solver.solve_constraint_modes
.. autofunction:: fluxfem.solver.fixed_interface_modes
.. autofunction:: fluxfem.solver.reduced_residual_from_full
.. autofunction:: fluxfem.solver.reduced_jacobian_from_full

Craig-Bampton Constraints and Dynamics
--------------------------------------

.. autoclass:: fluxfem.solver.LinearConstraintSystem
.. autoclass:: fluxfem.solver.ReducedLinearConstraintSystem
.. autoclass:: fluxfem.solver.ReducedCoupledSystem
.. autoclass:: fluxfem.solver.ReducedCoupledSystemBuilder
.. autoclass:: fluxfem.solver.ReducedEquationField
.. autoclass:: fluxfem.solver.ReducedEquationProblem
.. autoclass:: fluxfem.solver.ReducedEquationSolveInfo
.. autoclass:: fluxfem.solver.ReducedEquationBuilder
.. autofunction:: fluxfem.solver.solve_reduced_equation
.. autofunction:: fluxfem.solver.solve_reduced_equation_active
.. autofunction:: fluxfem.solver.make_reduced_equation_newmark_residual
.. autofunction:: fluxfem.solver.reduced_equation_newmark_step
.. autofunction:: fluxfem.solver.reduced_equation_active_newmark_step
.. autoclass:: fluxfem.solver.RBE3Patch
.. autoclass:: fluxfem.solver.ReferencePointFixture
.. autoclass:: fluxfem.solver.RBE3RemoteFixture
.. autofunction:: fluxfem.solver.vector_dofs_from_nodes
.. autofunction:: fluxfem.solver.retained_dofs_from_node_sets
.. autofunction:: fluxfem.solver.remote_reference_size
.. autofunction:: fluxfem.solver.remote_reference_direction
.. autofunction:: fluxfem.solver.rbe3_remote_reference_rank
.. autofunction:: fluxfem.solver.validate_rbe3_remote_reference_rank
.. autofunction:: fluxfem.solver.make_average_rigid_body_constraint
.. autofunction:: fluxfem.solver.linear_constraint_system_from_reference_fixtures
.. autofunction:: fluxfem.solver.assemble_reference_fixture_preload
.. autofunction:: fluxfem.solver.solve_linear_constraint_kkt
.. autoclass:: fluxfem.solver.NewmarkState
.. autoclass:: fluxfem.solver.NewmarkConfig
.. autofunction:: fluxfem.solver.newmark_step
.. autofunction:: fluxfem.solver.integrate_newmark
.. autofunction:: fluxfem.solver.active_contact_fixed_point_solve
.. autofunction:: fluxfem.solver.active_contact_newmark_step

Sparse
------

.. autoclass:: fluxfem.solver.SparsityPattern
.. autoclass:: fluxfem.solver.FluxSparseMatrix
.. autoclass:: fluxfem.solver.FluxSparseOperator
   :no-index:

Dirichlet
---------

.. autofunction:: fluxfem.solver.enforce_dirichlet_dense
.. autofunction:: fluxfem.solver.enforce_dirichlet_sparse
.. autofunction:: fluxfem.solver.free_dofs
.. autofunction:: fluxfem.solver.condense_dirichlet_fluxsparse
.. autofunction:: fluxfem.solver.condense_dirichlet_dense
.. autofunction:: fluxfem.solver.expand_dirichlet_solution

Iterative
---------

.. autofunction:: fluxfem.solver.cg_solve
.. autofunction:: fluxfem.solver.cg_solve_jax

Block matrices
--------------

.. autofunction:: fluxfem.solver.block_diag
.. autofunction:: fluxfem.solver.make_block_matrix
.. autoclass:: fluxfem.solver.FluxBlockMatrix

Nonlinear
---------

.. autofunction:: fluxfem.solver.newton_solve
.. autofunction:: fluxfem.solver.solve_nonlinear

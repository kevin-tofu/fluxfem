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
.. autofunction:: fluxfem.solver.make_craig_bampton_basis
.. autofunction:: fluxfem.solver.solve_constraint_modes
.. autofunction:: fluxfem.solver.fixed_interface_modes
.. autofunction:: fluxfem.solver.reduced_residual_from_full
.. autofunction:: fluxfem.solver.reduced_jacobian_from_full

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

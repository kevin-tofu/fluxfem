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

Sparse
------

.. autoclass:: fluxfem.solver.SparsityPattern
.. autoclass:: fluxfem.solver.FluxSparseMatrix

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

Nonlinear
---------

.. autofunction:: fluxfem.solver.newton_solve
.. autofunction:: fluxfem.solver.solve_nonlinear

Solvers
=======

This section summarizes the linear and nonlinear solver APIs in FluxFEM and how
they are typically used in workflows.

.. contents::
   :local:
   :depth: 2


Linear solves
-------------

FluxFEM provides a lightweight ``LinearSolver`` wrapper that can solve either
dense arrays or ``FluxSparseMatrix`` objects, with optional Dirichlet handling.

Typical usage:

.. code-block:: python

   import fluxfem as ff

   # assemble K and F (e.g., weak-form or tensor-based assembly)
   solver = ff.LinearSolver(method="spsolve")
   u, info = solver.solve(
       K,
       F,
       dirichlet=(dir_dofs, dir_vals),
       dirichlet_mode="condense",  # or "enforce"
   )

Common options:

- ``method="spsolve"``: direct solve on CPU.
- ``method="cg"``: conjugate gradient (JAX-based).
- ``dirichlet_mode="condense"``: eliminate Dirichlet DOFs (smaller system).
- ``dirichlet_mode="enforce"``: enforce constraints in the full system.


Nonlinear solves (Newton)
-------------------------

For nonlinear problems, FluxFEM uses a Newton loop with configurable tolerances,
line search, and linear solver settings.

Example:

.. code-block:: python

   analysis = ff.NonlinearAnalysis(
       space=space,
       residual_form=ff.neo_hookean_residual_form,
       params={"lam": lam, "mu": mu},
       base_external_vector=F_ext,
       dirichlet=(dir_dofs, None),
       jacobian_pattern=ff.make_sparsity_pattern(space, with_idx=False),
       dtype=jnp.float64,
   )

   newton_cfg = ff.NewtonLoopConfig(
       tol=1e-6,
       atol=1e-10,
       maxiter=50,
       line_search=False,
       linear_solver="cg",
       linear_preconditioner="jacobi",
       linear_tol=None,
       n_steps=1,
   )
   runner = ff.NewtonSolveRunner(analysis, newton_cfg)
   u, history = runner.run(u0=jnp.zeros(space.n_dofs, dtype=jnp.float64))

Runner usage
^^^^^^^^^^^^

``NewtonSolveRunner`` orchestrates load steps and keeps a history of iterations.
Use it when you need load stepping, convergence history, or callbacks.

.. code-block:: python

   def on_newton_step(cb):
       # cb may include residual norms, iteration counters, and load step info.
       pass

   runner = ff.NewtonSolveRunner(analysis, newton_cfg)
   u, history = runner.run(u0=u0, newton_callback=on_newton_step)

The returned ``history`` is important for judging whether the analysis is
actually converging (residual norms, iteration counts, and load-step progress).
It also provides a structured signal for future automation workflows, including
AI/LLM agents that monitor and adapt solver behavior.

Typical fields include residual norms, iteration counters, and per-step summaries.
These are commonly used for early stopping, adaptive stepping, or logging/alerts.

Inputs overview
^^^^^^^^^^^^^^^

``NonlinearAnalysis`` expects the following core inputs:

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``space``
     - ``FESpace`` object defining the mesh, basis, and quadrature.
   * - ``residual_form``
     - Nonlinear residual kernel ``(ctx, u_elem, params) -> (n_q, n_ldofs)``.
   * - ``params``
     - Material/physics parameters (dict-like or custom object).
   * - ``base_external_vector``
     - External force vector (optional).
   * - ``dirichlet``
     - Dirichlet DOFs and values, e.g. ``(dir_dofs, dir_vals)``.
   * - ``jacobian_pattern``
     - Sparsity pattern (use ``make_sparsity_pattern(..., with_idx=False)``).
   * - ``dtype``
     - Computation dtype (e.g., ``jnp.float64``).

NewtonLoopConfig overview
^^^^^^^^^^^^^^^^^^^^^^^^^

Key configuration fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``tol`` / ``atol``
     - Converges when the residual norm falls below ``tol * ||R0||`` or ``atol``.
   * - ``maxiter``
     - Max Newton iterations per load step.
   * - ``line_search``
     - Enable/disable line search.
   * - ``linear_solver``
     - Backend for linear solves (e.g., ``"cg"``, ``"spsolve"``).
   * - ``linear_preconditioner``
     - Preconditioner for ``cg`` (e.g., ``"jacobi"``).
   * - ``linear_tol``
     - Optional tolerance override for linear solves.
   * - ``n_steps``
     - Number of load steps in the analysis.


Choosing a linear solver
------------------------

- ``spsolve``: robust for small-to-medium problems on CPU.
- ``cg``: scalable iterative solver; use preconditioners for better convergence.
- ``spdirect_solve_gpu``: direct solver when running on GPU (when available).


Dirichlet handling
------------------

Dirichlet constraints can be handled in two ways:

- **condense**: removes constrained DOFs from the system (smaller, faster).
- **enforce**: keeps the system size but overwrites rows/cols to impose constraints.

For large systems, ``condense`` is typically preferred.

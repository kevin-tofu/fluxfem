MixedForm and Block Utilities
=============================

This page shows the minimal usage of mixed formulations and block utilities in FluxFEM.

Mixed form (minimal)
^^^^^^^^^^^^^^^^^^^^

Define a mixed space, write per-field residuals, and solve in a single system:

.. code-block:: python

   import fluxfem as ff
   import fluxfem.helpers_wf as h_wf
   import numpy as np

   mesh = ff.StructuredHexBox(nx=2, ny=1, nz=1, lx=1.0, ly=0.1, lz=0.1).build()
   space = ff.make_hex_space(mesh, dim=1, intorder=2)
   mixed = ff.MixedFESpace({"u": space, "T": space})

   def res_T(v, T, p):
       return (p.kappa * h_wf.gaction(v, h_wf.grad(T)) - v * p.q) * h_wf.dOmega()

   def res_u(v, u, p):
       T_ref = ff.unknown_ref("T")
       return p.E * h_wf.gaction(v, h_wf.grad(u)) * h_wf.dOmega() - p.alpha * h_wf.gaction(v, T_ref.val) * h_wf.dOmega()

   residuals = ff.make_mixed_residuals(u=res_u, T=res_T)
   params = ff.Params(kappa=1.0, q=1.0, E=1.0, alpha=1.0e-3)

   bc = mixed.make_dirichlet(
       u=([0], [0.0]),
       T=([0], [0.0]),
   )

   u0 = np.zeros(mixed.n_dofs)
   problem = ff.MixedProblem(mixed, residuals, params=params)
   K = problem.assemble_jacobian(u0, return_flux_matrix=True)
   R0 = problem.assemble_residual(u0)
   b = -R0
   sol, _ = ff.LinearSolver(method="spsolve").solve(
       K, b, dirichlet=bc.as_dirichlet_bc(), dirichlet_mode="condense"
   )
   fields = mixed.unpack_fields(sol)

Block utilities (minimal)
^^^^^^^^^^^^^^^^^^^^^^^^^

Build a named block dictionary (for use with `MixedFESpace.build_block_system`):

.. code-block:: python

   import numpy as np
   from fluxfem import solver as ff_solver

   diag = ff_solver.block_diag(a=np.eye(2), b=2.0 * np.eye(2))
   blocks = ff_solver.make_block_matrix(
       diag=diag,
       sizes={"a": 2, "b": 2},
   )

   # blocks["a"]["a"], blocks["a"]["b"], blocks["b"]["a"], blocks["b"]["b"]

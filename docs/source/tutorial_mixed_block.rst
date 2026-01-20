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

Related mixed tutorials:

- `tutorials/thermoelastic_bar_1d_mixed.py`
- `tutorials/nitsche_contact_supermesh_api.py`

Mixed PDE view
--------------

The residual definitions above correspond to a coupled continuum system:

.. math::

   -\nabla\cdot(\kappa\,\nabla T) &= q, \\
   -\nabla\cdot(E\,\nabla u) &= \alpha\,\nabla\cdot T.

The temperature residual ``res_T`` mirrors :math:`\int_\Omega \kappa \nabla v\cdot\nabla T - v q`
and ``res_u`` mimics :math:`\int_\Omega E \nabla v\cdot\nabla u - \alpha \nabla v\cdot T`.
Because both weak forms live in the same ``MixedProblem``, FluxFEM assembles the full
coupled Jacobian and RHS, including the off-diagonal coupling between ``u`` and ``T``.

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

Off-diagonal coupling (K12 / K21)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For coupled/contact problems, you often have off-diagonal blocks. Use `rel` to
insert `K12` and (optionally) its transpose:

.. code-block:: python

   rel = {
       ("u", "p"): K_up,
   }
   blocks = ff_solver.make_block_matrix(
       diag=ff_solver.block_diag(u=K_uu, p=K_pp),
       rel=rel,
       sizes={"u": K_uu.shape[0], "p": K_pp.shape[0]},
       symmetric=True,
       transpose_rule="T",
   )

This is commonly used in contact or mixed formulations. See the following
tutorial scripts for full examples:

- `tutorials/nitsche_contact_supermesh_api.py`
- `tutorials/nitsche_contact_supermesh_demo_fluxfem.py`

Block matrix intuition
----------------------

The assembled Jacobian from the mixed residuals can be viewed as a block operator

.. math::

   \begin{bmatrix}
       K_{uu} & K_{uT} \\
       K_{Tu} & K_{TT}
   \end{bmatrix},

where the diagonal entries come from the self-residuals (``res_u`` / ``res_T``)
and the off-diagonal blocks arise from cross-couplings.  The block helpers above
let you construct this matrix explicitly: ``diag`` supplies ``K_{uu}``/``K_{TT}``
and ``rel`` inserts pairs such as ``K_{uT}``.  Passing the resulting ``blocks``
dictionary into ``MixedFESpace.build_block_system`` makes it easy to swap in
Schur-complement solvers, block preconditioners, or other multiphysics strategies.

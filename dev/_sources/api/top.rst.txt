Top-level API
=============

Core
----

.. autoclass:: fluxfem.FESpace
.. autoclass:: fluxfem.FormContext
   :no-index:
.. autoclass:: fluxfem.LinearForm
   :no-index:
.. autoclass:: fluxfem.BilinearForm
   :no-index:
.. autoclass:: fluxfem.ResidualForm
   :no-index:

.. autofunction:: fluxfem.make_hex_space
.. autofunction:: fluxfem.make_tet_space

Physics
-------

.. autofunction:: fluxfem.lame_parameters
.. autofunction:: fluxfem.isotropic_3d_D
.. autofunction:: fluxfem.linear_elasticity_form
.. autofunction:: fluxfem.diffusion_form
.. autofunction:: fluxfem.neo_hookean_residual_form

Solvers
-------

.. autoclass:: fluxfem.LinearSolver
   :no-index:
.. autoclass:: fluxfem.NonlinearSolver
   :no-index:
.. autoclass:: fluxfem.NonlinearAnalysis
   :no-index:
.. autoclass:: fluxfem.NewtonSolveRunner
   :no-index:

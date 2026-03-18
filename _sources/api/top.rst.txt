Top-level API
=============

Core
----

For role-explicit assembly and mixed/contact setup, prefer the ``*Spaces``
family listed here. Lower-level constructors remain available in the deeper API
reference, but they are no longer the preferred public entry points.

.. autoclass:: fluxfem.FESpace
.. autoclass:: fluxfem.FormContext
   :no-index:
.. autoclass:: fluxfem.LinearForm
   :no-index:
.. autoclass:: fluxfem.BilinearForm
   :no-index:
.. autoclass:: fluxfem.ResidualForm
   :no-index:
.. autoclass:: fluxfem.NamedSpace
   :no-index:
.. autoclass:: fluxfem.BilinearSpaces
   :no-index:
.. autoclass:: fluxfem.LinearSpaces
   :no-index:
.. autoclass:: fluxfem.ResidualSpaces
   :no-index:
.. autoclass:: fluxfem.JacobianSpaces
   :no-index:
.. autoclass:: fluxfem.MixedSpaces
   :no-index:
.. autoclass:: fluxfem.OneSidedContactSpaces
   :no-index:

.. autofunction:: fluxfem.make_hex_space
.. autofunction:: fluxfem.make_tet_space
.. autofunction:: fluxfem.assemble_linear_form
   :no-index:
.. autofunction:: fluxfem.assemble_bilinear_form
   :no-index:
.. autofunction:: fluxfem.assemble_residual
   :no-index:
.. autofunction:: fluxfem.assemble_jacobian
   :no-index:

Physics
-------

.. autofunction:: fluxfem.lame_parameters
.. autofunction:: fluxfem.isotropic_3d_D
.. autofunction:: fluxfem.linear_elasticity_form
.. autofunction:: fluxfem.diffusion_form
.. autofunction:: fluxfem.neo_hookean_residual_form

Solvers
-------

.. autoclass:: fluxfem.FluxSparseOperator
   :no-index:
.. autoclass:: fluxfem.LinearSolver
   :no-index:
.. autoclass:: fluxfem.NonlinearSolver
   :no-index:
.. autoclass:: fluxfem.NonlinearAnalysis
   :no-index:
.. autoclass:: fluxfem.NewtonSolveRunner
   :no-index:

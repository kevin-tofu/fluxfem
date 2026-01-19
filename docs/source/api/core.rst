Core
====

Spaces
------

.. autoclass:: fluxfem.core.FESpaceBase
.. autoclass:: fluxfem.core.FESpace
.. autoclass:: fluxfem.core.FESpacePytree
.. autoclass:: fluxfem.core.MixedFESpace
.. autoclass:: fluxfem.core.MixedProblem

.. autofunction:: fluxfem.core.make_space
.. autofunction:: fluxfem.core.make_space_pytree
.. autofunction:: fluxfem.core.make_hex_space
.. autofunction:: fluxfem.core.make_hex_space_pytree
.. autofunction:: fluxfem.core.make_tet_space
.. autofunction:: fluxfem.core.make_tet_space_pytree

Forms
-----

.. autoclass:: fluxfem.core.FormContext
.. autoclass:: fluxfem.core.MixedFormContext
.. autoclass:: fluxfem.core.VolumeContext
.. autoclass:: fluxfem.core.SurfaceContext

.. autoclass:: fluxfem.core.LinearForm
.. autoclass:: fluxfem.core.BilinearForm
.. autoclass:: fluxfem.core.ResidualForm
.. autoclass:: fluxfem.core.MixedWeakForm
.. autofunction:: fluxfem.core.make_mixed_residuals

Example (MixedProblem)
----------------------

.. code-block:: python

   mixed = ff.MixedFESpace({"u": space, "p": space})
   residuals = ff.make_mixed_residuals(u=res_u, p=res_p)
   params = ff.Params(alpha=1.2, beta=-0.4)
   pattern = mixed.get_sparsity_pattern(with_idx=True)
   problem = ff.MixedProblem(mixed, residuals, params=params, pattern=pattern)

   u0 = jnp.zeros(mixed.n_dofs)
   K = problem.assemble_jacobian(u0, return_flux_matrix=True)
   R = problem.assemble_residual(u0)

Assembly
--------

.. autofunction:: fluxfem.core.make_sparsity_pattern

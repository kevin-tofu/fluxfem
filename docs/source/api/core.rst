Core
====

Spaces
------

For new public-facing mixed setups, prefer ``MixedSpaces(...).to_fe_space()``.
``MixedFESpace`` remains available as a lower-level constructor for advanced
or internal usage.

.. autoclass:: fluxfem.core.FESpaceBase
.. autoclass:: fluxfem.core.FESpace
.. autoclass:: fluxfem.core.FESpacePytree
.. autoclass:: fluxfem.core.MixedFESpace
.. autoclass:: fluxfem.core.MixedProblem
.. autoclass:: fluxfem.core.MixedBlockSystem

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
.. autofunction:: fluxfem.core.bind_mixed_residual
.. autofunction:: fluxfem.core.kernel

Kernel metadata (ff.kernel)
---------------------------

``@ff.kernel`` attaches metadata used by ``space.assemble`` to infer the
form kind and domain. The following combinations are supported:

.. list-table::
   :header-rows: 1

   * - kind
     - domain
     - expected kernel signature
   * - bilinear
     - volume
     - ``(ctx, params) -> (n_q, n_ldofs, n_ldofs)``
   * - linear
     - volume
     - ``(ctx, params) -> (n_q, n_ldofs)``
   * - linear
     - surface
     - ``(ctx, params) -> (n_q, n_ldofs)``
   * - residual
     - volume
     - ``(ctx, u_elem, params) -> (n_q, n_ldofs)``
   * - jacobian
     - volume
     - ``(u_elem, ctx) -> (n_ldofs, n_ldofs)``

Use ``domain="surface"`` with a surface-specific assembly helper. For most workflows
use ``SurfaceMesh.assemble_linear_form_on_space(...)``; lower-level options include
``assemble_surface_linear_form`` and ``assemble_surface_bilinear_form`` when you need
explicit ``dim``/``n_total_nodes`` control.

Example (MixedProblem)
----------------------

.. code-block:: python

   mixed = ff.MixedSpaces(
       {
           "u": ff.NamedSpace("U", space),
           "p": ff.NamedSpace("P", space),
       }
   ).to_fe_space()
   residuals = ff.make_mixed_residuals(
       u=ff.bind_mixed_residual("u", res_u, space="U"),
       p=ff.bind_mixed_residual("p", res_p, space="P"),
   )
   params = ff.Params(alpha=1.2, beta=-0.4)
   pattern = mixed.get_sparsity_pattern(with_idx=True)
   problem = ff.MixedProblem(mixed, residuals, params=params, pattern=pattern)

   u0 = jnp.zeros(mixed.n_dofs)
   K = problem.assemble_jacobian(u0)
   R = problem.assemble_residual(u0)

Mixed naming convention:

- ``ctx.test`` / ``ctx.trial`` for simple single-space code
- ``ctx.bindings["u"]`` for named mixed-field lookup
- ``ctx.spaces["U"]`` for explicit space-key lookup

Example (MixedBlockSystem)
--------------------------

.. code-block:: python

   blocks = {
       "u": {"u": Kuu, "p": Kup},
       "p": {"u": Kpu, "p": Kpp},
   }
   rhs = {"u": Fu, "p": Fp}
   constraints = {"u": (u_dofs, u_vals)}
   system = mixed.build_block_system(blocks=blocks, rhs=rhs, constraints=constraints)

   u_free = solver.solve(system.K, system.F)
   u_full = system.expand(u_free)
   fields = system.split(u_full)

Assembly
--------

.. autofunction:: fluxfem.core.make_sparsity_pattern

Contact Interface Usage
=======================

This page shows the recommended high-level API for contact assembly.

One-To-Many Contact From Meshes
-------------------------------

Use :class:`fluxfem.OneToManyContactSurfaceSpace` with mesh objects and facet selectors.
You do not need to manually pass surface coordinates/connectivity.

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   import fluxfem as ff
   import fluxfem.helpers_wf as h_wf

   mesh_master = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
   mesh_s1 = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
   mesh_s2 = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()

   def select_contact(mesh):
       return mesh.facets_on_plane(axis=2, value=0.0)

   contact = ff.OneToManyContactSurfaceSpace.from_meshes(
       master_mesh=mesh_master,
       slave_meshes=[mesh_s1, mesh_s2],
       master_facet_selector=select_contact,
       slave_facet_selectors=select_contact,
       value_dim_master=3,
       value_dim_slaves=3,
       quad_order=1,
       backend="jax",
   )

   def bilin(v1, v2, u1, u2, p):
       ju = u1.val - u2.val
       return (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju)) * h_wf.ds()

   params = ff.Params(alpha=10.0, inv_h=1.0)
   n = mesh_master.coords.shape[0] * 3
   u = jnp.zeros(n)
   K = contact.assemble_bilinear(bilin, u, [u, u], params)

Nitsche-Style Jacobian Assembly
-------------------------------

After building contact space, call ``assemble_bilinear`` with your weak form.

.. code-block:: python

   K = contact.assemble_bilinear(bilin, u, [u, u], params, sparse=False)

Contact Coupling Matrices (Mortar Path)
---------------------------------------

Coupling matrices for constraints are available from:

.. code-block:: python

   M_aa, M_ab = contact.assemble_contact_coupling_matrices()

KKT Assembly, FluxSparse, and BCOO
----------------------------------

The default KKT output is ``FluxSparseMatrix``.

.. code-block:: python

   KKT_flux = contact.assemble_contact_kkt(
       rho=5.0,
       multiplier_space="p0",   # or "nodal"
       backend="numpy",
   )

   KKT_bcoo = KKT_flux.to_bcoo()

You can also request formats explicitly:

.. code-block:: python

   KKT_dense = contact.assemble_contact_kkt(
       rho=5.0,
       multiplier_space="p0",
       backend="jax",
       format="dense",
   )

   KKT_bcoo_direct = contact.assemble_contact_kkt(
       rho=5.0,
       multiplier_space="p0",
       backend="jax",
       format="bcoo",
   )

KKT Solve
---------

Solve with ``solve_contact_kkt``. For JAX, use ``backend="jax"``.

.. code-block:: python

   rhs = jnp.linspace(0.2, 1.0, int(KKT_dense.shape[0]))
   u = ff.solve_contact_kkt(KKT_dense, rhs, backend="jax", diagonal_shift=1e-2)

Notes
-----

- ``master_facet_selector`` and ``slave_facet_selectors`` are recommended for robust workflows.
- For oblique interfaces, provide custom selectors that return facet IDs based on your geometric rule.
- ``multiplier_space="p0"`` gives facet-wise constant multipliers (common for mortar-like constraints).

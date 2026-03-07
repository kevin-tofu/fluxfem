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

Penalty-Style Jacobian Assembly
-------------------------------

After building contact space, call ``assemble_bilinear`` with your weak form.

.. code-block:: python

   K = contact.assemble_bilinear(bilin, u, [u, u], params, sparse=False)

Contact Operators API (Recommended)
-----------------------------------

Use explicit APIs by family:

- ``law``: physical contact model.
- ``formulation``: multiplier-family vs penalty-family intent.
- ``assemble_contact_constraint_operators``: coupling/B/Kuu.
- ``assemble_contact_penalty_operators``: residual/jacobian from weak form.

.. code-block:: python

   # Constraint path: returns coupling/B/Kuu operators
   ops = ff.assemble_contact_constraint_operators(
       contact,
       rho=5.0,
       multiplier_space="p0",   # or "nodal"
       backend="numpy",
   )

   # Penalty-family path: returns residual/jacobian from your weak form
   ops_nitsche = ff.assemble_contact_penalty_operators(
       contact,
       weak_form=res_form,
       state={"master": u_master, "slaves": [u_s1, u_s2]},
       params=params,
       backend="jax",
   )

Contact Coupling Matrices (Constraint)
--------------------------------------

Coupling matrices for constraints are available from:

.. code-block:: python

   M_aa, M_ab = contact.assemble_contact_coupling_matrices()

KKT Assembly, FluxSparse, and BCOO
----------------------------------

Assemble KKT from coupling matrices (or from ``ops.coupling_*``).  
The default output is ``FluxSparseMatrix``.

.. code-block:: python

   KKT_flux = ff.assemble_contact_kkt(
       ops.coupling_aa,
       ops.coupling_ab,
       rho=5.0,
       multiplier_space="p0",
       facet_conn_master=ops.facet_conn_master,
       backend="numpy",
   )

   KKT_bcoo = KKT_flux.to_bcoo()

You can also request formats explicitly:

.. code-block:: python

   KKT_dense = ff.assemble_contact_kkt(
       ops.coupling_aa,
       ops.coupling_ab,
       rho=5.0,
       multiplier_space="p0",
       facet_conn_master=ops.facet_conn_master,
       backend="jax",
       format="dense",
   )

   KKT_bcoo_direct = ff.assemble_contact_kkt(
       ops.coupling_aa,
       ops.coupling_ab,
       rho=5.0,
       multiplier_space="p0",
       facet_conn_master=ops.facet_conn_master,
       backend="jax",
       format="bcoo",
   )

KKT Solve
---------

Solve with ``solve_contact_kkt``. For JAX, use ``backend="jax"``.

.. code-block:: python

   rhs = jnp.linspace(0.2, 1.0, int(KKT_dense.shape[0]))
   sol = ff.solve_contact_kkt(KKT_dense, rhs, backend="jax", diagonal_shift=1e-2)

Notes
-----

- ``master_facet_selector`` and ``slave_facet_selectors`` are recommended for robust workflows.
- For oblique interfaces, provide custom selectors that return facet IDs based on your geometric rule.
- ``multiplier_space="p0"`` gives facet-wise constant multipliers (common for constraint-family coupling).
- ``assemble_contact_kkt`` is a low-level API. In most cases, prefer ``CoupledSystemBuilder.add_contact(...)``.

CoupledSystemBuilder (Penalty)
------------------------------

To avoid manual offset/node bookkeeping, use ``CoupledSystemBuilder``:

.. code-block:: python

   builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
   builder.register_blocks([
       ("top", top_space, {"value_dim": 1}),
       ("support", support_space, {"value_dim": 1}),
   ])
   builder.add_contact(ops_nitsche, master="top", slave="support", value_dim=1)
   system = builder.build()
   u = system.solve(dirichlet_dofs=dir_dofs, dirichlet_vals=0.0, format="csr")

``register_field`` is also auto-offset by default:

.. code-block:: python

   builder.register_field("u", n_dofs=nu, value_dim=1)   # offset=0
   builder.register_field("v", n_dofs=nv, value_dim=1)   # offset=nu

Constraint with Builder
-----------------------

``assemble_contact_constraint_operators(...)`` and builder are consistent:
assemble operators first, then pass them to ``add_contact`` (or ``add_contact_mortar``).

.. code-block:: python

   ops_mortar = ff.assemble_contact_constraint_operators(contact, rho=1.0, multiplier_space="p0", backend="numpy")
   builder.add_contact(ops_mortar, master="top", slave="support", value_dim=1)

When ``ops_mortar`` comes from ``assemble_contact_constraint_operators(...)``,
``rho`` and ``multiplier_space`` are inherited automatically. Override only when needed.

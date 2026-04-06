Contact Interface Usage
=======================

This page shows the recommended high-level API for contact assembly.

For the broader split between plain surface forms, mixed surface residuals, and
contact-specific paths, see :doc:`surface_api_status`.

For terminology/ownership/scope (``contact``, ``multiplier``, ``formulation``, ``ops``),
see :doc:`contact_api_boundaries`.

Role Specs
----------

For new code, prefer explicit role specs over ad-hoc positional wiring.

Pair contact:

.. code-block:: python

   pair_spec = ff.ContactSpaces(master=master_side, slave=slave_side)
   contact = pair_spec.to_contact_surface_space(quad_order=1)

One-sided contact:

.. code-block:: python

   one_sided_spec = ff.OneSidedContactSpaces(side=slave_side)
   contact = one_sided_spec.to_contact_surface_space(quad_order=2)

One-to-many contact:

.. code-block:: python

   group_spec = ff.ContactGroupSpaces(master=master_side, slaves=[slave_side_1, slave_side_2])
   contact = group_spec.to_contact_surface_space(quad_order=1)

Backend Selection
-----------------

Contact entry points accept ``backend=None`` by default.

- ``backend=None`` auto-selects from the inputs
- if any relevant state/parameter/input is JAX-like, FluxFEM prefers ``jax``
- otherwise FluxFEM falls back to the API's default backend
- use ``backend="numpy"`` or ``backend="jax"`` only when you want an explicit
  override

One-To-Many Contact
-------------------

For new code, prefer explicit role specs and build the one-to-many contact space
from ``ContactGroupSpaces``.

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

   master_side = ff.ContactSide.from_facets(
       mesh_master,
       select_contact(mesh_master),
       value_dim=3,
   )
   slave_side_1 = ff.ContactSide.from_facets(
       mesh_s1,
       select_contact(mesh_s1),
       value_dim=3,
   )
   slave_side_2 = ff.ContactSide.from_facets(
       mesh_s2,
       select_contact(mesh_s2),
       value_dim=3,
   )

   contact = ff.ContactGroupSpaces(
       master=master_side,
       slaves=[slave_side_1, slave_side_2],
   ).to_contact_surface_space(
      quad_order=1,
   )

   def bilin(v1, v2, u1, u2, p):
       ju = u1.val - u2.val
       return (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju)) * h_wf.ds()

   params = ff.Params(alpha=10.0, inv_h=1.0)
   n = mesh_master.coords.shape[0] * 3
   u = jnp.zeros(n)
   K = contact.assemble_bilinear(bilin, u, [u, u], params)

Low-level constructors such as
``OneToManyContactSurfaceSpace.from_meshes(...)`` remain available for advanced
setups, but they are no longer the preferred public tutorial path.

Penalty-Style Jacobian Assembly
-------------------------------

After building contact space, call ``assemble_bilinear`` with your weak form.

.. code-block:: python

   K = contact.assemble_bilinear(bilin, u, [u, u], params)

Call ``K.to_dense()`` or ``np.asarray(K)`` only when you explicitly need a
dense matrix for debugging, dense linear algebra, or comparison code.

If you assemble directly from a compiled mixed surface residual form and still
want the standard pair Nitsche/penalty NumPy fast path, use:

.. code-block:: python

   res_form = ff.compile_tagged_pair_nitsche_penalty_residual(
       {"a": res_a, "b": res_b},
       backend="numpy",
   )
   J = contact.assemble_jacobian(res_form, {"a": u_a, "b": u_b}, params, backend="numpy")

Volume vs Contact API Style
---------------------------

The recommended split is:

- volume single-space shortcut: ``space.assemble_*``
- volume role-explicit path: ``ff.assemble_*(*Spaces(...), ...)``
- contact role-explicit path: ``ContactSpaces`` / ``ContactGroupSpaces`` / ``OneSidedContactSpaces``

This keeps the short path for simple problems while making master/slave intent
explicit when contact wiring matters.

Contact Operators API (Recommended)
-----------------------------------

Use explicit APIs by family:

- ``law``: physical contact model.
- ``formulation``: multiplier-family vs penalty-family intent.
- ``assemble_contact_constraint_operators``: coupling/B/Kuu.
- ``assemble_contact_penalty_operators``: residual/jacobian from weak form.

.. code-block:: python

   lm_space = ff.ContactMultiplierSpace.from_contact(
       contact,
       family="p0",
       side="master",
   )

   # Constraint path: returns coupling/B/Kuu operators
   ops: ff.ContactOperators = ff.assemble_contact_constraint_operators(
       contact,
       rho=5.0,
       multiplier=lm_space,
       # Optional: also evaluate and store residual/jacobian metadata on the same ContactOperators.
       weak_form=res_form,
       state={"master": u_master, "slaves": [u_s1, u_s2]},
       params=params,
   )

   # Penalty-family path: returns residual/jacobian from your weak form.
   # This is the recommended public entry when you want to assemble contact
   # operators and manage the nonlinear solve yourself.
   ops_nitsche: ff.ContactOperators = ff.assemble_contact_operators(
       contact,
       enforcement="penalty",
       weak_form=res_form,
       state={"master": u_master, "slaves": [u_s1, u_s2]},
       params=params,
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
       multiplier=lm_space,
       facet_conn_master=ops.facet_conn_master,
   )

   KKT_bcoo = KKT_flux.to_bcoo()

You can also request formats explicitly:

.. code-block:: python

   KKT_dense = ff.assemble_contact_kkt(
       ops.coupling_aa,
       ops.coupling_ab,
       rho=5.0,
       multiplier=lm_space,
       facet_conn_master=ops.facet_conn_master,
       format="dense",
   )

   # Explicit override example: request the JAX/BCOO path directly.
   KKT_bcoo_direct = ff.assemble_contact_kkt(
       ops.coupling_aa,
       ops.coupling_ab,
       rho=5.0,
       multiplier=lm_space,
       facet_conn_master=ops.facet_conn_master,
       backend="jax",
       format="bcoo",
   )

KKT Solve
---------

Solve with ``solve_contact_kkt``. Leave ``backend`` unset unless you need to
force a specific solver path.

.. code-block:: python

   rhs = jnp.linspace(0.2, 1.0, int(KKT_dense.shape[0]))
   sol = ff.solve_contact_kkt(KKT_dense, rhs, diagonal_shift=1e-2)

Notes
-----

- ``assemble_contact_operators(..., enforcement=...)`` is the recommended public entry when you want FluxFEM to assemble contact contributions and manage the solve yourself.
- ``solve_contact_penalty_jax(...)`` and ``solve_contact_al_jax(...)`` should be treated as convenience / experimental helpers for narrow JAX contact workflows, not as the main production solve path.
- ``master_facet_selector`` and ``slave_facet_selectors`` are recommended for robust workflows.
- For oblique interfaces, provide custom selectors that return facet IDs based on your geometric rule.
- ``ContactMultiplierSpace(family="p0")`` gives facet-wise constant multipliers (common for constraint-family coupling).
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

   lm_space = ff.ContactMultiplierSpace.from_contact(contact, family="p0", side="master")
   ops_mortar = ff.assemble_contact_constraint_operators(contact, rho=1.0, multiplier=lm_space)
   builder.add_contact(ops_mortar, master="top", slave="support", value_dim=1)

Mixed NumPy/JAX Inputs
----------------------

Mixed inputs are allowed. If a contact path receives JAX-traced values together
with NumPy arrays or Python scalars, FluxFEM prefers the JAX path.

.. code-block:: python

   rho = jnp.array(5.0)
   KKT = ff.assemble_contact_kkt(
       ops.coupling_aa,
       ops.coupling_ab,
       rho=rho,
       multiplier=lm_space,
       facet_conn_master=ops.facet_conn_master,
   )

When ``ops_mortar`` comes from ``assemble_contact_constraint_operators(...)``,
``rho`` and ``multiplier`` are inherited automatically. Override only when needed.

Embedding Map (With Unmapped Report)
------------------------------------

For mesh-to-mesh tied constraints, you can request unmapped slave node IDs:

.. code-block:: python

   emb, unmapped = ff.build_barycentric_embedding_map_from_meshes(
       master_mesh,
       slave_mesh,
       slave_facet_selector=lambda m: m.facets_on_plane(axis=2, value=0.0),
       allow_unmapped="skip",           # "error" | "skip"
       return_unmapped_ids=True,
   )
   print("unmapped slave node ids:", unmapped)

   builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
   builder.register_field("master", n_dofs=n_master, value_dim=1)
   builder.register_field("slave", n_dofs=n_slave, value_dim=1)
   builder.add_embedding_constraint(emb, master="master", slave="slave", value_dim=1)

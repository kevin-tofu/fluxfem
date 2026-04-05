Remote Constraints and Virtual Springs
======================================

This page summarizes the current high-level workflow for remote-point coupling
with ``CoupledSystemBuilder``:

- ``RBE2``: rigid kinematic coupling from a remote point to slave nodes.
- ``RBE3``: weighted interpolation from slave nodes to a remote point.
- virtual springs: translational/rotational support on the remote point.

These APIs are useful when you want to model fixtures, compliant mounts, remote
loading points, or reduced support stiffness without manually building KKT
offsets and constraint matrices.

Core APIs
---------

The main entry points are:

- ``ff.CoupledSystemBuilder.append_remote_point(...)``
- ``ff.CoupledSystemBuilder.add_rbe2_constraint(...)``
- ``ff.CoupledSystemBuilder.add_rbe3_constraint(...)``
- ``ff.CoupledSystemBuilder.add_dof_spring(...)``
- ``ff.CoupledSystemBuilder.add_remote_spring(...)``
- ``ff.build_rbe3_weights(...)``
- ``ff.build_rbe3_remote_resultant(...)``

Remote Point Layout
-------------------

By default, ``append_remote_point(...)`` creates a 6-DOF field ordered as:

.. code-block:: text

   [u_ref(3), omega_ref(3)]

with:

- ``u_ref``: remote-point translation
- ``omega_ref``: remote-point rotation

If you pass ``include_rotation=False``, the remote point becomes a translational
3-DOF field.

.. code-block:: python

   builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
   builder.register_field("u", n_dofs=space.n_dofs, value_dim=1, offset=0)
   builder.append_remote_point("remote", point=x_ref)

RBE2
----

Use ``RBE2`` when the slave surface should follow the remote point as a rigid
body. The expected layout is:

- master: 6 DOFs, ``[u_ref(3), omega_ref(3)]``
- slave: 3 translational DOFs per slave node

.. code-block:: python

   builder.add_rbe2_constraint(
       master="remote",
       slave="support_face",
       ref_point=x_ref,
       slave_coords=slave_coords,
   )

Conceptually, this enforces:

.. code-block:: text

   u_slave_i = u_ref + omega_ref x (x_i - x_ref)

Use ``RBE2`` when you want rigid transfer of both displacement and rotation to
the slave nodes.

RBE3
----

Use ``RBE3`` when the remote point should be interpolated from slave-node
motion with user-defined weights. The expected layout is the same as ``RBE2``:

- master: 6 DOFs, ``[u_ref(3), omega_ref(3)]``
- slave: 3 translational DOFs per slave node

.. code-block:: python

   weights = ff.build_rbe3_weights(
       x_ref,
       slave_coords,
       method="facet_area",
       surface=surface,
   )

   builder.add_rbe3_constraint(
       master="remote",
       slave="support_face",
       ref_point=x_ref,
       slave_coords=slave_coords,
       weights=weights,
   )

``RBE3`` does not rigidly lock the slave nodes. It constructs a weighted
reconstruction of remote translation and rotation from the slave side. This is
typically the better choice for compliant load transfer or distributed support.

RBE3 Weight Helpers
-------------------

``ff.build_rbe3_weights(...)`` currently supports:

- ``method="equal"``: uniform nodal weights
- ``method="distance"``: inverse-distance weighting from the remote point
- ``method="facet_area"``: facet-area lumping to nodes, then normalization

For fixture/support style use, ``method="facet_area"`` is the closest built-in
option to a uniform surface-pressure style distribution.

Virtual Springs
---------------

For arbitrary field DOFs, use ``add_dof_spring(...)``.

.. code-block:: python

   builder.add_dof_spring(
       "support_face",
       local_dofs=[0, 1, 2],
       stiffness=[1.0e3, 1.0e3, 1.0e3],
       reference_value=[0.0, 0.0, 0.0],
   )

For a remote point, prefer ``add_remote_spring(...)``:

.. code-block:: python

   builder.add_remote_spring(
       "remote",
       translational_stiffness=[1.0e4, 1.0e4, 1.0e4],
       rotational_stiffness=[1.0e6, 1.0e6, 1.0e6],
       translational_target=[0.0, 0.0, 0.0],
       rotational_target=[0.0, 0.0, 0.0],
   )

This adds a spring-to-ground contribution of the form:

.. code-block:: text

   F_spring = K_s (u - u_ref)

implemented as:

.. code-block:: text

   K += K_s
   F += K_s @ u_ref

where ``reference_value`` / ``*_target`` acts as the spring target
displacement or rotation.

Inferring Spring Stiffness from Force and Target
------------------------------------------------

``add_remote_spring(...)`` also accepts force plus target and infers diagonal
stiffness values from:

.. code-block:: text

   k = force / target

This is useful when you know the intended restoring force at a prescribed
virtual displacement.

.. code-block:: python

   builder.add_remote_spring(
       "remote",
       translational_force=[100.0, 0.0, 0.0],
       translational_target=[0.01, 1.0, 1.0],
   )

Notes:

- For each block, provide either ``*_stiffness`` or ``*_force``, not both.
- If ``target == 0`` and ``force != 0``, stiffness inference is rejected.
- If ``force == 0`` and ``target == 0``, the inferred stiffness is zero.

Equivalent Remote Resultant for RBE3
------------------------------------

When the slave surface represents a distributed load patch, you often need the
equivalent force and moment at the remote point. Use
``ff.build_rbe3_remote_resultant(...)`` for that conversion.

Vector surface load:

.. code-block:: python

   F_remote = ff.build_rbe3_remote_resultant(
       x_ref,
       slave_coords,
       surface=surface,
       load=[0.0, 0.0, -2.0],
   )

Pressure acting along surface normals:

.. code-block:: python

   F_remote = ff.build_rbe3_remote_resultant(
       x_ref,
       slave_coords,
       surface=surface,
       pressure=pressure_value,
       outward_from=inside_point,
   )

The returned 6-vector is ordered as:

.. code-block:: text

   [Fx, Fy, Fz, Mx, My, Mz]

You can pass it directly when creating the remote point:

.. code-block:: python

   builder.append_remote_point("remote", point=x_ref, F_block=F_remote)

Typical RBE3 + Spring Workflow
------------------------------

The following pattern is the recommended public workflow for a compliant remote
support:

.. code-block:: python

   builder = ff.CoupledSystemBuilder.from_structural(K, F)
   builder.register_field("u", n_dofs=space.n_dofs, value_dim=1, offset=0)
   builder.append_remote_point("remote", point=x_ref)
   builder.append_field("support_face", n_dofs=left_local_dofs.size, value_dim=1)

   C_face = np.zeros((left_local_dofs.size, space.n_dofs + left_local_dofs.size))
   for row, dof in enumerate(left_local_dofs):
       C_face[row, dof] = 1.0
       C_face[row, space.n_dofs + row] = -1.0

   builder.add_constraint_matrix_dof(C_face, master="u", slave="support_face")

   weights = ff.build_rbe3_weights(
       x_ref,
       slave_coords,
       method="facet_area",
       surface=surface,
   )

   builder.add_rbe3_constraint(
       master="remote",
       slave="support_face",
       ref_point=x_ref,
       slave_coords=slave_coords,
       weights=weights,
   )

   builder.add_remote_spring(
       "remote",
       translational_stiffness=[k, k, k],
       rotational_stiffness=[1.0e12, 1.0e12, 1.0e12],
   )

   u = builder.build().solve(format="csr", diagonal_shift=1e-8)

Reference Script
----------------

For a working example, see:

- ``tutorials/remote_rbe3_spring_compliance.py``

Current Scope
-------------

- ``RBE2`` and ``RBE3`` currently assume 3D translational slave nodes.
- The remote-point helpers are designed around 3-DOF or 6-DOF remote layouts.
- ``build_rbe3_weights(method="facet_area")`` gives an area-lumped nodal
  weighting, which is suitable for many practical fixture/load-transfer cases.
- ``build_rbe3_remote_resultant(...)`` converts distributed surface load or
  pressure into an equivalent remote-point force and moment, but it does not
  replace a full surface weak-form load model when that level of fidelity is
  required.

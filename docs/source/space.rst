FE Spaces
=========

FluxFEM provides two closely related space types:

- ``FESpace``: the default space for fixed meshes and standard solves.
- ``FESpacePytree``: a pytree-compatible space for JAX transformations.

When to use which
-----------------

Use ``FESpace`` for typical PDE solves where the mesh is fixed. Use
``FESpacePytree`` when the mesh coordinates or basis need to participate in
``jax.jit``/``jax.grad``/``jax.vmap`` workflows (e.g., shape optimization,
geometry inference, or coordinate warping).

Building pytree spaces
----------------------

The easiest entry point is the pytree helpers:

.. code-block:: python

   import fluxfem as ff

   mesh, _, _ = ff.load_gmsh_tet_mesh("mesh.msh")
   space = ff.make_tet_space_pytree(mesh, dim=3, intorder=2)

If you need to update coordinates inside a JAX function, construct a pytree
mesh and space explicitly:

.. code-block:: python

   def space_with_coords(coords):
       mesh_py = ff.TetMeshPytree(
           coords=coords,
           conn=mesh.conn,
           cell_tags=mesh.cell_tags,
           node_tags=mesh.node_tags,
       )
       return ff.FESpacePytree(
           mesh=mesh_py,
           basis=space.basis,
           elem_dofs=space.elem_dofs,
           value_dim=space.value_dim,
           _n_dofs=space._n_dofs,
           _n_ldofs=space._n_ldofs,
       )

Notes
-----

- ``FESpacePytree`` shares the same API as ``FESpace``; only the pytree behavior
  differs.
- Prefer ``FESpace`` unless you need the mesh/basis to be part of a JAX pytree.
- For same-space Galerkin problems, ``space.assemble_*`` remains the shortest
  entry point.
- For role-explicit assembly, prefer the top-level ``ff.assemble_*`` helpers
  with ``ff.NamedSpace`` and the ``*Spaces`` family, such as
  ``ff.LinearSpaces``, ``ff.BilinearSpaces``, ``ff.ResidualSpaces``, and
  ``ff.JacobianSpaces``.

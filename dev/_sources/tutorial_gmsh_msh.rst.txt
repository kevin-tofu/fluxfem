Load a Gmsh .msh and use it in analysis
=======================================

This tutorial shows the minimal flow to load a ``.msh`` created by Gmsh
and use it in FluxFEM analysis.

Key points
^^^^^^^^^^

- ``ff.load_gmsh_mesh`` supports both hex and tet meshes.
- Boundary conditions can be selected by the Physical Surface tag ID in Gmsh.
- If the ``.msh`` lacks boundary faces (quad/tri), they are inferred from geometry.

Prerequisites
^^^^^^^^^^^^^

- ``meshio`` is required (already listed in ``pyproject.toml`` dependencies).
- Tag the faces used in analysis as Physical Surface in Gmsh.

Prepare the ``.msh`` (Gmsh side)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1) Tag the faces used in analysis as Physical Surface.  
   Example: set ``dirichlet_tag=1`` for fixed boundary, ``traction_tag=2`` for load boundary.
2) Export the ``.msh`` (either v2 or v4 is OK).

These tag IDs are later used to select boundary conditions via ``facet_tags``.

Load in FluxFEM
^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   import fluxfem as ff
   import fluxfem.helpers_wf as h_wf

   mesh, facets, facet_tags = ff.load_gmsh_mesh("mesh.msh")

   # Switch space by tet/hex
   if isinstance(mesh, ff.TetMesh):
       space = ff.make_tet_space(mesh, dim=3, intorder=2)
   else:
       space = ff.make_hex_space(mesh, dim=3, intorder=2)

   # 3D linear elastic material matrix
   E = 210_000.0
   nu = 0.3
   D = ff.isotropic_3d_D(E, nu)

   bilinear_form = ff.BilinearForm.volume(
       lambda u, v, D_mat: h_wf.ddot(v.sym_grad, h_wf.matmul_std(D_mat, u.sym_grad))
       * h_wf.dOmega()
   )
   K = space.assemble(bilinear_form, params=D)

Boundary conditions (Physical Surface tags)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extract boundaries from ``facet_tags`` and set Dirichlet and Neumann conditions.

.. code-block:: python

   if facets is None or facet_tags is None:
       raise RuntimeError("Boundary faces or tags are not included in the .msh.")

   dirichlet_tag = 1
   traction_tag = 2

   facets_np = np.asarray(facets)
   tags_np = np.asarray(facet_tags)

   # Dirichlet: fix nodes on tagged faces
   dirichlet_nodes = np.unique(facets_np[tags_np == dirichlet_tag].reshape(-1))
   dir_dofs = mesh.node_dofs(dirichlet_nodes, components="xyz")

   # Neumann: surface traction on tagged faces
   neumann_facets = facets_np[tags_np == traction_tag]
   surf = ff.make_surface_from_facets(np.asarray(mesh.coords), neumann_facets)
   surface_form = ff.LinearForm.surface(lambda v, p: (v | p) * h_wf.ds())
   traction_vec = np.array([1.0, 0.0, 0.0], dtype=float)
   F = surf.assemble_linear_form_on_space(
       space, surface_form.get_compiled(), params=traction_vec
   )

   solver = ff.LinearSolver(method="spsolve")
   u, _ = solver.solve(K, F, dirichlet=(dir_dofs, None), dirichlet_mode="condense")

Fallback when tags are missing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the ``.msh`` has no Physical Surface tags, you can select boundaries by coordinate.
For example, fix ``x = xmin`` and apply traction on ``x = xmax``:

.. code-block:: python

   coords_np = np.asarray(mesh.coords)
   xmin = float(coords_np[:, 0].min())
   xmax = float(coords_np[:, 0].max())

   dir_dofs = mesh.boundary_dofs_where(
       lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
       components="xyz",
   )

   neumann_facets = np.asarray(
       mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
   )
   surf = ff.make_surface_from_facets(coords_np, neumann_facets)

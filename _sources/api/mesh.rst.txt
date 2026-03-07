Mesh
====

Meshes
------

.. autoclass:: fluxfem.mesh.BaseMesh
.. autoclass:: fluxfem.mesh.BaseMeshPytree

.. autoclass:: fluxfem.mesh.HexMesh
.. autoclass:: fluxfem.mesh.HexMeshPytree
.. autoclass:: fluxfem.mesh.StructuredHexBox
.. autofunction:: fluxfem.mesh.tag_axis_minmax_facets

.. autoclass:: fluxfem.mesh.TetMesh
.. autoclass:: fluxfem.mesh.TetMeshPytree
.. autoclass:: fluxfem.mesh.StructuredTetBox
.. autoclass:: fluxfem.mesh.StructuredTetTensorBox

Surface
-------

.. autoclass:: fluxfem.mesh.SurfaceMesh
.. autoclass:: fluxfem.mesh.SurfaceMeshPytree

IO
--

.. autofunction:: fluxfem.mesh.load_gmsh_mesh
.. autofunction:: fluxfem.mesh.load_gmsh_hex_mesh
.. autofunction:: fluxfem.mesh.load_gmsh_tet_mesh
.. autofunction:: fluxfem.mesh.make_surface_from_facets

Predicates
----------

.. autofunction:: fluxfem.mesh.bbox_predicate
.. autofunction:: fluxfem.mesh.plane_predicate
.. autofunction:: fluxfem.mesh.axis_plane_predicate
.. autofunction:: fluxfem.mesh.slab_predicate

Contact
-------

.. autoclass:: fluxfem.mesh.ContactSide
.. autoclass:: fluxfem.mesh.ContactSurfaceSpace
.. autoclass:: fluxfem.mesh.OneToManyContactSurfaceSpace
.. autoclass:: fluxfem.mesh.OneSidedContactSurfaceSpace
.. autoclass:: fluxfem.mesh.ContactOperators

.. autofunction:: fluxfem.mesh.assemble_contact_constraint_operators
.. autofunction:: fluxfem.mesh.assemble_contact_penalty_operators
.. autofunction:: fluxfem.mesh.assemble_contact_interface_residual
.. autofunction:: fluxfem.mesh.assemble_contact_interface_jacobian
.. autofunction:: fluxfem.mesh.assemble_contact_coupling_matrices
.. autofunction:: fluxfem.mesh.assemble_contact_kkt
.. autofunction:: fluxfem.mesh.solve_contact_kkt

.. autofunction:: fluxfem.mesh.facet_gap_values
.. autofunction:: fluxfem.mesh.active_contact_facets

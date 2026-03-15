Role-Explicit API Migration
===========================

This note summarizes the current public API direction for trial/test/unknown and
master/slave role binding.

Recommended API
---------------

For new code, prefer the explicit spec family:

- volume linear forms: ``ff.LinearSpaces(test=...)``
- volume bilinear forms: ``ff.BilinearSpaces(test=..., trial=...)``
- volume residuals: ``ff.ResidualSpaces(test=..., unknown=...)``
- volume Jacobians: ``ff.JacobianSpaces(test=..., trial=...)``
- mixed spaces: ``ff.MixedSpaces({...}).to_fe_space()``
- pair contact: ``ff.ContactSpaces(master=..., slave=...).to_contact_surface_space()``
- one-to-many contact: ``ff.ContactGroupSpaces(master=..., slaves=[...]).to_contact_surface_space()``
- one-sided contact: ``ff.OneSidedContactSpaces(side=...).to_contact_surface_space()``

These APIs make roles explicit without changing the underlying assembly model.

Still Supported
---------------

The short single-space entry points remain supported:

- ``space.assemble_linear_form(...)``
- ``space.assemble_bilinear_form(...)``
- ``space.assemble_residual(...)``
- ``space.assemble_jacobian(...)``

Use them when the problem is standard same-space Galerkin and no explicit role
binding is needed.

This also applies to selected performance benchmarks: some benchmark scripts
intentionally keep ``space.assemble_*`` or
``space.assemble_bilinear_linear_pair(...)`` as the measured target, even when
newer role-explicit setup is used elsewhere in the same repository.

Solver and Benchmark Direction
------------------------------

For new solver tests, mixed setup, and benchmark problem definitions, prefer:

- ``ff.MixedSpaces({...}).to_fe_space()``
- ``ff.assemble_bilinear_form(ff.BilinearSpaces(...), ...)``
- ``ff.assemble_linear_form(ff.LinearSpaces(...), ...)``
- ``ff.ContactSpaces(...)`` / ``ff.ContactGroupSpaces(...)`` / ``ff.OneSidedContactSpaces(...)``

The main exception is benchmark code whose purpose is to measure the
same-space shortcut API itself.

Deprecated Compatibility Paths
------------------------------

The following public compatibility paths are deprecated:

- ``assemble_bilinear_form_pg(...)``
- dict-based role passing such as ``{"test": V, "trial": U}``
- dict-based linear role passing such as ``{"test": V}``

Use the corresponding ``*Spaces`` objects instead.

Low-Level Constructors
----------------------

Low-level constructors are still available for advanced usage and internal code,
for example:

- ``MixedFESpace(...)``
- ``ContactSurfaceSpace.from_*``
- ``OneToManyContactSurfaceSpace.from_sides(...)``
- ``OneSidedContactSurfaceSpace.from_side(...)``

They are no longer the preferred tutorial path, but they are not being removed
immediately.

At this stage they also remain warning-free at runtime. The migration pressure
is currently applied through documentation and examples rather than constructor
warnings.

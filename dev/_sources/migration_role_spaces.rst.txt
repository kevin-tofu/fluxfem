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

For mixed problems, each field entry may be either a plain ``NamedSpace`` or a
same-space role spec such as ``ResidualSpaces(test=..., unknown=...)`` when you
want separate symbolic test/unknown names for that field.

Example:

.. code-block:: python

   mixed = ff.MixedSpaces(
       {
           "u": ff.ResidualSpaces(
               test=ff.NamedSpace("V", space),
               unknown=ff.NamedSpace("U", space),
           ),
           "p": ff.ResidualSpaces(
               test=ff.NamedSpace("Qv", space),
               unknown=ff.NamedSpace("Q", space),
           ),
       }
   ).to_fe_space()

Experimental Role-Aware Mixed Prototype
---------------------------------------

For field-internal distinct role spaces, there is now an experimental
prototype:

.. code-block:: python

   mixed = ff.MixedRoleSpaces(
       {
           "u": ff.ResidualSpaces(
               test=ff.NamedSpace("V", V_test),
               unknown=ff.NamedSpace("U", U_unknown),
           ),
       }
   ).to_fe_space()

This uses a separate role-aware layout model rather than the current
``MixedFESpace`` one-vector-per-field model.

Current prototype scope:

- raw residual callables
- raw Jacobian callables
- compiled mixed residual/Jacobian bindings with explicit ``space=...``
- volume assembly only

Not migrated yet:

- mixed surface ``ds()`` residuals and facet integrals
- contact surface assembly
- block-system helpers such as ``mixed.build_block_system(...)``
- broad tutorial coverage
- a stable public recommendation

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

NumPy Backend Scope
-------------------

The ``numpy`` backend is supported, but not uniformly across every new public
API path.

Supported today:

- same-space volume bilinear assembly
  ``space.assemble_bilinear_form(..., backend="numpy")``
- same-space volume linear assembly
  ``space.assemble_linear_form(..., backend="numpy")``
- same-space volume residual assembly
  ``space.assemble_residual(..., backend="numpy")``
- same-space role-explicit bilinear assembly
  ``ff.assemble_bilinear_form(ff.BilinearSpaces(test=V, trial=U), ..., backend="numpy")``
  when ``V.space is U.space``
- same-space role-explicit linear assembly
  ``ff.assemble_linear_form(ff.LinearSpaces(test=V), ..., backend="numpy")``
- same-space role-explicit residual assembly
  ``ff.assemble_residual(ff.ResidualSpaces(test=V, unknown=U), ..., backend="numpy")``
  when ``V.space is U.space``
- distinct-space role-explicit residual assembly
  ``ff.assemble_residual(ff.ResidualSpaces(test=V, unknown=U), ..., backend="numpy")``
  as a functional volume-only path
- selected same-space paired convenience paths such as
  ``space.assemble_bilinear_linear_pair(..., backend="numpy")``

Not guaranteed today:

- broad optimized coverage for distinct-space role-explicit bilinear assembly
  with ``backend="numpy"``
- named Jacobian assembly with ``backend="numpy"``
- full contact/surface coverage under ``backend="numpy"`` across every path

In practice:

- use ``backend="numpy"`` for same-space volume assembly and comparison/debug flows
- use ``backend="numpy"`` for distinct-space bilinear assembly when functional
  coverage is sufficient and performance is not the primary concern
- use ``backend="numpy"`` for distinct-space residual assembly when functional
  coverage is sufficient and no Jacobian path is required
- use ``backend="jax"`` for distinct-space, nonlinear role-explicit, and most
  contact-oriented assembly workflows

Contact Backend Direction
-------------------------

For contact, the long-term direction is to support both ``numpy`` and ``jax``
across the main public paths.

This is useful for different reasons:

- ``numpy`` is valuable as a lightweight reference/debug backend for geometry,
  pairing, sign conventions, and small reproducible checks
- ``jax`` is valuable for autodiff, Jacobian assembly, and nonlinear contact
  workflows

The intended rollout order is:

1. pair contact parity first
2. one-to-many contact next
3. one-sided contact next
4. full weak-form Jacobian parity later

Until that parity is in place, contact users should treat ``backend="jax"``
as the default safe path for role-explicit and nonlinear assembly, and use
``numpy`` mainly for validated comparison/debug routes.

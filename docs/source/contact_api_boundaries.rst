Contact API Boundaries
======================

This page fixes terminology, ownership, and scope for contact APIs.

Object Model
------------

- ``contact``:
  geometric and pairing description of the interface.
  It owns master/slave surfaces, trace pairing/supermesh, normals, and interface quadrature.
- ``multiplier``:
  discrete LM space definition for constraint-family assembly.
  It owns LM family (``nodal``/``p0``), side, value dimension, and optional facet connectivity.
- ``formulation``:
  enforcement/formulation intent used by assembly routing.
  Typical values are multiplier-family (mortar/KKT) and penalty-family (Nitsche-like).
- ``ops``:
  assembled operator bundle to pass into system builders.
  For constraint-family: coupling/B/Kuu (+ optional residual/jacobian metadata).
  For penalty-family: residual/jacobian (+ metadata).

Dependency Direction
--------------------

The intended dependency chain is:

.. code-block:: text

   contact -> multiplier -> formulation/assembly -> ops -> CoupledSystemBuilder

``contact`` and ``multiplier`` are defined separately, but are coupled at assembly time.

Axes: family / enforcement / formulation
----------------------------------------

Current public interpretation:

- ``family``: high-level route (``constraint`` or ``penalty``)
- ``enforcement``: concrete route (``mortar`` or ``nitsche``)
- ``formulation``: variant within the route
  (examples: ``multiplier`` / ``augmented_lagrangian`` / ``penalty_consistent``)

Rule: keep one meaning per axis and use the same terms in builder/docstrings/tutorials.

Configuration Scope
-------------------

Settings are scoped as follows:

- ``contact`` scope:
  geometry, pairing/supermesh, quadrature, backend and Jacobian assembly behavior.
- ``multiplier`` scope:
  LM discretization parameters (family/side/value_dim/facet_conn).
- ``formulation`` scope:
  enforcement-specific parameters and weak-form variant choices.
- ``builder.add_contact(...)`` call scope:
  routing choice and per-contact runtime options; each call can use different values.

For multiple contacts:

- Separate ``add_contact(...)`` calls can use different family/enforcement/formulation/multiplier choices.
- ``OneToManyContactSurfaceSpace`` groups multiple pair contacts under one contact object; settings passed when creating that object are shared by that object.

Known Constraint: ``p0`` Side
-----------------------------

``ContactMultiplierSpace(family="p0")`` currently supports ``side="master"`` only.
This is an implementation limitation in the current code path, not a mathematical requirement.

Use ``side="master"`` for now and treat ``side="slave"`` for ``p0`` as future work.

Guarantees Covered by Tests
---------------------------

Current tests explicitly target these guarantees:

- multiple contact contributions can be added to one coupled system,
- mortar/KKT assembly remains valid when different contacts have different lambda sizes,
- route consistency for penalty vs constraint contact operators.


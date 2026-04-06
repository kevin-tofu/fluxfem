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

Backend Direction
-----------------

The intended backend direction for contact is ``numpy``/``jax`` parity on the
main public paths.

The reason is split by use case:

- ``numpy`` is the preferred lightweight reference/debug backend
- ``jax`` is the preferred backend for autodiff, Jacobians, and nonlinear
  contact assembly

Implementation priority should follow this order:

1. pair contact
2. one-to-many contact
3. one-sided contact
4. full weak-form Jacobian parity

Until that work is complete, treat the auto-selected JAX path as the default
route for advanced contact assembly and use explicit ``backend="numpy"``
primarily where parity has already been validated.

Coordinate Differentiation
--------------------------

Coordinate differentiation is not yet a general guarantee for contact.

Current practical status:

- volume assembly already has working JAX-based examples of differentiation with
  respect to coordinates
- surface linear-form assembly also has direct coordinate-differentiation tests
- contact assembly still contains several NumPy-only geometry conversion paths,
  so geometry tracing is not yet robust across the main contact routes

In particular, the current contact interface code still relies on NumPy geometry
materialization in several places such as:

- ``np.asarray(surface_a.coords, dtype=float)``
- ``np.asarray(surface_b.coords, dtype=float)``
- facet-area and facet-shape helper loops that build NumPy arrays from
  quadrature-point geometry data

So the current rule should be:

- assume coordinate differentiation is available for JAX volume/surface paths
- do not assume it for contact until the geometry pipeline is made tracer-safe

The natural implementation order is:

1. remove ``np.asarray(...coords...)`` barriers from pair contact geometry prep
2. replace NumPy facet-shape/facet-area accumulation loops on the pair path
3. validate JAX coordinate gradients on pair contact before extending to
   one-to-many and one-sided paths

# Contact ROM API audit

## Short answer

The current ROM layer is useful for active contact prototyping, but it is not
yet a complete active-contact interface.

What works now:

- Contact can be represented as a full-space residual contribution.
- The residual can be composed with structural residuals.
- Craig-Bampton projection preserves retained contact DOFs as physical
  coordinates.
- Reduced residuals and Newmark effective residuals remain differentiable via
  `jax.jacrev`.

What is missing for a library-level active-contact API:

- A typed contact candidate model.
- Active/inactive state storage and update policy.
- Surface/node helpers for selecting retained contact DOFs.
- Contact residual objects with inspectable gap/traction outputs.
- A solver loop that can update active sets or contact history per Newton/time
  step.

## Current library shape

- `core`: FE spaces, basis, weak forms, assembly, ROM.
- `core.contact`: contact DOF helpers, kinematics, active-state snapshots,
  penalty contact laws, friction history helpers, and quadrature contact
  prototypes.
- `mesh`: volume meshes, surface meshes, predicates.
- `physics`: reusable constitutive and operator forms.
- `solver`: sparse matrices, Dirichlet handling, Newton/load stepping, linear
  solve runners, surface load assembly.
- `tools`: JIT wrappers, timers, visualization.

This is a reasonable split. ROM currently lives in `core` because it is mostly a
linear algebra/projection layer. If contact grows beyond prototypes, it should
probably become either:

- `fluxfem.physics.contact` for residual laws, or
- `fluxfem.contact` for candidate search, active sets, and history management.

## Active contact readiness

The current `make_unilateral_plane_contact_residual` is a residual factory:

`u -> R_contact(u)`

That makes it easy to compose with:

`R_full(u) = R_structural(u) + R_contact(u)`

and then reduce:

`R_rom(q) = Phi.T R_full(Phi q)`.

This is enough for:

- node-to-rigid-plane penalty contact,
- smoothed active/inactive transition,
- AD tangents through contact,
- reduced Newmark dynamics with contact forces.

It is not yet enough for:

- node-to-surface or surface-to-surface contact,
- active pair search,
- frictional contact with stick/slip history,
- Lagrange multiplier contact,
- active-set Newton strategies,
- contact diagnostics such as gap, pressure, active mask, and penetration energy.

## API usability issues

1. `retained_dofs` is direct and precise, but inconvenient.
   Users will usually want `retained_dofs_from_nodes(nodes, dim)` or
   `retained_dofs_from_surface(surface, dim)`.

2. The contact residual factory has no object identity.
   A dataclass like `PlaneContact` would be easier to inspect, reuse, and extend:
   `contact.residual(u)`, `contact.gap(u)`, `contact.active_mask(u)`.

3. `NewmarkState` and `NewmarkStepInfo` are local ROM records.
   They are fine for now, but should eventually align with `SolverResult` /
   history conventions.

4. The ROM builder is dense.
   Good for first experiments, but large meshes need sparse solves/eigensolvers
   for fixed-interface modes and constraint modes.

5. `newmark_step` has no line search.
   For hard nonsmooth contact, an active-set aware loop or line search will be
   needed.

6. Surface load assembly is mostly NumPy-side, while volume assembly is JAX-side.
   That is acceptable for loads, but contact residuals that need AD should stay
   in JAX arrays and avoid NumPy in the residual path.

## Recommended next API layer

Add small helper types before implementing complex contact:

```python
@dataclass(frozen=True)
class ContactKinematics:
    dofs: Array
    normals: Array
    gaps0: Array

    def displacements(self, u): ...
    def gaps(self, u): ...


@dataclass(frozen=True)
class PlanePenaltyContact:
    kinematics: ContactKinematics
    penalty: float
    smoothing: float = 0.0

    def residual(self, u): ...
    def active_mask(self, u): ...
    def pressure(self, u): ...
```

Add DOF helpers:

```python
vector_dofs_from_nodes(nodes, dim)
retained_dofs_from_surface(surface, dim)
```

Then active contact can be introduced as a small stateful layer:

```python
@dataclass(frozen=True)
class ActiveContactState:
    active: Array
    previous_gap: Array | None = None
    friction_history: Array | None = None
```

This keeps AD residuals pure while allowing the outer time/Newton loop to update
state explicitly.

## Suggested implementation order

1. Add an active-set contact residual option.
2. Add FE tutorial using `SurfaceMesh` to select retained DOFs.
3. Add sparse CB builder later.

## Follow-up implemented

- Added `vector_dofs_from_nodes(nodes, dim)`.
- Added `retained_dofs_from_surface(surface, dim)`.
- Added `ContactKinematics`.
- Added `PlanePenaltyContact` with `gaps`, `penetration`, `active_mask`,
  `pressure`, `residual`, `penetration_energy`, `force_norm`, and
  `active_count`.
- Kept `make_unilateral_plane_contact_residual(...)` as compatibility sugar.
- Added `ActiveContactState` and `update_active_contact_state(...)`.
- Added `PlanePenaltyContact.residual_with_state(state)` for frozen active-set
  Newton/Newmark experiments.
- Added `plane_contact_kinematics_from_surface(surface, dim, normal,
  plane_offset)` for FE surface candidate construction.
- Added `PairedContactKinematics`, `PairedPenaltyContact`, and
  `paired_contact_kinematics_from_surfaces(...)` for fixed-normal nearest-node
  slave/master contact prototypes.
- Added `NodeSurfaceContactKinematics`, `NodeSurfacePenaltyContact`, and
  `node_surface_contact_kinematics_from_surfaces(...)` for fixed-normal
  slave-node to master-facet prototypes with facet shape weights.
- `node_surface_contact_kinematics_from_surfaces(..., displacement=u)` can now
  update facet pairing and shape weights on the deformed geometry while keeping
  the resulting kinematics frozen for AD residual evaluation.
- Node-surface facet selection now uses closest projected point distance on
  each candidate facet instead of facet-centroid distance.
- `node_surface_contact_kinematics_from_surfaces(..., normal=None)` now computes
  frozen normals from the selected master facet, using deformed facet geometry
  when `displacement` is provided.
- Added `active_contact_fixed_point_solve(...)` as a solver-independent outer
  loop for frozen contact state updates.
- Added `active_contact_newmark_step(...)` for one implicit Newmark step with
  active contact updates around the inner Newton solve.
- Added `residual_with_state(...)` and `state_from_displacement(...)` to paired
  and node-surface penalty contacts, so plane/paired/node-surface contacts now
  share the same frozen-active-set interface.
- Added the same contact diagnostics to paired and node-surface penalty
  contacts.
- Added `ContactUpdateSnapshot` so active contact callbacks can carry a frozen
  contact object plus active state without defining ad hoc tutorial-local
  dataclasses.
- `ContactUpdateSnapshot` now carries optional `history` for future friction or
  path-dependent contact laws. History is preserved but not included in
  `changed(...)` by default.
- Added `orthonormal_tangent_basis(normals)` as the geometry helper for
  tangential friction/history prototypes.
- Added `TangentialPenaltyHistory`,
  `update_tangential_penalty_history(...)`, `slip_norm(...)`, and
  `stick_count(...)` as the first friction-history layer. This currently
  updates and diagnoses clipped tangential penalty forces.
- Added `friction_residual_from_history(...)` and `make_friction_residual(...)`
  to scatter stored tangential friction forces for plane, paired, and
  node-surface contacts.
- Added `tutorials/craig_bampton_friction_history_rom.py` to show normal
  contact, frozen friction residuals, and explicit history updates in a reduced
  Newmark loop.
- Added `SurfaceQuadratureContactKinematics`,
  `SurfaceQuadraturePenaltyContact`, and
  `surface_quadrature_contact_kinematics_from_surfaces(...)` as a first
  quadrature-point contact prototype with centroid and vertex rules.
- Added `tutorials/craig_bampton_surface_quadrature_active_newmark.py` to show
  quadrature contact snapshots in an active reduced Newmark step.
- Added `tutorials/craig_bampton_node_surface_active_newmark.py` to exercise CB
  projection, node-surface contact, deformed contact-weight updates, and active
  Newmark in one workflow.
- Split the contact implementation from `core.rom` into `core.contact` while
  preserving public exports from `fluxfem.core` and `fluxfem`.
- Split the test coverage into `src/tests/test_rom.py` and
  `src/tests/test_contact.py` so ROM/Newmark tests and contact API tests can be
  run and triaged independently.
- Added section markers inside `core.contact` to make the large prototype file
  easier to scan before any future module split.
- Added `ContactSearchCache` and
  `contact_search_cache_from_kinematics(...)`.
- Node-surface and surface-quadrature contact kinematics now expose
  `master_facet_ids` and a `.search_cache()` convenience method.
- `node_surface_contact_kinematics_from_surfaces(...)` and
  `surface_quadrature_contact_kinematics_from_surfaces(...)` now accept
  `search_cache=...`, which freezes selected master facets but still rebuilds
  interpolation weights, gaps, and automatic normals from the current geometry.
- The active Newmark node-surface and surface-quadrature tutorials now use this
  cache path and report cached master facet ids.
- Added `ContactCandidateSet` and `candidate_facet_ids=...` support for
  node-surface and surface-quadrature contact builders. This gives callers a
  manual broad-phase pruning hook before a full automatic spatial search is
  added.
- `search_cache` is the stricter exact-pairing API and takes precedence over
  candidate pruning when both are supplied.
- Added `contact_candidate_set_from_bounding_boxes(...)` to generate a
  `ContactCandidateSet` from slave/master AABBs and `search_radius`.
- The AABB helper supports displaced geometry via `displacement=...` and is now
  used by `tutorials/node_surface_contact_demo.py`.
- Extended `ContactCandidateSet` to support per-contact candidate segments.
- Added `contact_candidate_set_from_per_contact(...)` for manual variable-length
  per-contact candidates.
- Added `node_surface_candidate_set_from_bounding_boxes(...)` for per-slave-node
  AABB candidate generation, including displaced geometry support.
- Added `ContactNeighborList` and
  `node_surface_neighbor_list_from_bounding_boxes(...)` for Verlet-style
  broad-phase candidate reuse with `search_radius + skin` and a max-drift
  refresh criterion.
- Added `ContactAABBIndex` and `contact_aabb_index_from_surface(...)` as a
  reusable uniform-grid index over master facet AABBs.
- Added `node_surface_candidate_set_from_aabb_index(...)` and
  `node_surface_neighbor_list_from_aabb_index(...)` so node-surface contact can
  build per-node candidates from the index instead of scanning all facets.
- Added `surface_quadrature_candidate_set_from_aabb_index(...)` and
  `surface_quadrature_neighbor_list_from_aabb_index(...)` so surface-quadrature
  contact can also use indexed per-quadrature-point candidates.
- Added `NodeSurfaceContactSearchManager` and
  `make_node_surface_contact_search_manager(...)` to own index rebuilds,
  neighbor-list refresh, exact search-cache updates, and node-surface penalty
  contact construction.
- Added `SurfaceQuadratureContactSearchManager` and
  `make_surface_quadrature_contact_search_manager(...)` with the same explicit
  `(contact, next_manager)` search-state update style for surface-quadrature
  penalty contact.
- Added `FrictionalContactUpdateSnapshot` and
  `TangentialPenaltyFrictionManager` so frozen tangential friction history can
  be composed with frozen active normal-contact residuals through an explicit
  snapshot/advance workflow.
- Added a composed node-surface search plus friction-history workflow test and
  tutorial, validating that the active contact API can be assembled from small
  explicit state managers instead of tutorial-local bookkeeping.
- Added an AD regression for a reduced objective with frozen active contact and
  frozen friction history, checking `Phi.T J_full Phi` consistency and a
  finite-difference gradient spot check.
- Fixed surface-quadrature friction residual scattering to apply quadrature
  weights before slave/master distribution.
- Added a composed surface-quadrature search plus friction-history workflow
  test and active Newmark tutorial.
- Added `ReducedContactDynamics` as a public facade for CB-reduced active
  contact dynamics with optional friction history.
- Updated the node-surface and surface-quadrature friction active Newmark
  tutorials to use the facade instead of tutorial-local callback closures.
- Added a facade-vs-manual callback equivalence test for a node-surface
  friction active Newmark step.
- Added a full-order vs CB-ROM active penalty-contact benchmark regression with
  mode-count convergence checks.
- Added `craig_bampton_full_order_contact_benchmark.py` to print full-order vs
  CB-ROM contact displacement error over `n_modes`.
- Added a closed-form 1D obstacle penalty-contact reference test and tutorial.
- Added an AD regression for a reduced dynamic Newmark contact/friction
  objective with frozen active contact and friction history.
- Reduced node-surface and surface-quadrature contact search-manager
  duplication by sharing validation, AABB-index rebuild, refresh decision, and
  candidate/cache selection helpers.
- Added public `ContactSearchManagerLike` and `FrictionManagerLike` protocols
  and runtime validation in `ReducedContactDynamics`.
- Added a surface-quadrature full-order vs CB-ROM reference test and tutorial.
- Added independent weighted-penalty surface-quadrature contact reference tests
  that assemble `B_q`, residual, and tangent outside the contact class,
  including a multi-facet mixed-active patch, a 3D quad surface-pair, and a
  search-manager refresh workflow.
- Added an independent surface-quadrature friction-history/scatter reference
  test covering stick/slip clipping, quadrature-weighted scatter, and the
  frozen-friction zero Jacobian.
- Added a `ReducedContactDynamics` surface-quadrature friction
  facade-vs-manual callback equivalence test for the public manager interface.
- Added `notes/contact_rom_api_guide.md` as the compact user-facing guide for
  the current CB contact ROM API.
- Added `notes/contact_independent_reference_pr_summary.md` as the merge/PR
  summary for this validation branch.

## Verification snapshot

- `PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton_surface_quadrature_friction_active_newmark.py`
  completed successfully.
- `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py -k "independent_penalty_form or independent_weighted_penalty_form"`
  passed with 3 tests.
- `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py -k "independent_reference or independent_penalty_form or independent_weighted_penalty_form"`
  passed with 4 tests.
- `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py -k "friction or independent_reference"`
  passed with 7 tests.
- `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "reduced_contact_dynamics"`
  passed with 4 tests.
- `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
  passed with 65 tests.
- `python -m py_compile` passed for the ROM/contact modules, public package
  exports, and all current contact tutorials.
- Top-level API smoke check passed for:
  - `CraigBamptonBasis`
  - `ContactSearchManagerLike`
  - `FrictionManagerLike`
  - `ReducedContactDynamics`
  - `ContactCandidateSet`
  - `ContactSearchCache`
  - `contact_candidate_set_from_bounding_boxes`
  - `contact_candidate_set_from_per_contact`
  - `node_surface_candidate_set_from_bounding_boxes`
  - `ContactNeighborList`
  - `node_surface_neighbor_list_from_bounding_boxes`
  - `ContactAABBIndex`
  - `contact_aabb_index_from_surface`
  - `node_surface_candidate_set_from_aabb_index`
  - `node_surface_neighbor_list_from_aabb_index`
  - `surface_quadrature_candidate_set_from_aabb_index`
  - `surface_quadrature_neighbor_list_from_aabb_index`
  - `NodeSurfaceContactSearchManager`
  - `make_node_surface_contact_search_manager`
  - `SurfaceQuadratureContactSearchManager`
  - `make_surface_quadrature_contact_search_manager`
  - `FrictionalContactUpdateSnapshot`
  - `TangentialPenaltyFrictionManager`
  - `PlanePenaltyContact`
  - `NodeSurfacePenaltyContact`
  - `SurfaceQuadraturePenaltyContact`
  - `TangentialPenaltyHistory`
  - `active_contact_newmark_step`
  - `contact_search_cache_from_kinematics`
  - `make_friction_residual`
- All current tutorials completed successfully:
  - `craig_bampton_contact_rom.py`
  - `craig_bampton_1d_obstacle_contact_reference.py`
  - `craig_bampton_full_order_contact_benchmark.py`
  - `craig_bampton_surface_contact_reference.py`
  - `craig_bampton_fe_surface_contact.py`
  - `node_surface_contact_demo.py`
  - `paired_surface_contact_demo.py`
  - `active_contact_outer_loop_demo.py`
  - `craig_bampton_node_surface_active_newmark.py`
  - `craig_bampton_friction_history_rom.py`
  - `craig_bampton_node_surface_friction_active_newmark.py`
  - `craig_bampton_surface_quadrature_active_newmark.py`
  - `craig_bampton_surface_quadrature_friction_active_newmark.py`
- The current JAX install emits CUDA/cuSPARSE plugin warnings in this
  environment, but all smoke/tutorial commands above exited successfully on CPU.

# Craig-Bampton ROM implementation notes

## Goal

Add a reduced-order modeling path that keeps contact candidate boundary DOFs as
physical retained coordinates while reducing the interior with fixed-interface
modes. This should make dynamic contact experiments cheaper without hiding the
gap/contact variables behind modal coordinates.

## Current first slice

- Added `fluxfem.core.rom`.
- `make_craig_bampton_basis(K, M, retained_dofs, n_modes)` builds a dense CB
  basis with reduced coordinates ordered as retained physical DOFs first, then
  internal modes.
- `CraigBamptonBasis.expand(q)` maps reduced coordinates to full DOFs.
- `CraigBamptonBasis.project_vector(r)` maps full residuals to reduced residuals.
- `CraigBamptonBasis.project_matrix(A)` maps full matrices to reduced matrices.
- `reduced_residual_from_full(cb, residual_fn)` keeps the full residual function
  intact and wraps it as `Phi.T @ residual_fn(Phi @ q)`.
- `reduced_jacobian_from_full(cb, residual_fn)` uses `jax.jacrev` on the reduced
  residual, so nonlinear/contact terms can remain differentiable.

## Current second slice

- Added reduced-coordinate Newmark-beta dynamics:
  - `NewmarkState(q, qd, qdd, t)`
  - `NewmarkConfig(dt, beta, gamma, tol, atol, maxiter)`
  - `make_newmark_effective_residual(...)`
  - `newmark_step(...)`
  - `integrate_newmark(...)`
- The implicit step solves
  `G(q_next) = Mr a_next + Cr v_next + Rr(q_next) - Fr_next = 0`.
- `Rr(q)` is a callable, so structural ROM residuals and contact residuals can
  be composed outside the time integrator.
- The effective tangent is formed with `jax.jacrev(G)`, preserving the AD path
  through nonlinear reduced/contact forces.

## Current third slice

- Added a first contact residual prototype:
  `make_unilateral_plane_contact_residual(...)`.
- It returns a full-space residual for frictionless penalty contact against a
  rigid plane:
  `gap = gap0 + dot(u_contact, normal)`,
  `R_contact = -penalty * max(-gap, 0) * normal`.
- Optional `smoothing > 0` uses softplus for a differentiable active/inactive
  transition.
- Added `compose_residuals(...)` so structural and contact residual terms can be
  combined before CB projection.

## Current fourth slice

- Added `tutorials/craig_bampton_contact_rom.py`.
- The tutorial uses a small spring-chain model to demonstrate the full workflow:
  dense CB basis, retained boundary DOF, plane contact residual, reduced
  internal force, projected external force, and implicit Newmark stepping.

## Current fifth slice

- Added `tutorials/craig_bampton_fe_surface_contact.py`.
- The tutorial builds a small 3D Hex FE space, assembles linear elasticity
  stiffness and mass, selects the +x `SurfaceMesh`, derives retained DOFs and
  plane-contact kinematics from that surface, and verifies reduced residual and
  AD Jacobian construction.

## Current sixth slice

- Added `tutorials/paired_surface_contact_demo.py`.
- The tutorial builds two toy surfaces, constructs nearest-node slave/master
  paired contact kinematics, evaluates paired penalty residuals, and checks that
  the AD Jacobian is available.

## Current seventh slice

- Added `tutorials/node_surface_contact_demo.py`.
- The tutorial builds one slave point and one master quad facet, constructs
  fixed-normal node-to-surface kinematics, distributes the reaction over master
  facet nodes with shape-function weights, and verifies force balance/Jacobian
  shape.
- Node-to-surface kinematics now accepts `displacement=u_current` for outer-loop
  active/contact update of facet pairing and shape-function weights.

## Current eighth slice

- Added `active_contact_fixed_point_solve(...)`.
- The outer loop freezes contact state, calls an inner solver, updates contact
  state from the new solution, and repeats until the active/contact state stops
  changing.
- Added `tutorials/active_contact_outer_loop_demo.py`.

## Current ninth slice

- Added `active_contact_newmark_step(...)`.
- This wraps one implicit Newmark step with the same active-contact outer loop:
  contact state is frozen, the Newmark/Newton solve is run, contact state is
  refreshed from the converged displacement, and the loop repeats until the
  active set/pairing/weights stop changing.
- The API is callback-based:
  `internal_force_from_contact_state(state)` builds the frozen reduced residual,
  and `update_contact_state(q_next)` refreshes the state. For CB ROMs, those
  callbacks can expand `u = cb.expand(q)` and project full contact forces with
  `cb.project_vector(...)`.
- This keeps AD local to each frozen-contact residual while making active-set or
  closest-point updates explicit outer-loop operations.

## Current tenth slice

- Added `residual_with_state(...)` and `state_from_displacement(...)` to
  `PairedPenaltyContact` and `NodeSurfacePenaltyContact`, matching the plane
  contact API.
- Added `ContactUpdateSnapshot`, a small library-level snapshot that stores a
  frozen contact object plus its `ActiveContactState`.
- `ContactUpdateSnapshot.changed(...)` compares active masks and contact
  kinematics, including node-surface master DOF maps, weights, normals, and
  reference gaps.
- Added `tutorials/craig_bampton_node_surface_active_newmark.py`.
- The tutorial combines:
  - retained contact-surface DOFs,
  - internal fixed-interface modes,
  - node-to-surface penalty contact,
  - deformed-geometry contact-weight refresh,
  - `active_contact_newmark_step(...)`.
- The contact snapshot stores both the frozen active state and the frozen
  node-surface kinematics, so the AD residual sees a fixed contact topology
  during each inner Newton solve.

## Current eleventh slice

- Updated `node_surface_contact_kinematics_from_surfaces(...)` to select master
  facets by the distance to the closest projected point on each facet, instead
  of by facet-centroid distance.
- The selected facet still produces frozen shape-function weights for the AD
  residual path.
- Added a regression test where centroid selection would choose the wrong
  nearby facet but projected-point selection chooses the long facet that the
  slave node is actually closest to.

## Current twelfth slice

- `node_surface_contact_kinematics_from_surfaces(...)` now accepts
  `normal=None`.
- With `normal=None`, normals are computed from the selected master facets and
  frozen into the returned kinematics:
  - 1D: `[1]`
  - 2D line facets: left normal from the edge orientation
  - 3D triangle/quad facets: cross-product normal from the facet ordering
- When `displacement` is provided, automatic normals are computed from the
  deformed master facet geometry. This lets active-contact snapshots update
  pairing, shape weights, and normals together while keeping each inner AD
  residual frozen.
- The CB node-surface active Newmark tutorial now uses automatic normals.

## Current thirteenth slice

- Added shared contact diagnostics to `PlanePenaltyContact`,
  `PairedPenaltyContact`, and `NodeSurfacePenaltyContact`:
  - `penetration_energy(u)`
  - `force_norm(u)`
  - `active_count(u)`
- `penetration_energy(u)` is a compact scalar diagnostic based on
  `0.5 * penalty * sum(penetration**2)`. With smoothing enabled this uses the
  smoothed penetration value and should be treated as a comparison diagnostic,
  not a rigorously integrated smooth potential.
- The CB node-surface active Newmark tutorial now prints active count, contact
  energy, and contact force norm.

## Current fourteenth slice

- Added optional `history` to `ContactUpdateSnapshot`.
- `ContactUpdateSnapshot.from_contact(..., history=...)` carries arbitrary
  contact-law state such as stick/slip flags, accumulated slip, previous
  tangential traction, or continuation parameters.
- `ContactUpdateSnapshot.with_history(history)` returns a copy with the same
  frozen contact object and active state but updated history.
- `ContactUpdateSnapshot.changed(...)` intentionally does not compare history.
  Active-set/pairing/kinematics convergence remains separate from material or
  friction history updates. A future friction law should update and validate
  history explicitly.
- History is outer-loop state; it is carried alongside frozen residuals and is
  not differentiated unless a specific contact law chooses to use differentiable
  history fields in its residual.

## Current fifteenth slice

- Added `orthonormal_tangent_basis(normals)`.
- Input shape is `(n_contact, dim)` and output shape is
  `(n_contact, dim - 1, dim)`.
- For `dim == 1`, the returned basis has zero tangent directions. For 2D and
  3D, tangent vectors are orthonormal and orthogonal to the normalized contact
  normal.
- This is the common geometry helper needed for tangential penalty friction,
  stick/slip history, and slip diagnostics.

## Current sixteenth slice

- Added `TangentialPenaltyHistory`.
  - `tangential_slip`: `(n_contact, dim - 1)`
  - `stick`: `(n_contact,)`
  - `friction_force`: `(n_contact, dim)`
- Added `update_tangential_penalty_history(...)`.
- The updater computes relative contact displacement increments, projects them
  to the tangent basis, accumulates trial tangential slip, forms a tangential
  penalty force, and clips the force by `mu * pressure`.
- Added `slip_norm(history)` and `stick_count(history)` diagnostics.
- This is intentionally history/diagnostics only. Friction residual assembly is
  the next layer, because force scatter differs between plane, paired, and
  node-surface contact.

## Current seventeenth slice

- Added `friction_residual_from_history(contact, history)`.
- Added `make_friction_residual(contact, history)`.
- The stored `history.friction_force` is now scattered consistently for:
  - plane contact: force at contact DOFs
  - paired contact: slave force plus master reaction
  - node-surface contact: slave force plus weighted master-facet reaction
- `make_friction_residual(...)` returns a frozen residual contribution suitable
  for `compose_residuals(...)`; update `history` explicitly between solves or
  outer contact iterations.

## Current eighteenth slice

- Added `tutorials/craig_bampton_friction_history_rom.py`.
- The tutorial demonstrates a small 2D Craig-Bampton ROM with:
  - retained contact-node DOFs,
  - normal plane penalty contact,
  - frozen tangential friction residual from previous history,
  - explicit `update_tangential_penalty_history(...)` after each Newmark step.
- This follows the intended history design: friction history is carried between
  time steps and is not part of active-set convergence checks by default.

## Current nineteenth slice

- Added `SurfaceQuadratureContactKinematics`.
- Added `SurfaceQuadraturePenaltyContact`.
- Added `surface_quadrature_contact_kinematics_from_surfaces(...)`.
- The prototype supports `quadrature_rule="centroid"` and
  `quadrature_rule="vertices"`:
  - centroid gives one contact point per slave facet,
  - vertices gives one contact point per slave facet node, so a quad facet
    gives four contact points.
- The residual multiplies normal penalty force by quadrature weights, distributes
  slave forces by slave interpolation weights, and distributes master reactions
  by master facet shape weights.
- This is a prototype quadrature contact path, not a mortar formulation.

## Current twentieth slice

- Added `tutorials/craig_bampton_surface_quadrature_active_newmark.py`.
- The tutorial demonstrates:
  - retained slave/master contact-surface DOFs,
  - internal fixed-interface modes,
  - `SurfaceQuadraturePenaltyContact`,
  - `quadrature_rule="vertices"`,
  - active-contact Newmark updates with frozen quadrature contact snapshots.
- The example reaches active contact at both quadrature points and reports
  quadrature gaps, active count, contact energy, force norm, and quadrature
  weights.

## Current twenty-first slice

- Split contact-related code out of `src/fluxfem/core/rom.py` into
  `src/fluxfem/core/contact.py`.
- `rom.py` now focuses on Craig-Bampton basis construction, reduced residual
  projection, Newmark dynamics, and active-contact outer-loop orchestration.
- `contact.py` now owns contact DOF helpers, kinematics builders, penalty
  contact laws, active contact state snapshots, friction history helpers, and
  quadrature contact prototypes.
- Public imports are preserved through `fluxfem.core` and top-level `fluxfem`.
  Existing tutorials/tests still use `import fluxfem as ff`.

## Current twenty-second slice

- Split tests by responsibility:
  - `src/tests/test_rom.py` now covers Craig-Bampton basis construction,
    reduced residual/Jacobian projection, Newmark dynamics, active-contact outer
    loop wrappers, and ROM/contact composition.
  - `src/tests/test_contact.py` now covers contact kinematics, penalty contact
    residuals, active snapshots, friction history/residual helpers, and
    quadrature contact.
- This keeps failures easier to triage as the contact API grows independently
  from the ROM/Newmark layer.

## Current twenty-third slice

- Added internal section markers to `src/fluxfem/core/contact.py`.
- The file is now organized into:
  - DOF helpers
  - geometry helpers
  - kinematics builders
  - projection and quadrature helpers
  - residual composition and active-update comparison helpers
  - kinematics dataclasses
  - penalty contact laws
  - active contact state and snapshots
  - tangential friction history and residual helpers
  - compatibility factories
- No public API or behavior changed in this slice.

## Current twenty-fourth slice

- Ran a consistency check across the current ROM/contact split.
- Test results:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 28 tests.
  - `python -m py_compile` passed for `core.rom`, `core.contact`, package
    exports, and all current Craig-Bampton/contact tutorials.
  - Top-level `import fluxfem as ff` smoke check found the expected ROM,
    contact, active-loop, and friction-history API symbols.
- Tutorial smoke results:
  - `craig_bampton_contact_rom.py`: reduced dynamic contact example completed.
  - `craig_bampton_fe_surface_contact.py`: FE surface retained-DOF setup and
    reduced residual/Jacobian probe completed.
  - `node_surface_contact_demo.py`: node-to-surface residual, action-reaction,
    and deformed-weight update completed.
  - `paired_surface_contact_demo.py`: paired-surface contact residual and
    action-reaction check completed.
  - `active_contact_outer_loop_demo.py`: frozen active-set outer loop converged
    in two iterations.
  - `craig_bampton_node_surface_active_newmark.py`: CB node-surface active
    Newmark step completed with deformed master weights and automatic normal.
  - `craig_bampton_friction_history_rom.py`: reduced Newmark loop with frozen
    normal contact and explicit tangential history completed.
  - `craig_bampton_surface_quadrature_active_newmark.py`: quadrature contact
    active Newmark example completed with both quadrature points active.
- The local JAX environment still emits CUDA/cuSPARSE plugin warnings before
  falling back to CPU. These warnings did not cause failures in this check.

## Current twenty-fifth slice

- Added `ContactSearchCache` and
  `contact_search_cache_from_kinematics(...)`.
- Node-surface and surface-quadrature kinematics now carry selected
  `master_facet_ids`, and can rebuild from `search_cache=...`.
- With a search cache, the selected master facet is frozen while shape weights,
  reference gaps, and automatic normals are recomputed from the current
  displaced geometry. This gives active-contact/Newmark loops a stable pairing
  option without making residual evaluation stateful.
- `ContactUpdateSnapshot.changed(...)` now also compares contact search ids, so
  topology changes are visible when a builder is allowed to re-search.
- Updated the node-surface and surface-quadrature active Newmark tutorials to
  reuse a contact search cache and report the cached master facet ids.
- Added tests showing:
  - node-surface cached pairing does not switch facets after displacement,
  - surface-quadrature cached pairing does not switch facets after displacement,
  - invalid cache shapes and facet ids are rejected.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 31 tests.
  - `python -m py_compile` passed for the updated modules/tutorials.
  - top-level API smoke passed for `ContactSearchCache` and
    `contact_search_cache_from_kinematics`.
  - the updated active Newmark tutorials completed successfully.

## Current twenty-sixth slice

- Added `ContactCandidateSet` as a typed broad-phase pruning hook.
- `node_surface_contact_kinematics_from_surfaces(...)` and
  `surface_quadrature_contact_kinematics_from_surfaces(...)` now accept
  `candidate_facet_ids=...`.
- The candidate set limits closest-facet search to selected master facet rows.
  This is less strict than `search_cache`: candidates prune search, while
  `search_cache` freezes exact pairing.
- If both are provided, `search_cache` wins and no candidate validation/search is
  performed.
- Added tests showing candidate pruning for node-surface and surface-quadrature
  contact, plus invalid candidate id rejection.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 33 tests.
  - `python -m py_compile` passed for the updated modules/tutorials.
  - top-level API smoke passed for `ContactCandidateSet`.
  - node-surface and surface-quadrature active Newmark tutorials completed.

## Current twenty-seventh slice

- Added `contact_candidate_set_from_bounding_boxes(...)`.
- The helper builds a `ContactCandidateSet` from slave/master surface bounding
  boxes and a `search_radius`.
- It supports the same `n_total_nodes` and `displacement` convention as the
  contact kinematics builders, so broad-phase pruning can be evaluated on the
  current displaced geometry.
- The helper currently returns a global master-facet subset, not per-contact
  candidates. This keeps the API simple while removing the need for users to
  hand-author candidate facet ids.
- Updated `tutorials/node_surface_contact_demo.py` to build and pass candidates
  from the AABB helper.
- Added tests for:
  - pruning distant master facets,
  - changing candidates after displacement,
  - rejecting empty broad-phase results.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 35 tests.
  - `python -m py_compile` passed for the updated modules/tutorials.
  - top-level API smoke passed for
    `contact_candidate_set_from_bounding_boxes`.
  - node-surface and surface-quadrature tutorials completed.

## Current twenty-eighth slice

- Extended `ContactCandidateSet` to support per-contact candidate segments via
  flat `master_facet_ids` plus `contact_offsets`.
- Added `contact_candidate_set_from_per_contact(...)` for manually building
  variable-length per-contact candidate sets.
- Node-surface and surface-quadrature kinematics builders now accept both global
  candidate sets and per-contact candidate sets.
- Added `node_surface_candidate_set_from_bounding_boxes(...)`, which builds
  per-slave-node candidates from point-expanded AABBs.
- Updated `tutorials/node_surface_contact_demo.py` to use the per-node AABB
  helper.
- Added tests for:
  - per-slave-node candidate pruning,
  - per-quadrature-point candidate pruning,
  - per-node AABB generation,
  - displaced per-node AABB generation,
  - candidate offset/contact count validation.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 39 tests.
  - `python -m py_compile` passed for the updated modules/tutorial.
  - top-level API smoke passed for
    `contact_candidate_set_from_per_contact` and
    `node_surface_candidate_set_from_bounding_boxes`.
  - node-surface and active Newmark tutorials completed.

## Current twenty-ninth slice

- Added `ContactNeighborList` for broad-phase candidate reuse.
- Added `node_surface_neighbor_list_from_bounding_boxes(...)`.
- The node-surface neighbor list builds per-node candidates with
  `search_radius + skin`, stores the reference displacement, and reports
  `needs_refresh(displacement)` when max nodal drift exceeds `skin / 2`.
- `ContactNeighborList.max_drift(...)` reports the nodal drift used by the
  refresh criterion.
- Updated `tutorials/node_surface_contact_demo.py` to build candidates through
  the neighbor-list helper and report whether refresh is needed.
- Added tests for:
  - no refresh under small displacement,
  - refresh after large displacement,
  - candidate rebuild after refresh,
  - skin validation.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 40 tests.
  - `python -m py_compile` passed for updated modules/tutorials.
  - top-level API smoke passed for `ContactNeighborList` and
    `node_surface_neighbor_list_from_bounding_boxes`.
  - node-surface and active Newmark tutorials completed.

## Current thirtieth slice

- Added `ContactAABBIndex`, a uniform-grid spatial index over master facet
  AABBs.
- Added `contact_aabb_index_from_surface(...)` to build the index from a
  master surface, including optional displaced geometry and explicit cell size.
- Added `node_surface_candidate_set_from_aabb_index(...)` to query per-slave-node
  candidates from the index.
- Added `node_surface_neighbor_list_from_aabb_index(...)` to build a
  `ContactNeighborList` from a reusable index.
- Updated `tutorials/node_surface_contact_demo.py` to use the indexed
  neighbor-list path.
- Added tests for:
  - direct AABB index query,
  - indexed per-node candidate generation,
  - displaced index construction,
  - indexed neighbor-list refresh behavior,
  - invalid cell size validation.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 43 tests.
  - `python -m py_compile` passed for the updated modules/tutorials.
  - top-level API smoke passed for `ContactAABBIndex`,
    `contact_aabb_index_from_surface`,
    `node_surface_candidate_set_from_aabb_index`, and
    `node_surface_neighbor_list_from_aabb_index`.
  - node-surface and active Newmark tutorials completed.

## Current thirty-first slice

- Added `surface_quadrature_candidate_set_from_aabb_index(...)`.
- Added `surface_quadrature_neighbor_list_from_aabb_index(...)`.
- The surface-quadrature helper queries the master `ContactAABBIndex` once per
  slave quadrature point, in the same order used by
  `surface_quadrature_contact_kinematics_from_surfaces(...)`, and returns a
  per-contact `ContactCandidateSet`.
- Updated `tutorials/craig_bampton_surface_quadrature_active_newmark.py` to use
  the indexed quadrature neighbor-list path before the exact contact search
  cache is established.
- Added tests for:
  - indexed per-quadrature-point candidate generation,
  - indexed surface-quadrature neighbor-list refresh behavior,
  - empty indexed quadrature candidate queries.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 45 tests.
  - `python -m py_compile` passed for the updated modules/tutorials.
  - top-level API smoke passed for
    `surface_quadrature_candidate_set_from_aabb_index` and
    `surface_quadrature_neighbor_list_from_aabb_index`.
  - node-surface and surface-quadrature tutorials completed.

## Current thirty-second slice

- Added `NodeSurfaceContactSearchManager`.
- Added `make_node_surface_contact_search_manager(...)`.
- The manager owns:
  - master `ContactAABBIndex`,
  - `ContactNeighborList`,
  - exact `ContactSearchCache`,
  - node-surface penalty parameters.
- `manager.build_contact(u)` returns `(contact, next_manager)`, so Active/Newmark
  loops can update search state explicitly without mutating residual evaluation.
- The manager refreshes the index/neighbor list when max displacement drift
  exceeds the neighbor-list skin criterion, otherwise it reuses the cached broad
  phase and exact pairing.
- Updated `tutorials/craig_bampton_node_surface_active_newmark.py` to use the
  manager instead of tutorial-local search-cache bookkeeping.
- Added tests for:
  - first contact build initializes index, neighbor list, and search cache,
  - small drift reuses search state,
  - large drift refreshes and can switch master facet.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 46 tests.
  - `python -m py_compile` passed for updated modules/tutorials.
  - top-level API smoke passed for `NodeSurfaceContactSearchManager` and
    `make_node_surface_contact_search_manager`.
  - node-surface and surface-quadrature tutorials completed.

## Current thirty-third slice

- Added `SurfaceQuadratureContactSearchManager`.
- Added `make_surface_quadrature_contact_search_manager(...)`.
- The manager mirrors the node-surface manager for surface-quadrature contact:
  it owns the master AABB index, neighbor list, exact search cache, quadrature
  rule, and penalty parameters.
- `manager.build_contact(u)` returns `(contact, next_manager)`, keeping search
  state explicit and residual evaluation pure.
- Updated `tutorials/craig_bampton_surface_quadrature_active_newmark.py` to use
  the manager instead of tutorial-local search-cache and neighbor-list
  bookkeeping.
- Added tests for:
  - first surface-quadrature contact build initializes index, neighbor list, and
    search cache,
  - small drift reuses search state,
  - large drift refreshes and can switch master facets.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 47 tests.
  - `python -m py_compile` passed for updated modules/tutorials.
  - top-level API smoke passed for `SurfaceQuadratureContactSearchManager` and
    `make_surface_quadrature_contact_search_manager`.
  - node-surface and surface-quadrature tutorials completed.

## Current thirty-fourth slice

- Added `FrictionalContactUpdateSnapshot`.
- Added `TangentialPenaltyFrictionManager`.
- `manager.snapshot(contact, u)` returns a frozen normal-contact snapshot plus
  the currently stored tangential friction history.
- `snapshot.residual()` composes the frozen active normal residual with
  `make_friction_residual(...)` when history is present.
- `manager.advance(contact, u)` explicitly updates tangential penalty history
  after a converged displacement and returns the next manager state.
- Added `manager.snapshot_and_advance(contact, u)` as convenience sugar.
- Updated `tutorials/craig_bampton_friction_history_rom.py` to use the friction
  manager instead of tutorial-local history updates and residual composition.
- Added tests for:
  - snapshot residual with and without history,
  - history update and frozen friction residual composition,
  - snapshot-and-advance convenience,
  - manager parameter validation.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 49 tests.
  - `python -m py_compile` passed for updated modules/tutorials.
  - top-level API smoke passed for `TangentialPenaltyFrictionManager` and
    `FrictionalContactUpdateSnapshot`.
  - friction, node-surface, and surface-quadrature tutorials completed.

## Current thirty-fifth slice

- Added a composed workflow test for `NodeSurfaceContactSearchManager` and
  `TangentialPenaltyFrictionManager`.
- The test verifies that search-managed node-surface contact can feed the
  friction snapshot API and produce the expected combined normal/tangential
  residual distribution.
- Added `tutorials/craig_bampton_node_surface_friction_active_newmark.py`.
- The tutorial combines:
  - Craig-Bampton projection,
  - active node-surface contact search,
  - frozen normal contact snapshots,
  - frozen tangential friction history,
  - explicit post-convergence friction-history advance.
- This makes the active-contact path closer to the intended production API:
  search managers own broad/exact contact state, friction managers own history
  state, and the structural solver sees only a reduced residual builder.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 50 tests.
  - `python -m py_compile` passed for the new tutorial and updated test.
  - `PYTHONPATH=src python tutorials/craig_bampton_node_surface_friction_active_newmark.py`
    completed successfully.

## Current thirty-sixth slice

- Added an AD regression test for a reduced contact/friction objective.
- The test freezes active normal-contact state and tangential friction history,
  then differentiates a scalar objective through:
  - `u = Phi q`,
  - full-space structural residual,
  - frozen active contact residual,
  - frozen friction residual,
  - `Rr(q) = Phi.T R(Phi q)`.
- The test checks the reduced objective gradient against
  `Phi.T J_full Phi` chain-rule assembly and a finite-difference spot check.
- This locks in the intended AD contract: search/active/history updates are
  outer-loop state, while residual evaluation after snapshotting remains
  differentiable in reduced coordinates.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py -k "frozen_contact_friction"`
    passed.
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 51 tests.

## Current thirty-seventh slice

- Extended the composed search/friction workflow to surface-quadrature contact.
- Fixed surface-quadrature friction residual scattering so stored tangential
  friction forces are multiplied by `quadrature_weights` before slave/master
  distribution. This matches the normal-contact residual integration path.
- Added a composed workflow test for `SurfaceQuadratureContactSearchManager`
  and `TangentialPenaltyFrictionManager`.
- The test verifies:
  - search-cache creation,
  - friction history creation,
  - quadrature weights in the contact kinematics,
  - combined normal/tangential residual distribution with quadrature weighting.
- Added `tutorials/craig_bampton_surface_quadrature_friction_active_newmark.py`.
- The tutorial combines CB projection, active quadrature contact search, frozen
  active snapshots, frozen friction history, and post-convergence friction
  history advance.
- Verification:
  - `PYTHONPATH=src pytest -q src/tests/test_contact.py -k "surface_quadrature_search_and_friction"`
    passed.
  - `PYTHONPATH=src python tutorials/craig_bampton_surface_quadrature_friction_active_newmark.py`
    completed successfully.
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 52 tests.

## Current thirty-eighth slice

- Added `ReducedContactDynamics`.
- The facade owns:
  - Craig-Bampton basis,
  - full-space `K`, `M`, optional `C`,
  - contact search manager,
  - optional tangential friction manager.
- It exposes:
  - `build_snapshot(q)`,
  - `internal_force_from_snapshot(snapshot)`,
  - `active_newmark_step(...)`,
  - `advance_friction(...)`,
  - full-force projection via `project_force(...)`.
- This removes tutorial-local closure plumbing for the common active contact
  ROM path while preserving the explicit manager-state model.
- Updated the node-surface and surface-quadrature friction active Newmark
  tutorials to use the facade and pass full-space external forces directly.
- Added a regression test that runs one search/friction active Newmark step
  through the facade and verifies search cache, friction history, and active
  contact state.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "reduced_contact_dynamics_facade"`
    passed.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton_node_surface_friction_active_newmark.py`
    completed successfully.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton_surface_quadrature_friction_active_newmark.py`
    completed successfully.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 53 tests.

## Current thirty-ninth slice

- Added a numerical equivalence test for `ReducedContactDynamics`.
- The test solves the same node-surface friction active Newmark step with:
  - the new facade,
  - the previous manual callback/closure wiring.
- It verifies that reduced displacement, velocity, acceleration, active contact
  state, and friction-history forces match.
- This improves confidence that the facade is an API simplification, not a
  behavioral change.
- Current correctness status:
  - CB projection chain rule is covered by `Phi.T J_full Phi` tests.
  - Frozen active contact and frozen friction residuals are AD-tested.
  - Node-surface and surface-quadrature contact residual scatter have force
    balance/weighting tests.
  - Active Newmark examples converge and now match manual callback wiring for
    the facade path.
- Remaining validation gap:
  - These are internal consistency and manufactured small-system tests.
  - They do not yet validate against an external FE/contact benchmark,
    analytical Hertz/contact solution, or mesh/time-step convergence study.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "matches_manual_callbacks"`
    passed.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 54 tests.

## Current fortieth slice

- Added the first full-order vs CB-ROM contact benchmark regression.
- The test solves one active Newmark penalty-contact step in full coordinates,
  then solves the same step with CB-ROMs using retained contact DOFs and
  increasing internal mode count.
- It verifies:
  - all internal modes kept by CB reproduces the full-order displacement within
    tolerance,
  - one retained internal mode improves the full-order displacement error over
    zero internal modes,
  - active contact solve converges in both full-order and reduced paths.
- This is still a small manufactured benchmark, but it checks the key ROM
  convergence contract more directly than the earlier internal consistency
  tests.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "full_order_when_all_internal_modes"`
    passed.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 55 tests.

## Current forty-first slice

- Added `tutorials/craig_bampton_full_order_contact_benchmark.py`.
- The script promotes the full-order vs CB-ROM contact regression into a
  readable benchmark table over `n_modes = 0..4`.
- Current output shows:
  - full-order active contact is active,
  - all ROM cases preserve the active state,
  - errors decrease as internal modes are added,
  - the all-mode CB basis reproduces the full-order displacement to near
    roundoff for this small dense system.
- Representative errors from the current run:
  - `n_modes=0`: abs error `1.963349e-03`,
  - `n_modes=1`: abs error `1.962958e-03`,
  - `n_modes=2`: abs error `1.537210e-03`,
  - `n_modes=3`: abs error `1.305716e-03`,
  - `n_modes=4`: abs error `1.947349e-09`.
- Verification:
  - `PYENV_VERSION=jaxfem python -m py_compile tutorials/craig_bampton_full_order_contact_benchmark.py`
    passed.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton_full_order_contact_benchmark.py`
    completed successfully.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 55 tests.

## Current forty-second slice

- Added a closed-form 1D obstacle penalty-contact validation.
- The reference problem is a linear spring chain with a unilateral penalty
  obstacle at the retained contact DOF.
- The closed-form active solution solves
  `(K + p e0 e0.T) u = f - p gap0 e0` after the inactive solution is found to
  violate the gap condition.
- Added a regression test verifying:
  - full-order active-contact fixed-point solve matches the closed-form
    reference,
  - all-mode CB-ROM solve matches the same reference,
  - both full-order and ROM active states are active.
- Added `tutorials/craig_bampton_1d_obstacle_contact_reference.py`, which
  prints the closed-form displacement, full-order displacement, and ROM error
  vs `n_modes`.
- Representative tutorial output:
  - full-order abs error against closed form: `2.137101e-09`,
  - `n_modes=0`: abs error `2.482725e-02`,
  - `n_modes=1`: abs error `1.938384e-02`,
  - `n_modes=2`: abs error `1.238996e-02`,
  - `n_modes=3`: abs error `2.085752e-09`.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "1d_obstacle"`
    passed.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton_1d_obstacle_contact_reference.py`
    completed successfully.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 56 tests.

## Current forty-third slice

- Added an AD regression test for a reduced dynamic Newmark contact/friction
  objective.
- The test builds a frozen active contact/friction snapshot, then differentiates
  through the effective Newmark residual:
  `M a(q_next) + C v(q_next) + Rr(q_next) - F`.
- It checks:
  - `jax.grad` returns finite gradients,
  - the scalar objective gradient matches `J_eff.T @ G(q_next)`,
  - one finite-difference spot check agrees with the AD gradient.
- This closes the earlier AD gap between static reduced residual objectives and
  the actual dynamic residual used by Newmark solves.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "dynamic_contact_friction_newmark_objective"`
    passed.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 57 tests.

## Current forty-fourth slice

- Reduced duplication between node-surface and surface-quadrature contact search
  managers.
- Shared helper coverage now includes:
  - scalar input validation for `dim`, `n_total_nodes`, `search_radius`,
    `skin`, `penalty`, and `smoothing`,
  - master-surface AABB index rebuild,
  - neighbor-list refresh decision,
  - candidate-set selection vs exact search-cache reuse.
- The contact-type-specific parts remain local:
  - node-surface neighbor-list construction,
  - surface-quadrature neighbor-list construction,
  - kinematics construction,
  - penalty contact object construction.
- `with_search_cache(...)` now uses `dataclasses.replace`, reducing the risk of
  accidentally dropping manager fields when adding future options.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py -k "contact_search_manager or search_and_friction"`
    passed: 4 tests.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py`
    passed: 43 tests.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py`
    passed: 14 tests.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 57 tests.
  - Node-surface and surface-quadrature friction active Newmark tutorials
    completed successfully.

## Current forty-fifth slice

- Added typed runtime-checkable manager protocols:
  - `ContactSearchManagerLike`,
  - `FrictionManagerLike`.
- `ReducedContactDynamics` now annotates its manager fields with these protocols
  instead of plain `object`.
- The facade validates manager capabilities at construction time:
  - `search_manager` must implement `build_contact(displacement)`,
  - `friction_manager`, when present, must implement `snapshot(contact, u)` and
    `advance(contact, u)`.
- Exported the protocols through `fluxfem.core` and top-level `fluxfem`.
- Added tests that invalid managers fail early with clear `TypeError` messages.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "manager_protocols or reduced_contact_dynamics_facade"`
    passed: 2 tests.
  - top-level smoke passed for `ContactSearchManagerLike` and
    `FrictionManagerLike`.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 58 tests.

## Current forty-sixth slice

- Added a surface-quadrature contact reference validation.
- The test solves the same active Newmark surface-contact problem in:
  - full-order coordinates,
  - CB-ROM coordinates with retained surface/contact DOFs.
- It verifies:
  - all internal modes kept by CB reproduces the full-order displacement,
  - reduced models preserve the active surface-contact state,
  - an intermediate mode count improves error over the retained-only ROM.
- Added `tutorials/craig_bampton_surface_contact_reference.py`.
- Representative tutorial output:
  - full active count: `2`,
  - full gaps: `[-0.00277478, -0.00277478]`,
  - `n_modes=0`: abs error `1.503429e-02`,
  - `n_modes=1`: abs error `1.502753e-02`,
  - `n_modes=2`: abs error `1.226242e-04`,
  - `n_modes=3`: abs error `1.226242e-04`,
  - `n_modes=4`: abs error `1.020350e-08`.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "surface_quadrature_contact_matches_full_order"`
    passed.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton_surface_contact_reference.py`
    completed successfully.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 59 tests.

## Current forty-seventh slice

- Started the independent-reference validation branch for contact.
- Added a low-level surface-quadrature penalty-contact check that does not use
  the `SurfaceQuadraturePenaltyContact.residual(...)` implementation as its
  reference.
- Extended it to a two-facet line-surface case with four quadrature points and
  a mixed active/inactive contact patch.
- Extended it again to a 3D quad surface-pair with four vertex quadrature points
  and a mixed active/inactive patch.
- Added the same independent residual/Jacobian check for a
  `SurfaceQuadratureContactSearchManager` workflow, including a refreshed
  pairing/cache after large slave-surface motion.
- Added an independent surface-quadrature friction-history/scatter check:
  tangential slip, stick/slip clipping, quadrature-weighted residual scatter,
  and frozen-friction zero Jacobian are verified outside the contact residual
  implementation.
- Added a `ReducedContactDynamics` facade-vs-manual callback equivalence test
  for surface-quadrature contact with a friction manager. This checks the public
  manager interface path, reduced Newmark state, active state, search cache, and
  advanced friction history.
- Added `notes/contact_rom_api_guide.md` as a user-facing entry point for the
  recommended CB contact ROM workflow, manager roles, AD behavior, and active
  contact applicability.
- The test builds an independent NumPy contact-row form:
  - `gap_q = gap0_q + B_q u`,
  - active residual `R += penalty * weight_q * gap_q * B_q`,
  - active tangent `K += penalty * weight_q * outer(B_q, B_q)`.
- It verifies residual, active mask, gaps, and AD Jacobian against that
  independently assembled weighted penalty form.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton_surface_quadrature_friction_active_newmark.py`
    completed successfully.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py -k "independent_penalty_form or independent_weighted_penalty_form"`
    passed: 3 tests.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py -k "independent_reference or independent_penalty_form or independent_weighted_penalty_form"`
    passed: 4 tests.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_contact.py -k "friction or independent_reference"`
    passed: 7 tests.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "reduced_contact_dynamics"`
    passed: 4 tests.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 65 tests.

## Current forty-eighth slice

- Started the sparse/iterative CB branch.
- Added `solve_constraint_modes(...)` as the first replaceable solve boundary
  inside CB basis construction.
- `make_craig_bampton_basis(...)` now accepts:
  - `constraint_solver="dense"` for the existing direct dense solve,
  - `constraint_solver="cg"` for an in-tree SPD conjugate-gradient multi-RHS
    solve,
  - a custom callable `solver(K_ii, rhs)` for external sparse/direct/iterative
    solvers.
- Fixed-interface modal extraction is still dense in this slice; the new solver
  option only targets static constraint modes.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "craig_bampton"`
    passed: 4 tests.
  - `PYENV_VERSION=jaxfem python -m py_compile src/fluxfem/core/rom.py src/fluxfem/core/__init__.py src/fluxfem/__init__.py`
    passed.
  - top-level smoke passed for `solve_constraint_modes`.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 67 tests.

## Current forty-ninth slice

- Extended the sparse/iterative CB branch to fixed-interface modal extraction.
- `fixed_interface_modes(...)` now accepts:
  - `solver="dense"` for the existing generalized dense eigensolve,
  - `solver="subspace"` for block inverse/subspace iteration,
  - a custom callable `solver(K_ii, M_ii, n_modes)`.
- `make_craig_bampton_basis(...)` forwards modal options:
  - `modal_solver`,
  - `modal_linear_solver`,
  - `modal_oversample`,
  - `modal_maxiter`,
  - `modal_tol`.
- The subspace path reuses `solve_constraint_modes(...)` as its internal
  linear-solve hook, so it can use dense, CG, or external callables.
- Verification:
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py -k "craig_bampton or fixed_interface_subspace"`
    passed: 7 tests.
  - `PYENV_VERSION=jaxfem python -m py_compile src/fluxfem/core/rom.py src/fluxfem/core/__init__.py src/fluxfem/__init__.py`
    passed.
  - top-level smoke passed for `fixed_interface_modes`.
  - `PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
    passed: 70 tests.

## AD/contact design

The important design choice is to avoid special ROM element kernels at first.
Contact/Nitsche/penalty residuals can be assembled in the full coordinate space,
including retained boundary DOFs, then projected by the CB basis. Since the only
ROM operation is `u = Phi q` and `Rr = Phi.T R(u)`, JAX can still differentiate
through the full residual path.

For contact, choose retained DOFs from the contact candidate surfaces. Internal
DOFs can be modalized. If the active contact patch moves far beyond the retained
surface set, the model will need either a larger retained set or enrichment.

## Next implementation steps

1. Add a production sparse eigensolver adapter, likely via optional SciPy
   `eigsh`/LOBPCG or an existing project sparse backend, while preserving the
   current callable modal-solver hook.
2. Extend the independent contact reference from the current one-facet
   weighted penalty form to a multi-facet FE assembly or external benchmark.
3. Consider a higher-level convenience constructor once multiple real examples
   use the same retained-surface/search/friction setup.

## Open questions

- Should retained DOFs be specified directly, by node ids, or by surface tags?
  Direct DOFs are implemented now; surface/node helpers will be more ergonomic.
- For moving contact, how large should the retained boundary band be?
- For frictional history variables, do we keep history in full contact-space
  coordinates and project only structural equilibrium?

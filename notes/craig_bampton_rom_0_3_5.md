# Craig-Bampton ROM Port for 0.3.5

## Current scope

- `fluxfem.solver.make_craig_bampton_basis` builds a CB basis from assembled stiffness and mass matrices.
- Retained DOFs stay physical: the retained rows of the retained-coordinate block are identity.
- Internal DOFs are represented by static constraint modes and fixed-interface modes.
- Dense, SciPy sparse direct (`spsolve`), and SciPy sparse modal (`eigsh`) paths are exposed through solver options.
- The legacy iterative paths are available: `constraint_solver="cg"` and `modal_solver="subspace"` with `modal_linear_solver="cg"`.
- `CraigBamptonBasis` is a JAX pytree and provides `expand`, `project_vector`, `project_matrix`, `project_operator`, `reduced_residual`, and `reduced_jacobian`.
- The previous ROM-level helper layer is restored in 0.3.5 form: `LinearConstraintSystem`, `ReducedLinearConstraintSystem`, `ReducedCoupledSystemBuilder`, `RBE3Patch`, `ReferencePointFixture`, `RBE3RemoteFixture`, KKT solve helpers, reduced Newmark helpers, and generic active-contact/state outer loops.

## Why this helps contact and fixtures

CB retained DOFs are the natural location for interface, contact, and remote fixture coordinates. A contact workflow can keep uncertain active-contact boundary DOFs in the retained set, while reducing only the interior. The existing 0.3.5 coupled-system and RBE3 APIs should remain the primary wrapper layer for full 3D fixtures and reference points; CB is the projection layer below that.

`RBE3Patch` and `ReferencePointFixture` are intentionally thin compatibility wrappers. They cover the older translational reference-point MPC/preload examples and can project through CB with `LinearConstraintSystem.project`. `RBE3RemoteFixture` is the CB-facing wrapper for both translational-only and 6-DOF rotational RBE3 reference fixtures. Its remote rotation coordinates are not structural CB retained DOFs; they are explicit appended reference DOFs preserved outside the workpiece basis via `LinearConstraintSystem.project(..., n_extra_dofs=...)`.

For new code, prefer `ReducedCoupledSystemBuilder`. It mirrors the name-based `CoupledSystemBuilder` style: register structural fields, name retained node/DOF/surface groups with `retain_node_set(...)`, `retain_dof_group(...)`, or `retain_surface_nodes(...)`, call `reduce_field(..., method="craig_bampton", retained_groups=[...])` per field, append remote points, connect them with `add_rbe3_constraint(...)`, add preload/Dirichlet through named remote fields, record contact-candidate pairs and Python metadata such as `normal`, `penalty`, or `search_radius` with `register_contact_pair(...)`, resolve their full/reduced DOF bundles with `contact_pair_dofs(...)`, and connect reduced subsystems with `tie_retained_groups(...)` or `add_dof_tie_constraint(...)`. This keeps the connection graph readable and avoids hand-written ROM/reference DOF offsets in tutorials. The CB builder records contact-ready metadata and retained/reduced interface DOF bundles; actual contact residuals, mortar operators, and active-search updates remain external solver responsibilities.

Small fixture utilities are now part of the public API rather than tutorial glue:
`vector_dofs_from_nodes`, `retained_dofs_from_node_sets`, `remote_reference_direction`,
and `validate_rbe3_remote_reference_rank`. Tutorials should use these helpers so
examples describe the model connection graph rather than low-level DOF indexing.

`tutorials/craig_bampton_rbe3_preload_component.py` is the compact FluxFEM counterpart of `skfem-Craig-Bampton-ROM/experiment-2`: it compares a full explicit-reference RBE3 preload KKT solve with the CB-projected KKT solve on a notched tetrahedral workpiece.

The same tutorial supports `--fixture-boundary preload`, `--fixture-boundary dirichlet`, and `--fixture-rotation none|rbe3`. The Dirichlet path prescribes the explicit RBE3 reference-point displacement using nonzero `fixed_values` in `LinearConstraintSystem.solve`, so a one-sided fixture can be represented either as a preload spring or as a prescribed support motion.

`tutorials/craig_bampton_reduced_coupled_builder.py` is the preferred short-form tutorial for new users. It exercises the same conceptual pieces with named fields:

1. register `workpiece`,
2. reduce it with `reduce_field(..., method="craig_bampton")`,
3. add a named remote `fixture`,
4. connect the fixture with `add_rbe3_fixture_from_nodes(...)`,
5. solve preload and Dirichlet variants without manual ROM/reference offset arithmetic.

The older `craig_bampton_rbe3_preload_mpc.py` and `craig_bampton_fluxfem_rbe3_preload_experiment2.py` files remain useful as low-level compatibility/reference examples, but they should not be the first tutorial path for the current API.

`tutorials/craig_bampton_multifield_builder.py` demonstrates the multi-subsystem path. It registers `part_a` and `part_b`, names support/interface retained groups, creates interface groups from surface objects, records an `interface_candidate` contact pair, reduces both fields from those group names, ties `part_a:interface` to `part_b:interface`, and checks the ROM solution against a full KKT reference.

The active-contact/Newmark helpers no longer depend on removed legacy contact classes. They accept user-provided contact-state callbacks:

- `residual_from_contact_state(state) -> residual_fn`
- `solve_fn(residual_fn, initial_solution) -> (solution, info)`
- `update_contact_state(solution) -> new_state`

This keeps residual/Jacobian evaluation pure and AD-friendly while active sets, broad-phase search, or friction history are updated outside the differentiated residual.

For future uncertain/contact-changing time evolution, the intended shape is:

1. keep candidate contact/interface DOFs in the CB retained set,
2. evaluate reduced Newmark residuals with `make_newmark_effective_residual`,
3. update contact state with `active_contact_newmark_step` outside the differentiated residual,
4. represent changing fixture states as either active preload springs, nonzero Dirichlet reference DOFs, or current `CoupledSystemBuilder` constraints.

Current contact tutorials follow this split: the reduced residual stays differentiable for a frozen contact snapshot, while broad-phase search, active-set updates, and tangential friction history are advanced explicitly between solves.

## Verification snapshot

- `PYTHONPATH=src pytest -q src/tests/test_craig_bampton.py src/tests/test_rbe_constraints.py`
- `PYTHONPATH=src python tutorials/craig_bampton_contact_rom.py`
- `PYTHONPATH=src python tutorials/craig_bampton_reduced_coupled_builder.py`
- `PYTHONPATH=src python tutorials/craig_bampton_multifield_builder.py`
- `PYTHONPATH=src python tutorials/craig_bampton_rbe3_preload_component.py --nx 12 --ny 9 --nz 1 --modes 3 --fixture-boundary preload --fixture-rotation rbe3`
- `PYTHONPATH=src python tutorials/craig_bampton_rbe3_preload_component.py --nx 12 --ny 9 --nz 1 --modes 3 --fixture-boundary dirichlet --fixture-rotation rbe3`
- `PYTHONPATH=src python tutorials/craig_bampton_1d_obstacle_contact_reference.py`
- `PYTHONPATH=src python tutorials/craig_bampton_full_order_contact_benchmark.py`
- `PYTHONPATH=src python tutorials/craig_bampton_surface_contact_reference.py`
- `PYTHONPATH=src python tutorials/craig_bampton_friction_history_rom.py`
- `PYTHONPATH=src python tutorials/craig_bampton_active_contact_newmark.py`
- `PYTHONPATH=src python tutorials/craig_bampton_node_surface_active_newmark.py`
- `PYTHONPATH=src python tutorials/craig_bampton_surface_quadrature_active_newmark.py`
- `PYTHONPATH=src python tutorials/craig_bampton_node_surface_friction_active_newmark.py`
- `PYTHONPATH=src python tutorials/craig_bampton_surface_quadrature_friction_active_newmark.py`
- `PYTHONPATH=src python tutorials/craig_bampton_fe_surface_contact.py`
- `PYTHONPATH=src python tutorials/craig_bampton_sparse_fe_basis.py`
- `PYTHONPATH=src python tutorials/craig_bampton_rbe3_preload_mpc.py`

## Important limitation

The current port still stores the final basis as a dense matrix. `project_matrix`
keeps SciPy/FluxFEM sparse inputs sparse for the `K @ Phi` multiplication, and
`project_operator` exposes the matrix-free reduced action `q -> Phi.T A(Phi q)`
for sparse/operator/callable inputs. Very large models will still need a
block/matrix-free basis representation before this becomes the best option for
production-scale DOF counts.

## Next implementation steps

1. Add a block/matrix-free basis representation so full dense `Phi` does not need to be materialized.
2. Use `ReducedCoupledSystem.contact_pair_adapter(...)` in contact/mortar tutorials that attach real contact operators.
3. Add an active-contact example where candidate contact DOFs are retained and only interior DOFs are reduced.
4. Extend multi-field coupling beyond DOF ties to named spring/damper/interface operators.

## Contact mortar and AL next tasks

The existing mortar path has standard nodal and P0 multiplier spaces.
`dual_nodal` is now the default multiplier family for mortar. It provides a
master-side biorthogonal nodal map:
`B_a = I` and `B_b = pinv(M_aa) M_ab`. For full-rank nodal mass blocks this is
the usual inverse-based dual map; for rank-deficient or inactive rows it becomes
the least-squares dual map. The original nodal multiplier remains available with
`MultiplierSpec(family="nodal")`.

The same multiplier object also supports an initial coarse mortar option:
`MultiplierSpec.coarse_dual_mortar()` selects a reduced row basis automatically
from the assembled `B` matrix using SVD energy/numerical-rank criteria. Users can
still pass `rank=k` for a QR-based fixed-rank projection, `max_rank=k` to cap the
automatic rank, or `projection=P` for an explicit projection matrix. This
reduces multiplier rows after the chosen mortar family is assembled.
`MultiplierSpec.coarse_p0_mortar(contact, patch_ids=...)` is different: it
builds an integrated coarse P0 multiplier by summing fine P0 facet/supermesh
integral rows over user-defined patch ids.
`MultiplierSpec.coarse_p1_mortar(basis=C)` is the first integrated coarse P1
path: rows of `C` define coarse master-side nodal multiplier shapes and assemble
`B_a = C M_aa`, `B_b = C M_ab`. For simple grouped coarse nodes, use
`coarse_p1_basis_from_node_groups(...)` to build `C`.

Current scope:

- supported default: `ContactMultiplierSpace()` / `MultiplierSpec()` resolve to
  `family="dual_nodal", side="master"`
- supported explicit legacy basis: `family="nodal"`
- supported user-facing constructors: `dual_mortar()`,
  `coarse_dual_mortar()`, `nodal_mortar()`, `p0_mortar(contact)`,
  `coarse_p0_mortar(contact, patch_ids=...)`, `coarse_p1_mortar(basis=C)`
- supported coarse options: auto SVD rank selection, fixed `rank=k`, capped
  `max_rank=k`, explicit algebraic `projection=P`, or integrated P0 patch
  grouping with `patch_ids`
- supported paths: `assemble_contact_constraint_operators`,
  `assemble_contact_kkt(..., format="dense|fluxsparse|bcoo")`
- limitation: the sparse KKT path currently builds the small dual transform in
  dense form before returning COO/BCOO
- limitation: no slave-side dual basis yet, because the current generic coupling
  API exposes `M_aa` and `M_ab`, but not an independent slave-side nodal mass
  block
- limitation: no P0/supermesh dual basis yet

The next natural step is an augmented-Lagrangian outer loop on top of these
constraint operators. The first generic API is now
`solve_augmented_lagrangian_outer_loop(...)`.

It is intentionally lower level than a contact-specific solver:

1. assemble or accept `(B, Kuu)` from `assemble_contact_constraint_operators`;
2. pass a `solve_subproblem(x, state)` callback that uses
   `state.lambda_values` and `state.rho`;
3. define constraints with either `constraint_fn(solution)` or
   `ContactOperators.B @ solution - offset`;
4. update multiplier state explicitly, with optional `projection="nonnegative"`
   or a custom update/projection callback;
5. stop on constraint and multiplier-update norms.

That keeps the contact law state outside differentiated residual evaluation,
matching the active-contact/Newmark design used by the CB-ROM utilities.

This API is enough for equality-style mortar constraints and for a unilateral
prototype where the user supplies a positive penetration constraint and a
nonnegative multiplier projection. A contact-specific AL convenience wrapper can
now be added on top without changing the low-level interface.

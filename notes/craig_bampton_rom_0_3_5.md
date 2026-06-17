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

For new code, prefer `ReducedCoupledSystemBuilder`. It mirrors the name-based `CoupledSystemBuilder` style: register structural fields, name retained node/DOF groups with `retain_node_set(...)` or `retain_dof_group(...)`, call `reduce_field(..., method="craig_bampton", retained_groups=[...])` per field, append remote points, connect them with `add_rbe3_constraint(...)`, add preload/Dirichlet through named remote fields, and connect reduced subsystems with `tie_retained_groups(...)` or `add_dof_tie_constraint(...)`. This keeps the connection graph readable and avoids hand-written ROM/reference DOF offsets in tutorials. For sparse KKT assembly and current contact multiplier blocks, use `NumpyCoupledSystemBuilder` / `CoupledSystemBuilder`.

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

`tutorials/craig_bampton_multifield_builder.py` demonstrates the multi-subsystem path. It registers `part_a` and `part_b`, names support/interface retained groups, reduces both fields from those group names, ties `part_a:interface` to `part_b:interface`, and checks the ROM solution against a full KKT reference.

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
2. Add direct contact integration so candidate contact DOFs can be retained and contact blocks can be attached by field name.
3. Add an active-contact example where candidate contact DOFs are retained and only interior DOFs are reduced.
4. Extend multi-field coupling beyond DOF ties to named spring/damper/interface operators.

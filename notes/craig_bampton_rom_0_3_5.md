# Craig-Bampton ROM Port for 0.3.5

## Current scope

- `fluxfem.solver.make_craig_bampton_basis` builds a CB basis from assembled stiffness and mass matrices.
- Retained DOFs stay physical: the retained rows of the retained-coordinate block are identity.
- Internal DOFs are represented by static constraint modes and fixed-interface modes.
- Dense, SciPy sparse direct (`spsolve`), and SciPy sparse modal (`eigsh`) paths are exposed through solver options.
- The legacy iterative paths are available: `constraint_solver="cg"` and `modal_solver="subspace"` with `modal_linear_solver="cg"`.
- `CraigBamptonBasis` is a JAX pytree and provides `expand`, `project_vector`, `project_matrix`, `reduced_residual`, and `reduced_jacobian`.
- The previous ROM-level helper layer is restored in 0.3.5 form: `LinearConstraintSystem`, `ReducedLinearConstraintSystem`, `RBE3Patch`, `ReferencePointFixture`, KKT solve helpers, reduced Newmark helpers, and generic active-contact/state outer loops.

## Why this helps contact and fixtures

CB retained DOFs are the natural location for interface, contact, and remote fixture coordinates. A contact workflow can keep uncertain active-contact boundary DOFs in the retained set, while reducing only the interior. The existing 0.3.5 coupled-system and RBE3 APIs should remain the primary wrapper layer for full 3D fixtures and reference points; CB is the projection layer below that.

`RBE3Patch` and `ReferencePointFixture` are intentionally thin compatibility wrappers. They cover the older translational reference-point MPC/preload examples and can project through CB with `LinearConstraintSystem.project`. For richer 6-DOF remote points, rotations, multiple named fields, sparse KKT assembly, and current contact multiplier blocks, use `NumpyCoupledSystemBuilder` / `CoupledSystemBuilder`.

The active-contact/Newmark helpers no longer depend on removed legacy contact classes. They accept user-provided contact-state callbacks:

- `residual_from_contact_state(state) -> residual_fn`
- `solve_fn(residual_fn, initial_solution) -> (solution, info)`
- `update_contact_state(solution) -> new_state`

This keeps residual/Jacobian evaluation pure and AD-friendly while active sets, broad-phase search, or friction history are updated outside the differentiated residual.

## Important limitation

The current port still stores the final basis as a dense matrix and `project_matrix` densifies the input matrix. Sparse solvers are used for partition solves and modal extraction, but very large models will need a matrix-free/sparse basis operator before this becomes the best option for production-scale DOF counts.

## Next implementation steps

1. Add a sparse/matrix-free basis operator so `Phi.T @ K @ Phi` can be assembled blockwise without materializing full dense `Phi`.
2. Add direct integration with `CoupledSystemBuilder` so retained CB coordinates can be coupled to named remote/RBE3 blocks without hand-written index mapping.
3. Add a full 3D fixture tutorial that compares full coupled-system solve vs CB-projected KKT solve across active fixture cases.
4. Add an active-contact example where candidate contact DOFs are retained and only interior DOFs are reduced.

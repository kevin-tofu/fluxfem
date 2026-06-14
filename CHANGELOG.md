# Changelog

## 0.2.0 - 2026-06-15

### Added

- Craig-Bampton ROM basis construction with replaceable sparse-oriented solve paths:
  - `constraint_solver="spsolve"` for static constraint modes,
  - `modal_solver="eigsh"` for fixed-interface modal extraction,
  - `modal_solver="subspace"` and callable solver hooks for custom workflows.
- Public ROM projection helpers through `CraigBamptonBasis`:
  - `expand`,
  - `project_vector`,
  - `project_matrix`,
  - reduced residual and reduced Jacobian adapters.
- Active contact ROM workflow support through `ReducedContactDynamics`.
- Linear MPC/KKT utilities:
  - `LinearConstraintSystem`,
  - `ReducedLinearConstraintSystem`,
  - `solve_linear_constraint_kkt`.
- Sparse KKT/preload options for larger systems:
  - `LinearConstraintSystem.solve(..., solver="spsolve")`,
  - `ReducedLinearConstraintSystem.solve(..., solver="spsolve")`,
  - `assemble_reference_fixture_preload(..., sparse=True)`.
- Fixture-oriented MPC wrappers:
  - `RBE3Patch`,
  - `ReferencePointFixture`,
  - `linear_constraint_system_from_reference_fixtures`,
  - `assemble_reference_fixture_preload`.
- Tutorials:
  - `tutorials/craig_bampton_sparse_fe_basis.py`,
  - `tutorials/craig_bampton_rbe3_preload_mpc.py`,
  - `tutorials/craig_bampton_fluxfem_rbe3_preload_experiment2.py`.

### Validation

- The FluxFEM experiment-2 RBE3/preload tutorial matches the scikit-fem reference setup with:
  - full DOFs: `3744`,
  - ROM DOFs: `322`,
  - master DOFs: `318`,
  - relative displacement error around `1e-13` in x64 mode.
- ROM/contact regression suite passed:
  - `PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py`
  - `79 passed`.

# Plate/Shell Status

Current status after merge to `main`.

Implemented:

- Q4 Mindlin/Reissner-Mindlin plate helpers with 3 DOF/node: `w, rx, ry`.
- Q4 Reissner-Mindlin shell helpers with 6 DOF/node: `ux, uy, uz, rx, ry, rz`.
- Selectable transverse shear treatment for plate/shell sections:
  `shear_mode="reduced"` keeps one-point selective reduced integration,
  `shear_mode="full"` uses full 2x2 shear integration, and
  `shear_mode="mitc4"` uses an edge-tying assumed-shear variant.
- Consistent plate/shell mass assembly using `section.rho`:
  `assemble_mindlin_plate_mass`, `assemble_flat_shell_mass`, and
  `assemble_shell_mass`.
- 3D shell coordinates through per-element local frames and global DOF rotation.
- Q4 surface VTU output for plate/shell visualization.
- Shell-to-beam style coupling through existing 6-DOF DOF ties.
- Shell-solid translational tie for coincident shell/solid surface nodes.
- Static full-KKT vs CB-projected-KKT regressions for coincident and nonmatching
  shell-solid translational ties with retained interface DOFs.
- Nonmatching shell-solid translational tie matrix for shell nodes projected to
  planar solid tri/quad surface facets with displacement interpolation.
- Solid patch to shell edge coupling through a 6-DOF RBE3-style remote point.
- Static full-KKT vs CB-projected-KKT regression for solid-patch-to-shell-edge
  RBE3 coupling with explicit source-copy and remote-point extra DOFs.
- Shell/solid cantilever benchmark against an Euler-Bernoulli beam estimate.
- Shell-solid tutorials and benchmarks accept the shell `shear_mode` setting.

Backend support:

| Feature | SciPy/NumPy path | JAX path | Current note |
|---|---|---|---|
| Plate stiffness/mass/load assembly | `format="csr"` / `"dense"`, NumPy load vectors | `format="fluxsparse"`, JAX load vectors | `shear_mode` is section-level and backend-neutral; mass requires `section.rho`. |
| Shell stiffness/mass/load assembly | `format="csr"` / `"dense"`, NumPy load vectors | `format="fluxsparse"`, JAX load vectors | 3D shell coordinates are local planar frames transformed to global DOFs; mass requires `section.rho`. |
| Coincident shell-solid tie | `NumpyCoupledSystemBuilder.add_dof_tie_constraint` | Same DOF rows are compatible with `JAXCoupledSystemBuilder` | Translations only. |
| Nonmatching shell-solid tie | CSR matrix from `shell_solid_nonmatching_translational_tie_matrix` | Use the same matrix as a dense/JAX array with `add_constraint_matrix_dof` | Node-to-surface interpolation; not mortar. |
| Solid patch to shell edge RBE3 coupling | `NumpyCoupledSystemBuilder.add_distributed_coupling` tutorials | JAX builder has matching distributed-coupling tests | RBE3-style weighted least-squares remote reconstruction; generated rows are checked for force/moment balance. |

Important limits:

- Kirchhoff/C1 plate elements are not implemented.
- The shell is a linear Reissner-Mindlin Q4 shell built from local planar element frames; it is not a geometrically nonlinear curved-shell formulation.
- The `mitc4` option is a small-strain, planar Q4 assumed-shear implementation; it is not a full general shell MITC library with curved geometry support.
- Drilling rotation uses a small diagonal stabilization.
- Shell drilling rotation mass uses the same through-thickness rotary inertia
  scale as the bending rotations for dynamic regularity; it is not a calibrated
  drilling inertia model.
- Direct solid-shell rotational continuity is not available because solid nodes do not have rotational DOFs. Use the RBE3 patch coupling path when average rotation transfer is needed.
- Nonmatching shell-solid ties currently support node-to-surface translational
  interpolation on planar tri/quad solid facets; this is not a mortar coupling
  and does not transfer rotational continuity.
- Low-order solid Hex bending can be much stiffer than the shell/beam estimate on coarse meshes; the benchmark reports this separately.

Main checks:

- `PYTHONPATH=src pytest -q src/tests/test_plate.py src/tests/test_solid_shell_coupling.py src/tests/test_shell_solid_benchmark.py`
- `PYTHONPATH=src pytest -q src/tests/test_cb_structural_elements.py`
- `PYTHONPATH=src pytest -q src/tests/test_jax_coupled_system.py -k "constraint_matrix or dof_tie or rbe3 or distributed_coupling or nonmatching_shell_solid_tie"`
- `PYTHONPATH=src pytest -q src/tests/test_plate_shear_locking_benchmark.py`
- `PYTHONPATH=src pytest -q src/tests/test_shell_shear_locking_benchmark.py`
- `PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_plate_shell_modes.py`
- `PYTHONPATH=src python tutorials/elasticity/mindlin_plate_shear_locking_benchmark.py`
- `PYTHONPATH=src python tutorials/elasticity/flat_shell_shear_locking_benchmark.py --tilt-z 0.25`
- `PYTHONPATH=src python tutorials/elasticity/flat_shell_cantilever.py --format fluxsparse --tilt-z 0.2`
- `PYTHONPATH=src python tutorials/elasticity/flat_shell_cantilever.py --format fluxsparse --tilt-z 0.2 --shear-mode mitc4`
- `PYTHONPATH=src python tutorials/elasticity/solid_shell_translational_tie.py --nx 2 --ny 1 --nz 1 --pressure-z -1.0`
- `PYTHONPATH=src python tutorials/elasticity/solid_shell_translational_tie.py --nx 2 --ny 1 --nz 1 --pressure-z -1.0 --shear-mode mitc4`
- `PYTHONPATH=src python tutorials/elasticity/solid_shell_nonmatching_tie.py --solid-nx 2 --solid-ny 1 --solid-nz 1 --shell-nx 4 --shell-ny 2 --pressure-z -1.0 --shear-mode mitc4`
- `PYTHONPATH=src python tutorials/elasticity/solid_shell_rbe3_patch_coupling.py --shell-nx 2 --shell-ny 1 --tip-load-y -1.0`
- `PYTHONPATH=src python tutorials/elasticity/shell_solid_cantilever_benchmark.py --shell-nx 8 --shell-ny 2 --solid-nx 12 --solid-ny 2 --solid-nz 2 --thickness 0.08 --tip-load-z -100.0`

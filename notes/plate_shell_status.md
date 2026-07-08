# Plate/Shell Status

Current branch: `feature/plate-shell-elements`.

Implemented:

- Q4 Mindlin/Reissner-Mindlin plate helpers with 3 DOF/node: `w, rx, ry`.
- Q4 Reissner-Mindlin shell helpers with 6 DOF/node: `ux, uy, uz, rx, ry, rz`.
- 3D shell coordinates through per-element local frames and global DOF rotation.
- Q4 surface VTU output for plate/shell visualization.
- Shell-to-beam style coupling through existing 6-DOF DOF ties.
- Shell-solid translational tie for coincident shell/solid surface nodes.
- Solid patch to shell edge coupling through a 6-DOF RBE3-style remote point.
- Shell/solid cantilever benchmark against an Euler-Bernoulli beam estimate.

Important limits:

- Kirchhoff/C1 plate elements are not implemented.
- The shell is a linear Reissner-Mindlin Q4 shell built from local planar element frames; it is not a geometrically nonlinear curved-shell formulation.
- Drilling rotation uses a small diagonal stabilization.
- Direct solid-shell rotational continuity is not available because solid nodes do not have rotational DOFs. Use the RBE3 patch coupling path when average rotation transfer is needed.
- Low-order solid Hex bending can be much stiffer than the shell/beam estimate on coarse meshes; the benchmark reports this separately.

Main checks:

- `PYTHONPATH=src pytest -q src/tests/test_plate.py src/tests/test_solid_shell_coupling.py src/tests/test_shell_solid_benchmark.py`
- `PYTHONPATH=src python tutorials/elasticity/flat_shell_cantilever.py --format fluxsparse --tilt-z 0.2`
- `PYTHONPATH=src python tutorials/elasticity/solid_shell_translational_tie.py --nx 2 --ny 1 --nz 1 --pressure-z -1.0`
- `PYTHONPATH=src python tutorials/elasticity/solid_shell_rbe3_patch_coupling.py --shell-nx 2 --shell-ny 1 --tip-load-y -1.0`
- `PYTHONPATH=src python tutorials/elasticity/shell_solid_cantilever_benchmark.py --shell-nx 8 --shell-ny 2 --solid-nx 12 --solid-ny 2 --solid-nz 2 --thickness 0.08 --tip-load-z -100.0`

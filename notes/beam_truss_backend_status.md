# Beam/Truss Backend Status

## Summary

Beam and truss helpers are exposed through the normal `fluxfem as ff` API and
can choose their matrix return format explicitly.

- `format="csr"` returns SciPy CSR matrices and is the default.
- `format="fluxsparse"` returns `FluxSparseMatrix`.
- `format="dense"` returns dense NumPy matrices for small checks.
- Load vectors support `array_backend="numpy"` and `array_backend="jax"`.

The older matrix `backend="jax" | "scipy" | "numpy"` spelling remains as a
compatibility alias, but the preferred API is `format=...` because these
helpers are selecting a matrix representation, not an element-evaluation
backend.

The implemented structural helpers are dedicated line elements and lumped
connectors:

- frame2d: x-z planar Euler-Bernoulli frame, 3 DOF per node (`ux, uz, ry`)
- beam: 3D Euler-Bernoulli frame, 6 DOF per node
- truss2d/bar: x-z planar axial element, 2 translational DOF per node
- truss/bar: 3D axial element, 3 translational DOF per node
- lumped: DOF-level springs, dashpots, nodal loads, Rayleigh damping helpers

This does not yet mean beam/truss elements are mixed into the continuum
`FESpace` mesh assembly path. They are separate structural-element assemblers
that can be solved through the shared solver API.

There is now a first continuum-to-beam coupling example:

- `tutorials/elasticity/solid_beam_rbe3_coupling.py` assembles a 3D continuum
  solid block, assembles a 3D Euler-Bernoulli beam, block-diagonalizes the two
  structural matrices, copies the solid interface face into an auxiliary field,
  reduces that face to a 6-DOF RBE3 remote point, and ties the beam root 6 DOFs
  to that remote point with `add_dof_tie_constraint(...)`.
- `tutorials/elasticity/solid_truss_rbe3_coupling.py` follows the same pattern
  for a 3D truss/bar, tying only the remote translational DOFs to the truss root
  translations.

`NumpyCoupledSystemBuilder` and `JAXCoupledSystemBuilder` now expose
`add_dof_tie_constraint(master=..., slave=..., master_dofs=..., slave_dofs=...,
rhs=0.0)`, which builds the DOF-level MPC rows for
`u_master[master_dofs] - u_slave[slave_dofs] = rhs`.

## Public API

Top-level exports are available through `fluxfem as ff`:

- `ff.BeamSection`
- `ff.assemble_beam_stiffness`
- `ff.assemble_beam_mass`
- `ff.assemble_beam_point_load`
- `ff.assemble_beam_point_loads`
- `ff.assemble_beam_uniform_load`
- `ff.assemble_frame2d_stiffness`
- `ff.assemble_frame2d_mass`
- `ff.assemble_frame2d_point_load`
- `ff.assemble_frame2d_point_loads`
- `ff.assemble_frame2d_uniform_load`
- `ff.beam_node_dofs`
- `ff.frame2d_node_dofs`
- `ff.structured_beam_chain`
- `ff.structured_frame2d_chain`
- `ff.TrussSection`
- `ff.assemble_truss_stiffness`
- `ff.assemble_truss_mass`
- `ff.assemble_truss_point_load`
- `ff.assemble_truss_point_loads`
- `ff.assemble_truss_uniform_load`
- `ff.assemble_truss2d_stiffness`
- `ff.assemble_truss2d_mass`
- `ff.assemble_truss2d_point_load`
- `ff.assemble_truss2d_point_loads`
- `ff.assemble_truss2d_uniform_load`
- `ff.truss_node_dofs`
- `ff.truss2d_node_dofs`
- `ff.structured_truss_chain`
- `ff.structured_truss2d_chain`

Typical matrix format usage:

```python
K = ff.assemble_beam_stiffness(coords, conn, section, format="fluxsparse")
F = ff.assemble_beam_point_load(n_nodes, tip, force=(0.0, 0.0, -1000.0), array_backend="jax")
u, info = ff.LinearSolver(method="spsolve_jax").solve(K, F, dirichlet=bc)
```

```python
K = ff.assemble_truss_stiffness(coords, conn, section, format="csr")
F = ff.assemble_truss_point_load(n_nodes, tip, force=(1200.0, 0.0, 0.0))
u, info = ff.LinearSolver(method="spsolve").solve(K, F, dirichlet=bc)
```

## Solver Integration

`LinearSolver` can solve the returned matrix formats with Dirichlet constraints:

- `format="csr"` pairs naturally with `method="spsolve"`.
- `format="fluxsparse"` pairs with `method="spsolve_jax"` when a JAX path is desired.
- `format="dense"` also works with `method="spsolve"` for small dense checks.

SciPy sparse Dirichlet condensation/enforcement is now handled directly, so a
SciPy CSR matrix returned by the structural assemblers can go through
`LinearSolver(..., dirichlet_mode="condense")`.

## Tutorials Updated

The following tutorials accept `--format {csr,fluxsparse,dense}` and
`--solver auto`:

- `tutorials/elasticity/beam_cantilever.py`
- `tutorials/elasticity/frame2d_cantilever.py`
- `tutorials/elasticity/beam_point_load.py`
- `tutorials/elasticity/beam_uniform_load.py`
- `tutorials/elasticity/solid_beam_rbe3_coupling.py`
- `tutorials/elasticity/truss2d_bar_cantilever.py`
- `tutorials/elasticity/truss_bar_cantilever.py`
- `tutorials/elasticity/solid_truss_rbe3_coupling.py`
- `tutorials/elasticity/truss_uniform_load.py`

`--solver auto` selects `spsolve_jax` for `format="fluxsparse"` and `spsolve`
for `format="csr"` / `format="dense"`.

## Verification

Commands run successfully:

```bash
PYTHONPATH=src python tutorials/elasticity/beam_cantilever.py --format fluxsparse
PYTHONPATH=src python tutorials/elasticity/beam_cantilever.py --format csr
PYTHONPATH=src python tutorials/elasticity/beam_cantilever.py --format dense
PYTHONPATH=src python tutorials/elasticity/truss_bar_cantilever.py --format fluxsparse
PYTHONPATH=src python tutorials/elasticity/truss_bar_cantilever.py --format csr
PYTHONPATH=src python tutorials/elasticity/truss_bar_cantilever.py --format dense
PYTHONPATH=src python tutorials/elasticity/frame2d_cantilever.py
PYTHONPATH=src python tutorials/elasticity/truss2d_bar_cantilever.py
PYTHONPATH=src python tutorials/elasticity/solid_beam_rbe3_coupling.py
PYTHONPATH=src python tutorials/elasticity/solid_truss_rbe3_coupling.py
PYTHONPATH=src pytest -q src/tests/test_beam.py src/tests/test_truss.py src/tests/test_lumped.py
PYTHONPATH=src pytest -q src/tests/test_structural_2d.py src/tests/test_beam.py src/tests/test_truss.py src/tests/test_lumped.py
PYTHONPATH=src pytest -q src/tests/test_solid_beam_coupling.py src/tests/test_rbe_constraints.py src/tests/test_structural_2d.py
PYTHONPATH=src pytest -q src/tests/test_rbe_constraints.py src/tests/test_solid_beam_coupling.py src/tests/test_jax_coupled_system.py -k "dof_tie_constraint or rbe_constraints or solid_beam"
PYTHONPATH=src pytest -q src/tests/test_solid_truss_coupling.py src/tests/test_solid_beam_coupling.py
```

The latest targeted DOF-tie/coupling check was `14 passed` with one existing JAX
deprecation warning from `core/space.py`.

## Remaining Work

Useful next steps:

- decide whether to add a coupled structural/continuum assembly path,
- consider Timoshenko beam elements if shear deformation is needed.

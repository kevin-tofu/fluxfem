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

The implemented structural helpers are 3D dedicated elements:

- beam: 3D Euler-Bernoulli frame, 6 DOF per node
- truss/bar: 3D axial element, 3 translational DOF per node
- lumped: DOF-level springs, dashpots, nodal loads, Rayleigh damping helpers

This does not yet mean beam/truss elements are mixed into the continuum
`FESpace` mesh assembly path. They are separate structural-element assemblers
that can be solved through the shared solver API.

## Public API

Top-level exports are available through `fluxfem as ff`:

- `ff.BeamSection`
- `ff.assemble_beam_stiffness`
- `ff.assemble_beam_mass`
- `ff.assemble_beam_point_load`
- `ff.assemble_beam_point_loads`
- `ff.assemble_beam_uniform_load`
- `ff.beam_node_dofs`
- `ff.structured_beam_chain`
- `ff.TrussSection`
- `ff.assemble_truss_stiffness`
- `ff.assemble_truss_mass`
- `ff.assemble_truss_point_load`
- `ff.assemble_truss_point_loads`
- `ff.assemble_truss_uniform_load`
- `ff.truss_node_dofs`
- `ff.structured_truss_chain`

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
- `tutorials/elasticity/beam_point_load.py`
- `tutorials/elasticity/beam_uniform_load.py`
- `tutorials/elasticity/truss_bar_cantilever.py`
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
PYTHONPATH=src pytest -q src/tests/test_beam.py src/tests/test_truss.py src/tests/test_lumped.py
```

The structural helper test result was `31 passed`.

## Remaining Work

Useful next steps:

- document the matrix format choice in user-facing docs,
- decide whether to add a coupled structural/continuum assembly path,
- add 2D convenience wrappers if many examples are planar,
- consider Timoshenko beam elements if shear deformation is needed.

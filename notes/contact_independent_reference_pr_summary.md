# Contact independent reference branch summary

Branch: `feature/contact-independent-reference`

This branch strengthens the CB contact ROM validation and documents the public
manager/facade workflow. It builds on:

- `feature/cb-contact-rom`
- `feature/contact-manager-protocols`
- `feature/surface-contact-reference`

## Main additions

### Independent normal-contact references

- Added independent NumPy reference assembly for surface-quadrature penalty
  contact:
  - contact row `B_q`,
  - gap `gap_q = gap0_q + B_q u`,
  - active residual `penalty * weight_q * gap_q * B_q`,
  - active tangent `penalty * weight_q * outer(B_q, B_q)`.
- Covered:
  - one 2D line facet,
  - multi-facet 2D line surfaces,
  - 3D quad surface-pair,
  - mixed active/inactive quadrature points.
- Compared implementation residuals and JAX AD Jacobians against the independent
  reference form.

### Search manager validation

- Added a surface-quadrature search-manager test that validates contact returned
  by `SurfaceQuadratureContactSearchManager` against the independent reference.
- Covers exact pairing/cache refresh after large slave-surface motion.

### Friction history validation

- Added independent surface-quadrature friction-history/scatter checks:
  - tangential slip,
  - stick/slip clipping,
  - quadrature-weighted friction residual scatter,
  - frozen-friction zero Jacobian.

### ReducedContactDynamics facade validation

- Added surface-quadrature friction facade-vs-manual callback equivalence.
- Confirms the public facade path matches a hand-written active Newmark setup
  for:
  - reduced Newmark state,
  - active state,
  - master facet pairing,
  - search cache,
  - advanced friction history.

### User-facing API guide

- Added `notes/contact_rom_api_guide.md`.
- Documents the recommended CB contact ROM workflow:
  - retain potential contact-surface DOFs,
  - build the CB basis,
  - choose a node-surface or surface-quadrature search manager,
  - add optional friction history,
  - solve through `ReducedContactDynamics`.

## Verification

Final branch check:

```bash
PYENV_VERSION=jaxfem PYTHONPATH=src pytest -q src/tests/test_rom.py src/tests/test_contact.py
```

Result:

```text
65 passed, 5 warnings in 40.57s
```

Representative tutorial check:

```bash
PYENV_VERSION=jaxfem PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_surface_quadrature_friction_active_newmark.py
```

The tutorial completed successfully. The current environment emits known
CUDA/cuSPARSE plugin warnings, but execution continues on CPU.

## Merge notes

- No production API breaking changes were introduced in this branch.
- The main behavioral additions are validation tests and documentation.
- The current CB basis builder still densifies internal blocks; sparse/iterative
  CB construction remains a separate scaling task.
- For large moving active contact patches, users should retain a sufficiently
  large surface band or add enrichment later.

## Suggested next branch

`feature/cb-sparse-iterative`

Primary target: replace the current dense internal solve/eigensolve path in the
CB basis builder with sparse or iterative options while preserving the validated
contact manager/facade behavior.

# JAX-Optional CB Backend Notes

## Current state

FluxFEM is not JAX-free today.

`pyproject.toml` lists `jax` and `jaxlib` as required project and Poetry
dependencies, and core solver modules import `jax` / `jax.numpy` at module
import time.  Craig-Bampton APIs also store and return `jnp.ndarray` values in
the public basis object.

This means a user cannot currently install or import the relevant FluxFEM
solver stack without JAX.

## What already works

The CB-ROM implementation can already use SciPy for some large-DOF work while
JAX remains installed as the array/autodiff backend:

- sparse stiffness and mass inputs can be accepted through SciPy CSR-like paths
  or FluxFEM sparse matrices,
- static constraint modes can use `constraint_solver="spsolve"`,
- fixed-interface modes can use `modal_solver="eigsh"`,
- sparse matrix projection preserves sparse input until the reduced dense
  matrix is formed.

This is the right release claim for the current code:

> SciPy-backed sparse Craig-Bampton construction is supported, while JAX remains
> the runtime array/autodiff backend.

Do not describe the current release as JAX-free.

## Practical limitation

Reduced matrices such as `K_r` and `M_r` are dense by nature in standard
Craig-Bampton projection.  That is acceptable for modest reduced coordinates,
but it can become a bottleneck if many retained interface DOFs or internal modes
are kept.

For very large reduced models, the better direction is to expose matrix-free
projected operators and iterative reduced solvers, not to promise sparse
`K_r`/`M_r` matrices.

## Proposed branch

Use a dedicated branch for the optional-backend work:

```bash
git switch -c feature/jax-optional-cb-backend
```

Initial scope:

1. Move JAX from mandatory runtime dependency to an optional extra only after
   import boundaries are clean.
2. Add a NumPy/SciPy-only CB basis path for linear structural ROMs.
3. Keep JAX-only APIs, such as autodiff residuals, contact Jacobians, and
   reduced-equation Newton wrappers, behind explicit lazy imports.
4. Add a no-JAX import smoke test for the package subset that is intended to be
   usable without JAX.
5. Add scipy-only CB tests covering `spsolve`, `eigsh`, projection, and
   expansion without importing `jax`.

## API direction

Prefer a small backend-neutral boundary instead of duplicating the full solver
stack.  The first useful JAX-free target is:

- build CB basis from NumPy/SciPy matrices,
- expand reduced coordinates,
- project vectors and linear operators,
- return NumPy arrays or SciPy-compatible linear operators.

Keep differentiable contact, active contact, and residual-first reduced
equations as JAX-backed features until there is a concrete non-JAX AD backend.

## First implementation slice

The immediate direction is backend selection, not removing JAX from the whole
library in one step.

`make_craig_bampton_basis(..., backend="jax")` remains the default and returns
the existing JAX-backed `CraigBamptonBasis`.  It is the right choice when the
ROM is used with autodiff residuals, contact Jacobians, active contact, or
reduced-equation Newton wrappers.

`make_craig_bampton_basis(..., backend="scipy")` returns a
`ScipyCraigBamptonBasis` with NumPy arrays and SciPy sparse linear algebra.  It
is the right choice for large linear structural ROM construction where the
inputs are sparse matrices and the user wants `spsolve` / `eigsh` without JAX
arrays in the CB basis object.

SciPy's advantage is largest in sparse full-order linear algebra.  JAX's
advantage is largest in differentiable nonlinear/contact workflows.  The public
API should therefore make the backend choice explicit rather than implying that
one backend is universally better.

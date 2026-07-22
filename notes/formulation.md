# Formulation status

This note summarizes the current Mortar/Nitsche/contact-KKT implementation state
after comparing FluxFEM with `notes/kktkit/solver-implementations-lecture`.

## Current branch state

- Work is on `dev`.
- `origin/main` was merged into `dev` by fast-forward before implementation.
- The current changes are focused on tracked FluxFEM source/tests.  The local
  `notes/kktkit/` lecture material remains a reference input.

## Implemented from the lecture notes

### Contact operator diagnostics

`ContactOperators` now has a lightweight `diagnostics` mapping for attaching
future assembly/solver metadata without changing the numerical blocks.

Penalty/Nitsche contributions now expose

```text
contact_energy(u) = 1/2 u^T K_contact u
penalty_energy(u) = contact_energy(u)
```

This covers the lecture note gap that Nitsche/penalty energy metrics were not
reported.  For nonlinear contact this is intentionally the quadratic energy of
the assembled tangent at the supplied state, not a full path-integrated
nonlinear potential.

Mortar/multiplier contributions now expose

```text
r_c(u) = B u
||r_c||_2
E_aug(u) = 1/2 rho ||B u||_2^2
```

This makes compatibility error and augmented-Lagrangian penalty energy available
directly from `MultiplierContactContribution`.

### Constraint matrix diagnostics

FluxFEM now has `ContactConstraintDiagnostics` and
`contact_constraint_matrix_diagnostics(B)`.

The reported fields match the lecture's useful mortar checks:

- shape and nnz
- zero row count
- row norm min/max/mean
- estimated rank
- rank deficiency
- condition number estimate
- leading singular values

`MultiplierContactContribution.constraint_diagnostics()` calls the same helper
for the assembled `B` matrix.

This does not remove redundant constraints.  It makes rank/scaling issues
observable before choosing coarse, SVD, or QR-style reduction.

### Constraint quality policy

FluxFEM now also has an opt-in quality-policy layer:

```python
report = assess_contact_constraint_quality(
    B,
    max_zero_rows=0,
    max_rank_deficiency=0,
    max_condition_number=1.0e6,
)
```

`report.status` is `pass`, `warn`, or `fail`.  The default policy fails on zero
rows and rank deficiency, while condition-number and minimum-row-norm checks are
enabled only when thresholds are provided.  Each issue also carries an advisory
`hint` that points to likely next checks such as overlap selection, coarser or
rank-reduced multiplier spaces, row scaling, and block-scaled KKT diagnostics.
The same path is available from assembled multiplier contributions:

```python
report = mortar_ops.constraint_quality(max_condition_number=1.0e6)
```

This remains opt-in and does not alter assembly defaults.

### KKT block scaling

`ContactKKTSolveConfig` now supports

```python
ContactKKTSolveConfig(
    backend="numpy",
    numpy_solver="block_scaled",
    n_primal=n_u,
)
```

The scaling follows the lecture:

```text
D A D y = D b
x = D y

d_u,i = 1 / sqrt(|K_ii|)
d_lambda,j = 1 / ||B_j diag(d_u)||_2
```

This path is implemented for the NumPy/SciPy solver backend and works for dense
or sparse inputs.  `n_primal` is required because the primal/dual split is not
recoverable from an arbitrary KKT matrix alone.

The non-breaking diagnostic entry point is

```python
result = solve_contact_kkt_with_info(A, b, config=cfg)
x = result.solution
info = result.info
```

For the NumPy block-scaled path, `info` reports primal/dual scaling ranges,
unscaled residuals, scaled residuals, and row/column norm ranges before and
after scaling.

### Nitsche supermesh public helper

FluxFEM now exposes a public helper for assembling pair-Nitsche contact terms on
the prepared contact supermesh:

```python
ops = assemble_pair_nitsche_supermesh(
    contact,
    params,
    sparse=False,
    use_penalty=1.0,
    use_traction=0.0,
)
K_contact = ops.jacobian
```

Prepared contacts also expose the same path as a method:

```python
ops = contact.assemble_pair_nitsche(params, sparse=False)
```

and the generic contact-operator entry point routes to it when no explicit weak
form/state is provided:

```python
ops = assemble_contact_operators(
    contact,
    formulation="pair_nitsche_penalty",
    params=params,
)
```

The helper builds the standard symmetric pair-Nitsche bilinear

```text
jump(u) = u_master - u_slave
t(u) = 1/2 (sigma(u_master) n + sigma(u_slave) n)

a_Gamma(u, v)
  = alpha / h (v_master - v_slave) dot jump(u)
    - v_master dot t(u)
    + v_slave dot t(u)
    - 1/2 sigma(v_master)n dot jump(u)
    - 1/2 sigma(v_slave)n dot jump(u)
```

and routes it through the existing `ContactSurfaceSpace` /
`OneToManyContactSurfaceSpace` supermesh integration path.  The return value is
a `PenaltyContactContribution` with `formulation="pair_nitsche_penalty"` and
diagnostics including the number of supermesh triangles when available.

The first test milestone is complete:

- penalty-only (`use_traction=0.0`) public-helper, method, and
  `assemble_contact_operators` routing parity against the direct tagged bilinear
  on a tet4 contact pair
- public API parity against scikit-fem for penalty-only and full
  penalty-plus-traction tet4 contact
- public API parity against scikit-fem for a nonmatching split-tet interface
  where the master contact face is one triangle and the slave contact face is
  split into two triangles over the same overlap
- lower-level direct/symbolic path parity against scikit-fem for tet4 and hex27

## Existing functionality already close to the lecture

FluxFEM already has several pieces that overlap with the kktkit lecture:

- supermesh contact integration infrastructure
- P0, active P0, and supermesh P0 mortar-like multiplier spaces
- dual-nodal and coarse-dual mortar choices
- SVD/QR-like algebraic row projection for coarse multipliers
- JAX and NumPy builder paths for explicit penalty and multiplier contact
- PETSc forwarding hooks for KKT solve configuration

The current work therefore focused on observability and solver conditioning
rather than rewriting contact assembly.

## Tests added or extended

Focused tests cover:

- penalty/contact energy
- mortar compatibility residual and augmented energy
- constraint diagnostics row norms and rank deficiency
- opt-in constraint quality pass/warn/fail policy
- NumPy block-scaled KKT solve matching the direct solve
- `solve_contact_kkt_with_info` reporting block-scaling ranges and residuals
- validation that block scaling requires `n_primal`
- public pair-Nitsche supermesh helper matching the direct tagged bilinear
- public pair-Nitsche supermesh API matching scikit-fem for matching and
  nonmatching tet4 interfaces
- a runnable Mortar/Nitsche split-tet supermesh comparison demo
- a runnable nonmatching hex fixture/workpiece Mortar/Nitsche diagnostics demo

The checked command was:

```bash
PYTHONPATH=src pytest \
  src/tests/test_contact_kkt_autodiff.py \
  src/tests/test_contact_spaces.py \
  src/tests/test_contact_interface_nitsche_vs_skfem.py \
  -q
```

Current result:

```text
85 passed, 3 warnings
```

The warnings are existing float32/JAX and deprecated compatibility-path
warnings.

The comparison demo command was also checked:

```bash
PYTHONPATH=src python tutorials/contact/mortar_nitsche_supermesh_comparison.py
```

Current output includes:

```text
supermesh triangles: 2
mortar B shape: (6, 27)
mortar estimated rank: 6
mortar rank deficiency: 0
mortar quality status: pass
nitsche penalty K shape: (27, 27)
nitsche full K shape: (27, 27)
KKT solver: block_scaled
KKT residual norm: 6.942e-19
KKT scaled row norm range: ('1.000e+00', '1.852e+00')
```

The larger comparison demo command was also checked:

```bash
PYTHONPATH=src python tutorials/contact/mortar_nitsche_fixture_workpiece_diagnostics.py
```

Current output includes:

```text
fixture facets: 4
workpiece facets: 9
supermesh triangles: 32
mortar B shape: (96, 150)
mortar estimated rank: 63
mortar rank deficiency: 33
mortar quality status: fail
nitsche penalty K shape: (150, 150)
nitsche full K shape: (150, 150)
KKT solver: block_scaled
KKT scaled row norm range: ('1.000e+00', '2.631e+00')
```

This larger nonmatching case intentionally exposes rank deficiency in the
supermesh P0 multiplier space.  That makes it a useful diagnostics example:
the split-tet smoke case passes, while the larger fixture/workpiece case shows
when quality policy starts asking for a coarser or rank-reduced multiplier.

## Remaining formulation gaps

### 1. CI smoke-test selection

The split-tet and fixture/workpiece demos are both runnable.  The next decision
is whether to make either of them part of CI:

- split-tet: fast pass case for public API and KKT diagnostics
- fixture/workpiece: larger diagnostic case that intentionally reports a rank
  deficiency

The fixture/workpiece case should probably remain a tutorial unless the quality
policy is configured to expect and assert the current fail status.

### 2. Sparse rank-revealing reduction

The lecture notes call out dense local SVD/QR as prototype-level for large
contact pairs.  FluxFEM currently has algebraic coarse projection paths, but a
true sparse rank-revealing reduction is still future work.

This is lower priority than CI smoke-test selection unless large contact pairs
become the immediate bottleneck.

## Recommended next step

Decide CI smoke-test coverage next.

Reasoning:

- diagnostics, block-scaled solve, public pair-Nitsche helper, method access,
  and generic routing now cover most of the low-risk lecture items
- matching and nonmatching public API paths are now covered by scikit-fem
  reference tests
- the split-tet demo now shows how to inspect Mortar and Nitsche on the same
  overlap
- block-scaled KKT diagnostics now show scaling ranges and solve residuals
- constraint quality now gives explicit pass/warn/fail policy without changing
  default assembly behavior
- the larger fixture/workpiece demo confirms the diagnostics stay useful on a
  less toy contact geometry
- quality issues now include advisory hints without making automatic assembly
  choices
- the next missing piece is deciding which diagnostics should be asserted in CI

Suggested first milestone:

```text
Add a small CI-friendly test for the split-tet demo's run_demo() output: status
pass, zero rank deficiency, finite KKT residual, and expected supermesh count.
```

Keep the fixture/workpiece diagnostics as a tutorial unless we explicitly assert
its current rank-deficient quality status.

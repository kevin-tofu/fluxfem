# Gridap-Style Trial/Test API Plan

## Goal

Add a public API that can express:

- standard Galerkin (`U == V`)
- Petrov-Galerkin (`U != V`)
- mixed formulations with named spaces
- future surface/contact reuse

without breaking the current `compile_residual(...)` and `space.assemble_*` workflows.

## Current State

The current weak-form DSL already distinguishes symbolic roles:

- `test_ref(...)`
- `unknown_ref(...)`
- `trial` and `test` fields inside `FormContext`

But for the single-space volume path, `space.build_form_contexts()` currently sets:

- `trial = test`

So the API can distinguish `u` vs `v` symbolically, but not yet as two different FE spaces in the public single-field interface.

## Current Implementation Status

The following are now implemented for volume problems:

- `NamedSpace(name, space)`
- `LinearSpaces(test=...)`
- `BilinearSpaces(test=..., trial=...)`
- `ResidualSpaces(test=..., unknown=...)`
- `JacobianSpaces(test=..., trial=...)`
- `build_form_contexts_pair(...)`
- `FluxSparseOperator` for rectangular/square operator assembly

The following are now implemented for mixed/contact specs:

- `MixedSpaces({...}).to_fe_space()`
- `ContactSpaces(master=..., slave=...).to_contact_surface_space()`
- `ContactGroupSpaces(master=..., slaves=[...]).to_contact_surface_space()`

Current public behavior:

- `assemble_linear_form(...)` accepts `LinearSpaces`
- `assemble_bilinear_form(...)` accepts `BilinearSpaces`
- `assemble_residual(...)` accepts `ResidualSpaces`
- `assemble_jacobian(...)` accepts `JacobianSpaces`
- same-space cases preserve parity with the legacy single-space path
- mixed public naming can be expressed through `MixedSpaces`
- contact public role binding can be expressed through `ContactSpaces`
- one-to-many contact public role binding can be expressed through `ContactGroupSpaces`

Compatibility status:

- legacy single-space APIs remain supported
- dict-based role passing is retained only as a compatibility path
- new examples/tests should prefer the `*Spaces` family
- `assemble_bilinear_form_pg(...)` is retained as a compatibility helper, not the preferred public entry point

## Deprecation Stance

The current direction is not to remove the old API wholesale.

Instead:

- keep the shortest single-space Galerkin entry points
- standardize new user-facing code on the role-explicit `*Spaces` family
- deprecate only transitional or redundant public entry points

### Deprecation Candidates

These should move toward deprecation first:

- `assemble_bilinear_form_pg(...)`
- dict-based role passing such as `{"test": ..., "trial": ...}`
- examples/docs that construct `MixedFESpace(...)` directly when `MixedSpaces(...).to_fe_space()` is sufficient
- examples/docs that construct `ContactSurfaceSpace.from_sides(...)` or `OneToManyContactSurfaceSpace.from_sides(...)`
  directly when `ContactSpaces(...)` or `ContactGroupSpaces(...)` is sufficient

### Keep Supported

These should remain supported as the shortest same-space path:

- `space.assemble_linear_form(...)`
- `space.assemble_bilinear_form(...)`
- `space.assemble_residual(...)`
- `space.assemble_jacobian(...)`
- `space.assemble_bilinear_linear_pair(...)` as a convenience wrapper over
  `assemble_bilinear_form(...)` + `assemble_linear_form(...)`

Rationale:

- they remain the simplest API for standard `U == V` Galerkin problems
- they align naturally with the existing residual/Jacobian workflows
- removing them would make the common single-space path more verbose without enough gain
- for the paired bilinear/linear case, a dedicated `PairSpaces` abstraction is
  not justified yet; the existing pair API is better treated as sugar over the
  role-explicit building blocks

### Low-Level APIs To Keep For Now

The following should remain available as low-level building blocks:

- `MixedFESpace(...)`
- `ContactSurfaceSpace.from_facets(...)`
- `ContactSurfaceSpace.from_sides(...)`
- `OneToManyContactSurfaceSpace.from_sides(...)`
- `ContactSide.from_facets(...)`

These are still useful for internal code, tests, and advanced workflows even if they
are no longer the preferred public tutorial path.

At the current stage, these should stay warning-free at runtime.
The preferred-vs-low-level distinction should be enforced through docs,
tutorials, and examples first, not through constructor warnings.

### Recommended Removal Order

1. Mark transitional helpers as deprecated.
2. Remove them from tutorials/examples/docs first.
3. Keep single-space APIs as supported long-term.
4. Revisit low-level constructor deprecation only after the `*Spaces` family is fully proven in real workflows.
5. Do not add runtime warnings for low-level constructors until there is a
   concrete migration burden reduction to justify the noise.

## Design Direction

Prefer a layered design:

1. Keep the current low-level implementation and DSL.
2. Add a thin public layer for named trial/test spaces.
3. Add sugar later if desired.

This keeps internal assembly stable while improving FE semantics at the API boundary.

## Recommended Public API

### Phase 1: Named spaces

Introduce a minimal named-space API:

```python
U = ff.NamedSpace("U", trial_space)
V = ff.NamedSpace("V", test_space)

u = ff.trial_ref(space="U")
v = ff.test_ref(space="V")
p = ff.param_ref()
```

Assembly should accept an explicit mapping from names to spaces.

Possible target shape:

```python
form = ff.compile_bilinear(my_form, trial_space="U", test_space="V")
A = ff.assemble_bilinear_form({"U": U, "V": V}, form, params)
```

This is the preferred first step because it aligns with the existing `space=...` mechanism already used in mixed and contact code.

### Phase 1.5: `*Spaces` family

Use a small family of explicit spec objects instead of dict-based role passing.

Recommended naming:

```python
ff.LinearSpaces(test=V)
ff.BilinearSpaces(test=V, trial=U)
ff.ResidualSpaces(test=V, unknown=U)
ff.JacobianSpaces(test=V, trial=U)
```

This keeps public semantics explicit while reusing the current context machinery.

### Phase 2: Sugar API

Optionally add a Gridap-like wrapper later:

```python
U = ff.TrialSpace(space_u, name="U")
V = ff.TestSpace(space_v, name="V")

u = U.trial()
v = V.test()
```

This should be implemented as sugar over the named-space API, not as a separate assembly mechanism.

## Non-Goals

Do not:

- break `space.assemble_bilinear_form(ff.diffusion_form, ...)`
- remove `compile_residual(...)`
- force all users into named spaces
- rewrite the internal contact/mixed assembly model first
- deprecate the current public API before the new path is proven

The new API should be additive.

## Internal Constraints

The implementation should reuse existing structures where possible:

- `FieldPair(test, trial, unknown)`
- `FormContext(..., spaces=..., default_space=...)`
- weak-form resolution through `space_key`

This suggests the main missing piece is not expression support, but a public assembly path that builds a context with distinct trial/test spaces.

## Existing Pieces

The following already exist:

- `trial_ref(...)`
- `test_ref(...)`
- `compile_bilinear(...)`
- `FieldPair(test, trial, unknown)`

So the gap is no longer symbolic expression support. The gap is public assembly over distinct test/trial spaces.

## Proposed Implementation Order

1. Add a named-space container type.
2. Add a volume bilinear assembly path that accepts separate test/trial spaces.
3. Build public wrappers around that path.
4. Add rectangular-operator tests.
5. Only after that, consider sugar such as `space.test()` / `space.trial()`.

## Concrete Task Breakdown

### Task 0: Confirm assumptions in code

Check and document the exact assumptions that currently force `trial = test`.

Relevant files:

- `src/fluxfem/core/space.py`
- `src/fluxfem/core/assembly.py`
- `src/fluxfem/core/assembly_jacobian.py`
- `src/fluxfem/core/forms.py`

Expected outcome:

- a short note on which paths assume square operators
- a short note on which paths already carry both `ctx.test` and `ctx.trial`

### Task 1: Introduce named-space container

Add a tiny public object such as:

```python
ff.NamedSpace(name: str, space: FESpaceClosure)
```

Requirements:

- no new assembly logic yet
- only validation and storage
- exported from `fluxfem.__init__`

Expected outcome:

- users can explicitly map symbolic space names to FE spaces

Status:

- done

### Task 2: Add distinct test/trial context builder

Implement an internal builder that constructs a `FormContext` from:

- `test_space`
- `trial_space`

instead of from a single `space`.

Requirements:

- preserve the current single-space builder unchanged
- build `ctx.test` from `test_space`
- build `ctx.trial` and `ctx.unknown` from `trial_space`
- attach both through `spaces={...}` when named spaces are provided

Likely touch points:

- `src/fluxfem/core/space.py`
- possibly a new helper module if that keeps concerns cleaner

Expected outcome:

- element-level kernels can evaluate `v` on one basis and `u` on another basis

Status:

- done for aligned volume spaces

### Task 3: Add rectangular bilinear assembly path

Implement a new assembly entry point for bilinear forms with separate spaces.

Possible shape:

```python
A = ff.assemble_bilinear_form_pg(
    test_space=V,
    trial_space=U,
    form=form,
    params=params,
)
```

Requirements:

- return a sparse matrix wrapper
- support rectangular shapes `(n_test, n_trial)`
- avoid reusing Jacobian code paths that assume square matrices
- make bilinear assembly, not Jacobian assembly, the first-class route for `U != V`

Likely touch points:

- `src/fluxfem/core/assembly.py`
- `src/fluxfem/solver/sparse.py`

Note:

`FluxSparseMatrix` is currently square-oriented (`n_dofs`). This task may require:

- extending it to `(n_rows, n_cols)`, or
- adding a separate rectangular sparse wrapper

This is the main design fork and should be decided explicitly before deeper implementation.

Status:

- done for a first JAX-only path via `FluxSparseOperator`

### Task 4: Add public wrapper API

Expose the rectangular assembly path through a user-facing API.

Candidate APIs:

```python
A = ff.assemble_bilinear_form(
    {"V": ff.NamedSpace("V", V), "U": ff.NamedSpace("U", U)},
    form,
    params,
)
```

or

```python
A = ff.assemble_bilinear_form_pg(V, U, form, params)
```

Recommendation:

- implement the explicit `*_pg` helper first
- optionally add the mapping-based API later if it improves mixed/contact consistency

Status:

- `assemble_bilinear_form_pg(...)` exists as a compatibility/helper path
- main public direction is now `assemble_bilinear_form(BilinearSpaces(...), ...)`

### Task 5: Add regression and capability tests

Add tests for:

- Galerkin parity: `U == V` matches current `space.assemble_bilinear_form(...)`
- Petrov-Galerkin rectangular shape
- distinct basis orders if supported
- `trial_ref(space="U")` and `test_ref(space="V")` resolving correctly
- dense comparison on a tiny mesh

Suggested new test file:

- `src/tests/test_petrov_galerkin_api.py`

Status:

- implemented

### Task 6: Documentation and examples

Add a minimal example showing:

- current single-space API
- new Gridap-like named-space API
- when to choose each

Suggested targets:

- docstring on the new assembly function
- one small tutorial or test-as-example
- release note / changelog note if this becomes public

## Priority Order

Recommended order for implementation work:

1. decide sparse rectangular matrix representation
2. implement internal distinct test/trial context builder
3. implement rectangular bilinear assembly
4. add public API wrapper
5. add tests
6. add sugar

Updated priority after current implementation:

1. keep docs/examples aligned to `*Spaces`
2. minimize public reliance on dict compatibility
3. extend the same family concept to mixed/contact
4. decide whether `assemble_bilinear_form_pg(...)` should remain public long-term
5. add sugar only after mixed/contact direction is stable

## Decision Points

These decisions should be made before coding too far:

### Decision A: Sparse matrix representation

Choose one:

- extend `FluxSparseMatrix` to support rectangular shape
- add `FluxSparseOperator` for generic rectangular operators

Recommendation:

- use a separate rectangular operator type if minimizing breakage is more important
- extend `FluxSparseMatrix` only if square-only assumptions are already shallow

#### Chosen direction for first implementation

Use a separate rectangular sparse wrapper first.

Working name:

```python
FluxSparseOperator
```

Reasoning:

- the immediate goal is distinct test/trial bilinear assembly
- current square-only solver paths should not be destabilized
- rectangular support should be introduced in the narrowest possible surface area first

This means:

- keep `FluxSparseMatrix` as the square matrix type for current workflows
- return `FluxSparseOperator` from new Petrov-Galerkin / distinct trial-test assembly paths
- defer any unification of square and rectangular sparse wrappers until usage is validated

#### Minimal `FluxSparseOperator` interface

The first implementation only needs:

```python
class FluxSparseOperator:
    rows
    cols
    data
    shape

    def to_coo(self): ...
    def to_dense(self): ...
    def to_csr(self): ...
    def matvec(self, x): ...
    def rmatvec(self, y): ...
```

Notes:

- `shape` should be `(n_rows, n_cols)`
- `matvec(x)` computes `A @ x`
- `rmatvec(y)` computes `A.T @ y`
- direct solver integration is not required in the first phase

#### Explicit non-goals for `FluxSparseOperator` v1

Do not require in the first implementation:

- Dirichlet elimination helpers
- Newton / Jacobian integration
- preconditioner support
- block factorization support
- PETSc or JAX sparse solver bindings

Those can be added later if rectangular operators become broadly used.

#### Immediate code impact

Likely files:

- `src/fluxfem/solver/sparse.py`
- `src/fluxfem/solver/__init__.py`
- new Petrov-Galerkin assembly entry point in core assembly

Likely tests:

- `to_dense()` parity on tiny examples
- `matvec()` and `rmatvec()` sanity checks
- rectangular shape checks

### Decision B: Public API entry point

Choose one:

- a new explicit Petrov-Galerkin assembly function
- overload the current `assemble_bilinear_form(...)`

Recommendation:

- start with a new explicit function
- only overload the current one after semantics are stable

### Decision C: Scope of first implementation

Choose one:

- scalar volume only
- scalar + vector volume
- include surfaces/contact immediately

Recommendation:

- start with scalar/vector volume only
- reuse the same abstraction for surfaces later

## Milestone Plan

### Milestone 1

- named-space object
- rectangular sparse operator decision
- internal context prototype

### Milestone 2

- working rectangular bilinear assembly on tiny volume problems
- parity tests for `U == V`

### Milestone 3

- public API polish
- docs
- optional sugar layer

## Expected Behavior

### Standard Galerkin

```python
V = ff.NamedSpace("V", space)
u = ff.trial_ref(space="V")
v = ff.test_ref(space="V")
```

This should behave the same as the current single-space path.

### Petrov-Galerkin

```python
U = ff.NamedSpace("U", trial_space)
V = ff.NamedSpace("V", test_space)
u = ff.trial_ref(space="U")
v = ff.test_ref(space="V")
```

This should assemble a rectangular or otherwise non-identical test/trial operator if the underlying spaces differ.

## Risks

- Rectangular operators may require widening assumptions currently hard-coded for square Jacobians.
- Existing `assemble_jacobian` semantics are tied to residual differentiation and may not be the right public surface for `U != V`.
- Bilinear-form assembly likely needs to be the first-class route for distinct trial/test spaces.

## Practical Guidance

Use the current API when:

- solving standard single-field Galerkin problems
- using nonlinear residual/Jacobian assembly on one space
- writing mixed/contact forms that already fit the named-field model

Use the new API, once implemented, when:

- trial and test spaces are conceptually different
- a Gridap-like formulation is desired
- Petrov-Galerkin semantics should be explicit in code review and maintenance

## Mixed And Contact Outlook

The current codebase already contains most of the machinery needed to extend this design:

- `FieldPair(test, trial, unknown)`
- `ctx.spaces`
- `default_space`
- mixed residual bindings keyed by field/space names
- contact assembly building named two-side contexts

That means the long-term direction can be unified.

### Mixed

Mixed already behaves close to a `*Spaces` API internally.

Current reality:

- each field already owns its own `FieldPair`
- `compile_mixed_residual(...)` already resolves symbolic refs through `space=...`

So the likely public direction is:

```python
ff.MixedSpaces({
    "u": ff.ResidualSpaces(test=Vu, unknown=Uu),
    "p": ff.ResidualSpaces(test=Vp, unknown=Up),
})
```

Current status:

- a first `MixedSpaces` public spec exists
- it currently focuses on named field-to-space binding and conversion to `MixedFESpace`
- mixed assembly itself still flows through `MixedFESpace`

Near-term recommendation:

- keep `MixedSpaces` as the public naming layer
- defer deeper mixed `*Spaces` decomposition until a concrete Petrov/mixed use case demands it

### Contact

Contact already has an even stronger notion of named sides:

- master/slave fields
- named space resolution in mixed surface residuals
- custom `spaces` dictionaries in the contact contexts

So the likely public direction is not to force volume-style `BilinearSpaces` directly onto contact.

Instead, contact should probably grow a parallel but aligned API family, for example:

```python
ff.ContactSpaces(master=A, slave=B)
ff.ContactResidualSpaces(master=A, slave=B)
```

The key idea is:

- keep the role-explicit public API
- keep the `FieldPair`/`spaces`-based internals
- do not collapse contact into a fake volume bilinear API

Current status:

- a first `ContactSpaces(master=..., slave=...)` spec exists
- it currently focuses on public role binding and conversion to `ContactSurfaceSpace`
- a first `ContactGroupSpaces(master=..., slaves=[...])` spec also exists
- deeper contact-family decomposition beyond pair/one-to-many is not yet added

### Design Principle Across Modules

The API family should align semantically, not necessarily use the same exact class everywhere.

Recommended alignment:

- volume linear/bilinear: `LinearSpaces`, `BilinearSpaces`
- volume nonlinear: `ResidualSpaces`, `JacobianSpaces`
- mixed: `MixedSpaces` built from those primitives
- contact/surface: role-specific specs that still resolve to named `spaces`/`FieldPair`

This preserves conceptual unity without forcing unnatural abstractions onto contact.

## Next Concrete Steps

### Short term

- keep new tests/examples on `LinearSpaces` / `BilinearSpaces` / `ResidualSpaces` / `JacobianSpaces`
- avoid introducing new dict-based APIs
- document `FluxSparseOperator` as the rectangular operator type
- prefer `MixedSpaces` and `ContactSpaces` in new high-level examples

### Mixed

Introduce a public `MixedSpaces` spec that wraps per-field specs rather than exposing bare dicts.

Candidate direction:

```python
ff.MixedSpaces({
    "u": ff.ResidualSpaces(test=Vu, unknown=Uu),
    "p": ff.ResidualSpaces(test=Vp, unknown=Up),
})
```

First implementation target:

- same-space parity with current mixed assembly
- preserve current `compile_mixed_residual(...)` and binding semantics

Status:

- basic public spec implemented
- deeper mixed assembly-family integration remains future work

### Contact

Introduce a contact-specific spec family rather than forcing volume APIs onto contact.

Candidate direction:

```python
ff.ContactSpaces(master=A, slave=B)
ff.ContactBilinearSpaces(master=A, slave=B)
```

First implementation target:

- align names and role semantics with volume `*Spaces`
- keep current `ContactSurfaceSpace` assembly internals intact
- preserve current master/slave terminology at the public boundary

Status:

- pair and one-to-many public specs implemented
- next likely step is docs/examples alignment, not replacement of current contact internals

## Summary

The current design is workable internally, but its FE semantics are under-expressed at the public API level.

The right move is not a rewrite. The right move is an additive named-space layer that:

- preserves current workflows
- exposes distinct trial/test spaces explicitly
- becomes the foundation for future Gridap-like sugar

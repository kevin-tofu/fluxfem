# Reduced Equation Builder Notes

Goal: keep the existing `ReducedCoupledSystemBuilder` focused on assembled
linear structural blocks, while adding a thin residual-first layer for future
differentiable formulations.

## Motivation

Current CB-ROM workflows handle:

- multiple structural subsystems with `K`, `M`, and `f`
- per-subsystem Craig-Bampton bases
- retained interface ties, fixtures, and contact-candidate metadata
- reduced residuals for a single full-order residual via `cb.reduced_residual`

The gap is a public API for composing arbitrary reduced residual equations
across multiple reduced fields. Future formulations may be nonlinear,
state-dependent, contact-like, multiphysics, matrix-free, or autodiff-native.
Those should not have to masquerade as a global linear `K`.

## Initial Scope

Add a small `ReducedEquationBuilder` API:

- register named reduced fields
- add field-local residual blocks
- add multi-field coupling residual blocks
- assemble one global reduced residual
- obtain a global Jacobian with `jax.jacrev`
- support CB bases by registering `CraigBamptonBasis` objects

This is intentionally residual-first. Linear systems can still use the existing
coupled-system builders.

## Non-goals for the first pass

- no global Newton driver wrapper yet
- no sparse Jacobian assembly
- no time integration state manager
- no contact search ownership
- no replacement for `ReducedCoupledSystemBuilder`

## API Sketch

```python
builder = ff.ReducedEquationBuilder()
builder.register_field("part_a", basis=cb_a)
builder.register_field("part_b", basis=cb_b)

builder.add_field_residual(
    "part_a",
    lambda qa: cb_a.project_vector(Ra(cb_a.expand(qa))),
)

builder.add_coupling_residual(
    ("part_a", "part_b"),
    lambda qa, qb: {
        "part_a": k * (qa[a] - qb[b]) * ea,
        "part_b": -k * (qa[a] - qb[b]) * eb,
    },
)

problem = builder.build()
r = problem.residual(q)
j = problem.jacobian(q)
```

## Design Rule

The builder owns only field slicing and residual scattering. Physics-specific
code stays in user residual functions or higher-level formulation modules.

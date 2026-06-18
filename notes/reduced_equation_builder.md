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

## Added Solver Layer

The residual-first layer now has:

- `solve_reduced_equation` for dense Newton solves on the assembled reduced
  residual
- `reduced_equation_newmark_step` for implicit Newmark dynamics using
  `problem.residual(q, params)` as the internal reduced force
- fixed reduced-DoF support shared by static and Newmark solves
- `ReducedEquationBuilder.add_constraint` for user-defined constraint
  residuals supplied as callables or objects with `fields` and `residual`

This keeps transient dynamics compatible with field-local residuals, nonlinear
coupling residuals, CB bases, and future contact residuals.

## Contact Constraint Pattern

Contact can be expressed as a user constraint object.  For a CB-reduced field,
the object expands the reduced coordinate, evaluates the full contact residual,
and projects it back:

```python
class CBPlaneContactConstraint:
    fields = ("body",)

    def residual(self, q):
        u = cb.expand(q)
        return cb.project_vector(contact.residual(u))
```

This keeps contact-specific logic outside the builder while preserving
autodiff and the same static/Newmark solve interfaces.

## Non-goals for the first pass

- no sparse Jacobian assembly
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

builder.add_constraint(
    CustomConstraint(fields=("part_a", "part_b"))
)

problem = builder.build()
q, solve_info = ff.solve_reduced_equation(problem, q0)
next_state, step_info = ff.reduced_equation_newmark_step(
    problem, mass, damping, external_force, state, config
)
```

## Design Rule

The builder owns only field slicing and residual scattering. Physics-specific
code stays in user residual functions or higher-level formulation modules.

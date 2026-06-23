# Solver API Unification Notes

## Current concern

Solver-facing APIs have grown in parallel:

- low-level `newton_solve(...)`
- convenience `solve_nonlinear(...)`
- stateful `NonlinearAnalysis` + `NewtonSolveRunner`
- new `NonlinearConstrainedProblem`
- linear `LinearAnalysis` + `LinearSolveRunner`

This is workable internally, but it can look confusing to users unless the
recommended entry points are clear.

## Proposed public hierarchy

Use three levels and document them explicitly.

1. Problem facades for user workflows:
   - `LinearAnalysis` / `LinearSolveRunner`
   - `NonlinearAnalysis` / `NewtonSolveRunner`
   - `NonlinearConstrainedProblem`

2. Convenience functions for scripts:
   - `solve_nonlinear(...)`
   - `solve_nonlinear_constrained_kkt(...)`

3. Low-level kernels for advanced users:
   - `newton_solve(...)`
   - sparse/direct solver helpers
   - residual/Jacobian assembly functions

## Next cleanup direction

The constrained nonlinear path should eventually share configuration objects
with the existing nonlinear runner:

- reuse `NewtonLoopConfig` fields where possible,
- expose `NonlinearConstrainedProblem.solve(config=...)`,
- keep `solve_nonlinear_constrained_kkt(...)` as the lower-level implementation,
- document that `newton_solve(...)` is not the first API for application code.

This keeps exact MPC/KKT support without making the solver namespace feel like
several unrelated solvers.

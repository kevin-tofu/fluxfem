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

## Current cleanup

The constrained nonlinear path now accepts the same `NewtonLoopConfig` object
for the Newton-loop fields it can faithfully support:

- `NonlinearConstrainedProblem.solve(config=...)`,
- shared `tol`, `atol`, and `maxiter` controls,
- `load_sequence` / `n_steps` load stepping by scaling the external vector,
- selectable constrained KKT backends:
  `linear_solver="spsolve"` for direct solves, `"gmres"` for SciPy
  `LinearOperator` solves, and `"petsc_shell"` for optional PETSc KSP solves,
- `constrained_kkt_config("direct" | "scipy-gmres" | "petsc-gmres" | "petsc-ilu")`
  returns a typed `NewtonLoopConfig` for user-facing backend selection,
- the tutorial uses the config object instead of a separate one-off signature.
- the tutorial can compare a stepped solve against the direct single-step
  result with `--compare-single-step`.

Unsupported `NewtonLoopConfig` fields are rejected explicitly for this path
instead of being silently ignored:

- `line_search`,
- linear solvers other than `spsolve`, `gmres`, and `petsc_shell`.

## Next cleanup direction

The remaining work is to make more of `NewtonLoopConfig` meaningful for exact
KKT/MPC solves:

- decide whether line search is useful for the saddle-point Newton update,
- add better preconditioner presets for iterative KKT solves on large constrained systems,
- keep `solve_nonlinear_constrained_kkt(...)` as the lower-level implementation,
- document that `newton_solve(...)` is not the first API for application code.

This keeps exact MPC/KKT support without making the solver namespace feel like
several unrelated solvers.

# Nonlinear ROM Notes

## Current implementation

`NonlinearReducedFEModel` provides a direct Galerkin nonlinear FE ROM:

- full displacement: `u = Phi q`,
- reduced residual: `R_r(q) = Phi.T R(Phi q)`,
- reduced tangent: `J_r(q) = Phi.T J(Phi q) Phi`.

The full residual and tangent are still assembled with the existing nonlinear
FE assembly path. This keeps geometric nonlinear material models, autodiff, and
existing reduced-equation composition available without introducing a separate
hyper-reduction layer.

## Scope

This is suitable for small-to-moderate nonlinear deformation studies and for
checking that a complete basis reproduces the full-order solve. A low-dimensional
basis can be used immediately, but the full residual assembly cost remains.

## Next step

Large nonlinear ROMs need hyper-reduction or sampled assembly. Candidate next
steps are element sampling, DEIM/GNAT-style residual bases, or using contact and
fixture active sets to restrict nonlinear evaluations.

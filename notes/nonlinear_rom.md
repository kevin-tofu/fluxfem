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
checking that an identity-full basis reproduces the full-order solve. A
low-dimensional basis can be used immediately, but the full residual assembly
cost remains.

## Next step

Large nonlinear ROMs need hyper-reduction or sampled assembly. Candidate next
steps are element sampling, DEIM/GNAT-style residual bases, or using contact and
fixture active sets to restrict nonlinear evaluations.

## Comparison tutorial

`tutorials/nonlinear_rom/compare_geometric_nonlinear_full_vs_rom.py` compares
full geometric nonlinear FEM against the direct Galerkin ROM. Use
`--basis identity-full` as a regression check that the ROM reproduces the
full-coordinate solve, `--basis free-dofs` to remove prescribed coordinates while
retaining the full free subspace, `--basis linearized-modes` to build a practical
low-dimensional basis from the initial linearized stiffness/mass eigenmodes, and
`--basis cantilever-bending-y` as a deliberately small bending-shape ROM that
exposes projection error. By default, JSON/PNG/VTU/Markdown summary outputs are
written under
`tutorials/nonlinear_rom/results/compare_geometric_nonlinear_full_vs_rom/`.

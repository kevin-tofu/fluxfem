# Craig-Bampton ROM Port for 0.3.5

## Current scope

- `fluxfem.solver.make_craig_bampton_basis` builds a CB basis from assembled stiffness and mass matrices.
- Retained DOFs stay physical: the retained rows of the retained-coordinate block are identity.
- Internal DOFs are represented by static constraint modes and fixed-interface modes.
- Dense, SciPy sparse direct (`spsolve`), and SciPy sparse modal (`eigsh`) paths are exposed through solver options.
- `CraigBamptonBasis` is a JAX pytree and provides `expand`, `project_vector`, `project_matrix`, `reduced_residual`, and `reduced_jacobian`.

## Why this helps contact and fixtures

CB retained DOFs are the natural location for interface, contact, and remote fixture coordinates. A contact workflow can keep uncertain active-contact boundary DOFs in the retained set, while reducing only the interior. The existing 0.3.5 coupled-system and RBE3 APIs should remain the wrapper layer for fixtures and reference points; CB is the projection layer below that.

## Important limitation

The first port still stores the final basis as a dense matrix and `project_matrix` densifies the input matrix. Sparse solvers are used for partition solves and modal extraction, but very large models will need a matrix-free/sparse basis operator before this becomes the best option for production-scale DOF counts.

## Next implementation steps

1. Add a sparse/matrix-free basis operator so `Phi.T @ K @ Phi` can be assembled blockwise without materializing full dense `Phi`.
2. Add direct integration with `CoupledSystemBuilder` so retained CB coordinates can be coupled to remote/RBE3 blocks without hand-written index mapping.
3. Add a dynamic tutorial using `newmark_solve_linear` on the reduced `M_r`, `C_r`, `K_r`.
4. Add an active-contact example where candidate contact DOFs are retained and only interior DOFs are reduced.

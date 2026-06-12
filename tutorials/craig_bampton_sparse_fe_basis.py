from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def main():
    mesh = ff.StructuredHexBox(nx=6, ny=1, nz=1, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    stiffness = space.assemble_bilinear_form(ff.diffusion_form, params=1.0)
    mass = space.assemble_mass_matrix()

    coords = np.asarray(mesh.coords)
    retained_nodes = np.flatnonzero(
        np.isclose(coords[:, 0], coords[:, 0].min()) | np.isclose(coords[:, 0], coords[:, 0].max())
    )
    retained = ff.vector_dofs_from_nodes(jnp.asarray(retained_nodes, dtype=jnp.int32), dim=1)

    print("CB sparse FE basis demo")
    print(f"full DOFs: {space.n_dofs}")
    print(f"retained DOFs: {int(retained.size)}")
    print(f"stiffness nnz: {stiffness.to_csr().nnz}")
    print()
    print("n_modes  n_red  eig_error     basis_projector_error")
    print("-------  -----  ------------  ---------------------")
    for n_modes in range(1, 5):
        dense = ff.make_craig_bampton_basis(
            stiffness.to_dense(),
            mass.to_dense(),
            retained_dofs=retained,
            n_modes=n_modes,
        )
        sparse = ff.make_craig_bampton_basis(
            stiffness,
            mass,
            retained_dofs=retained,
            n_modes=n_modes,
            constraint_solver="spsolve",
            modal_solver="eigsh",
            modal_tol=1e-9,
            modal_maxiter=500,
        )
        eig_error = float(jnp.linalg.norm(sparse.eigenvalues - dense.eigenvalues))
        projector_error = float(jnp.linalg.norm(sparse.basis @ sparse.basis.T - dense.basis @ dense.basis.T))
        print(f"{n_modes:7d}  {sparse.n_reduced:5d}  {eig_error:12.6e}  {projector_error:21.6e}")


if __name__ == "__main__":
    main()

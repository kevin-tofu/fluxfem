from __future__ import annotations

import numpy as np

import fluxfem as ff


def main():
    if not ff.petsc_is_available():
        raise SystemExit("petsc4py is not installed. Install with `poetry install --extras petsc`.")

    mesh = ff.StructuredHexBox(nx=8, ny=8, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    K = space.assemble_bilinear_form(ff.diffusion_form, params=1.0)
    F = space.assemble_linear_form(ff.scalar_body_force_form, params=1.0)

    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: (
            np.isclose(pts[:, 0], 0.0)
            | np.isclose(pts[:, 0], 1.0)
            | np.isclose(pts[:, 1], 0.0)
            | np.isclose(pts[:, 1], 1.0)
        ),
        components=[0],
        dof_per_node=1,
    )
    dir_vals = np.zeros(len(dir_dofs))

    K_bc, F_bc = ff.enforce_dirichlet_sparse(K, F, dir_dofs, dir_vals)
    u = ff.petsc_solve(K_bc, F_bc, ksp_type="cg", pc_type="jacobi")

    print("solution norm:", float(np.linalg.norm(u)))
    try:
        u_jax = ff.spdirect_solve_jax(K_bc, F_bc)
        diff = float(np.linalg.norm(u - u_jax))
        print("jax diff norm:", diff)
    except Exception as exc:
        print("jax solve skipped:", exc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Linear elastic tensile bar — minimal FluxFEM demo (weak-form assembly).

- Structured hex mesh.
- Clamp at x = xmin.
- Uniform traction applied on x = xmax (+x direction).
"""

import numpy as np
import jax
import jax.numpy as jnp
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

jax.config.update("jax_enable_x64", True)


def main():
    # --------------------
    # Parameters (edit here)
    # --------------------
    dtype = jnp.float64

    # geometry / mesh
    nx, ny, nz = 12, 2, 2
    lx, ly, lz = 1.0, 0.1, 0.1
    intorder = 2

    # material
    E, nu = 210_000.0, 0.3
    D = ff.isotropic_3d_D(E, nu)

    # loads
    traction = 1.0  # applied on x = xmax, along +x

    # output
    output_vtu = None  # set a path like "result.vtu" to write VTU

    # --------------------
    # Build mesh & space
    # --------------------
    mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    # --------------------
    # Bilinear form (weak-form)
    # --------------------
    bilinear_form = ff.compile_bilinear(
        lambda v, u, D_mat: h_wf.ddot(v.sym_grad, h_wf.matmul_std(D_mat, u.sym_grad))
        * h_wf.dOmega()
    )
    K = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=V, trial=U),
        bilinear_form,
        D,
    )

    # --------------------
    # Surface traction on x = xmax (weak-form)
    # --------------------
    coords_np = np.asarray(mesh.coords)
    xmax = float(coords_np[:, 0].max())
    facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    surface = ff.make_surface_from_facets(coords_np, facets)

    surface_form = ff.LinearForm.surface(lambda v, p: (v | p) * h_wf.ds())
    traction_vec = np.array([traction, 0.0, 0.0], dtype=float)
    F = surface.assemble_linear_form_on_space(
        space, surface_form.get_compiled(), params=traction_vec
    )
    F = jnp.asarray(F, dtype=dtype)

    # --------------------
    # Dirichlet DOFs (clamp x = xmin)
    # --------------------
    xmin = float(coords_np[:, 0].min())
    dir_dofs = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    ).dofs

    # --------------------
    # Solve
    # --------------------
    solver = ff.LinearSolver(method="spsolve")
    u, _ = solver.solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dir_dofs, None),
        dirichlet_mode="condense",
    )

    u_nodes = np.asarray(u).reshape(-1, 3)
    right_nodes = mesh.axis_extrema_nodes(axis=0, side="max")
    ux_max = float(np.max(u_nodes[right_nodes, 0])) if right_nodes.size else 0.0
    ux_theory = traction * lx / E

    print(f"bar solved: dofs={space.n_dofs}, ux_max@x=L={ux_max:.6e}")
    print(f"[theory] ux=L: {ux_theory:.6e}")

    # --------------------
    # Output (optional)
    # --------------------
    if output_vtu:
        ff.write_elastic_vtu(mesh, space, u, output_vtu, compute_j=True, deformed_scale=1.0)
        print("VTU written to", output_vtu)


if __name__ == "__main__":
    main()

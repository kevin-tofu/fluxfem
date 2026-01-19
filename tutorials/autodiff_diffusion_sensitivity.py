#!/usr/bin/env python3
"""
Autodiff demo: diffusion loss sensitivity to kappa and boundary traction.

- Scalar diffusion on a structured hex mesh (dim=1).
- Dirichlet u=0 on x=xmin.
- Neumann traction (scalar flux) on x=xmax.
"""

import numpy as np
import jax
import jax.numpy as jnp
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

jax.config.update("jax_enable_x64", True)


def main():
    # Mesh / space
    nx, ny, nz = 8, 4, 1
    mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=1.0, ly=0.5, lz=0.1).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())

    # Dirichlet dofs at x=xmin
    bc = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components=[0],
        dof_per_node=1,
    )
    free_dofs = bc.free_dofs(space.n_dofs)

    # Surface for Neumann traction on x=xmax
    facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    surface = ff.make_surface_from_facets(coords, facets)

    surface_form = ff.LinearForm.surface(lambda v, p: v * p * h_wf.ds())

    K0 = jnp.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense())
    F_base = surface.assemble_linear_form_on_space(
        space, surface_form.get_compiled(), params=1.0
    )
    F_base = jnp.asarray(F_base)

    def solve_u(kappa, traction):
        K = kappa * K0
        F = traction * F_base

        K_ff = K[jnp.asarray(free_dofs)][:, jnp.asarray(free_dofs)]
        F_ff = F[jnp.asarray(free_dofs)]
        u_free = jnp.linalg.solve(K_ff, F_ff)

        u = jnp.zeros(space.n_dofs, dtype=K.dtype)
        u = u.at[jnp.asarray(free_dofs)].set(u_free)
        return u

    def loss_fn(kappa, traction):
        u = solve_u(kappa, traction)
        return 0.5 * jnp.dot(u, u)

    kappa0 = jnp.array(2.0)
    traction0 = jnp.array(1.0)

    loss_val = loss_fn(kappa0, traction0)
    grad_kappa, grad_trac = jax.grad(loss_fn, argnums=(0, 1))(kappa0, traction0)

    print("loss:", float(loss_val))
    print("d(loss)/d(kappa):", float(grad_kappa))
    print("d(loss)/d(traction):", float(grad_trac))


if __name__ == "__main__":
    main()

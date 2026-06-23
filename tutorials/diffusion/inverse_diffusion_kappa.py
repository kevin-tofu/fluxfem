#!/usr/bin/env python3
"""
Inverse problem demo: recover diffusion coefficient kappa from synthetic data.

- Scalar diffusion on a structured hex mesh (dim=1).
- Dirichlet u=0 on x=xmin.
- Neumann traction (scalar flux) on x=xmax.
- Optimize kappa to match observed displacement data.
"""

from __future__ import annotations

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
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

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
    dir_dofs = bc.dofs
    free_dofs = bc.free_dofs(space.n_dofs)
    free_dofs_j = jnp.asarray(free_dofs)

    # Surface for Neumann traction on x=xmax
    facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    surface = ff.make_surface_from_facets(coords, facets)
    surface_form = ff.LinearForm.surface(lambda v, p: v * p * h_wf.ds())

    K0 = jnp.asarray(
        ff.assemble_bilinear_form(
            ff.BilinearSpaces(test=V, trial=U),
            ff.diffusion_form,
            1.0,
        ).to_dense(),
        dtype=jnp.float64,
    )
    F_base = surface.assemble_linear_form_on_space(
        space, surface_form, params=1.0
    )
    F_base = jnp.asarray(F_base, dtype=jnp.float64)

    def solve_u(kappa, traction):
        K = kappa * K0
        F = traction * F_base
        K_ff = K[free_dofs_j][:, free_dofs_j]
        F_ff = F[free_dofs_j]
        u_free = jnp.linalg.solve(K_ff, F_ff)
        u = jnp.zeros(space.n_dofs, dtype=K.dtype)
        return u.at[free_dofs_j].set(u_free)

    solve_u_jit = jax.jit(solve_u)

    # Synthetic observations
    kappa_true = jnp.array(2.5, dtype=jnp.float64)
    traction_true = jnp.array(1.0, dtype=jnp.float64)
    u_synth = solve_u(kappa_true, traction_true)

    noise_std = 1e-4
    rng = np.random.default_rng(0)
    u_obs = u_synth + jnp.asarray(
        rng.normal(scale=noise_std, size=space.n_dofs),
        dtype=jnp.float64,
    )

    # Randomize the number of observed samples on the boundary (x=xmax).
    boundary_dofs = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmax, atol=1e-8),
        components=[0],
        dof_per_node=1,
    ).dofs
    boundary_free_dofs = np.setdiff1d(boundary_dofs, dir_dofs)
    obs_min = max(2, boundary_free_dofs.size // 4)
    obs_max = max(obs_min, boundary_free_dofs.size // 2)
    obs_count = int(rng.integers(obs_min, obs_max + 1))
    obs_idx = rng.choice(boundary_free_dofs, size=obs_count, replace=False)
    obs_idx_j = jnp.asarray(obs_idx)

    def loss_theta(theta):
        kappa = jnp.exp(theta)
        u = solve_u_jit(kappa, traction_true)
        diff = u[obs_idx_j] - u_obs[obs_idx_j]
        return 0.5 * jnp.mean(diff * diff)

    loss_theta_jit = jax.jit(loss_theta)
    grad_fn = jax.jit(jax.grad(loss_theta))

    theta = jnp.log(jnp.array(1.0, dtype=jnp.float64))  # initial guess for kappa
    lr = 0.5
    steps = 120

    for step in range(steps):
        g = grad_fn(theta)
        theta = theta - lr * g
        if step % 10 == 0 or step == steps - 1:
            loss_val = loss_theta_jit(theta)
            kappa_est = float(jnp.exp(theta))
            print(
                f"step={step:02d} loss={float(loss_val):.3e} "
                f"kappa_est={kappa_est:.6f}"
            )

    kappa_est = float(jnp.exp(theta))
    rel_err = abs(kappa_est - float(kappa_true)) / float(kappa_true)
    print("kappa_true:", float(kappa_true))
    print("kappa_est:", kappa_est)
    print("relative error:", rel_err)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Autodiff demo: linear-elastic compliance sensitivity to Young's modulus.

- 3D tensile bar (structured hex mesh).
- Dirichlet clamp on x=xmin.
- Traction on x=xmax along +x.
"""

import numpy as np
import jax
import jax.numpy as jnp
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

jax.config.update("jax_enable_x64", True)


def isotropic_3d_D_jax(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return jnp.array(
        [
            [lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=E.dtype,
    )


def main():
    # Mesh / space
    nx, ny, nz = 6, 2, 2
    mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())

    # Dirichlet clamp on x=xmin
    bc = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    free_dofs = bc.free_dofs(space.n_dofs)

    # Surface traction on x=xmax
    facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    surface = ff.make_surface_from_facets(coords, facets)
    surface_form = ff.LinearForm.surface(lambda v, p: (v | p) * h_wf.ds())
    traction_vec = jnp.array([1.0, 0.0, 0.0])

    D0 = isotropic_3d_D_jax(jnp.array(1.0), jnp.array(0.3))
    K0 = jnp.asarray(
        ff.assemble_bilinear_form(
            ff.BilinearSpaces(test=V, trial=U),
            ff.linear_elasticity_form,
            D0,
        ).to_dense()
    )
    F_base = surface.assemble_linear_form_on_space(
        space, surface_form.get_compiled(), params=np.asarray(traction_vec)
    )
    F_base = jnp.asarray(F_base)

    def solve_u(E):
        K = E * K0
        F = F_base

        K_ff = K[jnp.asarray(free_dofs)][:, jnp.asarray(free_dofs)]
        F_ff = F[jnp.asarray(free_dofs)]
        u_free = jnp.linalg.solve(K_ff, F_ff)

        u = jnp.zeros(space.n_dofs, dtype=K.dtype)
        u = u.at[jnp.asarray(free_dofs)].set(u_free)
        return u, F

    def compliance(E):
        u, F = solve_u(E)
        return jnp.dot(F, u)

    E0 = jnp.array(210_000.0)
    comp = compliance(E0)
    dcomp_dE = jax.grad(compliance)(E0)

    print("compliance:", float(comp))
    print("d(compliance)/d(E):", float(dcomp_dE))


if __name__ == "__main__":
    main()

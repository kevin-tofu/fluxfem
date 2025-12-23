"""Poisson MMS Dirichlet solve accuracy check."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff


def exact_solution(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def body_force(x_q):
    x = x_q[..., 0]
    y = x_q[..., 1]
    return 2 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def test_poisson_mms_dirichlet():
    # Unit cube with one element in z; solution independent of z.
    mesh = ff.StructuredHexBox(nx=12, ny=12, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    # Assemble stiffness
    K = space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense()

    # RHS: ∫ f N
    rhs_form = ff.make_scalar_body_force_form(body_force)
    F = space.assemble_linear_form(rhs_form, params=None)

    # Dirichlet boundary: exact solution on all boundary nodes
    coords = np.asarray(mesh.coords)
    u_exact = exact_solution(coords).astype(np.float64)
    # boundary nodes where x=0/1 or y=0/1 (z free but solution independent of z)
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
    dir_vals = u_exact[dir_dofs]
    solver = ff.LinearSolver(method="spsolve")
    u_sol, _ = solver.solve(
        K,
        np.asarray(F),
        dirichlet=(dir_dofs, dir_vals),
        dirichlet_mode="enforce",
    )

    # L2 (nodal) relative error
    err = np.linalg.norm(u_sol - u_exact) / np.linalg.norm(u_exact)
    assert err < 2e-2

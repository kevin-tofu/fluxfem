import os
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_beam_with_tip_spring_dashpot_newmark_moves_toward_static_solution():
    coords, conn = ff.structured_beam_chain(n_elems=4, length=1.0)
    section = ff.BeamSection(
        E=50.0e9,
        G=20.0e9,
        A=1.0e-3,
        Iy=2.0e-6,
        Iz=2.0e-6,
        J=1.0e-6,
        rho=1000.0,
    )

    n_dofs = ff.BEAM_DOF_PER_NODE * coords.shape[0]
    tip = coords.shape[0] - 1
    tip_uz = ff.beam_node_dofs([tip], "uz")

    K = np.asarray(ff.assemble_beam_stiffness(coords, conn, section).to_dense()) + np.asarray(
        ff.assemble_dof_spring(n_dofs, tip_uz, 2.0e6).to_dense()
    )
    M = np.asarray(ff.assemble_beam_mass(coords, conn, section).to_dense())
    C = np.asarray(ff.assemble_dof_dashpot(n_dofs, tip_uz, 1.0e4).to_dense())

    force = np.zeros(n_dofs, dtype=float)
    force[tip_uz] = -500.0
    fixed = ff.beam_node_dofs([0])
    bc = ff.DirichletBC(fixed, 0.0)

    u_static, _ = ff.LinearSolver(method="spsolve").solve(K, force, dirichlet=bc, dirichlet_mode="condense")
    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=np.zeros(n_dofs),
        v0=np.zeros(n_dofs),
        dt=2.0e-4,
        n_steps=300,
        force=force,
        dirichlet=bc,
    )

    tip_hist = out.u[:, tip_uz[0]]
    assert np.min(tip_hist) < 0.0
    assert abs(tip_hist[-1] - u_static[tip_uz[0]]) < abs(tip_hist[1] - u_static[tip_uz[0]])

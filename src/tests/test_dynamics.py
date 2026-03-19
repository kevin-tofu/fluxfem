"""Transient dynamics checks for Newmark-beta time integration."""

import numpy as np
import jax
import scipy.linalg as la

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_newmark_sdof_free_vibration_energy_stable():
    omega = 4.0
    M = np.array([[1.0]], dtype=float)
    C = np.array([[0.0]], dtype=float)
    K = np.array([[omega * omega]], dtype=float)

    period = 2.0 * np.pi / omega
    dt = period / 200.0
    n_steps = 400

    out = ff.newmark_solve_linear(M, C, K, u0=np.array([1.0]), v0=np.array([0.0]), dt=dt, n_steps=n_steps)

    u = out.u[:, 0]
    v = out.v[:, 0]
    e = 0.5 * (v * v + (omega * omega) * (u * u))

    e0 = e[0]
    rel_drift = np.max(np.abs(e - e0)) / e0
    assert rel_drift < 5.0e-3

    i_period = int(round(period / dt))
    assert abs(u[i_period] - 1.0) < 2.0e-2


def test_newmark_preserves_constant_dirichlet_dof():
    M = np.eye(2, dtype=float)
    C = np.zeros((2, 2), dtype=float)
    K = np.array([[5.0, 0.0], [0.0, 3.0]], dtype=float)

    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=np.array([0.0, 0.0]),
        v0=np.array([0.0, 0.0]),
        dt=0.05,
        n_steps=20,
        force=lambda _t: np.array([0.0, 1.0]),
        dirichlet=(np.array([0]), np.array([2.0])),
    )

    np.testing.assert_allclose(out.u[:, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(out.v[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out.a[:, 0], 0.0, atol=1e-12)
    assert np.max(np.abs(out.u[:, 1])) > 0.0


def test_newmark_3d_bar_tracks_first_mode_period():
    # 3D solid bar (hex mesh) with small cross-section: axial-dominant dynamics.
    L = 1.0
    E = 200.0
    nu = 0.3
    rho = 1.0

    mesh = ff.StructuredHexBox(nx=8, ny=1, nz=1, lx=L, ly=0.1, lz=0.1).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)

    D = ff.isotropic_3d_D(E, nu)
    K = np.asarray(space.assemble(ff.linear_elasticity_form, D).to_dense())
    M = rho * np.asarray(space.assemble_mass_matrix().to_dense())
    C = np.zeros_like(K)

    left = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0, atol=1e-10),
        components=[0, 1, 2],
        dof_per_node=3,
    )
    free = ff.free_dofs(space.n_dofs, left)

    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]
    w2, vecs = la.eigh(K_ff, M_ff)
    i0 = int(np.argmax(w2 > 1.0e-10))
    omega1 = float(np.sqrt(w2[i0]))
    phi1 = np.asarray(vecs[:, i0], dtype=float)
    phi1 /= np.max(np.abs(phi1))

    u0 = np.zeros(space.n_dofs, dtype=float)
    v0 = np.zeros(space.n_dofs, dtype=float)
    u0[free] = 1.0e-4 * phi1

    period = 2.0 * np.pi / omega1
    n_steps = 240
    dt = period / n_steps

    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=u0,
        v0=v0,
        dt=dt,
        n_steps=n_steps,
        dirichlet=(left, np.zeros_like(left, dtype=float)),
    )

    q = out.u[:, free] @ (M_ff @ phi1)
    q0 = float(q[0])
    q_half = float(q[n_steps // 2])
    q_end = float(q[-1])

    assert q0 != 0.0
    assert abs(q_end - q0) / abs(q0) < 5.0e-2
    assert abs(q_half + q0) / abs(q0) < 8.0e-2

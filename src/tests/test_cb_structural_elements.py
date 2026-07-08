import os

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def _assert_cb_backend_projection_matches(k, m, retained, n_modes):
    cb_scipy = ff.make_craig_bampton_basis(
        k,
        m,
        retained_dofs=retained,
        n_modes=n_modes,
        backend="scipy",
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    cb_jax = ff.make_craig_bampton_basis(
        k,
        m,
        retained_dofs=retained,
        n_modes=n_modes,
        backend="jax",
        constraint_solver="spsolve",
        modal_solver="dense",
    )

    assert cb_scipy.n_full == k.shape[0]
    assert cb_scipy.n_reduced == retained.size + n_modes
    assert cb_jax.n_full == k.shape[0]
    assert cb_jax.n_reduced == retained.size + n_modes
    np.testing.assert_allclose(cb_scipy.basis[retained, : retained.size], np.eye(retained.size), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(cb_jax.basis)[retained, : retained.size], np.eye(retained.size), atol=1.0e-12)
    np.testing.assert_allclose(cb_scipy.eigenvalues, np.asarray(cb_jax.eigenvalues), rtol=1.0e-9, atol=1.0e-9)

    aligned_basis = cb_scipy.basis.copy()
    jax_basis = np.asarray(cb_jax.basis)
    for i in range(cb_scipy.n_modes):
        col = cb_scipy.n_retained + i
        sign = np.sign(float(np.dot(aligned_basis[:, col], jax_basis[:, col])))
        aligned_basis[:, col] *= 1.0 if sign == 0.0 else sign

    k_red_scipy = aligned_basis.T @ (k @ aligned_basis)
    m_red_scipy = aligned_basis.T @ (m @ aligned_basis)
    k_red_jax = np.asarray(cb_jax.project_matrix(k))
    m_red_jax = np.asarray(cb_jax.project_matrix(m))

    np.testing.assert_allclose(k_red_scipy, k_red_scipy.T, atol=1.0e-6)
    np.testing.assert_allclose(m_red_scipy, m_red_scipy.T, atol=1.0e-10)
    np.testing.assert_allclose(k_red_scipy, k_red_jax, rtol=1.0e-9, atol=1.0e-6)
    np.testing.assert_allclose(m_red_scipy, m_red_jax, rtol=1.0e-9, atol=1.0e-10)

    q = np.linspace(-0.2, 0.3, cb_scipy.n_reduced)
    np.testing.assert_allclose(aligned_basis @ q, np.asarray(cb_jax.expand(q)), rtol=1.0e-10, atol=1.0e-10)


def test_craig_bampton_accepts_assembled_beam_6dof_matrices():
    section = ff.BeamSection(
        E=210.0e9,
        G=80.0e9,
        A=2.0e-3,
        Iy=8.0e-6,
        Iz=5.0e-6,
        J=1.0e-5,
        rho=7800.0,
    )
    coords, conn = ff.structured_beam_chain(n_elems=3, length=2.0)
    k = ff.assemble_beam_stiffness(coords, conn, section, format="csr")
    m = ff.assemble_beam_mass(coords, conn, section, format="csr")
    retained = np.array(
        sorted(
            set(ff.beam_node_dofs([0], "ux uy uz rx ry rz")).union(
                ff.beam_node_dofs([coords.shape[0] - 1], "ux uy uz rx ry rz")
            )
        ),
        dtype=np.int32,
    )

    _assert_cb_backend_projection_matches(k, m, retained, n_modes=4)


def test_craig_bampton_accepts_assembled_truss_axial_matrices():
    section = ff.TrussSection(E=70.0e9, A=1.5e-3, rho=2700.0)
    coords, conn = ff.structured_truss_chain(n_elems=5, length=3.0)
    active = ff.truss_node_dofs(np.arange(coords.shape[0]), "x")
    k_full = ff.assemble_truss_stiffness(coords, conn, section, format="csr")
    m_full = ff.assemble_truss_mass(coords, conn, section, format="csr")
    k = k_full[active, :][:, active]
    m = m_full[active, :][:, active]
    retained = np.array([0, active.size - 1], dtype=np.int32)

    _assert_cb_backend_projection_matches(k, m, retained, n_modes=2)


def test_craig_bampton_accepts_assembled_mindlin_plate_matrices():
    coords, conn = ff.structured_plate_grid(nx=2, ny=2, length_x=2.0, length_y=1.0)
    section = ff.PlateSection(E=70.0e9, nu=0.33, thickness=0.05, rho=2700.0, shear_mode="mitc4")
    k = ff.assemble_mindlin_plate_stiffness(coords, conn, section, format="csr")
    m = ff.assemble_mindlin_plate_mass(coords, conn, section, format="csr")
    boundary_nodes = np.flatnonzero(
        np.isclose(coords[:, 0], coords[:, 0].min())
        | np.isclose(coords[:, 0], coords[:, 0].max())
        | np.isclose(coords[:, 1], coords[:, 1].min())
        | np.isclose(coords[:, 1], coords[:, 1].max())
    )
    retained = ff.plate_node_dofs(boundary_nodes).astype(np.int32)

    _assert_cb_backend_projection_matches(k, m, retained, n_modes=1)


def test_craig_bampton_accepts_assembled_flat_shell_matrices():
    coords, conn = ff.structured_plate_grid(nx=2, ny=2, length_x=2.0, length_y=1.0)
    section = ff.ShellSection(E=70.0e9, nu=0.33, thickness=0.05, rho=2700.0, shear_mode="mitc4")
    k = ff.assemble_shell_stiffness(coords, conn, section, format="csr")
    m = ff.assemble_shell_mass(coords, conn, section, format="csr")
    boundary_nodes = np.flatnonzero(
        np.isclose(coords[:, 0], coords[:, 0].min())
        | np.isclose(coords[:, 0], coords[:, 0].max())
        | np.isclose(coords[:, 1], coords[:, 1].min())
        | np.isclose(coords[:, 1], coords[:, 1].max())
    )
    retained = ff.shell_node_dofs(boundary_nodes).astype(np.int32)

    _assert_cb_backend_projection_matches(k, m, retained, n_modes=2)

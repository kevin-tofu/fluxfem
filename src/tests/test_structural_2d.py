import os

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_frame2d_cantilever_tip_load_matches_euler_bernoulli_solution():
    length = 2.0
    force = -1000.0
    section = ff.BeamSection(E=210.0e9, G=80.0e9, A=2.0e-3, Iy=8.0e-6, Iz=5.0e-6, J=1.0e-5)
    coords, conn = ff.structured_frame2d_chain(n_elems=4, length=length)
    K = ff.assemble_frame2d_stiffness(coords, conn, section)
    tip = coords.shape[0] - 1
    F = ff.assemble_frame2d_point_load(coords.shape[0], tip, force=(0.0, force))

    fixed = ff.frame2d_node_dofs([0])
    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(fixed, 0.0),
        dirichlet_mode="condense",
    )

    uz_tip = float(u[ff.frame2d_node_dofs([tip], "uz")][0])
    ry_tip = float(u[ff.frame2d_node_dofs([tip], "ry")][0])
    uz_exact = force * length**3 / (3.0 * section.E * section.Iy)
    ry_exact = -force * length**2 / (2.0 * section.E * section.Iy)

    np.testing.assert_allclose(uz_tip, uz_exact, rtol=2.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(ry_tip, ry_exact, rtol=2.0e-12, atol=1.0e-14)


def test_frame2d_format_choice_and_jax_load_backend():
    section = ff.BeamSection(
        E=210.0e9,
        G=80.0e9,
        A=2.0e-3,
        Iy=8.0e-6,
        Iz=5.0e-6,
        J=1.0e-5,
        rho=7800.0,
    )
    coords, conn = ff.structured_frame2d_chain(n_elems=2, length=1.0)
    K_csr = ff.assemble_frame2d_stiffness(coords, conn, section, backend="scipy")
    K_flux = ff.assemble_frame2d_stiffness(coords, conn, section, backend="jax")
    K_dense = ff.assemble_frame2d_stiffness(coords, conn, section, backend="numpy")
    M_dense = ff.assemble_frame2d_mass(coords, conn, section, format="dense")
    F_jax = ff.assemble_frame2d_uniform_load(coords, conn, (0.0, -5.0), array_backend="jax")

    assert sp.isspmatrix_csr(K_csr)
    assert isinstance(K_dense, np.ndarray)
    assert isinstance(M_dense, np.ndarray)
    assert isinstance(F_jax, jax.Array)
    np.testing.assert_allclose(K_dense, K_csr.toarray())
    np.testing.assert_allclose(np.asarray(K_flux.to_dense()), K_dense)
    np.testing.assert_allclose(np.asarray(F_jax[ff.frame2d_node_dofs(np.arange(coords.shape[0]), "uz")]).sum(), -5.0)


def test_frame2d_point_loads_accumulate_multiple_nodes():
    F = ff.assemble_frame2d_point_loads(
        4,
        [1, 3, 1],
        forces=[(1.0, 2.0), (0.0, -1.0), (4.0, 5.0)],
        moments=[0.1, 1.0, 0.4],
    )
    np.testing.assert_allclose(F[ff.frame2d_node_dofs([1])], np.array([5.0, 7.0, 0.5]))
    np.testing.assert_allclose(F[ff.frame2d_node_dofs([3])], np.array([0.0, -1.0, 1.0]))


def test_truss2d_cantilever_axial_load_matches_bar_solution():
    length = 3.0
    force = 1200.0
    section = ff.TrussSection(E=70.0e9, A=1.5e-3)
    coords, conn = ff.structured_truss2d_chain(n_elems=5, length=length)
    K = ff.assemble_truss2d_stiffness(coords, conn, section)
    tip = coords.shape[0] - 1
    F = ff.assemble_truss2d_point_load(coords.shape[0], tip, force=(force, 0.0))

    fixed = ff.truss2d_node_dofs([0], "xz")
    lateral = ff.truss2d_node_dofs(np.arange(1, coords.shape[0]), "z")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))
    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dirichlet, 0.0),
        dirichlet_mode="condense",
    )

    ux_tip = float(u[ff.truss2d_node_dofs([tip], "x")][0])
    ux_exact = force * length / (section.E * section.A)
    np.testing.assert_allclose(ux_tip, ux_exact, rtol=2.0e-12, atol=1.0e-14)


def test_truss2d_uniform_axial_load_matches_bar_solution():
    length = 3.0
    qx = 400.0
    section = ff.TrussSection(E=70.0e9, A=1.5e-3)
    coords, conn = ff.structured_truss2d_chain(n_elems=5, length=length)
    K = ff.assemble_truss2d_stiffness(coords, conn, section)
    F = ff.assemble_truss2d_uniform_load(coords, conn, [qx], frame="local")

    fixed = ff.truss2d_node_dofs([0], "xz")
    lateral = ff.truss2d_node_dofs(np.arange(1, coords.shape[0]), "z")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))
    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dirichlet, 0.0),
        dirichlet_mode="condense",
    )

    tip = coords.shape[0] - 1
    ux_tip = float(u[ff.truss2d_node_dofs([tip], "x")][0])
    ux_exact = qx * length**2 / (2.0 * section.E * section.A)
    np.testing.assert_allclose(ux_tip, ux_exact, rtol=2.0e-12, atol=1.0e-14)


def test_truss2d_format_choice_and_jax_load_backend():
    section = ff.TrussSection(E=70.0e9, A=1.5e-3, rho=2700.0)
    coords, conn = ff.structured_truss2d_chain(n_elems=3, length=2.0, direction=(1.0, 1.0))
    K_csr = ff.assemble_truss2d_stiffness(coords, conn, section, backend="scipy")
    K_flux = ff.assemble_truss2d_stiffness(coords, conn, section, backend="jax")
    K_dense = ff.assemble_truss2d_stiffness(coords, conn, section, backend="numpy")
    M_dense = ff.assemble_truss2d_mass(coords, conn, section, format="dense")
    F_jax = ff.assemble_truss2d_uniform_load(coords, conn, (3.0, -2.0), frame="global", array_backend="jax")

    assert sp.isspmatrix_csr(K_csr)
    assert isinstance(K_dense, np.ndarray)
    assert isinstance(M_dense, np.ndarray)
    assert isinstance(F_jax, jax.Array)
    np.testing.assert_allclose(K_dense, K_csr.toarray())
    np.testing.assert_allclose(np.asarray(K_flux.to_dense()), K_dense)
    np.testing.assert_allclose(np.asarray(F_jax[ff.truss2d_node_dofs(np.arange(coords.shape[0]), "x")]).sum(), 6.0)
    np.testing.assert_allclose(np.asarray(F_jax[ff.truss2d_node_dofs(np.arange(coords.shape[0]), "z")]).sum(), -4.0)

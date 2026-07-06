import os
import numpy as np
import scipy.sparse as sp

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_truss_cantilever_axial_load_matches_bar_solution():
    length = 3.0
    force = 1200.0
    section = ff.TrussSection(E=70.0e9, A=1.5e-3)
    coords, conn = ff.structured_truss_chain(n_elems=5, length=length)
    K = ff.assemble_truss_stiffness(coords, conn, section)

    tip = coords.shape[0] - 1
    F = ff.assemble_truss_point_load(coords.shape[0], tip, force=(force, 0.0, 0.0))

    fixed = ff.truss_node_dofs([0], "xyz")
    lateral = ff.truss_node_dofs(np.arange(1, coords.shape[0]), "yz")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))
    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dirichlet, 0.0),
        dirichlet_mode="condense",
    )

    ux_tip = float(u[ff.truss_node_dofs([tip], "x")][0])
    ux_exact = force * length / (section.E * section.A)
    np.testing.assert_allclose(ux_tip, ux_exact, rtol=2.0e-12, atol=1.0e-14)


def test_truss_stiffness_backend_choice_matches_between_scipy_and_jax():
    length = 3.0
    force = 1200.0
    section = ff.TrussSection(E=70.0e9, A=1.5e-3)
    coords, conn = ff.structured_truss_chain(n_elems=5, length=length)
    K_scipy = ff.assemble_truss_stiffness(coords, conn, section, backend="scipy")
    K_jax = ff.assemble_truss_stiffness(coords, conn, section, backend="jax")
    K_numpy = ff.assemble_truss_stiffness(coords, conn, section, backend="numpy")
    tip = coords.shape[0] - 1
    F_numpy = ff.assemble_truss_point_load(coords.shape[0], tip, force=(force, 0.0, 0.0))
    F_jax = ff.assemble_truss_point_load(coords.shape[0], tip, force=(force, 0.0, 0.0), backend="jax")

    assert sp.isspmatrix_csr(K_scipy)
    assert isinstance(K_numpy, np.ndarray)
    assert isinstance(F_jax, jax.Array)
    np.testing.assert_allclose(K_numpy, K_scipy.toarray())

    fixed = ff.truss_node_dofs([0], "xyz")
    lateral = ff.truss_node_dofs(np.arange(1, coords.shape[0]), "yz")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))
    bc = ff.DirichletBC(dirichlet, 0.0)
    u_scipy, _info_scipy = ff.LinearSolver(method="spsolve").solve(
        K_scipy,
        F_numpy,
        dirichlet=bc,
        dirichlet_mode="condense",
    )
    u_jax, _info_jax = ff.LinearSolver(method="spsolve_jax").solve(
        K_jax,
        F_jax,
        dirichlet=bc,
        dirichlet_mode="condense",
    )

    np.testing.assert_allclose(np.asarray(u_jax), u_scipy, rtol=1.0e-11, atol=1.0e-14)


def test_truss_cantilever_uniform_axial_load_matches_bar_solution():
    length = 3.0
    qx = 400.0
    section = ff.TrussSection(E=70.0e9, A=1.5e-3)
    coords, conn = ff.structured_truss_chain(n_elems=5, length=length)
    K = ff.assemble_truss_stiffness(coords, conn, section)
    F = ff.assemble_truss_uniform_load(coords, conn, [qx], frame="local")

    fixed = ff.truss_node_dofs([0], "xyz")
    lateral = ff.truss_node_dofs(np.arange(1, coords.shape[0]), "yz")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))
    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dirichlet, 0.0),
        dirichlet_mode="condense",
    )

    tip = coords.shape[0] - 1
    ux_tip = float(u[ff.truss_node_dofs([tip], "x")][0])
    ux_exact = qx * length**2 / (2.0 * section.E * section.A)
    np.testing.assert_allclose(ux_tip, ux_exact, rtol=2.0e-12, atol=1.0e-14)


def test_truss_point_loads_accumulate_multiple_nodes():
    F = ff.assemble_truss_point_loads(
        4,
        [1, 3, 1],
        forces=[(1.0, 2.0, 3.0), (0.0, -1.0, 0.0), (4.0, 5.0, 6.0)],
    )
    np.testing.assert_allclose(F[ff.truss_node_dofs([1])], np.array([5.0, 7.0, 9.0]))
    np.testing.assert_allclose(F[ff.truss_node_dofs([3])], np.array([0.0, -1.0, 0.0]))


def test_truss_uniform_global_load_preserves_total_force():
    coords, conn = ff.structured_truss_chain(n_elems=3, length=2.0, direction=(1.0, 1.0, 0.0))
    load = np.array([3.0, -2.0, 5.0])
    F = ff.assemble_truss_uniform_load(coords, conn, load, frame="global")

    for comp in range(3):
        dofs = ff.truss_node_dofs(np.arange(coords.shape[0]), [comp])
        np.testing.assert_allclose(np.sum(F[dofs]), load[comp] * 2.0)


def test_truss_load_helpers_support_jax_backend():
    coords, conn = ff.structured_truss_chain(n_elems=3, length=2.0, direction=(1.0, 1.0, 0.0))
    load = np.array([3.0, -2.0, 5.0])
    F = ff.assemble_truss_uniform_load(coords, conn, load, frame="global", backend="jax")

    assert isinstance(F, jax.Array)
    for comp in range(3):
        dofs = ff.truss_node_dofs(np.arange(coords.shape[0]), [comp])
        np.testing.assert_allclose(np.asarray(F[dofs]).sum(), load[comp] * 2.0)


def test_truss_element_stiffness_rotates_to_global_direction():
    section = ff.TrussSection(E=10.0, A=2.0)
    k = ff.truss_element_stiffness_global([0.0, 0.0, 0.0], [0.0, 2.0, 0.0], section)

    assert np.count_nonzero(np.abs(k) > 1.0e-12) == 4
    np.testing.assert_allclose(k[np.ix_([1, 4], [1, 4])], [[10.0, -10.0], [-10.0, 10.0]])
    np.testing.assert_allclose(k, k.T)


def test_truss_mass_matrix_preserves_total_mass_per_direction():
    section = ff.TrussSection(E=1.0, A=0.25, rho=8.0)
    coords, conn = ff.structured_truss_chain(n_elems=2, length=3.0)
    M = np.asarray(ff.assemble_truss_mass(coords, conn, section).to_dense())

    total_mass = section.rho * section.A * 3.0
    for comp in range(3):
        dofs = ff.truss_node_dofs(np.arange(coords.shape[0]), [comp])
        ones = np.ones(dofs.size)
        assert abs(float(ones @ M[np.ix_(dofs, dofs)] @ ones) - total_mass) < 1.0e-12


def test_truss_node_dofs():
    dofs = ff.truss_node_dofs([1, 3], "xz")
    np.testing.assert_array_equal(dofs, np.array([3, 5, 9, 11]))

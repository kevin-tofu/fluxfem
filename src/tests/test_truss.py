import os
import numpy as np

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

    F = np.zeros(ff.TRUSS_DOF_PER_NODE * coords.shape[0], dtype=float)
    tip = coords.shape[0] - 1
    F[ff.truss_node_dofs([tip], "x")] = force

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

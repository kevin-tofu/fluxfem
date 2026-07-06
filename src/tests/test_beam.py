import os
import numpy as np
import scipy.linalg as la

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_beam_element_local_stiffness_is_symmetric():
    section = ff.BeamSection(E=210.0, G=80.0, A=0.02, Iy=3.0e-5, Iz=4.0e-5, J=5.0e-5)
    k = ff.beam_element_stiffness_local(2.5, section)

    np.testing.assert_allclose(k, k.T, atol=1.0e-12)
    assert np.min(np.linalg.eigvalsh(k)) > -1.0e-10


def test_beam_element_mass_is_symmetric_and_preserves_translational_mass():
    length = 2.5
    section = ff.BeamSection(E=210.0, G=80.0, A=0.02, Iy=3.0e-5, Iz=4.0e-5, J=5.0e-5, rho=7.8)
    m = ff.beam_element_mass_local(length, section)

    np.testing.assert_allclose(m, m.T, atol=1.0e-12)
    total_mass = section.rho * section.A * length
    for dofs in ([0, 6], [1, 7], [2, 8]):
        ones = np.ones(2)
        assert abs(float(ones @ m[np.ix_(dofs, dofs)] @ ones) - total_mass) < 1.0e-12


def test_beam_cantilever_tip_load_matches_euler_bernoulli_solution():
    length = 2.0
    force = -1000.0
    section = ff.BeamSection(
        E=210.0e9,
        G=80.0e9,
        A=2.0e-3,
        Iy=8.0e-6,
        Iz=5.0e-6,
        J=1.0e-5,
    )
    coords, conn = ff.structured_beam_chain(n_elems=4, length=length)
    K = ff.assemble_beam_stiffness(coords, conn, section).to_dense()
    F = np.zeros(ff.BEAM_DOF_PER_NODE * coords.shape[0], dtype=float)
    tip = coords.shape[0] - 1
    F[ff.beam_node_dofs([tip], "uz")] = force

    fixed = ff.beam_node_dofs([0])
    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(fixed, 0.0),
        dirichlet_mode="condense",
    )

    uz_tip = float(u[ff.beam_node_dofs([tip], "uz")][0])
    ry_tip = float(u[ff.beam_node_dofs([tip], "ry")][0])
    uz_exact = force * length**3 / (3.0 * section.E * section.Iy)
    ry_exact = -force * length**2 / (2.0 * section.E * section.Iy)

    np.testing.assert_allclose(uz_tip, uz_exact, rtol=2.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(ry_tip, ry_exact, rtol=2.0e-12, atol=1.0e-14)


def test_beam_cantilever_first_bending_frequency_matches_reference():
    length = 2.0
    section = ff.BeamSection(
        E=210.0e9,
        G=80.0e9,
        A=2.0e-3,
        Iy=8.0e-6,
        Iz=8.0e-6,
        J=1.0e-5,
        rho=7800.0,
    )
    coords, conn = ff.structured_beam_chain(n_elems=12, length=length)
    K = np.asarray(ff.assemble_beam_stiffness(coords, conn, section).to_dense())
    M = np.asarray(ff.assemble_beam_mass(coords, conn, section).to_dense())

    fixed = ff.beam_node_dofs([0])
    free = ff.free_dofs(K.shape[0], fixed)
    w2 = la.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True)
    omega1 = float(np.sqrt(w2[w2 > 1.0e-8][0]))

    beta1 = 1.875104068711961
    omega_exact = beta1**2 * np.sqrt(section.E * section.Iy / (section.rho * section.A * length**4))
    assert abs(omega1 - omega_exact) / omega_exact < 2.0e-3


def test_beam_mass_requires_density():
    section = ff.BeamSection(E=210.0, G=80.0, A=0.02, Iy=3.0e-5, Iz=4.0e-5, J=5.0e-5)
    coords, conn = ff.structured_beam_chain(n_elems=1, length=1.0)

    try:
        ff.assemble_beam_mass(coords, conn, section)
    except ValueError as exc:
        assert "rho" in str(exc)
    else:
        raise AssertionError("assemble_beam_mass should require section.rho")


def test_beam_node_dofs_accept_named_components():
    dofs = ff.beam_node_dofs([2, 4], "ux rz")
    np.testing.assert_array_equal(dofs, np.array([12, 17, 24, 29]))

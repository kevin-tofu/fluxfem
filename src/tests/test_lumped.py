import os
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_dof_spring_to_ground_accepts_scalar_vector_and_matrix():
    K_scalar = np.asarray(ff.assemble_dof_spring(4, [1, 3], 5.0).to_dense())
    np.testing.assert_allclose(K_scalar[np.ix_([1, 3], [1, 3])], 5.0 * np.eye(2))

    K_vector = np.asarray(ff.assemble_dof_spring(4, [1, 3], [2.0, 7.0]).to_dense())
    np.testing.assert_allclose(K_vector[np.ix_([1, 3], [1, 3])], np.diag([2.0, 7.0]))

    block = np.array([[3.0, 1.0], [1.0, 4.0]])
    K_matrix = np.asarray(ff.assemble_dof_spring(4, [1, 3], block).to_dense())
    np.testing.assert_allclose(K_matrix[np.ix_([1, 3], [1, 3])], block)


def test_assemble_nodal_load_accumulates_duplicate_dofs():
    f = ff.assemble_nodal_load(4, [1, 3, 1], [2.0, -1.0, 5.0])
    np.testing.assert_allclose(f, np.array([0.0, 7.0, 0.0, -1.0]))


def test_dof_spring_between_dofs_uses_relative_displacement():
    K = np.asarray(ff.assemble_dof_spring(3, [0], 10.0, other_dofs=[2]).to_dense())
    expected = np.array(
        [
            [10.0, 0.0, -10.0],
            [0.0, 0.0, 0.0],
            [-10.0, 0.0, 10.0],
        ]
    )
    np.testing.assert_allclose(K, expected)
    np.testing.assert_allclose(K @ np.array([1.5, 3.0, 1.5]), np.zeros(3))


def test_dof_dashpot_builds_damping_matrix_for_newmark_decay():
    M = np.array([[1.0]], dtype=float)
    K = np.asarray(ff.assemble_dof_spring(1, [0], 4.0).to_dense())
    C = np.asarray(ff.assemble_dof_dashpot(1, [0], 0.4).to_dense())

    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=np.array([1.0]),
        v0=np.array([0.0]),
        dt=0.01,
        n_steps=300,
    )

    energy = 0.5 * (out.v[:, 0] ** 2 + 4.0 * out.u[:, 0] ** 2)
    assert energy[-1] < energy[0]


def test_rayleigh_damping_matrix_matches_linear_combination():
    M = np.diag([2.0, 3.0])
    K = np.array([[10.0, -2.0], [-2.0, 5.0]])
    C = ff.assemble_rayleigh_damping(M, K, alpha=0.1, beta=0.02)

    np.testing.assert_allclose(C, 0.1 * M + 0.02 * K)


def test_rayleigh_coefficients_match_target_modal_damping():
    alpha, beta = ff.rayleigh_coefficients_from_modal_damping(
        omega1=10.0,
        zeta1=0.02,
        omega2=100.0,
        zeta2=0.05,
    )

    zetas = ff.rayleigh_damping_ratio(np.array([10.0, 100.0]), alpha=alpha, beta=beta)
    np.testing.assert_allclose(zetas, np.array([0.02, 0.05]), rtol=1.0e-14, atol=1.0e-14)


def test_rayleigh_damping_accepts_flux_sparse_matrices():
    M = ff.assemble_dof_spring(2, [0, 1], [2.0, 3.0])
    K = ff.assemble_dof_spring(2, [0, 1], [10.0, 20.0])
    C = ff.assemble_rayleigh_damping(M, K, alpha=0.5, beta=0.1)

    np.testing.assert_allclose(C, np.diag([2.0, 3.5]))


def test_dof_connector_rejects_mismatched_sizes():
    try:
        ff.assemble_dof_spring(3, [0, 1], 1.0, other_dofs=[2])
    except ValueError as exc:
        assert "same size" in str(exc)
    else:
        raise AssertionError("mismatched connector DOFs should fail")

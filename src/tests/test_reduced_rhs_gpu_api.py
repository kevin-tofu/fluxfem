from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_project_reduced_rhs_cpu_matches_manual_projection():
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.5],
        ]
    )
    free = np.array([0, 2, 4])
    forces = np.array(
        [
            [1.0, 9.0, 2.0, 8.0, 3.0],
            [4.0, 7.0, 5.0, 6.0, 6.0],
        ]
    )

    rhs = ff.project_reduced_rhs_cpu(forces, basis, free_dofs=free, hub_dofs=1, n_constraints=2)
    expected_projected = forces[:, free] @ basis
    expected = np.concatenate([expected_projected, np.zeros((2, 3))], axis=1)

    np.testing.assert_allclose(rhs, expected)


def test_jitted_project_reduced_rhs_matches_cpu_projection():
    basis = np.array(
        [
            [1.0, 0.0],
            [0.25, 0.75],
            [0.0, 1.0],
        ]
    )
    free = np.array([0, 1, 3])
    forces = np.array(
        [
            [2.0, 3.0, 99.0, 4.0],
            [5.0, 6.0, 88.0, 7.0],
        ]
    )

    cpu = ff.project_reduced_rhs_cpu(forces, basis, free_dofs=free, hub_dofs=2)
    projector = ff.make_reduced_rhs_projector_jax(basis, free_dofs=free, hub_dofs=2, jit=True)
    jax_rhs = np.asarray(projector(forces))

    np.testing.assert_allclose(jax_rhs, cpu)


def test_factorized_reduced_dense_batch_jax_matches_direct_solve():
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.25],
        ]
    )
    kkt = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rhs = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )

    direct = ff.solve_reduced_dense_batch_jax(kkt, rhs, basis, n_reduced=2)
    factorization = ff.factor_reduced_dense_kkt_jax(kkt)
    factorized = ff.solve_reduced_dense_batch_jax_factorized(factorization, rhs, basis, n_reduced=2)
    solver = ff.make_reduced_dense_factorized_solver_jax(kkt, basis, n_reduced=2, jit=True)
    compiled = np.asarray(solver(rhs))

    np.testing.assert_allclose(factorized, direct, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(compiled, direct, rtol=1.0e-10, atol=1.0e-10)

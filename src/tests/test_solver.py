"""Linear solver, Dirichlet, and multi-RHS behavior checks."""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)

import fluxfem as ff
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from fluxfem.solver.bc import add_neumann_load, add_robin, facet_area


def test_enforce_dirichlet_dense_simple():
    K = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    F = np.array([1.0, 0.0], dtype=float)
    dofs = [0]
    vals = [1.0]

    Kc, Fc = ff.enforce_dirichlet_dense(K, F, dofs, vals)
    np.testing.assert_allclose(Kc[0, :], [1.0, 0.0])
    np.testing.assert_allclose(Kc[:, 0], [1.0, 0.0])
    assert Fc[0] == 1.0


def test_condense_and_expand_dirichlet():
    K = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    F = np.array([1.0, 0.0], dtype=float)
    dofs = [0]
    vals = [1.0]

    Kc, Fc, free, dir_dofs, dir_vals = ff.condense_dirichlet_dense(K, F, dofs, vals)
    # Reduced system is scalar: (2)*u1 = 1  => u1 = 0.5
    u_free = np.linalg.solve(Kc, Fc)
    u_full = ff.expand_dirichlet_solution(u_free, free, dir_dofs, dir_vals, n_total=2)
    np.testing.assert_allclose(u_full, [1.0, 0.5])


def test_enforce_dirichlet_sparse():
    # same system but via FluxSparseMatrix
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    data = np.array([2.0, -1.0, -1.0, 2.0])
    pattern = ff.SparsityPattern(rows=rows, cols=cols, n_dofs=2)
    A = ff.FluxSparseMatrix(pattern, data)
    F = np.array([1.0, 0.0], dtype=float)

    Kc, Fc = ff.enforce_dirichlet_sparse(A, F, [0], [1.0])
    Kc = Kc.toarray()
    np.testing.assert_allclose(Kc[0, :], [1.0, 0.0])
    np.testing.assert_allclose(Kc[:, 0], [1.0, 0.0])
    assert Fc[0] == 1.0


def test_cg_solve_matches_dense():
    # solve 2x2 SPD system with CG via FluxSparseMatrix
    K = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    F = np.array([1.0, 0.0], dtype=np.float32)
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    pattern = ff.SparsityPattern(rows=rows, cols=cols, n_dofs=2)
    A = ff.FluxSparseMatrix(pattern, data)

    u_dense = np.linalg.solve(K, F)
    u_cg, info = ff.cg_solve(A, F, tol=1e-10, maxiter=50)
    np.testing.assert_allclose(np.asarray(u_cg), u_dense, rtol=1e-6, atol=1e-6)
    assert float(info["residual_norm"]) < 1e-6


def test_neumann_load_added():
    # Quad face with unit area, traction (1,2,3) in dim=3 → total force shared by 4 nodes
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facet = np.array([0, 1, 2, 3], dtype=int)
    area = facet_area(coords, facet)
    assert np.isclose(area, 1.0)
    F = np.zeros(4 * 3, dtype=float)
    F_new = add_neumann_load(F, facet, traction=[1.0, 2.0, 3.0], dim=3, coords=coords)
    # each node gets area/4 * traction
    expected = np.tile(np.array([0.25, 0.5, 0.75]), 4)
    np.testing.assert_allclose(F_new, expected)


def test_robin_adds_diag_and_rhs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facet = np.array([0, 1, 2, 3], dtype=int)
    K = np.zeros((4, 4), dtype=float)
    F = np.zeros(4, dtype=float)
    F_new, K_new = add_robin(
        F, K, facet, alpha=2.0, g=3.0, dim=1, coords=coords
    )
    # weight = alpha * area / m = 2 * 1 / 4 = 0.5
    assert np.allclose(np.diag(K_new), 0.5)
    assert np.allclose(F_new, 0.5 * 3.0)


def test_linear_solver_multi_rhs_enforce_dirichlet():
    K = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=float)
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_bc, F_bc = ff.enforce_dirichlet_dense(K, F, dofs, vals)
    u_expected = spsolve(sp.csr_matrix(K_bc), F_bc)

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(K, F, dirichlet=(dofs, vals), dirichlet_mode="enforce")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def test_linear_solver_multi_rhs_condense_dirichlet():
    K = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=float)
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_ff, F_f, free, dir_dofs, dir_vals = ff.condense_dirichlet_dense(K, F, dofs, vals)
    u_free = spsolve(sp.csr_matrix(K_ff), F_f)
    u_expected = ff.expand_dirichlet_solution(u_free, free, dir_dofs, dir_vals, n_total=K.shape[0])

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(K, F, dirichlet=(dofs, vals), dirichlet_mode="condense")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def _flux_matrix_2x2():
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    data = np.array([4.0, -1.0, -1.0, 3.0])
    pattern = ff.SparsityPattern(rows=rows, cols=cols, n_dofs=2)
    return ff.FluxSparseMatrix(pattern, data)


def test_linear_solver_multi_rhs_enforce_dirichlet_fluxsparse():
    A = _flux_matrix_2x2()
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_bc, F_bc = ff.enforce_dirichlet_sparse(A, F, dofs, vals)
    u_expected = spsolve(K_bc, F_bc)

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(A, F, dirichlet=(dofs, vals), dirichlet_mode="enforce")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def test_linear_solver_multi_rhs_condense_dirichlet_fluxsparse():
    A = _flux_matrix_2x2()
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_ff, F_f, free, dir_dofs, dir_vals = ff.condense_dirichlet_fluxsparse(A, F, dofs, vals)
    u_free = spsolve(K_ff, F_f)
    u_expected = ff.expand_dirichlet_solution(u_free, free, dir_dofs, dir_vals, n_total=2)

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(A, F, dirichlet=(dofs, vals), dirichlet_mode="condense")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def test_cg_multi_rhs_matches_direct():
    A = _flux_matrix_2x2()
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    u_direct = spsolve(A.to_csr(), F)

    u_cg, info = ff.cg_solve(A, F, tol=1e-12, maxiter=200)
    np.testing.assert_allclose(np.asarray(u_cg), u_direct, rtol=1e-6, atol=1e-6)
    assert len(info.get("iters", [])) == F.shape[1]

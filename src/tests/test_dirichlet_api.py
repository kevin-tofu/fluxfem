import numpy as np
import jax.numpy as jnp

import fluxfem as ff


def test_dirichlet_api_methods_exist():
    bc = ff.DirichletBC([0], [1.0])
    expected = [
        "condense_system",
        "enforce_system",
        "split_matrix",
        "condense_flux",
        "enforce_flux",
        "expand_solution",
    ]
    for name in expected:
        assert hasattr(bc, name)


def test_dirichlet_enforce_system_dense_and_jax():
    K = np.array([[4.0, 1.0], [1.0, 3.0]])
    F = np.array([1.0, 2.0])
    bc = ff.DirichletBC([0], [0.5])
    K_bc, F_bc = bc.enforce_system(K, F)
    K_ref, F_ref = ff.enforce_dirichlet_dense(K, F, [0], [0.5])
    np.testing.assert_allclose(K_bc, K_ref)
    np.testing.assert_allclose(F_bc, F_ref)

    K_j = jnp.asarray(K)
    F_j = jnp.asarray(F)
    K_bc_j, F_bc_j = bc.enforce_system(K_j, F_j)
    np.testing.assert_allclose(np.asarray(K_bc_j), K_ref)
    np.testing.assert_allclose(np.asarray(F_bc_j), F_ref)

    try:
        import scipy.sparse as sp  # noqa: F401
    except Exception:
        return
    rows, cols = np.nonzero(K)
    data = K[rows, cols]
    K_sparse = ff.FluxSparseMatrix(rows, cols, data, K.shape[0])
    K_bc_s, F_bc_s = bc.enforce_system(K_sparse, F)
    K_ref_s, F_ref_s = ff.enforce_dirichlet_sparse(K_sparse, F, [0], [0.5])
    np.testing.assert_allclose(np.asarray(K_bc_s.todense()), np.asarray(K_ref_s.todense()))
    np.testing.assert_allclose(F_bc_s, F_ref_s)


def test_dirichlet_split_matrix_dense_and_sparse():
    K = np.array(
        [
            [2.0, 3.0, 0.0],
            [3.0, 5.0, 1.0],
            [0.0, 1.0, 4.0],
        ]
    )
    bc = ff.DirichletBC([1], [0.0])
    free, dir_dofs, K_ff, K_fd = bc.split_matrix(K)
    np.testing.assert_array_equal(dir_dofs, np.array([1]))
    np.testing.assert_array_equal(free, np.array([0, 2]))
    np.testing.assert_allclose(K_ff, K[np.ix_([0, 2], [0, 2])])
    np.testing.assert_allclose(K_fd, K[np.ix_([0, 2], [1])])

    try:
        import scipy.sparse as sp  # noqa: F401
    except Exception:
        return
    rows, cols = np.nonzero(K)
    data = K[rows, cols]
    K_sparse = ff.FluxSparseMatrix(rows, cols, data, K.shape[0])
    free2, dir2, K_ff2, K_fd2 = bc.split_matrix(K_sparse, n_total=K.shape[0])
    np.testing.assert_array_equal(free2, free)
    np.testing.assert_array_equal(dir2, dir_dofs)
    np.testing.assert_allclose(np.asarray(K_ff2.todense()), K_ff)
    np.testing.assert_allclose(np.asarray(K_fd2.todense()), K_fd)

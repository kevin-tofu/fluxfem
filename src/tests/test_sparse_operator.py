import numpy as np

import fluxfem as ff


def test_flux_sparse_operator_to_dense_and_coalesce():
    rows = np.array([0, 0, 1], dtype=int)
    cols = np.array([1, 1, 0], dtype=int)
    data = np.array([2.0, 3.0, 4.0], dtype=float)
    op = ff.FluxSparseOperator(rows, cols, data, shape=(2, 3))

    dense = np.asarray(op.to_dense())
    expected = np.array([[0.0, 5.0, 0.0], [4.0, 0.0, 0.0]])
    assert dense.shape == (2, 3)
    assert np.allclose(dense, expected)

    op_u = op.coalesce()
    rows_u, cols_u, data_u, shape_u = op_u.to_coo()
    assert shape_u == (2, 3)
    assert np.asarray(data_u).shape == (2,)
    assert np.allclose(np.asarray(op_u.to_dense()), expected)


def test_flux_sparse_operator_matvec_and_rmatvec():
    rows = np.array([0, 0, 1, 2], dtype=int)
    cols = np.array([0, 2, 1, 2], dtype=int)
    data = np.array([1.0, -1.0, 2.0, 3.0], dtype=float)
    op = ff.FluxSparseOperator(rows, cols, data, shape=(3, 3))

    x = np.array([2.0, 5.0, 7.0], dtype=float)
    y = np.array([11.0, 13.0, 17.0], dtype=float)

    dense = np.asarray(op.to_dense())
    assert np.allclose(np.asarray(op.matvec(x)), dense @ x)
    assert np.allclose(np.asarray(op.rmatvec(y)), dense.T @ y)


def test_flux_sparse_operator_to_csr_shape():
    rows = np.array([0, 1], dtype=int)
    cols = np.array([2, 0], dtype=int)
    data = np.array([1.5, -2.0], dtype=float)
    op = ff.FluxSparseOperator(rows, cols, data, shape=(2, 4))

    csr = op.to_csr()
    assert csr.shape == (2, 4)
    assert np.allclose(csr.toarray(), np.asarray(op.to_dense()))


def test_flux_sparse_matrix_and_operator_support_numpy_asarray():
    mat = ff.FluxSparseMatrix(
        np.array([0, 1], dtype=int),
        np.array([1, 0], dtype=int),
        np.array([2.0, 3.0], dtype=float),
        n_dofs=2,
    )
    op = ff.FluxSparseOperator(
        np.array([0, 1], dtype=int),
        np.array([1, 0], dtype=int),
        np.array([2.0, 3.0], dtype=float),
        shape=(2, 2),
    )

    expected = np.array([[0.0, 2.0], [3.0, 0.0]])
    assert np.allclose(np.asarray(mat), expected)
    assert np.allclose(np.asarray(op), expected)

import numpy as np
import pytest

from fluxfem import FluxSparseMatrix, make_block_jacobi_preconditioner


def test_block_jacobi_dof_per_node():
    rows = np.array([0, 1, 2, 3])
    cols = np.array([0, 1, 2, 3])
    data = np.array([2.0, 2.0, 4.0, 4.0])
    K = FluxSparseMatrix(rows, cols, data, 4, meta={"dof_layout": "blocked"})
    precon = make_block_jacobi_preconditioner(K, dof_per_node=2)
    r = np.array([2.0, 2.0, 4.0, 4.0])
    z = np.asarray(precon(r))
    assert z.shape == r.shape
    np.testing.assert_allclose(z, np.array([1.0, 1.0, 1.0, 1.0]), rtol=1e-7, atol=1e-7)


def test_block_jacobi_meta_layout_guard():
    rows = np.array([0, 1, 2, 3])
    cols = np.array([0, 1, 2, 3])
    data = np.array([1.0, 1.0, 1.0, 1.0])
    K = FluxSparseMatrix(rows, cols, data, 4, meta={"dof_layout": "interleaved", "dof_per_node": 2})
    with pytest.raises(ValueError):
        make_block_jacobi_preconditioner(K)

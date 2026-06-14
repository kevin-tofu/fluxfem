import numpy as np
import pytest
import warnings

import fluxfem as ff

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


def _poisson_2d_matrix(n: int):
    if sp is None:  # pragma: no cover
        pytest.skip("scipy is required for PETSc shell tests")
    e = np.ones(n, dtype=float)
    T = sp.diags([e, -2.0 * e, e], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    I = sp.eye(n, format="csr")
    return sp.kron(I, T, format="csr") + sp.kron(T, I, format="csr")


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
@pytest.mark.parametrize("preconditioner", [None, "diag0"])
def test_petsc_shell_poisson_matches_dense(preconditioner):
    n = 4
    A = _poisson_2d_matrix(n)
    x_true = np.linspace(0.1, 1.0, n * n, dtype=float)
    b = A @ x_true

    x = ff.petsc_shell_solve(
        A,
        b,
        ksp_type="cg",
        preconditioner=preconditioner,
        rtol=1e-10,
        max_it=200,
        options_prefix="fluxfem_test_",
        options={"fluxfem_test_ksp_max_it": 200},
    )
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_petsc_shell_poisson_matvec_callable():
    n = 4
    A = _poisson_2d_matrix(n)
    x_true = np.linspace(0.2, 0.9, n * n, dtype=float)
    b = A @ x_true

    def mv(v):
        return A @ v

    x = ff.petsc_shell_solve(
        mv,
        b,
        n_dofs=A.shape[0],
        ksp_type="cg",
        preconditioner=None,
        rtol=1e-10,
        max_it=200,
    )
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_petsc_shell_diag0_no_warning_for_matrix():
    n = 4
    A = _poisson_2d_matrix(n)
    x_true = np.linspace(0.1, 1.0, n * n, dtype=float)
    b = A @ x_true

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = ff.petsc_shell_solve(
            A,
            b,
            ksp_type="cg",
            preconditioner="diag0",
            rtol=1e-10,
            max_it=200,
        )
    diag0_warnings = [w for w in caught if "diag0" in str(w.message)]
    assert not diag0_warnings
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_petsc_shell_diag0_fallback_for_callable():
    n = 4
    A = _poisson_2d_matrix(n)
    x_true = np.linspace(0.2, 0.9, n * n, dtype=float)
    b = A @ x_true

    def mv(v):
        return A @ v

    with pytest.warns(RuntimeWarning, match="diag0"):
        x = ff.petsc_shell_solve(
            mv,
            b,
            n_dofs=A.shape[0],
            ksp_type="cg",
            preconditioner="diag0",
            rtol=1e-10,
            max_it=200,
        )
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_petsc_shell_diag0_fallback_ignores_pc_type():
    n = 4
    A = _poisson_2d_matrix(n)
    x_true = np.linspace(0.3, 1.1, n * n, dtype=float)
    b = A @ x_true

    def mv(v):
        return A @ v

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = ff.petsc_shell_solve(
            mv,
            b,
            n_dofs=A.shape[0],
            ksp_type="cg",
            pc_type="jacobi",
            preconditioner="diag0",
            rtol=1e-10,
            max_it=200,
        )
    messages = [str(w.message) for w in caught]
    assert any("diag0" in m for m in messages)
    assert any("pc_type" in m for m in messages)
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)

import numpy as np

import fluxfem as ff

try:
    import scipy.sparse as sp
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy is required for this demo") from exc


def poisson_2d_matrix(n: int):
    e = np.ones(n, dtype=float)
    T = sp.diags([e, -2.0 * e, e], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    I = sp.eye(n, format="csr")
    return sp.kron(I, T, format="csr") + sp.kron(T, I, format="csr")


def main():
    if not ff.petsc_is_available():
        raise SystemExit("petsc4py is required for PETSc shell demo")

    n = 8
    A = poisson_2d_matrix(n)
    x_true = np.linspace(0.1, 1.0, n * n, dtype=float)
    b = A @ x_true

    x = ff.petsc_shell_solve(
        A,
        b,
        ksp_type="cg",
        preconditioner="diag0",
        rtol=1e-10,
        max_it=200,
        options_prefix="fluxfem_demo_",
        # Example: override from CLI with -fluxfem_demo_ksp_type gmres
    )
    res = np.linalg.norm(A @ x - b)
    print(f"residual={res:.3e}")


if __name__ == "__main__":
    main()

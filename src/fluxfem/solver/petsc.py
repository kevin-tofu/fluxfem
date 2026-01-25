from __future__ import annotations

import numpy as np
import time
import warnings
from typing import Any, Callable, TypeAlias

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .sparse import FluxSparseMatrix

ArrayLike: TypeAlias = np.ndarray
MatVec: TypeAlias = Callable[[np.ndarray], np.ndarray]
SolveInfo: TypeAlias = dict[str, Any]


def petsc_is_available() -> bool:
    try:
        import petsc4py  # noqa: F401
        return True
    except Exception:
        return False


def _require_petsc4py():
    try:
        import petsc4py
        petsc4py.init([])
        from petsc4py import PETSc
        return PETSc
    except Exception as exc:  # pragma: no cover
        raise ImportError("petsc4py is required for PETSc solves. Install with the petsc extra.") from exc


def _coo_to_csr(
    rows: ArrayLike, cols: ArrayLike, data: ArrayLike, n_dofs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    d = np.asarray(data)
    if r.size == 0:
        indptr = np.zeros(n_dofs + 1, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        return indptr, indices, d
    order = np.lexsort((c, r))
    r_s = r[order]
    c_s = c[order]
    d_s = d[order]
    new_group = np.ones(r_s.size, dtype=bool)
    new_group[1:] = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1])
    starts = np.nonzero(new_group)[0]
    r_u = r_s[starts]
    c_u = c_s[starts]
    d_u = np.add.reduceat(d_s, starts)
    indptr = np.zeros(n_dofs + 1, dtype=np.int32)
    np.add.at(indptr, r_u + 1, 1)
    indptr = np.cumsum(indptr, dtype=np.int32)
    return indptr, c_u.astype(np.int32), d_u


def _infer_n_dofs(K: Any, F: Any | None, n_dofs: int | None) -> int:
    if n_dofs is not None:
        return int(n_dofs)
    if hasattr(K, "n_dofs"):
        return int(getattr(K, "n_dofs"))
    if hasattr(K, "shape"):
        shape = getattr(K, "shape")
        if shape is not None:
            return int(shape[0])
    if F is not None:
        F_arr = np.asarray(F)
        if F_arr.ndim >= 1:
            return int(F_arr.shape[0])
    raise ValueError("n_dofs is required when operator shape is not available.")


def _matvec_builder(A: Any) -> MatVec:
    if isinstance(A, FluxSparseMatrix):
        return lambda x: np.asarray(A.matvec(x))
    if hasattr(A, "matvec"):
        return lambda x: np.asarray(A.matvec(x))
    if callable(A):
        return lambda x: np.asarray(A(x))
    if isinstance(A, tuple) and len(A) == 4:
        return _matvec_builder(FluxSparseMatrix.from_bilinear(A))
    if sp is not None and sp.issparse(A):
        return lambda x: np.asarray(A @ x)

    def mv(x):
        return np.asarray(A) @ x

    return mv


def _diag_from_coo(rows: ArrayLike, cols: ArrayLike, data: ArrayLike, n_dofs: int) -> np.ndarray:
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    d = np.asarray(data)
    diag = np.zeros(n_dofs, dtype=d.dtype)
    mask = r == c
    if np.any(mask):
        np.add.at(diag, r[mask], d[mask])
    return diag


def _diag_from_operator(A: Any, n_dofs: int) -> np.ndarray:
    if isinstance(A, FluxSparseMatrix):
        return np.asarray(A.diag())
    if isinstance(A, tuple) and len(A) == 4:
        rows, cols, data, n_dofs_tuple = A
        n_use = int(n_dofs_tuple) if n_dofs_tuple is not None else n_dofs
        return _diag_from_coo(rows, cols, data, n_use)
    if sp is not None and sp.issparse(A):
        return np.asarray(A.diagonal())
    if hasattr(A, "diag"):
        return np.asarray(A.diag())
    if hasattr(A, "diagonal"):
        return np.asarray(A.diagonal())
    A_np = np.asarray(A)
    if A_np.ndim == 2 and A_np.shape[0] == A_np.shape[1]:
        return np.diag(A_np)
    raise ValueError("diag0 preconditioner requires access to the matrix diagonal.")


def _as_csr(K: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if isinstance(K, FluxSparseMatrix):
        rows, cols, data, n_dofs = K.to_coo()
        indptr, indices, data = _coo_to_csr(rows, cols, data, int(n_dofs))
        return indptr, indices, data, int(n_dofs)
    if isinstance(K, tuple) and len(K) == 4:
        rows, cols, data, n_dofs = K
        indptr, indices, data = _coo_to_csr(rows, cols, data, int(n_dofs))
        return indptr, indices, data, int(n_dofs)
    if sp is not None and sp.issparse(K):
        K_csr = K.tocsr()
        return (
            K_csr.indptr.astype(np.int32, copy=False),
            K_csr.indices.astype(np.int32, copy=False),
            K_csr.data,
            K_csr.shape[0],
        )
    if hasattr(K, "to_csr"):
        K_csr = K.to_csr()
        return K_csr.indptr.astype(np.int32, copy=False), K_csr.indices.astype(np.int32, copy=False), K_csr.data, K_csr.shape[0]
    K_np = np.asarray(K)
    if K_np.ndim != 2 or K_np.shape[0] != K_np.shape[1]:
        raise ValueError("K must be square for PETSc solve.")
    rows, cols = np.nonzero(K_np)
    data = K_np[rows, cols]
    indptr, indices, data = _coo_to_csr(rows, cols, data, K_np.shape[0])
    return indptr, indices, data, int(K_np.shape[0])


def petsc_solve(
    K: Any,
    F: Any,
    *,
    ksp_type: str = "preonly",
    pc_type: str = "lu",
    rtol: float | None = None,
    atol: float | None = None,
    max_it: int | None = None,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Solve K u = F using PETSc.

    Parameters
    ----------
    K : FluxSparseMatrix | COO tuple | ndarray | scipy.sparse matrix
        Assembled system matrix. COO tuple is (rows, cols, data, n_dofs).
    F : array-like
        RHS vector (n_dofs,) or matrix (n_dofs, n_rhs).
    ksp_type / pc_type : str
        PETSc KSP/PC type, e.g., "cg"/"gamg" or "preonly"/"lu".
    options : dict
        Extra PETSc options (name -> value).
    """
    PETSc = _require_petsc4py()
    indptr, indices, data, n_dofs = _as_csr(K)

    mat = PETSc.Mat().createAIJ(size=(n_dofs, n_dofs), csr=(indptr, indices, np.asarray(data)))
    mat.assemble()

    ksp = PETSc.KSP().create()
    ksp.setOperators(mat)
    if ksp_type:
        ksp.setType(ksp_type)
    if pc_type:
        ksp.getPC().setType(pc_type)
    if rtol is not None or atol is not None or max_it is not None:
        ksp.setTolerances(
            rtol=rtol if rtol is not None else PETSc.DEFAULT,
            atol=atol if atol is not None else PETSc.DEFAULT,
            max_it=max_it if max_it is not None else PETSc.DEFAULT,
        )
    if options:
        opts = PETSc.Options()
        for key, value in options.items():
            opts[str(key)] = str(value)
        ksp.setFromOptions()

    F_arr = np.asarray(F)
    if F_arr.ndim == 1:
        if F_arr.shape[0] != n_dofs:
            raise ValueError("F has incompatible size for K.")
        b = PETSc.Vec().createWithArray(F_arr)
        x = PETSc.Vec().createSeq(n_dofs)
        ksp.solve(b, x)
        return np.asarray(x.getArray(), copy=True)

    if F_arr.ndim == 2:
        if F_arr.shape[0] != n_dofs:
            raise ValueError("F has incompatible size for K.")
        out = []
        for i in range(F_arr.shape[1]):
            b = PETSc.Vec().createWithArray(F_arr[:, i])
            x = PETSc.Vec().createSeq(n_dofs)
            ksp.solve(b, x)
            out.append(np.asarray(x.getArray(), copy=True))
        return np.stack(out, axis=1)

    raise ValueError("F must be a vector or a 2D array.")


def petsc_shell_solve(
    A: Any,
    F: Any,
    *,
    n_dofs: int | None = None,
    ksp_type: str = "gmres",
    pc_type: str = "none",
    preconditioner: str | None | Callable[[np.ndarray], np.ndarray] = "diag0",
    pmat: Any | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_it: int | None = None,
    options: dict[str, Any] | None = None,
    options_prefix: str | None = "fluxfem_",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, SolveInfo]:
    """
    Solve A x = F using PETSc with a matrix-free Shell Mat.

    Parameters
    ----------
    A : callable | object with matvec | FluxSparseMatrix | tuple | ndarray
        Operator to apply in matvec form.
    F : array-like
        RHS vector (n_dofs,) or matrix (n_dofs, n_rhs).
    preconditioner : "diag0" | callable | None
        "diag0" builds a diagonal preconditioner if available.
    pmat : optional
        Assembled matrix used only as the preconditioner operator.
    return_info : bool
        When True, return a (solution, info) tuple.
    """
    PETSc = _require_petsc4py()
    n = _infer_n_dofs(A, F, n_dofs)
    matvec = _matvec_builder(A)

    class _ShellMatContext:
        def __init__(self, mv):
            self.mv = mv

        def mult(self, mat, x, y):
            x_arr = x.getArray(readonly=True)
            y_arr = np.asarray(self.mv(x_arr), dtype=x_arr.dtype)
            y.setArray(y_arr)

    mat_ctx = _ShellMatContext(matvec)
    mat = PETSc.Mat().createPython([n, n])
    mat.setPythonContext(mat_ctx)
    mat.setUp()

    pmat_aij = None
    pmat_build_time = None
    if pmat is not None:
        t_pmat = time.perf_counter()
        indptr, indices, data, n_p = _as_csr(pmat)
        if n_p != n:
            raise ValueError("pmat has incompatible size for A.")
        pmat_aij = PETSc.Mat().createAIJ(size=(n, n), csr=(indptr, indices, np.asarray(data)))
        pmat_aij.assemble()
        pmat_build_time = time.perf_counter() - t_pmat

    ksp = PETSc.KSP().create()
    if pmat_aij is None:
        ksp.setOperators(mat)
    else:
        ksp.setOperators(mat, pmat_aij)
    if options_prefix:
        ksp.setOptionsPrefix(options_prefix)

    if ksp_type:
        ksp.setType(ksp_type)
    if rtol is not None or atol is not None or max_it is not None:
        ksp.setTolerances(
            rtol=rtol if rtol is not None else PETSc.DEFAULT,
            atol=atol if atol is not None else PETSc.DEFAULT,
            max_it=max_it if max_it is not None else PETSc.DEFAULT,
        )
    if options or options_prefix:
        if options:
            opts = PETSc.Options()
            for key, value in options.items():
                opts[str(key)] = str(value)
        ksp.setFromOptions()
        if ksp_type:
            ksp.setType(ksp_type)
        if rtol is not None or atol is not None or max_it is not None:
            ksp.setTolerances(
                rtol=rtol if rtol is not None else PETSc.DEFAULT,
                atol=atol if atol is not None else PETSc.DEFAULT,
                max_it=max_it if max_it is not None else PETSc.DEFAULT,
            )

    pc = ksp.getPC()
    if preconditioner is None:
        if pc_type:
            pc.setType(pc_type)
    else:
        if preconditioner == "diag0":
            diag_source = pmat if pmat is not None else A
            try:
                diag = _diag_from_operator(diag_source, n)
            except Exception as exc:
                warnings.warn(
                    f"diag0 preconditioner unavailable ({exc}); falling back to no preconditioner.",
                    RuntimeWarning,
                )
                preconditioner = None
                diag = None

            if diag is not None:
                inv_diag = np.where(diag != 0.0, 1.0 / diag, 0.0)

                class _DiagPCContext:
                    def __init__(self, inv):
                        self.inv = inv

                    def apply(self, pc, x, y):
                        x_arr = x.getArray(readonly=True)
                        y.setArray(self.inv * x_arr)

                pc.setType(PETSc.PC.Type.PYTHON)
                pc.setPythonContext(_DiagPCContext(inv_diag))
            elif pc_type and pc_type not in ("none", "NONE"):
                warnings.warn(
                    f"pc_type='{pc_type}' requires a usable diagonal; falling back to pc_type='none'.",
                    RuntimeWarning,
                )
                pc.setType("none")
            elif pc_type:
                pc.setType(pc_type)
        elif callable(preconditioner):

            class _CallablePCContext:
                def __init__(self, fn):
                    self.fn = fn

                def apply(self, pc, x, y):
                    x_arr = x.getArray(readonly=True)
                    y_arr = np.asarray(self.fn(x_arr), dtype=x_arr.dtype)
                    y.setArray(y_arr)

            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(_CallablePCContext(preconditioner))
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner}")

    def _ksp_info(solve_time=None, pc_setup_time=None):
        try:
            reason = ksp.getConvergedReason()
        except Exception:  # pragma: no cover - defensive
            reason = None
        try:
            iters = ksp.getIterationNumber()
        except Exception:  # pragma: no cover - defensive
            iters = None
        try:
            res = ksp.getResidualNorm()
        except Exception:  # pragma: no cover - defensive
            res = None
        converged = None
        if reason is not None:
            converged = reason > 0
        return {
            "iters": iters,
            "residual_norm": res,
            "converged": converged,
            "reason": reason,
            "pmat_build_time": pmat_build_time,
            "pc_setup_time": pc_setup_time,
            "solve_time": solve_time,
        }

    F_arr = np.asarray(F)
    if F_arr.ndim == 1:
        if F_arr.shape[0] != n:
            raise ValueError("F has incompatible size for A.")
        b = PETSc.Vec().createWithArray(F_arr)
        x = PETSc.Vec().createSeq(n)
        t_setup = time.perf_counter()
        ksp.setUp()
        pc_setup_time = time.perf_counter() - t_setup
        t_solve = time.perf_counter()
        ksp.solve(b, x)
        solve_time = time.perf_counter() - t_solve
        x_out = np.asarray(x.getArray(), copy=True)
        if return_info:
            return x_out, _ksp_info(solve_time=solve_time, pc_setup_time=pc_setup_time)
        return x_out

    if F_arr.ndim == 2:
        if F_arr.shape[0] != n:
            raise ValueError("F has incompatible size for A.")
        out = []
        infos = []
        t_setup = time.perf_counter()
        ksp.setUp()
        pc_setup_time = time.perf_counter() - t_setup
        for i in range(F_arr.shape[1]):
            b = PETSc.Vec().createWithArray(F_arr[:, i])
            x = PETSc.Vec().createSeq(n)
            t_solve = time.perf_counter()
            ksp.solve(b, x)
            solve_time = time.perf_counter() - t_solve
            out.append(np.asarray(x.getArray(), copy=True))
            infos.append(_ksp_info(solve_time=solve_time, pc_setup_time=pc_setup_time))
        x_out = np.stack(out, axis=1)
        if return_info:
            info = {
                "iters": [i.get("iters") for i in infos],
                "residual_norm": np.asarray([i.get("residual_norm") for i in infos]),
                "converged": [i.get("converged") for i in infos],
                "reason": [i.get("reason") for i in infos],
            }
            return x_out, info
        return x_out

    raise ValueError("F must be a vector or a 2D array.")

from __future__ import annotations

import numpy as np
from typing import Any

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .sparse import FluxSparseMatrix


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


def _coo_to_csr(rows, cols, data, n_dofs: int):
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


def _as_csr(K: Any):
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

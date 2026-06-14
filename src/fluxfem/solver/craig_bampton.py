from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


def _optional_scipy_sparse():
    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover
        return None
    return sp


def _matrix_shape(matrix) -> tuple[int, int]:
    if hasattr(matrix, "shape"):
        return tuple(int(v) for v in matrix.shape)
    if hasattr(matrix, "to_csr"):
        return tuple(int(v) for v in matrix.to_csr().shape)
    arr = jnp.asarray(matrix)
    return tuple(int(v) for v in arr.shape)


def _as_dense_array(matrix) -> Array:
    if hasattr(matrix, "to_dense"):
        return jnp.asarray(matrix.to_dense())
    if hasattr(matrix, "toarray"):
        return jnp.asarray(matrix.toarray())
    return jnp.asarray(matrix)


def _matrix_dtype(matrix):
    if hasattr(matrix, "data"):
        return jnp.asarray(matrix.data).dtype
    if hasattr(matrix, "dtype"):
        return jnp.asarray(np.empty((), dtype=matrix.dtype)).dtype
    return jnp.asarray(matrix).dtype


def _result_dtype(*matrices):
    return jnp.result_type(*[jnp.empty((), dtype=_matrix_dtype(matrix)) for matrix in matrices])


def _take_block(matrix, rows: Array, cols: Array):
    sp = _optional_scipy_sparse()
    if hasattr(matrix, "to_csr"):
        matrix = matrix.to_csr()
    if sp is not None and sp.issparse(matrix):
        rows_np = np.asarray(rows, dtype=np.int32).reshape(-1)
        cols_np = np.asarray(cols, dtype=np.int32).reshape(-1)
        return matrix.tocsr()[rows_np, :][:, cols_np].tocsr()
    return jnp.asarray(matrix)[jnp.asarray(rows, dtype=jnp.int32)[:, None], jnp.asarray(cols, dtype=jnp.int32)[None, :]]


def complement_dofs(n_dofs: int, retained_dofs: Array) -> Array:
    retained = jnp.asarray(retained_dofs, dtype=jnp.int32)
    mask = jnp.ones((int(n_dofs),), dtype=bool).at[retained].set(False)
    return jnp.nonzero(mask, size=int(n_dofs) - int(retained.size))[0].astype(jnp.int32)


def _mass_normalize(modes: Array, mass) -> Array:
    mass_dense = _as_dense_array(mass)
    norms2 = jnp.einsum("ia,ij,ja->a", modes, mass_dense, modes)
    return modes / jnp.sqrt(jnp.maximum(norms2, jnp.finfo(modes.dtype).eps))[None, :]


def solve_constraint_modes(
    stiffness_ii,
    stiffness_ir,
    *,
    solver: str | Callable[[object, Array], Array] = "dense",
) -> Array:
    """Solve Craig-Bampton static constraint modes `K_ii Psi = -K_ir`."""
    stiffness_shape = _matrix_shape(stiffness_ii)
    stiffness_ir = _as_dense_array(stiffness_ir)
    if stiffness_shape[0] != stiffness_shape[1]:
        raise ValueError("stiffness_ii must be square.")
    if stiffness_ir.ndim != 2 or stiffness_ir.shape[0] != stiffness_shape[0]:
        raise ValueError("stiffness_ir must have shape (n_internal, n_retained).")
    if stiffness_ir.shape[1] == 0:
        return jnp.zeros((stiffness_shape[0], 0), dtype=_result_dtype(stiffness_ii, stiffness_ir))

    rhs = -stiffness_ir
    if callable(solver):
        return jnp.asarray(solver(stiffness_ii, rhs))
    if solver == "dense":
        return jnp.linalg.solve(_as_dense_array(stiffness_ii), rhs)
    if solver == "spsolve":
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for constraint_solver='spsolve'.") from exc
        if hasattr(stiffness_ii, "to_csr"):
            k_csr = stiffness_ii.to_csr()
        elif sp.issparse(stiffness_ii):
            k_csr = stiffness_ii.tocsr()
        else:
            k_csr = sp.csr_matrix(np.asarray(stiffness_ii))
        solution = spla.spsolve(k_csr, np.asarray(rhs))
        if solution.ndim == 1:
            solution = solution[:, None]
        return jnp.asarray(solution, dtype=_result_dtype(stiffness_ii, rhs))
    raise ValueError("constraint_solver must be 'dense', 'spsolve', or a callable.")


def _dense_fixed_interface_modes(stiffness_ii, mass_ii, n_modes: int) -> tuple[Array, Array]:
    stiffness_ii = _as_dense_array(stiffness_ii)
    mass_ii = _as_dense_array(mass_ii)
    chol_m = jnp.linalg.cholesky(mass_ii)
    tmp = jnp.linalg.solve(chol_m, stiffness_ii)
    standard_op = jnp.linalg.solve(chol_m, tmp.T).T
    standard_op = 0.5 * (standard_op + standard_op.T)
    eigvals, eigvecs = jnp.linalg.eigh(standard_op)
    z = eigvecs[:, :n_modes]
    modes = jnp.linalg.solve(chol_m.T, z)
    modes = _mass_normalize(modes, mass_ii)
    return modes, eigvals[:n_modes]


def _scipy_eigsh_fixed_interface_modes(stiffness_ii, mass_ii, n_modes: int, *, tol: float, maxiter: int):
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for modal_solver='eigsh'.") from exc

    n_internal = int(_matrix_shape(stiffness_ii)[0])
    if n_modes >= n_internal:
        return _dense_fixed_interface_modes(stiffness_ii, mass_ii, n_modes)

    if hasattr(stiffness_ii, "to_csr"):
        k_csr = stiffness_ii.to_csr()
    elif sp.issparse(stiffness_ii):
        k_csr = stiffness_ii.tocsr()
    else:
        k_csr = sp.csr_matrix(np.asarray(stiffness_ii))
    if hasattr(mass_ii, "to_csr"):
        m_csr = mass_ii.to_csr()
    elif sp.issparse(mass_ii):
        m_csr = mass_ii.tocsr()
    else:
        m_csr = sp.csr_matrix(np.asarray(mass_ii))

    eigvals, eigvecs = spla.eigsh(k_csr, k=int(n_modes), M=m_csr, which="SM", tol=float(tol), maxiter=int(maxiter))
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    modes = _mass_normalize(jnp.asarray(eigvecs), mass_ii)
    return modes, jnp.asarray(eigvals, dtype=modes.dtype)


def fixed_interface_modes(
    stiffness_ii,
    mass_ii,
    n_modes: int,
    *,
    solver: str | Callable[[object, object, int], tuple[Array, Array]] = "dense",
    modal_tol: float = 1e-8,
    modal_maxiter: int = 300,
) -> tuple[Array, Array]:
    """Compute fixed-interface modes from `K_ii phi = lambda M_ii phi`."""
    if n_modes < 0:
        raise ValueError("n_modes must be non-negative.")
    n_internal = int(_matrix_shape(stiffness_ii)[0])
    if n_modes == 0 or n_internal == 0:
        dtype = _result_dtype(stiffness_ii, mass_ii)
        return jnp.zeros((n_internal, 0), dtype=dtype), jnp.zeros((0,), dtype=dtype)

    n_keep = min(int(n_modes), n_internal)
    if callable(solver):
        modes, eigenvalues = solver(stiffness_ii, mass_ii, n_keep)
        return _mass_normalize(jnp.asarray(modes), mass_ii), jnp.asarray(eigenvalues)[:n_keep]
    if solver == "dense":
        return _dense_fixed_interface_modes(stiffness_ii, mass_ii, n_keep)
    if solver == "eigsh":
        return _scipy_eigsh_fixed_interface_modes(stiffness_ii, mass_ii, n_keep, tol=modal_tol, maxiter=modal_maxiter)
    raise ValueError("modal_solver must be 'dense', 'eigsh', or a callable.")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CraigBamptonBasis:
    """Craig-Bampton basis with retained physical DOFs followed by internal modes."""

    basis: Array
    retained_dofs: Array
    internal_dofs: Array
    eigenvalues: Array

    @property
    def n_full(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_reduced(self) -> int:
        return int(self.basis.shape[1])

    @property
    def n_retained(self) -> int:
        return int(self.retained_dofs.size)

    @property
    def n_modes(self) -> int:
        return int(self.eigenvalues.size)

    def expand(self, q: Array) -> Array:
        return self.basis @ q

    def project_vector(self, vector: Array) -> Array:
        return self.basis.T @ jnp.asarray(vector)

    def project_matrix(self, matrix) -> Array:
        return self.basis.T @ _as_dense_array(matrix) @ self.basis

    def reduced_residual(self, residual_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
        def _residual(q: Array) -> Array:
            return self.project_vector(residual_fn(self.expand(q)))

        return _residual

    def reduced_jacobian(self, residual_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
        return jax.jacrev(self.reduced_residual(residual_fn))

    def tree_flatten(self):
        return (self.basis, self.retained_dofs, self.internal_dofs, self.eigenvalues), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        basis, retained_dofs, internal_dofs, eigenvalues = children
        return cls(basis, retained_dofs, internal_dofs, eigenvalues)


def make_craig_bampton_basis(
    stiffness,
    mass,
    retained_dofs: Array,
    n_modes: int,
    *,
    constraint_solver: str | Callable[[object, Array], Array] = "dense",
    modal_solver: str | Callable[[object, object, int], tuple[Array, Array]] = "dense",
    modal_tol: float = 1e-8,
    modal_maxiter: int = 300,
) -> CraigBamptonBasis:
    """Build a Craig-Bampton basis for assembled full-order matrices."""
    stiffness_shape = _matrix_shape(stiffness)
    mass_shape = _matrix_shape(mass)
    if stiffness_shape[0] != stiffness_shape[1]:
        raise ValueError("stiffness must be square.")
    if mass_shape != stiffness_shape:
        raise ValueError("mass must have the same shape as stiffness.")

    n_full = int(stiffness_shape[0])
    retained_np = np.asarray(retained_dofs, dtype=np.int32).reshape(-1)
    if retained_np.size and (retained_np.min() < 0 or retained_np.max() >= n_full):
        raise ValueError("retained_dofs contains an index outside the full DOF range.")
    if np.unique(retained_np).size != retained_np.size:
        raise ValueError("retained_dofs must not contain duplicates.")

    retained = jnp.asarray(retained_np, dtype=jnp.int32)
    internal = complement_dofs(n_full, retained)
    if internal.size == 0:
        return CraigBamptonBasis(
            basis=jnp.eye(n_full, dtype=_result_dtype(stiffness, mass)),
            retained_dofs=retained,
            internal_dofs=internal,
            eigenvalues=jnp.zeros((0,), dtype=_result_dtype(stiffness, mass)),
        )

    k_ii = _take_block(stiffness, internal, internal)
    k_ir = _take_block(stiffness, internal, retained)
    m_ii = _take_block(mass, internal, internal)

    if retained.size:
        constraint_modes = solve_constraint_modes(k_ii, k_ir, solver=constraint_solver)
    else:
        constraint_modes = jnp.zeros((internal.size, 0), dtype=_result_dtype(stiffness, mass))
    normal_modes, eigenvalues = fixed_interface_modes(
        k_ii,
        m_ii,
        n_modes,
        solver=modal_solver,
        modal_tol=modal_tol,
        modal_maxiter=modal_maxiter,
    )

    n_reduced = int(retained.size) + int(normal_modes.shape[1])
    basis = jnp.zeros((n_full, n_reduced), dtype=_result_dtype(stiffness, mass))
    if retained.size:
        basis = basis.at[internal, : retained.size].set(constraint_modes)
        basis = basis.at[retained, : retained.size].set(jnp.eye(int(retained.size), dtype=basis.dtype))
    if normal_modes.shape[1]:
        basis = basis.at[internal, retained.size :].set(normal_modes)

    return CraigBamptonBasis(basis=basis, retained_dofs=retained, internal_dofs=internal, eigenvalues=eigenvalues)


def reduced_residual_from_full(cb: CraigBamptonBasis, residual_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return cb.reduced_residual(residual_fn)


def reduced_jacobian_from_full(cb: CraigBamptonBasis, residual_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return cb.reduced_jacobian(residual_fn)


__all__ = [
    "CraigBamptonBasis",
    "complement_dofs",
    "fixed_interface_modes",
    "make_craig_bampton_basis",
    "reduced_jacobian_from_full",
    "reduced_residual_from_full",
    "solve_constraint_modes",
]

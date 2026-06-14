from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

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


def _cg_solve_matrix(matrix, rhs: Array, *, tol: float, maxiter: int | None) -> Array:
    """Solve SPD multi-RHS systems with a small JAX CG loop."""
    matrix = _as_dense_array(matrix)
    rhs = jnp.asarray(rhs)
    if rhs.ndim != 2:
        raise ValueError("rhs must have shape (n, n_rhs).")
    n = int(matrix.shape[0])
    if matrix.shape != (n, n):
        raise ValueError("matrix must be square.")
    if rhs.shape[0] != n:
        raise ValueError("rhs row count must match matrix size.")
    if rhs.shape[1] == 0 or n == 0:
        return jnp.zeros_like(rhs)

    maxiter = n if maxiter is None else int(maxiter)
    if maxiter <= 0:
        raise ValueError("maxiter must be positive.")
    tol = float(tol)
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    x = jnp.zeros_like(rhs)
    r = rhs - matrix @ x
    p = r
    rs_old = jnp.sum(r * r, axis=0)
    rhs_norm = jnp.sqrt(jnp.maximum(jnp.sum(rhs * rhs, axis=0), jnp.finfo(rhs.dtype).eps))
    eps = jnp.finfo(rhs.dtype).eps
    active = jnp.ones((rhs.shape[1],), dtype=bool)

    for _ in range(maxiter):
        ap = matrix @ p
        denom = jnp.sum(p * ap, axis=0)
        alpha = jnp.where(active, rs_old / jnp.maximum(denom, eps), 0.0)
        x = x + p * alpha[None, :]
        r = r - ap * alpha[None, :]
        rs_new = jnp.sum(r * r, axis=0)
        converged = jnp.sqrt(jnp.maximum(rs_new, 0.0)) <= tol * rhs_norm
        active = active & ~converged
        beta = jnp.where(active, rs_new / jnp.maximum(rs_old, eps), 0.0)
        p = r + p * beta[None, :]
        rs_old = rs_new
        if not bool(jnp.any(active)):
            break
    return x


def solve_constraint_modes(
    stiffness_ii,
    stiffness_ir,
    *,
    solver: str | Callable[[object, Array], Array] = "dense",
    cg_tol: float = 1e-10,
    cg_maxiter: int | None = None,
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
    if solver == "cg":
        return _cg_solve_matrix(stiffness_ii, rhs, tol=cg_tol, maxiter=cg_maxiter)
    raise ValueError("constraint_solver must be 'dense', 'cg', 'spsolve', or a callable.")


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


def _m_orthonormalize(vectors: Array, mass) -> Array:
    mass_dense = _as_dense_array(mass)
    gram = vectors.T @ mass_dense @ vectors
    chol = jnp.linalg.cholesky(0.5 * (gram + gram.T))
    return jnp.linalg.solve(chol, vectors.T).T


def _subspace_fixed_interface_modes(
    stiffness_ii,
    mass_ii,
    n_modes: int,
    *,
    oversample: int,
    maxiter: int,
    tol: float,
    linear_solver: str | Callable[[object, Array], Array],
    cg_tol: float,
    cg_maxiter: int | None,
) -> tuple[Array, Array]:
    """Block inverse/subspace iteration for the lowest fixed-interface modes."""
    n_internal = int(_matrix_shape(stiffness_ii)[0])
    mass_dense = _as_dense_array(mass_ii)
    block_size = min(n_internal, max(int(n_modes), int(n_modes) + int(oversample)))
    if block_size <= 0:
        dtype = _result_dtype(stiffness_ii, mass_ii)
        return jnp.zeros((n_internal, 0), dtype=dtype), jnp.zeros((0,), dtype=dtype)
    maxiter = int(maxiter)
    if maxiter <= 0:
        raise ValueError("modal_maxiter must be positive.")
    tol = float(tol)
    if tol < 0.0:
        raise ValueError("modal_tol must be non-negative.")

    dtype = _result_dtype(stiffness_ii, mass_ii)
    q = _m_orthonormalize(jnp.eye(n_internal, block_size, dtype=dtype), mass_dense)
    previous = None
    theta = None

    for _ in range(maxiter):
        z = solve_constraint_modes(
            stiffness_ii,
            -(mass_dense @ q),
            solver=linear_solver,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
        q = _m_orthonormalize(z, mass_dense)
        projected_k = q.T @ _as_dense_array(stiffness_ii) @ q
        projected_k = 0.5 * (projected_k + projected_k.T)
        theta_all, y = jnp.linalg.eigh(projected_k)
        q = q @ y
        theta = theta_all
        current = theta_all[:n_modes]
        if previous is not None:
            denom = jnp.maximum(jnp.linalg.norm(current), jnp.finfo(dtype).eps)
            if float(jnp.linalg.norm(current - previous) / denom) <= tol:
                break
        previous = current

    modes = _mass_normalize(q[:, :n_modes], mass_dense)
    eigenvalues = jnp.einsum("ia,ij,ja->a", modes, _as_dense_array(stiffness_ii), modes)
    if theta is not None:
        eigenvalues = theta[:n_modes]
    return modes, eigenvalues


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
    modal_linear_solver: str | Callable[[object, Array], Array] = "dense",
    modal_oversample: int = 2,
    modal_tol: float = 1e-8,
    modal_maxiter: int = 30,
    cg_tol: float = 1e-10,
    cg_maxiter: int | None = None,
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
    if solver == "subspace":
        return _subspace_fixed_interface_modes(
            stiffness_ii,
            mass_ii,
            n_keep,
            oversample=modal_oversample,
            maxiter=modal_maxiter,
            tol=modal_tol,
            linear_solver=modal_linear_solver,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
    if solver == "eigsh":
        return _scipy_eigsh_fixed_interface_modes(stiffness_ii, mass_ii, n_keep, tol=modal_tol, maxiter=modal_maxiter)
    raise ValueError("modal_solver must be 'dense', 'subspace', 'eigsh', or a callable.")


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


@dataclass(frozen=True)
class LinearConstraintSystem:
    """Dense linear equality constraints `C u = rhs` for KKT solves and ROM projection."""

    matrix: Array
    rhs: Array | None = None

    def __post_init__(self):
        matrix = jnp.asarray(self.matrix)
        if matrix.ndim != 2:
            raise ValueError("constraint matrix must have shape (n_constraints, n_dofs).")
        rhs = jnp.zeros((matrix.shape[0],), dtype=matrix.dtype) if self.rhs is None else jnp.asarray(self.rhs)
        if rhs.shape != (matrix.shape[0],):
            raise ValueError("constraint rhs must have shape (n_constraints,).")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "rhs", rhs)

    @property
    def n_constraints(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_dofs(self) -> int:
        return int(self.matrix.shape[1])

    def residual(self, u: Array) -> Array:
        return self.matrix @ jnp.asarray(u) - self.rhs

    def project(self, basis: CraigBamptonBasis, *, n_extra_dofs: int = 0) -> "ReducedLinearConstraintSystem":
        """Project constraints through a CB basis, optionally preserving appended DOFs."""
        n_extra_dofs = int(n_extra_dofs)
        if n_extra_dofs < 0:
            raise ValueError("n_extra_dofs must be non-negative.")
        if self.n_dofs != basis.n_full + n_extra_dofs:
            raise ValueError("constraint matrix column count must match basis.n_full + n_extra_dofs.")
        dtype = jnp.result_type(self.matrix, basis.basis)
        transform = jnp.zeros((self.n_dofs, basis.n_reduced + n_extra_dofs), dtype=dtype)
        transform = transform.at[: basis.n_full, : basis.n_reduced].set(basis.basis)
        if n_extra_dofs:
            transform = transform.at[basis.n_full :, basis.n_reduced :].set(jnp.eye(n_extra_dofs, dtype=dtype))
        return ReducedLinearConstraintSystem(
            matrix=self.matrix @ transform,
            rhs=self.rhs,
            basis=basis,
            n_extra_dofs=n_extra_dofs,
        )

    def solve(
        self,
        stiffness,
        force: Array,
        *,
        fixed_dofs: Array | None = None,
        fixed_values: Array | None = None,
        solver: str = "dense",
    ) -> Array:
        return solve_linear_constraint_kkt(
            stiffness,
            force,
            self,
            fixed_dofs=fixed_dofs,
            fixed_values=fixed_values,
            solver=solver,
        )


@dataclass(frozen=True)
class ReducedLinearConstraintSystem:
    """Linear constraints projected into `[q_cb, u_extra]` coordinates."""

    matrix: Array
    rhs: Array
    basis: CraigBamptonBasis
    n_extra_dofs: int = 0

    @property
    def n_reduced(self) -> int:
        return int(self.matrix.shape[1])

    def residual(self, q: Array) -> Array:
        return self.matrix @ jnp.asarray(q) - self.rhs

    def expand(self, q: Array) -> Array:
        q = jnp.asarray(q)
        structural = self.basis.expand(q[: self.basis.n_reduced])
        if self.n_extra_dofs == 0:
            return structural
        return jnp.concatenate([structural, q[self.basis.n_reduced :]])

    def solve(
        self,
        stiffness,
        force: Array,
        *,
        fixed_dofs: Array | None = None,
        fixed_values: Array | None = None,
        solver: str = "dense",
    ) -> Array:
        return LinearConstraintSystem(self.matrix, self.rhs).solve(
            stiffness,
            force,
            fixed_dofs=fixed_dofs,
            fixed_values=fixed_values,
            solver=solver,
        )


@dataclass(frozen=True)
class RBE3Patch:
    """Thin translational RBE3-style weighted average over structural DOFs."""

    dofs: Array
    weights: Array

    def __post_init__(self):
        dofs = jnp.asarray(self.dofs, dtype=jnp.int32)
        weights = jnp.asarray(self.weights)
        if dofs.ndim != 2:
            raise ValueError("RBE3Patch dofs must have shape (n_points, dim).")
        if weights.shape != (dofs.shape[0],):
            raise ValueError("RBE3Patch weights must have shape (n_points,).")
        if dofs.shape[0] == 0 or dofs.shape[1] == 0:
            raise ValueError("RBE3Patch must contain at least one point and one component.")
        if bool(jnp.any(dofs < 0)):
            raise ValueError("RBE3Patch dofs must be non-negative.")
        weight_sum = jnp.sum(weights)
        if float(jnp.abs(weight_sum)) <= 0.0:
            raise ValueError("RBE3Patch weights must have a non-zero sum.")
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "weights", weights / weight_sum)

    @property
    def dim(self) -> int:
        return int(self.dofs.shape[1])

    @property
    def retained_dofs(self) -> Array:
        return jnp.unique(self.dofs.reshape(-1)).astype(jnp.int32)

    def average_matrix(self, n_structural_dofs: int) -> Array:
        n_structural_dofs = int(n_structural_dofs)
        if n_structural_dofs <= 0:
            raise ValueError("n_structural_dofs must be positive.")
        if int(jnp.max(self.dofs)) >= n_structural_dofs:
            raise ValueError("RBE3Patch dofs contain an index outside n_structural_dofs.")
        b = jnp.zeros((self.dim, n_structural_dofs), dtype=self.weights.dtype)
        for point in range(int(self.dofs.shape[0])):
            for component in range(self.dim):
                b = b.at[component, self.dofs[point, component]].add(self.weights[point])
        return b


@dataclass(frozen=True)
class ReferencePointFixture:
    """Reference-point fixture backed by an RBE3-style patch MPC."""

    name: str
    patch: RBE3Patch
    reference_dofs: Array
    direction: Array | None = None
    stiffness: float = 0.0
    target_displacement: float = 0.0

    def __post_init__(self):
        reference_dofs = jnp.asarray(self.reference_dofs, dtype=jnp.int32)
        if reference_dofs.shape != (self.patch.dim,):
            raise ValueError("reference_dofs must have shape (patch.dim,).")
        if bool(jnp.any(reference_dofs < 0)):
            raise ValueError("reference_dofs must be non-negative.")
        if np.unique(np.asarray(reference_dofs)).size != int(reference_dofs.size):
            raise ValueError("reference_dofs must not contain duplicates.")
        if self.direction is None:
            direction = jnp.zeros((self.patch.dim,), dtype=self.patch.weights.dtype).at[-1].set(1.0)
        else:
            direction = jnp.asarray(self.direction, dtype=self.patch.weights.dtype)
        if direction.shape != (self.patch.dim,):
            raise ValueError("direction must have shape (patch.dim,).")
        norm = jnp.linalg.norm(direction)
        if float(norm) <= 0.0:
            raise ValueError("direction must have non-zero norm.")
        object.__setattr__(self, "reference_dofs", reference_dofs)
        object.__setattr__(self, "direction", direction / norm)

    @property
    def retained_dofs(self) -> Array:
        return self.patch.retained_dofs

    def constraint_matrix(self, n_structural_dofs: int, total_dofs: int) -> Array:
        n_structural_dofs = int(n_structural_dofs)
        total_dofs = int(total_dofs)
        if total_dofs < n_structural_dofs:
            raise ValueError("total_dofs must be at least n_structural_dofs.")
        if int(jnp.max(self.reference_dofs)) >= total_dofs:
            raise ValueError("reference_dofs contain an index outside total_dofs.")
        c = jnp.zeros((self.patch.dim, total_dofs), dtype=self.patch.weights.dtype)
        c = c.at[:, :n_structural_dofs].set(-self.patch.average_matrix(n_structural_dofs))
        c = c.at[jnp.arange(self.patch.dim), self.reference_dofs].set(1.0)
        return c

    def preload_stiffness_force(self, total_dofs: int, *, active: bool = True) -> tuple[Array, Array]:
        total_dofs = int(total_dofs)
        k = jnp.zeros((total_dofs, total_dofs), dtype=self.patch.weights.dtype)
        f = jnp.zeros((total_dofs,), dtype=self.patch.weights.dtype)
        if not active or float(self.stiffness) == 0.0:
            return k, f
        if int(jnp.max(self.reference_dofs)) >= total_dofs:
            raise ValueError("reference_dofs contain an index outside total_dofs.")
        d = jnp.asarray(self.direction, dtype=self.patch.weights.dtype)
        kk = float(self.stiffness) * jnp.outer(d, d)
        ff = float(self.stiffness) * float(self.target_displacement) * d
        refs = self.reference_dofs
        k = k.at[refs[:, None], refs[None, :]].add(kk)
        f = f.at[refs].add(ff)
        return k, f


def linear_constraint_system_from_reference_fixtures(
    fixtures: tuple[ReferencePointFixture, ...] | list[ReferencePointFixture],
    *,
    n_structural_dofs: int,
    total_dofs: int,
) -> LinearConstraintSystem:
    if len(fixtures) == 0:
        return LinearConstraintSystem(jnp.zeros((0, int(total_dofs))))
    rows = [
        fixture.constraint_matrix(n_structural_dofs=n_structural_dofs, total_dofs=total_dofs)
        for fixture in fixtures
    ]
    return LinearConstraintSystem(jnp.vstack(rows))


def assemble_reference_fixture_preload(
    fixtures: tuple[ReferencePointFixture, ...] | list[ReferencePointFixture],
    *,
    total_dofs: int,
    active_names: set[str] | frozenset[str] | None = None,
    sparse: bool = False,
) -> tuple[Array, Array]:
    total_dofs = int(total_dofs)
    dtype = fixtures[0].patch.weights.dtype if len(fixtures) else jnp.float32
    if sparse:
        try:
            import scipy.sparse as sp
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for sparse=True.") from exc
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        f_np = np.zeros((total_dofs,), dtype=np.dtype(dtype))
        for fixture in fixtures:
            active = active_names is None or fixture.name in active_names
            if not active or float(fixture.stiffness) == 0.0:
                continue
            refs = np.asarray(fixture.reference_dofs, dtype=np.int32)
            if refs.size and (refs.min() < 0 or refs.max() >= total_dofs):
                raise ValueError("reference_dofs contain an index outside total_dofs.")
            d = np.asarray(fixture.direction, dtype=np.dtype(dtype))
            kk = float(fixture.stiffness) * np.outer(d, d)
            ff = float(fixture.stiffness) * float(fixture.target_displacement) * d
            rr, cc = np.meshgrid(refs, refs, indexing="ij")
            rows.extend(rr.reshape(-1).tolist())
            cols.extend(cc.reshape(-1).tolist())
            data.extend(kk.reshape(-1).tolist())
            f_np[refs] += ff
        return sp.coo_matrix((data, (rows, cols)), shape=(total_dofs, total_dofs)).tocsr(), jnp.asarray(f_np)
    k = jnp.zeros((total_dofs, total_dofs), dtype=dtype)
    f = jnp.zeros((total_dofs,), dtype=dtype)
    for fixture in fixtures:
        active = active_names is None or fixture.name in active_names
        kk, ff = fixture.preload_stiffness_force(total_dofs, active=active)
        k = k + kk
        f = f + ff
    return k, f


def _free_dofs(n_dofs: int, fixed_dofs: Array | None) -> np.ndarray:
    if fixed_dofs is None:
        return np.arange(n_dofs, dtype=np.int32)
    fixed = np.asarray(fixed_dofs, dtype=np.int32).reshape(-1)
    if fixed.size and (fixed.min() < 0 or fixed.max() >= n_dofs):
        raise ValueError("fixed_dofs contains an index outside the unknown range.")
    if np.unique(fixed).size != fixed.size:
        raise ValueError("fixed_dofs must not contain duplicates.")
    mask = np.ones((n_dofs,), dtype=bool)
    mask[fixed] = False
    return np.flatnonzero(mask).astype(np.int32)


def solve_linear_constraint_kkt(
    stiffness,
    force: Array,
    constraints: LinearConstraintSystem,
    *,
    fixed_dofs: Array | None = None,
    fixed_values: Array | None = None,
    solver: str = "dense",
) -> Array:
    """Solve `K u + C.T lambda = f`, `C u = rhs`, with optional fixed DOFs."""
    force = jnp.asarray(force)
    stiffness_shape = _matrix_shape(stiffness)
    n_dofs = int(stiffness_shape[0])
    if stiffness_shape != (n_dofs, n_dofs):
        raise ValueError("stiffness must be square.")
    if force.shape != (n_dofs,):
        raise ValueError("force must have shape (n_dofs,).")
    if constraints.n_dofs != n_dofs:
        raise ValueError("constraint column count must match stiffness size.")

    free_np = _free_dofs(n_dofs, fixed_dofs)
    free = jnp.asarray(free_np, dtype=jnp.int32)
    fixed_np = np.asarray([], dtype=np.int32) if fixed_dofs is None else np.asarray(fixed_dofs, dtype=np.int32).reshape(-1)
    fixed = jnp.asarray(fixed_np, dtype=jnp.int32)
    if fixed_values is None:
        fixed_vals = jnp.zeros((fixed.size,), dtype=force.dtype)
    else:
        fixed_vals = jnp.asarray(fixed_values, dtype=force.dtype)
        if fixed_vals.shape != (fixed.size,):
            raise ValueError("fixed_values must have shape (fixed_dofs.size,).")
    if solver == "dense":
        stiffness_dense = _as_dense_array(stiffness)
        k_ff = stiffness_dense[free[:, None], free[None, :]]
        k_fc = stiffness_dense[free[:, None], fixed[None, :]] if fixed.size else jnp.zeros((free.size, 0), dtype=stiffness_dense.dtype)
        c_f = constraints.matrix[:, free]
        c_c = constraints.matrix[:, fixed] if fixed.size else jnp.zeros((constraints.n_constraints, 0), dtype=constraints.matrix.dtype)
        rhs = jnp.concatenate([force[free] - k_fc @ fixed_vals, constraints.rhs - c_c @ fixed_vals])
        lhs = jnp.block(
            [
                [k_ff, c_f.T],
                [c_f, jnp.zeros((constraints.n_constraints, constraints.n_constraints), dtype=stiffness_dense.dtype)],
            ]
        )
        sol = jnp.linalg.solve(lhs, rhs)
    elif solver == "spsolve":
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for solver='spsolve'.") from exc
        if hasattr(stiffness, "to_csr"):
            k_csr = stiffness.to_csr()
        elif sp.issparse(stiffness):
            k_csr = stiffness.tocsr()
        else:
            k_csr = sp.csr_matrix(np.asarray(stiffness))
        k_ff = k_csr[free_np, :][:, free_np].tocsr()
        k_fc = k_csr[free_np, :][:, fixed_np].tocsr() if fixed_np.size else sp.csr_matrix((free_np.size, 0), dtype=k_ff.dtype)
        c_f = sp.csr_matrix(np.asarray(constraints.matrix[:, free]))
        c_c = np.asarray(constraints.matrix[:, fixed]) if fixed_np.size else np.zeros((constraints.n_constraints, 0), dtype=np.asarray(constraints.matrix).dtype)
        rhs = jnp.concatenate([force[free] - jnp.asarray(k_fc @ np.asarray(fixed_vals)), constraints.rhs - jnp.asarray(c_c @ np.asarray(fixed_vals))])
        zero = sp.csr_matrix((constraints.n_constraints, constraints.n_constraints), dtype=k_ff.dtype)
        lhs = sp.bmat([[k_ff, c_f.T], [c_f, zero]], format="csr")
        sol_np = spla.spsolve(lhs, np.asarray(rhs))
        sol = jnp.asarray(sol_np, dtype=jnp.result_type(force, constraints.matrix))
    else:
        raise ValueError("solver must be 'dense' or 'spsolve'.")
    u = jnp.zeros((n_dofs,), dtype=sol.dtype).at[free].set(sol[: free.size])
    if fixed.size:
        u = u.at[fixed].set(fixed_vals.astype(sol.dtype))
    return u


def make_craig_bampton_basis(
    stiffness,
    mass,
    retained_dofs: Array,
    n_modes: int,
    *,
    constraint_solver: str | Callable[[object, Array], Array] = "dense",
    modal_solver: str | Callable[[object, object, int], tuple[Array, Array]] = "dense",
    modal_linear_solver: str | Callable[[object, Array], Array] = "dense",
    modal_oversample: int = 2,
    modal_tol: float = 1e-8,
    modal_maxiter: int = 30,
    cg_tol: float = 1e-10,
    cg_maxiter: int | None = None,
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
        constraint_modes = solve_constraint_modes(
            k_ii,
            k_ir,
            solver=constraint_solver,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
    else:
        constraint_modes = jnp.zeros((internal.size, 0), dtype=_result_dtype(stiffness, mass))
    normal_modes, eigenvalues = fixed_interface_modes(
        k_ii,
        m_ii,
        n_modes,
        solver=modal_solver,
        modal_linear_solver=modal_linear_solver,
        modal_oversample=modal_oversample,
        modal_tol=modal_tol,
        modal_maxiter=modal_maxiter,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
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


@dataclass(frozen=True)
class NewmarkState:
    """Reduced-coordinate state for second-order dynamics."""

    q: Array
    qd: Array
    qdd: Array
    t: float = 0.0


@dataclass(frozen=True)
class NewmarkConfig:
    """Controls for an implicit Newmark-beta step."""

    dt: float
    beta: float = 0.25
    gamma: float = 0.5
    tol: float = 1e-8
    atol: float = 0.0
    maxiter: int = 20


@dataclass(frozen=True)
class NewmarkStepInfo:
    converged: bool
    iters: int
    residual_norm: float
    residual0: float
    rel_residual: float
    stop_reason: str


@dataclass(frozen=True)
class ActiveContactIterationRecord:
    """One outer active/contact-state update record."""

    iter: int
    active_changed: bool
    solve_info: object | None = None


@dataclass(frozen=True)
class ActiveContactSolveInfo:
    """Summary for an active/contact-state fixed-point solve."""

    converged: bool
    iters: int
    contact_state: object
    records: tuple[ActiveContactIterationRecord, ...]
    stop_reason: str


@dataclass(frozen=True)
class ActiveContactNewmarkStepInfo:
    """Summary for one implicit Newmark step with outer contact-state updates."""

    converged: bool
    iters: int
    contact_info: ActiveContactSolveInfo
    step_infos: tuple[NewmarkStepInfo, ...]
    contact_state: object
    stop_reason: str


@runtime_checkable
class ContactSearchManagerLike(Protocol):
    """Protocol for stateful contact-search managers used by ROM facades."""

    def build_contact(self, displacement: Array) -> tuple[Any, "ContactSearchManagerLike"]:
        ...


@runtime_checkable
class FrictionManagerLike(Protocol):
    """Protocol for explicit friction-history managers used by ROM facades."""

    def snapshot(self, contact: Any, u: Array) -> Any:
        ...

    def advance(self, contact: Any, u: Array) -> "FrictionManagerLike":
        ...


@dataclass
class ReducedContactDynamics:
    """Convenience facade for CB-reduced dynamics with explicit contact managers."""

    cb: CraigBamptonBasis
    stiffness: Array
    mass: Array
    damping: Array | None
    search_manager: ContactSearchManagerLike
    friction_manager: FrictionManagerLike | None = None

    def __post_init__(self):
        if not isinstance(self.search_manager, ContactSearchManagerLike):
            raise TypeError("search_manager must implement build_contact(displacement).")
        if self.friction_manager is not None and not isinstance(self.friction_manager, FrictionManagerLike):
            raise TypeError("friction_manager must implement snapshot(contact, u) and advance(contact, u).")
        stiffness = _as_dense_array(self.stiffness)
        mass = _as_dense_array(self.mass)
        if stiffness.shape != (self.cb.n_full, self.cb.n_full):
            raise ValueError("stiffness must have shape (cb.n_full, cb.n_full).")
        if mass.shape != stiffness.shape:
            raise ValueError("mass must have the same shape as stiffness.")
        damping = None if self.damping is None else _as_dense_array(self.damping)
        if damping is not None and damping.shape != stiffness.shape:
            raise ValueError("damping must have the same shape as stiffness.")
        self.stiffness = stiffness
        self.mass = mass
        self.damping = damping
        self.reduced_mass = self.cb.project_matrix(mass)
        self.reduced_damping = None if damping is None else self.cb.project_matrix(damping)

    def expand(self, q: Array) -> Array:
        return self.cb.expand(q)

    def project_force(self, full_force: Array) -> Array:
        return self.cb.project_vector(full_force)

    def build_contact(self, u_full: Array):
        contact, next_manager = self.search_manager.build_contact(u_full)
        self.search_manager = next_manager
        return contact

    def build_snapshot(self, q: Array):
        u_full = self.expand(q)
        contact = self.build_contact(u_full)
        if self.friction_manager is None:
            if hasattr(contact, "snapshot"):
                return contact.snapshot(u_full)
            return contact
        return self.friction_manager.snapshot(contact, u_full)

    def internal_force_from_snapshot(self, snapshot) -> Callable[[Array], Array]:
        contact_residual = snapshot.residual()

        def full_residual(u: Array) -> Array:
            return self.stiffness @ u + contact_residual(u)

        return self.cb.reduced_residual(full_residual)

    def advance_friction(self, snapshot, q: Array) -> None:
        if self.friction_manager is None:
            return
        contact = snapshot.contact if hasattr(snapshot, "contact") else snapshot
        self.friction_manager = self.friction_manager.advance(contact, self.expand(q))

    def active_newmark_step(
        self,
        external_force: Array,
        state: NewmarkState,
        config: NewmarkConfig,
        *,
        force_is_reduced: bool = False,
        advance_friction: bool = True,
        max_active_updates: int = 8,
        q_initial: Array | None = None,
    ) -> tuple[NewmarkState, ActiveContactNewmarkStepInfo]:
        force = jnp.asarray(external_force)
        reduced_force = force if force_is_reduced else self.project_force(force)
        next_state, info = active_contact_newmark_step(
            self.reduced_mass,
            self.reduced_damping,
            self.internal_force_from_snapshot,
            reduced_force,
            state,
            config,
            initial_contact_state=self.build_snapshot(state.q),
            update_contact_state=self.build_snapshot,
            max_active_updates=max_active_updates,
            q_initial=q_initial,
        )
        if advance_friction and info.converged:
            self.advance_friction(info.contact_state, next_state.q)
        return next_state, info


def newmark_kinematics(q_next: Array, state: NewmarkState, config: NewmarkConfig) -> tuple[Array, Array]:
    """Return Newmark velocity and acceleration implied by q_next."""
    dt = float(config.dt)
    beta = float(config.beta)
    gamma = float(config.gamma)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    q = jnp.asarray(state.q)
    qd = jnp.asarray(state.qd)
    qdd = jnp.asarray(state.qdd)
    q_pred = q + dt * qd + dt**2 * (0.5 - beta) * qdd
    qdd_next = (jnp.asarray(q_next) - q_pred) / (beta * dt**2)
    qd_next = qd + dt * ((1.0 - gamma) * qdd + gamma * qdd_next)
    return qd_next, qdd_next


def make_newmark_effective_residual(
    mass: Array,
    damping: Array | None,
    internal_force: Callable[[Array], Array],
    external_force: Array,
    state: NewmarkState,
    config: NewmarkConfig,
) -> Callable[[Array], Array]:
    """Build `G(q_next) = M a_next + C v_next + R(q_next) - F_next`."""
    mass = jnp.asarray(mass)
    damping_arr = None if damping is None else jnp.asarray(damping)
    force = jnp.asarray(external_force)

    def _residual(q_next: Array) -> Array:
        qd_next, qdd_next = newmark_kinematics(q_next, state, config)
        residual = mass @ qdd_next + internal_force(q_next) - force
        if damping_arr is not None:
            residual = residual + damping_arr @ qd_next
        return residual

    return _residual


def newmark_step(
    mass: Array,
    damping: Array | None,
    internal_force: Callable[[Array], Array],
    external_force: Array,
    state: NewmarkState,
    config: NewmarkConfig,
    *,
    q_initial: Array | None = None,
) -> tuple[NewmarkState, NewmarkStepInfo]:
    """Solve one implicit reduced Newmark step with dense Newton iterations."""
    q = jnp.asarray(state.q)
    dt = float(config.dt)
    beta = float(config.beta)
    q_pred = q + dt * state.qd + dt**2 * (0.5 - beta) * state.qdd
    q_next = jnp.asarray(q_initial) if q_initial is not None else q_pred
    residual_fn = make_newmark_effective_residual(mass, damping, internal_force, external_force, state, config)
    jacobian_fn = jax.jacrev(residual_fn)

    residual = residual_fn(q_next)
    residual0 = float(jax.block_until_ready(jnp.linalg.norm(residual, ord=2)))
    crit = max(float(config.atol), float(config.tol) * residual0)
    if residual0 <= crit:
        qd_next, qdd_next = newmark_kinematics(q_next, state, config)
        return (
            NewmarkState(q=q_next, qd=qd_next, qdd=qdd_next, t=float(state.t) + dt),
            NewmarkStepInfo(True, 0, residual0, residual0, 0.0, "converged"),
        )

    residual_norm = residual0
    converged = False
    stop_reason = "maxiter"
    iters = 0
    for k in range(int(config.maxiter)):
        jacobian = jacobian_fn(q_next)
        delta = jnp.linalg.solve(jacobian, -residual)
        q_next = q_next + delta
        residual = residual_fn(q_next)
        residual_norm = float(jax.block_until_ready(jnp.linalg.norm(residual, ord=2)))
        iters = k + 1
        if not np.isfinite(residual_norm):
            stop_reason = "nan"
            break
        if residual_norm <= crit:
            converged = True
            stop_reason = "converged"
            break

    qd_next, qdd_next = newmark_kinematics(q_next, state, config)
    rel = residual_norm / residual0 if residual0 > 0.0 else 0.0
    return (
        NewmarkState(q=q_next, qd=qd_next, qdd=qdd_next, t=float(state.t) + dt),
        NewmarkStepInfo(converged, iters, residual_norm, residual0, rel, stop_reason),
    )


def integrate_newmark(
    mass: Array,
    damping: Array | None,
    internal_force: Callable[[Array], Array],
    external_force: Callable[[float], Array],
    initial_state: NewmarkState,
    config: NewmarkConfig,
    n_steps: int,
) -> tuple[NewmarkState, list[NewmarkState], list[NewmarkStepInfo]]:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    state = initial_state
    states: list[NewmarkState] = [state]
    infos: list[NewmarkStepInfo] = []
    for _ in range(int(n_steps)):
        f_next = external_force(float(state.t) + float(config.dt))
        state, info = newmark_step(mass, damping, internal_force, f_next, state, config)
        states.append(state)
        infos.append(info)
        if not info.converged:
            break
    return state, states, infos


def _default_contact_state_changed(old_state, new_state) -> bool:
    if hasattr(new_state, "changed"):
        return bool(jax.block_until_ready(new_state.changed(old_state)))
    if hasattr(old_state, "active") and hasattr(new_state, "active"):
        changed = jnp.any(jnp.asarray(old_state.active, dtype=bool) != jnp.asarray(new_state.active, dtype=bool))
        return bool(jax.block_until_ready(changed))
    return old_state != new_state


def active_contact_fixed_point_solve(
    initial_solution: Array,
    initial_contact_state,
    residual_from_contact_state: Callable[[object], Callable[[Array], Array]],
    solve_fn: Callable[[Callable[[Array], Array], Array], tuple[Array, object]],
    update_contact_state: Callable[[Array], object],
    *,
    state_changed: Callable[[object, object], bool] | None = None,
    max_active_updates: int = 8,
) -> tuple[Array, ActiveContactSolveInfo]:
    """Solve with an outer loop that freezes and updates contact/active state."""
    if max_active_updates <= 0:
        raise ValueError("max_active_updates must be positive.")
    changed_fn = state_changed or _default_contact_state_changed
    solution = jnp.asarray(initial_solution)
    contact_state = initial_contact_state
    records: list[ActiveContactIterationRecord] = []
    for outer_iter in range(int(max_active_updates)):
        residual_fn = residual_from_contact_state(contact_state)
        solution, solve_info = solve_fn(residual_fn, solution)
        new_contact_state = update_contact_state(solution)
        changed = bool(changed_fn(contact_state, new_contact_state))
        records.append(ActiveContactIterationRecord(iter=outer_iter, active_changed=changed, solve_info=solve_info))
        contact_state = new_contact_state
        if not changed:
            return solution, ActiveContactSolveInfo(
                converged=True,
                iters=outer_iter + 1,
                contact_state=contact_state,
                records=tuple(records),
                stop_reason="active_converged",
            )
    return solution, ActiveContactSolveInfo(
        converged=False,
        iters=int(max_active_updates),
        contact_state=contact_state,
        records=tuple(records),
        stop_reason="max_active_updates",
    )


def active_contact_newmark_step(
    mass: Array,
    damping: Array | None,
    internal_force_from_contact_state: Callable[[object], Callable[[Array], Array]],
    external_force: Array,
    state: NewmarkState,
    config: NewmarkConfig,
    initial_contact_state,
    update_contact_state: Callable[[Array], object],
    *,
    state_changed: Callable[[object, object], bool] | None = None,
    max_active_updates: int = 8,
    q_initial: Array | None = None,
) -> tuple[NewmarkState, ActiveContactNewmarkStepInfo]:
    """Solve one implicit Newmark step with an outer active/contact update loop."""
    q = jnp.asarray(state.q)
    dt = float(config.dt)
    beta = float(config.beta)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    q_pred = q + dt * state.qd + dt**2 * (0.5 - beta) * state.qdd
    initial_solution = jnp.asarray(q_initial) if q_initial is not None else q_pred
    step_infos: list[NewmarkStepInfo] = []

    def solve_fn(internal_force: Callable[[Array], Array], q0: Array) -> tuple[Array, NewmarkStepInfo]:
        next_state, step_info = newmark_step(
            mass,
            damping,
            internal_force,
            external_force,
            state,
            config,
            q_initial=q0,
        )
        step_infos.append(step_info)
        return next_state.q, step_info

    q_next, contact_info = active_contact_fixed_point_solve(
        initial_solution,
        initial_contact_state,
        internal_force_from_contact_state,
        solve_fn,
        update_contact_state,
        state_changed=state_changed,
        max_active_updates=max_active_updates,
    )
    qd_next, qdd_next = newmark_kinematics(q_next, state, config)
    next_state = NewmarkState(q=q_next, qd=qd_next, qdd=qdd_next, t=float(state.t) + dt)
    inner_converged = all(info.converged for info in step_infos)
    converged = bool(contact_info.converged and inner_converged)
    if not contact_info.converged:
        stop_reason = contact_info.stop_reason
    elif not inner_converged:
        stop_reason = "inner_newmark_not_converged"
    else:
        stop_reason = "converged"
    return next_state, ActiveContactNewmarkStepInfo(
        converged=converged,
        iters=contact_info.iters,
        contact_info=contact_info,
        step_infos=tuple(step_infos),
        contact_state=contact_info.contact_state,
        stop_reason=stop_reason,
    )


__all__ = [
    "ActiveContactIterationRecord",
    "ActiveContactNewmarkStepInfo",
    "ActiveContactSolveInfo",
    "ContactSearchManagerLike",
    "CraigBamptonBasis",
    "FrictionManagerLike",
    "LinearConstraintSystem",
    "NewmarkConfig",
    "NewmarkState",
    "NewmarkStepInfo",
    "RBE3Patch",
    "ReducedContactDynamics",
    "ReducedLinearConstraintSystem",
    "ReferencePointFixture",
    "active_contact_fixed_point_solve",
    "active_contact_newmark_step",
    "assemble_reference_fixture_preload",
    "complement_dofs",
    "fixed_interface_modes",
    "integrate_newmark",
    "linear_constraint_system_from_reference_fixtures",
    "make_craig_bampton_basis",
    "make_newmark_effective_residual",
    "newmark_kinematics",
    "newmark_step",
    "reduced_jacobian_from_full",
    "reduced_residual_from_full",
    "solve_linear_constraint_kkt",
    "solve_constraint_modes",
]

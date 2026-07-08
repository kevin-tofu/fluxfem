from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import fluxfem as ff


def csr(matrix) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "to_csr"):
        return matrix.to_csr()
    if hasattr(matrix, "toarray"):
        return sp.csr_matrix(matrix.toarray())
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def constrained_free_matrices(K, M, C, fixed):
    free = np.asarray(ff.free_dofs(K.shape[0], np.asarray(fixed, dtype=int)), dtype=int)
    K_ff = K[free, :][:, free].toarray() if sp.issparse(K) else np.asarray(K)[np.ix_(free, free)]
    M_ff = M[free, :][:, free].toarray() if sp.issparse(M) else np.asarray(M)[np.ix_(free, free)]
    C_f = C[:, free].toarray() if sp.issparse(C) else np.asarray(C)[:, free]
    Z = la.null_space(C_f)
    return free, Z, Z.T @ K_ff @ Z, Z.T @ M_ff @ Z


def constrained_omegas(K, M, C, fixed, n_modes: int) -> np.ndarray:
    _free, Z, k_c, m_c = constrained_free_matrices(K, M, C, fixed)
    assert Z.shape[1] >= n_modes
    w2 = la.eigh(k_c, m_c, eigvals_only=True)
    w2 = np.asarray(w2, dtype=float)
    w2 = w2[w2 > 1.0e-8]
    return np.sqrt(w2[:n_modes])


def assert_static_cb_projection_matches_full(K, F, C, fixed, retained_full):
    full = np.asarray(
        ff.LinearConstraintSystem(C.toarray()).solve(
            K,
            F,
            fixed_dofs=fixed,
            solver="spsolve",
        ),
        dtype=float,
    )

    free = np.asarray(ff.free_dofs(K.shape[0], fixed), dtype=int)
    retained = np.flatnonzero(np.isin(free, np.asarray(retained_full, dtype=int))).astype(np.int32)
    k_free = K[free, :][:, free]
    f_free = np.asarray(F, dtype=float)[free]
    c_free = C[:, free]

    cb = ff.make_craig_bampton_basis(
        k_free,
        sp.eye(k_free.shape[0], format="csr"),
        retained_dofs=retained,
        n_modes=k_free.shape[0] - retained.size,
        backend="scipy",
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    reduced_constraints = ff.LinearConstraintSystem(c_free.toarray()).project(cb)
    q = np.asarray(
        reduced_constraints.solve(
            cb.project_matrix(k_free),
            cb.project_vector(f_free),
            solver="spsolve",
        ),
        dtype=float,
    )
    rom_free = np.asarray(reduced_constraints.expand(q), dtype=float)
    rom = np.zeros_like(full)
    rom[free] = rom_free

    np.testing.assert_allclose(rom, full, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(C @ rom, np.zeros((C.shape[0],), dtype=float), atol=1.0e-9)


def projected_cb_system(K, M, C, fixed, retained_structural, n_structural: int, n_extra: int):
    fixed_arr = np.asarray(fixed, dtype=int)
    structural_fixed = fixed_arr[fixed_arr < n_structural]
    structural_free = np.asarray(ff.free_dofs(n_structural, structural_fixed), dtype=int)
    retained = np.flatnonzero(np.isin(structural_free, np.asarray(retained_structural, dtype=int))).astype(np.int32)
    k_struct = K[:n_structural, :n_structural].tocsr()
    m_struct = M[:n_structural, :n_structural].tocsr()
    k_free = k_struct[structural_free, :][:, structural_free]
    m_free = m_struct[structural_free, :][:, structural_free]
    cb = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained_dofs=retained,
        n_modes=k_free.shape[0] - retained.size,
        backend="scipy",
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    K_rom = sp.block_diag(
        (sp.csr_matrix(cb.project_matrix(k_free)), K[n_structural : n_structural + n_extra, n_structural : n_structural + n_extra]),
        format="csr",
    )
    M_rom = sp.block_diag((sp.csr_matrix(cb.project_matrix(m_free)), sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")
    C_rom = sp.csr_matrix(
        np.hstack(
            [
                np.asarray(C[:, structural_free] @ cb.basis),
                C[:, n_structural : n_structural + n_extra].toarray(),
            ]
        )
    )
    return structural_free, cb, K_rom, M_rom, C_rom


def assert_static_cb_projection_with_extra_matches_full(K, F, C, fixed, retained_structural, n_structural: int, n_extra: int):
    full = np.asarray(
        ff.LinearConstraintSystem(C.toarray()).solve(
            K,
            F,
            fixed_dofs=fixed,
            solver="spsolve",
        ),
        dtype=float,
    )

    fixed_arr = np.asarray(fixed, dtype=int)
    structural_fixed = fixed_arr[fixed_arr < n_structural]
    structural_free = np.asarray(ff.free_dofs(n_structural, structural_fixed), dtype=int)
    retained = np.flatnonzero(np.isin(structural_free, np.asarray(retained_structural, dtype=int))).astype(np.int32)
    k_struct = K[:n_structural, :n_structural].tocsr()
    k_free = k_struct[structural_free, :][:, structural_free]

    cb = ff.make_craig_bampton_basis(
        k_free,
        sp.eye(k_free.shape[0], format="csr"),
        retained_dofs=retained,
        n_modes=k_free.shape[0] - retained.size,
        backend="scipy",
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    f_reduced = np.concatenate(
        [
            np.asarray(cb.project_vector(np.asarray(F[:n_structural], dtype=float)[structural_free]), dtype=float),
            np.asarray(F[n_structural : n_structural + n_extra], dtype=float),
        ]
    )
    k_reduced = cb.project_matrix(k_free)
    k_aug = sp.block_diag(
        (sp.csr_matrix(k_reduced), K[n_structural : n_structural + n_extra, n_structural : n_structural + n_extra]),
        format="csr",
    )
    c_reduced = np.hstack(
        [
            np.asarray(C[:, structural_free] @ cb.basis),
            C[:, n_structural : n_structural + n_extra].toarray(),
        ]
    )

    q = np.asarray(
        ff.LinearConstraintSystem(c_reduced).solve(
            k_aug,
            f_reduced,
            solver="spsolve",
        ),
        dtype=float,
    )
    rom = np.zeros_like(full)
    rom[structural_free] = np.asarray(cb.expand(q[: cb.n_reduced]), dtype=float)
    rom[n_structural : n_structural + n_extra] = q[cb.n_reduced :]

    np.testing.assert_allclose(rom, full, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(C @ rom, np.zeros((C.shape[0],), dtype=float), atol=1.0e-9)

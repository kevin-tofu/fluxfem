from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import fluxfem as ff


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / "solid_beam_rbe3_coupling.py"
    spec = importlib.util.spec_from_file_location("solid_beam_rbe3_coupling", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _csr(matrix):
    if sp.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "to_csr"):
        return matrix.to_csr()
    if hasattr(matrix, "toarray"):
        return sp.csr_matrix(matrix.toarray())
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def _assert_static_cb_projection_with_extra_matches_full(K, F, C, fixed, retained_structural, n_structural: int, n_extra: int):
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
    k_aug = sp.block_diag((sp.csr_matrix(k_reduced), K[n_structural : n_structural + n_extra, n_structural : n_structural + n_extra]), format="csr")
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


def _constrained_omegas(K, M, C, fixed, n_modes: int) -> np.ndarray:
    free = np.asarray(ff.free_dofs(K.shape[0], np.asarray(fixed, dtype=int)), dtype=int)
    K_ff = K[free, :][:, free].toarray() if sp.issparse(K) else np.asarray(K)[np.ix_(free, free)]
    M_ff = M[free, :][:, free].toarray() if sp.issparse(M) else np.asarray(M)[np.ix_(free, free)]
    C_f = C[:, free].toarray() if sp.issparse(C) else np.asarray(C)[:, free]
    Z = la.null_space(C_f)
    assert Z.shape[1] >= n_modes
    w2 = la.eigh(Z.T @ K_ff @ Z, Z.T @ M_ff @ Z, eigvals_only=True)
    w2 = np.asarray(w2, dtype=float)
    w2 = w2[w2 > 1.0e-8]
    return np.sqrt(w2[:n_modes])


def test_solid_beam_rbe3_coupling_ties_remote_to_beam_root():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_beam_coupling(solid_nx=2, solid_ny=1, solid_nz=1, beam_elems=2, tip_load_z=-5.0)
    system = model["system"]
    fixed = model["fixed_dofs"]

    u_all = np.asarray(
        system.solve(
            format="csr",
            dirichlet_dofs=fixed,
            dirichlet_vals=np.zeros((fixed.size,), dtype=float),
        ),
        dtype=float,
    )

    remote_q = u_all[model["remote_dofs"]]
    beam_root_q = u_all[model["beam_root_dofs"]]
    beam_tip_q = u_all[model["beam_tip_dofs"]]

    np.testing.assert_allclose(remote_q, beam_root_q, rtol=1.0e-9, atol=1.0e-9)
    assert float(beam_tip_q[2]) < float(beam_root_q[2])

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False
    assert float(np.linalg.norm(residual[free])) < 1.0e-6


def test_solid_beam_rbe3_coupling_projects_through_cb_rom():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_beam_coupling(solid_nx=2, solid_ny=1, solid_nz=1, beam_elems=2, tip_load_z=-5.0)
    K_lifted, F_lifted = model["system"].assemble(format="csr")

    n_structural = model["solid_space"].n_dofs + model["beam_coords"].shape[0] * 6
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    F = np.asarray(F_lifted[:n_primary], dtype=float)
    C = K_lifted[n_primary:, :n_primary].tocsr()
    retained_structural = np.unique(np.concatenate([model["face_dofs"], model["beam_root_dofs"], model["beam_tip_dofs"]]))

    _assert_static_cb_projection_with_extra_matches_full(
        K,
        F,
        C,
        np.asarray(model["fixed_dofs"], dtype=int),
        retained_structural,
        n_structural,
        n_extra,
    )


def test_solid_beam_rbe3_coupling_cb_matches_constrained_frequencies():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_beam_coupling(solid_nx=2, solid_ny=1, solid_nz=1, beam_elems=2, tip_load_z=0.0)
    K_lifted, _F_lifted = model["system"].assemble(format="csr")

    n_solid = model["solid_space"].n_dofs
    n_beam = model["beam_coords"].shape[0] * 6
    n_structural = n_solid + n_beam
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    C = K_lifted[n_primary:, :n_primary].tocsr()

    M_solid = _csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    beam_section = ff.BeamSection(E=2.0e5, G=7.7e4, A=1.0e-2, Iy=1.5e-5, Iz=1.5e-5, J=3.0e-5, rho=1.0)
    M_beam = ff.assemble_beam_mass(model["beam_coords"], model["beam_conn"], beam_section, format="csr")
    M = sp.block_diag((M_solid, M_beam, sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")

    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    structural_fixed = fixed[fixed < n_structural]
    structural_free = np.asarray(ff.free_dofs(n_structural, structural_fixed), dtype=int)
    retained_structural = np.unique(np.concatenate([model["face_dofs"], model["beam_root_dofs"], model["beam_tip_dofs"]]))
    retained = np.flatnonzero(np.isin(structural_free, retained_structural)).astype(np.int32)
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
    K_rom = sp.block_diag((sp.csr_matrix(cb.project_matrix(k_free)), K[n_structural:n_primary, n_structural:n_primary]), format="csr")
    M_rom = sp.block_diag((sp.csr_matrix(cb.project_matrix(m_free)), sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")
    C_rom = sp.csr_matrix(np.hstack([np.asarray(C[:, structural_free] @ cb.basis), C[:, n_structural:n_primary].toarray()]))

    full = _constrained_omegas(K, M, C, fixed, n_modes=6)
    rom = _constrained_omegas(K_rom, M_rom, C_rom, np.array([], dtype=int), n_modes=6)

    np.testing.assert_allclose(rom, full, rtol=1.0e-6, atol=1.0e-5)

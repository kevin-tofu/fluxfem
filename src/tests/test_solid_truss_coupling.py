from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import fluxfem as ff
from tests.cb_test_utils import (
    assert_static_cb_projection_with_extra_matches_full,
    constrained_free_matrices,
    constrained_omegas,
    csr,
    projected_cb_system,
)


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / "solid_truss_rbe3_coupling.py"
    spec = importlib.util.spec_from_file_location("solid_truss_rbe3_coupling", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_solid_truss_rbe3_coupling_ties_remote_translation_to_truss_root():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_truss_coupling(solid_nx=2, solid_ny=1, solid_nz=1, truss_elems=2, tip_load_x=5.0)
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
    truss_root_q = u_all[model["truss_root_dofs"]]
    truss_tip_q = u_all[model["truss_tip_dofs"]]

    np.testing.assert_allclose(remote_q[:3], truss_root_q, rtol=1.0e-9, atol=1.0e-9)
    assert float(truss_tip_q[0]) > float(truss_root_q[0])

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False
    assert float(np.linalg.norm(residual[free])) < 1.0e-6


def test_solid_truss_rbe3_coupling_projects_through_cb_rom():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_truss_coupling(solid_nx=2, solid_ny=1, solid_nz=1, truss_elems=2, tip_load_x=5.0)
    K_lifted, F_lifted = model["system"].assemble(format="csr")

    n_structural = model["solid_space"].n_dofs + model["truss_coords"].shape[0] * 3
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    F = np.asarray(F_lifted[:n_primary], dtype=float)
    C = K_lifted[n_primary:, :n_primary].tocsr()
    retained_structural = np.unique(np.concatenate([model["face_dofs"], model["truss_root_dofs"], model["truss_tip_dofs"]]))

    assert_static_cb_projection_with_extra_matches_full(
        K,
        F,
        C,
        np.asarray(model["fixed_dofs"], dtype=int),
        retained_structural,
        n_structural,
        n_extra,
    )


def test_solid_truss_rbe3_coupling_cb_matches_constrained_frequencies():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_truss_coupling(solid_nx=2, solid_ny=1, solid_nz=1, truss_elems=2, tip_load_x=0.0)
    K_lifted, _F_lifted = model["system"].assemble(format="csr")

    n_solid = model["solid_space"].n_dofs
    n_truss = model["truss_coords"].shape[0] * 3
    n_structural = n_solid + n_truss
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    C = K_lifted[n_primary:, :n_primary].tocsr()

    M_solid = csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    truss_section = ff.TrussSection(E=2.0e5, A=1.0e-2, rho=1.0)
    M_truss = ff.assemble_truss_mass(model["truss_coords"], model["truss_conn"], truss_section, format="csr")
    M = sp.block_diag((M_solid, M_truss, sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")

    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    retained_structural = np.unique(np.concatenate([model["face_dofs"], model["truss_root_dofs"], model["truss_tip_dofs"]]))
    _structural_free, _cb, K_rom, M_rom, C_rom = projected_cb_system(K, M, C, fixed, retained_structural, n_structural, n_extra)

    full = constrained_omegas(K, M, C, fixed, n_modes=6)
    rom = constrained_omegas(K_rom, M_rom, C_rom, np.array([], dtype=int), n_modes=6)

    np.testing.assert_allclose(rom, full, rtol=1.0e-6, atol=1.0e-5)


def test_solid_truss_rbe3_coupling_cb_matches_constrained_newmark_history():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_truss_coupling(solid_nx=2, solid_ny=1, solid_nz=1, truss_elems=2, tip_load_x=0.0)
    K_lifted, _F_lifted = model["system"].assemble(format="csr")

    n_solid = model["solid_space"].n_dofs
    n_truss = model["truss_coords"].shape[0] * 3
    n_structural = n_solid + n_truss
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    C = K_lifted[n_primary:, :n_primary].tocsr()

    M_solid = csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    truss_section = ff.TrussSection(E=2.0e5, A=1.0e-2, rho=1.0)
    M_truss = ff.assemble_truss_mass(model["truss_coords"], model["truss_conn"], truss_section, format="csr")
    M = sp.block_diag((M_solid, M_truss, sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")

    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    free, Z, Kz, Mz = constrained_free_matrices(K, M, C, fixed)
    w2, modes = la.eigh(Kz, Mz)
    positive = np.flatnonzero(np.asarray(w2, dtype=float) > 1.0e-8)
    assert positive.size > 0
    mode = modes[:, positive[0]]
    u_free0 = Z @ mode
    u_free0 *= 1.0e-4 / np.max(np.abs(u_free0))
    q0 = la.lstsq(Z, u_free0)[0]
    period = 2.0 * np.pi / np.sqrt(float(w2[positive[0]]))

    out_full = ff.newmark_solve_linear(
        Mz,
        np.zeros_like(Kz),
        Kz,
        u0=q0,
        v0=np.zeros_like(q0),
        dt=period / 80.0,
        n_steps=24,
    )
    full_free_hist = out_full.u @ Z.T

    retained_structural = np.unique(np.concatenate([model["face_dofs"], model["truss_root_dofs"], model["truss_tip_dofs"]]))
    structural_free, cb, K_rom, M_rom, C_rom = projected_cb_system(K, M, C, fixed, retained_structural, n_structural, n_extra)
    assert np.array_equal(free[: structural_free.size], structural_free)
    Z_rom = la.null_space(C_rom.toarray())
    Kzr = Z_rom.T @ K_rom.toarray() @ Z_rom
    Mzr = Z_rom.T @ M_rom.toarray() @ Z_rom
    q_cb0 = la.lstsq(np.asarray(cb.basis, dtype=float), u_free0[: structural_free.size])[0]
    q_aug0 = np.concatenate([q_cb0, u_free0[structural_free.size : structural_free.size + n_extra]])
    ar0 = la.lstsq(Z_rom, q_aug0)[0]

    out_rom = ff.newmark_solve_linear(
        Mzr,
        np.zeros_like(Kzr),
        Kzr,
        u0=ar0,
        v0=np.zeros_like(ar0),
        dt=period / 80.0,
        n_steps=24,
    )
    q_rom_hist = out_rom.u @ Z_rom.T
    rom_free_hist = np.zeros_like(full_free_hist)
    rom_free_hist[:, : structural_free.size] = q_rom_hist[:, : cb.n_reduced] @ np.asarray(cb.basis, dtype=float).T
    rom_free_hist[:, structural_free.size : structural_free.size + n_extra] = q_rom_hist[:, cb.n_reduced :]

    tip_free = np.flatnonzero(np.isin(free, model["truss_tip_dofs"]))
    assert tip_free.size > 0
    np.testing.assert_allclose(rom_free_hist[:, tip_free], full_free_hist[:, tip_free], rtol=1.0e-7, atol=1.0e-9)
    np.testing.assert_allclose(rom_free_hist[:, : structural_free.size], full_free_hist[:, : structural_free.size], rtol=1.0e-7, atol=1.0e-9)

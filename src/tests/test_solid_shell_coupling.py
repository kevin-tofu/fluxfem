from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import fluxfem as ff


def _load_tutorial_module(name: str):
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _coincident_tie_constraint(n_solid: int, n_shell: int, solid_dofs: np.ndarray, shell_dofs: np.ndarray) -> sp.csr_matrix:
    rows = np.repeat(np.arange(solid_dofs.size, dtype=int), 2)
    cols = np.empty((2 * solid_dofs.size,), dtype=int)
    data = np.empty((2 * solid_dofs.size,), dtype=float)
    cols[0::2] = np.asarray(solid_dofs, dtype=int)
    cols[1::2] = n_solid + np.asarray(shell_dofs, dtype=int)
    data[0::2] = 1.0
    data[1::2] = -1.0
    return sp.csr_matrix((data, (rows, cols)), shape=(solid_dofs.size, n_solid + n_shell))


def _constrained_omegas(K, M, C, fixed, n_modes: int) -> np.ndarray:
    _free, Z, k_c, m_c = _constrained_free_matrices(K, M, C, fixed)
    assert Z.shape[1] >= n_modes
    w2 = la.eigh(k_c, m_c, eigvals_only=True)
    w2 = np.asarray(w2, dtype=float)
    w2 = w2[w2 > 1.0e-8]
    return np.sqrt(w2[:n_modes])


def _constrained_free_matrices(K, M, C, fixed):
    free = np.asarray(ff.free_dofs(K.shape[0], np.asarray(fixed, dtype=int)), dtype=int)
    K_ff = K[free, :][:, free].toarray() if sp.issparse(K) else np.asarray(K)[np.ix_(free, free)]
    M_ff = M[free, :][:, free].toarray() if sp.issparse(M) else np.asarray(M)[np.ix_(free, free)]
    C_f = C[:, free].toarray() if sp.issparse(C) else np.asarray(C)[:, free]
    Z = la.null_space(C_f)
    return free, Z, Z.T @ K_ff @ Z, Z.T @ M_ff @ Z


def _csr(matrix):
    if sp.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "to_csr"):
        return matrix.to_csr()
    if hasattr(matrix, "toarray"):
        return sp.csr_matrix(matrix.toarray())
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def _assert_static_cb_projection_matches_full(K, F, C, fixed, retained_full):
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
    q_fixed = np.asarray([np.flatnonzero(structural_free == dof)[0] for dof in structural_fixed if dof in structural_free], dtype=int)
    assert q_fixed.size == 0

    f_reduced = np.concatenate(
        [
            np.asarray(cb.project_vector(np.asarray(F[:n_structural], dtype=float)[structural_free]), dtype=float),
            np.asarray(F[n_structural : n_structural + n_extra], dtype=float),
        ]
    )
    k_reduced = cb.project_matrix(k_free)
    k_aug = sp.block_diag((sp.csr_matrix(k_reduced), K[n_structural : n_structural + n_extra, n_structural : n_structural + n_extra]), format="csr")
    c_struct_free = C[:, structural_free]
    c_extra = C[:, n_structural : n_structural + n_extra]
    c_reduced = np.hstack([np.asarray(c_struct_free @ cb.basis), c_extra.toarray()])

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


def test_solid_shell_translational_tie_matches_interface_displacements():
    tutorial = _load_tutorial_module("solid_shell_translational_tie")
    model = tutorial.build_solid_shell_tie(nx=2, ny=1, nz=1, pressure_z=-1.0)
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

    solid_tie_u = u_all[model["solid_tie_dofs"]].reshape(-1, 3)
    shell_tie_u = u_all[model["shell_tie_dofs"]].reshape(-1, 3)
    np.testing.assert_allclose(solid_tie_u, shell_tie_u, rtol=1.0e-9, atol=1.0e-9)

    shell_u = u_all[model["shell_offset"] : model["shell_offset"] + model["shell_n_dofs"]].reshape(-1, 6)
    assert float(np.min(shell_u[:, 2])) < 0.0

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False
    assert float(np.linalg.norm(residual[free])) < 1.0e-6


def test_solid_shell_translational_tie_projects_through_cb_rom():
    tutorial = _load_tutorial_module("solid_shell_translational_tie")
    model = tutorial.build_solid_shell_tie(nx=2, ny=1, nz=1, pressure_z=-1.0, shear_mode="mitc4")
    system = model["system"]
    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    K_lifted, F_lifted = system.assemble(format="csr")

    n_solid = model["solid_space"].n_dofs
    n_shell = model["shell_n_dofs"]
    K = K_lifted[: n_solid + n_shell, : n_solid + n_shell].tocsr()
    F = np.asarray(F_lifted[: n_solid + n_shell], dtype=float)
    C = _coincident_tie_constraint(n_solid, n_shell, model["solid_tie_dofs"], model["shell_tie_dofs"] - model["shell_offset"])

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
    retained_full = np.unique(
        np.concatenate(
            [
                np.asarray(model["solid_tie_dofs"], dtype=int),
                np.asarray(model["shell_tie_dofs"], dtype=int),
                model["shell_offset"]
                + ff.shell_node_dofs(
                    np.flatnonzero(np.isclose(model["shell_coords"][:, 0], model["shell_coords"][:, 0].max()))
                ),
            ]
        )
    )
    retained = np.flatnonzero(np.isin(free, retained_full)).astype(np.int32)
    k_free = K[free, :][:, free]
    f_free = F[free]
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


def test_solid_shell_translational_tie_cb_matches_constrained_frequencies():
    tutorial = _load_tutorial_module("solid_shell_translational_tie")
    model = tutorial.build_solid_shell_tie(nx=2, ny=1, nz=1, pressure_z=-1.0, shear_mode="mitc4")
    n_solid = model["solid_space"].n_dofs
    n_shell = model["shell_n_dofs"]

    K_lifted, _F_lifted = model["system"].assemble(format="csr")
    K = K_lifted[: n_solid + n_shell, : n_solid + n_shell].tocsr()
    C = _coincident_tie_constraint(n_solid, n_shell, model["solid_tie_dofs"], model["shell_tie_dofs"] - model["shell_offset"])
    M_solid = 1.0 * _csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02, rho=1.0, shear_mode=model["shear_mode"])
    M_shell = ff.assemble_shell_mass(model["shell_coords"], model["shell_conn"], shell_section, format="csr")
    M = sp.block_diag((M_solid, M_shell), format="csr")

    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    free = np.asarray(ff.free_dofs(K.shape[0], fixed), dtype=int)
    retained_full = np.unique(np.concatenate([np.asarray(model["solid_tie_dofs"], dtype=int), np.asarray(model["shell_tie_dofs"], dtype=int)]))
    retained = np.flatnonzero(np.isin(free, retained_full)).astype(np.int32)
    k_free = K[free, :][:, free]
    m_free = M[free, :][:, free]
    c_free = C[:, free]

    cb = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained_dofs=retained,
        n_modes=k_free.shape[0] - retained.size,
        backend="scipy",
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    full = _constrained_omegas(K, M, C, fixed, n_modes=6)
    rom = _constrained_omegas(
        sp.csr_matrix(cb.project_matrix(k_free)),
        sp.csr_matrix(cb.project_matrix(m_free)),
        sp.csr_matrix(c_free @ cb.basis),
        np.array([], dtype=int),
        n_modes=6,
    )

    np.testing.assert_allclose(rom, full, rtol=1.0e-8, atol=1.0e-7)


def test_solid_shell_translational_tie_cb_matches_constrained_newmark_history():
    tutorial = _load_tutorial_module("solid_shell_translational_tie")
    model = tutorial.build_solid_shell_tie(nx=2, ny=1, nz=1, pressure_z=0.0, shear_mode="mitc4")
    n_solid = model["solid_space"].n_dofs
    n_shell = model["shell_n_dofs"]

    K_lifted, _F_lifted = model["system"].assemble(format="csr")
    K = K_lifted[: n_solid + n_shell, : n_solid + n_shell].tocsr()
    C = _coincident_tie_constraint(n_solid, n_shell, model["solid_tie_dofs"], model["shell_tie_dofs"] - model["shell_offset"])
    M_solid = _csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02, rho=1.0, shear_mode=model["shear_mode"])
    M_shell = ff.assemble_shell_mass(model["shell_coords"], model["shell_conn"], shell_section, format="csr")
    M = sp.block_diag((M_solid, M_shell), format="csr")

    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    free, Z, Kz, Mz = _constrained_free_matrices(K, M, C, fixed)
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

    retained_full = np.unique(np.concatenate([np.asarray(model["solid_tie_dofs"], dtype=int), np.asarray(model["shell_tie_dofs"], dtype=int)]))
    retained = np.flatnonzero(np.isin(free, retained_full)).astype(np.int32)
    k_free = K[free, :][:, free]
    m_free = M[free, :][:, free]
    c_free = C[:, free]
    cb = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained_dofs=retained,
        n_modes=k_free.shape[0] - retained.size,
        backend="scipy",
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    basis = np.asarray(cb.basis, dtype=float)
    K_rom = np.asarray(cb.project_matrix(k_free), dtype=float)
    M_rom = np.asarray(cb.project_matrix(m_free), dtype=float)
    C_rom = np.asarray(c_free @ basis, dtype=float)
    Z_rom = la.null_space(C_rom)
    Kzr = Z_rom.T @ K_rom @ Z_rom
    Mzr = Z_rom.T @ M_rom @ Z_rom
    q_cb0 = la.lstsq(basis, u_free0)[0]
    ar0 = la.lstsq(Z_rom, q_cb0)[0]

    out_rom = ff.newmark_solve_linear(
        Mzr,
        np.zeros_like(Kzr),
        Kzr,
        u0=ar0,
        v0=np.zeros_like(ar0),
        dt=period / 80.0,
        n_steps=24,
    )
    rom_free_hist = out_rom.u @ Z_rom.T @ basis.T

    shell_tip_nodes = np.flatnonzero(np.isclose(model["shell_coords"][:, 0], model["shell_coords"][:, 0].max()))
    tip_uz = model["shell_offset"] + ff.shell_node_dofs(shell_tip_nodes, "uz")
    tip_free = np.flatnonzero(np.isin(free, tip_uz))
    assert tip_free.size > 0

    np.testing.assert_allclose(rom_free_hist[:, tip_free], full_free_hist[:, tip_free], rtol=1.0e-7, atol=1.0e-9)
    np.testing.assert_allclose(rom_free_hist, full_free_hist, rtol=1.0e-7, atol=1.0e-9)


def test_solid_shell_translational_tie_accepts_mitc4_shell():
    tutorial = _load_tutorial_module("solid_shell_translational_tie")
    model = tutorial.build_solid_shell_tie(nx=2, ny=1, nz=1, pressure_z=-1.0, shear_mode="mitc4")
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

    solid_tie_u = u_all[model["solid_tie_dofs"]].reshape(-1, 3)
    shell_tie_u = u_all[model["shell_tie_dofs"]].reshape(-1, 3)
    np.testing.assert_allclose(solid_tie_u, shell_tie_u, rtol=1.0e-9, atol=1.0e-9)


def test_solid_shell_rbe3_patch_coupling_ties_shell_root_to_remote():
    tutorial = _load_tutorial_module("solid_shell_rbe3_patch_coupling")
    model = tutorial.build_solid_shell_rbe3_patch_coupling(
        solid_nx=2,
        solid_ny=1,
        solid_nz=1,
        shell_nx=2,
        shell_ny=1,
        tip_load_y=-1.0,
    )
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
    shell_root_q = u_all[model["shell_root_dofs"]].reshape(-1, 6)
    shell_tip_q = u_all[model["shell_tip_dofs"]].reshape(-1, 6)
    np.testing.assert_allclose(shell_root_q, np.tile(remote_q[None, :], (shell_root_q.shape[0], 1)), rtol=1.0e-9, atol=1.0e-9)
    assert float(np.mean(shell_tip_q[:, 1])) < float(remote_q[1])

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False
    assert float(np.linalg.norm(residual[free])) < 1.0e-6


def test_solid_shell_rbe3_patch_coupling_accepts_mitc4_shell():
    tutorial = _load_tutorial_module("solid_shell_rbe3_patch_coupling")
    model = tutorial.build_solid_shell_rbe3_patch_coupling(
        solid_nx=2,
        solid_ny=1,
        solid_nz=1,
        shell_nx=2,
        shell_ny=1,
        tip_load_y=-1.0,
        shear_mode="mitc4",
    )
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
    shell_root_q = u_all[model["shell_root_dofs"]].reshape(-1, 6)
    np.testing.assert_allclose(shell_root_q, np.tile(remote_q[None, :], (shell_root_q.shape[0], 1)), rtol=1.0e-9, atol=1.0e-9)


def test_solid_shell_rbe3_patch_coupling_projects_through_cb_rom():
    tutorial = _load_tutorial_module("solid_shell_rbe3_patch_coupling")
    model = tutorial.build_solid_shell_rbe3_patch_coupling(
        solid_nx=2,
        solid_ny=1,
        solid_nz=1,
        shell_nx=2,
        shell_ny=1,
        tip_load_y=-1.0,
        shear_mode="mitc4",
    )
    K_lifted, F_lifted = model["system"].assemble(format="csr")
    n_structural = model["shell_offset"] + model["shell_n_dofs"]
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)

    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    F = np.asarray(F_lifted[:n_primary], dtype=float)
    C = K_lifted[n_primary:, :n_primary].tocsr()

    face_dofs = ff.vector_dofs_from_nodes(model["face_nodes"], dim=3)
    retained_structural = np.unique(
        np.concatenate(
            [
                face_dofs,
                model["shell_root_dofs"],
                model["shell_tip_dofs"],
            ]
        )
    )

    _assert_static_cb_projection_with_extra_matches_full(
        K,
        F,
        C,
        np.asarray(model["fixed_dofs"], dtype=int),
        retained_structural,
        n_structural,
        n_extra=n_extra,
    )


def test_solid_shell_rbe3_patch_coupling_cb_matches_constrained_frequencies():
    tutorial = _load_tutorial_module("solid_shell_rbe3_patch_coupling")
    model = tutorial.build_solid_shell_rbe3_patch_coupling(
        solid_nx=2,
        solid_ny=1,
        solid_nz=1,
        shell_nx=2,
        shell_ny=1,
        tip_load_y=-1.0,
        shear_mode="mitc4",
    )
    K_lifted, _F_lifted = model["system"].assemble(format="csr")
    n_structural = model["shell_offset"] + model["shell_n_dofs"]
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    C = K_lifted[n_primary:, :n_primary].tocsr()

    M_solid = _csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02, rho=1.0, shear_mode=model["shear_mode"])
    M_shell = ff.assemble_shell_mass(model["shell_coords"], model["shell_conn"], shell_section, format="csr")
    M = sp.block_diag((M_solid, M_shell, sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")

    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    structural_free = np.asarray(ff.free_dofs(n_structural, fixed[fixed < n_structural]), dtype=int)
    face_dofs = ff.vector_dofs_from_nodes(model["face_nodes"], dim=3)
    retained_structural = np.unique(np.concatenate([face_dofs, model["shell_root_dofs"], model["shell_tip_dofs"]]))
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
    k_reduced = cb.project_matrix(k_free)
    m_reduced = cb.project_matrix(m_free)
    c_reduced = np.hstack(
        [
            np.asarray(C[:, structural_free] @ cb.basis),
            C[:, n_structural:n_primary].toarray(),
        ]
    )
    K_rom = sp.block_diag((sp.csr_matrix(k_reduced), K[n_structural:n_primary, n_structural:n_primary]), format="csr")
    M_rom = sp.block_diag((sp.csr_matrix(m_reduced), sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")

    full = _constrained_omegas(K, M, C, fixed, n_modes=6)
    rom = _constrained_omegas(K_rom, M_rom, sp.csr_matrix(c_reduced), np.array([], dtype=int), n_modes=6)

    np.testing.assert_allclose(rom, full, rtol=1.0e-6, atol=1.0e-5)


def test_solid_shell_nonmatching_tie_interpolates_interface_displacements():
    tutorial = _load_tutorial_module("solid_shell_nonmatching_tie")
    model = tutorial.build_solid_shell_nonmatching_tie(
        solid_nx=2,
        solid_ny=1,
        solid_nz=1,
        shell_nx=4,
        shell_ny=2,
        pressure_z=-1.0,
        shear_mode="mitc4",
    )
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

    solid_u = u_all[: model["solid_space"].n_dofs]
    shell_u = u_all[model["shell_offset"] : model["shell_offset"] + model["shell_n_dofs"]]
    tie_residual = model["constraint_matrix"] @ np.concatenate([solid_u, shell_u])
    np.testing.assert_allclose(tie_residual, np.zeros_like(tie_residual), rtol=1.0e-9, atol=1.0e-9)
    assert model["matched_solid_nodes"].shape[0] == model["shell_coords"].shape[0]
    assert float(np.min(shell_u.reshape(-1, 6)[:, 2])) < 0.0


def test_solid_shell_nonmatching_tie_projects_through_cb_rom():
    tutorial = _load_tutorial_module("solid_shell_nonmatching_tie")
    model = tutorial.build_solid_shell_nonmatching_tie(
        solid_nx=2,
        solid_ny=1,
        solid_nz=1,
        shell_nx=4,
        shell_ny=2,
        pressure_z=-1.0,
        shear_mode="mitc4",
    )
    K_lifted, F_lifted = model["system"].assemble(format="csr")
    n_solid = model["solid_space"].n_dofs
    n_shell = model["shell_n_dofs"]
    K = K_lifted[: n_solid + n_shell, : n_solid + n_shell].tocsr()
    F = np.asarray(F_lifted[: n_solid + n_shell], dtype=float)
    C = model["constraint_matrix"].tocsr()

    matched_solid_dofs = np.asarray([3 * int(node) + comp for nodes in model["matched_solid_nodes"] for node in nodes for comp in range(3)], dtype=int)
    shell_nodes = np.arange(model["shell_coords"].shape[0], dtype=int)
    retained_full = np.unique(
        np.concatenate(
            [
                matched_solid_dofs,
                model["shell_offset"] + ff.shell_node_dofs(shell_nodes, "uxuyuz"),
                model["shell_offset"]
                + ff.shell_node_dofs(
                    np.flatnonzero(np.isclose(model["shell_coords"][:, 0], model["shell_coords"][:, 0].max()))
                ),
            ]
        )
    )

    _assert_static_cb_projection_matches_full(K, F, C, np.asarray(model["fixed_dofs"], dtype=int), retained_full)


def test_solid_shell_nonmatching_tie_cb_matches_constrained_frequencies():
    tutorial = _load_tutorial_module("solid_shell_nonmatching_tie")
    model = tutorial.build_solid_shell_nonmatching_tie(
        solid_nx=2,
        solid_ny=1,
        solid_nz=1,
        shell_nx=4,
        shell_ny=2,
        pressure_z=-1.0,
        shear_mode="mitc4",
    )
    K_lifted, _F_lifted = model["system"].assemble(format="csr")
    n_solid = model["solid_space"].n_dofs
    n_shell = model["shell_n_dofs"]
    K = K_lifted[: n_solid + n_shell, : n_solid + n_shell].tocsr()
    C = model["constraint_matrix"].tocsr()
    M_solid = _csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02, rho=1.0, shear_mode=model["shear_mode"])
    M_shell = ff.assemble_shell_mass(model["shell_coords"], model["shell_conn"], shell_section, format="csr")
    M = sp.block_diag((M_solid, M_shell), format="csr")

    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    free = np.asarray(ff.free_dofs(K.shape[0], fixed), dtype=int)
    matched_solid_dofs = np.asarray([3 * int(node) + comp for nodes in model["matched_solid_nodes"] for node in nodes for comp in range(3)], dtype=int)
    shell_nodes = np.arange(model["shell_coords"].shape[0], dtype=int)
    retained_full = np.unique(
        np.concatenate(
            [
                matched_solid_dofs,
                model["shell_offset"] + ff.shell_node_dofs(shell_nodes, "uxuyuz"),
                model["shell_offset"]
                + ff.shell_node_dofs(
                    np.flatnonzero(np.isclose(model["shell_coords"][:, 0], model["shell_coords"][:, 0].max()))
                ),
            ]
        )
    )
    retained = np.flatnonzero(np.isin(free, retained_full)).astype(np.int32)
    k_free = K[free, :][:, free]
    m_free = M[free, :][:, free]
    c_free = C[:, free]

    cb = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained_dofs=retained,
        n_modes=k_free.shape[0] - retained.size,
        backend="scipy",
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    full = _constrained_omegas(K, M, C, fixed, n_modes=6)
    rom = _constrained_omegas(
        sp.csr_matrix(cb.project_matrix(k_free)),
        sp.csr_matrix(cb.project_matrix(m_free)),
        sp.csr_matrix(c_free @ cb.basis),
        np.array([], dtype=int),
        n_modes=6,
    )

    np.testing.assert_allclose(rom, full, rtol=1.0e-6, atol=1.0e-5)

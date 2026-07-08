from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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

"""Mortar/Nitsche diagnostics on a nonmatching hex fixture/workpiece interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging
import os
import warnings

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*", category=UserWarning)

import fluxfem as ff


@dataclass(frozen=True)
class DemoConfig:
    nx_fixture: int = 2
    ny_fixture: int = 2
    nx_workpiece: int = 3
    ny_workpiece: int = 3
    thickness_fixture: float = 0.25
    thickness_workpiece: float = 0.25
    E: float = 1.0
    nu: float = 0.3
    alpha: float = 20.0
    rho: float = 10.0
    quad_order: int = 2
    jump_z: float = -1.0e-3
    kkt_regularization: float = 1.0e-2


def build_fixture_workpiece_contact(config: DemoConfig | None = None) -> tuple[Any, Any, Any]:
    cfg = DemoConfig() if config is None else config
    fixture_box = ff.StructuredHexBox(
        nx=cfg.nx_fixture,
        ny=cfg.ny_fixture,
        nz=1,
        lx=1.0,
        ly=1.0,
        lz=float(cfg.thickness_fixture),
        origin=(0.0, 0.0, 0.0),
        order=1,
    )
    workpiece_box = ff.StructuredHexBox(
        nx=cfg.nx_workpiece,
        ny=cfg.ny_workpiece,
        nz=1,
        lx=1.0,
        ly=1.0,
        lz=float(cfg.thickness_workpiece),
        origin=(0.0, 0.0, -float(cfg.thickness_workpiece)),
        order=1,
    )
    fixture_mesh = fixture_box.build()
    workpiece_mesh = workpiece_box.build()
    fixture_space = ff.make_hex_space(fixture_mesh, dim=3)
    workpiece_space = ff.make_hex_space(workpiece_mesh, dim=3)

    fixture_facets = np.asarray(fixture_mesh.facets_on_plane(axis=2, value=0.0, tol=1.0e-8), dtype=int)
    workpiece_facets = np.asarray(workpiece_mesh.facets_on_plane(axis=2, value=0.0, tol=1.0e-8), dtype=int)
    fixture_side = ff.ContactSide.from_facets(fixture_mesh, fixture_facets, fixture_space)
    workpiece_side = ff.ContactSide.from_facets(workpiece_mesh, workpiece_facets, workpiece_space)
    contact = ff.ContactSurfaceSpace.from_sides(
        fixture_side,
        workpiece_side,
        field_master="fixture",
        field_slave="workpiece",
        quad_order=int(cfg.quad_order),
        backend="numpy",
        normal_sign=-1.0,
    )
    return contact, fixture_space, workpiece_space


def demo_state(fixture_space, workpiece_space, *, jump_z: float) -> tuple[np.ndarray, np.ndarray]:
    u_fixture = np.zeros(int(fixture_space.n_dofs), dtype=float)
    u_workpiece = np.zeros(int(workpiece_space.n_dofs), dtype=float)
    u_workpiece[2::3] = float(jump_z)
    return u_fixture, u_workpiece


def _contact_params(cfg: DemoConfig) -> Any:
    lam, mu = ff.lame_parameters(cfg.E, cfg.nu)
    h = min(1.0 / cfg.nx_fixture, 1.0 / cfg.ny_fixture, 1.0 / cfg.nx_workpiece, 1.0 / cfg.ny_workpiece)
    return ff.Params(
        alpha=float(cfg.alpha),
        inv_h=float(1.0 / h),
        lam=float(lam),
        mu=float(mu),
    )


def run_demo(config: DemoConfig | None = None) -> dict[str, Any]:
    cfg = DemoConfig() if config is None else config
    contact, fixture_space, workpiece_space = build_fixture_workpiece_contact(cfg)
    state = demo_state(fixture_space, workpiece_space, jump_z=cfg.jump_z)

    multiplier = ff.MultiplierSpec.from_contact(
        contact,
        family="p0_supermesh",
        side="master",
        value_dim=3,
    )
    mortar = contact.assemble_multiplier(rho=cfg.rho, multiplier=multiplier, backend="numpy")
    mortar_diag = mortar.constraint_diagnostics(max_singular_values=10)
    mortar_quality = mortar.constraint_quality(max_condition_number=1.0e6)
    mortar_residual = mortar.constraint_residual(state)
    patch_qr_multiplier = ff.MultiplierSpec.patch_qr_mortar(
        contact,
        family="p0_supermesh",
        side="master",
        value_dim=3,
        constraint_scaling="l2",
    )
    mortar_patch_qr = contact.assemble_multiplier(rho=cfg.rho, multiplier=patch_qr_multiplier, backend="numpy")
    mortar_patch_qr_diag = mortar_patch_qr.constraint_diagnostics(max_singular_values=10)
    mortar_patch_qr_quality = mortar_patch_qr.constraint_quality(max_condition_number=1.0e6)
    mortar_patch_qr_residual = mortar_patch_qr.constraint_residual(state)
    qr_multiplier = ff.MultiplierSpec.algebraic_qr_mortar(
        contact,
        family="p0_supermesh",
        side="master",
        value_dim=3,
        constraint_scaling="l2",
    )
    mortar_qr = contact.assemble_multiplier(rho=cfg.rho, multiplier=qr_multiplier, backend="numpy")
    mortar_qr_diag = mortar_qr.constraint_diagnostics(max_singular_values=10)
    mortar_qr_quality = mortar_qr.constraint_quality(max_condition_number=1.0e6)
    mortar_qr_residual = mortar_qr.constraint_residual(state)

    params = _contact_params(cfg)
    nitsche_penalty = contact.assemble_pair_nitsche(
        params,
        sparse=False,
        use_penalty=1.0,
        use_traction=0.0,
    )
    nitsche_full = contact.assemble_pair_nitsche(
        params,
        sparse=False,
        use_penalty=1.0,
        use_traction=1.0,
    )

    B = np.asarray(mortar_qr.B, dtype=float)
    u = np.concatenate([np.asarray(state[0], dtype=float), np.asarray(state[1], dtype=float)])
    K_primal = np.asarray(nitsche_penalty.jacobian, dtype=float)
    K_primal = K_primal + float(cfg.kkt_regularization) * np.eye(B.shape[1], dtype=float)
    KKT = np.block(
        [
            [K_primal, B.T],
            [B, np.zeros((B.shape[0], B.shape[0]), dtype=float)],
        ]
    )
    solve_rhs = np.concatenate([np.zeros((B.shape[1],), dtype=float), -B @ u])
    solve_result = ff.solve_contact_kkt_with_info(
        KKT,
        solve_rhs,
        config=ff.ContactKKTSolveConfig(
            backend="numpy",
            numpy_solver="block_scaled",
            n_primal=int(B.shape[1]),
        ),
    )

    return {
        "fixture_dofs": int(fixture_space.n_dofs),
        "workpiece_dofs": int(workpiece_space.n_dofs),
        "fixture_facets": int(contact.surface_master.conn.shape[0]),
        "workpiece_facets": int(contact.surface_slave.conn.shape[0]),
        "supermesh_triangles": int(np.asarray(contact.supermesh_conn).shape[0]),
        "mortar": {
            "B_shape": tuple(int(v) for v in np.asarray(mortar.B).shape),
            "zero_row_count": int(mortar_diag.zero_row_count),
            "estimated_rank": int(mortar_diag.estimated_rank),
            "rank_deficiency": int(mortar_diag.rank_deficiency),
            "condition_number": float(mortar_diag.condition_number),
            "quality_status": str(mortar_quality.status),
            "quality_issues": tuple(issue.check for issue in mortar_quality.issues),
            "quality_hints": tuple((issue.check, issue.hint) for issue in mortar_quality.issues),
            "constraint_residual_norm": float(np.linalg.norm(mortar_residual)),
            "augmentation_energy": float(mortar.augmentation_energy(state)),
        },
        "mortar_patch_qr": {
            "B_shape": tuple(int(v) for v in np.asarray(mortar_patch_qr.B).shape),
            "zero_row_count": int(mortar_patch_qr_diag.zero_row_count),
            "estimated_rank": int(mortar_patch_qr_diag.estimated_rank),
            "rank_deficiency": int(mortar_patch_qr_diag.rank_deficiency),
            "condition_number": float(mortar_patch_qr_diag.condition_number),
            "quality_status": str(mortar_patch_qr_quality.status),
            "quality_issues": tuple(issue.check for issue in mortar_patch_qr_quality.issues),
            "quality_hints": tuple((issue.check, issue.hint) for issue in mortar_patch_qr_quality.issues),
            "row_norm_min": float(mortar_patch_qr_diag.row_norm_min),
            "row_norm_max": float(mortar_patch_qr_diag.row_norm_max),
            "constraint_residual_norm": float(np.linalg.norm(mortar_patch_qr_residual)),
            "augmentation_energy": float(mortar_patch_qr.augmentation_energy(state)),
        },
        "mortar_algebraic_qr": {
            "B_shape": tuple(int(v) for v in np.asarray(mortar_qr.B).shape),
            "zero_row_count": int(mortar_qr_diag.zero_row_count),
            "estimated_rank": int(mortar_qr_diag.estimated_rank),
            "rank_deficiency": int(mortar_qr_diag.rank_deficiency),
            "condition_number": float(mortar_qr_diag.condition_number),
            "quality_status": str(mortar_qr_quality.status),
            "quality_issues": tuple(issue.check for issue in mortar_qr_quality.issues),
            "quality_hints": tuple((issue.check, issue.hint) for issue in mortar_qr_quality.issues),
            "row_norm_min": float(mortar_qr_diag.row_norm_min),
            "row_norm_max": float(mortar_qr_diag.row_norm_max),
            "constraint_residual_norm": float(np.linalg.norm(mortar_qr_residual)),
            "augmentation_energy": float(mortar_qr.augmentation_energy(state)),
        },
        "nitsche": {
            "penalty_K_shape": tuple(int(v) for v in np.asarray(nitsche_penalty.jacobian).shape),
            "full_K_shape": tuple(int(v) for v in np.asarray(nitsche_full.jacobian).shape),
            "penalty_energy": float(nitsche_penalty.penalty_energy(state)),
            "penalty_matrix_norm": float(np.linalg.norm(np.asarray(nitsche_penalty.jacobian))),
            "full_matrix_norm": float(np.linalg.norm(np.asarray(nitsche_full.jacobian))),
        },
        "kkt_solve": {
            "solver": solve_result.info.solver,
            "residual_norm": float(solve_result.info.residual_norm),
            "scaled_residual_norm": float(solve_result.info.scaled_residual_norm),
            "primal_scaling_range": (
                float(solve_result.info.primal_scaling_min),
                float(solve_result.info.primal_scaling_max),
            ),
            "dual_scaling_range": (
                float(solve_result.info.dual_scaling_min),
                float(solve_result.info.dual_scaling_max),
            ),
            "scaled_row_norm_range": (
                float(solve_result.info.scaled_matrix_row_norm_min),
                float(solve_result.info.scaled_matrix_row_norm_max),
            ),
        },
    }


def main() -> None:
    result = run_demo()
    print("fixture/workpiece hex contact diagnostics")
    print("fixture dofs:", result["fixture_dofs"])
    print("workpiece dofs:", result["workpiece_dofs"])
    print("fixture facets:", result["fixture_facets"])
    print("workpiece facets:", result["workpiece_facets"])
    print("supermesh triangles:", result["supermesh_triangles"])
    print("mortar B shape:", result["mortar"]["B_shape"])
    print("mortar estimated rank:", result["mortar"]["estimated_rank"])
    print("mortar rank deficiency:", result["mortar"]["rank_deficiency"])
    print("mortar condition number:", f"{result['mortar']['condition_number']:.3e}")
    print("mortar quality status:", result["mortar"]["quality_status"])
    print("mortar quality issues:", result["mortar"]["quality_issues"])
    for check, hint in result["mortar"]["quality_hints"]:
        print(f"mortar quality hint [{check}]: {hint}")
    print("mortar constraint residual norm:", f"{result['mortar']['constraint_residual_norm']:.3e}")
    print("mortar augmentation energy:", f"{result['mortar']['augmentation_energy']:.3e}")
    print("mortar patch-qr B shape:", result["mortar_patch_qr"]["B_shape"])
    print("mortar patch-qr estimated rank:", result["mortar_patch_qr"]["estimated_rank"])
    print("mortar patch-qr rank deficiency:", result["mortar_patch_qr"]["rank_deficiency"])
    print("mortar patch-qr quality status:", result["mortar_patch_qr"]["quality_status"])
    print(
        "mortar patch-qr row norm range:",
        (
            f"{result['mortar_patch_qr']['row_norm_min']:.3e}",
            f"{result['mortar_patch_qr']['row_norm_max']:.3e}",
        ),
    )
    print("mortar patch-qr constraint residual norm:", f"{result['mortar_patch_qr']['constraint_residual_norm']:.3e}")
    print("mortar algebraic-qr B shape:", result["mortar_algebraic_qr"]["B_shape"])
    print("mortar algebraic-qr estimated rank:", result["mortar_algebraic_qr"]["estimated_rank"])
    print("mortar algebraic-qr rank deficiency:", result["mortar_algebraic_qr"]["rank_deficiency"])
    print("mortar algebraic-qr quality status:", result["mortar_algebraic_qr"]["quality_status"])
    print(
        "mortar algebraic-qr row norm range:",
        (
            f"{result['mortar_algebraic_qr']['row_norm_min']:.3e}",
            f"{result['mortar_algebraic_qr']['row_norm_max']:.3e}",
        ),
    )
    print("mortar algebraic-qr constraint residual norm:", f"{result['mortar_algebraic_qr']['constraint_residual_norm']:.3e}")
    print("nitsche penalty K shape:", result["nitsche"]["penalty_K_shape"])
    print("nitsche full K shape:", result["nitsche"]["full_K_shape"])
    print("nitsche penalty energy:", f"{result['nitsche']['penalty_energy']:.3e}")
    print("nitsche penalty matrix norm:", f"{result['nitsche']['penalty_matrix_norm']:.3e}")
    print("nitsche full matrix norm:", f"{result['nitsche']['full_matrix_norm']:.3e}")
    print("KKT solver:", result["kkt_solve"]["solver"])
    print("KKT residual norm:", f"{result['kkt_solve']['residual_norm']:.3e}")
    print("KKT scaled residual norm:", f"{result['kkt_solve']['scaled_residual_norm']:.3e}")
    print("KKT primal scaling range:", tuple(f"{v:.3e}" for v in result["kkt_solve"]["primal_scaling_range"]))
    print("KKT dual scaling range:", tuple(f"{v:.3e}" for v in result["kkt_solve"]["dual_scaling_range"]))
    print("KKT scaled row norm range:", tuple(f"{v:.3e}" for v in result["kkt_solve"]["scaled_row_norm_range"]))


if __name__ == "__main__":
    main()

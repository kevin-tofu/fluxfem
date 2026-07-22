"""Compare mortar and pair-Nitsche assembly on the same supermesh overlap."""

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
    E: float = 1.0
    nu: float = 0.3
    alpha: float = 20.0
    inv_h: float = 1.0
    rho: float = 10.0
    quad_order: int = 2
    jump_z: float = -1.0e-3


def build_split_tet_contact(quad_order: int = 2):
    """One master tet face against two slave tet faces covering the same triangle."""
    coords_master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn_master = np.array([[0, 1, 2, 3]], dtype=int)
    coords_slave = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn_slave = np.array(
        [
            [0, 1, 3, 4],
            [0, 3, 2, 4],
        ],
        dtype=int,
    )

    mesh_master = ff.TetMesh(coords=coords_master, conn=conn_master)
    mesh_slave = ff.TetMesh(coords=coords_slave, conn=conn_slave)

    def contact_facets(mesh):
        return mesh.facets_on_plane(axis=2, value=0.0)

    return ff.ContactSurfaceSpace.from_facets(
        coords_master,
        contact_facets(mesh_master),
        coords_slave,
        contact_facets(mesh_slave),
        elem_conn_master=conn_master,
        elem_conn_slave=conn_slave,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=int(quad_order),
        normal_sign=-1.0,
        backend="numpy",
    )


def demo_state(contact, *, jump_z: float) -> tuple[np.ndarray, np.ndarray]:
    n_master = int(contact.surface_master.n_nodes)
    n_slave = int(contact.surface_slave.n_nodes)
    u_master = np.zeros(n_master * 3, dtype=float)
    u_slave = np.zeros(n_slave * 3, dtype=float)
    u_slave[2::3] = float(jump_z)
    return u_master, u_slave


def run_demo(config: DemoConfig | None = None) -> dict[str, Any]:
    cfg = DemoConfig() if config is None else config
    contact = build_split_tet_contact(quad_order=cfg.quad_order)
    state = demo_state(contact, jump_z=cfg.jump_z)

    multiplier = ff.MultiplierSpec.from_contact(
        contact,
        family="p0_supermesh",
        side="master",
        value_dim=3,
    )
    mortar = contact.assemble_multiplier(rho=cfg.rho, multiplier=multiplier, backend="numpy")
    mortar_diag = mortar.constraint_diagnostics(max_singular_values=8)
    mortar_quality = mortar.constraint_quality(max_condition_number=1.0e6)
    mortar_residual = mortar.constraint_residual(state)

    lam, mu = ff.lame_parameters(cfg.E, cfg.nu)
    params = ff.Params(
        alpha=float(cfg.alpha),
        inv_h=float(cfg.inv_h),
        lam=float(lam),
        mu=float(mu),
    )
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

    K_aug = np.asarray(mortar.Kuu, dtype=float)
    u = np.concatenate([np.asarray(state[0], dtype=float), np.asarray(state[1], dtype=float)])
    kkt_residual = np.concatenate([K_aug @ u, np.asarray(mortar.B, dtype=float) @ u])
    B = np.asarray(mortar.B, dtype=float)
    K_primal = np.asarray(nitsche_penalty.jacobian, dtype=float) + 1.0e-2 * np.eye(B.shape[1], dtype=float)
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
        "supermesh_triangles": int(np.asarray(contact.supermesh_conn).shape[0]),
        "mortar": {
            "B_shape": tuple(int(v) for v in mortar.B.shape),
            "zero_row_count": int(mortar_diag.zero_row_count),
            "estimated_rank": int(mortar_diag.estimated_rank),
            "rank_deficiency": int(mortar_diag.rank_deficiency),
            "condition_number": float(mortar_diag.condition_number),
            "quality_status": str(mortar_quality.status),
            "quality_issues": tuple(issue.check for issue in mortar_quality.issues),
            "constraint_residual_norm": float(np.linalg.norm(mortar_residual)),
            "augmentation_energy": float(mortar.augmentation_energy(state)),
            "kkt_residual_norm_at_zero_lambda": float(np.linalg.norm(kkt_residual)),
        },
        "nitsche": {
            "penalty_K_shape": tuple(int(v) for v in np.asarray(nitsche_penalty.jacobian).shape),
            "full_K_shape": tuple(int(v) for v in np.asarray(nitsche_full.jacobian).shape),
            "penalty_energy": float(nitsche_penalty.penalty_energy(state)),
            "penalty_matrix_norm": float(np.linalg.norm(np.asarray(nitsche_penalty.jacobian))),
            "full_matrix_norm": float(np.linalg.norm(np.asarray(nitsche_full.jacobian))),
            "supermesh_triangles": int(nitsche_full.diagnostics["supermesh_triangles"]),
        },
        "kkt_solve": {
            "solver": solve_result.info.solver,
            "residual_norm": float(solve_result.info.residual_norm),
            "relative_residual_norm": float(solve_result.info.relative_residual_norm),
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
    print("supermesh triangles:", result["supermesh_triangles"])
    print("mortar B shape:", result["mortar"]["B_shape"])
    print("mortar estimated rank:", result["mortar"]["estimated_rank"])
    print("mortar rank deficiency:", result["mortar"]["rank_deficiency"])
    print("mortar condition number:", f"{result['mortar']['condition_number']:.3e}")
    print("mortar quality status:", result["mortar"]["quality_status"])
    print("mortar quality issues:", result["mortar"]["quality_issues"])
    print("mortar constraint residual norm:", f"{result['mortar']['constraint_residual_norm']:.3e}")
    print("mortar augmentation energy:", f"{result['mortar']['augmentation_energy']:.3e}")
    print("mortar KKT residual norm at zero lambda:", f"{result['mortar']['kkt_residual_norm_at_zero_lambda']:.3e}")
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

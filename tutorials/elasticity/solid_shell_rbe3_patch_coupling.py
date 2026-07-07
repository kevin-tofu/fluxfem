#!/usr/bin/env python3
"""
3D solid face coupled to a Q4 shell root edge through an RBE3 remote point.

The solid right face is reduced to a 6-DOF remote point by weighted RBE3-style
distributed coupling. The shell root edge is then tied to that remote point in
all six DOFs, so the shell receives both average translation and average
rotation reconstructed from the solid face patch.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import scipy.sparse as sp
import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def _as_csr(matrix):
    if sp.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "to_csr"):
        return matrix.to_csr()
    if hasattr(matrix, "toarray"):
        return sp.csr_matrix(matrix.toarray())
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def build_solid_shell_rbe3_patch_coupling(
    *,
    solid_nx: int = 3,
    solid_ny: int = 2,
    solid_nz: int = 2,
    solid_length: float = 1.0,
    solid_width: float = 0.4,
    solid_height: float = 0.3,
    shell_nx: int = 4,
    shell_ny: int = 2,
    shell_length: float = 1.0,
    tip_load_y: float = -5.0,
):
    solid_mesh = ff.StructuredHexBox(
        nx=solid_nx,
        ny=solid_ny,
        nz=solid_nz,
        lx=solid_length,
        ly=solid_width,
        lz=solid_height,
    ).build()
    solid_space = ff.make_hex_space(solid_mesh, dim=3, intorder=2)
    solid_K = _as_csr(solid_space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(2.0e5, 0.30)))
    solid_F = np.zeros((solid_space.n_dofs,), dtype=float)
    solid_coords = np.asarray(solid_mesh.coords, dtype=float)

    x_min = float(np.min(solid_coords[:, 0]))
    x_max = float(np.max(solid_coords[:, 0]))
    face_nodes = np.flatnonzero(np.isclose(solid_coords[:, 0], x_max, atol=1.0e-10)).astype(int)
    face_coords = solid_coords[face_nodes]
    face_dofs = ff.vector_dofs_from_nodes(face_nodes, dim=3)
    x_ref = face_coords.mean(axis=0)

    shell_xy, shell_conn = ff.structured_plate_grid(
        nx=shell_nx,
        ny=shell_ny,
        length_x=shell_length,
        length_y=solid_height,
    )
    shell_coords = np.column_stack(
        [
            x_ref[0] + shell_xy[:, 0],
            np.full(shell_xy.shape[0], x_ref[1]),
            shell_xy[:, 1],
        ]
    )
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02)
    shell_K = ff.assemble_shell_stiffness(shell_coords, shell_conn, shell_section, format="csr")

    free_edge = np.flatnonzero(np.isclose(shell_xy[:, 0], shell_length, atol=1.0e-10))
    forces = np.zeros((free_edge.size, 3), dtype=float)
    forces[:, 1] = tip_load_y / free_edge.size
    shell_F = ff.assemble_flat_shell_point_loads(shell_coords.shape[0], free_edge, forces=forces)

    structural_K = sp.block_diag((solid_K, shell_K), format="csr")
    structural_F = np.concatenate([solid_F, np.asarray(shell_F, dtype=float)])
    shell_offset = solid_space.n_dofs

    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_K, structural_F)
    builder.register_field("solid", n_dofs=solid_space.n_dofs, value_dim=1, offset=0)
    builder.register_field("shell", n_dofs=shell_K.shape[0], value_dim=1, offset=shell_offset)

    weights = ff.build_rbe3_weights(x_ref, face_coords, method="equal")
    builder.add_distributed_coupling(
        source="solid",
        source_dofs=face_dofs,
        remote="interface_remote",
        point=x_ref,
        slave_coords=face_coords,
        weights=weights,
        backend="numpy",
    )

    root_edge = np.flatnonzero(np.isclose(shell_xy[:, 0], 0.0, atol=1.0e-10))
    builder.add_dof_tie_constraint(
        master="interface_remote",
        slave="shell",
        master_dofs=np.tile(np.arange(6, dtype=int), root_edge.size),
        slave_dofs=ff.shell_node_dofs(root_edge),
    )

    fixed_solid = solid_mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_min, atol=1.0e-10),
        components="xyz",
        dof_per_node=3,
    )

    return {
        "builder": builder,
        "system": builder.build(),
        "solid_mesh": solid_mesh,
        "solid_space": solid_space,
        "solid_coords": solid_coords,
        "shell_coords": shell_coords,
        "shell_conn": shell_conn,
        "shell_offset": shell_offset,
        "shell_n_dofs": shell_K.shape[0],
        "face_nodes": face_nodes,
        "fixed_dofs": fixed_solid,
        "remote_dofs": builder.resolve_block_dofs("interface_remote", local_dofs=np.arange(6)),
        "shell_root_dofs": shell_offset + ff.shell_node_dofs(root_edge),
        "shell_tip_dofs": shell_offset + ff.shell_node_dofs(free_edge),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Couple a solid face to a shell root edge through an RBE3 remote point.")
    p.add_argument("--shell-nx", type=int, default=4)
    p.add_argument("--shell-ny", type=int, default=2)
    p.add_argument("--tip-load-y", type=float, default=-5.0)
    p.add_argument("--shell-vtu", default="", help="Optional VTU path for shell visualization.")
    return p.parse_args()


def main():
    args = parse_args()
    model = build_solid_shell_rbe3_patch_coupling(
        shell_nx=args.shell_nx,
        shell_ny=args.shell_ny,
        tip_load_y=args.tip_load_y,
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

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False

    remote_q = u_all[model["remote_dofs"]]
    shell_root_q = u_all[model["shell_root_dofs"]].reshape(-1, 6)
    shell_tip_q = u_all[model["shell_tip_dofs"]].reshape(-1, 6)
    print("solid nodes:          ", model["solid_coords"].shape[0])
    print("shell nodes:          ", model["shell_coords"].shape[0])
    print("coupled unknowns:     ", K_full.shape[0])
    print("interface remote q:   ", remote_q)
    print("max root mismatch:    ", f"{float(np.max(np.abs(shell_root_q - remote_q[None, :]))):.8e}")
    print("shell tip mean u:     ", np.mean(shell_tip_q[:, :3], axis=0))
    print("free residual norm:   ", f"{float(np.linalg.norm(residual[free])):.8e}")

    if args.shell_vtu:
        shell_u = u_all[model["shell_offset"] : model["shell_offset"] + model["shell_n_dofs"]].reshape(-1, 6)
        disp = shell_u[:, :3]
        rot = shell_u[:, 3:6]
        ff.write_q4_surface_vtu(
            model["shell_coords"] + disp,
            model["shell_conn"],
            args.shell_vtu,
            point_data={"reference_coords": model["shell_coords"], "displacement": disp, "rotation": rot},
        )
        print(f"wrote shell VTU: {args.shell_vtu}")


if __name__ == "__main__":
    main()

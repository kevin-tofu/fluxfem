#!/usr/bin/env python3
"""
3D solid block tied to a Q4 Reissner-Mindlin shell skin by translations.

The shell nodes lie on the solid top surface. Their translational DOFs are tied
to the coincident solid surface nodes. Shell rotations are retained as shell
DOFs; this is a translational shell-solid tie, not a rotational continuum-shell
constraint.
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


def build_solid_shell_tie(
    *,
    nx: int = 3,
    ny: int = 2,
    nz: int = 1,
    length: float = 1.0,
    width: float = 0.4,
    height: float = 0.2,
    pressure_z: float = -5.0,
):
    solid_mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=length, ly=width, lz=height).build()
    solid_space = ff.make_hex_space(solid_mesh, dim=3, intorder=2)
    solid_K = _as_csr(solid_space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(2.0e5, 0.30)))
    solid_F = np.zeros((solid_space.n_dofs,), dtype=float)
    solid_coords = np.asarray(solid_mesh.coords, dtype=float)

    shell_xy, shell_conn = ff.structured_plate_grid(nx=nx, ny=ny, length_x=length, length_y=width)
    shell_coords = np.column_stack([shell_xy[:, 0], shell_xy[:, 1], np.full(shell_xy.shape[0], height)])
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02)
    shell_K = ff.assemble_shell_stiffness(shell_coords, shell_conn, shell_section, format="csr")
    shell_F = ff.assemble_shell_uniform_load(shell_coords, shell_conn, (0.0, 0.0, pressure_z))

    structural_K = sp.block_diag((solid_K, shell_K), format="csr")
    structural_F = np.concatenate([solid_F, np.asarray(shell_F, dtype=float)])
    shell_offset = solid_space.n_dofs

    top_nodes = np.flatnonzero(np.isclose(solid_coords[:, 2], height, atol=1.0e-10))
    matched_shell, matched_solid, shell_local_dofs, solid_dofs = ff.shell_solid_translational_tie_dofs(
        shell_coords,
        solid_coords,
        solid_nodes=top_nodes,
    )

    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_K, structural_F)
    builder.register_field("solid", n_dofs=solid_space.n_dofs, value_dim=1, offset=0)
    builder.register_field("shell", n_dofs=shell_K.shape[0], value_dim=1, offset=shell_offset)
    builder.add_dof_tie_constraint(
        master="solid",
        slave="shell",
        master_dofs=solid_dofs,
        slave_dofs=shell_local_dofs,
    )

    fixed_solid = solid_mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0, atol=1.0e-10),
        components="xyz",
        dof_per_node=3,
    )
    fixed_shell_rot = shell_offset + ff.shell_node_dofs(np.flatnonzero(np.isclose(shell_coords[:, 0], 0.0)), "rxryrz")
    fixed_dofs = np.unique(np.concatenate([fixed_solid, fixed_shell_rot]))

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
        "matched_shell_nodes": matched_shell,
        "matched_solid_nodes": matched_solid,
        "solid_tie_dofs": solid_dofs,
        "shell_tie_dofs": shell_offset + shell_local_dofs,
        "fixed_dofs": fixed_dofs,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Tie a shell skin to a solid surface by translational DOFs.")
    p.add_argument("--nx", type=int, default=3)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--pressure-z", type=float, default=-5.0)
    p.add_argument("--shell-vtu", default="", help="Optional VTU path for shell visualization.")
    return p.parse_args()


def main():
    args = parse_args()
    model = build_solid_shell_tie(nx=args.nx, ny=args.ny, nz=args.nz, pressure_z=args.pressure_z)
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

    solid_tie_u = u_all[model["solid_tie_dofs"]].reshape(-1, 3)
    shell_tie_u = u_all[model["shell_tie_dofs"]].reshape(-1, 3)
    shell_u = u_all[model["shell_offset"] : model["shell_offset"] + model["shell_n_dofs"]].reshape(-1, 6)
    print("solid nodes:        ", model["solid_coords"].shape[0])
    print("shell nodes:        ", model["shell_coords"].shape[0])
    print("matched tie nodes:  ", model["matched_shell_nodes"].size)
    print("max tie mismatch:   ", f"{float(np.max(np.abs(solid_tie_u - shell_tie_u))):.8e}")
    print("shell min uz:       ", f"{float(np.min(shell_u[:, 2])):.8e}")
    print("free residual norm: ", f"{float(np.linalg.norm(residual[free])):.8e}")

    if args.shell_vtu:
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

#!/usr/bin/env python3
"""
3D solid surface tied to a nonmatching Q4 shell skin by translational interpolation.

Shell translational DOFs are constrained to the displacement interpolated from
the containing solid surface facet. Shell rotations remain independent shell
DOFs. This is a small planar-surface helper, not a mortar shell-solid coupling.
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


def build_solid_shell_nonmatching_tie(
    *,
    solid_nx: int = 2,
    solid_ny: int = 1,
    solid_nz: int = 1,
    shell_nx: int = 4,
    shell_ny: int = 2,
    length: float = 1.0,
    width: float = 0.4,
    height: float = 0.2,
    pressure_z: float = -1.0,
    shear_mode: str = "reduced",
):
    solid_mesh = ff.StructuredHexBox(nx=solid_nx, ny=solid_ny, nz=solid_nz, lx=length, ly=width, lz=height).build()
    solid_space = ff.make_hex_space(solid_mesh, dim=3, intorder=2)
    solid_K = _as_csr(solid_space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(2.0e5, 0.30)))
    solid_F = np.zeros((solid_space.n_dofs,), dtype=float)
    solid_coords = np.asarray(solid_mesh.coords, dtype=float)

    shell_xy, shell_conn = ff.structured_plate_grid(nx=shell_nx, ny=shell_ny, length_x=length, length_y=width)
    shell_coords = np.column_stack([shell_xy[:, 0], shell_xy[:, 1], np.full(shell_xy.shape[0], height)])
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02, shear_mode=shear_mode)
    shell_K = ff.assemble_shell_stiffness(shell_coords, shell_conn, shell_section, format="csr")
    shell_F = ff.assemble_shell_uniform_load(shell_coords, shell_conn, (0.0, 0.0, pressure_z))

    structural_K = sp.block_diag((solid_K, shell_K), format="csr")
    structural_F = np.concatenate([solid_F, np.asarray(shell_F, dtype=float)])
    shell_offset = solid_space.n_dofs

    top_facets = np.asarray(solid_mesh.boundary_facets_plane(axis=2, value=height, tol=1.0e-10), dtype=int)
    C, matched_facets, matched_solid_nodes, matched_weights = ff.shell_solid_nonmatching_translational_tie_matrix(
        shell_coords,
        solid_coords,
        top_facets,
    )

    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_K, structural_F)
    builder.register_field("solid", n_dofs=solid_space.n_dofs, value_dim=1, offset=0)
    builder.register_field("shell", n_dofs=shell_K.shape[0], value_dim=1, offset=shell_offset)
    builder.add_constraint_matrix_dof(C, master="solid", slave="shell")

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
        "constraint_matrix": C,
        "matched_facets": matched_facets,
        "matched_solid_nodes": matched_solid_nodes,
        "matched_weights": matched_weights,
        "fixed_dofs": fixed_dofs,
        "shear_mode": shear_mode,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Tie a nonmatching shell skin to a solid surface by interpolated translations.")
    p.add_argument("--solid-nx", type=int, default=2)
    p.add_argument("--solid-ny", type=int, default=1)
    p.add_argument("--solid-nz", type=int, default=1)
    p.add_argument("--shell-nx", type=int, default=4)
    p.add_argument("--shell-ny", type=int, default=2)
    p.add_argument("--pressure-z", type=float, default=-1.0)
    p.add_argument("--shear-mode", choices=("reduced", "full", "mitc4"), default="reduced")
    p.add_argument("--shell-vtu", default="", help="Optional VTU path for shell visualization.")
    return p.parse_args()


def main():
    args = parse_args()
    model = build_solid_shell_nonmatching_tie(
        solid_nx=args.solid_nx,
        solid_ny=args.solid_ny,
        solid_nz=args.solid_nz,
        shell_nx=args.shell_nx,
        shell_ny=args.shell_ny,
        pressure_z=args.pressure_z,
        shear_mode=args.shear_mode,
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

    solid_u = u_all[: model["solid_space"].n_dofs]
    shell_u = u_all[model["shell_offset"] : model["shell_offset"] + model["shell_n_dofs"]]
    tie_residual = model["constraint_matrix"] @ np.concatenate([solid_u, shell_u])
    shell_q = shell_u.reshape(-1, 6)

    print("solid nodes:          ", model["solid_coords"].shape[0])
    print("shell nodes:          ", model["shell_coords"].shape[0])
    print("shell shear mode:     ", model["shear_mode"])
    print("nonmatching tie rows: ", model["constraint_matrix"].shape[0])
    print("max tie residual:     ", f"{float(np.max(np.abs(tie_residual))):.8e}")
    print("shell min uz:         ", f"{float(np.min(shell_q[:, 2])):.8e}")
    print("free residual norm:   ", f"{float(np.linalg.norm(residual[free])):.8e}")

    if args.shell_vtu:
        disp = shell_q[:, :3]
        rot = shell_q[:, 3:6]
        ff.write_q4_surface_vtu(
            model["shell_coords"] + disp,
            model["shell_conn"],
            args.shell_vtu,
            point_data={"reference_coords": model["shell_coords"], "displacement": disp, "rotation": rot},
        )
        print(f"wrote shell VTU: {args.shell_vtu}")


if __name__ == "__main__":
    main()

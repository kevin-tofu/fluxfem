#!/usr/bin/env python3
"""
Flat Q4 Reissner-Mindlin shell cantilever.

Each node has shell DOFs [ux, uy, uz, theta_x, theta_y, theta_z]. The example
clamps the left edge and applies an equal transverse point load over the right
edge.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def parse_args():
    p = argparse.ArgumentParser(description="Cantilever with flat Q4 Reissner-Mindlin shell elements.")
    p.add_argument("--format", choices=("csr", "fluxsparse", "dense"), default="csr", help="Matrix assembly format.")
    p.add_argument("--solver", choices=("auto", "spsolve", "spsolve_jax"), default="auto", help="Linear solver backend.")
    p.add_argument("--nx", type=int, default=8, help="Elements along the cantilever length.")
    p.add_argument("--ny", type=int, default=2, help="Elements across the width.")
    p.add_argument("--length", type=float, default=2.0, help="Shell length.")
    p.add_argument("--width", type=float, default=0.4, help="Shell width.")
    p.add_argument("--thickness", type=float, default=0.04, help="Shell thickness.")
    p.add_argument("--shear-mode", choices=("reduced", "full", "mitc4"), default="reduced", help="Transverse shear integration/assumed-strain mode.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--nu", type=float, default=0.3, help="Poisson ratio.")
    p.add_argument("--tip-load-z", type=float, default=-1000.0, help="Total transverse load on the free edge.")
    p.add_argument("--tilt-z", type=float, default=0.0, help="Optional z = tilt_z * x coordinate for a 3D planar shell.")
    p.add_argument("--output-vtu", default="", help="Optional VTU path for deformed shell visualization.")
    p.add_argument("--deformed-scale", type=float, default=1.0, help="Scale factor for VTU deformed coordinates.")
    return p.parse_args()


def _auto_solver(format: str, solver: str) -> str:
    if solver != "auto":
        return solver
    return "spsolve_jax" if format == "fluxsparse" else "spsolve"


def _matrix_nnz(matrix) -> int:
    if hasattr(matrix, "nnz"):
        return int(matrix.nnz)
    return int(np.count_nonzero(np.asarray(matrix)))


def main():
    args = parse_args()
    coords, conn = ff.structured_plate_grid(nx=args.nx, ny=args.ny, length_x=args.length, length_y=args.width)
    shell_coords = coords
    if args.tilt_z != 0.0:
        shell_coords = np.column_stack([coords[:, 0], coords[:, 1], args.tilt_z * coords[:, 0]])
    section = ff.ShellSection(E=args.E, nu=args.nu, thickness=args.thickness, shear_mode=args.shear_mode)
    K = ff.assemble_shell_stiffness(shell_coords, conn, section, format=args.format)

    left_nodes = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    right_nodes = np.flatnonzero(np.isclose(coords[:, 0], args.length))
    forces = np.zeros((right_nodes.size, 3), dtype=float)
    forces[:, 2] = args.tip_load_z / right_nodes.size
    F = ff.assemble_flat_shell_point_loads(
        coords.shape[0],
        right_nodes,
        forces=forces,
        array_backend="jax" if args.format == "fluxsparse" else "numpy",
    )

    fixed = ff.shell_node_dofs(left_nodes)
    solver = _auto_solver(args.format, args.solver)
    u, _info = ff.LinearSolver(method=solver).solve(
        K,
        F,
        dirichlet=ff.DirichletBC(fixed, 0.0),
        dirichlet_mode="condense",
    )

    uz_nodes = np.asarray(u)[2::6]
    ux_nodes = np.asarray(u)[0::6]
    right_uz = uz_nodes[right_nodes]
    tip_mid = right_nodes[np.argmin(np.abs(coords[right_nodes, 1] - 0.5 * args.width))]

    print(
        f"flat shell solved: format={args.format}, solver={solver}, "
        f"shear_mode={args.shear_mode}, nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={6 * coords.shape[0]}"
    )
    print(f"tip mid-node uz={float(uz_nodes[tip_mid]):.6e}")
    print(f"free-edge mean uz={float(np.mean(right_uz)):.6e}, min uz={float(np.min(right_uz)):.6e}")
    print(f"max |ux|={float(np.max(np.abs(ux_nodes))):.6e}, stiffness nnz={_matrix_nnz(K)}")

    if args.output_vtu:
        coords3 = np.zeros((coords.shape[0], 3), dtype=float)
        if np.asarray(shell_coords).shape[1] == 2:
            coords3[:, :2] = shell_coords
        else:
            coords3 = np.asarray(shell_coords, dtype=float)
        disp = np.asarray(u).reshape(-1, 6)[:, :3]
        rot = np.asarray(u).reshape(-1, 6)[:, 3:6]
        ff.write_q4_surface_vtu(
            coords3 + args.deformed_scale * disp,
            conn,
            args.output_vtu,
            point_data={
                "reference_coords": coords3,
                "displacement": disp,
                "rotation": rot,
            },
        )
        print(f"wrote VTU: {args.output_vtu}")


if __name__ == "__main__":
    main()

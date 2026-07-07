#!/usr/bin/env python3
"""
Q4 Reissner-Mindlin plate cantilever.

Each node has plate DOFs [w, theta_x, theta_y]. The example clamps the left
edge, applies an equal transverse point load over the right edge, and solves the
linear plate system.
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
    p = argparse.ArgumentParser(description="Cantilever plate with Q4 Reissner-Mindlin elements.")
    p.add_argument("--format", choices=("csr", "fluxsparse", "dense"), default="csr", help="Matrix assembly format.")
    p.add_argument("--solver", choices=("auto", "spsolve", "spsolve_jax"), default="auto", help="Linear solver backend.")
    p.add_argument("--nx", type=int, default=8, help="Elements along the cantilever length.")
    p.add_argument("--ny", type=int, default=2, help="Elements across the width.")
    p.add_argument("--length", type=float, default=2.0, help="Plate length.")
    p.add_argument("--width", type=float, default=0.4, help="Plate width.")
    p.add_argument("--thickness", type=float, default=0.04, help="Plate thickness.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--nu", type=float, default=0.3, help="Poisson ratio.")
    p.add_argument("--tip-load", type=float, default=-1000.0, help="Total transverse load on the free edge.")
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
    section = ff.PlateSection(E=args.E, nu=args.nu, thickness=args.thickness)
    K = ff.assemble_mindlin_plate_stiffness(coords, conn, section, format=args.format)

    left_nodes = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    right_nodes = np.flatnonzero(np.isclose(coords[:, 0], args.length))
    F = ff.assemble_mindlin_plate_point_loads(
        coords.shape[0],
        right_nodes,
        forces=np.full(right_nodes.size, args.tip_load / right_nodes.size),
        array_backend="jax" if args.format == "fluxsparse" else "numpy",
    )

    fixed = ff.plate_node_dofs(left_nodes)
    solver = _auto_solver(args.format, args.solver)
    u, _info = ff.LinearSolver(method=solver).solve(
        K,
        F,
        dirichlet=ff.DirichletBC(fixed, 0.0),
        dirichlet_mode="condense",
    )

    w_nodes = np.asarray(u)[0::3]
    right_w = w_nodes[right_nodes]
    tip_mid = right_nodes[np.argmin(np.abs(coords[right_nodes, 1] - 0.5 * args.width))]
    D = args.E * args.thickness**3 / (12.0 * (1.0 - args.nu**2))

    print(
        f"mindlin plate solved: format={args.format}, solver={solver}, "
        f"nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={3 * coords.shape[0]}"
    )
    print(f"tip mid-node w={float(w_nodes[tip_mid]):.6e}")
    print(f"free-edge mean w={float(np.mean(right_w)):.6e}, min w={float(np.min(right_w)):.6e}")
    print(f"bending rigidity D={D:.6e}, stiffness nnz={_matrix_nnz(K)}")


if __name__ == "__main__":
    main()

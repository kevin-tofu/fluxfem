#!/usr/bin/env python3
"""
2D truss/bar cantilever in the x-z plane.

Each node has translational DOFs [ux, uz]. The example fixes the left end,
keeps the chain lateral z DOFs fixed to model a 1D bar, and compares the tip
displacement with

  u(L) = P L / (E A).
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
    p = argparse.ArgumentParser(description="Cantilever bar with 2D truss elements.")
    p.add_argument("--format", choices=("csr", "fluxsparse", "dense"), default="csr", help="Matrix assembly format.")
    p.add_argument("--solver", choices=("auto", "spsolve", "spsolve_jax"), default="auto", help="Linear solver backend.")
    p.add_argument("--n-elems", type=int, default=8, help="Number of truss/bar elements.")
    p.add_argument("--length", type=float, default=3.0, help="Bar length.")
    p.add_argument("--E", type=float, default=70.0e9, help="Young's modulus.")
    p.add_argument("--A", type=float, default=1.5e-3, help="Cross-sectional area.")
    p.add_argument("--rho", type=float, default=2700.0, help="Density for optional mass assembly.")
    p.add_argument("--tip-load-x", type=float, default=1200.0, help="Tip load in global x.")
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

    coords, conn = ff.structured_truss2d_chain(n_elems=args.n_elems, length=args.length)
    section = ff.TrussSection(E=args.E, A=args.A, rho=args.rho)
    K = ff.assemble_truss2d_stiffness(coords, conn, section, format=args.format)
    M = ff.assemble_truss2d_mass(coords, conn, section, format=args.format)

    n_dofs = ff.TRUSS2D_DOF_PER_NODE * coords.shape[0]
    tip_node = coords.shape[0] - 1
    F = ff.assemble_truss2d_point_load(
        coords.shape[0],
        tip_node,
        force=(args.tip_load_x, 0.0),
        array_backend="jax" if args.format == "fluxsparse" else "numpy",
    )

    fixed = ff.truss2d_node_dofs([0], "xz")
    lateral = ff.truss2d_node_dofs(np.arange(1, coords.shape[0]), "z")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))

    solver = _auto_solver(args.format, args.solver)
    u, _info = ff.LinearSolver(method=solver).solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dirichlet, 0.0),
        dirichlet_mode="condense",
    )

    ux_tip = float(u[ff.truss2d_node_dofs([tip_node], "x")][0])
    ux_exact = args.tip_load_x * args.length / (args.E * args.A)
    rel_err = abs(ux_tip - ux_exact) / abs(ux_exact) if ux_exact != 0.0 else 0.0

    print(f"truss2d/bar solved: format={args.format}, solver={solver}, nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={n_dofs}")
    print(f"tip ux={ux_tip:.6e}, bar exact={ux_exact:.6e}, rel.err={rel_err:.3e}")
    print(f"stiffness nnz={_matrix_nnz(K)}, mass nnz={_matrix_nnz(M)}")


if __name__ == "__main__":
    main()

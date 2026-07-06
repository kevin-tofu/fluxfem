#!/usr/bin/env python3
"""
3D truss/bar cantilever under a uniform axial load.

The equivalent nodal load vector is assembled with ``assemble_truss_uniform_load``.
For a fixed-free bar with axial load q, the tip displacement is

  u(L) = q L^2 / (2 E A).
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
    p = argparse.ArgumentParser(description="Cantilever truss/bar with uniform axial load.")
    p.add_argument("--backend", choices=("jax", "scipy", "numpy"), default="jax", help="Matrix assembly backend.")
    p.add_argument("--solver", choices=("auto", "spsolve", "spsolve_jax"), default="auto", help="Linear solver backend.")
    p.add_argument("--n-elems", type=int, default=8, help="Number of truss/bar elements.")
    p.add_argument("--length", type=float, default=3.0, help="Bar length.")
    p.add_argument("--E", type=float, default=70.0e9, help="Young's modulus.")
    p.add_argument("--A", type=float, default=1.5e-3, help="Cross-sectional area.")
    p.add_argument("--qx", type=float, default=400.0, help="Uniform local axial load per unit length.")
    return p.parse_args()


def _auto_solver(backend: str, solver: str) -> str:
    if solver != "auto":
        return solver
    return "spsolve_jax" if backend == "jax" else "spsolve"


def main():
    args = parse_args()

    coords, conn = ff.structured_truss_chain(n_elems=args.n_elems, length=args.length)
    section = ff.TrussSection(E=args.E, A=args.A)
    K = ff.assemble_truss_stiffness(coords, conn, section, backend=args.backend)
    F = ff.assemble_truss_uniform_load(
        coords,
        conn,
        [args.qx],
        frame="local",
        backend="jax" if args.backend == "jax" else "numpy",
    )

    fixed = ff.truss_node_dofs([0], "xyz")
    lateral = ff.truss_node_dofs(np.arange(1, coords.shape[0]), "yz")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))
    solver = _auto_solver(args.backend, args.solver)
    u, _info = ff.LinearSolver(method=solver).solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dirichlet, 0.0),
        dirichlet_mode="condense",
    )

    tip = coords.shape[0] - 1
    ux_tip = float(u[ff.truss_node_dofs([tip], "x")][0])
    ux_exact = args.qx * args.length**2 / (2.0 * args.E * args.A)
    rel_err = abs(ux_tip - ux_exact) / abs(ux_exact) if ux_exact != 0.0 else 0.0

    print(f"truss uniform load solved: backend={args.backend}, solver={solver}, nodes={coords.shape[0]}, elems={conn.shape[0]}")
    print(f"tip ux={ux_tip:.6e}, bar exact={ux_exact:.6e}, rel.err={rel_err:.3e}")
    print(f"total applied Fx={np.sum(F[ff.truss_node_dofs(np.arange(coords.shape[0]), 'x')]):.6e}")


if __name__ == "__main__":
    main()

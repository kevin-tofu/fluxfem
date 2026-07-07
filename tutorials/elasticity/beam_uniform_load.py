#!/usr/bin/env python3
"""
Euler-Bernoulli cantilever beam under a uniform distributed load.

The equivalent nodal load vector is assembled with ``assemble_beam_uniform_load``.
For a transverse load qz, the tip deflection is compared with

  w(L) = qz L^4 / (8 E Iy).
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
    p = argparse.ArgumentParser(description="Cantilever beam with uniform distributed load.")
    p.add_argument("--format", choices=("csr", "fluxsparse", "dense"), default="csr", help="Matrix assembly format.")
    p.add_argument("--solver", choices=("auto", "spsolve", "spsolve_jax"), default="auto", help="Linear solver backend.")
    p.add_argument("--n-elems", type=int, default=8, help="Number of beam elements.")
    p.add_argument("--length", type=float, default=2.0, help="Beam length.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--G", type=float, default=80.0e9, help="Shear modulus for torsion.")
    p.add_argument("--A", type=float, default=2.0e-3, help="Cross-sectional area.")
    p.add_argument("--Iy", type=float, default=8.0e-6, help="Second moment about local y.")
    p.add_argument("--Iz", type=float, default=5.0e-6, help="Second moment about local z.")
    p.add_argument("--J", type=float, default=1.0e-5, help="Torsion constant.")
    p.add_argument("--qz", type=float, default=-1000.0, help="Uniform load in global z per unit length.")
    return p.parse_args()


def _auto_solver(format: str, solver: str) -> str:
    if solver != "auto":
        return solver
    return "spsolve_jax" if format == "fluxsparse" else "spsolve"


def main():
    args = parse_args()

    coords, conn = ff.structured_beam_chain(n_elems=args.n_elems, length=args.length)
    section = ff.BeamSection(E=args.E, G=args.G, A=args.A, Iy=args.Iy, Iz=args.Iz, J=args.J)
    K = ff.assemble_beam_stiffness(coords, conn, section, format=args.format)
    F = ff.assemble_beam_uniform_load(
        coords,
        conn,
        [0.0, 0.0, args.qz],
        frame="global",
        array_backend="jax" if args.format == "fluxsparse" else "numpy",
    )

    fixed = ff.beam_node_dofs([0])
    solver = _auto_solver(args.format, args.solver)
    u, _info = ff.LinearSolver(method=solver).solve(
        K,
        F,
        dirichlet=ff.DirichletBC(fixed, 0.0),
        dirichlet_mode="condense",
    )

    tip = coords.shape[0] - 1
    uz_tip = float(u[ff.beam_node_dofs([tip], "uz")][0])
    uz_exact = args.qz * args.length**4 / (8.0 * args.E * args.Iy)
    rel_err = abs(uz_tip - uz_exact) / abs(uz_exact) if uz_exact != 0.0 else 0.0

    print(f"beam uniform load solved: format={args.format}, solver={solver}, nodes={coords.shape[0]}, elems={conn.shape[0]}")
    print(f"tip uz={uz_tip:.6e}, EB exact={uz_exact:.6e}, rel.err={rel_err:.3e}")
    print(f"total applied Fz={np.sum(F[ff.beam_node_dofs(np.arange(coords.shape[0]), 'uz')]):.6e}")


if __name__ == "__main__":
    main()

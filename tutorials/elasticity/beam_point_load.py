#!/usr/bin/env python3
"""
Euler-Bernoulli cantilever beam with tip force and tip moment.

The nodal RHS is assembled with ``assemble_beam_point_load``. For a transverse
tip force Pz and end moment My, the tip displacement follows the superposition

  uz(L) = Pz L^3 / (3 E Iy) - My L^2 / (2 E Iy).
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
    p = argparse.ArgumentParser(description="Cantilever beam with tip force and moment.")
    p.add_argument("--backend", choices=("jax", "scipy", "numpy"), default="jax", help="Matrix assembly backend.")
    p.add_argument("--solver", choices=("auto", "spsolve", "spsolve_jax"), default="auto", help="Linear solver backend.")
    p.add_argument("--n-elems", type=int, default=8, help="Number of beam elements.")
    p.add_argument("--length", type=float, default=2.0, help="Beam length.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--G", type=float, default=80.0e9, help="Shear modulus for torsion.")
    p.add_argument("--A", type=float, default=2.0e-3, help="Cross-sectional area.")
    p.add_argument("--Iy", type=float, default=8.0e-6, help="Second moment about local y.")
    p.add_argument("--Iz", type=float, default=5.0e-6, help="Second moment about local z.")
    p.add_argument("--J", type=float, default=1.0e-5, help="Torsion constant.")
    p.add_argument("--tip-force-z", type=float, default=-1000.0, help="Tip force in global z.")
    p.add_argument("--tip-moment-y", type=float, default=500.0, help="Tip moment about global y.")
    return p.parse_args()


def _auto_solver(backend: str, solver: str) -> str:
    if solver != "auto":
        return solver
    return "spsolve_jax" if backend == "jax" else "spsolve"


def main():
    args = parse_args()

    coords, conn = ff.structured_beam_chain(n_elems=args.n_elems, length=args.length)
    section = ff.BeamSection(E=args.E, G=args.G, A=args.A, Iy=args.Iy, Iz=args.Iz, J=args.J)
    K = ff.assemble_beam_stiffness(coords, conn, section, backend=args.backend)

    tip = coords.shape[0] - 1
    F = ff.assemble_beam_point_load(
        coords.shape[0],
        tip,
        force=(0.0, 0.0, args.tip_force_z),
        moment=(0.0, args.tip_moment_y, 0.0),
        backend="jax" if args.backend == "jax" else "numpy",
    )

    fixed = ff.beam_node_dofs([0])
    solver = _auto_solver(args.backend, args.solver)
    u, _info = ff.LinearSolver(method=solver).solve(
        K,
        F,
        dirichlet=ff.DirichletBC(fixed, 0.0),
        dirichlet_mode="condense",
    )

    uz_tip = float(u[ff.beam_node_dofs([tip], "uz")][0])
    ry_tip = float(u[ff.beam_node_dofs([tip], "ry")][0])
    uz_exact = (
        args.tip_force_z * args.length**3 / (3.0 * args.E * args.Iy)
        - args.tip_moment_y * args.length**2 / (2.0 * args.E * args.Iy)
    )
    ry_exact = (
        -args.tip_force_z * args.length**2 / (2.0 * args.E * args.Iy)
        + args.tip_moment_y * args.length / (args.E * args.Iy)
    )
    rel_err = abs(uz_tip - uz_exact) / abs(uz_exact) if uz_exact != 0.0 else 0.0

    print(f"beam point load solved: backend={args.backend}, solver={solver}, nodes={coords.shape[0]}, elems={conn.shape[0]}")
    print(f"tip uz={uz_tip:.6e}, EB exact={uz_exact:.6e}, rel.err={rel_err:.3e}")
    print(f"tip ry={ry_tip:.6e}, EB exact={ry_exact:.6e}")


if __name__ == "__main__":
    main()

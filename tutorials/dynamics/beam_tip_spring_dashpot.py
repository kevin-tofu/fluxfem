#!/usr/bin/env python3
"""
Euler-Bernoulli beam with a lumped tip spring/dashpot support.

This combines:
  - beam stiffness and mass matrices,
  - a grounded spring on the tip uz DOF,
  - a grounded dashpot on the tip uz DOF,
  - Newmark time integration.
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
    p = argparse.ArgumentParser(description="Beam tip spring/dashpot transient response.")
    p.add_argument("--n-elems", type=int, default=12, help="Number of beam elements.")
    p.add_argument("--length", type=float, default=2.0, help="Beam length.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--G", type=float, default=80.0e9, help="Shear modulus for torsion.")
    p.add_argument("--A", type=float, default=2.0e-3, help="Cross-sectional area.")
    p.add_argument("--Iy", type=float, default=8.0e-6, help="Second moment about local y.")
    p.add_argument("--Iz", type=float, default=5.0e-6, help="Second moment about local z.")
    p.add_argument("--J", type=float, default=1.0e-5, help="Torsion constant.")
    p.add_argument("--rho", type=float, default=7800.0, help="Density.")
    p.add_argument("--tip-spring", type=float, default=2.0e6, help="Ground spring on tip uz.")
    p.add_argument("--tip-dashpot", type=float, default=2.0e4, help="Ground dashpot on tip uz.")
    p.add_argument("--tip-load-z", type=float, default=-1000.0, help="Constant tip load in z.")
    p.add_argument("--dt", type=float, default=1.0e-4, help="Time step.")
    p.add_argument("--steps", type=int, default=1200, help="Number of time steps.")
    return p.parse_args()


def main():
    args = parse_args()

    coords, conn = ff.structured_beam_chain(n_elems=args.n_elems, length=args.length)
    section = ff.BeamSection(
        E=args.E,
        G=args.G,
        A=args.A,
        Iy=args.Iy,
        Iz=args.Iz,
        J=args.J,
        rho=args.rho,
    )

    K_beam = ff.assemble_beam_stiffness(coords, conn, section).toarray()
    M = ff.assemble_beam_mass(coords, conn, section).toarray()

    n_dofs = ff.BEAM_DOF_PER_NODE * coords.shape[0]
    tip_node = coords.shape[0] - 1
    tip_uz = ff.beam_node_dofs([tip_node], "uz")
    K = K_beam + ff.assemble_dof_spring(n_dofs, tip_uz, args.tip_spring).toarray()
    C = ff.assemble_dof_dashpot(n_dofs, tip_uz, args.tip_dashpot).toarray()

    force = np.zeros(n_dofs, dtype=float)
    force[tip_uz] = args.tip_load_z
    fixed = ff.beam_node_dofs([0])
    bc = ff.DirichletBC(fixed, 0.0)

    u_static, _ = ff.LinearSolver(method="spsolve").solve(
        K,
        force,
        dirichlet=bc,
        dirichlet_mode="condense",
    )

    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=np.zeros(n_dofs, dtype=float),
        v0=np.zeros(n_dofs, dtype=float),
        dt=args.dt,
        n_steps=args.steps,
        force=force,
        dirichlet=bc,
    )

    tip_history = out.u[:, tip_uz[0]]
    final_err = abs(tip_history[-1] - u_static[tip_uz[0]]) / max(abs(u_static[tip_uz[0]]), 1.0e-30)

    print(f"beam tip spring/dashpot solved: nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={n_dofs}")
    print(f"tip uz static={u_static[tip_uz[0]]:.6e}")
    print(f"tip uz final={tip_history[-1]:.6e}, max_abs={np.max(np.abs(tip_history)):.6e}")
    print(f"relative final/static error={final_err:.3e}")


if __name__ == "__main__":
    main()

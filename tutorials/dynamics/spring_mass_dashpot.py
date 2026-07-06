#!/usr/bin/env python3
"""
Single-DOF spring-mass-dashpot oscillator using lumped element helpers.

The system is

  m u_ddot + c u_dot + k u = 0

and is integrated with the existing Newmark solver.
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
    p = argparse.ArgumentParser(description="Spring-mass-dashpot oscillator.")
    p.add_argument("--mass", type=float, default=1.0, help="Mass m.")
    p.add_argument("--stiffness", type=float, default=4.0, help="Spring stiffness k.")
    p.add_argument("--damping", type=float, default=0.4, help="Dashpot damping c.")
    p.add_argument("--u0", type=float, default=1.0, help="Initial displacement.")
    p.add_argument("--v0", type=float, default=0.0, help="Initial velocity.")
    p.add_argument("--dt", type=float, default=0.01, help="Time step.")
    p.add_argument("--steps", type=int, default=500, help="Number of time steps.")
    return p.parse_args()


def main():
    args = parse_args()

    M = np.array([[args.mass]], dtype=float)
    K = ff.assemble_dof_spring(1, [0], args.stiffness)
    C = ff.assemble_dof_dashpot(1, [0], args.damping)

    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=np.array([args.u0], dtype=float),
        v0=np.array([args.v0], dtype=float),
        dt=args.dt,
        n_steps=args.steps,
    )

    omega_n = np.sqrt(args.stiffness / args.mass)
    zeta = args.damping / (2.0 * np.sqrt(args.stiffness * args.mass))
    energy = 0.5 * args.mass * out.v[:, 0] ** 2 + 0.5 * args.stiffness * out.u[:, 0] ** 2

    print(f"spring-mass-dashpot solved: steps={args.steps}, dt={args.dt}")
    print(f"omega_n={omega_n:.6e}, damping_ratio={zeta:.6e}")
    print(f"u_final={out.u[-1, 0]:.6e}, v_final={out.v[-1, 0]:.6e}")
    print(f"energy_initial={energy[0]:.6e}, energy_final={energy[-1]:.6e}")


if __name__ == "__main__":
    main()

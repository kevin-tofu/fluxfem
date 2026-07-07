#!/usr/bin/env python3
"""
Modal check for a 3D Euler-Bernoulli beam cantilever.

The first bending frequency is compared with

  omega_1 = beta_1^2 * sqrt(E I / (rho A L^4)), beta_1 = 1.875104...
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import scipy.linalg as la

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def parse_args():
    p = argparse.ArgumentParser(description="Cantilever beam modal check.")
    p.add_argument("--n-elems", type=int, default=12, help="Number of beam elements.")
    p.add_argument("--length", type=float, default=2.0, help="Beam length.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--G", type=float, default=80.0e9, help="Shear modulus for torsion.")
    p.add_argument("--A", type=float, default=2.0e-3, help="Cross-sectional area.")
    p.add_argument("--I", type=float, default=8.0e-6, help="Bending second moment for both local y/z.")
    p.add_argument("--J", type=float, default=1.0e-5, help="Torsion constant.")
    p.add_argument("--rho", type=float, default=7800.0, help="Density.")
    p.add_argument("--n-report", type=int, default=4, help="Number of frequencies to print.")
    return p.parse_args()


def main():
    args = parse_args()

    coords, conn = ff.structured_beam_chain(n_elems=args.n_elems, length=args.length)
    section = ff.BeamSection(
        E=args.E,
        G=args.G,
        A=args.A,
        Iy=args.I,
        Iz=args.I,
        J=args.J,
        rho=args.rho,
    )
    K = ff.assemble_beam_stiffness(coords, conn, section).toarray()
    M = ff.assemble_beam_mass(coords, conn, section).toarray()

    fixed = ff.beam_node_dofs([0])
    free = ff.free_dofs(K.shape[0], fixed)
    w2 = la.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True)
    omegas = np.sqrt(w2[w2 > 1.0e-8])

    beta1 = 1.875104068711961
    omega_exact = beta1**2 * np.sqrt(args.E * args.I / (args.rho * args.A * args.length**4))
    rel_err = abs(float(omegas[0]) - omega_exact) / omega_exact

    print(f"beam modal solved: nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={K.shape[0]}")
    print(f"omega_1={omegas[0]:.6e}, EB exact={omega_exact:.6e}, rel.err={rel_err:.3e}")
    print("first omegas:", " ".join(f"{omega:.6e}" for omega in omegas[: args.n_report]))


if __name__ == "__main__":
    main()

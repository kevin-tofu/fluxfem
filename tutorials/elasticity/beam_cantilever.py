#!/usr/bin/env python3
"""
3D Euler-Bernoulli beam cantilever using the dedicated frame-element assembler.

Each beam node has 6 DOFs:
  [ux, uy, uz, rx, ry, rz]

The example fixes the left end and applies a transverse tip load in z. The tip
deflection is compared with the Euler-Bernoulli cantilever result

  w(L) = P L^3 / (3 E Iy).
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
    p = argparse.ArgumentParser(description="Cantilever beam with 3D frame elements.")
    p.add_argument("--n-elems", type=int, default=8, help="Number of beam elements.")
    p.add_argument("--length", type=float, default=2.0, help="Beam length.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--G", type=float, default=80.0e9, help="Shear modulus for torsion.")
    p.add_argument("--A", type=float, default=2.0e-3, help="Cross-sectional area.")
    p.add_argument("--Iy", type=float, default=8.0e-6, help="Second moment about local y.")
    p.add_argument("--Iz", type=float, default=5.0e-6, help="Second moment about local z.")
    p.add_argument("--J", type=float, default=1.0e-5, help="Torsion constant.")
    p.add_argument("--rho", type=float, default=7800.0, help="Density for optional mass assembly.")
    p.add_argument("--tip-load-z", type=float, default=-1000.0, help="Tip load in global z.")
    return p.parse_args()


def main():
    args = parse_args()

    coords, conn = ff.structured_beam_chain(n_elems=args.n_elems, length=args.length)
    section = ff.BeamSection(E=args.E, G=args.G, A=args.A, Iy=args.Iy, Iz=args.Iz, J=args.J, rho=args.rho)
    K = ff.assemble_beam_stiffness(coords, conn, section)
    M = ff.assemble_beam_mass(coords, conn, section)

    n_dofs = ff.BEAM_DOF_PER_NODE * coords.shape[0]
    F = np.zeros(n_dofs, dtype=float)
    tip_node = coords.shape[0] - 1
    F[ff.beam_node_dofs([tip_node], "uz")] = args.tip_load_z

    fixed = ff.beam_node_dofs([0])
    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(fixed, 0.0),
        dirichlet_mode="condense",
    )

    uz_tip = float(u[ff.beam_node_dofs([tip_node], "uz")][0])
    ry_tip = float(u[ff.beam_node_dofs([tip_node], "ry")][0])
    uz_exact = args.tip_load_z * args.length**3 / (3.0 * args.E * args.Iy)
    ry_exact = -args.tip_load_z * args.length**2 / (2.0 * args.E * args.Iy)
    rel_err = abs(uz_tip - uz_exact) / abs(uz_exact) if uz_exact != 0.0 else 0.0

    print(f"beam solved: nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={n_dofs}")
    print(f"tip uz={uz_tip:.6e}, EB exact={uz_exact:.6e}, rel.err={rel_err:.3e}")
    print(f"tip ry={ry_tip:.6e}, EB exact={ry_exact:.6e}")
    print(f"stiffness nnz={K.nnz}, mass nnz={M.nnz}")


if __name__ == "__main__":
    main()

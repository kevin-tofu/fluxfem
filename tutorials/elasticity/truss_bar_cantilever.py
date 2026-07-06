#!/usr/bin/env python3
"""
3D truss/bar cantilever using the dedicated axial-element assembler.

Each node has translational DOFs [ux, uy, uz]. The example fixes the left end,
keeps the chain lateral DOFs fixed to model a 1D bar, and applies an axial tip
load. The tip displacement is compared with

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
    p = argparse.ArgumentParser(description="Cantilever bar with 3D truss elements.")
    p.add_argument("--n-elems", type=int, default=8, help="Number of truss/bar elements.")
    p.add_argument("--length", type=float, default=3.0, help="Bar length.")
    p.add_argument("--E", type=float, default=70.0e9, help="Young's modulus.")
    p.add_argument("--A", type=float, default=1.5e-3, help="Cross-sectional area.")
    p.add_argument("--rho", type=float, default=2700.0, help="Density for optional mass assembly.")
    p.add_argument("--tip-load-x", type=float, default=1200.0, help="Tip load in global x.")
    return p.parse_args()


def main():
    args = parse_args()

    coords, conn = ff.structured_truss_chain(n_elems=args.n_elems, length=args.length)
    section = ff.TrussSection(E=args.E, A=args.A, rho=args.rho)
    K = ff.assemble_truss_stiffness(coords, conn, section)
    M = ff.assemble_truss_mass(coords, conn, section)

    n_dofs = ff.TRUSS_DOF_PER_NODE * coords.shape[0]
    F = np.zeros(n_dofs, dtype=float)
    tip_node = coords.shape[0] - 1
    F[ff.truss_node_dofs([tip_node], "x")] = args.tip_load_x

    fixed = ff.truss_node_dofs([0], "xyz")
    lateral = ff.truss_node_dofs(np.arange(1, coords.shape[0]), "yz")
    dirichlet = np.unique(np.concatenate([fixed, lateral]))

    u, _info = ff.LinearSolver(method="spsolve").solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dirichlet, 0.0),
        dirichlet_mode="condense",
    )

    ux_tip = float(u[ff.truss_node_dofs([tip_node], "x")][0])
    ux_exact = args.tip_load_x * args.length / (args.E * args.A)
    rel_err = abs(ux_tip - ux_exact) / abs(ux_exact) if ux_exact != 0.0 else 0.0

    print(f"truss/bar solved: nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={n_dofs}")
    print(f"tip ux={ux_tip:.6e}, bar exact={ux_exact:.6e}, rel.err={rel_err:.3e}")
    print(f"stiffness nnz={K.nnz}, mass nnz={M.nnz}")


if __name__ == "__main__":
    main()

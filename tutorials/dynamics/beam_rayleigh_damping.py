#!/usr/bin/env python3
"""
Free vibration of an Euler-Bernoulli beam with Rayleigh damping.

The Rayleigh coefficients are chosen to match a target damping ratio at the
first two bending frequencies, then the damped response is integrated with
Newmark.
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
    p = argparse.ArgumentParser(description="Beam free vibration with Rayleigh damping.")
    p.add_argument("--n-elems", type=int, default=12, help="Number of beam elements.")
    p.add_argument("--length", type=float, default=2.0, help="Beam length.")
    p.add_argument("--E", type=float, default=210.0e9, help="Young's modulus.")
    p.add_argument("--G", type=float, default=80.0e9, help="Shear modulus for torsion.")
    p.add_argument("--A", type=float, default=2.0e-3, help="Cross-sectional area.")
    p.add_argument("--I", type=float, default=8.0e-6, help="Bending second moment for both local y/z.")
    p.add_argument("--J", type=float, default=1.0e-5, help="Torsion constant.")
    p.add_argument("--rho", type=float, default=7800.0, help="Density.")
    p.add_argument("--zeta", type=float, default=0.02, help="Target damping ratio at first two modes.")
    p.add_argument("--dt", type=float, default=1.0e-4, help="Time step.")
    p.add_argument("--steps", type=int, default=1000, help="Number of time steps.")
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
    w2, modes = la.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    positive = np.flatnonzero(w2 > 1.0e-8)
    omega1 = float(np.sqrt(w2[positive[0]]))
    omega2 = next(float(np.sqrt(w2[i])) for i in positive[1:] if np.sqrt(w2[i]) > omega1 * (1.0 + 1.0e-6))

    alpha, beta = ff.rayleigh_coefficients_from_modal_damping(omega1, args.zeta, omega2)
    C = ff.assemble_rayleigh_damping(M, K, alpha=alpha, beta=beta)

    u0 = np.zeros(K.shape[0], dtype=float)
    first_mode = np.asarray(modes[:, positive[0]], dtype=float)
    first_mode /= np.max(np.abs(first_mode))
    u0[free] = 1.0e-3 * first_mode

    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=u0,
        v0=np.zeros_like(u0),
        dt=args.dt,
        n_steps=args.steps,
        dirichlet=ff.DirichletBC(fixed, 0.0),
    )

    energy = np.einsum("ni,ij,nj->n", out.v, M, out.v)
    energy += np.einsum("ni,ij,nj->n", out.u, K, out.u)
    energy *= 0.5

    print(f"beam Rayleigh damping solved: nodes={coords.shape[0]}, elems={conn.shape[0]}, dofs={K.shape[0]}")
    print(f"omega1={omega1:.6e}, omega2={omega2:.6e}")
    print(f"alpha={alpha:.6e}, beta={beta:.6e}, zeta={args.zeta:.6e}")
    print(f"energy_initial={energy[0]:.6e}, energy_final={energy[-1]:.6e}")


if __name__ == "__main__":
    main()

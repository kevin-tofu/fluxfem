#!/usr/bin/env python3
"""
Steady-state heat conduction in a 3D bar with uniform volumetric heat source.

Model:
  -k * ΔT = q  in Ω
  T = 0 on x=0 and x=L

Analytical 1D solution (uniform in y,z):
  T(x) = q/(2k) * x * (L - x)
  T_max at x=L/2: q * L^2 / (8k)
"""

from __future__ import annotations

import argparse
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff  # noqa: E402
import fluxfem.helpers_wf as h_wf  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Linear heat conduction in a 3D bar.")
    p.add_argument("--nx", type=int, default=12, help="Elements along x (length).")
    p.add_argument("--ny", type=int, default=2, help="Elements along y (thickness).")
    p.add_argument("--nz", type=int, default=2, help="Elements along z (width).")
    p.add_argument("--lx", type=float, default=1.0, help="Bar length.")
    p.add_argument("--ly", type=float, default=0.1, help="Bar thickness.")
    p.add_argument("--lz", type=float, default=0.1, help="Bar width.")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--kappa", type=float, default=1.0, help="Thermal conductivity.")
    p.add_argument("--source", type=float, default=1.0, help="Uniform heat source q.")
    return p.parse_args()


def main():
    args = parse_args()

    mesh = ff.StructuredHexBox(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
    ).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=args.intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    bilinear = ff.compile_bilinear(
        lambda v, u, p: p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
    )
    linear = ff.compile_linear(
        lambda v, p: (v * p.q) * h_wf.dOmega()
    )

    K = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=V, trial=U),
        bilinear,
        ff.Params(kappa=args.kappa),
    )
    F = ff.assemble_linear_form(
        ff.LinearSpaces(test=V),
        linear,
        ff.Params(q=args.source),
    )

    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())

    dir_left = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="x",
    )
    dir_right = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmax, atol=1e-8),
        components="x",
    )
    dir_dofs = np.unique(np.concatenate([dir_left.dofs, dir_right.dofs]))

    solver = ff.LinearSolver(method="spsolve")
    u, _ = solver.solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dir_dofs, None),
        dirichlet_mode="condense",
    )

    u_nodes = np.asarray(u).reshape(-1)
    x_coords = coords[:, 0]
    mid_x = 0.5 * (xmin + xmax)
    mid_mask = np.isclose(x_coords, mid_x, atol=1e-8)
    if not np.any(mid_mask):
        mid_mask = np.isclose(x_coords, mid_x, atol=(xmax - xmin) / args.nx)

    u_mid = float(np.max(u_nodes[mid_mask])) if np.any(mid_mask) else float(np.max(u_nodes))
    u_theory = args.source * args.lx * args.lx / (8.0 * args.kappa)
    rel_err = abs(u_mid - u_theory) / abs(u_theory) if u_theory != 0.0 else 0.0

    print(
        f"heat solved: dofs={space.n_dofs}, "
        f"T_max≈{u_mid:.6e} (q={args.source}, k={args.kappa})"
    )
    print(f"[theory] T_max≈{u_theory:.6e} (rel.err={rel_err:.3e})")


if __name__ == "__main__":
    main()

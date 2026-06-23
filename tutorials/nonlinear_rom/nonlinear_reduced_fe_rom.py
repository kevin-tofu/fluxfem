#!/usr/bin/env python3
"""Nonlinear FE Galerkin ROM with full residual projection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)
TUTORIALS_ROOT = Path(__file__).resolve().parents[1]
if str(TUTORIALS_ROOT) not in sys.path:
    sys.path.insert(0, str(TUTORIALS_ROOT))

from common.basis import DenseBasis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--E", type=float, default=250.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--force", type=float, default=-0.001)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = jnp.float64
    mesh = ff.StructuredHexBox(nx=args.nx, ny=args.ny, nz=args.nz, lx=1.0, ly=0.25, lz=0.25).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())
    ymax = float(coords[:, 1].max())
    zmax = float(coords[:, 2].max())
    tool_node = int(
        np.flatnonzero(
            np.isclose(coords[:, 0], xmax)
            & np.isclose(coords[:, 1], ymax)
            & np.isclose(coords[:, 2], zmax)
        )[0]
    )
    force = jnp.zeros(space.n_dofs, dtype=dtype).at[tool_node * 3 + 1].set(args.force)
    lam, mu = ff.lame_parameters(args.E, args.nu)
    params = {"lam": lam, "mu": mu}
    dirichlet = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin),
        components="xyz",
    )

    full = ff.NonlinearAnalysis(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params=params,
        base_external_vector=force,
        dirichlet=dirichlet,
        dtype=dtype,
    )
    full_u, history = ff.NewtonSolveRunner(
        full,
        ff.NewtonLoopConfig(tol=args.tol, atol=args.tol, maxiter=args.maxiter, linear_solver="spsolve"),
    ).run(u0=jnp.zeros(space.n_dofs, dtype=dtype), newton_callback=lambda _cb: None)

    complete_basis = DenseBasis(jnp.eye(space.n_dofs, dtype=dtype))
    rom = ff.NonlinearReducedFEModel(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params=params,
        basis=complete_basis,
        external_vector=force,
    )
    q, info = rom.as_problem("body").solve(
        jnp.zeros(complete_basis.n_reduced, dtype=dtype),
        fixed_dofs=dirichlet.dofs,
        fixed_values=jnp.zeros_like(jnp.asarray(dirichlet.dofs, dtype=dtype)),
        tol=args.tol,
        atol=args.tol,
        maxiter=args.maxiter,
    )
    rom_u = rom.expand(q)
    print("nonlinear reduced FE ROM demo")
    print(f"full_converged: {history[-1].info.converged}")
    print(f"full_rom_u_inf: {float(jnp.linalg.norm(full_u - rom_u, ord=jnp.inf)):.6e}")
    print(f"rom_converged: {info.converged}")
    print(f"rom_iters: {info.iters}")
    if not history[-1].info.converged or not info.converged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

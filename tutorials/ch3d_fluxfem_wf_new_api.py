#!/usr/bin/env python3
"""
Cahn-Hilliard 3D (semi-implicit) example using the explicit mixed bindings API.

This mirrors `tutorials/ch3d_fluxfem_wf.py`, but keeps the short local trial/test
arguments and only uses explicit bindings where they help:
  - field names distinct from space keys
  - explicit residual labels via bind_mixed_residual(...)
  - context access through ctx.bindings[...] for named mixed fields
  - context access through ctx.spaces[...] when selecting a specific mixed field
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def write_pvd_collection(path: str, files: List[str], times: List[float]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="ascii") as io:
        io.write('<?xml version="1.0"?>\n')
        io.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        io.write("  <Collection>\n")
        for f, t in zip(files, times):
            io.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{f}"/>\n')
        io.write("  </Collection>\n")
        io.write("</VTKFile>\n")


def parse_args():
    p = argparse.ArgumentParser(description="Cahn-Hilliard 3D (new mixed API).")
    p.add_argument("--nx", type=int, default=16, help="Elements along x.")
    p.add_argument("--ny", type=int, default=16, help="Elements along y.")
    p.add_argument("--nz", type=int, default=16, help="Elements along z.")
    p.add_argument("--lx", type=float, default=1.0, help="Domain length.")
    p.add_argument("--ly", type=float, default=1.0, help="Domain height.")
    p.add_argument("--lz", type=float, default=1.0, help="Domain width.")
    p.add_argument("--order", type=int, default=1, help="Polynomial order hint.")
    p.add_argument("--kappa", type=float, default=1e-3, help="Gradient-energy coefficient.")
    p.add_argument("--dt", type=float, default=1e-4, help="Time step.")
    p.add_argument("--steps", type=int, default=200, help="Number of time steps.")
    p.add_argument("--plot-interval", type=int, default=1, help="Write every N steps.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("result", "tutorials", "ch_output_3d_fluxfem_wf_new_api"),
        help="Output directory.",
    )
    return p.parse_args()


def run_ch_3d():
    args = parse_args()
    phase_unknown_space = "C"
    phase_test_space = "W"
    chemical_unknown_space = "MU"
    chemical_test_space = "Q"

    mesh = ff.StructuredHexBox(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
    ).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2 * args.order)
    mixed = ff.MixedSpaces(
        {
            "phase": ff.ResidualSpaces(
                test=ff.NamedSpace(phase_test_space, space),
                unknown=ff.NamedSpace(phase_unknown_space, space),
            ),
            "chem_potential": ff.ResidualSpaces(
                test=ff.NamedSpace(chemical_test_space, space),
                unknown=ff.NamedSpace(chemical_unknown_space, space),
            ),
        }
    ).to_fe_space()

    rng = np.random.default_rng(0)
    mean_c0 = 0.0
    c_vec = mean_c0 + 0.01 * rng.standard_normal(space.n_dofs)
    c_vec = c_vec - (np.mean(c_vec) - mean_c0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    vtufiles: List[str] = []
    times: List[float] = []

    ff.write_vtu(mesh, os.path.join(out_dir, "ch_0000.vtu"), point_data={"c": c_vec})
    vtufiles.append("ch_0000.vtu")
    times.append(0.0)

    def res_c(v, u, p):
        mu = ff.unknown_ref("chemical_potential", space=chemical_unknown_space)
        return (v * (u.val - p.c_old) + p.dt * h_wf.gaction(v, h_wf.grad(mu))) * h_wf.dOmega()

    def res_mu(q, u, p):
        c = ff.unknown_ref("concentration", space=phase_unknown_space)
        return (
            q * (u.val - (p.c_old**3 - p.c_old))
            - p.kappa * h_wf.gaction(q, h_wf.grad(c))
        ) * h_wf.dOmega()

    residuals = ff.make_mixed_residuals(
        phase_balance=ff.bind_mixed_residual("phase", res_c, space=phase_unknown_space),
        chemical_equilibrium=ff.bind_mixed_residual("chem_potential", res_mu, space=chemical_unknown_space),
    )

    solver = ff.LinearSolver(method="spsolve")
    pattern = mixed.get_sparsity_pattern(with_idx=True)

    for step in range(1, args.steps + 1):
        c_old_elems = jnp.asarray(c_vec)[space.elem_dofs]

        def params_fn(ctx, c_old_elems=c_old_elems, kappa=args.kappa, dt=args.dt):
            c_old_elem = c_old_elems[ctx.elem_id]
            # This is the place where the explicit mixed-space lookup is useful:
            # params_fn receives the full mixed context, not one local residual view.
            c_old_q = ctx.spaces[phase_unknown_space].trial.eval(c_old_elem)
            return {"kappa": kappa, "dt": dt, "c_old": c_old_q}

        u0 = jnp.zeros(mixed.n_dofs)
        problem = ff.MixedProblem(mixed, residuals, params=params_fn, pattern=pattern)
        K = problem.assemble_jacobian(u0)
        R0 = problem.assemble_residual(u0)
        b = -R0

        u_new, _info = solver.solve(K, b)
        solution_fields = mixed.unpack_fields(u_new)
        c_vec = np.asarray(solution_fields["phase"])

        if (step % args.plot_interval == 0) or (step == args.steps):
            fname = f"ch_{step:04d}.vtu"
            ff.write_vtu(mesh, os.path.join(out_dir, fname), point_data={"c": c_vec})
            vtufiles.append(fname)
            times.append(step * args.dt)
            print(f"[step {step}] wrote {fname}")

    write_pvd_collection(os.path.join(out_dir, "ch_series.pvd"), vtufiles, times)
    print(f"VTK series written to {os.path.join(out_dir, 'ch_series.pvd')}")


if __name__ == "__main__":
    run_ch_3d()

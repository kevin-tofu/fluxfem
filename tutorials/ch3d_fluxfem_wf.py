#!/usr/bin/env python3
"""
Cahn-Hilliard 3D (semi-implicit) example using FluxFEM mixed weak-form DSL.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import jax.numpy as jnp

import fluxfem as ff
from fluxfem.core.mixed_space import MixedFESpace
import fluxfem.helpers_wf as h_wf
from fluxfem.mixed_weakform import (
    MixedResidualForm,
    assemble_mixed_jacobian_wf,
    assemble_mixed_residual_wf,
)


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


def run_ch_3d():
    kappa = 1e-3
    dt = 1e-4
    n_steps = 200
    plot_interval = 1
    order = 1

    mesh = ff.StructuredHexBox(nx=16, ny=16, nz=16, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2 * order)
    mixed = MixedFESpace({"c": space, "mu": space})

    rng = np.random.default_rng(0)
    mean_c0 = 0.0
    c_vec = mean_c0 + 0.01 * rng.standard_normal(space.n_dofs)
    c_vec = c_vec - (np.mean(c_vec) - mean_c0)

    out_dir = os.path.join("result", "tutorials", "ch_output_3d_fluxfem_wf")
    os.makedirs(out_dir, exist_ok=True)
    vtufiles: List[str] = []
    times: List[float] = []

    ff.write_vtu(mesh, os.path.join(out_dir, "ch_0000.vtu"), point_data={"c": c_vec})
    vtufiles.append("ch_0000.vtu")
    times.append(0.0)

    def res_c(v, u, p):
        mu = ff.unknown_ref("mu")
        return (v * (u.val - p.c_old) + p.dt * h_wf.gaction(v, h_wf.grad(mu))) * h_wf.dOmega()

    def res_mu(q, u, p):
        c = ff.unknown_ref("c")
        return (q * (u.val - (p.c_old**3 - p.c_old)) - p.kappa * h_wf.gaction(q, h_wf.grad(c))) * h_wf.dOmega()

    mixed_form = MixedResidualForm({"c": res_c, "mu": res_mu})

    solver = ff.LinearSolver(method="spsolve")
    pattern = mixed.get_sparsity_pattern(with_idx=True)

    for step in range(1, n_steps + 1):
        c_old_elems = jnp.asarray(c_vec)[space.elem_dofs]

        def params_fn(ctx, c_old_elems=c_old_elems, kappa=kappa, dt=dt):
            c_old_elem = c_old_elems[ctx.elem_id]
            c_old_q = ctx.fields["c"].trial.eval(c_old_elem)
            return {"kappa": kappa, "dt": dt, "c_old": c_old_q}

        u0 = jnp.zeros(mixed.n_dofs)
        K = assemble_mixed_jacobian_wf(
            mixed,
            mixed_form,
            u0,
            params_fn,
            pattern=pattern,
            return_flux_matrix=True,
        )
        R0 = assemble_mixed_residual_wf(mixed, mixed_form, u0, params_fn)
        b = -R0

        u_new, _info = solver.solve(K, b)
        fields = mixed.unpack_fields(u_new)
        c_vec = np.asarray(fields["c"])

        if (step % plot_interval == 0) or (step == n_steps):
            fname = f"ch_{step:04d}.vtu"
            ff.write_vtu(mesh, os.path.join(out_dir, fname), point_data={"c": c_vec})
            vtufiles.append(fname)
            times.append(step * dt)
            print(f"[step {step}] wrote {fname}")

    write_pvd_collection(os.path.join(out_dir, "ch_series.pvd"), vtufiles, times)
    print(f"VTK series written to {os.path.join(out_dir, 'ch_series.pvd')}")


if __name__ == "__main__":
    run_ch_3d()

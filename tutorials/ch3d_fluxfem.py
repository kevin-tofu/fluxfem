#!/usr/bin/env python3
"""
Cahn-Hilliard 3D (semi-implicit) example using FluxFEM mixed assembly.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import jax.numpy as jnp

import fluxfem as ff
from fluxfem.core.mixed_space import MixedFESpace


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


def ch_residual(ctx: ff.MixedFormContext, u_elem: dict, params) -> dict[str, jnp.ndarray]:
    c_elem = u_elem["c"]
    mu_elem = u_elem["mu"]
    c_old_elem = params["c_old_elems"][ctx.elem_id]

    c_field = ctx.fields["c"].trial
    mu_field = ctx.fields["mu"].trial
    v = ctx.fields["c"].test
    q = ctx.fields["mu"].test

    c_q = c_field.eval(c_elem)
    mu_q = mu_field.eval(mu_elem)
    c_old_q = c_field.eval(c_old_elem)

    grad_mu = mu_field.grad(mu_elem)
    grad_c = c_field.grad(c_elem)

    res_c = v.N * (c_q - c_old_q)
    res_c = res_c + params["dt"] * jnp.einsum("qj,qaj->qa", grad_mu, v.gradN)

    res_mu = q.N * (mu_q - (c_old_q**3 - c_old_q))
    res_mu = res_mu - params["kappa"] * jnp.einsum("qj,qaj->qa", grad_c, q.gradN)

    return {"c": res_c, "mu": res_mu}


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

    out_dir = os.path.join("result", "tutorials", "ch_output_3d_fluxfem")
    os.makedirs(out_dir, exist_ok=True)
    vtufiles: List[str] = []
    times: List[float] = []

    ff.write_vtu(mesh, os.path.join(out_dir, "ch_0000.vtu"), point_data={"c": c_vec})
    vtufiles.append("ch_0000.vtu")
    times.append(0.0)

    solver = ff.LinearSolver(method="spsolve")
    pattern = mixed.get_sparsity_pattern(with_idx=True)

    for step in range(1, n_steps + 1):
        c_old_elems = jnp.asarray(c_vec)[space.elem_dofs]
        params = {"kappa": kappa, "dt": dt, "c_old_elems": c_old_elems}

        u0 = jnp.zeros(mixed.n_dofs)
        K = mixed.assemble_jacobian(ch_residual, u0, params, pattern=pattern)
        R0 = mixed.assemble_residual(ch_residual, u0, params)
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

#!/usr/bin/env python3
"""
3D solid bar transient dynamics demo (axial-dominant) using Newmark-beta.

Workflow:
- Build a thin 3D hex bar.
- Assemble K (linear elasticity) and M (consistent mass).
- Fix x=0 face (all components).
- Compute first free-vibration mode from K_ff phi = w^2 M_ff phi.
- Use the first mode as initial displacement, then run Newmark time integration.
- Write VTU snapshots and a PVD collection for visualization.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import scipy.linalg as la

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D bar transient dynamics demo (Newmark-beta).")
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--nz", type=int, default=2)
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--ly", type=float, default=0.1)
    p.add_argument("--lz", type=float, default=0.1)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--E", type=float, default=200.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--rho", type=float, default=1.0)
    p.set_defaults(axial_only=True)
    p.add_argument(
        "--axial-only",
        dest="axial_only",
        action="store_true",
        help="Constrain uy=uz at all nodes and keep only axial dynamics (default).",
    )
    p.add_argument(
        "--full-3d",
        dest="axial_only",
        action="store_false",
        help="Disable axial-only constraint and run unconstrained 3D vibration.",
    )
    p.add_argument("--amp", type=float, default=1.0e-4, help="Initial modal displacement amplitude.")
    p.add_argument("--steps-per-period", type=int, default=240)
    p.add_argument("--periods", type=float, default=2.0)
    p.add_argument("--write-every", type=int, default=20)
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "results" / "dynamic_bar_3d"))
    return p.parse_args()


def write_pvd_collection(path: str, files: list[str], times: list[float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as io:
        io.write('<?xml version="1.0"?>\n')
        io.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        io.write("  <Collection>\n")
        for f, t in zip(files, times):
            io.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{f}"/>\n')
        io.write("  </Collection>\n")
        io.write("</VTKFile>\n")


def main() -> None:
    args = parse_args()

    mesh = ff.StructuredHexBox(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
    ).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=args.intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    D = ff.isotropic_3d_D(args.E, args.nu)
    K = np.asarray(
        ff.assemble_bilinear_form(
            ff.BilinearSpaces(test=V, trial=U),
            ff.linear_elasticity_form,
            D,
        ).to_dense()
    )
    M = args.rho * np.asarray(space.assemble_mass_matrix().to_dense())
    C = np.zeros_like(K)

    xmin = float(np.min(np.asarray(mesh.coords)[:, 0]))
    left_xyz = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1.0e-10),
        components=[0, 1, 2],
        dof_per_node=3,
    )
    left_x = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1.0e-10),
        components=[0],
        dof_per_node=3,
    )
    if args.axial_only:
        nodes = np.arange(mesh.n_nodes, dtype=int)
        yz = np.concatenate([3 * nodes + 1, 3 * nodes + 2])
        dirichlet_dofs = np.unique(np.concatenate([left_x, yz]))
    else:
        dirichlet_dofs = np.asarray(left_xyz, dtype=int)
    free = ff.free_dofs(space.n_dofs, dirichlet_dofs)

    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]

    w2, vecs = la.eigh(K_ff, M_ff)
    i0 = int(np.argmax(w2 > 1.0e-12))
    omega1 = float(np.sqrt(w2[i0]))
    phi1 = np.asarray(vecs[:, i0], dtype=float)
    phi1 /= np.max(np.abs(phi1))

    constrained_modulus = args.E * (1.0 - args.nu) / ((1.0 + args.nu) * (1.0 - 2.0 * args.nu))
    c_rod = np.sqrt((constrained_modulus if args.axial_only else args.E) / args.rho)
    omega1_theory = 0.5 * np.pi * c_rod / args.lx

    T1 = 2.0 * np.pi / omega1
    dt = T1 / float(args.steps_per_period)
    n_steps = int(round(args.periods * args.steps_per_period))

    u0 = np.zeros(space.n_dofs, dtype=float)
    v0 = np.zeros(space.n_dofs, dtype=float)
    u0[free] = args.amp * phi1

    out = ff.newmark_solve_linear(
        M,
        C,
        K,
        u0=u0,
        v0=v0,
        dt=dt,
        n_steps=n_steps,
        dirichlet=(dirichlet_dofs, np.zeros_like(dirichlet_dofs, dtype=float)),
    )

    # modal coordinate for phase/amplitude check
    q = out.u[:, free] @ (M_ff @ phi1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vtus: list[str] = []
    times: list[float] = []
    for step in range(n_steps + 1):
        if step % args.write_every != 0 and step != n_steps:
            continue
        vtu = out_dir / f"bar_{step:05d}.vtu"
        ff.write_vtu(mesh, str(vtu), point_data={"u": out.u[step].reshape(mesh.n_nodes, 3)})
        vtus.append(vtu.name)
        times.append(float(out.t[step]))

    pvd = out_dir / "bar_series.pvd"
    write_pvd_collection(str(pvd), vtus, times)

    rel_omega_err = abs(omega1 - omega1_theory) / max(abs(omega1_theory), 1.0e-30)
    n_period_int = int(np.floor(args.periods + 1.0e-12))
    if n_period_int >= 1:
        step_period = int(round(n_period_int * args.steps_per_period))
        step_period = min(step_period, n_steps)
        rel_period_return = abs(q[step_period] - q[0]) / max(abs(q[0]), 1.0e-30)
    else:
        rel_period_return = np.nan

    print(f"dofs={space.n_dofs}, free={free.size}, dt={dt:.3e}, steps={n_steps}")
    print(f"axial_only={args.axial_only}")
    print(f"omega1(FE)={omega1:.6e}, omega1(rod)={omega1_theory:.6e}, rel.err={rel_omega_err:.3e}")
    if np.isfinite(rel_period_return):
        print(f"modal return over {n_period_int} period(s): rel.diff={rel_period_return:.3e}")
    else:
        print("modal return: skipped (periods < 1)")
    print(f"output: {os.fspath(pvd)}")


if __name__ == "__main__":
    main()

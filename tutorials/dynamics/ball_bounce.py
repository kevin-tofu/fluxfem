#!/usr/bin/env python3
"""
3D elastic ball bounce demo (linear elasticity + Newmark).

Notes:
- Contact is a smooth penalty force applied on the boundary surface.
- Time integration uses implicit Newmark (average acceleration).
- This is a minimal demo; no friction and contact stiffness Jacobian.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import fluxfem as ff
from fluxfem.solver.bc import assemble_surface_linear_form
from fluxfem.tools.visualizer import write_displacement_vtu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Elastic ball bounce demo (3D).")
    p.add_argument("--mesh", type=str, default="data/sphere_r10_lc0p7.msh")
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "results" / "ball_bounce"))
    p.add_argument("--dt", type=float, default=2.0e-4)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--write-every", type=int, default=10)
    p.add_argument("--vx", type=float, default=2.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--vz", type=float, default=0.0)
    p.add_argument("--gravity", type=float, default=-9.81)
    p.add_argument("--density", type=float, default=1.0)
    p.add_argument("--E", type=float, default=1.0e5)
    p.add_argument("--nu", type=float, default=0.35)
    p.add_argument("--contact-k", type=float, default=5.0e5)
    p.add_argument("--contact-c", type=float, default=2.0e2)
    p.add_argument("--contact-beta", type=float, default=40.0)
    p.add_argument("--floor-z", type=float, default=0.0)
    p.add_argument("--init-height", type=float, default=12.0)
    return p.parse_args()


def write_pvd_collection(path: str, files: List[str], times: List[float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as io:
        io.write('<?xml version="1.0"?>\n')
        io.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        io.write("  <Collection>\n")
        for f, t in zip(files, times):
            io.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{f}"/>\n')
        io.write("  </Collection>\n")
        io.write("</VTKFile>\n")


def _softplus(x: np.ndarray, beta: float) -> np.ndarray:
    # numerically stable softplus(beta*x)/beta
    bx = beta * x
    out = np.where(bx > 50.0, bx, np.log1p(np.exp(bx)))
    return out / beta


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- mesh & space ---
    mesh, _, _ = ff.load_gmsh_tet_mesh(args.mesh)
    coords0 = np.asarray(mesh.coords)
    coords0 = coords0 + np.array([0.0, 0.0, args.init_height], dtype=coords0.dtype)
    mesh = ff.TetMesh(coords=coords0, conn=mesh.conn, cell_tags=mesh.cell_tags, node_tags=mesh.node_tags)

    space = ff.make_tet_space(mesh, dim=3, intorder=2)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    # --- material & operators ---
    D = ff.isotropic_3d_D(args.E, args.nu)
    K = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=V, trial=U),
        ff.linear_elasticity_form,
        D,
    )
    M_lumped = np.asarray(space.assemble_mass_matrix(lumped=True), dtype=float)

    g_vec = np.array([0.0, 0.0, args.gravity], dtype=float)
    f_grav = np.asarray(
        ff.assemble_linear_form(
            ff.LinearSpaces(test=V),
            ff.vector_body_force_form,
            args.density * g_vec,
        ),
        dtype=float,
    )

    n_nodes = mesh.n_nodes
    n_dofs = int(space.n_dofs)

    # --- boundary surface for contact ---
    facets = mesh.boundary_facets_where(lambda face: True)
    surface = mesh.surface_from_facets(facets)
    facets_np = np.asarray(surface.conn, dtype=int)

    # --- time integration params (Newmark average acceleration) ---
    dt = float(args.dt)
    beta = 0.25
    gamma = 0.5
    a0 = 1.0 / (beta * dt * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 0.5 / beta - 1.0

    # Prebuild effective stiffness
    K_csr = K.to_csr()
    K_eff = K_csr + sp.diags(M_lumped * a0, 0, shape=(n_dofs, n_dofs), format="csr")

    # --- initial conditions ---
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    v_nodes = v.reshape(n_nodes, 3)
    v_nodes[:, 0] = float(args.vx)
    v_nodes[:, 1] = float(args.vy)
    v_nodes[:, 2] = float(args.vz)
    v = v_nodes.reshape(-1)

    # initial acceleration from static balance (no contact)
    rhs0 = f_grav - K_csr @ u
    a = rhs0 / M_lumped

    # --- contact force assembly ---
    def contact_form(ctx, params):
        N = np.asarray(ctx.v.N)  # (n_q, n_nodes_f)
        facet = params["facets"][int(ctx.facet_id)]
        u_nodes = params["u_nodes"][facet]
        v_nodes = params["v_nodes"][facet]
        u_q = N @ u_nodes
        v_q = N @ v_nodes
        x_q = np.asarray(ctx.x_q) + u_q

        g = x_q[:, 2] - params["floor_z"]
        neg_g = -g
        delta = _softplus(neg_g, params["beta"])
        sigma = _sigmoid(params["beta"] * neg_g)
        delta_dot = sigma * (-v_q[:, 2])

        p = params["k"] * delta + params["c"] * delta_dot
        traction = np.zeros_like(v_q)
        traction[:, 2] = p  # push upward

        fe = N[:, :, None] * traction[:, None, :]
        return fe.reshape(fe.shape[0], -1)

    def assemble_contact(u_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
        u_nodes = u_vec.reshape(n_nodes, 3)
        v_nodes = v_vec.reshape(n_nodes, 3)
        params = {
            "u_nodes": u_nodes,
            "v_nodes": v_nodes,
            "facets": facets_np,
            "floor_z": float(args.floor_z),
            "k": float(args.contact_k),
            "c": float(args.contact_c),
            "beta": float(args.contact_beta),
        }
        return assemble_surface_linear_form(
            surface, contact_form, params, dim=3, n_total_nodes=n_nodes
        )

    # --- time loop ---
    vtufiles: List[str] = []
    times: List[float] = []
    for step in range(args.steps + 1):
        if step % args.write_every == 0:
            out_path = out_dir / f"ball_{step:05d}.vtu"
            write_displacement_vtu(mesh, u, str(out_path))
            vtufiles.append(out_path.name)
            times.append(step * dt)
            print(f"[step {step:05d}] wrote {out_path}")

        f_contact = assemble_contact(u, v)

        rhs = f_grav + f_contact + M_lumped * (a0 * u + a2 * v + a3 * a)
        u_new = spsolve(K_eff, rhs)

        a_new = a0 * (u_new - u) - a2 * v - a3 * a
        v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)

        u, v, a = u_new, v_new, a_new

    pvd_path = out_dir / "ball_series.pvd"
    write_pvd_collection(str(pvd_path), vtufiles, times)
    print(f"VTK series written to {pvd_path}")


if __name__ == "__main__":
    main()

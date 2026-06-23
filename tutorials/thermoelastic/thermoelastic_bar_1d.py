#!/usr/bin/env python3
"""
Thermoelastic 1D bar (scalar displacement) via staggered solve.

Heat:
  -kappa * Laplacian(T) = q  in Omega
  T = 0 on x=0 and x=L

Thermoelastic (axial):
  div(E * grad(u)) = div(E * alpha * T * e_x)
  u = 0 on x=0, traction-free on x=L

Analytical temperature:
  T(x) = q/(2kappa) * x * (L - x)
Analytical displacement (u(0)=0, traction-free at x=L):
  u(x) = alpha * integral_0^x T(s) ds
  u(L) = alpha * q * L^3 / (12 kappa)
"""

from __future__ import annotations

import argparse
import os
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import fluxfem as ff  # noqa: E402
import fluxfem.helpers_wf as h_wf  # noqa: E402


def build_bar_mesh(*, nx: int, ny: int, nz: int, lx: float, ly: float, lz: float):
    return ff.StructuredHexBox(
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
    ).build()


def x_bounds(mesh):
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())
    return xmin, xmax, coords


def default_output_path(filename: str) -> str:
    return os.path.join("result", "tutorials", filename)


def parse_args():
    p = argparse.ArgumentParser(description="Thermoelastic 1D bar (scalar displacement).")
    p.add_argument("--nx", type=int, default=12, help="Elements along x (length).")
    p.add_argument("--ny", type=int, default=2, help="Elements along y (thickness).")
    p.add_argument("--nz", type=int, default=2, help="Elements along z (width).")
    p.add_argument("--lx", type=float, default=1.0, help="Bar length.")
    p.add_argument("--ly", type=float, default=0.1, help="Bar thickness.")
    p.add_argument("--lz", type=float, default=0.1, help="Bar width.")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--kappa", type=float, default=1.0, help="Thermal conductivity.")
    p.add_argument("--source", type=float, default=1.0, help="Uniform heat source q.")
    p.add_argument("--E", type=float, default=1.0, help="Young's modulus for axial stiffness.")
    p.add_argument("--alpha", type=float, default=1.0e-3, help="Thermal expansion coefficient.")
    p.add_argument(
        "--output",
        type=str,
        default=default_output_path("thermoelastic_bar_1d.vtu"),
        help="VTU output path.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    mesh = build_bar_mesh(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
    )
    space = ff.make_hex_space(mesh, dim=1, intorder=args.intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    xmin, xmax, coords = x_bounds(mesh)
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
    dir_temp = np.unique(np.concatenate([dir_left.dofs, dir_right.dofs]))

    # --- Heat solve ---
    bilinear_T = ff.BilinearForm.volume(
        lambda u, v, p: p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
    )
    linear_T = ff.LinearForm.volume(lambda v, p: (v * p.q) * h_wf.dOmega())

    K_T = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=V, trial=U),
        bilinear_T,
        ff.Params(kappa=args.kappa),
    )
    F_T = ff.assemble_linear_form(
        ff.LinearSpaces(test=V),
        linear_T,
        ff.Params(q=args.source),
    )

    solver = ff.LinearSolver(method="spsolve")
    T_vec, _ = solver.solve(
        K_T,
        F_T,
        dirichlet=ff.DirichletBC(dir_temp, None),
        dirichlet_mode="condense",
    )
    T_nodes = np.asarray(T_vec).reshape(-1)

    # --- Thermoelastic solve (scalar axial displacement) ---
    bilinear_u = ff.BilinearForm.volume(
        lambda u, v, p: p.E * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
    )

    def thermal_rhs(ctx: ff.FormContext, params) -> jnp.ndarray:
        T_elem = params["T_elem"][ctx.elem_id]
        T_q = ctx.trial.eval(T_elem)  # (n_q,)
        flux = jnp.zeros((T_q.shape[0], 3), dtype=T_q.dtype)
        flux = flux.at[:, 0].set(params["E"] * params["alpha"] * T_q)
        return jnp.einsum("qaj,qj->qa", ctx.test.gradN, flux)

    K_u = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=V, trial=U),
        bilinear_u,
        ff.Params(E=args.E),
    )
    params_u = {
        "E": args.E,
        "alpha": args.alpha,
        "T_elem": jnp.asarray(T_nodes)[space.elem_dofs],
    }
    F_u = ff.assemble_linear_form(
        ff.LinearSpaces(test=V),
        thermal_rhs,
        params_u,
    )

    u_vec, _ = solver.solve(
        K_u,
        F_u,
        dirichlet=dir_left,
        dirichlet_mode="condense",
    )
    u_nodes = np.asarray(u_vec).reshape(-1)

    # --- Diagnostics ---
    x_coords = coords[:, 0]
    uL = float(np.max(u_nodes[np.isclose(x_coords, xmax, atol=1e-8)]))
    uL_theory = args.alpha * args.source * (args.lx ** 3) / (12.0 * args.kappa)
    rel_err = abs(uL - uL_theory) / abs(uL_theory) if uL_theory != 0.0 else 0.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ff.write_vtu(mesh, args.output, point_data={"T": T_nodes, "u": u_nodes})

    print(
        f"thermoelastic solved: dofs={space.n_dofs}, output={args.output}"
    )
    print(f"T max≈{T_nodes.max():.6e}, u(L)≈{uL:.6e}")
    print(f"[theory] u(L)≈{uL_theory:.6e} (rel.err={rel_err:.3e})")


if __name__ == "__main__":
    main()

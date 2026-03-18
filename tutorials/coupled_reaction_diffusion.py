#!/usr/bin/env python3
"""
Coupled reaction-diffusion system in a 3D box.

Model:
  -k1 * Laplacian(u) + alpha * (u - v) = f1
  -k2 * Laplacian(v) + alpha * (v - u) = f2
  u = v = 0 on the boundary
"""

from __future__ import annotations

import argparse
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import fluxfem as ff  # noqa: E402
import fluxfem.helpers_wf as h_wf  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Coupled reaction-diffusion system.")
    p.add_argument("--nx", type=int, default=8, help="Elements along x.")
    p.add_argument("--ny", type=int, default=8, help="Elements along y.")
    p.add_argument("--nz", type=int, default=8, help="Elements along z.")
    p.add_argument("--lx", type=float, default=1.0, help="Box length.")
    p.add_argument("--ly", type=float, default=1.0, help="Box height.")
    p.add_argument("--lz", type=float, default=1.0, help="Box width.")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--k1", type=float, default=1.0, help="Diffusivity for u.")
    p.add_argument("--k2", type=float, default=0.5, help="Diffusivity for v.")
    p.add_argument("--alpha", type=float, default=5.0, help="Coupling strength.")
    p.add_argument("--f1", type=float, default=1.0, help="Source term for u.")
    p.add_argument("--f2", type=float, default=0.0, help="Source term for v.")
    p.add_argument("--output", type=str, default="coupled_reaction_diffusion.vtu", help="VTU output path.")
    return p.parse_args()


def main():
    args = parse_args()
    space_u_unknown = "U"
    space_u_test = "W"
    space_v_unknown = "V"
    space_v_test = "Z"

    mesh = ff.StructuredHexBox(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
    ).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=args.intorder)
    mixed = ff.MixedSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace(space_u_test, space),
                unknown=ff.NamedSpace(space_u_unknown, space),
            ),
            "v": ff.ResidualSpaces(
                test=ff.NamedSpace(space_v_test, space),
                unknown=ff.NamedSpace(space_v_unknown, space),
            ),
        }
    ).to_fe_space()

    def res_u(v, u, p):
        v_ref = ff.unknown_ref("v", space=space_v_unknown)
        return (
            p.k1 * h_wf.gaction(v, h_wf.grad(u))
            + v * (p.alpha * (u.val - v_ref.val))
            - v * p.f1
        ) * h_wf.dOmega()

    def res_v(q, v, p):
        u_ref = ff.unknown_ref("u", space=space_u_unknown)
        return (
            p.k2 * h_wf.gaction(q, h_wf.grad(v))
            + q * (p.alpha * (v.val - u_ref.val))
            - q * p.f2
        ) * h_wf.dOmega()

    residuals = ff.make_mixed_residuals(
        u=ff.bind_mixed_residual("u", res_u, space=space_u_unknown),
        v=ff.bind_mixed_residual("v", res_v, space=space_v_unknown),
    )
    params = ff.Params(k1=args.k1, k2=args.k2, alpha=args.alpha, f1=args.f1, f2=args.f2)

    u0 = jnp.zeros(mixed.n_dofs)
    pattern = mixed.get_sparsity_pattern(with_idx=True)
    problem = ff.MixedProblem(mixed, residuals, params=params, pattern=pattern)
    K = problem.assemble_jacobian(u0)
    R0 = problem.assemble_residual(u0)
    b = -R0

    boundary = mesh.boundary_dofs_bbox(components=[0], dof_per_node=1)
    bc = mixed.make_dirichlet(
        u=(boundary, None),
        v=(boundary, None),
    )

    solver = ff.LinearSolver(method="spsolve")
    sol, _ = solver.solve(
        K,
        b,
        dirichlet=bc.as_dirichlet_bc(),
        dirichlet_mode="condense",
    )
    solution_fields = mixed.unpack_fields(sol)
    u = np.asarray(solution_fields["u"])
    v = np.asarray(solution_fields["v"])

    ff.write_vtu(mesh, args.output, point_data={"u": u, "v": v})

    print(f"coupled solve: dofs={mixed.n_dofs}, output={args.output}")
    print(f"u range: [{u.min():.3e}, {u.max():.3e}]")
    print(f"v range: [{v.min():.3e}, {v.max():.3e}]")


if __name__ == "__main__":
    main()

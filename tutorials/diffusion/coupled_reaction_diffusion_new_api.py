#!/usr/bin/env python3
"""
Coupled reaction-diffusion system using the explicit mixed-field bindings API.

This mirrors `tutorials/diffusion/coupled_reaction_diffusion.py`, but only makes the
mixed-specific parts explicit:
  - field names distinct from space keys
  - explicit residual labels via bind_mixed_residual(...)
  - explicit cross-field references via unknown_ref(..., space=...)

Within each residual, the local trial/test arguments still use the short form.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import fluxfem as ff  # noqa: E402
import fluxfem.helpers_wf as h_wf  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Coupled reaction-diffusion system (new mixed API).")
    p.add_argument("--nx", type=int, default=8, help="Elements along x.")
    p.add_argument("--ny", type=int, default=8, help="Elements along y.")
    p.add_argument("--nz", type=int, default=8, help="Elements along z.")
    p.add_argument("--lx", type=float, default=1.0, help="Box length.")
    p.add_argument("--ly", type=float, default=1.0, help="Box height.")
    p.add_argument("--lz", type=float, default=1.0, help="Box width.")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--k1", type=float, default=1.0, help="Diffusivity for the first field.")
    p.add_argument("--k2", type=float, default=0.5, help="Diffusivity for the second field.")
    p.add_argument("--alpha", type=float, default=5.0, help="Coupling strength.")
    p.add_argument("--f1", type=float, default=1.0, help="Source term for the first field.")
    p.add_argument("--f2", type=float, default=0.0, help="Source term for the second field.")
    p.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "coupled_reaction_diffusion_new_api.vtu"),
        help="VTU output path.",
    )
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
            "species_u": ff.ResidualSpaces(
                test=ff.NamedSpace(space_u_test, space),
                unknown=ff.NamedSpace(space_u_unknown, space),
            ),
            "species_v": ff.ResidualSpaces(
                test=ff.NamedSpace(space_v_test, space),
                unknown=ff.NamedSpace(space_v_unknown, space),
            ),
        }
    ).to_fe_space()

    def res_u(v, u, p):
        # The local field `u` is the short-form trial/unknown on space U.
        other = ff.unknown_ref("other_species", space=space_v_unknown)
        return (
            p.k1 * h_wf.gaction(v, h_wf.grad(u))
            + v * (p.alpha * (u.val - other.val))
            - v * p.f1
        ) * h_wf.dOmega()

    def res_v(q, v, p):
        # Cross-coupling is where the explicit space key pays off.
        other = ff.unknown_ref("other_species", space=space_u_unknown)
        return (
            p.k2 * h_wf.gaction(q, h_wf.grad(v))
            + q * (p.alpha * (v.val - other.val))
            - q * p.f2
        ) * h_wf.dOmega()

    residuals = ff.make_mixed_residuals(
        balance_u=ff.bind_mixed_residual("species_u", res_u, space=space_u_unknown),
        balance_v=ff.bind_mixed_residual("species_v", res_v, space=space_v_unknown),
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
        species_u=(boundary, None),
        species_v=(boundary, None),
    )

    solver = ff.LinearSolver(method="spsolve")
    sol, _ = solver.solve(
        K,
        b,
        dirichlet=bc.as_dirichlet_bc(),
        dirichlet_mode="condense",
    )
    solution_fields = mixed.unpack_fields(sol)
    u = np.asarray(solution_fields["species_u"])
    v = np.asarray(solution_fields["species_v"])

    ff.write_vtu(mesh, args.output, point_data={"u": u, "v": v})

    print(f"coupled solve (new API): dofs={mixed.n_dofs}, output={args.output}")
    print(f"u range: [{u.min():.3e}, {u.max():.3e}]")
    print(f"v range: [{v.min():.3e}, {v.max():.3e}]")


if __name__ == "__main__":
    main()

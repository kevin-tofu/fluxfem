#!/usr/bin/env python3
"""
Thermoelastic 1D bar (scalar displacement) solved as a mixed system.

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
from fluxfem.core.mixed_space import MixedFESpace  # noqa: E402
from fluxfem.core.weakform import einsum  # noqa: E402
from fluxfem.core.mixed_weakform import (  # noqa: E402
    MixedResidualForm,
    assemble_mixed_jacobian_wf,
    assemble_mixed_residual_wf,
)
from tutorials._thermoelastic_utils import (  # noqa: E402
    boundary_dofs_at_x,
    build_bar_mesh,
    default_output_path,
    x_bounds,
)


def parse_args():
    p = argparse.ArgumentParser(description="Thermoelastic 1D bar (mixed solve).")
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
        default=default_output_path("thermoelastic_bar_1d_mixed.vtu"),
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
    mixed = MixedFESpace({"u": space, "T": space})

    xmin, xmax, coords = x_bounds(mesh)
    dir_left = boundary_dofs_at_x(mesh, xmin)
    dir_right = boundary_dofs_at_x(mesh, xmax)

    dir_u = dir_left + mixed.field_offsets["u"]
    dir_T = np.unique(np.concatenate([dir_left, dir_right])) + mixed.field_offsets["T"]
    dir_dofs = np.unique(np.concatenate([dir_u, dir_T]))

    def res_T(v, T, p):
        return (p.kappa * h_wf.gaction(v, h_wf.grad(T)) - v * p.q) * h_wf.dOmega()

    def res_u(v, u, p):
        T_ref = ff.unknown_ref("T")
        e_x = einsum("q,i->qi", T_ref.val, p.ex)
        return (
            p.E * h_wf.gaction(v, h_wf.grad(u))
            - p.E * p.alpha * h_wf.gaction(v, e_x)
        ) * h_wf.dOmega()

    mixed_form = MixedResidualForm({"u": res_u, "T": res_T})
    params = ff.Params(
        kappa=args.kappa,
        q=args.source,
        E=args.E,
        alpha=args.alpha,
        ex=jnp.asarray([1.0, 0.0, 0.0]),
    )

    pattern = mixed.get_sparsity_pattern(with_idx=True)
    u0 = jnp.zeros(mixed.n_dofs)
    K = assemble_mixed_jacobian_wf(
        mixed, mixed_form, u0, params, pattern=pattern, return_flux_matrix=True
    )
    R0 = assemble_mixed_residual_wf(mixed, mixed_form, u0, params)
    b = -R0

    solver = ff.LinearSolver(method="spsolve")
    sol, _ = solver.solve(K, b, dirichlet=(dir_dofs, None), dirichlet_mode="condense")
    fields = mixed.unpack_fields(sol)
    u_nodes = np.asarray(fields["u"])
    T_nodes = np.asarray(fields["T"])

    x_coords = coords[:, 0]
    uL = float(np.max(u_nodes[np.isclose(x_coords, xmax, atol=1e-8)]))
    uL_theory = args.alpha * args.source * (args.lx ** 3) / (12.0 * args.kappa)
    rel_err = abs(uL - uL_theory) / abs(uL_theory) if uL_theory != 0.0 else 0.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ff.write_vtu(mesh, args.output, point_data={"T": T_nodes, "u": u_nodes})

    print(f"thermoelastic mixed: dofs={mixed.n_dofs}, output={args.output}")
    print(f"T max≈{T_nodes.max():.6e}, u(L)≈{uL:.6e}")
    print(f"[theory] u(L)≈{uL_theory:.6e} (rel.err={rel_err:.3e})")


if __name__ == "__main__":
    main()

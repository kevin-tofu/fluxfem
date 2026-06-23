#!/usr/bin/env python3
"""
Linear material + geometric nonlinearity (St. Venant–Kirchhoff) demo.

Runs both:
  - Linear small-strain elasticity (material + geometric linear)
  - Geometric nonlinear (finite strain) with linear material law
"""

from __future__ import annotations

import argparse
import numpy as np
import jax
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
import fluxfem.helpers_ts as h_ts
from fluxfem.core.basis import build_B_matrices_finite
from fluxfem.physics.elasticity.hyperelastic import deformation_gradient, green_lagrange_strain

jax.config.update("jax_enable_x64", True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Linear material + geometric nonlinearity (StVK) tutorial."
    )
    p.add_argument("--nx", type=int, default=12)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--nz", type=int, default=2)
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--ly", type=float, default=0.1)
    p.add_argument("--lz", type=float, default=0.1)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--E", type=float, default=210_000.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--traction", type=float, default=1.0)
    p.add_argument(
        "--mode",
        choices=["linear", "geononlinear", "hyperelastic", "both", "all"],
        default="both",
        help="Which model(s) to run.",
    )
    p.add_argument("--nstep", type=int, default=30)
    p.add_argument("--maxiter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--line-search", action="store_true")
    p.add_argument("--output-linear-vtu", type=str, default="")
    p.add_argument("--output-nonlinear-vtu", type=str, default="")
    p.add_argument("--output-hyperelastic-vtu", type=str, default="")
    return p.parse_args()


def stvk_residual_form(ctx: ff.FormContext, u_elem: jnp.ndarray, params) -> jnp.ndarray:
    D = params["D"]
    F = deformation_gradient(ctx, u_elem)               # (n_q, 3, 3)
    E = green_lagrange_strain(F)                        # (n_q, 3, 3)
    E_voigt = jnp.stack(
        [E[..., 0, 0], E[..., 1, 1], E[..., 2, 2], E[..., 0, 1], E[..., 1, 2], E[..., 2, 0]],
        axis=-1,
    )  # (n_q, 6)
    S_voigt = jnp.einsum("ij,qj->qi", D, E_voigt)        # (n_q, 6)
    B = build_B_matrices_finite(ctx.trial.gradN, F)      # (n_q, 6, n_ldofs)
    BT = jnp.swapaxes(B, 1, 2)
    return jnp.einsum("qik,qk->qi", BT, S_voigt)


stvk_residual_form._ff_kind = "residual"
stvk_residual_form._ff_domain = "volume"


def main():
    args = parse_args()

    dtype = jnp.float64
    max_disp_linear = None
    max_disp_geo = None
    max_disp_hyper = None
    mesh = ff.StructuredHexBox(
        nx=args.nx, ny=args.ny, nz=args.nz, lx=args.lx, ly=args.ly, lz=args.lz
    ).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=args.intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    D = jnp.asarray(ff.isotropic_3d_D(args.E, args.nu), dtype=dtype)
    lam, mu = ff.lame_parameters(args.E, args.nu)

    coords_np = np.asarray(mesh.coords)
    xmin = float(coords_np[:, 0].min())
    xmax = float(coords_np[:, 0].max())
    facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    surface = ff.make_surface_from_facets(coords_np, facets)
    traction_vec = np.array([args.traction, 0.0, 0.0], dtype=float)

    def traction_form(ctx: ff.SurfaceFormContext, t: np.ndarray) -> np.ndarray:
        return h_ts.dot(ctx.v, t)

    F_ext = surface.assemble_linear_form_on_space(
        space, traction_form, params=traction_vec
    )
    F_ext = jnp.asarray(F_ext, dtype=dtype)

    dir_dofs = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    ).dofs
    bc = ff.DirichletBC(dir_dofs, None)

    if args.mode in ("linear", "both", "all"):
        print("[linear] start")
        bilinear_form = ff.BilinearForm.volume(
            lambda u, v, D_mat: h_wf.ddot(v.sym_grad, h_wf.matmul_std(D_mat, u.sym_grad))
            * h_wf.dOmega()
        )
        K = ff.assemble_bilinear_form(
            ff.BilinearSpaces(test=V, trial=U),
            bilinear_form,
            D,
        )
        solver = ff.LinearSolver(method="spsolve")
        u_lin, _ = solver.solve(K, F_ext, dirichlet=bc, dirichlet_mode="condense")

        u_nodes = np.asarray(u_lin).reshape(-1, 3)
        max_disp = float(np.linalg.norm(u_nodes, axis=1).max()) if u_nodes.size else 0.0
        max_disp_linear = max_disp
        print(f"[linear] max |u| = {max_disp:.6e}", flush=True)

        if args.output_linear_vtu:
            ff.write_elastic_vtu(
                mesh,
                space,
                u_lin,
                args.output_linear_vtu,
                compute_j=True,
                deformed_scale=1.0,
            )
            print("VTU written to", args.output_linear_vtu)

    if args.mode in ("geononlinear", "both", "all"):
        print("[geo-nonlinear] start")
        J_pattern = ff.make_sparsity_pattern(space, with_idx=False)
        analysis = ff.NonlinearAnalysis(
            space=space,
            residual_form=stvk_residual_form,
            params={"D": D},
            base_external_vector=F_ext,
            dirichlet=bc,
            jacobian_pattern=J_pattern,
            dtype=dtype,
        )
        newton_cfg = ff.NewtonLoopConfig(
            tol=args.tol,
            atol=args.atol,
            maxiter=args.maxiter,
            line_search=args.line_search,
            linear_solver="cg",
            linear_preconditioner="jacobi",
            n_steps=args.nstep,
        )
        runner = ff.NewtonSolveRunner(analysis, newton_cfg)
        u0 = jnp.zeros(space.n_dofs, dtype=dtype)
        u_nl, _history = runner.run(u0=u0, newton_callback=lambda _cb: None)

        u_nodes = np.asarray(u_nl).reshape(-1, 3)
        max_disp = float(np.linalg.norm(u_nodes, axis=1).max()) if u_nodes.size else 0.0
        max_disp_geo = max_disp
        print(f"[geo-nonlinear] max |u| = {max_disp:.6e}", flush=True)

        if args.output_nonlinear_vtu:
            ff.write_elastic_vtu(
                mesh,
                space,
                u_nl,
                args.output_nonlinear_vtu,
                compute_j=True,
                deformed_scale=1.0,
            )
            print("VTU written to", args.output_nonlinear_vtu)

    if args.mode in ("hyperelastic", "all"):
        print("[hyperelastic] start")
        J_pattern = ff.make_sparsity_pattern(space, with_idx=False)
        analysis = ff.NonlinearAnalysis(
            space=space,
            residual_form=ff.neo_hookean_residual_form,
            params={"lam": lam, "mu": mu},
            base_external_vector=F_ext,
            dirichlet=bc,
            jacobian_pattern=J_pattern,
            dtype=dtype,
        )
        newton_cfg = ff.NewtonLoopConfig(
            tol=args.tol,
            atol=args.atol,
            maxiter=args.maxiter,
            line_search=args.line_search,
            linear_solver="cg",
            linear_preconditioner="jacobi",
            n_steps=args.nstep,
        )
        runner = ff.NewtonSolveRunner(analysis, newton_cfg)
        u0 = jnp.zeros(space.n_dofs, dtype=dtype)
        u_nh, _history = runner.run(u0=u0, newton_callback=lambda _cb: None)

        u_nodes = np.asarray(u_nh).reshape(-1, 3)
        max_disp = float(np.linalg.norm(u_nodes, axis=1).max()) if u_nodes.size else 0.0
        max_disp_hyper = max_disp
        print(f"[hyperelastic] max |u| = {max_disp:.6e}", flush=True)

        if args.output_hyperelastic_vtu:
            ff.write_elastic_vtu(
                mesh,
                space,
                u_nh,
                args.output_hyperelastic_vtu,
                compute_j=True,
                deformed_scale=1.0,
            )
            print("VTU written to", args.output_hyperelastic_vtu)

    print(
        "[summary] max|u| "
        f"linear={max_disp_linear} "
        f"geo={max_disp_geo} "
        f"hyper={max_disp_hyper}",
        flush=True,
    )


if __name__ == "__main__":
    main()

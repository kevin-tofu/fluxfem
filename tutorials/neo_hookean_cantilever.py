#!/usr/bin/env python3
"""
Hyperelastic cantilever (Neo-Hookean) — minimal FluxFEM demo.

- Uses a structured hex mesh (no external mesh file).
- Clamped at x = xmin.
- Uniform traction applied on x = xmax (y-direction).
- Newton solve, then write VTU (optional).
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import fluxfem as ff
import fluxfem.helpers_ts as h_ts


def parse_args():
    p = argparse.ArgumentParser(description="Neo-Hookean cantilever demo.")
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=4)
    p.add_argument("--nz", type=int, default=4)
    p.add_argument("--lx", type=float, default=100.0)
    p.add_argument("--ly", type=float, default=10.0)
    p.add_argument("--lz", type=float, default=10.0)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--E", type=float, default=210_000.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--traction", type=float, default=1e-2)
    p.add_argument("--nstep", type=int, default=200)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--line-search", action="store_true")
    p.add_argument(
        "--linear-solver",
        type=str,
        default="cg_matfree",
        choices=["cg", "cg_matfree", "spsolve"],
    )
    p.add_argument(
        "--linear-precond",
        type=str,
        default="none",
        choices=["none", "diag0", "jacobi", "block_jacobi"],
    )
    p.add_argument(
        "--matfree-mode",
        type=str,
        default="linearize",
        choices=["linearize", "jvp"],
    )
    output_group = p.add_mutually_exclusive_group()
    output_group.add_argument("--output-vtu", type=str, default="result.vtu")
    output_group.add_argument("--no-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --------------------
    # Parameters (edit here)
    # --------------------
    dtype = jnp.float64

    # geometry / mesh
    nx, ny, nz = args.nx, args.ny, args.nz
    lx, ly, lz = args.lx, args.ly, args.lz
    intorder = args.intorder

    # material
    E, nu = args.E, args.nu
    lam, mu = ff.lame_parameters(E, nu)
    params = {"lam": lam, "mu": mu}

    # loads
    traction = args.traction      # applied on x = xmax face, along +y
    body_force = (0.0, 0.0, 0.0)

    # nonlinear solve
    nstep = args.nstep
    maxiter = args.maxiter
    tol = args.tol
    atol = args.atol
    line_search = args.line_search

    # linear solver options
    linear_solver = args.linear_solver
    linear_precond = args.linear_precond
    matfree_mode = args.matfree_mode

    # output
    output_vtu = None if args.no_output else args.output_vtu

    # --------------------
    # Build mesh & space
    # --------------------
    mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=dtype),
        conn=jnp.asarray(mesh.conn, dtype=mesh.conn.dtype),
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )

    space = ff.make_hex_space(mesh, dim=3, intorder=intorder)

    # --------------------
    # External force vector
    # --------------------
    f_body = jnp.array(body_force, dtype=dtype)
    F_ext = space.assemble_linear_form(ff.vector_body_force_form, params=f_body, sparse=False)

    # traction on x = xmax
    coords_np = np.asarray(mesh.coords)
    xmax = float(coords_np[:, 0].max())
    neumann_facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    surf = ff.make_surface_from_facets(coords_np, neumann_facets)

    def traction_form(ctx: ff.SurfaceFormContext, traction_vec: np.ndarray) -> np.ndarray:
        return h_ts.dot(ctx.v, traction_vec)

    traction_vec = np.array([0.0, traction, 0.0], dtype=float)
    F_ext = surf.assemble_linear_form_on_space(space, traction_form, params=traction_vec, F0=F_ext)
    F_ext = jnp.asarray(F_ext, dtype=dtype)

    # --------------------
    # Dirichlet DOFs (clamp x = xmin)
    # --------------------
    xmin = float(coords_np[:, 0].min())
    dir_dofs = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    ).dofs

    # --------------------
    # Nonlinear analysis (Neo-Hookean)
    # --------------------
    J_pattern = ff.make_sparsity_pattern(space, with_idx=False)

    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params=params,
        base_external_vector=F_ext,
        dirichlet=ff.DirichletBC(dir_dofs, None),
        jacobian_pattern=J_pattern,
        dtype=dtype,
    )

    newton_cfg = ff.NewtonLoopConfig(
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        line_search=line_search,
        linear_solver=linear_solver,
        linear_preconditioner=(None if linear_precond == "none" else linear_precond),
        matfree_mode=matfree_mode,
        n_steps=nstep,
    )
    runner = ff.NewtonSolveRunner(analysis, newton_cfg)

    # solve
    u0 = jnp.zeros(space.n_dofs, dtype=dtype)
    u, history = runner.run(u0=u0, newton_callback=lambda cb: None)

    u_nodes = np.asarray(u).reshape(-1, 3)
    max_disp = float(np.linalg.norm(u_nodes, axis=1).max()) if u_nodes.size else 0.0
    print("Final max |u| =", max_disp)

    # --------------------
    # Output (optional)
    # --------------------
    if output_vtu:
        ff.write_elastic_vtu(
            mesh, space, u, output_vtu, compute_j=True, deformed_scale=1.0
        )
        print("VTU written to", output_vtu)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hyperelastic cantilever demo (Tet/Hex; supports gmsh mesh-file + VTU output).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import fluxfem as ff


def env_default(name: str, default, cast):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return cast(val)
    except Exception:
        return default


def parse_args():
    p = argparse.ArgumentParser(description="Hyperelastic cantilever demo.")
    p.add_argument("--mesh-file", type=str, default=os.environ.get("MESH_FILE", "data/tension_bar.msh"))
    p.add_argument("--nx", type=int, default=env_default("NX", 20, int))
    p.add_argument("--ny", type=int, default=env_default("NY", 4, int))
    p.add_argument("--nz", type=int, default=env_default("NZ", 4, int))
    p.add_argument("--lx", type=float, default=env_default("LX", 100.0, float))
    p.add_argument("--ly", type=float, default=env_default("LY", 10.0, float))
    p.add_argument("--lz", type=float, default=env_default("LZ", 10.0, float))
    p.add_argument("--traction", type=float, default=env_default("TRACTION", 1e-2, float))
    p.add_argument("--fx", type=float, default=env_default("FX", 0.0, float))
    p.add_argument("--fy", type=float, default=env_default("FY", 0.0, float))
    p.add_argument("--fz", type=float, default=env_default("FZ", 0.0, float))
    p.add_argument("--E", type=float, default=env_default("E", 210_000.0, float))
    p.add_argument("--nu", type=float, default=env_default("NU", 0.3, float))
    p.add_argument("--intorder", type=int, default=env_default("INTORDER", 2, int))
    p.add_argument("--nstep", type=int, default=env_default("NSTEP", 200, int))
    p.add_argument("--maxiter", type=int, default=env_default("MAXITER", 80, int))
    p.add_argument("--tol", type=float, default=env_default("TOL", 1e-4, float))
    p.add_argument("--atol", type=float, default=env_default("ATOL", 1e-6, float))
    p.add_argument("--line-search", action="store_true")
    p.add_argument("--max-ls", type=int, default=env_default("MAX_LS", 30, int))
    p.add_argument("--ls-c", type=float, default=env_default("LS_C", 1e-4, float))
    p.add_argument("--linear-solver", type=str, choices=("spsolve", "cg"), default=env_default("LINEAR_SOLVER", "cg", str))
    p.add_argument("--linear-precond", type=str, choices=("none", "jacobi", "block_jacobi"), default=env_default("LINEAR_PRECOND", "jacobi", str))
    p.add_argument("--linear-tol", type=float, default=env_default("LINEAR_TOL", None, float))
    p.add_argument("--dirichlet-tag", type=int, default=env_default("DIRICHLET_TAG", 1, int))
    p.add_argument("--traction-tag", type=int, default=env_default("TRACTION_TAG", 2, int))
    p.add_argument(
        "--output-vtu",
        type=str,
        default=os.environ.get("OUTPUT_VTU", "result/tutorials/hyperelastic_cantilever/hyperelastic_cantilever.vtu").strip(),
    )
    return p.parse_args()


def main():
    args = parse_args()
    dtype = jnp.float64

    lam, mu = ff.lame_parameters(args.E, args.nu)
    params = {"mu": mu, "lam": lam}

    # --- Load mesh (gmsh if exists; else structured hex) ---
    facet_data = None
    if args.mesh_file and os.path.exists(args.mesh_file):
        mesh, facets, facet_tags = ff.load_gmsh_mesh(args.mesh_file)
        if facets is not None and (facet_tags is None or np.all(np.asarray(facet_tags) == 0)):
            # geometric fallback tagging by x-min/x-max
            coords_np = np.asarray(mesh.coords)
            xs = coords_np[facets.reshape(-1), 0].reshape(facets.shape)
            xmin = coords_np[:, 0].min()
            xmax = coords_np[:, 0].max()
            tags_geom = np.zeros(facets.shape[0], dtype=np.int32)
            tags_geom[np.all(np.isclose(xs, xmin, atol=1e-8), axis=1)] = args.dirichlet_tag
            tags_geom[np.all(np.isclose(xs, xmax, atol=1e-8), axis=1)] = args.traction_tag
            facet_tags = tags_geom
        facet_data = (facets, facet_tags)
    else:
        mesh = ff.StructuredHexBox(nx=args.nx, ny=args.ny, nz=args.nz, lx=args.lx, ly=args.ly, lz=args.lz).build()

    # dtype promote (keep n_nodes consistent)
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=dtype),
        conn=jnp.asarray(mesh.conn, dtype=mesh.conn.dtype),
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )

    # space
    if isinstance(mesh, ff.TetMesh):
        space = ff.make_tet_space(mesh, dim=3, intorder=args.intorder)
    else:
        space = ff.make_hex_space(mesh, dim=3, intorder=args.intorder)

    J_pattern = ff.make_sparsity_pattern(space, with_idx=False)

    def surface_traction_form(ctx: ff.SurfaceFormContext, traction_vec: np.ndarray) -> np.ndarray:
        return ff.dot(ctx.v, traction_vec)

    # external vector
    f_body = jnp.array([args.fx, args.fy, args.fz], dtype=dtype)
    F_ext = space.assemble_linear_form(
        ff.vector_body_force_form, params=f_body, sparse=False
    )

    if abs(args.traction) > 0:
        coords_np = np.asarray(mesh.coords)
        xmax = float(coords_np[:, 0].max())

        if facet_data is not None and facet_data[0] is not None and facet_data[1] is not None:
            facets, tags = facet_data
            neumann_facets = np.asarray(facets)[np.asarray(tags) == args.traction_tag]
        else:
            neumann_facets = np.asarray(
                mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
            )

        if neumann_facets.size == 0:
            raise RuntimeError("No Neumann facets found for traction.")

        surf = ff.make_surface_from_facets(coords_np, neumann_facets)
        traction_vec = np.array([0.0, args.traction, 0.0], dtype=float)
        F_ext = surf.assemble_linear_form_on_space(
            space,
            surface_traction_form,
            params=traction_vec,
            F0=F_ext,
        )

    F_ext = jnp.asarray(F_ext, dtype=dtype)
    print("||F_ext||_inf =", float(jnp.max(jnp.abs(F_ext))), "nonzero =", int(jnp.count_nonzero(F_ext)))

    # dirichlet dofs
    if facet_data is not None and facet_data[0] is not None and facet_data[1] is not None:
        facets, tags = facet_data
        clamp_nodes = np.unique(np.asarray(facets)[np.asarray(tags) == args.dirichlet_tag].reshape(-1))
        dir_dofs = mesh.node_dofs(clamp_nodes, components="xyz")
    else:
        xmin = float(np.asarray(mesh.coords)[:, 0].min())
        dir_dofs = mesh.boundary_dofs_where(
            lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
            components="xyz",
        )

    internal_res = ff.neo_hookean_residual_form

    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=internal_res,
        params=params,
        base_external_vector=F_ext,
        dirichlet=(dir_dofs, None),
        jacobian_pattern=J_pattern,
        dtype=dtype,
    )
    newton_cfg = ff.NewtonLoopConfig(
        tol=args.tol,
        atol=args.atol,
        maxiter=args.maxiter,
        line_search=args.line_search,
        max_ls=args.max_ls,
        ls_c=args.ls_c,
        linear_solver=args.linear_solver,
        linear_preconditioner=(None if args.linear_precond == "none" else args.linear_precond),
        linear_tol=args.linear_tol,
        n_steps=args.nstep,
    )
    runner = ff.NewtonSolveRunner(analysis, newton_cfg)

    u0 = jnp.zeros(space.n_dofs, dtype=dtype)
    u, history = runner.run(u0=u0, newton_callback=lambda cb: None)

    u_nodes = np.asarray(u).reshape(-1, 3)
    max_disp = float(np.linalg.norm(u_nodes, axis=1).max()) if u_nodes.size else 0.0
    # max_disp = float(np.linalg.norm(u_nodes, axis=1).max()) if u_nodes.size else 0.0
    print("Final max |u| =", max_disp)

    if args.output_vtu:
        ff.write_elastic_vtu(mesh, space, u, args.output_vtu, compute_j=True, deformed_scale=1.0)
        print(f"VTU written to {args.output_vtu}")


if __name__ == "__main__":
    main()

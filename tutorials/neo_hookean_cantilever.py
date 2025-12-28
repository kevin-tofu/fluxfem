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
import fluxfem.helpers_ts as h_ts
from scripts.render_deformed_vtu import render_deformed_vtu


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
        "--output-dir",
        type=str,
        default=os.environ.get(
            "OUTPUT_DIR",
            "result/tutorials/neo_hookean_cantilever",
        ).strip(),
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
        if facet_tags is not None:
            tags_np = np.asarray(facet_tags)
            tags_unique = np.unique(tags_np)
            print("facet_tags unique:", tags_unique)
            for t in tags_unique:
                print(f"facet_tags count[{int(t)}] =", int(np.sum(tags_np == t)))
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

    ctxs = space.build_form_contexts()
    ctx0 = jax.tree_util.tree_map(lambda x: x[0], ctxs)
    detJ = np.asarray(ctx0.test.detJ)
    print("detJ finite:", bool(np.all(np.isfinite(detJ))), "min/max:", float(detJ.min()), float(detJ.max()))
    gradN = np.asarray(ctx0.trial.gradN)
    print("gradN finite:", bool(np.all(np.isfinite(gradN))))
    elem_coords = np.asarray(mesh.element_coords())
    print("elem_coords finite:", bool(np.all(np.isfinite(elem_coords))))
    basis = space.basis
    detJ_all = jax.vmap(lambda Xe: basis.spatial_grads_and_detJ(Xe)[1])(jnp.asarray(elem_coords))
    detJ_all_np = np.asarray(detJ_all)
    print(
        "detJ all finite:",
        bool(np.all(np.isfinite(detJ_all_np))),
        "min/max:",
        float(detJ_all_np.min()),
        float(detJ_all_np.max()),
        "nonpos:",
        int(np.count_nonzero(detJ_all_np <= 0.0)),
    )
    u0_check = jnp.zeros(space.n_dofs, dtype=dtype)
    res_kernel = ff.make_element_residual_kernel(ff.neo_hookean_residual_form, params)
    u_elems0 = u0_check[space.elem_dofs]
    elem_res0 = jax.vmap(res_kernel)(ctxs, u_elems0)
    elem_bad = ~jnp.all(jnp.isfinite(elem_res0), axis=1)
    n_bad = int(jnp.count_nonzero(elem_bad))
    print("elem residual nonfinite:", n_bad)
    if n_bad > 0:
        bad_idx = int(np.nonzero(np.asarray(elem_bad))[0][0])
        ctx_bad = jax.tree_util.tree_map(lambda x: x[bad_idx], ctxs)
        u_bad = u0_check[space.elem_dofs[bad_idx]]
        F_bad = ff.deformation_gradient(ctx_bad, u_bad)
        J_bad = jnp.sqrt(jnp.linalg.det(ff.right_cauchy_green(F_bad)))
        print(
            "bad elem:",
            bad_idx,
            "F finite:",
            bool(jnp.all(jnp.isfinite(F_bad))),
            "J min/max:",
            float(jnp.min(J_bad)),
            float(jnp.max(J_bad)),
        )

    J_pattern = ff.make_sparsity_pattern(space, with_idx=False)

    def surface_traction_form(ctx: ff.SurfaceFormContext, traction_vec: np.ndarray) -> np.ndarray:
        return h_ts.dot(ctx.v, traction_vec)

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
        print("dirichlet nodes:", clamp_nodes.size, "dirichlet dofs:", dir_dofs.size)
        if dir_dofs.size > 0:
            print("dirichlet dofs min/max:", int(np.min(dir_dofs)), int(np.max(dir_dofs)))
        if clamp_nodes.size > 0:
            print("dirichlet nodes min/max:", int(np.min(clamp_nodes)), int(np.max(clamp_nodes)))
        if clamp_nodes.size == 0:
            raise RuntimeError("No Dirichlet facets found for dirichlet_tag; check mesh tags.")
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

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        vtu_path = os.path.join(args.output_dir, "result.vtu")
        ff.write_elastic_vtu(
            mesh, space, u, vtu_path,
            compute_j=True,
            deformed_scale=1.0
        )
        print(f"VTU written to {vtu_path}")
        img_path = render_deformed_vtu(
            vtu_path,
            output_path=os.path.join(args.output_dir, "result_deformed_x20000.png"),
            scale=20000.0,
            title="neo_hookean_cantilever",
            azimuth=35.0,
            elevation=8.0,
            view="xy",
        )
        print(f"Image written to {img_path}")


if __name__ == "__main__":
    main()

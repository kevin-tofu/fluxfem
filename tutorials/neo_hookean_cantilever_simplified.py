#!/usr/bin/env python3
"""
Hyperelastic cantilever (Neo-Hookean) — minimal FluxFEM demo.

- Uses a structured hex mesh (no external mesh file).
- Clamped at x = xmin.
- Uniform traction applied on x = xmax (y-direction).
- Newton solve, then write VTU (optional).
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import fluxfem as ff
import fluxfem.helpers_ts as h_ts

jax.config.update("jax_enable_x64", True)


def main():
    def _env_int(name: str, default: int) -> int:
        val = os.environ.get(name)
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            return default
    def _env_float(name: str, default: float) -> float:
        val = os.environ.get(name)
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            return default
    def _env_str(name: str, default: str) -> str:
        val = os.environ.get(name)
        if not val:
            return default
        return val

    # --------------------
    # Parameters (edit here)
    # --------------------
    dtype = jnp.float64

    # geometry / mesh
    nx = _env_int("FF_NX", 20)
    ny = _env_int("FF_NY", 4)
    nz = _env_int("FF_NZ", 4)
    lx, ly, lz = 100.0, 10.0, 10.0
    intorder = 2

    # material
    E, nu = 210_000.0, 0.3
    lam, mu = ff.lame_parameters(E, nu)
    params = {"lam": lam, "mu": mu}

    # loads
    traction = 1e-2               # applied on x = xmax face, along +y
    body_force = (0.0, 0.0, 0.0)  # (fx, fy, fz)

    # nonlinear solve
    nstep = _env_int("FF_NSTEP", 200)
    maxiter = _env_int("FF_MAXITER", 80)
    tol = _env_float("FF_TOL", 1e-4)
    atol = _env_float("FF_ATOL", 1e-10)
    line_search = False

    # linear solver options
    linear_solver = "cg_matfree"  # "cg", "cg_matfree", or "spsolve"
    linear_precond = _env_str("FF_PRECOND", "none")  # "none", "diag0", "jacobi", "block_jacobi"
    linear_tol = None
    matfree_mode = _env_str("FF_MATFREE_MODE", "linearize")  # "linearize" or "jvp"

    # output
    output_vtu = _env_str("FF_OUTPUT_VTU", "result.vtu")  # set None to disable

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
    dir_dofs = mesh.boundary_dofs_where(lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8), components="xyz")

    # --------------------
    # Nonlinear analysis (Neo-Hookean)
    # --------------------
    J_pattern = ff.make_sparsity_pattern(space, with_idx=False)

    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params=params,
        base_external_vector=F_ext,
        dirichlet=(dir_dofs, None),
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
        linear_tol=linear_tol,
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

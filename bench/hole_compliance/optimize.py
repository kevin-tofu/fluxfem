#!/usr/bin/env python3
"""
Compliance sensitivity for a plate-with-hole (tet mesh) via AD on hole radius.

- Linear elasticity in 3D (thin plate).
- Dirichlet clamp at x = xmin.
- Uniform traction at x = xmax (along +x).
- Shape parameter: hole radius r (mesh warp around the hole boundary).
"""

from __future__ import annotations

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

jax.config.update("jax_enable_x64", True)


def warp_coords_radial(coords: jnp.ndarray, center_xy: jnp.ndarray, r0: float, r: jnp.ndarray, delta: float) -> jnp.ndarray:
    xy = coords[:, :2] - center_xy
    rho = jnp.linalg.norm(xy, axis=1)
    smooth = jnp.exp(-((rho - r0) / delta) ** 2)
    dr = r - r0
    scale = dr * smooth / jnp.maximum(rho, 1e-12)
    xy_new = coords[:, :2] + scale[:, None] * xy
    return coords.at[:, :2].set(xy_new)


def main():
    p = argparse.ArgumentParser(description="Hole-radius compliance sensitivity (FluxFEM + AD).")
    p.add_argument("--mesh", type=str, default="bench/hole_compliance/mesh.msh")
    p.add_argument("--r0", type=float, default=0.2, help="Base hole radius used for the mesh.")
    p.add_argument("--r", type=float, default=0.2, help="Initial hole radius for optimization.")
    p.add_argument("--r-min", type=float, default=0.1)
    p.add_argument("--r-max", type=float, default=0.35)
    p.add_argument("--delta", type=float, default=0.15, help="Warp influence width around r0.")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--traction", type=float, default=1.0)
    p.add_argument("--E", type=float, default=210_000.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--fd-check", action="store_true", help="Run finite-difference check for dJ/dr.")
    p.add_argument(
        "--eps-list",
        type=str,
        default="1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,1e-6",
        help="Comma-separated eps values for FD check.",
    )
    p.add_argument(
        "--rebuild-force",
        action="store_true",
        help="Reassemble traction load F for each r using warped coords.",
    )
    p.add_argument(
        "--compare-force",
        action="store_true",
        help="Compare fixed-F vs rebuilt-F compliance/gradient at the current r.",
    )
    p.add_argument(
        "--compare-force-eps",
        type=float,
        default=1e-4,
        help="Finite-difference step for rebuild-force gradient comparison.",
    )
    p.add_argument(
        "--fd-plot",
        type=str,
        default="",
        help="Save FD relative-error plot to this path (requires matplotlib).",
    )
    args = p.parse_args()

    mesh, _, _ = ff.load_gmsh_tet_mesh(args.mesh)
    coords_np = np.asarray(mesh.coords)
    conn = jnp.asarray(mesh.conn)

    dtype = jnp.float64
    xmin = float(coords_np[:, 0].min())
    xmax = float(coords_np[:, 0].max())
    ymin = float(coords_np[:, 1].min())
    ymax = float(coords_np[:, 1].max())
    center_xy = jnp.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)], dtype=dtype)

    base_space = ff.make_tet_space_pytree(mesh, dim=3, intorder=args.intorder)
    basis = base_space.basis
    elem_dofs = base_space.elem_dofs

    bc = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    free_dofs = jnp.asarray(bc.free_dofs(base_space.n_dofs))

    facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    surface_form = ff.LinearForm.surface(lambda v, p: (v | p) * h_wf.ds())
    traction_vec = np.array([args.traction, 0.0, 0.0], dtype=float)
    surface_base = ff.make_surface_from_facets(coords_np, facets)
    F_base = surface_base.assemble_linear_form_on_space(
        base_space, surface_form.get_compiled(), params=traction_vec
    )
    F_base = jnp.asarray(F_base, dtype=dtype)

    D = ff.isotropic_3d_D(args.E, args.nu)
    coords0 = jnp.asarray(coords_np, dtype=dtype)

    def space_with_coords(coords: jnp.ndarray) -> ff.FESpacePytree:
        mesh_new = ff.TetMeshPytree(
            coords=coords,
            conn=conn,
            cell_tags=mesh.cell_tags,
            node_tags=mesh.node_tags,
        )
        return ff.FESpacePytree(
            mesh=mesh_new,
            basis=basis,
            elem_dofs=elem_dofs,
            value_dim=base_space.value_dim,
            _n_dofs=base_space._n_dofs,
            _n_ldofs=base_space._n_ldofs,
        )

    r0 = jnp.array(args.r0, dtype=dtype)
    delta = jnp.array(args.delta, dtype=dtype)

    def compliance(r: jnp.ndarray, *, rebuild_force: bool) -> jnp.ndarray:
        coords = warp_coords_radial(coords0, center_xy, r0, r, delta)
        space = space_with_coords(coords)
        K = space.assemble_bilinear_form(ff.linear_elasticity_form, params=D).to_dense()

        if rebuild_force:
            surface = ff.make_surface_from_facets(coords, facets)
            F = surface.assemble_linear_form_on_space(
                space, surface_form.get_compiled(), params=traction_vec
            )
            F = jnp.asarray(F, dtype=dtype)
        else:
            F = F_base

        K_ff = K[free_dofs][:, free_dofs]
        F_ff = F[free_dofs]
        u_free = jnp.linalg.solve(K_ff, F_ff)

        u = jnp.zeros(base_space.n_dofs, dtype=K.dtype)
        u = u.at[free_dofs].set(u_free)
        return jnp.dot(F, u)

    def compliance_active(r: jnp.ndarray) -> jnp.ndarray:
        return compliance(r, rebuild_force=args.rebuild_force)

    r = jnp.clip(jnp.array(args.r, dtype=dtype), args.r_min, args.r_max)
    if args.fd_check:
        grad_ad = jax.grad(compliance_active)(r)
        eps_list = [float(x) for x in args.eps_list.split(",") if x.strip()]
        print(f"[fd] r={float(r):.6f} grad_ad={float(grad_ad):.6e}")
        rel_errs = []
        for eps in eps_list:
            eps_j = jnp.array(eps, dtype=dtype)
            j_plus = compliance_active(r + eps_j)
            j_minus = compliance_active(r - eps_j)
            grad_fd = (j_plus - j_minus) / (2.0 * eps_j)
            rel = float(jnp.abs((grad_fd - grad_ad) / jnp.maximum(jnp.abs(grad_ad), 1e-30)))
            print(
                f"[fd] eps={eps:.1e} grad_fd={float(grad_fd):.6e} rel_err={rel:.3e}"
            )
            rel_errs.append(rel)
        if args.fd_plot:
            try:
                import matplotlib.pyplot as plt
            except Exception as exc:
                print(f"[fd] matplotlib unavailable ({exc}); skipping plot.")
            else:
                x = np.log10(np.asarray(eps_list, dtype=float))
                y = np.log10(np.asarray(rel_errs, dtype=float))
                plt.figure(figsize=(4.0, 3.0))
                plt.plot(x, y, marker="o")
                plt.xlabel("log10(eps)")
                plt.ylabel("log10(relative error)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(args.fd_plot, dpi=200)
                print(f"[fd] saved plot -> {args.fd_plot}")
    if args.compare_force:
        comp_fixed = compliance(r, rebuild_force=False)
        comp_rebuild = compliance(r, rebuild_force=True)
        grad_fixed = jax.grad(lambda rr: compliance(rr, rebuild_force=False))(r)
        eps_c = jnp.array(args.compare_force_eps, dtype=dtype)
        grad_rebuild = (
            compliance(r + eps_c, rebuild_force=True)
            - compliance(r - eps_c, rebuild_force=True)
        ) / (2.0 * eps_c)
        comp_rel = float(jnp.abs((comp_rebuild - comp_fixed) / jnp.maximum(jnp.abs(comp_fixed), 1e-30)))
        grad_rel = float(jnp.abs((grad_rebuild - grad_fixed) / jnp.maximum(jnp.abs(grad_fixed), 1e-30)))
        print(f"[force] comp_fixed={float(comp_fixed):.6e} comp_rebuild={float(comp_rebuild):.6e} rel_diff={comp_rel:.3e}")
        print(f"[force] grad_fixed={float(grad_fixed):.6e} grad_rebuild(FD)={float(grad_rebuild):.6e} rel_diff={grad_rel:.3e}")
    for step in range(args.steps + 1):
        comp = compliance_active(r)
        grad = jax.grad(compliance_active)(r)
        print(
            f"step={step:02d} r={float(r):.6f} compliance={float(comp):.6e} dJ/dr={float(grad):.6e}"
        )
        if step == args.steps:
            break
        r = jnp.clip(r - args.lr * grad, args.r_min, args.r_max)


if __name__ == "__main__":
    main()

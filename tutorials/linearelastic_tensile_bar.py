#!/usr/bin/env python3
"""
Simple tensile bar (linear elasticity) to showcase a compact weak-form workflow.

- Build a 3D hex mesh for a slender bar.
- Assemble the bilinear form from a locally defined weak form.
- Apply a uniform traction on the +x face and clamp x=0.
"""

from __future__ import annotations

import argparse
import numpy as np
import time

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff
import fluxfem.helpers_num as h_num
import fluxfem.helpers_wf as h_wf


def parse_args():
    p = argparse.ArgumentParser(description="Tensile bar demo: simple weak form + solve.")
    p.add_argument("--nx", type=int, default=12, help="Elements along x (length).")
    p.add_argument("--ny", type=int, default=2, help="Elements along y (thickness).")
    p.add_argument("--nz", type=int, default=2, help="Elements along z (width).")
    p.add_argument("--lx", type=float, default=1.0, help="Bar length.")
    p.add_argument("--ly", type=float, default=0.1, help="Bar thickness.")
    p.add_argument("--lz", type=float, default=0.1, help="Bar width.")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--E", type=float, default=210_000.0, help="Young's modulus.")
    p.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio.")
    p.add_argument("--traction", type=float, default=1.0, help="Uniform traction on +x face.")
    p.add_argument("--normal-traction", action="store_true", help="Apply traction along outward normals.")
    return p.parse_args()


def main():
    args = parse_args()

    def linear_elasticity_form(ctx: ff.FormContext, D: np.ndarray) -> ff.jnp.ndarray:
        Bu = h_num.sym_grad(ctx.u)
        Bv = h_num.sym_grad(ctx.v)
        return h_num.ddot(Bv, D, Bu)

    def surface_traction_form(ctx: ff.SurfaceFormContext, traction_vec: np.ndarray) -> np.ndarray:
        return h_num.dot(ctx.v, traction_vec)

    def surface_normal_traction_form(ctx: ff.SurfaceFormContext, traction_scalar: float) -> np.ndarray:
        normal = ctx.normal
        if normal is None:
            raise RuntimeError("surface normal is not available in context")
        return h_num.dot(ctx.v, float(traction_scalar) * normal)

    mesh = ff.StructuredHexBox(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
    ).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=args.intorder)

    # --- Weak form (scikit-fem style) --- 
    D = ff.isotropic_3d_D(args.E, args.nu)
    t0 = time.perf_counter()
    K = space.assemble_bilinear_form(linear_elasticity_form, params=D)
    print(f"[timing] assemble K (numeric): {time.perf_counter() - t0:.3f}s")

    # --- Weak form ---
    bilinear_form = ff.BilinearForm.volume(
        lambda u, v, D: h_wf.ddot(v.sym_grad, D @ u.sym_grad) * h_wf.dOmega()
    )
    t0 = time.perf_counter()
    K_wf = space.assemble_bilinear_form(
        bilinear_form.bilinear_form(),
        params=D,
    )
    print(f"[timing] assemble K_wf (weakform): {time.perf_counter() - t0:.3f}s")
    same_pattern = (
        np.array_equal(np.asarray(K.pattern.rows), np.asarray(K_wf.pattern.rows))
        and np.array_equal(np.asarray(K.pattern.cols), np.asarray(K_wf.pattern.cols))
        and K.pattern.n_dofs == K_wf.pattern.n_dofs
    )
    same_data = np.allclose(np.asarray(K.data), np.asarray(K_wf.data))
    if not (same_pattern and same_data):
        raise RuntimeError("K and K_wf do not match; check weak-form definitions.")

    # --- Linear form (surface traction) ---
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())

    def on_xmax(face: np.ndarray) -> bool:
        return np.allclose(face[:, 0], xmax, atol=1e-8)

    facets = mesh.boundary_facets_where(on_xmax)
    if facets.shape[0] == 0:
        raise RuntimeError("No facets found on +x face; check mesh/geometry.")
    surface = ff.SurfaceMesh.from_hex_mesh(mesh, facets)
    traction_form = surface_normal_traction_form if args.normal_traction else surface_traction_form
    traction_param = float(args.traction) if args.normal_traction else np.array([args.traction, 0.0, 0.0], dtype=float)

    if args.normal_traction:
        surface_form = ff.LinearForm.surface(
            lambda v, p: (v | (p * h_wf.normal())) * h_wf.ds()
        )
    else:
        surface_form = ff.LinearForm.surface(
            lambda v, p: (v | p) * h_wf.ds()
        )

    t0 = time.perf_counter()
    F_num = surface.assemble_linear_form_on_space(
        space,
        traction_form,
        params=traction_param,
    )
    print(f"[timing] assemble F_num (surface numeric): {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()
    F_wf = surface.assemble_linear_form_on_space(
        space,
        surface_form.linear_form(),
        params=traction_param,
    )
    print(f"[timing] assemble F_wf (surface weakform): {time.perf_counter() - t0:.3f}s")
    if not np.allclose(F_num, F_wf):
        raise RuntimeError("Surface linear form mismatch between numeric and form APIs.")
    F = F_wf

    # --- Dirichlet BC (clamp x=0) ---
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    solver = ff.LinearSolver(method="spsolve")
    t0 = time.perf_counter()
    u, _ = solver.solve(
        K,
        F,
        dirichlet=(dir_dofs, None),
        dirichlet_mode="condense",
    )
    print(f"[timing] solve: {time.perf_counter() - t0:.3f}s")

    u_nodes = np.asarray(u).reshape(-1, 3)
    right_nodes = mesh.axis_extrema_nodes(axis=0, side="max")
    ux_max = float(np.max(u_nodes[right_nodes, 0])) if right_nodes.size else 0.0
    ux_theory = args.traction * args.lx / args.E

    print(
        f"bar solved: dofs={space.n_dofs}, "
        f"ux_max@x=L={ux_max:.6e} (traction={args.traction})"
    )
    if ux_theory != 0.0:
        rel_err = abs(ux_max - ux_theory) / abs(ux_theory)
        print(
            f"[theory] ux=L: {ux_theory:.6e} (rel.err={rel_err:.3e})"
        )
    else:
        print(f"[theory] ux=L: {ux_theory:.6e}")


if __name__ == "__main__":
    main()

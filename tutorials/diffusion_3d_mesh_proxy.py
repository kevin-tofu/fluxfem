#!/usr/bin/env python3
"""
3D Poisson (diffusion) with an exact solution, and mesh-move proxy comparison.

Flow:
1) Compute grad wrt node coords of the true error (u_h - u_exact), fixing boundary nodes.
2) Build a ZZ-style proxy (recovered gradient mismatch) and take its coord gradient.
3) Compare the two directions (cosine + top-k overlap).
4) Save results and a side-by-side gradient plot to result/tutorials.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import fluxfem as ff


def exact_solution(coords: jnp.ndarray) -> jnp.ndarray:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)


def body_force(x_q: jnp.ndarray) -> jnp.ndarray:
    x, y, z = x_q[..., 0], x_q[..., 1], x_q[..., 2]
    return 3.0 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)


def exact_grad(x_q: jnp.ndarray) -> jnp.ndarray:
    x, y, z = x_q[..., 0], x_q[..., 1], x_q[..., 2]
    gx = jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
    gy = jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.sin(jnp.pi * z)
    gz = jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.cos(jnp.pi * z)
    return jnp.stack([gx, gy, gz], axis=-1)


def element_volumes(space) -> jnp.ndarray:
    ctxs = space.build_form_contexts()

    def per_elem(ctx):
        return jnp.sum(ctx.w * jnp.abs(ctx.test.detJ))

    return jax.vmap(per_elem)(ctxs)


def element_gradients(space, u: jnp.ndarray) -> jnp.ndarray:
    ctxs = space.build_form_contexts()
    u_elems = u[space.elem_dofs]

    def per_elem(ctx, u_elem):
        grad_q = ctx.trial.grad(u_elem)  # (n_q, 3)
        return grad_q.mean(axis=0)

    return jax.vmap(per_elem)(ctxs, u_elems)


def h1_seminorm_loss(space, u: jnp.ndarray, u_exact: jnp.ndarray, *, exact_grad_q: bool) -> jnp.ndarray:
    ctxs = space.build_form_contexts()
    u_elems = u[space.elem_dofs]
    uex_elems = u_exact[space.elem_dofs]

    def per_elem(ctx, u_elem, uex_elem):
        g_u = ctx.trial.grad(u_elem)  # (n_q, 3)
        if exact_grad_q:
            g_ex = exact_grad(ctx.x_q)
        else:
            g_ex = ctx.trial.grad(uex_elem)
        diff = g_u - g_ex
        wJ = ctx.w * jnp.abs(ctx.test.detJ)
        return jnp.sum(jnp.sum(diff * diff, axis=1) * wJ)

    return 0.5 * jnp.sum(jax.vmap(per_elem)(ctxs, u_elems, uex_elems))


def recover_nodal_gradients(
    conn: jnp.ndarray,
    elem_grad: jnp.ndarray,
    n_nodes: int,
    elem_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    grad_acc = jnp.zeros((n_nodes, 3), dtype=elem_grad.dtype)
    count = jnp.zeros((n_nodes,), dtype=elem_grad.dtype)
    if elem_weight is None:
        grad_acc = grad_acc.at[conn].add(elem_grad[:, None, :])
        count = count.at[conn].add(1.0)
    else:
        w = elem_weight[:, None]
        grad_acc = grad_acc.at[conn].add(elem_grad[:, None, :] * w[..., None])
        count = count.at[conn].add(w)
    return grad_acc / jnp.maximum(count[:, None], 1.0)


def detJ_barrier(space, eps: float = 1e-6) -> jnp.ndarray:
    ctxs = space.build_form_contexts()

    def per_elem(ctx):
        min_det = jnp.min(ctx.test.detJ)
        return jnp.square(jax.nn.softplus(eps - min_det))

    return jnp.sum(jax.vmap(per_elem)(ctxs))


def build_space(coords: jnp.ndarray, conn: jnp.ndarray, intorder: int):
    mesh = ff.HexMeshPytree(coords=coords, conn=conn, cell_tags=None, node_tags=None)
    return ff.make_hex_space_pytree(mesh, dim=1, intorder=intorder)


def solve_poisson(
    coords: jnp.ndarray,
    conn: jnp.ndarray,
    dir_dofs: jnp.ndarray,
    dir_vals: jnp.ndarray,
    intorder: int,
):
    space = build_space(coords, conn, intorder)
    K = space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense()
    rhs_form = ff.make_scalar_body_force_form(body_force)
    F = space.assemble_linear_form(rhs_form, params=None)
    bc = ff.DirichletBC(dir_dofs, dir_vals)
    K_bc, F_bc = bc.enforce_system(K, F)
    u = jnp.linalg.solve(K_bc, F_bc)
    return u, space


def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    dot = jnp.sum(a * b)
    na = jnp.sqrt(jnp.sum(a * a))
    nb = jnp.sqrt(jnp.sum(b * b))
    return dot / (na * nb + eps)


def topk_overlap(a: jnp.ndarray, b: jnp.ndarray, k: int) -> jnp.ndarray:
    _, idx_a = lax.top_k(a, k)
    _, idx_b = lax.top_k(b, k)
    return jnp.mean(jnp.isin(idx_a, idx_b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--intorder", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--topk_frac", type=float, default=0.1)
    parser.add_argument("--perturb", type=float, default=0.1)
    parser.add_argument("--outdir", type=str, default="result/tutorials/diffusion_3d_mesh_proxy")
    parser.add_argument("--slice_z", type=float, default=0.5)
    parser.add_argument("--loss", choices=("l2", "h1"), default="l2")
    parser.add_argument("--h1_exact_grad", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--alpha_c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_mesh = ff.StructuredHexBox(
        nx=args.nx, ny=args.ny, nz=args.nz, lx=1.0, ly=1.0, lz=1.0
    ).build()
    base_coords_np = np.asarray(base_mesh.coords)
    conn = jnp.asarray(base_mesh.conn)

    mins = base_coords_np.min(axis=0)
    maxs = base_coords_np.max(axis=0)
    bbox_pred = ff.bbox_predicate(mins, maxs, tol=1e-8)
    bnodes = base_mesh.node_indices_where(bbox_pred)
    bmask = np.zeros(base_coords_np.shape[0], dtype=bool)
    bmask[bnodes] = True
    interior_mask = jnp.asarray(~bmask)
    h = min(1.0 / args.nx, 1.0 / args.ny, 1.0 / args.nz)

    coords_np = base_coords_np.copy()
    if args.perturb > 0.0:
        rng = np.random.default_rng(args.seed)
        delta = args.perturb * h
        noise = rng.normal(scale=delta, size=coords_np.shape)
        noise[bmask] = 0.0
        coords_np = coords_np + noise

    coords0 = jnp.asarray(coords_np)
    dir_dofs = jnp.asarray(bnodes, dtype=jnp.int32)
    dir_vals = exact_solution(coords0[dir_dofs])

    def loss_true(coords):
        u, space = solve_poisson(coords, conn, dir_dofs, dir_vals, args.intorder)
        u_ex = exact_solution(coords)
        if args.loss == "h1":
            return h1_seminorm_loss(space, u, u_ex, exact_grad_q=args.h1_exact_grad)
        diff = u - u_ex
        m_lumped = space.assemble_mass_matrix(lumped=True)
        if m_lumped.ndim == 2:
            m_lumped = jnp.diag(m_lumped)
        return 0.5 * jnp.sum(m_lumped * diff * diff)

    def loss_proxy(coords):
        u, space = solve_poisson(coords, conn, dir_dofs, dir_vals, args.intorder)
        g_elem = element_gradients(space, u)
        vol = element_volumes(space)
        g_node = recover_nodal_gradients(conn, g_elem, coords.shape[0], elem_weight=vol)
        g_star_elem = g_node[conn].mean(axis=1)
        eta = jnp.sum(jnp.sum((g_star_elem - g_elem) ** 2, axis=1) * vol)
        return eta + args.beta * detJ_barrier(space)

    g_true = jax.grad(loss_true)(coords0) * interior_mask[:, None]
    g_proxy = jax.grad(loss_proxy)(coords0) * interior_mask[:, None]

    g_true_flat = g_true[interior_mask].reshape(-1)
    g_proxy_flat = g_proxy[interior_mask].reshape(-1)
    cos_sim = cosine_similarity(g_true_flat, g_proxy_flat)

    norm_true = jnp.linalg.norm(g_true, axis=1)
    norm_proxy = jnp.linalg.norm(g_proxy, axis=1)
    norm_true_i = norm_true[interior_mask]
    norm_proxy_i = norm_proxy[interior_mask]
    k = max(1, int(args.topk_frac * int(norm_true_i.shape[0])))
    k = min(k, int(norm_true_i.shape[0]))
    overlap = topk_overlap(norm_true_i, norm_proxy_i, k)

    os.makedirs(args.outdir, exist_ok=True)

    def _auto_alpha(g):
        g_norm = jnp.linalg.norm(g[interior_mask], axis=1)
        gmax = jnp.max(g_norm)
        alpha = args.alpha_c * h / (gmax + 1e-12)
        return alpha, gmax

    if args.alpha > 0.0:
        alpha_true = jnp.array(args.alpha, dtype=coords0.dtype)
        alpha_proxy = jnp.array(args.alpha, dtype=coords0.dtype)
        gmax_true = jnp.max(jnp.linalg.norm(g_true[interior_mask], axis=1))
        gmax_proxy = jnp.max(jnp.linalg.norm(g_proxy[interior_mask], axis=1))
        alpha_mode = "fixed"
    else:
        alpha_true, gmax_true = _auto_alpha(g_true)
        alpha_proxy, gmax_proxy = _auto_alpha(g_proxy)
        alpha_mode = "auto"

    alpha_cap = 0.25 * h
    alpha_true = jnp.minimum(alpha_true, alpha_cap)
    alpha_proxy = jnp.minimum(alpha_proxy, alpha_cap)

    coords1_true = coords0 - alpha_true * g_true
    coords1_proxy = coords0 - alpha_proxy * g_proxy

    loss_true0 = float(loss_true(coords0))
    loss_proxy0 = float(loss_proxy(coords0))
    loss_true1_true = float(loss_true(coords1_true))
    loss_true1_proxy = float(loss_true(coords1_proxy))
    loss_proxy1_true = float(loss_proxy(coords1_true))
    loss_proxy1_proxy = float(loss_proxy(coords1_proxy))

    def _rel(a0, a1):
        return (a1 - a0) / (a0 + 1e-12)

    results = {
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "intorder": args.intorder,
        "beta": args.beta,
        "perturb": args.perturb,
        "seed": args.seed,
        "loss": args.loss,
        "h1_exact_grad": bool(args.h1_exact_grad),
        "alpha_mode": alpha_mode,
        "alpha_c": float(args.alpha_c),
        "alpha_fixed": float(args.alpha),
        "alpha_true": float(alpha_true),
        "alpha_proxy": float(alpha_proxy),
        "alpha_cap": float(alpha_cap),
        "gmax_true": float(gmax_true),
        "gmax_proxy": float(gmax_proxy),
        "nodes": int(coords0.shape[0]),
        "elems": int(conn.shape[0]),
        "interior_nodes": int(interior_mask.sum()),
        "loss_true": loss_true0,
        "loss_proxy": loss_proxy0,
        "loss_true_step_true": loss_true1_true,
        "loss_true_step_proxy": loss_true1_proxy,
        "loss_proxy_step_true": loss_proxy1_true,
        "loss_proxy_step_proxy": loss_proxy1_proxy,
        "loss_true_rel_step_true": float(_rel(loss_true0, loss_true1_true)),
        "loss_true_rel_step_proxy": float(_rel(loss_true0, loss_true1_proxy)),
        "loss_proxy_rel_step_true": float(_rel(loss_proxy0, loss_proxy1_true)),
        "loss_proxy_rel_step_proxy": float(_rel(loss_proxy0, loss_proxy1_proxy)),
        "cosine_similarity": float(cos_sim),
        "topk": int(k),
        "topk_overlap": float(overlap),
    }

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    np.save(os.path.join(args.outdir, "coords.npy"), np.asarray(coords0))
    np.save(os.path.join(args.outdir, "g_true.npy"), np.asarray(g_true))
    np.save(os.path.join(args.outdir, "g_proxy.npy"), np.asarray(g_proxy))
    np.save(os.path.join(args.outdir, "interior_mask.npy"), np.asarray(interior_mask))

    try:
        import matplotlib.pyplot as plt

        coords_np = np.asarray(coords0)
        z = coords_np[:, 2]
        z_target = float(args.slice_z)
        h = min(1.0 / args.nx, 1.0 / args.ny, 1.0 / args.nz)
        slice_mask = np.abs(z - z_target) < 0.25 * h
        if not np.any(slice_mask):
            k_idx = int(np.argmin(np.abs(z - z_target)))
            z_pick = float(z[k_idx])
            slice_mask = np.isclose(z, z_pick, atol=1e-8)
        else:
            z_pick = float(np.mean(z[slice_mask]))

        xy = coords_np[slice_mask][:, :2]
        gt = np.asarray(g_true)[slice_mask][:, :2]
        gp = np.asarray(g_proxy)[slice_mask][:, :2]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        axes[0].quiver(
            xy[:, 0],
            xy[:, 1],
            gt[:, 0],
            gt[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
        )
        axes[0].set_title(f"true grad (z≈{z_pick:.3f})")
        axes[1].quiver(
            xy[:, 0],
            xy[:, 1],
            gp[:, 0],
            gp[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
        )
        axes[1].set_title(f"proxy grad (z≈{z_pick:.3f})")
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig_path = os.path.join(args.outdir, "gradients_slice.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    except Exception as exc:
        print("[warn] matplotlib plot skipped:", exc)

    print("[mesh] nodes:", coords0.shape[0], "elems:", conn.shape[0])
    print("[mesh] interior nodes:", int(interior_mask.sum()))
    print("[loss] true:", float(results["loss_true"]), "proxy:", float(results["loss_proxy"]))
    print(
        "[step] loss_true (true/proxy):",
        float(results["loss_true_step_true"]),
        float(results["loss_true_step_proxy"]),
    )
    print(
        "[step] loss_proxy (true/proxy):",
        float(results["loss_proxy_step_true"]),
        float(results["loss_proxy_step_proxy"]),
    )
    print("[compare] cosine(g_true, g_proxy) =", float(results["cosine_similarity"]))
    print(f"[compare] top-{k} overlap =", float(results["topk_overlap"]))
    print("[save] outdir:", args.outdir)


if __name__ == "__main__":
    main()

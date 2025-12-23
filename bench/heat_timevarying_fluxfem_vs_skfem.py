#!/usr/bin/env python3
"""
Compare implicit heat solves between fluxfem and scikit-fem with time-varying kappa.
kappa(t) = kappa0 * (1 + alpha * t), rebuild LHS each step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

import numpy as np

import jax
from jax import config as jax_config

import fluxfem as ff
from fluxfem.tools.timer import SectionTimer

jax_config.update("jax_enable_x64", True)
print("[JAX]", jax.devices(), "backend=", jax.default_backend())


def kappa_t(t: float, kappa0: float, alpha: float) -> float:
    return kappa0 * (1.0 + alpha * t)


def _kappa_integral(t: float, kappa0: float, alpha: float) -> float:
    return kappa0 * (t + 0.5 * alpha * t * t)


def exact_u(coords: np.ndarray, t: float, kappa0: float, alpha: float) -> np.ndarray:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    phi = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    decay = np.exp(-3.0 * (np.pi**2) * _kappa_integral(t, kappa0, alpha))
    return phi * decay


def _apply_dirichlet(A_csr, b, bc_dofs, bc_vals):
    A = A_csr.tolil(copy=True)
    b = b.copy()
    b -= A[:, bc_dofs] @ bc_vals
    A[:, bc_dofs] = 0.0
    A[bc_dofs, :] = 0.0
    A[bc_dofs, bc_dofs] = 1.0
    b[bc_dofs] = bc_vals
    return A.tocsr(), b


def _grid_from_slice(xy: np.ndarray, values: np.ndarray):
    x = xy[:, 0]
    y = xy[:, 1]
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    x_vals.sort()
    y_vals.sort()
    nx = len(x_vals)
    ny = len(y_vals)
    grid = np.full((ny, nx), np.nan)
    key_vals = {
        (round(float(xi), 12), round(float(yi), 12)): values[i]
        for i, (xi, yi) in enumerate(zip(x, y))
    }
    for iy, yv in enumerate(y_vals):
        yk = round(float(yv), 12)
        for ix, xv in enumerate(x_vals):
            xk = round(float(xv), 12)
            grid[iy, ix] = key_vals.get((xk, yk), np.nan)
    return x_vals, y_vals, grid


def save_compare_plot(
    coords: np.ndarray,
    u_exact: np.ndarray,
    u_flux: np.ndarray,
    u_skfem: np.ndarray,
    t: float,
    out_path: str,
    slice_z: float = 0.5,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    z = coords[:, 2]
    k = np.argmin(np.abs(z - slice_z))
    z_target = z[k]
    mask = np.isclose(z, z_target, atol=1e-8)
    xy = coords[mask][:, :2]
    ue = u_exact[mask]
    uf = u_flux[mask]
    us = u_skfem[mask]
    err_f = uf - ue
    err_s = us - ue
    vmax = max(np.max(np.abs(ue)), np.max(np.abs(uf)), np.max(np.abs(us)))
    emax = max(np.max(np.abs(err_f)), np.max(np.abs(err_s)))
    x_vals, y_vals, grid_e = _grid_from_slice(xy, ue)
    _, _, grid_f = _grid_from_slice(xy, uf)
    _, _, grid_s = _grid_from_slice(xy, us)
    _, _, grid_ef = _grid_from_slice(xy, err_f)
    _, _, grid_es = _grid_from_slice(xy, err_s)
    extent = [float(x_vals.min()), float(x_vals.max()), float(y_vals.min()), float(y_vals.max())]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=False)
    for ax, data, title in zip(
        axes[0],
        [grid_e, grid_f, grid_s, grid_ef, grid_es],
        [
            f"|u_exact| t={t} z≈{z_target:.2f}",
            f"|u_fluxfem| t={t} z≈{z_target:.2f}",
            f"|u_skfem| t={t} z≈{z_target:.2f}",
            f"u_flux - u_exact t={t}",
            f"u_skfem - u_exact t={t}",
        ],
    ):
        is_err = "diff" in title or "u_" in title and "u_exact" in title and "flux" in title
        if "u_flux - u_exact" in title or "u_skfem - u_exact" in title:
            vmin, vmax_use, cmap = -emax, emax, "coolwarm"
        else:
            vmin, vmax_use, cmap = -vmax, vmax, "viridis"
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_use,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[heat] saved compare plot -> {out_path}")


def save_benchmark_plot(out_path: str, backend: str, timings: dict, rel_last: dict):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    flux_asm = timings["assemble_k_flux_s"]
    sk_asm = timings["assemble_k_skfem_s"]
    flux_sol = timings["solve_flux_s"]
    sk_sol = timings["solve_skfem_s"]
    has_sk = np.isfinite(sk_asm).any() and np.isfinite(sk_sol).any()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    def _violin(ax, data, labels, title):
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#2b6cb0")
            pc.set_edgecolor("black")
            pc.set_alpha(0.5)
        ax.set_xticks(list(range(1, len(labels) + 1)), labels)
        ax.set_ylabel("seconds")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    if has_sk:
        _violin(axes[0], [flux_asm, sk_asm], ["fluxfem", "skfem"], f"Assemble K ({backend})")
        _violin(axes[1], [flux_sol, sk_sol], ["fluxfem", "skfem"], f"Solve ({backend})")
        total_means = [
            float(np.mean(flux_asm + flux_sol)),
            float(np.mean(sk_asm + sk_sol)),
        ]
        axes[2].bar(["fluxfem", "skfem"], total_means, color=["#2b6cb0", "#4a5568"])
    else:
        _violin(axes[0], [flux_asm], ["fluxfem"], f"Assemble K ({backend})")
        _violin(axes[1], [flux_sol], ["fluxfem"], f"Solve ({backend})")
        total_means = [float(np.mean(flux_asm + flux_sol))]
        axes[2].bar(["fluxfem"], total_means, color=["#2b6cb0"])
    axes[2].set_ylabel("seconds (mean)")
    axes[2].set_title(f"Assemble+Solve (mean, {backend})")
    axes[2].grid(True, axis="y", alpha=0.3)

    if has_sk:
        err_text = (
            f"rel_err_flux_last={rel_last['flux']:.3e}\n"
            f"rel_err_skfem_last={rel_last['skfem']:.3e}\n"
            f"rel_err_flux_vs_skfem_last={rel_last['flux_vs_skfem']:.3e}"
        )
        axes[1].text(
            1.02,
            0.5,
            err_text,
            transform=axes[1].transAxes,
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[heat] saved benchmark plot -> {out_path}")


def _build_fluxfem_system(size: int):
    mesh = ff.StructuredHexBox(nx=size, ny=size, nz=size, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    M = space.assemble_mass_matrix().to_csr()
    coords = np.asarray(mesh.coords)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox_pred = ff.bbox_predicate(mins, maxs, tol=1e-8)
    bc_dofs = mesh.boundary_dofs_where(bbox_pred, components=[0], dof_per_node=1)
    return space, M, coords, bc_dofs


def assemble_fluxfem_stiffness(space, kappa: float):
    # FluxFEM weak form: diffusion_form -> assemble_bilinear_form
    return space.assemble_bilinear_form(ff.diffusion_form, params=kappa)


def _build_skfem_basis(size: int, coords_flux: np.ndarray):
    import importlib.util

    if importlib.util.find_spec("skfem") is None:
        raise RuntimeError("scikit-fem not installed; please `pip install scikit-fem`.")

    from skfem import MeshHex, ElementHex1, Basis

    xs = np.linspace(0.0, 1.0, size + 1)
    ys = np.linspace(0.0, 1.0, size + 1)
    zs = np.linspace(0.0, 1.0, size + 1)
    mesh = MeshHex().init_tensor(xs, ys, zs)
    basis = Basis(mesh, ElementHex1(), intorder=2)

    coords = mesh.p.T

    def grid_key(coord: np.ndarray) -> tuple[int, int, int]:
        idx = np.rint(coord * size).astype(int)
        expected = idx / float(size)
        if not np.allclose(coord, expected, atol=1e-8):
            raise RuntimeError("Non-grid coordinate encountered when mapping nodes.")
        return int(idx[0]), int(idx[1]), int(idx[2])

    coord_map = {grid_key(c): i for i, c in enumerate(coords)}
    perm_nodes = []
    for c in coords_flux:
        key = grid_key(c)
        if key not in coord_map:
            raise RuntimeError("Failed to map scikit-fem nodes to fluxfem ordering.")
        perm_nodes.append(coord_map[key])
    perm_nodes = np.array(perm_nodes, dtype=int)
    return basis, perm_nodes


def solve_heat_compare(
    size: int,
    kappa0: float,
    alpha: float,
    t0: float,
    dt: float,
    nsteps: int,
    record_times: List[float],
    flux_solver: str = "scipy",
    *,
    include_skfem: bool = True,
):
    timer = SectionTimer()
    import scipy.sparse.linalg as sla
    space, M_flux, coords, bc_dofs = _build_fluxfem_system(size)
    if include_skfem:
        basis_sf, perm_nodes = _build_skfem_basis(size, coords)

        import skfem
        from skfem import asm
        from skfem.models.poisson import laplace
        import scipy.sparse.linalg as sla

        @skfem.BilinearForm
        def mass(u, v, w):
            return u * v

        M_sf = asm(mass, basis_sf).tocsr()
        M_sf = M_sf[perm_nodes][:, perm_nodes]
    else:
        basis_sf = None
        perm_nodes = None
        M_sf = None

    u0_nodal = exact_u(coords, t0, kappa0, alpha)
    free = ff.free_dofs(len(coords), bc_dofs)
    M_ff = M_flux[free][:, free]
    M_fd = M_flux[free][:, bc_dofs]
    rhs0 = M_flux @ u0_nodal
    rhs0_free = rhs0[free] - M_fd @ u0_nodal[bc_dofs]
    u0_free = np.asarray(sla.spsolve(M_ff, rhs0_free))
    u_flux = u0_nodal.copy()
    u_flux[free] = u0_free
    u_flux[bc_dofs] = u0_nodal[bc_dofs]
    record_flux = {t0: u_flux.copy()}
    if include_skfem:
        u_sk = u_flux.copy()
        record_sk = {t0: u_sk.copy()}
    else:
        u_sk = None
        record_sk = {}
    times_sorted = sorted(record_times)
    rec_idx = 1

    import scipy.sparse.linalg as sla

    assemble_k_flux_times = []
    assemble_k_sk = []
    solve_flux = []
    solve_sk = []

    assemble_k_flux = jax.jit(
        lambda k: assemble_fluxfem_stiffness(space, k)
    )
    for step in range(nsteps):
        t_n = t0 + step * dt
        t_np1 = t_n + dt
        k_t = kappa_t(t_np1, kappa0, alpha)

        with timer.section("assemble_flux"):
            K_flux = assemble_k_flux(k_t)
            jax.block_until_ready(K_flux.data)
            K_flux = K_flux.to_csr()
        asm_flux_dt = timer.last("assemble_flux")
        assemble_k_flux_times.append(asm_flux_dt)

        if include_skfem:
            with timer.section("assemble_skfem"):
                K_sf = (asm(laplace, basis_sf) * k_t).tocsr()
                K_sf = K_sf[perm_nodes][:, perm_nodes]
            asm_sk_dt = timer.last("assemble_skfem")
            assemble_k_sk.append(asm_sk_dt)

        A_flux = (M_flux / dt) + K_flux
        rhs_flux = (M_flux @ u_flux) / dt
        if include_skfem:
            A_sf = (M_sf / dt) + K_sf
            rhs_sf = (M_sf @ u_sk) / dt

        bc_vals = exact_u(coords[bc_dofs], t_np1, kappa0, alpha)
        A_flux_bc, rhs_flux_bc = _apply_dirichlet(A_flux, rhs_flux, bc_dofs, bc_vals)
        if include_skfem:
            A_sf_bc, rhs_sf_bc = _apply_dirichlet(A_sf, rhs_sf, bc_dofs, bc_vals)

        with timer.section("solve_flux"):
            if flux_solver == "jax":
                u_flux = ff.spdirect_solve_jax(A_flux_bc, rhs_flux_bc)
            else:
                u_flux = np.asarray(sla.spsolve(A_flux_bc, rhs_flux_bc))
        sol_flux_dt = timer.last("solve_flux")
        solve_flux.append(sol_flux_dt)

        if include_skfem:
            with timer.section("solve_skfem"):
                u_sk = np.asarray(sla.spsolve(A_sf_bc, rhs_sf_bc))
            sol_sk_dt = timer.last("solve_skfem")
            solve_sk.append(sol_sk_dt)

        if include_skfem and (step % max(1, nsteps // 10) == 0 or step == nsteps - 1):
            max_diff = float(np.max(np.abs(u_flux - u_sk)))
            print(f"[heat] step {step+1}/{nsteps} t={t_np1:.3e} max|flux-skfem|={max_diff:.3e}")

        while rec_idx < len(times_sorted) and t_np1 + 1e-12 >= times_sorted[rec_idx]:
            record_flux[times_sorted[rec_idx]] = u_flux.copy()
            if include_skfem:
                record_sk[times_sorted[rec_idx]] = u_sk.copy()
            rec_idx += 1

    timings = {
        "assemble_k_flux_s": np.asarray(assemble_k_flux_times),
        "assemble_k_skfem_s": np.asarray(assemble_k_sk),
        "solve_flux_s": np.asarray(solve_flux),
        "solve_skfem_s": np.asarray(solve_sk),
    }

    return coords, record_flux, record_sk, timings


def _run_for_backend(args, backend: str):
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = backend
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-backend",
        "--size",
        str(args.size),
        "--kappa0",
        str(args.kappa0),
        "--alpha",
        str(args.alpha),
        "--times",
        args.times,
        "--dt",
        str(args.dt),
        "--nsteps",
        str(args.nsteps),
        "--out-dir",
        args.out_dir,
    ]
    if args.plot:
        cmd.extend(["--plot", args.plot])
    print(f"[heat] running backend={backend}")
    subprocess.run(cmd, env=env, check=True)


def _make_backend_path(path: str, backend: str):
    if not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{backend}{ext}"


def main():
    p = argparse.ArgumentParser(description="3D heat (time-varying kappa): fluxfem vs scikit-fem.")
    p.add_argument("--size", type=int, default=5, help="elements per axis")
    p.add_argument("--kappa0", type=float, default=0.1, help="base diffusivity")
    p.add_argument("--alpha", type=float, default=1.0, help="kappa(t)=kappa0*(1+alpha*t)")
    p.add_argument("--times", type=str, default="0.01,0.02", help="comma-separated times to evaluate or 'all'")
    p.add_argument("--dt", type=float, default=1e-3, help="time step for implicit solve")
    p.add_argument("--nsteps", type=int, default=20, help="number of time steps")
    p.add_argument(
        "--flux-solver",
        choices=("scipy", "jax"),
        default="scipy",
        help="FluxFEM linear solver backend.",
    )
    p.add_argument(
        "--plot",
        type=str,
        default="result/bench/fluxfem_vs_skfem_timevarying/compare_steps",
        help="save comparison plots (dir or file); default saves all steps to a subfolder",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="result/bench/fluxfem_vs_skfem_timevarying",
        help="directory to save npz results",
    )
    p.add_argument("--backends", type=str, default="cpu", help="comma-separated backends to run (cpu,gpu)")
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--compare", action="store_true", help="generate CPU/GPU comparison plot from npz results")
    p.add_argument(
        "--compare-out",
        type=str,
        default="result/bench/fluxfem_vs_skfem_timevarying/compare_cpu_gpu.png",
        help="output PNG path for CPU/GPU comparison plot",
    )
    p.add_argument("--no-skfem", action="store_true", help="Skip scikit-fem comparisons.")
    args = p.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not args.single_backend and (len(backends) > 1 or (backends and backends[0] != jax.default_backend())):
        for backend in backends:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--single-backend",
                    "--size",
                    str(args.size),
                    "--kappa0",
                    str(args.kappa0),
                    "--alpha",
                    str(args.alpha),
                    "--times",
                    args.times,
                    "--dt",
                    str(args.dt),
                    "--nsteps",
                    str(args.nsteps),
                    "--flux-solver",
                    args.flux_solver,
                    "--out-dir",
                    args.out_dir,
                    "--backends",
                    backend,
                ]
                + (["--plot", args.plot] if args.plot else [])
                + (["--compare"] if args.compare else [])
                + (["--compare-out", args.compare_out] if args.compare else []),
                env={**os.environ, "JAX_PLATFORM_NAME": backend},
                check=False,
            )
            if proc.returncode != 0:
                if backend == "gpu":
                    print("[heat] GPU backend unavailable; skipping GPU run.")
                    continue
                raise SystemExit(proc.returncode)
        if args.compare:
            cpu_npz = _make_backend_path(os.path.join(args.out_dir, "results.npz"), "cpu")
            gpu_npz = _make_backend_path(os.path.join(args.out_dir, "results.npz"), "gpu")
            if os.path.exists(cpu_npz) and os.path.exists(gpu_npz):
                cpu = np.load(cpu_npz)
                gpu = np.load(gpu_npz)

                def _stats(arr):
                    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))

                cpu_asm = _stats(cpu["assemble_k_flux_s"])
                gpu_asm = _stats(gpu["assemble_k_flux_s"])
                cpu_sol = _stats(cpu["solve_flux_s"])
                gpu_sol = _stats(gpu["solve_flux_s"])
                sk_asm = _stats(cpu["assemble_k_skfem_s"])
                sk_sol = _stats(cpu["solve_skfem_s"])

                try:
                    import matplotlib.pyplot as plt
                except Exception as exc:
                    raise RuntimeError("matplotlib is required for plotting") from exc

                fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

                def _violin(ax, data, title):
                    parts = ax.violinplot(data, showmeans=True, showextrema=True)
                    for pc in parts["bodies"]:
                        pc.set_facecolor("#2b6cb0")
                        pc.set_edgecolor("black")
                        pc.set_alpha(0.5)
                    ax.set_xticks([1, 2, 3], ["FluxFEM CPU", "FluxFEM GPU", "scikit-fem"])
                    ax.set_ylabel("seconds")
                    ax.set_title(title)
                    ax.grid(True, axis="y", alpha=0.3)

                asm_data = [cpu["assemble_k_flux_s"], gpu["assemble_k_flux_s"], cpu["assemble_k_skfem_s"]]
                sol_data = [cpu["solve_flux_s"], gpu["solve_flux_s"], cpu["solve_skfem_s"]]
                _violin(axes[0], asm_data, "Assemble K (distribution)")
                _violin(axes[1], sol_data, "Solve (distribution)")

                totals = [
                    float(np.mean(cpu["assemble_k_flux_s"] + cpu["solve_flux_s"])),
                    float(np.mean(gpu["assemble_k_flux_s"] + gpu["solve_flux_s"])),
                    float(np.mean(cpu["assemble_k_skfem_s"] + cpu["solve_skfem_s"])),
                ]
                axes[2].bar(
                    ["FluxFEM CPU", "FluxFEM GPU", "scikit-fem"],
                    totals,
                    color=["#2b6cb0", "#c05621", "#4a5568"],
                )
                axes[2].set_ylabel("seconds (mean)")
                axes[2].set_title("Assemble+Solve (mean)")
                axes[2].grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                os.makedirs(os.path.dirname(args.compare_out) or ".", exist_ok=True)
                fig.savefig(args.compare_out, dpi=200)
                plt.close(fig)
                print(f"[heat] saved comparison plot -> {args.compare_out}")
        return

    if args.times.strip().lower() == "all":
        t0 = 0.0
        times = [t0 + i * args.dt for i in range(args.nsteps + 1)]
    else:
        times = [float(tok) for tok in args.times.split(",")]
        t0 = times[0]

    backend = jax.default_backend()
    include_skfem = (backend != "gpu") and (not args.no_skfem)
    if not include_skfem:
        print("[heat] scikit-fem comparison skipped on GPU backend.")

    coords, record_flux, record_sk, timings = solve_heat_compare(
        args.size,
        args.kappa0,
        args.alpha,
        t0,
        args.dt,
        args.nsteps,
        times,
        flux_solver=args.flux_solver,
        include_skfem=include_skfem,
    )

    rel_flux_list = []
    rel_sk_list = []
    rel_fs_list = []
    for t in times:
        u_flux = record_flux[t]
        u_ex = exact_u(coords, t, args.kappa0, args.alpha)
        rel_flux = float(np.linalg.norm(u_flux - u_ex) / max(np.linalg.norm(u_ex), 1e-14))
        if include_skfem:
            u_sk = record_sk[t]
            rel_sk = float(np.linalg.norm(u_sk - u_ex) / max(np.linalg.norm(u_ex), 1e-14))
            rel_fs = float(np.linalg.norm(u_flux - u_sk) / max(np.linalg.norm(u_ex), 1e-14))
            print(f"[heat] t={t} rel_err_flux={rel_flux:.3e} rel_err_sk={rel_sk:.3e} rel_flux_vs_sk={rel_fs:.3e}")
        else:
            rel_sk = float("nan")
            rel_fs = float("nan")
            print(f"[heat] t={t} rel_err_flux={rel_flux:.3e}")
        rel_flux_list.append(rel_flux)
        rel_sk_list.append(rel_sk)
        rel_fs_list.append(rel_fs)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f"results_{backend}.npz")
    flux_stack = np.stack([record_flux[t] for t in times], axis=0)
    if include_skfem:
        sk_stack = np.stack([record_sk[t] for t in times], axis=0)
    else:
        sk_stack = np.full_like(flux_stack, np.nan)
    np.savez(
        out_npz,
        coords=coords,
        times=np.asarray(times),
        flux=flux_stack,
        skfem=sk_stack,
        backend=backend,
        size=args.size,
        kappa0=args.kappa0,
        alpha=args.alpha,
        dt=args.dt,
        nsteps=args.nsteps,
        rel_err_flux=np.asarray(rel_flux_list),
        rel_err_skfem=np.asarray(rel_sk_list),
        rel_err_flux_vs_skfem=np.asarray(rel_fs_list),
        assemble_k_flux_s=timings["assemble_k_flux_s"],
        assemble_k_skfem_s=timings["assemble_k_skfem_s"],
        solve_flux_s=timings["solve_flux_s"],
        solve_skfem_s=timings["solve_skfem_s"],
    )
    print(f"[heat] saved results -> {out_npz}")

    bench_plot = os.path.join(args.out_dir, f"benchmark_{backend}.png")
    rel_last = {
        "flux": rel_flux_list[-1],
        "skfem": rel_sk_list[-1],
        "flux_vs_skfem": rel_fs_list[-1],
    }
    save_benchmark_plot(bench_plot, backend, timings, rel_last)

    if args.plot:
        plot_path = _make_backend_path(args.plot, backend)
        if plot_path.lower().endswith(".png"):
            last_t = times[-1]
            if last_t in record_flux and last_t in record_sk:
                u_flux = record_flux[last_t]
                u_sk = record_sk[last_t]
                u_ex = exact_u(coords, last_t, args.kappa0, args.alpha)
                save_compare_plot(coords, u_ex, u_flux, u_sk, last_t, plot_path, slice_z=0.5)
        else:
            steps_dir = os.path.join(plot_path, f"steps_{backend}")
            os.makedirs(steps_dir, exist_ok=True)
            for t in times:
                if t in record_flux and t in record_sk:
                    u_flux = record_flux[t]
                    u_sk = record_sk[t]
                    u_ex = exact_u(coords, t, args.kappa0, args.alpha)
                    out_path = os.path.join(steps_dir, f"compare_t{t:.3f}.png")
                    save_compare_plot(coords, u_ex, u_flux, u_sk, t, out_path, slice_z=0.5)


if __name__ == "__main__":
    main()

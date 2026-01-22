#!/usr/bin/env python3
"""
Benchmark linear solvers on a small 3D elasticity problem.

Compares:
  - SciPy spsolve (CSR direct)
  - CG (fluxfem custom) with selectable matvec (Flux/BCOO) and preconditioner
  - CG (jax.scipy) with selectable matvec (Flux/BCOO) and preconditioner

Default problem: n=16 (ny=nz=16), isotropic linear elasticity, 3x3 block clamp on xmin.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fluxfem.tools.timer import SectionTimer
from fluxfem import (  # noqa: E402
    DirichletBC,
    FluxSparseMatrix,
    StructuredHexBox,
    isotropic_3d_D,
    linear_elasticity_form,
    make_hex_space,
    spdirect_solve_cpu,
    spdirect_solve_jax,
    build_cg_operator,
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark CG/spsolve on a small 3D elasticity system.")
    p.add_argument("--n", type=int, default=16, help="Elements in x (ny=nz=n by default).")
    p.add_argument("--ny-mult", type=float, default=1.0, help="ny = round(n * ny_mult)")
    p.add_argument("--nz-mult", type=float, default=1.0, help="nz = round(n * nz_mult)")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--repeats", type=int, default=3, help="Repetitions per solver timing.")
    p.add_argument("--E", type=float, default=210_000.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--cg-tol", type=float, default=1e-8)
    p.add_argument("--cg-maxiter", type=int, default=5000)
    p.add_argument(
        "--plot",
        type=str,
        default="result/bench/bench_linear_solvers/bench_linear_solvers.png",
        help="Output PNG for timings plot.",
    )
    p.add_argument(
        "--json",
        type=str,
        default="result/bench/bench_linear_solvers/results.json",
        help="Output JSON path for results",
    )
    p.add_argument("--no-spsolve-jax", action="store_true", help="Skip jax.experimental.sparse.spsolve timing.")
    p.add_argument("--backends", type=str, default="cpu", help="Comma-separated backends to run (cpu,gpu)")
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--compare", action="store_true", help="Generate CPU/GPU comparison plot from JSON results")
    p.add_argument(
        "--compare-out",
        type=str,
        default="result/bench/bench_linear_solvers/compare_cpu_gpu.png",
        help="Output PNG path for CPU/GPU comparison plot",
    )
    return p.parse_args()


def make_structured_mesh(n: int, ny_mult: float, nz_mult: float):
    ny = max(1, int(round(n * ny_mult)))
    nz = max(1, int(round(n * nz_mult)))
    mesh = StructuredHexBox(nx=n, ny=ny, nz=nz, lx=100.0, ly=10.0, lz=10.0).build()
    return mesh, ny, nz


def compute_dirichlet_dofs(mesh) -> DirichletBC:
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    return DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
        dof_per_node=3,
        values=0.0,
    )


def build_flux_matrix(n: int, args) -> tuple[FluxSparseMatrix, np.ndarray]:
    mesh, _, _ = make_structured_mesh(n, args.ny_mult, args.nz_mult)
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=dtype),
        conn=jnp.asarray(mesh.conn, dtype=mesh.conn.dtype),
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )
    space = make_hex_space(mesh, dim=3, intorder=args.intorder)
    bc = compute_dirichlet_dofs(mesh)
    D = isotropic_3d_D(args.E, args.nu)

    assemble_K = jax.jit(
        lambda: space.assemble(
            linear_elasticity_form,
            D,
        )
    )
    K_full = assemble_K()
    jax.block_until_ready(K_full.data)
    condensed = bc.condense_system(K_full, np.zeros(K_full.n_dofs, dtype=float))
    K_ff = condensed.K
    free = condensed.free_dofs
    return K_ff, free


def make_rhs(n_free: int):
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal(n_free))


def time_solver(name: str, cg_op, b, repeats: int, *, tol: float, maxiter: int | None) -> tuple[np.ndarray, np.ndarray, object]:
    timer = SectionTimer()
    times = []
    iters = []
    last_x = None
    for _ in range(repeats):
        section_name = f"solve_{name}"
        with timer.section(section_name):
            x, info = cg_op.solve(b, tol=tol, maxiter=maxiter)
            if hasattr(x, "block_until_ready"):
                x.block_until_ready()
            else:
                jax.block_until_ready(x)
        times.append(timer.last(section_name))
        iters.append(int(info.get("iters", 0)) if info.get("iters", 0) is not None else 0)
        last_x = x
    return np.asarray(times, dtype=float), np.asarray(iters, dtype=float), last_x


def time_spsolve(K_csr, b, repeats: int):
    timer = SectionTimer()
    times = []
    for _ in range(repeats):
        with timer.section("solve_spsolve_cpu"):
            _ = spdirect_solve_cpu(K_csr, np.asarray(b))
        times.append(timer.last("solve_spsolve_cpu"))
    return np.asarray(times, dtype=float)


def time_spsolve_jax(K_flux: FluxSparseMatrix, b, repeats: int):
    timer = SectionTimer()
    times = []
    for _ in range(repeats):
        with timer.section("solve_spsolve_jax"):
            _ = spdirect_solve_jax(K_flux, b)
        times.append(timer.last("solve_spsolve_jax"))
    return np.asarray(times, dtype=float)


def summarize(arr):
    return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))


def residual_norm_numpy(A, x, b):
    b_np = np.asarray(b)
    r = A @ np.asarray(x) - b_np
    num = np.linalg.norm(r)
    den = np.linalg.norm(b_np)
    return float(num / den if den != 0 else num)


def residual_norm_jax(A, x, b):
    b_arr = jnp.asarray(b)
    x_arr = jnp.asarray(x)
    if hasattr(A, "matvec"):
        r = A.matvec(x_arr) - b_arr
    else:
        r = A @ x_arr - b_arr
    r_np = np.asarray(r)
    b_np = np.asarray(b_arr)
    num = np.linalg.norm(r_np)
    den = np.linalg.norm(b_np)
    return float(num / den if den != 0 else num)


def main():
    args = parse_args()
    def _make_backend_path(path: str, backend: str):
        root, ext = os.path.splitext(path)
        return f"{root}_{backend}{ext}"

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not args.single_backend and (len(backends) > 1 or (backends and backends[0] != jax.default_backend())):
        for backend in backends:
            env = os.environ.copy()
            env["JAX_PLATFORM_NAME"] = backend
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--single-backend",
                "--n",
                str(args.n),
                "--ny-mult",
                str(args.ny_mult),
                "--nz-mult",
                str(args.nz_mult),
                "--intorder",
                str(args.intorder),
                "--repeats",
                str(args.repeats),
                "--E",
                str(args.E),
                "--nu",
                str(args.nu),
                "--cg-tol",
                str(args.cg_tol),
                "--cg-maxiter",
                str(args.cg_maxiter),
                "--plot",
                _make_backend_path(args.plot, backend),
                "--json",
                _make_backend_path(args.json, backend),
            ]
            proc = subprocess.run(cmd, env=env, check=False)
            if proc.returncode != 0:
                if backend == "gpu":
                    print("[bench] GPU backend unavailable; skipping GPU run.")
                    continue
                raise SystemExit(proc.returncode)
        if args.compare:
            cpu_json = Path(_make_backend_path(args.json, "cpu"))
            gpu_json = Path(_make_backend_path(args.json, "gpu"))
            if cpu_json.exists() and gpu_json.exists():
                with cpu_json.open("r", encoding="utf-8") as f:
                    cpu = json.load(f)
                with gpu_json.open("r", encoding="utf-8") as f:
                    gpu = json.load(f)

                cpu_map = {r["name"]: r for r in cpu.get("results", [])}
                gpu_map = {r["name"]: r for r in gpu.get("results", [])}
                names = [n for n in cpu_map.keys() if n in gpu_map]
                x = np.arange(len(names))
                width = 0.36

                cpu_mean = np.array([cpu_map[n]["tmean"] for n in names], dtype=float)
                gpu_mean = np.array([gpu_map[n]["tmean"] for n in names], dtype=float)
                cpu_err = np.array([
                    [cpu_map[n]["tmean"] - cpu_map[n]["tmin"] for n in names],
                    [cpu_map[n]["tmax"] - cpu_map[n]["tmean"] for n in names],
                ])
                gpu_err = np.array([
                    [gpu_map[n]["tmean"] - gpu_map[n]["tmin"] for n in names],
                    [gpu_map[n]["tmax"] - gpu_map[n]["tmean"] for n in names],
                ])

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(x - width / 2, cpu_mean, width, yerr=cpu_err, capsize=4, label="CPU")
                ax.bar(x + width / 2, gpu_mean, width, yerr=gpu_err, capsize=4, label="GPU")
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=45, ha="right")
                ax.set_ylabel("Solve time [s]")
                ax.set_yscale("log")
                ax.set_title("CPU vs GPU solve time (mean with min/max)")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend()

                out_cmp = Path(args.compare_out)
                out_cmp.parent.mkdir(parents=True, exist_ok=True)
                fig.tight_layout()
                fig.savefig(out_cmp, dpi=180)
                print(f"Saved comparison plot to {out_cmp}")
        return
    K_flux, free = build_flux_matrix(args.n, args)
    K_csr = K_flux.to_csr()
    b = make_rhs(K_flux.n_dofs)

    results = []

    # SciPy direct
    sps_times = time_spsolve(K_csr, b, args.repeats)
    sps_min, sps_mean, sps_max = summarize(sps_times)
    sps_x = spdirect_solve_cpu(K_csr, np.asarray(b))
    sps_res = residual_norm_numpy(K_csr, sps_x, np.asarray(b))
    results.append(
        {"name": "spsolve", "tmin": sps_min, "tmean": sps_mean, "tmax": sps_max, "iters": None, "residual": sps_res}
    )

    # JAX direct
    if not args.no_spsolve_jax:
        sps_jax_times = time_spsolve_jax(K_flux, b, args.repeats)
        sps_jax_min, sps_jax_mean, sps_jax_max = summarize(sps_jax_times)
        sps_jax_x = spdirect_solve_jax(K_flux, b)
        sps_jax_res = residual_norm_jax(K_flux, sps_jax_x, b)
        results.append(
            {
                "name": "spsolve_jax",
                "tmin": sps_jax_min,
                "tmean": sps_jax_mean,
                "tmax": sps_jax_max,
                "iters": None,
                "residual": sps_jax_res,
            }
        )

    # CG variants
    cg_specs = []
    def _add_cg(name: str, *, matvec: str, precon, solver: str):
        try:
            cg_op = build_cg_operator(
                K_flux,
                matvec=matvec,
                preconditioner=precon,
                solver=solver,
                dof_per_node=3,
            )
        except Exception:
            return
        cg_specs.append((name, cg_op))

    _add_cg("cg_custom_flux_jacobi", matvec="flux", precon="jacobi", solver="cg")
    _add_cg("cg_custom_flux_block", matvec="flux", precon="block_jacobi", solver="cg")
    _add_cg("cg_jax_flux_jacobi", matvec="flux", precon="jacobi", solver="cg_jax")
    _add_cg("cg_jax_flux_block", matvec="flux", precon="block_jacobi", solver="cg_jax")
    _add_cg("cg_custom_bcoo_jacobi", matvec="bcoo", precon="jacobi", solver="cg")
    _add_cg("cg_custom_bcoo_block", matvec="bcoo", precon="block_jacobi", solver="cg")
    _add_cg("cg_jax_bcoo_jacobi", matvec="bcoo", precon="jacobi", solver="cg_jax")
    _add_cg("cg_jax_bcoo_block", matvec="bcoo", precon="block_jacobi", solver="cg_jax")

    print(f"Problem: n={args.n}, free DOFs={K_flux.n_dofs}, repeats={args.repeats}, dtype={'float64' if jax.config.read('jax_enable_x64') else 'float32'}")

    # Direct solve timings (already computed)
    print(f"spsolve: mean={sps_mean:.3e}s [min={sps_min:.3e}, max={sps_max:.3e}]")
    if not args.no_spsolve_jax:
        print(f"spsolve_jax: mean={sps_jax_mean:.3e}s [min={sps_jax_min:.3e}, max={sps_jax_max:.3e}]")

    # CG timings
    for name, cg_op in cg_specs:
        A = cg_op.A
        times, iters, last_x = time_solver(
            name, cg_op, b, args.repeats, tol=args.cg_tol, maxiter=args.cg_maxiter
        )
        tmin, tmean, tmax = summarize(times)
        it_med = float(np.median(iters))
        res_val = residual_norm_jax(A, last_x, b)
        print(f"{name}: mean={tmean:.3e}s [min={tmin:.3e}, max={tmax:.3e}], iters~{it_med:.1f}")
        results.append(
            {
                "name": name,
                "tmin": tmin,
                "tmean": tmean,
                "tmax": tmax,
                "iters": it_med,
                "residual": res_val,
            }
        )

    # Plot results
    if args.plot:
        from pathlib import Path
        names = [r["name"] for r in results]
        tmeans = [r["tmean"] for r in results]
        tmins = [r["tmin"] for r in results]
        tmaxs = [r["tmax"] for r in results]
        iters_vals = [r["iters"] for r in results if r["iters"] is not None]
        iter_names = [r["name"] for r in results if r["iters"] is not None]

        residuals = [r["residual"] for r in results]

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.2))
        x = np.arange(len(names))
        ax0.bar(x, tmeans, yerr=[np.array(tmeans) - np.array(tmins), np.array(tmaxs) - np.array(tmeans)], capsize=4)
        ax0.set_xticks(x)
        ax0.set_xticklabels(names, rotation=45, ha="right")
        ax0.set_ylabel("Solve time [s]")
        ax0.set_yscale("log")
        ax0.set_title("Solve time (mean with min/max)")
        ax0.grid(True, axis="y", alpha=0.3)

        if iter_names:
            xi = np.arange(len(iter_names))
            ax1.bar(xi, iters_vals)
            ax1.set_xticks(xi)
            ax1.set_xticklabels(iter_names, rotation=45, ha="right")
            ax1.set_ylabel("CG iterations (median)")
            ax1.set_title("CG iteration counts")
            ax1.grid(True, axis="y", alpha=0.3)
        else:
            ax1.axis("off")
            ax1.set_title("CG iteration counts")

        ax2.bar(x, residuals)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("Relative residual")
        ax2.set_yscale("log")
        ax2.set_title("Residual norms")
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = Path(args.plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        print(f"Saved plot to {out_path}")

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": jax.default_backend(),
        "n": args.n,
        "ny_mult": args.ny_mult,
        "nz_mult": args.nz_mult,
        "intorder": args.intorder,
        "repeats": args.repeats,
        "E": args.E,
        "nu": args.nu,
        "cg_tol": args.cg_tol,
        "cg_maxiter": args.cg_maxiter,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()

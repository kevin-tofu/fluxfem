#!/usr/bin/env python3
"""
Performance benchmark (FluxFEM CPU/GPU vs scikit-fem).

Generates a markdown-style table suitable for README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from fluxfem.tools.timer import SectionTimer


def parse_args():
    p = argparse.ArgumentParser(description="FluxFEM CPU/GPU vs scikit-fem benchmark (Poisson).")
    p.add_argument("--min-k", type=int, default=6, help="Minimum k for N=2**(k/3).")
    p.add_argument("--max-k", type=int, default=20, help="Maximum k for N=2**(k/3).")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--backends", type=str, default="cpu,gpu", help="Comma-separated backends to run (cpu,gpu).")
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--json",
        type=str,
        default="result/bench/bench_performance/results.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--plot",
        type=str,
        default="result/bench/bench_performance/compare_cpu_gpu.png",
        help="Output PNG path for comparison plot (JIT/assembly/total/memory).",
    )
    p.add_argument(
        "--plot-solve",
        type=str,
        default="result/bench/bench_performance/solve.png",
        help="Output PNG path for solve-only plot.",
    )
    p.add_argument("--no-solve", action="store_true", help="Skip linear solve timing.")
    p.add_argument(
        "--cg-tol",
        type=float,
        default=1e-8,
        help="CG tolerance for fluxfem solve timing.",
    )
    p.add_argument(
        "--cg-maxiter",
        type=int,
        default=2000,
        help="CG max iterations for fluxfem solve timing.",
    )
    p.add_argument(
        "--n-chunks",
        type=int,
        default=8,
        help="Number of chunks for FluxFEM assembly (reduces JIT compile pressure).",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat assembly timing per mesh size (JIT reuse).",
    )
    p.add_argument(
        "--include-skfem",
        action="store_true",
        default=True,
        help="Include scikit-fem baseline in compare mode (default: enabled).",
    )
    p.add_argument(
        "--no-include-skfem",
        dest="include_skfem",
        action="store_false",
        help="Disable scikit-fem baseline in compare mode.",
    )
    return p.parse_args()


def mesh_sizes(min_k: int, max_k: int):
    sizes = []
    for k in range(min_k, max_k + 1):
        n = int(2 ** (k / 3))
        sizes.append(n)
    return sizes


def run_fluxfem_backend(args, backend: str):
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = backend
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-backend",
        "--min-k",
        str(args.min_k),
        "--max-k",
        str(args.max_k),
        "--intorder",
        str(args.intorder),
        "--json",
        _make_backend_path(args.json, backend),
    ]
    subprocess.run(cmd, env=env, check=True)


def _make_backend_path(path: str, backend: str):
    root, ext = os.path.splitext(path)
    return f"{root}_{backend}{ext}"


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return int(ru.ru_maxrss)
        return int(ru.ru_maxrss * 1024)
    except Exception:
        return 0


def fluxfem_cases(args, backend: str):
    import jax
    jax.config.update("jax_enable_x64", True)
    import scipy.sparse.linalg as sla

    from fluxfem import (
        AssemblyPolicy,
        StructuredTetTensorBox,
        make_tet_space,
        diffusion_form,
        scalar_body_force_form,
        DirichletBC,
        make_element_bilinear_kernel,
        make_element_linear_kernel,
    )

    results = []
    for n in mesh_sizes(args.min_k, args.max_k):
        timer = SectionTimer()
        rss0 = _rss_bytes()
        rss_peak = rss0
        phase_peak: dict[str, int] = {}
        phase_delta: dict[str, int] = {}

        def _mark_phase(name: str, start_rss: int):
            nonlocal rss_peak
            cur = _rss_bytes()
            rss_peak = max(rss_peak, cur)
            prev = phase_peak.get(name, 0)
            if cur > prev:
                phase_peak[name] = cur
            delta = max(cur - start_rss, 0)
            prev_delta = phase_delta.get(name, 0)
            if delta > prev_delta:
                phase_delta[name] = delta

        mesh = StructuredTetTensorBox(nx=n, ny=n, nz=n, lx=1.0, ly=1.0, lz=1.0).build()
        space = make_tet_space(mesh, dim=1, intorder=args.intorder)
        bc = DirichletBC.from_bbox(mesh, components="x", tol=1e-8)
        _mark_phase("setup", rss0)
        policy = AssemblyPolicy.chunked(
            int(args.n_chunks),
            include_x_q=False,
            lightweight_context=True,
        )
        # Reuse jitted per-element kernels across warmup/repeat calls.
        bilinear_kernel = make_element_bilinear_kernel(diffusion_form, 1.0, jit=True)
        linear_kernel = make_element_linear_kernel(scalar_body_force_form, 1.0, jit=True)

        def assemble_KF():
            return space.assemble_bilinear_linear_pair(
                diffusion_form,
                1.0,
                scalar_body_force_form,
                1.0,
                policy=policy,
                bilinear_kernel=bilinear_kernel,
                linear_kernel=linear_kernel,
            )

        def assemble_K():
            K, _ = assemble_KF()
            return K

        def _block_ready(K, F):
            jax.block_until_ready(K.data)
            jax.block_until_ready(F)

        with timer.section("total"):
            K0, F0 = assemble_KF()
            _block_ready(K0, F0)
        total_time = timer.last("total")
        _mark_phase("assemble", rss0)

        assemble_times = []
        for _ in range(max(1, args.repeat)):
            with timer.section("assemble"):
                K = assemble_K()
                jax.block_until_ready(K.data)
            assemble_times.append(timer.last("assemble"))
        assemble_time = float(np.mean(assemble_times))
        jit_compile_time = max(total_time - assemble_time, 0.0)

        if args.no_solve:
            solve_time = float("nan")
            _mark_phase("condense", rss0)
        else:
            rss_before_condense = _rss_bytes()
            system = bc.condense_system(K0, F0)
            K_ff, F_free, free, dir_vals = system.K, system.F, system.free_dofs, system.dir_vals
            _mark_phase("condense", rss_before_condense)
            if K_ff.shape[0] > 1e5:
                solve_time = float("nan")
            else:
                solve_fn = lambda: sla.spsolve(K_ff, F_free)
                with timer.section("solve"):
                    solve_fn()
                solve_time = timer.last("solve")
            _mark_phase("solve", rss_before_condense)

        results.append(
            {
                "n": n,
                "dofs": int(space.n_dofs),
                "jit_compile_s": float(jit_compile_time),
                "assembly_s": float(assemble_time),
                "total_s": float(total_time),
                "solve_s": float(solve_time),
                "rss_mb": float(rss_peak / (1024.0 * 1024.0)),
                "rss_delta_mb": float(max(rss_peak - rss0, 0) / (1024.0 * 1024.0)),
                "rss_phase_mb": {
                    key: float(val / (1024.0 * 1024.0))
                    for key, val in sorted(phase_peak.items())
                },
                "rss_phase_delta_mb": {
                    key: float(val / (1024.0 * 1024.0))
                    for key, val in sorted(phase_delta.items())
                },
            }
        )

    return results


def skfem_cases(args):
    try:
        from skfem import MeshTet, ElementTetP1, Basis, asm, condense, solve  # type: ignore
        from skfem.models.poisson import laplace, unit_load  # type: ignore
    except Exception:
        return None

    def _make_skfem_tet_mesh(n: int):
        nodes = np.linspace(0.0, 1.0, n + 1)
        return MeshTet.init_tensor(*(3 * (nodes,)))

    results = []
    for n in mesh_sizes(args.min_k, args.max_k):
        timer = SectionTimer()
        rss0 = _rss_bytes()
        rss_peak = rss0
        mesh = _make_skfem_tet_mesh(n)
        basis = Basis(mesh, ElementTetP1(), intorder=args.intorder)

        def assemble():
            return laplace.assemble(basis), unit_load.assemble(basis)

        with timer.section("assemble"):
            A, b = assemble()
        assemble_time = timer.last("assemble")
        rss_peak = max(rss_peak, _rss_bytes())
        D = mesh.boundary_nodes()

        if args.no_solve or A.shape[0] > 1e5:
            solve_time = float("nan")
        else:
            with timer.section("solve"):
                solve(*condense(A, b, D=D))
            solve_time = timer.last("solve")

        results.append(
            {
                "n": n,
                "dofs": int(basis.N),
                "assembly_s": float(assemble_time),
                "total_s": float(assemble_time),
                "solve_s": float(solve_time),
                "rss_mb": float(rss_peak / (1024.0 * 1024.0)),
                "rss_delta_mb": float(max(rss_peak - rss0, 0) / (1024.0 * 1024.0)),
            }
        )
    return results


def plot_compare(cpu: dict, gpu: dict, sk: list[dict] | None, out_plot: Path, out_plot_solve: Path | None, no_solve: bool):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    def _series(payload, key):
        items = payload.get("results", [])
        dofs = np.array([r["dofs"] for r in items], dtype=float)
        vals = np.array([r.get(key, float("nan")) for r in items], dtype=float)
        return dofs, vals

    def _series_list(items, key):
        dofs = np.array([r["dofs"] for r in items], dtype=float)
        vals = np.array([r.get(key, float("nan")) for r in items], dtype=float)
        return dofs, vals

    cpu_jit = _series(cpu, "jit_compile_s")
    gpu_jit = _series(gpu, "jit_compile_s")
    cpu_asm = _series(cpu, "assembly_s")
    gpu_asm = _series(gpu, "assembly_s")
    cpu_tot = _series(cpu, "total_s")
    gpu_tot = _series(gpu, "total_s")
    cpu_sol = _series(cpu, "solve_s")
    gpu_sol = _series(gpu, "solve_s")
    cpu_mem = _series(cpu, "rss_mb")
    gpu_mem = _series(gpu, "rss_mb")
    sk_asm = (np.array([]), np.array([]))
    sk_sol = (np.array([]), np.array([]))
    sk_tot = (np.array([]), np.array([]))
    sk_mem = (np.array([]), np.array([]))
    if sk is not None:
        sk_asm = _series_list(sk, "assembly_s")
        sk_sol = _series_list(sk, "solve_s")
        sk_tot = _series_list(sk, "total_s")
        sk_mem = _series_list(sk, "rss_mb")

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(17.2, 4.2))

    if cpu_jit[0].size:
        ax0.loglog(cpu_jit[0], cpu_jit[1], "o-", label="FluxFEM CPU")
    if gpu_jit[0].size:
        ax0.loglog(gpu_jit[0], gpu_jit[1], "s--", label="FluxFEM GPU")
    ax0.set_xlabel("DOFs")
    ax0.set_ylabel("JIT compile time [s]")
    ax0.set_title("JIT compile")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend()

    if cpu_asm[0].size:
        ax1.loglog(cpu_asm[0], cpu_asm[1], "o-", label="FluxFEM CPU")
    if gpu_asm[0].size:
        ax1.loglog(gpu_asm[0], gpu_asm[1], "s--", label="FluxFEM GPU")
    if sk_asm[0].size:
        ax1.loglog(sk_asm[0], sk_asm[1], "d-.", label="scikit-fem")
    ax1.set_xlabel("DOFs")
    ax1.set_ylabel("Assembly time [s]")
    ax1.set_title("Assembly")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    if cpu_tot[0].size:
        ax2.loglog(cpu_tot[0], cpu_tot[1], "o-", label="FluxFEM CPU")
    if gpu_tot[0].size:
        ax2.loglog(gpu_tot[0], gpu_tot[1], "s--", label="FluxFEM GPU")
    if sk_tot[0].size:
        ax2.loglog(sk_tot[0], sk_tot[1], "d-.", label="scikit-fem")
    ax2.set_xlabel("DOFs")
    ax2.set_ylabel("Total time [s]")
    ax2.set_title("JIT + assembly")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    if cpu_mem[0].size:
        ax3.loglog(cpu_mem[0], cpu_mem[1], "o-", label="FluxFEM CPU")
    if gpu_mem[0].size:
        ax3.loglog(gpu_mem[0], gpu_mem[1], "s--", label="FluxFEM GPU")
    if sk_mem[0].size:
        ax3.loglog(sk_mem[0], sk_mem[1], "d-.", label="scikit-fem")
    ax3.set_xlabel("DOFs")
    ax3.set_ylabel("RSS peak [MB]")
    ax3.set_title("Memory")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    print(f"Saved comparison plot to {out_plot}")

    if not no_solve and out_plot_solve is not None:
        fig_solve, ax_solve = plt.subplots(1, 1, figsize=(6.2, 4.2))
        if cpu_sol[0].size:
            ax_solve.loglog(cpu_sol[0], cpu_sol[1], "o-", label="FluxFEM CPU")
        if gpu_sol[0].size:
            ax_solve.loglog(gpu_sol[0], gpu_sol[1], "s--", label="FluxFEM GPU")
        if sk_sol[0].size:
            ax_solve.loglog(sk_sol[0], sk_sol[1], "d-.", label="scikit-fem")
        ax_solve.set_xlabel("DOFs")
        ax_solve.set_ylabel("Solve time [s]")
        ax_solve.set_title("Solve")
        ax_solve.grid(True, which="both", alpha=0.3)
        ax_solve.legend()
        out_plot_solve.parent.mkdir(parents=True, exist_ok=True)
        fig_solve.tight_layout()
        fig_solve.savefig(out_plot_solve, dpi=180)
        print(f"Saved solve plot to {out_plot_solve}")


def main():
    args = parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if (len(backends) > 1 or (backends and backends[0] != "cpu")) and not args.single_backend:
        for backend in backends:
            run_fluxfem_backend(args, backend)

        cpu_json = Path(_make_backend_path(args.json, "cpu"))
        gpu_json = Path(_make_backend_path(args.json, "gpu"))
        cpu = json.loads(cpu_json.read_text()) if cpu_json.exists() else {}
        gpu = json.loads(gpu_json.read_text()) if gpu_json.exists() else {}
        sk = skfem_cases(args) if args.include_skfem else None

        out_json = Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cpu": cpu, "gpu": gpu, "skfem": sk, "no_solve": args.no_solve}
        out_json.write_text(json.dumps(payload, indent=2))

        if args.plot:
            plot_compare(
                cpu=cpu,
                gpu=gpu,
                sk=sk,
                out_plot=Path(args.plot),
                out_plot_solve=Path(args.plot_solve) if args.plot_solve else None,
                no_solve=args.no_solve,
            )
        return

    backend = backends[0] if backends else "cpu"
    results = fluxfem_cases(args, backend=backend)
    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {"backend": backend, "results": results, "no_solve": args.no_solve},
            indent=2
        )
    )


if __name__ == "__main__":
    main()

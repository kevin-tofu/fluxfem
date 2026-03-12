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
import time
from pathlib import Path

import numpy as np

from fluxfem.tools.timer import SectionTimer


def parse_args():
    p = argparse.ArgumentParser(description="FluxFEM CPU/GPU vs scikit-fem benchmark (Poisson).")
    p.add_argument(
        "--element",
        choices=["tet-p1", "tet-p2", "hex-p1", "hex-p2"],
        default="tet-p1",
        help="Element family/order to benchmark.",
    )
    p.add_argument("--min-k", type=int, default=6, help="Minimum k for N=2**(k/3).")
    p.add_argument("--max-k", type=int, default=20, help="Maximum k for N=2**(k/3).")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument(
        "--backends",
        type=str,
        default="cpu,gpu",
        help="Comma-separated benchmark backends to run (cpu,gpu,numpy).",
    )
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--assembly-backend",
        choices=["jax", "numpy"],
        default="jax",
        help=argparse.SUPPRESS,
    )
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


def mesh_cases(min_k: int, max_k: int):
    cases = []
    for k in range(min_k, max_k + 1):
        n = int(2 ** (k / 3))
        cases.append((k, n))
    return cases


def run_fluxfem_backend(args, backend: str):
    env = os.environ.copy()
    assembly_backend = "numpy" if backend == "numpy" else "jax"
    if assembly_backend == "jax":
        env["JAX_PLATFORM_NAME"] = backend
    else:
        env["JAX_PLATFORM_NAME"] = "cpu"
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    script_path = os.path.abspath(__file__)
    if backend == "numpy":
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_numpy.py")
    cmd = [
        sys.executable,
        script_path,
        "--min-k",
        str(args.min_k),
        "--max-k",
        str(args.max_k),
        "--intorder",
        str(args.intorder),
        "--json",
        _make_backend_path(args.json, backend),
        "--cg-tol",
        str(args.cg_tol),
        "--cg-maxiter",
        str(args.cg_maxiter),
        "--n-chunks",
        str(args.n_chunks),
        "--repeat",
        str(args.repeat),
        "--element",
        str(args.element),
    ]
    if backend != "numpy":
        cmd[2:2] = ["--single-backend", "--backends", backend, "--assembly-backend", assembly_backend]
    if args.no_solve:
        cmd.append("--no-solve")
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        print(
            f"[bench] backend '{backend}' exited with code {proc.returncode}; continuing with available partial results.",
            file=sys.stderr,
            flush=True,
        )
    return int(proc.returncode)


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


def _write_backend_json(
    path: str | Path,
    backend: str,
    results: list[dict],
    no_solve: bool,
    *,
    runtime_init_s: float | None = None,
):
    out_json = Path(path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "backend": backend,
                "results": results,
                "no_solve": no_solve,
                "runtime_init_s": runtime_init_s,
            },
            indent=2,
        )
    )


def fluxfem_cases(args, backend: str):
    if str(args.assembly_backend) == "numpy":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    import scipy.sparse.linalg as sla

    from fluxfem import (
        AssemblyPolicy,
        StructuredTetTensorBox,
        StructuredTetBox,
        StructuredHexBox,
        make_tet_space,
        make_tet10_space,
        make_hex_space,
        make_hex27_space,
        diffusion_form,
        scalar_body_force_form,
        DirichletBC,
        make_element_bilinear_kernel,
        make_element_linear_kernel,
    )

    assembly_backend = str(args.assembly_backend)
    element = str(args.element)
    runtime_init_time: float | None = None
    if assembly_backend == "numpy":
        t0 = time.perf_counter()
        jax.block_until_ready(jax.numpy.asarray(0.0))
        runtime_init_time = float(time.perf_counter() - t0)

    results = []
    for k, n in mesh_cases(args.min_k, args.max_k):
        try:
            _run_one = True
            timer = SectionTimer()
            rss0 = _rss_bytes()
            rss_peak = rss0
            phase_peak: dict[str, int] = {}
            phase_delta: dict[str, int] = {}
            print(f"[fluxfem:{backend}] k={k} N={n} start", flush=True)
        except Exception as exc:
            print(
                f"[fluxfem:{backend}] k={k} N={n} setup failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            _run_one = False
        if not _run_one:
            continue

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

        try:
            with timer.section("mesh_build"):
                if element == "tet-p1":
                    mesh = StructuredTetTensorBox(nx=n, ny=n, nz=n, lx=1.0, ly=1.0, lz=1.0).build()
                elif element == "tet-p2":
                    mesh = StructuredTetBox(nx=n, ny=n, nz=n, lx=1.0, ly=1.0, lz=1.0, order=2).build()
                elif element == "hex-p1":
                    mesh = StructuredHexBox(nx=n, ny=n, nz=n, lx=1.0, ly=1.0, lz=1.0).build()
                elif element == "hex-p2":
                    mesh = StructuredHexBox(nx=n, ny=n, nz=n, lx=1.0, ly=1.0, lz=1.0, order=3).build()
                else:
                    raise ValueError(f"Unsupported element: {element}")
            mesh_build_time = timer.last("mesh_build")

            with timer.section("space_build"):
                if element == "tet-p1":
                    space = make_tet_space(mesh, dim=1, intorder=args.intorder)
                elif element == "tet-p2":
                    space = make_tet10_space(mesh, dim=1, intorder=args.intorder)
                elif element == "hex-p1":
                    space = make_hex_space(mesh, dim=1, intorder=args.intorder)
                elif element == "hex-p2":
                    space = make_hex27_space(mesh, dim=1, intorder=args.intorder)
                else:
                    raise ValueError(f"Unsupported element: {element}")
            space_build_time = timer.last("space_build")
            setup_build_time = mesh_build_time + space_build_time
            print(f"[fluxfem:{backend}] k={k} N={n} dofs={space.n_dofs}", flush=True)
            bc = DirichletBC.from_bbox(mesh, components="x", tol=1e-8)
            _mark_phase("setup", rss0)
            policy = None
            if assembly_backend == "jax":
                policy = AssemblyPolicy.chunked(
                    int(args.n_chunks),
                    include_x_q=False,
                    lightweight_context=True,
                )
            with timer.section("pattern_build"):
                pattern = space.get_sparsity_pattern(with_idx=True)
            pattern_build_time = timer.last("pattern_build")
            # Reuse per-element kernels across warmup/repeat calls.
            bilinear_kernel = make_element_bilinear_kernel(
                diffusion_form,
                1.0,
                jit=(assembly_backend == "jax"),
            )
            linear_kernel = make_element_linear_kernel(
                scalar_body_force_form,
                1.0,
                jit=(assembly_backend == "jax"),
            )

            def assemble_KF():
                return space.assemble_bilinear_linear_pair(
                    diffusion_form,
                    1.0,
                    scalar_body_force_form,
                    1.0,
                    backend=assembly_backend,
                    policy=policy,
                    pattern=pattern,
                    bilinear_kernel=bilinear_kernel,
                    linear_kernel=linear_kernel,
                )

            def assemble_K():
                K, _ = assemble_KF()
                return K

            def _block_ready(K, F):
                if assembly_backend == "jax":
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
                    if assembly_backend == "jax":
                        jax.block_until_ready(K.data)
                assemble_times.append(timer.last("assemble"))
            assemble_time = float(np.mean(assemble_times))
            first_call_overhead_time = max(total_time - assemble_time, 0.0)
            first_assemble_extra_time = max(first_call_overhead_time - setup_build_time, 0.0)

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
                    "k": int(k),
                    "n": n,
                    "element": element,
                    "dofs": int(space.n_dofs),
                    "assembly_backend": assembly_backend,
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
            if assembly_backend == "jax":
                results[-1]["jit_compile_s"] = float(first_call_overhead_time)
            else:
                results[-1]["mesh_build_s"] = float(mesh_build_time)
                results[-1]["space_build_s"] = float(space_build_time)
                results[-1]["pattern_build_s"] = float(pattern_build_time)
                results[-1]["setup_build_s"] = float(setup_build_time)
                results[-1]["first_call_overhead_s"] = float(first_call_overhead_time)
                results[-1]["first_assemble_extra_s"] = float(first_assemble_extra_time)
                results[-1]["first_assemble_core_s"] = float(
                    max(first_assemble_extra_time - pattern_build_time, 0.0)
                )
            if args.single_backend:
                _write_backend_json(
                    args.json,
                    backend,
                    results,
                    args.no_solve,
                    runtime_init_s=runtime_init_time,
                )
            print(
                (
                    f"[fluxfem:{backend}] k={k} N={n} done "
                    f"(assembly_backend={assembly_backend}, assembly={assemble_time:.3f}s, "
                    f"solve={solve_time:.3f}s, rss={rss_peak / (1024.0 * 1024.0):.1f}MB)"
                ),
                flush=True,
            )
            if assembly_backend == "numpy":
                print(
                    (
                        f"[fluxfem:{backend}] phases "
                        f"(runtime_init={runtime_init_time or 0.0:.3f}s, "
                        f"setup={setup_build_time:.3f}s, "
                        f"pattern={pattern_build_time:.3f}s, "
                        f"first_assemble={max(first_assemble_extra_time - pattern_build_time, 0.0):.3f}s)"
                    ),
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[fluxfem:{backend}] k={k} N={n} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

    return results, runtime_init_time


def skfem_cases(args):
    try:
        from skfem import MeshTet, MeshHex, Basis, asm, condense, solve  # type: ignore
        from skfem.models.poisson import laplace, unit_load  # type: ignore
        try:
            from skfem import ElementTetP1, ElementTetP2, ElementHex1, ElementHex2  # type: ignore
        except Exception:
            from skfem.element import ElementTetP1, ElementTetP2, ElementHex1, ElementHex2  # type: ignore
    except Exception:
        return None

    element = str(args.element)

    def _make_skfem_tet_tensor_mesh(n: int):
        nodes = np.linspace(0.0, 1.0, n + 1)
        return MeshTet.init_tensor(*(3 * (nodes,)))

    results = []
    for k, n in mesh_cases(args.min_k, args.max_k):
        try:
            timer = SectionTimer()
            rss0 = _rss_bytes()
            rss_peak = rss0
            print(f"[scikit-fem] k={k} N={n} start", flush=True)
            if element == "tet-p1":
                mesh = _make_skfem_tet_tensor_mesh(n)
                basis = Basis(mesh, ElementTetP1(), intorder=args.intorder)
            elif element == "tet-p2":
                mesh = _make_skfem_tet_tensor_mesh(n)
                basis = Basis(mesh, ElementTetP2(), intorder=args.intorder)
            elif element == "hex-p1":
                nodes = np.linspace(0.0, 1.0, n + 1)
                mesh = MeshHex.init_tensor(*(3 * (nodes,)))
                basis = Basis(mesh, ElementHex1(), intorder=args.intorder)
            elif element == "hex-p2":
                nodes = np.linspace(0.0, 1.0, n + 1)
                mesh = MeshHex.init_tensor(*(3 * (nodes,)))
                basis = Basis(mesh, ElementHex2(), intorder=args.intorder)
            else:
                raise ValueError(f"Unsupported element: {element}")
            print(f"[scikit-fem] k={k} N={n} dofs={basis.N}", flush=True)

            def assemble():
                return laplace.assemble(basis), unit_load.assemble(basis)

            with timer.section("assemble"):
                A, b = assemble()
            assemble_time = timer.last("assemble")
            rss_peak = max(rss_peak, _rss_bytes())
            D = mesh.boundary_nodes()

            if args.no_solve or A.shape[0] > 1e5 or element.endswith("p2"):
                solve_time = float("nan")
            else:
                with timer.section("solve"):
                    solve(*condense(A, b, D=D))
                solve_time = timer.last("solve")

            results.append(
                {
                    "k": int(k),
                    "n": n,
                    "element": element,
                    "dofs": int(basis.N),
                    "assembly_s": float(assemble_time),
                    "total_s": float(assemble_time),
                    "solve_s": float(solve_time),
                    "rss_mb": float(rss_peak / (1024.0 * 1024.0)),
                    "rss_delta_mb": float(max(rss_peak - rss0, 0) / (1024.0 * 1024.0)),
                }
            )
            print(
                f"[scikit-fem] k={k} N={n} done (assembly={assemble_time:.3f}s, solve={solve_time:.3f}s, rss={rss_peak / (1024.0 * 1024.0):.1f}MB)",
                flush=True,
            )
        except Exception as exc:
            print(f"[scikit-fem] k={k} N={n} failed: {exc}", file=sys.stderr, flush=True)
            continue
    return results


def plot_compare(
    flux_payloads: dict[str, dict],
    sk: list[dict] | None,
    out_plot: Path,
    out_plot_solve: Path | None,
    no_solve: bool,
):
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

    def _numpy_phase_series(payload):
        items = payload.get("results", [])
        dofs = np.array([r["dofs"] for r in items], dtype=float)
        runtime = np.full(dofs.shape, float(payload.get("runtime_init_s", float("nan"))), dtype=float)
        setup = np.array([r.get("setup_build_s", float("nan")) for r in items], dtype=float)
        pattern = np.array([r.get("pattern_build_s", float("nan")) for r in items], dtype=float)
        first_assemble = np.array([r.get("first_assemble_core_s", float("nan")) for r in items], dtype=float)
        return dofs, runtime, setup, pattern, first_assemble

    styles = {
        "cpu": ("o-", "FluxFEM CPU"),
        "gpu": ("s--", "FluxFEM GPU"),
        "numpy": ("^-", "FluxFEM NumPy"),
    }
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

    for backend, payload in flux_payloads.items():
        style, label = styles.get(backend, ("o-", f"FluxFEM {backend}"))
        items = payload.get("results", [])
        if items and items[0].get("assembly_backend") == "numpy":
            dofs, runtime, setup, pattern, first_assemble = _numpy_phase_series(payload)
            if dofs.size:
                ax0.loglog(dofs, runtime, "o-", label=f"{label} runtime_init")
                ax0.loglog(dofs, setup, "s-", label=f"{label} setup")
                ax0.loglog(dofs, pattern, "^-", label=f"{label} pattern")
                ax0.loglog(dofs, first_assemble, "d-", label=f"{label} first_assemble")
        else:
            dofs, vals = _series(payload, "jit_compile_s")
            if dofs.size:
                ax0.loglog(dofs, vals, style, label=label)
    ax0.set_xlabel("DOFs")
    ax0.set_ylabel("Initial overhead [s]")
    ax0.set_title("Init Breakdown")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend()

    for backend, payload in flux_payloads.items():
        style, label = styles.get(backend, ("o-", f"FluxFEM {backend}"))
        dofs, vals = _series(payload, "assembly_s")
        if dofs.size:
            ax1.loglog(dofs, vals, style, label=label)
    if sk_asm[0].size:
        ax1.loglog(sk_asm[0], sk_asm[1], "d-.", label="scikit-fem")
    ax1.set_xlabel("DOFs")
    ax1.set_ylabel("Assembly time [s]")
    ax1.set_title("Assembly")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    for backend, payload in flux_payloads.items():
        style, label = styles.get(backend, ("o-", f"FluxFEM {backend}"))
        dofs, vals = _series(payload, "total_s")
        if dofs.size:
            ax2.loglog(dofs, vals, style, label=label)
    if sk_tot[0].size:
        ax2.loglog(sk_tot[0], sk_tot[1], "d-.", label="scikit-fem")
    ax2.set_xlabel("DOFs")
    ax2.set_ylabel("Total time [s]")
    ax2.set_title("JIT + assembly")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    for backend, payload in flux_payloads.items():
        style, label = styles.get(backend, ("o-", f"FluxFEM {backend}"))
        dofs, vals = _series(payload, "rss_mb")
        if dofs.size:
            ax3.loglog(dofs, vals, style, label=label)
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
        for backend, payload in flux_payloads.items():
            style, label = styles.get(backend, ("o-", f"FluxFEM {backend}"))
            dofs, vals = _series(payload, "solve_s")
            if dofs.size:
                ax_solve.loglog(dofs, vals, style, label=label)
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
        backend_exit_codes: dict[str, int] = {}
        for backend in backends:
            backend_exit_codes[backend] = run_fluxfem_backend(args, backend)
        flux_payloads = {}
        for backend in backends:
            backend_json = Path(_make_backend_path(args.json, backend))
            flux_payloads[backend] = json.loads(backend_json.read_text()) if backend_json.exists() else {}
        sk = skfem_cases(args) if args.include_skfem else None

        out_json = Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backends": flux_payloads,
            "skfem": sk,
            "no_solve": args.no_solve,
            "backend_exit_codes": backend_exit_codes,
        }
        for backend, backend_payload in flux_payloads.items():
            payload[backend] = backend_payload
        out_json.write_text(json.dumps(payload, indent=2))

        if args.plot:
            try:
                plot_compare(
                    flux_payloads=flux_payloads,
                    sk=sk,
                    out_plot=Path(args.plot),
                    out_plot_solve=Path(args.plot_solve) if args.plot_solve else None,
                    no_solve=args.no_solve,
                )
            except Exception as exc:
                print(f"[bench] plot generation failed: {exc}", file=sys.stderr, flush=True)
        return

    backend = backends[0] if backends else "cpu"
    results, runtime_init_time = fluxfem_cases(args, backend=backend)
    _write_backend_json(
        args.json,
        backend,
        results,
        args.no_solve,
        runtime_init_s=runtime_init_time,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Performance benchmark (FluxFEM pad vs nopad).

Based on bench/performance.py, but compares n_chunks (pad) vs None (nopad).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from fluxfem.tools.timer import SectionTimer


def parse_args():
    p = argparse.ArgumentParser(description="FluxFEM pad vs nopad benchmark (Poisson).")
    p.add_argument("--min-k", type=int, default=6, help="Minimum k for N=2**(k/3).")
    p.add_argument("--max-k", type=int, default=20, help="Maximum k for N=2**(k/3).")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--backends", type=str, default="cpu", help="Comma-separated backends to run (cpu,gpu).")
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=["compare", "pad", "nopad"], default="compare")
    p.add_argument(
        "--n-chunks",
        type=int,
        default=64,
        help="Number of chunks for padded assembly (n_chunks).",
    )
    p.add_argument(
        "--json",
        type=str,
        default="result/bench/bench_performance_pad/results.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--plot",
        type=str,
        default="",
        help="Optional PNG path for comparison plot (assembly/total/memory).",
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
    p.add_argument("--no-solve", action="store_true", help="Skip linear solve timing.")
    p.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat assembly timing per mesh size (JIT reuse).",
    )
    return p.parse_args()


def mesh_sizes(min_k: int, max_k: int):
    sizes = []
    for k in range(min_k, max_k + 1):
        n = int(2 ** (k / 3))
        sizes.append((k, n))
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
        "--mode",
        str(args.mode),
        "--n-chunks",
        str(args.n_chunks),
        "--repeat",
        str(args.repeat),
        "--json",
        _make_backend_path(args.json, backend),
    ]
    if args.no_solve:
        cmd.append("--no-solve")
    subprocess.run(cmd, env=env, check=True)


def _make_backend_path(path: str, backend: str):
    root, ext = os.path.splitext(path)
    return f"{root}_{backend}{ext}"


def skfem_cases(args):
    try:
        from skfem import MeshTet, ElementTetP1, Basis, asm  # type: ignore
        from skfem.models.poisson import laplace, unit_load  # type: ignore
    except Exception:
        return None

    def _make_skfem_tet_mesh(n: int):
        nodes = np.linspace(0.0, 1.0, n + 1)
        return MeshTet.init_tensor(*(3 * (nodes,)))

    results = []
    for k, n in mesh_sizes(args.min_k, args.max_k):
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

        results.append(
            {
                "k": int(k),
                "n": n,
                "dofs": int(basis.N),
                "assembly_s": float(assemble_time),
                "total_s": float(assemble_time),
                "rss_mb": float(rss_peak / (1024.0 * 1024.0)),
                "rss_delta_mb": float(max(rss_peak - rss0, 0) / (1024.0 * 1024.0)),
            }
        )
        print(
            f"[mem][skfem] k={k} n={n} dofs={int(basis.N)} "
            f"rss_mb={rss_peak / (1024.0 * 1024.0):.1f} "
            f"rss_delta_mb={max(rss_peak - rss0, 0) / (1024.0 * 1024.0):.1f}",
            flush=True,
        )
    return results


def fluxfem_cases(args, backend: str, *, n_chunks: int | None):
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
    for k, n in mesh_sizes(args.min_k, args.max_k):
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

        selected_chunks = n_chunks

        mesh = StructuredTetTensorBox(nx=n, ny=n, nz=n, lx=1.0, ly=1.0, lz=1.0).build()
        space = make_tet_space(mesh, dim=1, intorder=args.intorder)
        bc = DirichletBC.from_bbox(mesh, components="x", tol=1e-8)
        _mark_phase("setup", rss0)
        policy = None
        if selected_chunks is not None:
            policy = AssemblyPolicy.chunked(
                int(selected_chunks),
                include_x_q=False,
                lightweight_context=True,
                chunk_build_context=True,
            )
        # Reuse jitted per-element kernels across warmup/repeat calls.
        bilinear_kernel = make_element_bilinear_kernel(diffusion_form, 1.0, jit=True)
        linear_kernel = make_element_linear_kernel(scalar_body_force_form, 1.0, jit=True)
        elem_data_prefetched = None
        if selected_chunks is None:
            rss_before_build = _rss_bytes()
            elem_data_prefetched = space.build_form_contexts(
                dep=None,
                include_x_q=False,
                lightweight=True,
            )
            _mark_phase("build_context", rss_before_build)
        else:
            # Chunked path builds contexts incrementally; no global prebuild phase.
            _mark_phase("build_context", _rss_bytes())

        def assemble_KF():
            return space.assemble_bilinear_linear_pair(
                diffusion_form,
                1.0,
                scalar_body_force_form,
                1.0,
                policy=policy,
                elem_data=elem_data_prefetched,
                bilinear_kernel=bilinear_kernel,
                linear_kernel=linear_kernel,
            )

        def assemble_K():
            K, _ = assemble_KF()
            return K

        assemble_K_run = assemble_K

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
                K = assemble_K_run()
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
                "k": int(k),
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
                "n_chunks": None if selected_chunks is None else int(selected_chunks),
            }
        )
        mode = "nopad" if selected_chunks is None else f"pad(n_chunks={int(selected_chunks)})"
        print(
            f"[mem][{backend}][{mode}] k={k} n={n} dofs={int(space.n_dofs)} "
            f"rss_mb={rss_peak / (1024.0 * 1024.0):.1f} "
            f"rss_delta_mb={max(rss_peak - rss0, 0) / (1024.0 * 1024.0):.1f} "
            f"phase_delta={{"
            + ", ".join(
                f"{k0}:{v0 / (1024.0 * 1024.0):.1f}"
                for k0, v0 in sorted(phase_delta.items())
            )
            + "}",
            flush=True,
        )
        gc.collect()

    return results


def _plot_compare(payload, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    def _series(items, key):
        dofs = np.array([r["dofs"] for r in items], dtype=float)
        vals = np.array([r.get(key, float("nan")) for r in items], dtype=float)
        return dofs, vals

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14.2, 4.2))
    pad = payload.get("pad", [])
    nopad = payload.get("nopad", [])
    skfem = payload.get("skfem", []) or []

    if pad:
        dofs, vals = _series(pad, "assembly_s")
        ax0.loglog(dofs, vals, "o-", label="pad")
    if nopad:
        dofs, vals = _series(nopad, "assembly_s")
        ax0.loglog(dofs, vals, "s--", label="nopad")
    if skfem:
        dofs, vals = _series(skfem, "assembly_s")
        ax0.loglog(dofs, vals, "d-.", label="scikit-fem")
    ax0.set_xlabel("DOFs")
    ax0.set_ylabel("Assembly time [s]")
    ax0.set_title("Assembly")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend()

    if pad:
        dofs, vals = _series(pad, "total_s")
        ax1.loglog(dofs, vals, "o-", label="pad")
    if nopad:
        dofs, vals = _series(nopad, "total_s")
        ax1.loglog(dofs, vals, "s--", label="nopad")
    if skfem:
        dofs, vals = _series(skfem, "total_s")
        ax1.loglog(dofs, vals, "d-.", label="scikit-fem")
    ax1.set_xlabel("DOFs")
    ax1.set_ylabel("Total time [s]")
    ax1.set_title("JIT + assembly")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    if pad:
        dofs, vals = _series(pad, "rss_mb")
        ax2.loglog(dofs, vals, "o-", label="pad")
    if nopad:
        dofs, vals = _series(nopad, "rss_mb")
        ax2.loglog(dofs, vals, "s--", label="nopad")
    if skfem:
        dofs, vals = _series(skfem, "rss_mb")
        ax2.loglog(dofs, vals, "d-.", label="scikit-fem")
    ax2.set_xlabel("DOFs")
    ax2.set_ylabel("RSS [MB]")
    ax2.set_title("Memory (sampled peak RSS)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Saved comparison plot to {out_path}")


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


def main():
    args = parse_args()
    fixed_chunks: int | None = None
    if args.n_chunks <= 0:
        raise ValueError("--n-chunks must be positive")
    fixed_chunks = int(args.n_chunks)
    plot_path = None
    if args.plot:
        plot_path = Path(args.plot)
        if not plot_path.is_absolute() and plot_path.parent == Path("."):
            plot_path = Path(args.json).resolve().parent / plot_path
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if (len(backends) > 1 or (backends and backends[0] != "cpu")) and not args.single_backend:
        for backend in backends:
            run_fluxfem_backend(args, backend)

        sk = skfem_cases(args) if args.include_skfem else None
        out_json = Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"skfem": sk, "no_solve": args.no_solve}
        for backend in backends:
            backend_json = Path(_make_backend_path(args.json, backend))
            payload[backend] = json.loads(backend_json.read_text()) if backend_json.exists() else {}
        out_json.write_text(json.dumps(payload, indent=2))

        if plot_path and args.mode == "compare":
            for backend in backends:
                data = payload.get(backend, {})
                if not data:
                    continue
                out_plot = plot_path.with_name(f"{plot_path.stem}_{backend}{plot_path.suffix}")
                _plot_compare(
                    {"pad": data.get("pad", []), "nopad": data.get("nopad", []), "skfem": sk},
                    out_plot,
                )
        return

    backend = backends[0] if backends else "cpu"
    pad = None
    nopad = None
    if args.mode in ("compare", "pad"):
        pad = fluxfem_cases(args, backend=backend, n_chunks=fixed_chunks)
    if args.mode in ("compare", "nopad"):
        nopad = fluxfem_cases(args, backend=backend, n_chunks=None)

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    sk = skfem_cases(args) if args.include_skfem and args.mode == "compare" else None
    if args.mode == "compare":
        payload = {
            "backend": backend,
            "pad": pad,
            "nopad": nopad,
            "skfem": sk,
            "no_solve": args.no_solve,
        }
    elif args.mode == "pad":
        payload = {"backend": backend, "results": pad, "no_solve": args.no_solve}
    else:
        payload = {"backend": backend, "results": nopad, "no_solve": args.no_solve}
    out_json.write_text(json.dumps(payload, indent=2))

    if plot_path and args.mode == "compare":
        _plot_compare(
            {"pad": pad or [], "nopad": nopad or [], "skfem": sk or []},
            plot_path,
        )


if __name__ == "__main__":
    main()

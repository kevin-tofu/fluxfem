#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="FluxFEM NumPy benchmark (lightweight entrypoint).")
    p.add_argument(
        "--element",
        choices=["tet-p1", "tet-p2", "hex-p1", "hex-p2"],
        default="tet-p1",
    )
    p.add_argument("--min-k", type=int, default=6)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument(
        "--json",
        type=str,
        default="result/bench/bench_performance/results_numpy.json",
    )
    p.add_argument("--no-solve", action="store_true")
    p.add_argument("--cg-tol", type=float, default=1e-8)
    p.add_argument("--cg-maxiter", type=int, default=2000)
    p.add_argument("--n-chunks", type=int, default=8)
    p.add_argument("--repeat", type=int, default=3)
    return p.parse_args()


def mesh_cases(min_k: int, max_k: int):
    cases = []
    for k in range(min_k, max_k + 1):
        n = int(2 ** (k / 3))
        cases.append((k, n))
    return cases


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


def _write_backend_json(path: str | Path, results: list[dict], no_solve: bool, runtime_init_s: float | None):
    out_json = Path(path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "backend": "numpy",
                "results": results,
                "no_solve": no_solve,
                "runtime_init_s": runtime_init_s,
            },
            indent=2,
        )
    )


def fluxfem_cases_numpy(args):
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import jax

    from fluxfem.tools.timer import SectionTimer
    from fluxfem.mesh.hex import StructuredHexBox
    from fluxfem.mesh.tet import StructuredTetBox, StructuredTetTensorBox
    from fluxfem.core.space import (
        make_hex27_space,
        make_hex_space,
        make_tet10_space,
        make_tet_space,
    )
    from fluxfem.core.assembly import (
        make_element_bilinear_kernel,
        make_element_linear_kernel,
        scalar_body_force_form,
    )
    from fluxfem.physics.diffusion import diffusion_form
    jax.config.update("jax_enable_x64", True)
    t0 = time.perf_counter()
    jax.block_until_ready(jax.numpy.asarray(0.0))
    runtime_init_time = float(time.perf_counter() - t0)

    sla = None
    DirichletBC = None
    if not args.no_solve:
        import scipy.sparse.linalg as sla  # type: ignore
        from fluxfem.solver.dirichlet import DirichletBC  # type: ignore

    results = []
    element = str(args.element)
    for k, n in mesh_cases(args.min_k, args.max_k):
        timer = SectionTimer()
        rss0 = _rss_bytes()
        rss_peak = rss0
        phase_peak: dict[str, int] = {}
        phase_delta: dict[str, int] = {}
        print(f"[fluxfem:numpy-lite] k={k} N={n} start", flush=True)

        def _mark_phase(name: str, start_rss: int):
            nonlocal rss_peak
            cur = _rss_bytes()
            rss_peak = max(rss_peak, cur)
            phase_peak[name] = max(phase_peak.get(name, 0), cur)
            phase_delta[name] = max(phase_delta.get(name, 0), max(cur - start_rss, 0))

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
        print(f"[fluxfem:numpy-lite] k={k} N={n} dofs={space.n_dofs}", flush=True)

        bc = None if DirichletBC is None else DirichletBC.from_bbox(mesh, components="x", tol=1e-8)
        _mark_phase("setup", rss0)

        with timer.section("pattern_build"):
            pattern = space.get_sparsity_pattern(with_idx=True)
        pattern_build_time = timer.last("pattern_build")

        bilinear_kernel = make_element_bilinear_kernel(diffusion_form, 1.0, jit=False)
        linear_kernel = make_element_linear_kernel(scalar_body_force_form, 1.0, jit=False)

        def assemble_KF():
            # Intentional same-space benchmark target: keep the convenience
            # pair path here so NumPy timings stay comparable to the JAX path.
            return space.assemble_bilinear_linear_pair(
                diffusion_form,
                1.0,
                scalar_body_force_form,
                1.0,
                backend="numpy",
                pattern=pattern,
                bilinear_kernel=bilinear_kernel,
                linear_kernel=linear_kernel,
            )

        def assemble_K():
            K, _ = assemble_KF()
            return K

        with timer.section("total"):
            K0, F0 = assemble_KF()
        total_time = timer.last("total")
        _mark_phase("assemble", rss0)

        assemble_times = []
        for _ in range(max(1, args.repeat)):
            with timer.section("assemble"):
                K = assemble_K()
            assemble_times.append(timer.last("assemble"))
        assemble_time = float(np.mean(assemble_times))
        first_call_overhead_time = max(total_time - assemble_time, 0.0)
        first_assemble_extra_time = max(first_call_overhead_time - setup_build_time, 0.0)
        first_assemble_core_time = max(first_assemble_extra_time - pattern_build_time, 0.0)

        if args.no_solve:
            solve_time = float("nan")
            _mark_phase("condense", rss0)
        else:
            rss_before_condense = _rss_bytes()
            assert bc is not None
            assert sla is not None
            system = bc.condense_system(K0, F0)
            K_ff, F_free = system.K, system.F
            _mark_phase("condense", rss_before_condense)
            if K_ff.shape[0] > 1e5:
                solve_time = float("nan")
            else:
                with timer.section("solve"):
                    sla.spsolve(K_ff, F_free)
                solve_time = timer.last("solve")
            _mark_phase("solve", rss_before_condense)

        results.append(
            {
                "k": int(k),
                "n": n,
                "element": element,
                "dofs": int(space.n_dofs),
                "assembly_backend": "numpy",
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
                "mesh_build_s": float(mesh_build_time),
                "space_build_s": float(space_build_time),
                "pattern_build_s": float(pattern_build_time),
                "setup_build_s": float(setup_build_time),
                "first_call_overhead_s": float(first_call_overhead_time),
                "first_assemble_extra_s": float(first_assemble_extra_time),
                "first_assemble_core_s": float(first_assemble_core_time),
            }
        )
        print(
            (
                f"[fluxfem:numpy-lite] k={k} N={n} done "
                f"(assembly={assemble_time:.3f}s, solve={solve_time:.3f}s, "
                f"rss={rss_peak / (1024.0 * 1024.0):.1f}MB)"
            ),
            flush=True,
        )
        print(
            (
                f"[fluxfem:numpy-lite] phases "
                f"(runtime_init={runtime_init_time:.3f}s, "
                f"setup={setup_build_time:.3f}s, "
                f"pattern={pattern_build_time:.3f}s, "
                f"first_assemble={first_assemble_core_time:.3f}s)"
            ),
            flush=True,
        )

    return results, runtime_init_time


def main():
    args = parse_args()
    results, runtime_init_time = fluxfem_cases_numpy(args)
    _write_backend_json(args.json, results, args.no_solve, runtime_init_time)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark assembly time and peak RSS for different n_chunks settings.

Runs each configuration in a subprocess by default to isolate memory peaks.
Use --memory-profile-dir to save per-run device memory profiles (best effort).

Example:
  PYTHONPATH=src python bench/assembly_n_chunks_bench.py --sizes 8,12,16 --n-chunks-list None,2,3,4,5 --kind bilinear --repeats 5 --out result/bench/assembly_n_chunks_bench.png --memory-profile-dir result/bench/memory_profiles
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Iterable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.profiler

import fluxfem as ff


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_n_chunks(raw: str) -> list[int | None]:
    out: list[int | None] = []
    for token in _parse_list(raw):
        lower = token.lower()
        if lower in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(token))
    return out


def _block_ready(out):
    if isinstance(out, ff.FluxSparseMatrix):
        jax.block_until_ready(out.data)
    else:
        jax.block_until_ready(out)


def _build_assemble_fn(space, kind: str, n_chunks: int | None, params: dict):
    policy = ff.AssemblyPolicy(n_chunks=n_chunks)
    # This benchmark intentionally targets the single-space shortcut APIs to
    # measure chunking behavior in the shortest common path.
    if kind == "bilinear":
        pattern = space.get_sparsity_pattern(with_idx=True)

        def assemble():
            return space.assemble(
                ff.diffusion_form,
                params=params["kappa"],
                pattern=pattern,
                policy=policy,
            )

    elif kind == "linear":
        def assemble():
            return space.assemble(
                ff.scalar_body_force_form,
                params=params["load"],
                policy=policy,
            )

    elif kind == "mass":
        def assemble():
            return space.assemble_mass_matrix(policy=policy)

    else:
        raise ValueError(f"Unknown kind: {kind}")

    return jax.jit(assemble)


def _get_device_memory_stats():
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    return stats or None


def _read_proc_status_value(key: str) -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])  # kB
    except OSError:
        return None
    return None


def _run_child(args) -> dict:
    mesh = ff.StructuredHexBox(nx=args.nx, ny=args.ny, nz=args.nz, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=args.intorder)

    params = {"kappa": args.kappa, "load": args.load}
    assemble_jit = _build_assemble_fn(space, args.kind, args.n_chunks, params)

    t0 = time.perf_counter()
    out0 = assemble_jit()
    _block_ready(out0)
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(args.repeats):
        t1 = time.perf_counter()
        out = assemble_jit()
        _block_ready(out)
        times.append(time.perf_counter() - t1)

    dev_stats = _get_device_memory_stats()
    device_peak_bytes = None
    if dev_stats is not None:
        device_peak_bytes = dev_stats.get("peak_bytes_in_use", dev_stats.get("bytes_in_use"))
    cpu_peak_kb = _read_proc_status_value("VmHWM:") or _read_proc_status_value("VmRSS:")

    mem_profile_path = None
    if args.memory_profile_dir:
        os.makedirs(args.memory_profile_dir, exist_ok=True)
        tag = f"{args.kind}_n{args.nx}_chunks{args.n_chunks}"
        mem_profile_path = os.path.join(args.memory_profile_dir, f"memory_profile_{tag}.txt")
        try:
            jax.profiler.save_device_memory_profile(mem_profile_path)
        except Exception as exc:  # pragma: no cover - best-effort profiling
            mem_profile_path = f"failed: {exc}"

    return {
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "n_dofs": int(space.n_dofs),
        "kind": args.kind,
        "n_chunks": args.n_chunks,
        "compile_s": compile_s,
        "avg_s": statistics.mean(times),
        "med_s": statistics.median(times),
        "device_peak_bytes": device_peak_bytes,
        "cpu_peak_kb": cpu_peak_kb,
        "memory_profile": mem_profile_path,
    }


def _run_parent(args) -> list[dict]:
    sizes = [int(s) for s in _parse_list(args.sizes)]
    n_chunks_list = _parse_n_chunks(args.n_chunks_list)

    results = []
    for s in sizes:
        for n_chunks in n_chunks_list:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--child",
                "--nx",
                str(s),
                "--ny",
                str(s),
                "--nz",
                str(s),
                "--intorder",
                str(args.intorder),
                "--kind",
                args.kind,
                "--repeats",
                str(args.repeats),
                "--kappa",
                str(args.kappa),
                "--load",
                str(args.load),
                "--n-chunks",
                str(n_chunks) if n_chunks is not None else "None",
            ]
            if args.memory_profile_dir:
                cmd.extend(["--memory-profile-dir", args.memory_profile_dir])
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            line = proc.stdout.strip().splitlines()[-1]
            results.append(json.loads(line))
    return results


def _plot_results(rows: Iterable[dict], out_path: str):
    rows = list(rows)
    if not rows:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kinds = sorted({row["kind"] for row in rows})
    sizes = sorted({row["nx"] for row in rows})
    n_chunks_list = sorted({row["n_chunks"] for row in rows}, key=lambda x: (x is None, x))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_compile, ax_time, ax_mem = axes
    mem_label = "Peak device memory (MiB)"
    fallback_label = "Peak RSS (MiB)"

    for kind in kinds:
        for n_chunks in n_chunks_list:
            label = f"{kind}, n_chunks={n_chunks}"
            xs = []
            ys_compile = []
            ys_time = []
            ys_mem = []
            for s in sizes:
                for row in rows:
                    if row["kind"] == kind and row["n_chunks"] == n_chunks and row["nx"] == s:
                        xs.append(s)
                        ys_compile.append(row["compile_s"])
                        ys_time.append(row["avg_s"])
                        if row.get("device_peak_bytes") is not None:
                            ys_mem.append(row["device_peak_bytes"] / (1024.0 * 1024.0))
                        elif row.get("cpu_peak_kb") is not None:
                            ys_mem.append(row["cpu_peak_kb"] / 1024.0)
                            mem_label = fallback_label
                        else:
                            ys_mem.append(float("nan"))
                        break
            if xs:
                ax_compile.plot(xs, ys_compile, marker="o", label=label)
                ax_time.plot(xs, ys_time, marker="o", label=label)
                ax_mem.plot(xs, ys_mem, marker="o", label=label)

    ax_compile.set_title("Compile Time")
    ax_compile.set_xlabel("nx = ny = nz")
    ax_compile.set_ylabel("seconds")
    ax_compile.set_yscale("log")
    ax_compile.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax_time.set_title("Assembly Time (avg)")
    ax_time.set_xlabel("nx = ny = nz")
    ax_time.set_ylabel("seconds")
    ax_time.set_yscale("log")
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax_mem.set_title(mem_label)
    ax_mem.set_xlabel("nx = ny = nz")
    ax_mem.set_ylabel("MiB")
    ax_mem.set_yscale("log")
    ax_mem.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = ax_time.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150)
    print(f"[plot] saved -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Benchmark assembly time and RSS across n_chunks.")
    p.add_argument("--sizes", default="8,12,16", help="Comma-separated sizes for nx=ny=nz.")
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--load", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--kind",
        choices=("bilinear", "linear", "mass"),
        default="bilinear",
        help="Which assembly to benchmark.",
    )
    p.add_argument(
        "--n-chunks-list",
        default="None,2,3,4,5",
        help="Comma-separated n_chunks list (use None for no chunking).",
    )
    p.add_argument(
        "--out",
        default="result/bench/assembly_n_chunks_bench.png",
        help="Output PNG path for the plot.",
    )
    p.add_argument(
        "--memory-profile-dir",
        default=None,
        help="Directory to save JAX device memory profiles per run.",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--nz", type=int, default=8)
    p.add_argument("--n-chunks", type=str, default="None")

    args = p.parse_args()

    if args.child:
        if args.n_chunks.lower() in {"none", "null"}:
            args.n_chunks = None
        else:
            args.n_chunks = int(args.n_chunks)
        result = _run_child(args)
        print(json.dumps(result, ensure_ascii=True))
        return

    rows = _run_parent(args)
    _plot_results(rows, args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot numerical-agreement and warm-run timing comparisons for Gridap vs FluxFEM."
    )
    p.add_argument(
        "--shared-json",
        default="result/bench/neo_hookean_fluxfem_vs_gridap_wp/results.json",
        help="Shared-mesh comparison JSON from bench/neo_hookean_fluxfem_vs_gridap.py.",
    )
    p.add_argument(
        "--gridap-warmrun",
        default="result/bench/gridap_warmrun/results.json",
        help="Gridap same-session warm-run JSON.",
    )
    p.add_argument(
        "--flux",
        action="append",
        default=[],
        help="FluxFEM warm-run JSON. May be passed multiple times.",
    )
    p.add_argument(
        "--petsc-json",
        default="",
        help="Optional FluxFEM PETSc warm-run JSON for stacked timing comparison.",
    )
    p.add_argument(
        "--out-dir",
        default="result/bench/fluxfem_bucketed_warmrun",
        help="Directory for generated PNG files.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _series_label(payload: dict[str, Any], source: Path) -> str:
    mode = str(payload.get("mode", source.stem))
    if mode == "plain":
        return "FluxFEM plain"
    if mode == "bucketed":
        cfg = payload.get("config", {})
        return f"FluxFEM bucketed ({cfg.get('bucket_size')}/{cfg.get('chunk_size')})"
    if mode == "fixed_chunk_tail":
        cfg = payload.get("config", {})
        return f"FluxFEM fixed_chunk_tail ({cfg.get('chunk_size')})"
    return f"FluxFEM {mode}"


def _mesh_name(item: dict[str, Any]) -> str:
    return Path(item["mesh"]).name


def plot_numeric(shared_payload: dict[str, Any], out_path: Path) -> None:
    cases = sorted(shared_payload["summary"]["cases"], key=lambda item: float(item["lc"]), reverse=True)
    lc_labels = [f"lc={case['lc']:g}" for case in cases]
    max_abs = [float(case["numeric"]["max_abs_diff"]["mean"]) for case in cases]
    rms = [float(case["numeric"]["rms_diff"]["mean"]) for case in cases]
    rel_l2 = [float(case["numeric"]["relative_l2_diff"]["mean"]) for case in cases]

    x = np.arange(len(cases), dtype=float)
    w = 0.24

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - w, max_abs, width=w, label="max abs diff", color="#355070")
    ax.bar(x, rms, width=w, label="RMS diff", color="#6d597a")
    ax.bar(x + w, rel_l2, width=w, label="rel L2 diff", color="#b56576")
    ax.set_yscale("log")
    ax.set_xticks(x, lc_labels)
    ax.set_ylabel("error")
    ax.set_title("FluxFEM vs Gridap: Shared-Mesh Numerical Agreement")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_warmrun(
    gridap_payload: dict[str, Any],
    flux_payloads: list[tuple[Path, dict[str, Any]]],
    out_path: Path,
    *,
    log_scale: bool = False,
) -> None:
    gridap_items = {_mesh_name(item): item for item in gridap_payload["results"]}
    mesh_order = list(gridap_items.keys())
    for _path, payload in flux_payloads:
        for item in payload["results"]:
            mesh = _mesh_name(item)
            if mesh not in mesh_order:
                mesh_order.append(mesh)

    series: list[tuple[str, list[float]]] = []
    series.append(
        (
            "Gridap total",
            [float(gridap_items[mesh]["wall_time_s"]) if mesh in gridap_items else np.nan for mesh in mesh_order],
        )
    )
    for path, payload in flux_payloads:
        label = _series_label(payload, path)
        item_map = {_mesh_name(item): item for item in payload["results"]}
        vals = [float(item_map[mesh]["wall_time_s"]) if mesh in item_map else np.nan for mesh in mesh_order]
        series.append((label, vals))

    x = np.arange(len(mesh_order), dtype=float)
    width = 0.8 / max(len(series), 1)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    colors = ["#2a9d8f", "#355070", "#e76f51", "#f4a261", "#6d597a"]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for i, (label, vals) in enumerate(series):
        ax.bar(x + offsets[i], vals, width=width * 0.92, label=label, color=colors[i % len(colors)])
    ax.set_xticks(x, mesh_order, rotation=0)
    ax.set_ylabel("wall time [s]")
    ax.set_title("Same-Session Warm-Run Timing" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_compiled_only(
    gridap_payload: dict[str, Any],
    flux_payloads: list[tuple[Path, dict[str, Any]]],
    petsc_payload: dict[str, Any] | None,
    out_path: Path,
    *,
    log_scale: bool = False,
) -> None:
    gridap_items = {_mesh_name(item): item for item in gridap_payload["results"]}
    mesh_order = list(gridap_items.keys())
    for _path, payload in flux_payloads:
        for item in payload["results"]:
            mesh = _mesh_name(item)
            if mesh not in mesh_order:
                mesh_order.append(mesh)

    def _compiled_wall(item: dict[str, Any]) -> float:
        first = item.get("first_step_s")
        rest = item.get("remaining_steps_avg_s")
        nstep = int(item.get("load_steps", 0))
        if first is None or rest is None or nstep <= 1:
            return float("nan")
        return float(rest) * float(nstep)

    gridap_vals = []
    for mesh in mesh_order:
        item = gridap_items.get(mesh)
        if item is None:
            gridap_vals.append(np.nan)
            continue
        first = item.get("first_step_s")
        rest = item.get("remaining_steps_avg_s")
        if first is None or rest is None or np.isnan(float(rest)):
            gridap_vals.append(np.nan)
            continue
        nstep = 20
        gridap_vals.append(float(rest) * nstep)

    series: list[tuple[str, list[float]]] = [("Gridap steady-state proxy", gridap_vals)]
    for path, payload in flux_payloads:
        label = _series_label(payload, path)
        item_map = {_mesh_name(item): item for item in payload["results"]}
        vals = [_compiled_wall(item_map[mesh]) if mesh in item_map else np.nan for mesh in mesh_order]
        series.append((f"{label} steady-state proxy", vals))
    if petsc_payload is not None:
        item_map = {_mesh_name(item): item for item in petsc_payload["results"]}
        vals = [_compiled_wall(item_map[mesh]) if mesh in item_map else np.nan for mesh in mesh_order]
        series.append(("FluxFEM petsc_shell steady-state proxy", vals))

    x = np.arange(len(mesh_order), dtype=float)
    width = 0.8 / max(len(series), 1)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    colors = ["#2f5d8a", "#355070", "#c96a1b", "#6d597a", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for i, (label, vals) in enumerate(series):
        ax.bar(x + offsets[i], vals, width=width * 0.92, label=label, color=colors[i % len(colors)])
    ax.set_xticks(x, mesh_order, rotation=0)
    ax.set_ylabel("estimated steady-state wall [s]")
    ax.set_title("Compile-Removed Warm-Run Proxy" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_compile_overhead_proxy(
    gridap_payload: dict[str, Any],
    flux_payloads: list[tuple[Path, dict[str, Any]]],
    petsc_payload: dict[str, Any] | None,
    out_path: Path,
    *,
    log_scale: bool = False,
) -> None:
    gridap_items = {_mesh_name(item): item for item in gridap_payload["results"]}
    mesh_order = list(gridap_items.keys())
    for _path, payload in flux_payloads:
        for item in payload["results"]:
            mesh = _mesh_name(item)
            if mesh not in mesh_order:
                mesh_order.append(mesh)

    gridap_vals = []
    for mesh in mesh_order:
        item = gridap_items.get(mesh)
        if item is None:
            gridap_vals.append(np.nan)
            continue
        direct_proxy = item.get("compile_only_proxy_s")
        if direct_proxy is not None:
            gridap_vals.append(float(direct_proxy))
            continue
        first = item.get("first_step_s")
        rest = item.get("remaining_steps_avg_s")
        if first is None or rest is None:
            gridap_vals.append(np.nan)
            continue
        gridap_vals.append(max(0.0, float(first) - float(rest)))

    series: list[tuple[str, list[float]]] = [("Gridap compile/init proxy", gridap_vals)]

    for path, payload in flux_payloads:
        label = _series_label(payload, path)
        item_map = {_mesh_name(item): item for item in payload["results"]}
        vals = []
        for mesh in mesh_order:
            item = item_map.get(mesh)
            if item is None:
                vals.append(np.nan)
                continue
            first = item.get("first_step_s")
            rest = item.get("remaining_steps_avg_s")
            if first is None or rest is None:
                vals.append(np.nan)
            else:
                vals.append(max(0.0, float(first) - float(rest)))
        series.append((f"{label} first-step overhead", vals))

    if petsc_payload is not None:
        item_map = {_mesh_name(item): item for item in petsc_payload["results"]}
        vals = []
        for mesh in mesh_order:
            item = item_map.get(mesh)
            if item is None:
                vals.append(np.nan)
                continue
            first = item.get("first_step_s")
            rest = item.get("remaining_steps_avg_s")
            if first is None or rest is None:
                vals.append(np.nan)
            else:
                vals.append(max(0.0, float(first) - float(rest)))
        series.append(("FluxFEM petsc_shell first-step overhead", vals))

    x = np.arange(len(mesh_order), dtype=float)
    width = 0.8 / max(len(series), 1)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    colors = ["#2f5d8a", "#355070", "#c96a1b", "#6d597a", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    for i, (label, vals) in enumerate(series):
        ax.bar(x + offsets[i], vals, width=width * 0.92, label=label, color=colors[i % len(colors)])
    ax.set_xticks(x, mesh_order, rotation=0)
    ax.set_ylabel("time [s]")
    ax.set_title("Compile / Initialization Overhead Proxy" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_warmrun_stacked(
    gridap_payload: dict[str, Any],
    spsolve_payload: dict[str, Any] | None,
    petsc_payload: dict[str, Any] | None,
    out_path: Path,
    *,
    log_scale: bool = False,
) -> None:
    gridap_map = {_mesh_name(item): item for item in gridap_payload["results"]}
    mesh_order = list(gridap_map.keys())

    def _mesh_labels(meshes: list[str]) -> list[str]:
        return [f"{mesh}\n" for mesh in meshes]

    x = np.arange(len(mesh_order), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(11.2, 5.2))

    def _stack(series_x, base0, base1, base2, *, colors, label_prefix, hatch: str = ""):
        ax.bar(
            series_x,
            base0,
            width=width * 0.92,
            color=colors[0],
            label=f"{label_prefix} assembly/linear",
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.4,
        )
        ax.bar(
            series_x,
            base1,
            width=width * 0.92,
            bottom=base0,
            color=colors[1],
            label=f"{label_prefix} eval/solve",
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.4,
        )
        ax.bar(
            series_x,
            base2,
            width=width * 0.92,
            bottom=np.asarray(base0) + np.asarray(base1),
            color=colors[2],
            label=f"{label_prefix} other",
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.4,
        )

    gridap_primary = []
    gridap_secondary = []
    gridap_other = []
    for mesh in mesh_order:
        item = gridap_map[mesh]
        asm = float(item.get("assembly_time_s", 0.0))
        solve = float(item.get("solve_time_s", 0.0))
        other = max(0.0, float(item["wall_time_s"]) - asm - solve)
        gridap_primary.append(asm)
        gridap_secondary.append(solve)
        gridap_other.append(other)
    _stack(
        x + offsets[0],
        gridap_primary,
        gridap_secondary,
        gridap_other,
        colors=["#2f5d8a", "#5f88b0", "#c8d8ea"],
        label_prefix="Gridap",
    )

    if spsolve_payload is not None:
        spsolve_map = {_mesh_name(item): item for item in spsolve_payload["results"]}
        lin = []
        ev = []
        other = []
        for mesh in mesh_order:
            item = spsolve_map.get(mesh)
            if item is None:
                lin.append(np.nan)
                ev.append(np.nan)
                other.append(np.nan)
                continue
            lin.append(float(item.get("linear_solve_total_s", np.nan)))
            ev.append(float(item.get("residual_eval_total_s", np.nan)))
            other.append(float(item.get("other_total_s", np.nan)))
        _stack(
            x + offsets[1],
            lin,
            ev,
            other,
            colors=["#c96a1b", "#e39d5f", "#f3d4b5"],
            label_prefix="FluxFEM spsolve",
        )

    if petsc_payload is not None:
        petsc_map = {_mesh_name(item): item for item in petsc_payload["results"]}
        lin = []
        ev = []
        other = []
        for mesh in mesh_order:
            item = petsc_map.get(mesh)
            if item is None:
                lin.append(np.nan)
                ev.append(np.nan)
                other.append(np.nan)
                continue
            lin.append(float(item.get("linear_solve_total_s", np.nan)))
            ev.append(float(item.get("residual_eval_total_s", np.nan)))
            other.append(float(item.get("other_total_s", np.nan)))
        _stack(
            x + offsets[2],
            lin,
            ev,
            other,
            colors=["#c96a1b", "#e39d5f", "#f3d4b5"],
            label_prefix="FluxFEM PETSc",
            hatch="//",
        )

    ax.set_xticks(x, _mesh_labels(mesh_order))
    ax.set_ylabel("time [s]")
    ax.set_title("Warm-Run Timing Breakdown" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h = []
    uniq_l = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    shared_path = (ROOT / args.shared_json).resolve()
    gridap_path = (ROOT / args.gridap_warmrun).resolve()
    flux_paths = [(ROOT / item).resolve() for item in args.flux]
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_payload = _load_json(shared_path)
    gridap_payload = _load_json(gridap_path)
    flux_payloads = [(path, _load_json(path)) for path in flux_paths]
    petsc_payload = _load_json((ROOT / args.petsc_json).resolve()) if args.petsc_json else None

    plot_numeric(shared_payload, out_dir / "gridap_fluxfem_numeric_comparison.png")
    plot_warmrun(
        gridap_payload,
        flux_payloads,
        out_dir / "gridap_fluxfem_warmrun_timing.png",
    )
    plot_warmrun(
        gridap_payload,
        flux_payloads,
        out_dir / "gridap_fluxfem_warmrun_timing_log.png",
        log_scale=True,
    )
    plot_compiled_only(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "gridap_fluxfem_compiled_only_timing.png",
    )
    plot_compiled_only(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "gridap_fluxfem_compiled_only_timing_log.png",
        log_scale=True,
    )
    plot_compiled_only(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "gridap_fluxfem_warmrun_timing_nocompile.png",
    )
    plot_compiled_only(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "gridap_fluxfem_warmrun_timing_nocompile_log.png",
        log_scale=True,
    )
    plot_compile_overhead_proxy(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "gridap_fluxfem_compile_overhead_proxy.png",
    )
    plot_compile_overhead_proxy(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "bar_plot_below_compile_overhead_proxy.png",
    )
    plot_compile_overhead_proxy(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "gridap_fluxfem_compile_overhead_proxy_log.png",
        log_scale=True,
    )
    plot_compile_overhead_proxy(
        gridap_payload,
        flux_payloads,
        petsc_payload,
        out_dir / "bar_plot_below_compile_overhead_proxy_log.png",
        log_scale=True,
    )
    spsolve_payload = flux_payloads[0][1] if flux_payloads else None
    plot_warmrun_stacked(
        gridap_payload,
        spsolve_payload,
        petsc_payload,
        out_dir / "gridap_fluxfem_warmrun_timing_stacked.png",
    )
    plot_warmrun_stacked(
        gridap_payload,
        spsolve_payload,
        petsc_payload,
        out_dir / "gridap_fluxfem_warmrun_timing_stacked_log.png",
        log_scale=True,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

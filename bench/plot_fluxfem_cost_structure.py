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
    p = argparse.ArgumentParser(description="Plot FluxFEM cost-structure breakdowns.")
    p.add_argument("--spsolve-20step", required=True, help="Warm-run JSON with cost breakdown fields.")
    p.add_argument("--spsolve-2step", required=True, help="2-step spsolve JSON.")
    p.add_argument("--petsc-lc3", required=True, help="2-step petsc JSON for lc=3.0.")
    p.add_argument("--petsc-lc2", required=True, help="2-step petsc JSON for lc=2.0.")
    p.add_argument(
        "--petsc-20step",
        default="result/bench/fluxfem_bucketed_warmrun/plain_results_petsc.json",
        help="20-step PETSc warm-run JSON.",
    )
    p.add_argument(
        "--gridap-warmrun",
        default="result/bench/gridap_warmrun/results.json",
        help="Gridap warm-run JSON.",
    )
    p.add_argument(
        "--out-dir",
        default="result/bench/fluxfem_bucketed_warmrun",
        help="Directory for output PNG files.",
    )
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_spsolve_breakdown(payload: dict[str, Any], out_path: Path, *, log_scale: bool = False) -> None:
    rows = payload["results"]
    labels = [f"{row['n_elems']} elems" for row in rows]
    linear = np.asarray([float(row["linear_solve_total_s"]) for row in rows], dtype=float)
    eval_s = np.asarray([float(row["residual_eval_total_s"]) for row in rows], dtype=float)
    init_r = np.asarray([float(row.get("initial_residual_total_s", 0.0)) for row in rows], dtype=float)
    init_j = np.asarray([float(row.get("initial_jacobian_total_s", 0.0)) for row in rows], dtype=float)
    control = np.asarray([float(row.get("control_total_s", 0.0)) for row in rows], dtype=float)
    other = np.asarray([float(row["other_total_s"]) for row in rows], dtype=float)

    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x, linear, label="linear solve", color="#355070")
    ax.bar(x, init_j, bottom=linear, label="initial Jacobian", color="#7b6d8d")
    ax.bar(x, init_r, bottom=linear + init_j, label="initial residual", color="#b56576")
    ax.bar(x, eval_s, bottom=linear + init_j + init_r, label="residual eval", color="#d9a6b3")
    ax.bar(x, control, bottom=linear + init_j + init_r + eval_s, label="control/sync", color="#f0c9a4")
    ax.bar(x, other, bottom=linear + init_j + init_r + eval_s + control, label="other", color="#e8e8e8")
    ax.set_xticks(x, labels)
    ax.set_ylabel("time [s]")
    ax.set_title("FluxFEM spsolve Cost Breakdown (20 steps)" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _row_by_elems(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["n_elems"]): row for row in payload["results"]}


def plot_solver_compare(
    spsolve_payload: dict[str, Any],
    petsc_lc3_payload: dict[str, Any],
    petsc_lc2_payload: dict[str, Any],
    out_path: Path,
) -> None:
    spsolve_map = _row_by_elems(spsolve_payload)
    petsc_map = {
        int(petsc_lc3_payload["results"][0]["n_elems"]): petsc_lc3_payload["results"][0],
        int(petsc_lc2_payload["results"][0]["n_elems"]): petsc_lc2_payload["results"][0],
    }
    elems_order = sorted(spsolve_map.keys())
    x = np.arange(len(elems_order), dtype=float)
    width = 0.18

    spsolve_total = np.asarray([float(spsolve_map[e]["wall_time_s"]) for e in elems_order], dtype=float)
    petsc_total = np.asarray([float(petsc_map[e]["wall_time_s"]) if e in petsc_map else np.nan for e in elems_order], dtype=float)
    spsolve_linear = np.asarray(
        [float(spsolve_map[e].get("linear_solve_total_s", np.nan)) for e in elems_order],
        dtype=float,
    )
    petsc_linear = np.asarray(
        [float(petsc_map[e].get("linear_solve_total_s", np.nan)) if e in petsc_map else np.nan for e in elems_order],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 1.5 * width, spsolve_total, width=width, label="spsolve total", color="#355070")
    ax.bar(x - 0.5 * width, spsolve_linear, width=width, label="spsolve linear", color="#6d597a")
    ax.bar(x + 0.5 * width, petsc_total, width=width, label="petsc total", color="#2a9d8f")
    ax.bar(x + 1.5 * width, petsc_linear, width=width, label="petsc linear", color="#8ab17d")
    ax.set_xticks(x, [f"{e} elems" for e in elems_order])
    ax.set_ylabel("time [s]")
    ax.set_title("FluxFEM Solver Path Comparison (2 steps)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cross_framework_breakdown(
    gridap_payload: dict[str, Any],
    spsolve_payload: dict[str, Any],
    petsc_payload: dict[str, Any],
    out_path: Path,
    *,
    log_scale: bool = False,
) -> None:
    gridap_rows = {int(row["n_nodes"]): row for row in gridap_payload["results"]}
    spsolve_rows = {int(row["n_nodes"]): row for row in spsolve_payload["results"]}
    petsc_rows = {int(row["n_nodes"]): row for row in petsc_payload["results"]}
    node_order = sorted(spsolve_rows.keys())
    mesh_labels = [f"{spsolve_rows[n]['n_elems']} elems" for n in node_order]

    x = np.arange(len(node_order), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]

    solve_color = "#355070"
    assembly_color = "#b56576"
    eval_color = "#d9a6b3"
    control_color = "#f0c9a4"
    other_color = "#e8e8e8"

    fig, ax = plt.subplots(figsize=(11.4, 5.2))

    def _stack(xx, segments, *, hatch: str = "", prefix: str = ""):
        bottom = np.zeros(len(xx), dtype=float)
        for values, label, color in segments:
            ax.bar(
                xx,
                values,
                width=width * 0.92,
                bottom=bottom,
                color=color,
                label=f"{prefix}{label}",
                hatch=hatch,
                edgecolor="#333333",
                linewidth=0.4,
            )
            bottom = bottom + np.asarray(values, dtype=float)

    gridap_asm = []
    gridap_solve = []
    gridap_other = []
    for n in node_order:
        row = gridap_rows[n]
        asm = float(row.get("assembly_time_s", 0.0))
        solve = float(row.get("solve_time_s", 0.0))
        other = max(0.0, float(row["wall_time_s"]) - asm - solve)
        gridap_asm.append(asm)
        gridap_solve.append(solve)
        gridap_other.append(other)
    _stack(
        x + offsets[0],
        [
            (gridap_solve, "solve", solve_color),
            (gridap_asm, "assembly", assembly_color),
            (gridap_other, "other", other_color),
        ],
        prefix="Gridap ",
    )

    spsolve_linear = []
    spsolve_init_j = []
    spsolve_init_r = []
    spsolve_eval = []
    spsolve_control = []
    spsolve_other = []
    for n in node_order:
        row = spsolve_rows[n]
        spsolve_linear.append(float(row.get("linear_solve_total_s", 0.0)))
        spsolve_init_j.append(float(row.get("initial_jacobian_total_s", 0.0)))
        spsolve_init_r.append(float(row.get("initial_residual_total_s", 0.0)))
        spsolve_eval.append(float(row.get("residual_eval_total_s", 0.0)))
        spsolve_control.append(float(row.get("control_total_s", 0.0)))
        spsolve_other.append(float(row.get("other_total_s", 0.0)))
    _stack(
        x + offsets[1],
        [
            (spsolve_linear, "linear solve", solve_color),
            (spsolve_init_j, "initial Jacobian", assembly_color),
            (spsolve_init_r, "initial residual", eval_color),
            (spsolve_eval, "residual eval", "#ead2d8"),
            (spsolve_control, "control/sync", control_color),
            (spsolve_other, "other", other_color),
        ],
        prefix="FluxFEM spsolve ",
    )

    petsc_linear = []
    petsc_eval = []
    petsc_other = []
    for n in node_order:
        row = petsc_rows[n]
        petsc_linear.append(float(row.get("linear_solve_total_s", 0.0)))
        petsc_eval.append(float(row.get("residual_eval_total_s", 0.0)))
        petsc_other.append(float(row.get("other_total_s", 0.0)))
    _stack(
        x + offsets[2],
        [
            (petsc_linear, "linear solve", solve_color),
            (petsc_eval, "residual eval", eval_color),
            (petsc_other, "other", other_color),
        ],
        hatch="//",
        prefix="FluxFEM PETSc ",
    )

    ax.set_xticks(x, mesh_labels)
    ax.set_ylabel("time [s]")
    ax.set_title("Gridap vs FluxFEM Cost Breakdown" + (" (log scale)" if log_scale else ""))
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
    ax.legend(uniq_h, uniq_l, frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cross_framework_breakdown_nocompile(
    gridap_payload: dict[str, Any],
    spsolve_payload: dict[str, Any],
    petsc_payload: dict[str, Any],
    out_path: Path,
    *,
    log_scale: bool = False,
) -> None:
    spsolve_rows = {int(row["n_nodes"]): row for row in spsolve_payload["results"]}
    petsc_rows = {int(row["n_nodes"]): row for row in petsc_payload["results"]}
    node_order = sorted(spsolve_rows.keys())
    mesh_labels = [f"{spsolve_rows[n]['n_elems']} elems" for n in node_order]

    x = np.arange(len(node_order), dtype=float)
    width = 0.28
    offsets = [-0.5 * width, 0.5 * width]

    solve_color = "#355070"
    assembly_color = "#b56576"
    eval_color = "#d9a6b3"
    control_color = "#f0c9a4"
    other_color = "#e8e8e8"

    fig, ax = plt.subplots(figsize=(11.4, 5.2))

    def _stack(xx, segments, *, hatch: str = "", prefix: str = ""):
        bottom = np.zeros(len(xx), dtype=float)
        for values, label, color in segments:
            ax.bar(
                xx,
                values,
                width=width * 0.92,
                bottom=bottom,
                color=color,
                label=f"{prefix}{label}",
                hatch=hatch,
                edgecolor="#333333",
                linewidth=0.4,
            )
            bottom = bottom + np.asarray(values, dtype=float)

    def _flux_segments(rows: dict[int, dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[float]]:
        linear = []
        residual = []
        control = []
        other = []
        for n in node_order:
            row = rows[n]
            linear.append(float(row.get("linear_solve_total_s", 0.0)))
            residual.append(float(row.get("residual_eval_total_s", 0.0)))
            control.append(float(row.get("control_total_s", 0.0)))
            other.append(float(row.get("other_total_s", 0.0)))
        return linear, residual, control, other

    s_lin, s_res, s_ctrl, s_oth = _flux_segments(spsolve_rows)
    _stack(
        x + offsets[0],
        [
            (s_lin, "linear solve", solve_color),
            (s_res, "residual eval", eval_color),
            (s_ctrl, "control/sync", control_color),
            (s_oth, "other", other_color),
        ],
        prefix="FluxFEM spsolve ",
    )

    p_lin, p_res, p_ctrl, p_oth = _flux_segments(petsc_rows)
    _stack(
        x + offsets[1],
        [
            (p_lin, "linear solve", solve_color),
            (p_res, "residual eval", eval_color),
            (p_ctrl, "control/sync", control_color),
            (p_oth, "other", other_color),
        ],
        hatch="//",
        prefix="FluxFEM PETSc ",
    )

    ax.set_xticks(x, mesh_labels)
    ax.set_ylabel("time [s]")
    ax.set_title("FluxFEM Cost Breakdown (compile removed proxy)" + (" (log scale)" if log_scale else ""))
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
    ax.legend(uniq_h, uniq_l, frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cross_framework_breakdown_nocompile_coarse(
    gridap_payload: dict[str, Any],
    spsolve_payload: dict[str, Any],
    petsc_payload: dict[str, Any],
    out_path: Path,
    *,
    log_scale: bool = False,
) -> None:
    gridap_rows = {int(row["n_nodes"]): row for row in gridap_payload["results"]}
    spsolve_rows = {int(row["n_nodes"]): row for row in spsolve_payload["results"]}
    petsc_rows = {int(row["n_nodes"]): row for row in petsc_payload["results"]}
    node_order = sorted(spsolve_rows.keys())
    mesh_labels = [f"{spsolve_rows[n]['n_elems']} elems" for n in node_order]

    x = np.arange(len(node_order), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]

    solve_color = "#355070"
    assembly_color = "#b56576"
    other_color = "#e8e8e8"

    fig, ax = plt.subplots(figsize=(11.4, 5.2))

    def _stack(xx, solve_like, assembly_like, other_like, *, hatch: str = "", prefix: str = ""):
        ax.bar(
            xx,
            solve_like,
            width=width * 0.92,
            color=solve_color,
            label=f"{prefix}solver-like",
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.4,
        )
        ax.bar(
            xx,
            assembly_like,
            width=width * 0.92,
            bottom=solve_like,
            color=assembly_color,
            label=f"{prefix}assembly-like",
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.4,
        )
        ax.bar(
            xx,
            other_like,
            width=width * 0.92,
            bottom=np.asarray(solve_like) + np.asarray(assembly_like),
            color=other_color,
            label=f"{prefix}other",
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.4,
        )

    gridap_solver = []
    gridap_assembly = []
    gridap_other = []
    for n in node_order:
        row = gridap_rows[n]
        rem = row.get("remaining_steps_avg_s")
        if rem is None or np.isnan(float(rem)):
            gridap_solver.append(np.nan)
        else:
            gridap_solver.append(float(rem) * 20.0)
        gridap_assembly.append(float(row.get("assembly_time_s", 0.0)))
        gridap_other.append(0.0)
    _stack(x + offsets[0], gridap_solver, gridap_assembly, gridap_other, prefix="Gridap ")

    def _rest_proxy(values: list[float], load_steps: int) -> float:
        if not values:
            return float("nan")
        if len(values) == 1:
            return float(values[0]) * float(load_steps)
        rest = values[1:]
        return float(sum(rest) / len(rest)) * float(load_steps)

    def _flux_coarse(rows: dict[int, dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
        solve_like = []
        assembly_like = []
        other_like = []
        for n in node_order:
            row = rows[n]
            load_steps = int(row.get("load_steps", 1))
            solve_like.append(_rest_proxy(list(row.get("per_step_linear_solve_s", [])), load_steps))
            assembly_like.append(_rest_proxy(list(row.get("per_step_residual_eval_s", [])), load_steps))
            control_proxy = _rest_proxy(list(row.get("per_step_control_s", [])), load_steps)
            other_proxy = _rest_proxy(list(row.get("per_step_other_s", [])), load_steps)
            other_like.append(control_proxy + other_proxy)
        return solve_like, assembly_like, other_like

    s_solve, s_asm, s_other = _flux_coarse(spsolve_rows)
    _stack(x + offsets[1], s_solve, s_asm, s_other, prefix="FluxFEM spsolve ")

    p_solve, p_asm, p_other = _flux_coarse(petsc_rows)
    _stack(x + offsets[2], p_solve, p_asm, p_other, hatch="//", prefix="FluxFEM PETSc ")

    ax.set_xticks(x, mesh_labels)
    ax.set_ylabel("time [s]")
    ax.set_title("No-Compile Breakdown Proxy (coarse 3-category)" + (" (log scale)" if log_scale else ""))
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
    ax.legend(uniq_h, uniq_l, frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_assembly_only_compare(
    gridap_payload: dict[str, Any],
    spsolve_payload: dict[str, Any],
    petsc_payload: dict[str, Any],
    out_path: Path,
    *,
    log_scale: bool = False,
    compile_removed: bool = False,
) -> None:
    gridap_rows = {int(row["n_nodes"]): row for row in gridap_payload["results"]}
    spsolve_rows = {int(row["n_nodes"]): row for row in spsolve_payload["results"]}
    petsc_rows = {int(row["n_nodes"]): row for row in petsc_payload["results"]}
    node_order = sorted(spsolve_rows.keys())
    mesh_labels = [f"{spsolve_rows[n]['n_elems']} elems" for n in node_order]

    x = np.arange(len(node_order), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]

    def _rest_proxy(values: list[float], load_steps: int) -> float:
        if not values:
            return float("nan")
        if len(values) == 1:
            return float(values[0]) * float(load_steps)
        rest = values[1:]
        return float(sum(rest) / len(rest)) * float(load_steps)

    gridap_vals = []
    spsolve_vals = []
    petsc_vals = []
    for n in node_order:
        g = gridap_rows[n]
        s = spsolve_rows[n]
        p = petsc_rows[n]
        if compile_removed:
            g_steps = int(g.get("load_steps", 1))
            gridap_vals.append(float(g.get("remaining_steps_avg_s", np.nan)) * float(g_steps))
            s_steps = int(s.get("load_steps", 1))
            p_steps = int(p.get("load_steps", 1))
            spsolve_vals.append(_rest_proxy(list(s.get("per_step_residual_eval_s", [])), s_steps))
            petsc_vals.append(_rest_proxy(list(p.get("per_step_residual_eval_s", [])), p_steps))
        else:
            gridap_vals.append(float(g.get("assembly_time_s", 0.0)))
            spsolve_vals.append(
                float(s.get("initial_jacobian_total_s", 0.0))
                + float(s.get("initial_residual_total_s", 0.0))
                + float(s.get("residual_eval_total_s", 0.0))
            )
            petsc_vals.append(
                float(p.get("initial_jacobian_total_s", 0.0))
                + float(p.get("initial_residual_total_s", 0.0))
                + float(p.get("residual_eval_total_s", 0.0))
            )

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(x + offsets[0], gridap_vals, width=width * 0.92, color="#2f5d8a", label="Gridap assembly")
    ax.bar(x + offsets[1], spsolve_vals, width=width * 0.92, color="#c96a1b", label="FluxFEM spsolve assembly-like")
    ax.bar(
        x + offsets[2],
        petsc_vals,
        width=width * 0.92,
        color="#c96a1b",
        hatch="//",
        edgecolor="#333333",
        linewidth=0.4,
        label="FluxFEM PETSc assembly-like",
    )
    ax.set_xticks(x, mesh_labels)
    ax.set_ylabel("time [s]")
    title = "Assembly-Only Comparison"
    if compile_removed:
        title += " (compile removed proxy)"
    if log_scale:
        title += " (log scale)"
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spsolve20 = _load((ROOT / args.spsolve_20step).resolve())
    spsolve2 = _load((ROOT / args.spsolve_2step).resolve())
    petsc3 = _load(Path(args.petsc_lc3).resolve())
    petsc2 = _load(Path(args.petsc_lc2).resolve())
    petsc20 = _load((ROOT / args.petsc_20step).resolve())
    gridap = _load((ROOT / args.gridap_warmrun).resolve())

    plot_spsolve_breakdown(spsolve20, out_dir / "fluxfem_spsolve_cost_breakdown.png")
    plot_spsolve_breakdown(spsolve20, out_dir / "fluxfem_spsolve_cost_breakdown_log.png", log_scale=True)
    plot_solver_compare(spsolve2, petsc3, petsc2, out_dir / "fluxfem_spsolve_vs_petsc_2step.png")
    plot_cross_framework_breakdown(gridap, spsolve20, petsc20, out_dir / "gridap_fluxfem_cost_breakdown.png")
    plot_cross_framework_breakdown(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "gridap_fluxfem_cost_breakdown_log.png",
        log_scale=True,
    )
    plot_cross_framework_breakdown_nocompile(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "fluxfem_cost_breakdown_nocompile.png",
    )
    plot_cross_framework_breakdown_nocompile(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "fluxfem_cost_breakdown_nocompile_log.png",
        log_scale=True,
    )
    plot_cross_framework_breakdown_nocompile_coarse(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "gridap_fluxfem_cost_breakdown_nocompile_coarse.png",
    )
    plot_cross_framework_breakdown_nocompile_coarse(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "gridap_fluxfem_cost_breakdown_nocompile_coarse_log.png",
        log_scale=True,
    )
    plot_assembly_only_compare(gridap, spsolve20, petsc20, out_dir / "gridap_fluxfem_assembly_only.png")
    plot_assembly_only_compare(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "gridap_fluxfem_assembly_only_log.png",
        log_scale=True,
    )
    plot_assembly_only_compare(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "gridap_fluxfem_assembly_only_nocompile.png",
        compile_removed=True,
    )
    plot_assembly_only_compare(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "bar_plot_below_assembly_only_nocompile.png",
        compile_removed=True,
    )
    plot_assembly_only_compare(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "gridap_fluxfem_assembly_only_nocompile_log.png",
        log_scale=True,
        compile_removed=True,
    )
    plot_assembly_only_compare(
        gridap,
        spsolve20,
        petsc20,
        out_dir / "bar_plot_below_assembly_only_nocompile_log.png",
        log_scale=True,
        compile_removed=True,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

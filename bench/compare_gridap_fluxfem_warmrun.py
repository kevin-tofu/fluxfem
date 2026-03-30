#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize same-session warm-run results for Gridap and FluxFEM modes."
    )
    p.add_argument("--gridap", required=True, help="Path to Gridap warm-run JSON.")
    p.add_argument(
        "--flux",
        action="append",
        default=[],
        help="Path to FluxFEM warm-run JSON. May be passed multiple times.",
    )
    p.add_argument(
        "--out",
        default="result/bench/fluxfem_bucketed_warmrun/compare_gridap_fluxfem_warmrun.md",
        help="Output markdown path.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mesh_key(item: dict[str, Any]) -> str:
    return Path(item["mesh"]).name


def _series_label(payload: dict[str, Any], source: Path) -> str:
    mode = str(payload.get("mode", source.stem))
    config = payload.get("config", {}) or {}
    if mode == "gridap_same_session":
        return "Gridap"
    if mode == "plain":
        linear_solver = str(config.get("linear_solver", "plain"))
        if linear_solver == "petsc_shell":
            return "FluxFEM petsc_shell"
        if linear_solver == "spsolve":
            return "FluxFEM spsolve"
        return f"FluxFEM {linear_solver}"
    if mode == "bucketed":
        return f"FluxFEM bucketed ({config.get('bucket_size')}/{config.get('chunk_size')})"
    if mode == "fixed_chunk_tail":
        return f"FluxFEM fixed_chunk_tail ({config.get('chunk_size')})"
    return f"FluxFEM {mode}"


def _step_metrics(item: dict[str, Any]) -> tuple[str, str]:
    first = item.get("first_step_s")
    rest = item.get("remaining_steps_avg_s")
    first_s = f"{float(first):.3f}" if first is not None else "-"
    rest_s = f"{float(rest):.3f}" if rest is not None else "-"
    return first_s, rest_s


def _format_pad(item: dict[str, Any]) -> str:
    if "pad_ratio" not in item:
        return "-"
    return f"{float(item['pad_ratio']):.3f}"


def render_summary(gridap_payload: dict[str, Any], flux_payloads: list[tuple[Path, dict[str, Any]]]) -> str:
    mesh_order: list[str] = []
    mesh_stats: dict[str, dict[str, Any]] = {}

    for item in gridap_payload["results"]:
        key = _mesh_key(item)
        if key not in mesh_stats:
            mesh_order.append(key)
            mesh_stats[key] = {
                "mesh": key,
                "nodes": int(item.get("n_nodes", 0)),
                "elems": item.get("n_elems"),
            }

    for _path, payload in flux_payloads:
        for item in payload["results"]:
            key = _mesh_key(item)
            if key not in mesh_stats:
                mesh_order.append(key)
                mesh_stats[key] = {
                    "mesh": key,
                    "nodes": int(item.get("n_nodes", 0)),
                    "elems": item.get("n_elems"),
                }
            elif mesh_stats[key].get("elems") is None and item.get("n_elems") is not None:
                mesh_stats[key]["elems"] = item.get("n_elems")

    lines = [
        "# Gridap vs FluxFEM Warm-Run",
        "",
        "Same-session warm-run summary from existing benchmark JSON files.",
        "",
        "## Timing",
        "",
        "| series | mesh | nodes | elems | wall [s] | first step [s] | rest avg [s] | pad ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in gridap_payload["results"]:
        key = _mesh_key(item)
        stats = mesh_stats[key]
        lines.append(
            f"| Gridap | {stats['mesh']} | {stats['nodes']} | "
            f"{stats['elems'] if stats['elems'] is not None else '-'} | "
            f"{float(item['wall_time_s']):.3f} | - | - | - |"
        )

    for path, payload in flux_payloads:
        label = _series_label(payload, path)
        for item in payload["results"]:
            key = _mesh_key(item)
            stats = mesh_stats[key]
            first_s, rest_s = _step_metrics(item)
            lines.append(
                f"| {label} | {stats['mesh']} | {stats['nodes']} | "
                f"{stats['elems'] if stats['elems'] is not None else '-'} | "
                f"{float(item['wall_time_s']):.3f} | {first_s} | {rest_s} | {_format_pad(item)} |"
            )

    lines.extend(
        [
            "",
            "## Matched Conditions",
            "",
            "- Common target setup: same mesh order (`lc=3.0 -> lc=2.0`), same traction (`0.01`), same 20 load steps, and line search enabled.",
            "- FluxFEM rows differ only in linear solver backend (`spsolve` vs `petsc_shell`).",
            "- Gridap follows the same problem family and load schedule, but its JSON does not currently expose per-step timing.",
            "",
            "## Notes",
            "",
            "- Gridap JSON currently reports total/assembly/solve wall times but not per-step timing.",
            "- FluxFEM `first step` and `rest avg` are derived from `newton_solve` timer records when available.",
            "- `pad ratio` is only meaningful for FluxFEM policies that expose padding metadata.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    gridap_path = (ROOT / args.gridap).resolve()
    flux_paths = [(ROOT / path).resolve() for path in args.flux]
    out_path = (ROOT / args.out).resolve()

    gridap_payload = _load_json(gridap_path)
    flux_payloads = [(path, _load_json(path)) for path in flux_paths]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_summary(gridap_payload, flux_payloads), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

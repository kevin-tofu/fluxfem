#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fluxfem as ff  # noqa: E402
import fluxfem.helpers_ts as h_ts  # noqa: E402
from fluxfem.tools.timer import SectionTimer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Same-session warm-run experiment for FluxFEM bucketed assembly.")
    p.add_argument("--lc-values", type=float, nargs="+", default=[3.0, 2.0])
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--traction", type=float, default=1e-2)
    p.add_argument("--nstep", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--linear-solver", choices=["spsolve", "cg", "cg_matfree", "petsc_shell"], default="petsc_shell")
    p.add_argument("--linear-precond", choices=["none", "diag0"], default="none")
    p.add_argument("--petsc-ksp-type", default="preonly")
    p.add_argument("--petsc-pc-type", default="lu")
    p.add_argument("--petsc-use-pmat", action="store_true", default=True)
    p.add_argument("--no-petsc-use-pmat", dest="petsc_use_pmat", action="store_false")
    p.add_argument("--line-search", action="store_true", default=True)
    p.add_argument("--no-line-search", dest="line_search", action="store_false")
    p.add_argument("--bucket-size", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--chunk-build-context", action="store_true", default=False)
    p.add_argument("--gmsh", default=os.environ.get("GMSH", "gmsh"))
    p.add_argument(
        "--out",
        default="result/bench/fluxfem_bucketed_warmrun/results.json",
        help="Output JSON path.",
    )
    return p.parse_args()


def build_policy(args: argparse.Namespace) -> ff.AssemblyPolicy | None:
    if args.bucket_size is None and args.chunk_size is None:
        return None
    if args.chunk_size is None:
        raise ValueError("fixed-chunk mode requires --chunk-size.")
    if args.bucket_size is None:
        return ff.AssemblyPolicy.fixed_chunk(
            chunk_size=int(args.chunk_size),
            chunk_build_context=bool(args.chunk_build_context),
            lightweight_context=True,
        )
    return ff.AssemblyPolicy.bucketed(
        bucket_size=int(args.bucket_size),
        chunk_size=int(args.chunk_size),
        chunk_build_context=bool(args.chunk_build_context),
        lightweight_context=True,
    )


def policy_stats(n_elems: int, policy: ff.AssemblyPolicy | None) -> dict[str, Any]:
    if policy is None:
        stats = ff.chunk_pad_stats(int(n_elems), None)
        return {
            "bucketed": False,
            "bucket_size": None,
            "chunk_size": None,
            "n_pad": int(stats["n_pad"]),
            "pad": int(stats["pad"]),
            "pad_ratio": float(stats["pad_ratio"]),
        }
    if policy.max_padded_elems is None and policy.allow_tail_chunk:
        return {
            "bucketed": False,
            "bucket_size": None,
            "chunk_size": int(policy.fixed_chunk_size) if policy.fixed_chunk_size is not None else None,
            "n_pad": int(n_elems),
            "pad": 0,
            "pad_ratio": 0.0,
        }
    stats = ff.chunk_pad_stats(
        int(n_elems),
        None,
        fixed_chunk_size=policy.fixed_chunk_size,
        max_padded_elems=policy.max_padded_elems,
    )
    return {
        "bucketed": True,
        "bucket_size": int(policy.max_padded_elems) if policy.max_padded_elems is not None else None,
        "chunk_size": int(policy.fixed_chunk_size) if policy.fixed_chunk_size is not None else None,
        "n_pad": int(stats["n_pad"]),
        "pad": int(stats["pad"]),
        "pad_ratio": float(stats["pad_ratio"]),
    }


def step_time_breakdown(history: list[Any], timer_records: dict[str, list[float]]) -> dict[str, Any]:
    step_times = [float(getattr(step, "solve_time", 0.0)) for step in history]
    if step_times and max(step_times) <= 0.0:
        step_times = [
            float(v)
            for v in timer_records.get("run_total>preprocess>step", [])
        ]
    if not step_times:
        return {
            "step_solve_times_s": [],
            "first_step_s": None,
            "remaining_steps_s": [],
            "remaining_steps_total_s": 0.0,
            "remaining_steps_avg_s": None,
        }
    remaining = step_times[1:]
    return {
        "step_solve_times_s": step_times,
        "first_step_s": float(step_times[0]),
        "remaining_steps_s": remaining,
        "remaining_steps_total_s": float(sum(remaining)),
        "remaining_steps_avg_s": (float(sum(remaining) / len(remaining)) if remaining else None),
    }


def nonlinear_cost_breakdown(history: list[Any]) -> dict[str, Any]:
    def _sum_attr(step: Any, attr: str) -> float:
        total = 0.0
        for rec in getattr(step, "iter_history", []):
            val = getattr(rec, attr, None)
            if val is not None:
                total += float(val)
        return total

    per_step_linear = [_sum_attr(step, "linear_time") for step in history]
    per_step_eval = [_sum_attr(step, "eval_time") for step in history]
    per_step_rhs = [_sum_attr(step, "rhs_time") for step in history]
    per_step_pre = [_sum_attr(step, "preconditioner_time") for step in history]
    per_step_linearize = [_sum_attr(step, "linearize_time") for step in history]
    per_step_control = [_sum_attr(step, "control_time") for step in history]
    per_step_init_residual = [_sum_attr(step, "initial_residual_time") for step in history]
    per_step_init_jacobian = [_sum_attr(step, "initial_jacobian_time") for step in history]
    step_solve = [float(getattr(step, "solve_time", 0.0)) for step in history]
    per_step_other = [
        max(0.0, s - (lin + ev + rhs + pre + linz + ctrl + init_r + init_j))
        for s, lin, ev, rhs, pre, linz, ctrl, init_r, init_j in zip(
            step_solve,
            per_step_linear,
            per_step_eval,
            per_step_rhs,
            per_step_pre,
            per_step_linearize,
            per_step_control,
            per_step_init_residual,
            per_step_init_jacobian,
        )
    ]
    total_solve = float(sum(step_solve))

    def _share(vals: list[float]) -> float | None:
        if total_solve <= 0.0:
            return None
        return float(sum(vals) / total_solve)

    return {
        "per_step_linear_solve_s": per_step_linear,
        "per_step_residual_eval_s": per_step_eval,
        "per_step_rhs_s": per_step_rhs,
        "per_step_preconditioner_s": per_step_pre,
        "per_step_linearize_s": per_step_linearize,
        "per_step_control_s": per_step_control,
        "per_step_initial_residual_s": per_step_init_residual,
        "per_step_initial_jacobian_s": per_step_init_jacobian,
        "per_step_other_s": per_step_other,
        "linear_solve_total_s": float(sum(per_step_linear)),
        "residual_eval_total_s": float(sum(per_step_eval)),
        "rhs_total_s": float(sum(per_step_rhs)),
        "preconditioner_total_s": float(sum(per_step_pre)),
        "linearize_total_s": float(sum(per_step_linearize)),
        "control_total_s": float(sum(per_step_control)),
        "initial_residual_total_s": float(sum(per_step_init_residual)),
        "initial_jacobian_total_s": float(sum(per_step_init_jacobian)),
        "other_total_s": float(sum(per_step_other)),
        "linear_solve_share": _share(per_step_linear),
        "residual_eval_share": _share(per_step_eval),
        "rhs_share": _share(per_step_rhs),
        "preconditioner_share": _share(per_step_pre),
        "linearize_share": _share(per_step_linearize),
        "control_share": _share(per_step_control),
        "initial_residual_share": _share(per_step_init_residual),
        "initial_jacobian_share": _share(per_step_init_jacobian),
        "other_share": _share(per_step_other),
    }


def ensure_meshes(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    meshes: list[Path] = []
    for lc in args.lc_values:
        mesh_path = out_dir / f"tension_bar_lc{str(lc).replace('.', '_')}.msh"
        proc = subprocess.run(
            [args.gmsh, "-3", "-setnumber", "lc", f"{lc}", "tension_bar.geo", "-o", str(mesh_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gmsh failed for lc={lc}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        meshes.append(mesh_path)
    return meshes


def run_case(mesh_path: Path, args: argparse.Namespace, policy: ff.AssemblyPolicy | None) -> dict[str, Any]:
    dtype = jnp.float64
    mesh, facets, facet_tags = ff.load_gmsh_tet_mesh(str(mesh_path))
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=dtype),
        conn=jnp.asarray(mesh.conn, dtype=mesh.conn.dtype),
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )
    space = ff.make_tet_space(mesh, dim=3, intorder=args.intorder)
    V = ff.NamedSpace("V", space)

    lam, mu = ff.lame_parameters(210_000.0, 0.3)
    params = {"lam": lam, "mu": mu}

    F_ext = ff.assemble_linear_form(
        ff.LinearSpaces(test=V),
        ff.vector_body_force_form,
        jnp.array([0.0, 0.0, 0.0], dtype=dtype),
    )

    right_facets = np.asarray(facets)[np.asarray(facet_tags) == 2]
    surface = ff.make_surface_from_facets(np.asarray(mesh.coords), right_facets)

    def traction_form(ctx: ff.SurfaceFormContext, traction_vec: np.ndarray) -> np.ndarray:
        return h_ts.dot(ctx.v, traction_vec)

    F_ext = surface.assemble_linear_form_on_space(
        space,
        traction_form,
        params=np.array([0.0, args.traction, 0.0], dtype=float),
        F0=F_ext,
    )

    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    ).dofs

    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params=params,
        base_external_vector=F_ext,
        dirichlet=ff.DirichletBC(dir_dofs, None),
        assembly_policy=policy,
        jacobian_pattern=ff.make_sparsity_pattern(space, with_idx=False),
        dtype=dtype,
    )
    cfg = ff.NewtonLoopConfig(
        tol=args.tol,
        atol=args.atol,
        maxiter=args.maxiter,
        linear_solver=args.linear_solver,
        linear_preconditioner=(None if args.linear_precond == "none" else args.linear_precond),
        petsc_ksp_type=args.petsc_ksp_type,
        petsc_pc_type=args.petsc_pc_type,
        petsc_use_pmat=args.petsc_use_pmat,
        line_search=args.line_search,
        n_steps=args.nstep,
    )
    runner = ff.NewtonSolveRunner(analysis, cfg)
    timer = SectionTimer(hierarchical=True)
    t0 = time.perf_counter()
    u, history = runner.run(
        u0=jnp.zeros(space.n_dofs, dtype=dtype),
        timer=timer,
        report_timing=False,
    )
    wall = float(time.perf_counter() - t0)
    u_nodes = np.asarray(u).reshape(-1, 3)
    timer_records = {name: [float(v) for v in vals] for name, vals in sorted(timer._records.items())}
    policy_meta = policy_stats(int(np.asarray(mesh.conn).shape[0]), policy)
    step_meta = step_time_breakdown(history, timer_records)
    cost_meta = nonlinear_cost_breakdown(history)
    return {
        "mesh": str(mesh_path),
        "n_nodes": int(np.asarray(mesh.coords).shape[0]),
        "n_elems": int(np.asarray(mesh.conn).shape[0]),
        "wall_time_s": wall,
        "load_steps": len(history),
        "iters": [int(getattr(step.info, "iters", 0)) for step in history],
        "iters_sum": int(sum(int(getattr(step.info, "iters", 0)) for step in history)),
        "max_disp": float(np.linalg.norm(u_nodes, axis=1).max()) if u_nodes.size else 0.0,
        **policy_meta,
        **step_meta,
        **cost_meta,
        "timer_records_s": timer_records,
    }


def render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# FluxFEM bucketed warm-run experiment",
        "",
        f"- mode: `{payload['mode']}`",
        f"- lc values: `{payload['config']['lc_values']}`",
        f"- nstep: `{payload['config']['nstep']}`",
        "",
        "| order | mesh | nodes | elems | n_pad | pad ratio | wall [s] | first step [s] | rest avg [s] | iters |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, item in enumerate(payload["results"], start=1):
        lines.append(
            f"| {i} | {Path(item['mesh']).name} | {item['n_nodes']} | {item['n_elems']} | "
            f"{item['n_pad']} | {item['pad_ratio']:.3f} | {item['wall_time_s']:.3f} | "
            f"{(item['first_step_s'] if item['first_step_s'] is not None else float('nan')):.3f} | "
            f"{(item['remaining_steps_avg_s'] if item['remaining_steps_avg_s'] is not None else float('nan')):.3f} | "
            f"{item['iters_sum']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    policy = build_policy(args)
    if policy is None:
        mode = "plain"
    elif policy.max_padded_elems is None and policy.allow_tail_chunk:
        mode = "fixed_chunk_tail"
    else:
        mode = "bucketed"
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meshes = ensure_meshes(args, out_path.parent)

    results = [run_case(mesh, args, policy) for mesh in meshes]
    payload = {
        "mode": mode,
        "config": {
            "lc_values": [float(v) for v in args.lc_values],
            "intorder": int(args.intorder),
            "traction": float(args.traction),
            "nstep": int(args.nstep),
            "maxiter": int(args.maxiter),
            "tol": float(args.tol),
            "atol": float(args.atol),
            "linear_solver": args.linear_solver,
            "linear_precond": args.linear_precond,
            "petsc_ksp_type": args.petsc_ksp_type,
            "petsc_pc_type": args.petsc_pc_type,
            "petsc_use_pmat": bool(args.petsc_use_pmat),
            "line_search": bool(args.line_search),
            "bucket_size": args.bucket_size,
            "chunk_size": args.chunk_size,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_path.with_suffix(".md").write_text(render_summary(payload), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

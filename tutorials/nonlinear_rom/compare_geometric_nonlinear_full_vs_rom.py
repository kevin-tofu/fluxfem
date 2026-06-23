#!/usr/bin/env python3
"""Compare geometric nonlinear full FEM and nonlinear Galerkin ROM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

import fluxfem as ff


jax.config.update("jax_enable_x64", True)
TUTORIALS_ROOT = Path(__file__).resolve().parents[1]
if str(TUTORIALS_ROOT) not in sys.path:
    sys.path.insert(0, str(TUTORIALS_ROOT))

from common.basis import DenseBasis


BASIS_LABELS = {
    "identity-full": "full-coordinate check",
    "free-dofs": "free-coordinate check",
    "linearized-modes": "linearized modal ROM",
    "cantilever-bending-y": "1-mode bending ROM",
}

BASIS_ROLES = {
    "identity-full": "Regression check: the ROM basis is the full coordinate basis.",
    "free-dofs": "Constraint-handling check: prescribed coordinates are removed, but all free DOFs are retained.",
    "linearized-modes": "Low-dimensional ROM: lowest vibration modes of the initial linearized elastic model.",
    "cantilever-bending-y": "Deliberately tiny ROM: one assumed cantilever bending shape in the loading direction.",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=0.25)
    parser.add_argument("--lz", type=float, default=0.25)
    parser.add_argument("--E", type=float, default=250.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--force", type=float, default=-0.001)
    parser.add_argument(
        "--basis",
        choices=("identity-full", "free-dofs", "linearized-modes", "cantilever-bending-y", "all"),
        default="all",
        help=(
            "ROM basis to compare: identity-full is an exact full-coordinate "
            "regression check, free-dofs removes fixed coordinates, and "
            "linearized-modes uses low vibration modes of the initial linearized "
            "model. cantilever-bending-y is a one-coordinate bending-shape ROM."
        ),
    )
    parser.add_argument("--modal-modes", type=int, default=6, help="Number of linearized modes for --basis linearized-modes.")
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / Path(__file__).stem),
    )
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-plot", type=str, default="")
    return parser.parse_args()


def _tool_node(coords: np.ndarray) -> int:
    xmax = float(coords[:, 0].max())
    ymax = float(coords[:, 1].max())
    zmax = float(coords[:, 2].max())
    ids = np.flatnonzero(
        np.isclose(coords[:, 0], xmax)
        & np.isclose(coords[:, 1], ymax)
        & np.isclose(coords[:, 2], zmax)
    )
    if ids.size != 1:
        raise RuntimeError("expected one tool node at the upper free corner.")
    return int(ids[0])


def _coordinate_selection_basis(n_full: int, dofs) -> DenseBasis:
    dofs = jnp.asarray(dofs, dtype=jnp.int32)
    return DenseBasis(jnp.eye(n_full, dtype=jnp.float64)[:, dofs])


def _cantilever_bending_y_shape_basis(space, coords: np.ndarray) -> DenseBasis:
    x = coords[:, 0]
    xmin = float(x.min())
    length = max(float(x.max() - xmin), 1.0e-30)
    xi = (x - xmin) / length
    shape = xi * xi * (3.0 - 2.0 * xi)
    phi = np.zeros(space.n_dofs, dtype=float)
    phi[1::3] = shape
    norm = np.linalg.norm(phi)
    if norm <= 0.0:
        raise ValueError("failed to build cantilever-bending-y basis: zero shape vector")
    return DenseBasis(jnp.asarray(phi[:, None] / norm, dtype=jnp.float64))


def _linearized_modal_basis(space, dirichlet, *, elastic_modulus: float, poisson_ratio: float, n_modes: int) -> DenseBasis:
    if n_modes <= 0:
        raise ValueError("modal-modes must be positive")
    stiffness = space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(elastic_modulus, poisson_ratio))
    mass = space.assemble_mass_matrix()
    fixed = set(np.asarray(dirichlet.dofs, dtype=int).tolist())
    free = np.asarray([dof for dof in range(space.n_dofs) if dof not in fixed], dtype=int)
    if free.size == 0:
        raise ValueError("cannot build linearized modal basis with no free DOFs")
    n_keep = min(int(n_modes), int(free.size))
    k_ff = np.asarray(stiffness.to_dense(), dtype=float)[np.ix_(free, free)]
    m_ff = np.asarray(mass.to_dense(), dtype=float)[np.ix_(free, free)]
    eigvals, eigvecs = la.eigh(k_ff, m_ff)
    order = np.argsort(eigvals)
    phi_free = eigvecs[:, order[:n_keep]]
    phi = np.zeros((space.n_dofs, n_keep), dtype=float)
    phi[free, :] = phi_free
    return DenseBasis(jnp.asarray(phi, dtype=jnp.float64))


def _make_basis(
    kind: str,
    space,
    coords: np.ndarray,
    dirichlet,
    *,
    elastic_modulus: float,
    poisson_ratio: float,
    modal_modes: int,
) -> DenseBasis:
    eye = jnp.eye(space.n_dofs, dtype=jnp.float64)
    if kind == "identity-full":
        return DenseBasis(eye)
    if kind == "free-dofs":
        fixed = set(np.asarray(dirichlet.dofs, dtype=int).tolist())
        free = np.asarray([dof for dof in range(space.n_dofs) if dof not in fixed], dtype=np.int32)
        return _coordinate_selection_basis(space.n_dofs, free)
    if kind == "linearized-modes":
        return _linearized_modal_basis(
            space,
            dirichlet,
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            n_modes=modal_modes,
        )
    if kind == "cantilever-bending-y":
        return _cantilever_bending_y_shape_basis(space, coords)
    raise ValueError(f"unknown basis kind: {kind}")


def _solve_full(space, residual_form, params, force, dirichlet, tol: float, maxiter: int):
    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=residual_form,
        params=params,
        base_external_vector=force,
        dirichlet=dirichlet,
        dtype=jnp.float64,
    )
    runner = ff.NewtonSolveRunner(
        analysis,
        ff.NewtonLoopConfig(tol=tol, atol=tol, maxiter=maxiter, linear_solver="spsolve"),
    )
    t0 = time.perf_counter()
    u, history = runner.run(u0=jnp.zeros(space.n_dofs, dtype=jnp.float64), newton_callback=lambda _cb: None)
    solve_time = time.perf_counter() - t0
    return u, history[-1].info, solve_time


def _solve_rom(space, residual_form, params, force, basis, dirichlet, tol: float, maxiter: int):
    t_build = time.perf_counter()
    model = ff.NonlinearReducedFEModel(
        space=space,
        residual_form=residual_form,
        params=params,
        basis=basis,
        external_vector=force,
    )
    build_time = time.perf_counter() - t_build
    t_solve = time.perf_counter()
    q, info = model.as_problem("body").solve(
        jnp.zeros(basis.n_reduced, dtype=jnp.float64),
        fixed_dofs=dirichlet.dofs if basis.n_reduced == space.n_dofs else None,
        fixed_values=jnp.zeros_like(jnp.asarray(dirichlet.dofs, dtype=jnp.float64)) if basis.n_reduced == space.n_dofs else None,
        tol=tol,
        atol=tol,
        maxiter=maxiter,
    )
    solve_time = time.perf_counter() - t_solve
    return model.expand(q), info, build_time, solve_time


def _write_outputs(output_dir: str, basis_name: str, mesh, space, full_u, rom_u):
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ff.write_elastic_vtu(mesh, space, full_u, str(out / "full.vtu"), compute_j=True, deformed_scale=1.0)
    ff.write_elastic_vtu(mesh, space, rom_u, str(out / f"rom_{basis_name}.vtu"), compute_j=True, deformed_scale=1.0)
    ff.write_elastic_vtu(mesh, space, full_u - rom_u, str(out / f"error_{basis_name}.vtu"), compute_j=False, deformed_scale=1.0)
    print(f"VTU written to {out}")


def _write_json(path: str, payload: dict):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"JSON written to {out}")


def _accuracy_verdict(relative_error: float) -> str:
    if relative_error < 1.0e-5:
        return "matches full-order solution"
    if relative_error < 5.0e-2:
        return "usable approximation for this small case"
    return "too inaccurate for this deformation"


def _write_summary(path: str, payload: dict):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Geometric Nonlinear Full FEM vs ROM",
        "",
        "## Bottom Line",
        "",
    ]
    exact_like = [case for case in payload["cases"] if case["relative_error_inf"] < 1.0e-5]
    rough = [case for case in payload["cases"] if case["relative_error_inf"] >= 5.0e-2]
    if exact_like:
        names = ", ".join(BASIS_LABELS.get(case["basis"], case["basis"]) for case in exact_like)
        lines.append(f"- {names} reproduce the full-order displacement for this setup.")
    if rough:
        names = ", ".join(BASIS_LABELS.get(case["basis"], case["basis"]) for case in rough)
        verb = "is" if len(rough) == 1 else "are"
        lines.append(f"- {names} {verb} intentionally small and not accurate enough here.")
    lines.extend(
        [
            "- This tutorial currently demonstrates direct Galerkin projection; it is not yet a production hyper-reduced nonlinear ROM benchmark.",
            "",
            "## Reference Full FEM",
            "",
            f"- DOFs: {payload['problem']['n_full']}",
            f"- converged: {payload['full']['converged']}",
            f"- Newton iterations: {payload['full']['iters']}",
            f"- tool y displacement: {payload['full']['tool_uy']:.6e}",
            f"- build time: {payload['full']['build_time_s']:.6e} s",
            f"- solve time: {payload['full']['solve_time_s']:.6e} s",
            "",
            "## ROM Cases",
            "",
            "| basis | role | reduced DOFs | relative error | tool y displacement | verdict |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for case in payload["cases"]:
        name = BASIS_LABELS.get(case["basis"], case["basis"])
        role = BASIS_ROLES.get(case["basis"], "")
        verdict = _accuracy_verdict(case["relative_error_inf"])
        lines.append(
            f"| {name} | {role} | {case['n_reduced']} | "
            f"{case['relative_error_inf']:.6e} | {case['rom_tool_uy']:.6e} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `identity-full` does not match the full FEM, the projection implementation is wrong.",
            "- If `free-dofs` does not match the full FEM, the handling of prescribed coordinates is wrong.",
            "- `linearized-modes` is the first meaningful low-dimensional basis in this tutorial; its error should drop as `--modal-modes` increases.",
            "- If `cantilever-bending-y` is inaccurate, that is expected: one bending shape cannot represent the full 3D nonlinear displacement field.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to {out}")


def _print_case_summary(payload: dict):
    print("\nresult summary")
    for case in payload["cases"]:
        label = BASIS_LABELS.get(case["basis"], case["basis"])
        verdict = _accuracy_verdict(case["relative_error_inf"])
        print(
            f"- {label}: n={case['n_reduced']}, "
            f"rel_error={case['relative_error_inf']:.3e}, verdict={verdict}"
        )


def _write_plot(path: str, payload: dict):
    if not path:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxfem_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = payload["cases"]
    labels = [BASIS_LABELS.get(case["basis"], case["basis"]) for case in cases]
    error = [case["relative_error_inf"] for case in cases]
    full_build = [payload["full"]["build_time_s"] for _case in cases]
    full_solve = [payload["full"]["solve_time_s"] for _case in cases]
    rom_build = [case["rom_build_time_s"] for case in cases]
    rom_solve = [case["rom_solve_time_s"] for case in cases]
    x = np.arange(len(labels))

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].bar(x, error, color="#4c78a8")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative displacement error inf-norm")
    axes[0].set_title("ROM displacement error")

    width = 0.2
    axes[1].bar(x - 1.5 * width, full_build, width, label="full build", color="#72b7b2")
    axes[1].bar(x - 0.5 * width, full_solve, width, label="full solve", color="#54a24b")
    axes[1].bar(x + 0.5 * width, rom_build, width, label="rom build", color="#f58518")
    axes[1].bar(x + 1.5 * width, rom_solve, width, label="rom solve", color="#e45756")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Build / solve time")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Generated {generated_at}", fontsize=9)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    fig.savefig(out, dpi=180, metadata={"Creation Time": generated_at})
    plt.close(fig)
    print(f"PNG refreshed at {out}")


def main():
    args = parse_args()
    t_build = time.perf_counter()
    mesh = ff.StructuredHexBox(nx=args.nx, ny=args.ny, nz=args.nz, lx=args.lx, ly=args.ly, lz=args.lz).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    tool_node = _tool_node(coords)
    force = jnp.zeros(space.n_dofs, dtype=jnp.float64).at[tool_node * 3 + 1].set(args.force)
    lam, mu = ff.lame_parameters(args.E, args.nu)
    params = {"lam": lam, "mu": mu}
    dirichlet = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin),
        components="xyz",
    )
    full_build_time = time.perf_counter() - t_build

    full_u, full_info, full_solve_time = _solve_full(
        space,
        ff.neo_hookean_residual_form,
        params,
        force,
        dirichlet,
        args.tol,
        args.maxiter,
    )
    full_nodes = np.asarray(full_u).reshape(-1, 3)
    full_inf = float(jnp.linalg.norm(full_u, ord=jnp.inf))
    basis_names = (
        ("identity-full", "free-dofs", "linearized-modes", "cantilever-bending-y")
        if args.basis == "all"
        else (args.basis,)
    )
    cases = []

    print("geometric nonlinear full FEM vs ROM")
    print(f"n_full: {space.n_dofs}")
    print(f"full_converged: {full_info.converged}")
    print(f"full_iters: {full_info.iters}")
    print(f"full_tool_uy: {full_nodes[tool_node, 1]:.6e}")
    print(f"full_build_time_s: {full_build_time:.6e}")
    print(f"full_solve_time_s: {full_solve_time:.6e}")

    for basis_name in basis_names:
        basis = _make_basis(
            basis_name,
            space,
            coords,
            dirichlet,
            elastic_modulus=args.E,
            poisson_ratio=args.nu,
            modal_modes=args.modal_modes,
        )
        rom_u, rom_info, rom_build_time, rom_solve_time = _solve_rom(
            space,
            ff.neo_hookean_residual_form,
            params,
            force,
            basis,
            dirichlet,
            args.tol,
            args.maxiter,
        )
        rom_nodes = np.asarray(rom_u).reshape(-1, 3)
        error = full_u - rom_u
        error_inf = float(jnp.linalg.norm(error, ord=jnp.inf))
        error_l2 = float(jnp.linalg.norm(error))
        rel_inf = error_inf / max(full_inf, 1.0e-30)
        case = {
            "basis": basis_name,
            "basis_label": BASIS_LABELS.get(basis_name, basis_name),
            "basis_role": BASIS_ROLES.get(basis_name, ""),
            "basis_parameters": {"modal_modes": int(args.modal_modes)} if basis_name == "linearized-modes" else {},
            "n_reduced": basis.n_reduced,
            "rom_converged": bool(rom_info.converged),
            "rom_iters": int(rom_info.iters),
            "rom_tool_uy": float(rom_nodes[tool_node, 1]),
            "error_inf": error_inf,
            "error_l2": error_l2,
            "relative_error_inf": rel_inf,
            "rom_build_time_s": rom_build_time,
            "rom_solve_time_s": rom_solve_time,
        }
        cases.append(case)
        print(f"[{basis_name}] n_reduced: {basis.n_reduced}")
        print(f"[{basis_name}] rom_converged: {rom_info.converged}")
        print(f"[{basis_name}] rom_iters: {rom_info.iters}")
        print(f"[{basis_name}] rom_tool_uy: {rom_nodes[tool_node, 1]:.6e}")
        print(f"[{basis_name}] error_inf: {error_inf:.6e}")
        print(f"[{basis_name}] relative_error_inf: {rel_inf:.6e}")
        print(f"[{basis_name}] rom_build_time_s: {rom_build_time:.6e}")
        print(f"[{basis_name}] rom_solve_time_s: {rom_solve_time:.6e}")
        _write_outputs(args.output_dir, basis_name, mesh, space, full_u, rom_u)

    payload = {
        "problem": {
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "n_full": int(space.n_dofs),
            "force": float(args.force),
            "modal_modes": int(args.modal_modes),
            "tol": float(args.tol),
            "maxiter": int(args.maxiter),
        },
        "full": {
            "converged": bool(full_info.converged),
            "iters": int(full_info.iters),
            "tool_uy": float(full_nodes[tool_node, 1]),
            "build_time_s": full_build_time,
            "solve_time_s": full_solve_time,
        },
        "cases": cases,
    }
    output_json = args.output_json or (str(Path(args.output_dir) / "metrics.json") if args.output_dir else "")
    output_plot = args.output_plot or (str(Path(args.output_dir) / "comparison.png") if args.output_dir else "")
    output_summary = str(Path(args.output_dir) / "summary.md") if args.output_dir else ""
    _print_case_summary(payload)
    _write_json(output_json, payload)
    _write_plot(output_plot, payload)
    _write_summary(output_summary, payload)

    if not full_info.converged or any(not case["rom_converged"] for case in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

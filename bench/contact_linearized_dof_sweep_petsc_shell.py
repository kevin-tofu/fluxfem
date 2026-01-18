#!/usr/bin/env python3
"""
Benchmark a single linearized contact step (non-SPD) for solver comparisons.

Compares:
- fluxfem assembly time (elastic + contact)
- fluxfem solve time with PETSc AIJ (optional)
- fluxfem solve time with PETSc shell (matrix-free, none/diag0)
- (optional) fluxfem CG solve (may fail for non-SPD)

USAGE
-----
PYTHONPATH=src python bench/contact_linearized_dof_sweep_petsc_shell.py --sizes 4 6 8 --repeats 3 --warmup 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import fluxfem as ff  # noqa: E402
import fluxfem.helpers_wf as h_wf  # noqa: E402
from fluxfem.core.weakform import einsum as wf_einsum  # noqa: E402


def env_default(name: str, default, cast):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return cast(val)
    except Exception:
        return default


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark a single linearized contact step (non-SPD).")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=env_default("SIZES", [4, 6, 8], lambda v: [int(x) for x in str(v).split(",")]),
        help="Element counts per axis for both top and bottom meshes.",
    )
    p.add_argument("--repeats", type=int, default=env_default("REPEATS", 3, int))
    p.add_argument("--warmup", type=int, default=env_default("WARMUP", 1, int))
    p.add_argument("--intorder", type=int, default=env_default("INTORDER", 1, int))
    p.add_argument("--E", type=float, default=env_default("E", 210_000.0, float))
    p.add_argument("--nu", type=float, default=env_default("NU", 0.3, float))
    p.add_argument("--contact-backend", choices=["jax", "numpy"], default=os.environ.get("CONTACT_BACKEND", "jax"))
    p.add_argument("--quad-order", type=int, default=env_default("QUAD_ORDER", 2, int))
    p.add_argument("--alpha", type=float, default=env_default("ALPHA", 1.0, float))
    p.add_argument("--total-force", type=float, default=env_default("TOTAL_FORCE", -1.0, float))
    p.add_argument("--no-cg", action="store_true", help="Disable CG benchmark (CG may fail for non-SPD).")
    p.add_argument("--cg-tol", type=float, default=env_default("CG_TOL", 1e-8, float))
    p.add_argument("--cg-maxiter", type=int, default=env_default("CG_MAXITER", 2000, int))
    p.add_argument(
        "--petsc",
        action=argparse.BooleanOptionalAction,
        default=env_default("PETSCSOLVE", True, lambda v: str(v).lower() not in {"0", "false", "no"}),
        help="Enable PETSc AIJ solve benchmark when petsc4py is available.",
    )
    p.add_argument("--petsc-ksp", type=str, default=os.environ.get("PETSCSOLVE_KSP", "gmres"))
    p.add_argument("--petsc-pc", type=str, default=os.environ.get("PETSCSOLVE_PC", "ilu"))
    p.add_argument(
        "--petsc-shell",
        action=argparse.BooleanOptionalAction,
        default=env_default("PETSCSHELL", True, lambda v: str(v).lower() not in {"0", "false", "no"}),
        help="Enable PETSc shell (matrix-free) solve benchmark when petsc4py is available.",
    )
    p.add_argument("--petsc-shell-ksp", type=str, default=os.environ.get("PETSCSHELL_KSP", "gmres"))
    p.add_argument("--petsc-shell-pc", type=str, default=os.environ.get("PETSCSHELL_PC", "none"))
    p.add_argument(
        "--petsc-shell-precon",
        choices=["none", "diag0", "pmat", "both", "all"],
        default=os.environ.get("PETSCSHELL_PRECON", "both"),
        help="Preconditioner mode for PETSc shell: none, diag0, pmat, both, or all.",
    )
    p.add_argument("--petsc-shell-rtol", type=float, default=env_default("PETSCSHELL_RTOL", 1e-8, float))
    p.add_argument("--petsc-shell-atol", type=float, default=env_default("PETSCSHELL_ATOL", 0.0, float))
    p.add_argument("--petsc-shell-maxiter", type=int, default=env_default("PETSCSHELL_MAXITER", 2000, int))
    p.add_argument(
        "--petsc-shell-pmat",
        action=argparse.BooleanOptionalAction,
        default=env_default("PETSCSHELL_PMAT", False, lambda v: str(v).lower() in {"1", "true", "yes"}),
        help="Use AIJ pmat for PETSc shell (A=Shell, P=AIJ).",
    )
    p.add_argument(
        "--petsc-shell-gmres-restart",
        type=int,
        default=env_default("PETSCSHELL_GMRES_RESTART", None, lambda v: int(v)),
        help="GMRES restart (optional; only used for gmres/fgmres).",
    )
    p.add_argument(
        "--petsc-shell-monitor-true-residual",
        action=argparse.BooleanOptionalAction,
        default=env_default("PETSCSHELL_MONITOR_TRUE_RESIDUAL", False, lambda v: str(v).lower() in {"1", "true", "yes"}),
        help="Enable PETSc true residual monitor output.",
    )
    p.add_argument(
        "--petsc-shell-converged-reason",
        action=argparse.BooleanOptionalAction,
        default=env_default("PETSCSHELL_CONVERGED_REASON", False, lambda v: str(v).lower() in {"1", "true", "yes"}),
        help="Print PETSc converged reason.",
    )
    p.add_argument(
        "--petsc-shell-norm-type",
        type=str,
        default=os.environ.get("PETSCSHELL_NORM_TYPE", ""),
        help="KSP norm type (e.g., unpreconditioned).",
    )
    p.add_argument(
        "--json",
        type=str,
        default="result/bench/bench_contact_linearized_petsc_shell/results.json",
        help="Output JSON path for results",
    )
    return p.parse_args()


def _mesh_spacing(box):
    return box.lx / box.nx, box.ly / box.ny, box.lz / box.nz


def _residual_error(K_free, F_free, u_free) -> float:
    r = np.asarray(K_free.matvec(u_free)) - np.asarray(F_free)
    denom = np.linalg.norm(F_free) + 1e-30
    return float(np.linalg.norm(r) / denom)


def assemble_contact_system(n: int, args):
    box_top = ff.StructuredTetTensorBox(nx=n, ny=n, nz=n, lx=2.0, ly=2.0, lz=1.0, origin=(0.0, 0.0, 0.0))
    box_bot = ff.StructuredTetTensorBox(nx=n, ny=n, nz=n, lx=1.0, ly=1.0, lz=0.5, origin=(0.5, 0.5, -0.5))
    mesh_top = box_top.build()
    mesh_bot = box_bot.build()

    space_top = ff.make_tet_space(mesh_top, dim=3, intorder=args.intorder)
    space_bot = ff.make_tet_space(mesh_bot, dim=3, intorder=args.intorder)

    D = ff.isotropic_3d_D(args.E, args.nu)
    K1 = space_top.assemble_bilinear_form(ff.linear_elasticity_form, params=D)
    K2 = space_bot.assemble_bilinear_form(ff.linear_elasticity_form, params=D)

    contact_facets_bot = mesh_bot.facets_on_plane(axis=2, value=0.0)
    x0, y0, _ = box_bot.origin
    x1 = x0 + box_bot.lx
    y1 = y0 + box_bot.ly
    dx_top, dy_top, _ = _mesh_spacing(box_top)
    pad = 2.0 * min(dx_top, dy_top)
    contact_facets_top = mesh_top.facets_on_plane_box(
        axis=2,
        value=0.0,
        x=(x0 - pad, x1 + pad),
        y=(y0 - pad, y1 + pad),
        mode="centroid",
    )

    side_top = ff.ContactSide.from_facets(mesh_top, contact_facets_top, space_top)
    side_bot = ff.ContactSide.from_facets(mesh_bot, contact_facets_bot, space_bot)
    contact = ff.ContactSurfaceSpace.from_sides(
        side_top,
        side_bot,
        quad_order=int(args.quad_order),
        backend=args.contact_backend,
    )

    dx_top, dy_top, dz_top = _mesh_spacing(box_top)
    dx_bot, dy_bot, dz_bot = _mesh_spacing(box_bot)
    h = min(dx_top, dy_top, dz_top, dx_bot, dy_bot, dz_bot)
    lam, mu = ff.lame_parameters(args.E, args.nu)
    params_contact = ff.Params(alpha=float(args.alpha), inv_h=float(1.0 / h), lam=float(lam), mu=float(mu))

    def bilin(v1, v2, u1, u2, p):
        n = h_wf.normal()
        ju = u1.val - u2.val
        t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
        t_v1 = h_wf.traction(v1, n, p)
        t_v2 = h_wf.traction(v2, n, p)
        penalty = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        traction = -h_wf.dot(v1, t_u) + h_wf.dot(v2, t_u)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
        return (penalty + traction) * h_wf.ds()

    if args.contact_backend == "numpy":
        u_top0 = np.zeros(space_top.n_dofs)
        u_bot0 = np.zeros(space_bot.n_dofs)
    else:
        u_top0 = jnp.zeros(space_top.n_dofs)
        u_bot0 = jnp.zeros(space_bot.n_dofs)

    contact_coo = contact.assemble_bilinear(bilin, (u_top0, u_bot0), params_contact, sparse=True)
    K_contact = ff.FluxSparseMatrix.from_bilinear(contact_coo)

    K_block = ff.block_diag_flux(K1, K2)
    K = ff.concat_flux(K_block, K_contact, n_dofs=K_block.n_dofs)

    force_facets_top = mesh_top.facets_on_plane(axis=2, value=1.0)
    force_surface = ff.SurfaceMesh.from_facets(mesh_top.coords, force_facets_top)
    area = float(np.sum(force_surface.facet_areas()))
    pressure = float(args.total_force) / area
    top_F = force_surface.assemble_load(
        load=np.array([0.0, 0.0, pressure], dtype=float),
        dim=3,
        n_total_nodes=mesh_top.n_nodes,
    )
    bot_F = np.zeros(space_bot.n_dofs, dtype=float)
    F = np.hstack([top_F, bot_F])

    dir_dofs_bot = mesh_bot.boundary_dofs_plane(axis=2, value=-0.5, dof_per_node=3)
    dir_dofs = dir_dofs_bot + space_top.n_dofs
    dir_vals = np.zeros(dir_dofs.shape[0], dtype=float)

    n_total = int(K.n_dofs)
    dir_dofs = np.asarray(dir_dofs, dtype=int)
    free_mask = np.ones(n_total, dtype=bool)
    free_mask[dir_dofs] = False
    free = np.nonzero(free_mask)[0]

    if np.all(dir_vals == 0.0):
        F_free = np.asarray(F, dtype=float)[free]
    else:
        u_dir = jnp.zeros(n_total, dtype=jnp.asarray(F).dtype).at[dir_dofs].set(jnp.asarray(dir_vals))
        F_free = np.asarray(jnp.asarray(F) - K.matvec(u_dir), dtype=float)[free]

    K_free = ff.restrict_flux_to_free(K, free)
    return {
        "K_free": K_free,
        "F_free": F_free,
        "free_dofs": int(K_free.n_dofs),
        "total_dofs": int(K.n_dofs),
    }


def time_solver_samples(fn, repeats: int):
    times = []
    outputs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        outputs.append(fn())
        times.append(time.perf_counter() - t0)
    return np.asarray(times, dtype=float), outputs


def summarize(samples: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(samples)),
        "mean": float(np.mean(samples)),
        "max": float(np.max(samples)),
        "median": float(np.median(samples)),
    }


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def main():
    args = parse_args()
    petsc_avail = ff.petsc_is_available()
    shell_options = {}
    if args.petsc_shell_monitor_true_residual:
        shell_options["fluxfem_ksp_monitor_true_residual"] = ""
    if args.petsc_shell_converged_reason:
        shell_options["fluxfem_ksp_converged_reason"] = ""
    if args.petsc_shell_norm_type:
        shell_options["fluxfem_ksp_norm_type"] = args.petsc_shell_norm_type
    if args.petsc_shell_gmres_restart is not None:
        shell_options["fluxfem_ksp_gmres_restart"] = str(args.petsc_shell_gmres_restart)

    results = []
    for n in args.sizes:
        # Assembly timing
        warmup_times = []
        if args.warmup:
            for _ in range(args.warmup):
                t0 = time.perf_counter()
                _ = assemble_contact_system(n, args)
                warmup_times.append(time.perf_counter() - t0)

        assembly_times = []
        last = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            last = assemble_contact_system(n, args)
            assembly_times.append(time.perf_counter() - t0)
        if last is None:
            raise RuntimeError("Failed to assemble contact system.")

        K_free = last["K_free"]
        F_free = last["F_free"]

        # CG (optional)
        solve_cg = np.full((args.repeats,), np.nan, dtype=float)
        residual_cg = np.full((args.repeats,), np.nan, dtype=float)
        cg_iters = np.full((args.repeats,), np.nan, dtype=float)
        if not args.no_cg:
            def _cg_once():
                return ff.cg_solve(K_free, F_free, tol=args.cg_tol, maxiter=args.cg_maxiter, preconditioner=None)

            times, outputs = time_solver_samples(_cg_once, args.repeats)
            solve_cg = times
            for i, out in enumerate(outputs):
                try:
                    u, info = out
                    cg_iters[i] = float(info.get("iters", np.nan))
                    residual_cg[i] = _residual_error(K_free, F_free, np.asarray(u))
                except Exception:
                    cg_iters[i] = np.nan
                    residual_cg[i] = np.nan

        # PETSc AIJ (optional)
        solve_petsc = np.full((args.repeats,), np.nan, dtype=float)
        residual_petsc = np.full((args.repeats,), np.nan, dtype=float)
        if args.petsc and petsc_avail:
            from fluxfem.solver.petsc import petsc_solve

            def _petsc_once():
                return petsc_solve(K_free, F_free, ksp_type=args.petsc_ksp, pc_type=args.petsc_pc)

            times, outputs = time_solver_samples(_petsc_once, args.repeats)
            solve_petsc = times
            for i, u in enumerate(outputs):
                residual_petsc[i] = _residual_error(K_free, F_free, np.asarray(u))

        # PETSc shell (matrix-free)
        shell_samples = {}
        if args.petsc_shell_precon == "both":
            shell_modes = ["none", "diag0"]
        elif args.petsc_shell_precon == "all":
            shell_modes = ["none", "diag0", "pmat"]
        else:
            shell_modes = [args.petsc_shell_precon]
        for mode in shell_modes:
            shell_samples[mode] = {
                "solve": np.full((args.repeats,), np.nan, dtype=float),
                "residual": np.full((args.repeats,), np.nan, dtype=float),
                "iters": np.full((args.repeats,), np.nan, dtype=float),
            }

        if args.petsc_shell and petsc_avail:
            from fluxfem.solver.petsc import petsc_shell_solve
            for mode in shell_modes:
                if mode == "pmat":
                    precon = None
                    use_pmat = True
                else:
                    precon = None if mode == "none" else "diag0"
                    use_pmat = args.petsc_shell_pmat

                def _shell_once():
                    return petsc_shell_solve(
                        K_free,
                        F_free,
                        ksp_type=args.petsc_shell_ksp,
                        pc_type=args.petsc_shell_pc,
                        preconditioner=precon,
                        rtol=args.petsc_shell_rtol,
                        atol=args.petsc_shell_atol,
                        max_it=args.petsc_shell_maxiter,
                        pmat=K_free if use_pmat else None,
                        options=shell_options if shell_options else None,
                        return_info=True,
                    )

                times, outputs = time_solver_samples(_shell_once, args.repeats)
                shell_samples[mode]["solve"] = times
                for i, out in enumerate(outputs):
                    u, info = out
                    shell_samples[mode]["iters"][i] = float(info.get("iters", np.nan))
                    shell_samples[mode]["residual"][i] = _residual_error(K_free, F_free, np.asarray(u))

        payload = {
            "n": n,
            "free_dofs": int(last["free_dofs"]),
            "total_dofs": int(last["total_dofs"]),
            "assembly_warmup_samples": np.asarray(warmup_times, dtype=float),
            "assembly_samples": np.asarray(assembly_times, dtype=float),
            "solve_cg_samples": solve_cg,
            "cg_iters_samples": cg_iters,
            "residual_cg_samples": residual_cg,
            "solve_petsc_samples": solve_petsc,
            "residual_petsc_samples": residual_petsc,
            "petsc_shell": shell_samples,
        }
        results.append(payload)

        asm_stats = summarize(payload["assembly_samples"])
        msg = (
            f"\n--- n={n} ---\n"
            f"assembly mean={asm_stats['mean']:.3e}s [min={asm_stats['min']:.3e}, max={asm_stats['max']:.3e}]"
        )
        if not args.no_cg:
            cg_stats = summarize(payload["solve_cg_samples"])
            it_stats = summarize(payload["cg_iters_samples"])
            res_stats = summarize(payload["residual_cg_samples"])
            msg += (
                f", cg mean={cg_stats['mean']:.3e}s [min={cg_stats['min']:.3e}, max={cg_stats['max']:.3e}]"
                f" (iters median~{it_stats['median']:.1f}, residual med={res_stats['median']:.3e})"
            )
        if args.petsc and petsc_avail:
            petsc_stats = summarize(payload["solve_petsc_samples"])
            res_stats = summarize(payload["residual_petsc_samples"])
            msg += (
                f", petsc mean={petsc_stats['mean']:.3e}s [min={petsc_stats['min']:.3e}, max={petsc_stats['max']:.3e}]"
                f", residual med={res_stats['median']:.3e}"
            )
        if args.petsc_shell and petsc_avail:
            for mode, samples in payload["petsc_shell"].items():
                if np.any(np.isfinite(samples["solve"])):
                    shell_stats = summarize(samples["solve"])
                    it_stats = summarize(samples["iters"])
                    res_stats = summarize(samples["residual"])
                    msg += (
                        f", petsc_shell[{mode}] mean={shell_stats['mean']:.3e}s"
                        f" [min={shell_stats['min']:.3e}, max={shell_stats['max']:.3e}]"
                        f" (iters median~{it_stats['median']:.1f}, residual med={res_stats['median']:.3e})"
                    )
        print(msg)

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args), "results": _to_jsonable(results)}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()

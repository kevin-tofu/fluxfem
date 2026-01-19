#!/usr/bin/env python3
"""
Benchmark linear-elastic assembly/solve time vs DOF.

Compares:
- fluxfem assembly time (includes first-call JIT compile in the samples)
- fluxfem solve time with SciPy `spsolve`
- fluxfem solve time with PETSc AIJ (if petsc4py available)
- fluxfem solve time with PETSc shell (matrix-free, none/diag0)
- (optional) fluxfem solve time with in-house `cg_solve`
- scikit-fem assembly + `solve` (if installed)

USAGE
-----
Basic (default: includes CG benchmark):
  PYTHONPATH=src python bench/linearelastic_dof_sweep_petsc_shell.py

Disable CG benchmark (skip cg_solve timing + plotting + CSV column):
  PYTHONPATH=src python bench/linearelastic_dof_sweep_petsc_shell.py --no-cg

Custom sweep and settings:
  PYTHONPATH=src python bench/linearelastic_dof_sweep_petsc_shell.py --sizes 8 12 16 20 --repeats 5 --intorder 2 --warmup 1

Environment variable equivalents (optional):
  SIZES="8,12,16,20" NY_MULT=1.0 NZ_MULT=1.0 REPEATS=5 WARMUP=1 INTORDER=2 PLOT="solve_benchmark.png"

Notes
-----
- This script enables JAX x64 by default (jax_enable_x64=True).
- scikit-fem comparison is skipped if scikit-fem is not installed.
- IMPORTANT: Use --warmup to exclude JIT compilation from timing statistics.
  We report the distribution over `--repeats` samples (min/mean/max), not just the mean.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import matplotlib
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fluxfem import (
    FluxSparseMatrix,
    SurfaceMesh,
    constant_body_force_vector_form,
    lame_parameters,
    isotropic_3d_D,
    linear_elasticity_form,
    make_hex_space,
    make_element_bilinear_kernel,
    petsc_is_available,
    tag_axis_minmax_facets,
)

from fluxfem.tools.timer import SectionTimer


from linearelastic_dof_sweep_common import (
    make_structured_mesh,
    compute_dirichlet_dofs,
    make_block_jacobi_preconditioner,
    _residual_error,
    time_flux_cg_samples,
    time_spsolve_samples,
    time_petsc_samples,
    time_petsc_shell_samples,
    summarize,
    prepare_kernel_breakdown,
)


def env_default(name: str, default, cast):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return cast(val)
    except Exception:
        return default


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark linear-elastic assembly/solve time vs DOF.")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=env_default("SIZES", [8, 12, 16, 20], lambda v: [int(x) for x in str(v).split(",")]),
        help="Element counts in x; ny/nz scale with multipliers.",
    )
    p.add_argument("--ny-mult", type=float, default=env_default("NY_MULT", 1.0, float), help="ny = round(n * ny_mult)")
    p.add_argument("--nz-mult", type=float, default=env_default("NZ_MULT", 1.0, float), help="nz = round(n * nz_mult)")
    p.add_argument("--lx", type=float, default=env_default("LX", 100.0, float))
    p.add_argument("--ly", type=float, default=env_default("LY", 10.0, float))
    p.add_argument("--lz", type=float, default=env_default("LZ", 10.0, float))
    p.add_argument("--fx", type=float, default=env_default("FX", 0.0, float))
    p.add_argument("--fy", type=float, default=env_default("FY", 0.0, float))
    p.add_argument("--fz", type=float, default=env_default("FZ", 0.0, float))
    p.add_argument(
        "--traction",
        type=float,
        default=env_default("TRACTION", 0.01, float),
        help="Uniform traction on +x face (applied in Fx direction)",
    )
    p.add_argument("--E", type=float, default=env_default("E", 210_000.0, float))
    p.add_argument("--nu", type=float, default=env_default("NU", 0.3, float))
    p.add_argument("--intorder", type=int, default=env_default("INTORDER", 2, int))
    p.add_argument("--repeats", type=int, default=env_default("REPEATS", 3, int), help="Number of repetitions per timing (>=1).")
    p.add_argument(
        "--warmup",
        type=int,
        default=env_default("WARMUP", 1, int),
        help="Warmup iterations (excluded from timing statistics).",
    )
    p.add_argument(
        "--kernel-jit",
        action=argparse.BooleanOptionalAction,
        default=env_default("KERNEL_JIT", True, lambda v: str(v).lower() not in {"0", "false", "no"}),
        help="JIT the element kernel inside assemble_bilinear_form (default: enabled).",
    )
    p.add_argument(
        "--breakdown",
        action=argparse.BooleanOptionalAction,
        default=env_default("BREAKDOWN", True, lambda v: str(v).lower() not in {"0", "false", "no"}),
        help="Measure kernel/backend timing breakdown (default: enabled).",
    )
    p.add_argument(
        "--kernel-breakdown",
        action=argparse.BooleanOptionalAction,
        default=env_default("KERNEL_BREAKDOWN", False, lambda v: str(v).lower() in {"1", "true", "yes"}),
        help="Measure sym_grad vs B^T D B timing inside the element kernel (default: disabled).",
    )
    p.add_argument("--cg-tol", type=float, default=env_default("CG_TOL", 1e-8, float))
    p.add_argument("--cg-maxiter", type=int, default=env_default("CG_MAXITER", 5000, int))
    p.add_argument(
        "--spsolve-impl",
        choices=["scipy", "jax"],
        default=os.environ.get("SPSOLVE_IMPL", "jax"),
        help="spsolve backend for fluxfem: jax.experimental.sparse.spsolve (default, falls back to scipy) or scipy",
    )
    p.add_argument(
        "--skfem-on-gpu",
        action="store_true",
        help="Run scikit-fem comparison even when backend is GPU (default: skip on GPU).",
    )
    p.add_argument(
        "--gpu-spsolve",
        action="store_true",
        help="Enable spsolve on GPU (default is CG-only on GPU; spsolve may be slow or fail).",
    )
    p.add_argument(
        "--cg-impl",
        choices=["custom", "jax"],
        default=os.environ.get("CG_IMPL", "custom"),
        help="CG implementation: 'custom' (fluxfem) or 'jax' (jax.scipy.sparse.linalg.cg).",
    )
    p.add_argument(
        "--cg-precon",
        choices=["jacobi", "block_jacobi"],
        default=os.environ.get("CG_PRECON", "jacobi"),
        help="Preconditioner for CG.",
    )
    p.add_argument(
        "--petsc",
        action=argparse.BooleanOptionalAction,
        default=env_default("PETSCSOLVE", True, lambda v: str(v).lower() not in {"0", "false", "no"}),
        help="Enable PETSc solve benchmark when petsc4py is available.",
    )
    p.add_argument(
        "--petsc-ksp",
        type=str,
        default=os.environ.get("PETSCSOLVE_KSP", "preonly"),
        help="PETSc KSP type (default: preonly).",
    )
    p.add_argument(
        "--petsc-pc",
        type=str,
        default=os.environ.get("PETSCSOLVE_PC", "lu"),
        help="PETSc PC type (default: lu).",
    )
    p.add_argument(
        "--petsc-shell",
        action=argparse.BooleanOptionalAction,
        default=env_default("PETSCSHELL", True, lambda v: str(v).lower() not in {"0", "false", "no"}),
        help="Enable PETSc shell (matrix-free) solve benchmark when petsc4py is available.",
    )
    p.add_argument(
        "--petsc-shell-ksp",
        type=str,
        default=os.environ.get("PETSCSHELL_KSP", "cg"),
        help="PETSc shell KSP type (default: cg).",
    )
    p.add_argument(
        "--petsc-shell-pc",
        type=str,
        default=os.environ.get("PETSCSHELL_PC", "none"),
        help="PETSc shell PC type (default: none).",
    )
    p.add_argument(
        "--petsc-shell-precon",
        choices=["none", "diag0", "both"],
        default=os.environ.get("PETSCSHELL_PRECON", "both"),
        help="Preconditioner mode for PETSc shell: none, diag0, or both (default).",
    )
    p.add_argument(
        "--petsc-shell-rtol",
        type=float,
        default=env_default("PETSCSHELL_RTOL", 1e-8, float),
        help="PETSc shell relative tolerance (default: 1e-8).",
    )
    p.add_argument(
        "--petsc-shell-atol",
        type=float,
        default=env_default("PETSCSHELL_ATOL", 0.0, float),
        help="PETSc shell absolute tolerance (default: 0.0).",
    )
    p.add_argument(
        "--petsc-shell-maxiter",
        type=int,
        default=env_default("PETSCSHELL_MAXITER", 2000, int),
        help="PETSc shell maximum iterations (default: 2000).",
    )
    p.add_argument(
        "--cg-matvec",
        choices=["flux", "bcoo"],
        default=os.environ.get("CG_MATVEC", "flux"),
        help="Matvec backend for CG: 'flux' (FluxSparseMatrix) or 'bcoo' (jax.experimental.sparse.BCOO).",
    )
    p.add_argument(
        "--no-cg",
        action="store_true",
        help="Disable fluxfem CG solve benchmark (skip cg_solve timing + plotting + CSV column).",
    )
    p.add_argument(
        "--plot",
        type=str,
        default=os.environ.get("PLOT", "result/bench/bench_linearelastic_dof_sweep_petsc_shell/solve_benchmark.png"),
        help="Output PNG for timing plot.",
    )
    p.add_argument(
        "--json",
        type=str,
        default="result/bench/bench_linearelastic_dof_sweep_petsc_shell/results.json",
        help="Output JSON path for results",
    )
    p.add_argument("--backends", type=str, default="cpu", help="Comma-separated backends to run (cpu,gpu)")
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--compare", action="store_true", help="Generate CPU/GPU comparison plot from JSON results")
    p.add_argument(
        "--compare-out",
        type=str,
        default="result/bench/bench_linearelastic_dof_sweep_petsc_shell/compare_cpu_gpu.png",
        help="Output PNG path for CPU/GPU comparison plot",
    )
    return p.parse_args()


def assemble_fluxfem_case(n: int, args, dtype):
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    mesh, ny, nz = make_structured_mesh(n, args.ny_mult, args.nz_mult, args.lx, args.ly, args.lz)
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=dtype),
        conn=jnp.asarray(mesh.conn, dtype=mesh.conn.dtype),
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )
    space = make_hex_space(mesh, dim=3, intorder=args.intorder)
    bc = compute_dirichlet_dofs(mesh)
    pattern = None

    facets, tags = tag_axis_minmax_facets(mesh, axis=0, dirichlet_tag=1, neumann_tag=2)
    neumann_facets = facets[np.asarray(tags) == 2]
    traction_surf = (
        SurfaceMesh.from_hex_mesh(mesh, neumann_facets)
        if abs(args.traction) > 0 and neumann_facets.size > 0
        else None
    )

    D = isotropic_3d_D(args.E, args.nu)
    D = jnp.asarray(D)
    f_body = jnp.array([args.fx, args.fy, args.fz], dtype=dtype)

    form_const = lambda ctx, _p: linear_elasticity_form(ctx, D)
    kernel = make_element_bilinear_kernel(form_const, None, jit=args.kernel_jit)
    elem_data = space.build_form_contexts()

    assemble_K = jax.jit(
        lambda: space.assemble(
            form_const,
            None,
            kind="bilinear",
            kernel=kernel,
            pattern=pattern,
        )
    )
    assemble_F = jax.jit(
        lambda: space.assemble_linear_form(
            constant_body_force_vector_form, params=f_body, sparse=False
        )
    )

    kernel_stage, backend_stage, sym_stage, bdb_stage = prepare_kernel_breakdown(
        elem_data, kernel, D, pattern,
        breakdown=args.breakdown,
        kernel_breakdown=args.kernel_breakdown,
    )

    # Warmup: exclude compile/first-call overhead from timing stats.
    warmup_times = []
    kernel_warm = []
    backend_warm = []
    sym_warm = []
    bdb_warm = []
    if args.warmup:
        for _ in range(args.warmup):
            t0 = time.perf_counter()
            if args.breakdown:
                Ke_all = kernel_stage()
                jax.block_until_ready(Ke_all)
                t_kernel = time.perf_counter() - t0

                t1 = time.perf_counter()
                K_tmp = backend_stage(Ke_all)
                jax.block_until_ready(K_tmp.data)
                t_backend = time.perf_counter() - t1
            else:
                K_tmp = assemble_K()
                jax.block_until_ready(K_tmp.data)
                t_kernel = float("nan")
                t_backend = float("nan")

            if args.kernel_breakdown:
                t_sym0 = time.perf_counter()
                Bu, Bv = sym_stage()
                jax.block_until_ready(Bu)
                t_sym = time.perf_counter() - t_sym0

                t_bdb0 = time.perf_counter()
                bdb = bdb_stage(Bu, Bv)
                jax.block_until_ready(bdb)
                t_bdb = time.perf_counter() - t_bdb0
            else:
                t_sym = float("nan")
                t_bdb = float("nan")

            F_tmp = assemble_F()
            jax.block_until_ready(F_tmp)
            if traction_surf is not None:
                F_tmp = traction_surf.assemble_load(
                    load=np.array([args.traction, 0.0, 0.0], dtype=float),
                    dim=3,
                    n_total_nodes=mesh.n_nodes,
                    F0=F_tmp,
                )
            warmup_times.append(time.perf_counter() - t0)
            kernel_warm.append(t_kernel)
            backend_warm.append(t_backend)
            sym_warm.append(t_sym)
            bdb_warm.append(t_bdb)

    assembly_times = []
    kernel_times = []
    backend_times = []
    sym_times = []
    bdb_times = []
    last_K = None
    last_F = None

    for _ in range(args.repeats):
        t0 = time.perf_counter()
        if args.breakdown:
            Ke_all = kernel_stage()
            jax.block_until_ready(Ke_all)
            t_kernel = time.perf_counter() - t0

            t1 = time.perf_counter()
            K_tmp = backend_stage(Ke_all)
            jax.block_until_ready(K_tmp.data)
            t_backend = time.perf_counter() - t1
        else:
            K_tmp = assemble_K()
            jax.block_until_ready(K_tmp.data)
            t_kernel = float("nan")
            t_backend = float("nan")

        if args.kernel_breakdown:
            t_sym0 = time.perf_counter()
            Bu, Bv = sym_stage()
            jax.block_until_ready(Bu)
            t_sym = time.perf_counter() - t_sym0

            t_bdb0 = time.perf_counter()
            bdb = bdb_stage(Bu, Bv)
            jax.block_until_ready(bdb)
            t_bdb = time.perf_counter() - t_bdb0
        else:
            t_sym = float("nan")
            t_bdb = float("nan")

        F_tmp = assemble_F()
        jax.block_until_ready(F_tmp)

        if traction_surf is not None:
            F_tmp = traction_surf.assemble_load(
                load=np.array([args.traction, 0.0, 0.0], dtype=float),
                dim=3,
                n_total_nodes=mesh.n_nodes,
                F0=F_tmp,
            )

        assembly_times.append(time.perf_counter() - t0)
        kernel_times.append(t_kernel)
        backend_times.append(t_backend)
        sym_times.append(t_sym)
        bdb_times.append(t_bdb)
        last_K = K_tmp
        last_F = np.asarray(F_tmp, dtype=float)

    if last_K is None or last_F is None:
        raise RuntimeError("Internal error: no assembly samples collected.")

    condensed = bc.condense_system(last_K, last_F)
    K_ff = condensed.K
    F_free = condensed.F
    free = condensed.free_dofs

    backend = jax.default_backend()
    solve_sps_warm = np.full((args.warmup,), np.nan, dtype=float)
    residual_sps_warm = np.full((args.warmup,), np.nan, dtype=float)
    if backend == "gpu" and not args.gpu_spsolve:
        solve_sps_times = np.full((args.repeats,), np.nan, dtype=float)
        residual_sps = np.full((args.repeats,), np.nan, dtype=float)
    else:
        if args.warmup:
            solve_sps_warm, residual_sps_warm = time_spsolve_samples(
                last_K,
                last_F,
                bc,
                K_ff,
                F_free,
                free,
                args.warmup,
                backend,
                args.spsolve_impl,
            )
        solve_sps_times, residual_sps = time_spsolve_samples(
            last_K,
            last_F,
            bc,
            K_ff,
            F_free,
            free,
            args.repeats,
            backend,
            args.spsolve_impl,
        )

    solve_cg_warm = np.full((args.warmup,), np.nan, dtype=float)
    cg_iters_warm = np.full((args.warmup,), np.nan, dtype=float)
    residual_cg_warm = np.full((args.warmup,), np.nan, dtype=float)
    if args.no_cg:
        solve_cg_times = np.full((args.repeats,), np.nan, dtype=float)
        cg_iters = np.full((args.repeats,), np.nan, dtype=float)
    else:
        if args.warmup:
            solve_cg_warm, cg_iters_warm, residual_cg_warm = time_flux_cg_samples(
                K_ff, F_free, args.warmup, args.cg_tol, args.cg_maxiter, args.cg_impl, args.cg_matvec, args.cg_precon
            )
        solve_cg_times, cg_iters, residual_cg = time_flux_cg_samples(
            K_ff, F_free, args.repeats, args.cg_tol, args.cg_maxiter, args.cg_impl, args.cg_matvec, args.cg_precon
        )
    if args.no_cg:
        residual_cg = np.full((args.repeats,), np.nan, dtype=float)

    petsc_avail = petsc_is_available()
    use_petsc = args.petsc and petsc_avail and backend != "gpu"
    if args.petsc and not petsc_avail:
        print("[bench] petsc4py not available; skipping PETSc solve benchmark.")
    if args.petsc and backend == "gpu":
        print("[bench] PETSc solve benchmark skipped on GPU backend.")
    solve_petsc_warm = np.full((args.warmup,), np.nan, dtype=float)
    residual_petsc_warm = np.full((args.warmup,), np.nan, dtype=float)
    solve_petsc_times = np.full((args.repeats,), np.nan, dtype=float)
    residual_petsc = np.full((args.repeats,), np.nan, dtype=float)
    if use_petsc:
        if args.warmup:
            solve_petsc_warm, residual_petsc_warm = time_petsc_samples(
                K_ff,
                F_free,
                args.warmup,
                args.petsc_ksp,
                args.petsc_pc,
            )
        solve_petsc_times, residual_petsc = time_petsc_samples(
            K_ff,
            F_free,
            args.repeats,
            args.petsc_ksp,
            args.petsc_pc,
        )

    use_petsc_shell = args.petsc_shell and petsc_avail and backend != "gpu"
    if args.petsc_shell and not petsc_avail:
        print("[bench] petsc4py not available; skipping PETSc shell benchmark.")
    if args.petsc_shell and backend == "gpu":
        print("[bench] PETSc shell benchmark skipped on GPU backend.")
    shell_modes = []
    if args.petsc_shell_precon == "both":
        shell_modes = ["none", "diag0"]
    else:
        shell_modes = [args.petsc_shell_precon]

    shell_samples = {}
    for mode in shell_modes:
        shell_samples[mode] = {
            "solve_warm": np.full((args.warmup,), np.nan, dtype=float),
            "residual_warm": np.full((args.warmup,), np.nan, dtype=float),
            "iters_warm": np.full((args.warmup,), np.nan, dtype=float),
            "solve": np.full((args.repeats,), np.nan, dtype=float),
            "residual": np.full((args.repeats,), np.nan, dtype=float),
            "iters": np.full((args.repeats,), np.nan, dtype=float),
        }

    if use_petsc_shell:
        for mode in shell_modes:
            precon = None if mode == "none" else "diag0"
            if args.warmup:
                s_warm, r_warm, i_warm = time_petsc_shell_samples(
                    K_ff,
                    F_free,
                    args.warmup,
                    args.petsc_shell_ksp,
                    args.petsc_shell_pc,
                    precon,
                    args.petsc_shell_rtol,
                    args.petsc_shell_atol,
                    args.petsc_shell_maxiter,
                )
                shell_samples[mode]["solve_warm"] = s_warm
                shell_samples[mode]["residual_warm"] = r_warm
                shell_samples[mode]["iters_warm"] = i_warm
            s_times, r_times, i_times = time_petsc_shell_samples(
                K_ff,
                F_free,
                args.repeats,
                args.petsc_shell_ksp,
                args.petsc_shell_pc,
                precon,
                args.petsc_shell_rtol,
                args.petsc_shell_atol,
                args.petsc_shell_maxiter,
            )
            shell_samples[mode]["solve"] = s_times
            shell_samples[mode]["residual"] = r_times
            shell_samples[mode]["iters"] = i_times

    return {
        "n": n,
        "ny": ny,
        "nz": nz,
        "dofs_total": int(space.n_dofs),
        "free_dofs": int(K_ff.shape[0]),
        "assembly_warmup_samples": np.asarray(warmup_times, dtype=float),
        "assembly_samples": np.asarray(assembly_times, dtype=float),
        "kernel_warmup_samples": np.asarray(kernel_warm, dtype=float),
        "backend_warmup_samples": np.asarray(backend_warm, dtype=float),
        "kernel_samples": np.asarray(kernel_times, dtype=float),
        "backend_samples": np.asarray(backend_times, dtype=float),
        "kernel_sym_warmup_samples": np.asarray(sym_warm, dtype=float),
        "kernel_bdb_warmup_samples": np.asarray(bdb_warm, dtype=float),
        "kernel_sym_samples": np.asarray(sym_times, dtype=float),
        "kernel_bdb_samples": np.asarray(bdb_times, dtype=float),
        "solve_spsolve_warmup_samples": solve_sps_warm,
        "solve_spsolve_samples": solve_sps_times,
        "residual_spsolve_warmup_samples": residual_sps_warm,
        "residual_spsolve_samples": residual_sps,
        "solve_cg_warmup_samples": solve_cg_warm,
        "solve_cg_samples": solve_cg_times,
        "residual_cg_warmup_samples": residual_cg_warm,
        "residual_cg_samples": residual_cg,
        "cg_iters_warmup_samples": cg_iters_warm,
        "cg_iters_samples": cg_iters,
        "solve_petsc_warmup_samples": solve_petsc_warm,
        "solve_petsc_samples": solve_petsc_times,
        "residual_petsc_warmup_samples": residual_petsc_warm,
        "residual_petsc_samples": residual_petsc,
        "petsc_shell": shell_samples,
    }


def assemble_skfem_case(n: int, ny: int, nz: int, args):
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if importlib.util.find_spec("skfem") is None:
        return None

    from skfem import MeshHex, ElementHex1, ElementVectorH1, Basis, asm, LinearForm, condense, solve  # type: ignore
    from skfem.models.elasticity import linear_elasticity  # type: ignore
    from skfem.helpers import dot  # type: ignore

    xs = np.linspace(0.0, args.lx, n + 1)
    ys = np.linspace(0.0, args.ly, ny + 1)
    zs = np.linspace(0.0, args.lz, nz + 1)
    mesh = MeshHex().init_tensor(xs, ys, zs)
    basis = Basis(mesh, ElementVectorH1(ElementHex1()), intorder=args.intorder)

    lam, mu = lame_parameters(args.E, args.nu)
    f_body = np.array([args.fx, args.fy, args.fz], dtype=float)
    traction_vec = np.array([args.traction, 0.0, 0.0], dtype=float)

    @LinearForm
    def body_force(v, w):
        return dot(f_body, v)

    @LinearForm
    def traction_load(v, w):
        return dot(traction_vec, v)

    # Precompute boundary basis once (avoid mixing "setup" cost into every repetition).
    fbasis = None
    if abs(args.traction) > 0:
        facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], args.lx, atol=1e-8))
        fbasis = basis.boundary(facets=facets)

    # Warmup samples (excluded from timing stats).
    warmup_times = []
    if args.warmup:
        warm_timer = SectionTimer()
        for _ in range(args.warmup):
            with warm_timer.section("assemble_skfem_warm"):
                K = asm(linear_elasticity(Lambda=lam, Mu=mu), basis)
                F = asm(body_force, basis)
                if fbasis is not None:
                    F += asm(traction_load, fbasis)
            warmup_times.append(warm_timer.last("assemble_skfem_warm"))

    # Assembly samples.
    assembly_times = []
    last_K = None
    last_F = None

    timer = SectionTimer()
    for _ in range(args.repeats):
        with timer.section("assemble_skfem"):
            K = asm(linear_elasticity(Lambda=lam, Mu=mu), basis)
            F = asm(body_force, basis)
            if fbasis is not None:
                F += asm(traction_load, fbasis)
        assembly_times.append(timer.last("assemble_skfem"))
        last_K = K
        last_F = F

    if last_K is None or last_F is None:
        raise RuntimeError("Internal error: no skfem assembly samples collected.")

    dir_dofs = basis.get_dofs(lambda x: np.isclose(x[0], 0.0, atol=1e-8))
    Kc, Fc = condense(last_K, last_F, D=dir_dofs, expand=False)

    solve_warm = []
    residual_warm = []
    if args.warmup:
        solve_timer = SectionTimer()
        for _ in range(args.warmup):
            with solve_timer.section("solve_skfem_warm"):
                u = solve(Kc, Fc)
            solve_warm.append(solve_timer.last("solve_skfem_warm"))
            residual_warm.append(_residual_error(Kc, Fc, u))

    solve_times = []
    residual_errors = []
    solve_timer = SectionTimer()
    for _ in range(args.repeats):
        with solve_timer.section("solve_skfem"):
            u = solve(Kc, Fc)
        solve_times.append(solve_timer.last("solve_skfem"))
        residual_errors.append(_residual_error(Kc, Fc, u))

    return {
        "dofs_total": int(basis.N),
        "free_dofs": int(Kc.shape[0]),
        "assembly_warmup_samples": np.asarray(warmup_times, dtype=float),
        "assembly_samples": np.asarray(assembly_times, dtype=float),
        "solve_warmup_samples": np.asarray(solve_warm, dtype=float),
        "solve_samples": np.asarray(solve_times, dtype=float),
        "residual_solve_warmup_samples": np.asarray(residual_warm, dtype=float),
        "residual_solve_samples": np.asarray(residual_errors, dtype=float),
    }


if __name__ == "__main__":
    args = parse_args()
    def _make_backend_path(path: str, backend: str):
        root, ext = os.path.splitext(path)
        return f"{root}_{backend}{ext}"

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if "gpu" in backends:
        backends = ["gpu"] + [b for b in backends if b != "gpu"]
    if not args.single_backend and (len(backends) > 1 or (backends and backends[0] != jax.default_backend())):
        for backend in backends:
            env = os.environ.copy()
            env["JAX_PLATFORM_NAME"] = backend
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--single-backend",
                "--sizes",
                *[str(n) for n in args.sizes],
                "--ny-mult",
                str(args.ny_mult),
                "--nz-mult",
                str(args.nz_mult),
                "--lx",
                str(args.lx),
                "--ly",
                str(args.ly),
                "--lz",
                str(args.lz),
                "--fx",
                str(args.fx),
                "--fy",
                str(args.fy),
                "--fz",
                str(args.fz),
                "--traction",
                str(args.traction),
                "--E",
                str(args.E),
                "--nu",
                str(args.nu),
                "--intorder",
                str(args.intorder),
                "--repeats",
                str(args.repeats),
                "--warmup",
                str(args.warmup),
                "--breakdown" if args.breakdown else "--no-breakdown",
                "--kernel-breakdown" if args.kernel_breakdown else "--no-kernel-breakdown",
                "--cg-tol",
                str(args.cg_tol),
                "--cg-maxiter",
                str(args.cg_maxiter),
                "--cg-impl",
                str(args.cg_impl),
                "--cg-precon",
                str(args.cg_precon),
                "--cg-matvec",
                str(args.cg_matvec),
                "--spsolve-impl",
                str(args.spsolve_impl),
                "--petsc-ksp",
                str(args.petsc_ksp),
                "--petsc-pc",
                str(args.petsc_pc),
                "--petsc-shell-ksp",
                str(args.petsc_shell_ksp),
                "--petsc-shell-pc",
                str(args.petsc_shell_pc),
                "--petsc-shell-precon",
                str(args.petsc_shell_precon),
                "--petsc-shell-rtol",
                str(args.petsc_shell_rtol),
                "--petsc-shell-atol",
                str(args.petsc_shell_atol),
                "--petsc-shell-maxiter",
                str(args.petsc_shell_maxiter),
                "--plot",
                _make_backend_path(args.plot, backend),
                "--json",
                _make_backend_path(args.json, backend),
            ]
            if args.gpu_spsolve:
                cmd.append("--gpu-spsolve")
            if args.no_cg:
                cmd.append("--no-cg")
            if args.petsc:
                cmd.append("--petsc")
            else:
                cmd.append("--no-petsc")
            if args.petsc_shell:
                cmd.append("--petsc-shell")
            else:
                cmd.append("--no-petsc-shell")
            proc = subprocess.run(cmd, env=env, check=False)
            if proc.returncode != 0:
                if backend == "gpu":
                    print("[bench] GPU backend unavailable; skipping GPU run.")
                    continue
                raise SystemExit(proc.returncode)
        if args.compare:
            cpu_json = Path(_make_backend_path(args.json, "cpu"))
            gpu_json = Path(_make_backend_path(args.json, "gpu"))
            if cpu_json.exists() and gpu_json.exists():
                with cpu_json.open("r", encoding="utf-8") as f:
                    cpu = json.load(f)
                with gpu_json.open("r", encoding="utf-8") as f:
                    gpu = json.load(f)

                try:
                    from scripts.bench_plot_utils import plot_bench_linearelastic_compare
                except ModuleNotFoundError:
                    import sys as _sys
                    from pathlib import Path as _Path

                    _root = _Path(__file__).resolve().parents[1]
                    if str(_root) not in _sys.path:
                        _sys.path.insert(0, str(_root))
                    from scripts.bench_plot_utils import plot_bench_linearelastic_compare

                plot_bench_linearelastic_compare(cpu, gpu, args.compare_out)
        sys.exit(0)
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    flux_results = []
    skfem_results = []

    backend = jax.default_backend()
    for n in args.sizes:
        flux_res = assemble_fluxfem_case(n, args, dtype)
        flux_results.append(flux_res)
        print(f"\n--- n={n}, ny={flux_res['ny']}, nz={flux_res['nz']} ---")

        asm_stats = summarize(flux_res["assembly_samples"])
        asm_warm_stats = summarize(flux_res["assembly_warmup_samples"]) if args.warmup else None
        sps_stats = summarize(flux_res["solve_spsolve_samples"])
        residual_sps_stats = summarize(flux_res["residual_spsolve_samples"])

        msg = (
            f"fluxfem: dof={flux_res['free_dofs']}, "
            f"assembly mean={asm_stats['mean']:.3e}s [min={asm_stats['min']:.3e}, max={asm_stats['max']:.3e}], "
            f"spsolve mean={sps_stats['mean']:.3e}s [min={sps_stats['min']:.3e}, max={sps_stats['max']:.3e}]"
            f", residual_spsolve med={residual_sps_stats['median']:.3e}"
        )
        if asm_warm_stats is not None:
            msg += (
                f", assembly warm mean={asm_warm_stats['mean']:.3e}s"
                f" [min={asm_warm_stats['min']:.3e}, max={asm_warm_stats['max']:.3e}]"
            )
        if args.breakdown:
            kern_stats = summarize(flux_res["kernel_samples"])
            back_stats = summarize(flux_res["backend_samples"])
            msg += f", kernel mean={kern_stats['mean']:.3e}s, backend mean={back_stats['mean']:.3e}s"
        if args.kernel_breakdown:
            sym_stats = summarize(flux_res["kernel_sym_samples"])
            bdb_stats = summarize(flux_res["kernel_bdb_samples"])
            msg += f", sym_grad mean={sym_stats['mean']:.3e}s, bdb mean={bdb_stats['mean']:.3e}s"
        if not args.no_cg:
            cg_stats = summarize(flux_res["solve_cg_samples"])
            it_stats = summarize(flux_res["cg_iters_samples"])
            residual_cg_stats = summarize(flux_res["residual_cg_samples"])
            msg += (
                f", cg mean={cg_stats['mean']:.3e}s [min={cg_stats['min']:.3e}, max={cg_stats['max']:.3e}]"
                f" (iters median~{it_stats['median']:.1f}, residual_cg med={residual_cg_stats['median']:.3e})"
            )
        if args.petsc and np.any(np.isfinite(flux_res["solve_petsc_samples"])):
            petsc_stats = summarize(flux_res["solve_petsc_samples"])
            residual_petsc_stats = summarize(flux_res["residual_petsc_samples"])
            msg += (
                f", petsc mean={petsc_stats['mean']:.3e}s [min={petsc_stats['min']:.3e}, max={petsc_stats['max']:.3e}]"
                f", residual_petsc med={residual_petsc_stats['median']:.3e}"
            )
        if args.petsc_shell and flux_res.get("petsc_shell"):
            for mode, samples in flux_res["petsc_shell"].items():
                if np.any(np.isfinite(samples["solve"])):
                    shell_stats = summarize(samples["solve"])
                    res_shell_stats = summarize(samples["residual"])
                    it_shell_stats = summarize(samples["iters"])
                    msg += (
                        f", petsc_shell[{mode}] mean={shell_stats['mean']:.3e}s"
                        f" [min={shell_stats['min']:.3e}, max={shell_stats['max']:.3e}]"
                        f" (iters median~{it_shell_stats['median']:.1f},"
                        f" residual med={res_shell_stats['median']:.3e})"
                    )
        print(msg)

        if backend == "gpu" and not args.skfem_on_gpu:
            print("scikit-fem comparison skipped on GPU backend.")
            continue

        sk_res = assemble_skfem_case(n, flux_res["ny"], flux_res["nz"], args)
        if sk_res is None:
            print("scikit-fem not installed; skipping skfem comparison.")
        else:
            skfem_results.append(sk_res)
            sk_asm = summarize(sk_res["assembly_samples"])
            sk_sol = summarize(sk_res["solve_samples"])
            sk_residual = summarize(sk_res["residual_solve_samples"])
            print(
                f"scikit-fem: dof={sk_res['free_dofs']}, "
                f"assembly mean={sk_asm['mean']:.3e}s [min={sk_asm['min']:.3e}, max={sk_asm['max']:.3e}], "
                f"solve mean={sk_sol['mean']:.3e}s [min={sk_sol['min']:.3e}, max={sk_sol['max']:.3e}], "
                f"residual_spsolve med={sk_residual['median']:.3e}"
            )

    if not flux_results:
        raise SystemExit("No fluxfem results collected.")

    # CSV-style table: report mean and range (min/max). (No longer only mean.)
    header = "\nDOF (free), flux_asm_mean[s], flux_asm_min[s], flux_asm_max[s]"
    if args.warmup:
        header += ", flux_asm_warm_mean[s], flux_asm_warm_min[s], flux_asm_warm_max[s]"
    if args.breakdown:
        header += ", flux_kernel_mean[s], flux_backend_mean[s]"
    if args.kernel_breakdown:
        header += ", flux_sym_grad_mean[s], flux_bdb_mean[s]"
    header += ", flux_sps_mean[s], flux_sps_min[s], flux_sps_max[s], flux_res_sps_median"
    if not args.no_cg:
        header += ", flux_cg_mean[s], flux_cg_min[s], flux_cg_max[s], flux_cg_iters_median, flux_res_cg_median"
    if args.petsc:
        header += ", flux_petsc_mean[s], flux_petsc_min[s], flux_petsc_max[s], flux_res_petsc_median"
    if args.petsc_shell:
        if args.petsc_shell_precon in ("both", "none"):
            header += ", flux_petsc_shell_none_mean[s], flux_petsc_shell_none_min[s], flux_petsc_shell_none_max[s], flux_petsc_shell_none_iters_median, flux_res_petsc_shell_none_median"
        if args.petsc_shell_precon in ("both", "diag0"):
            header += ", flux_petsc_shell_diag0_mean[s], flux_petsc_shell_diag0_min[s], flux_petsc_shell_diag0_max[s], flux_petsc_shell_diag0_iters_median, flux_res_petsc_shell_diag0_median"
    if skfem_results:
        header += ", sk_asm_mean[s], sk_asm_min[s], sk_asm_max[s], sk_solve_mean[s], sk_solve_min[s], sk_solve_max[s], sk_res_sps_median"
    print(header)

    sk_map = {r["free_dofs"]: r for r in skfem_results} if skfem_results else {}

    for res in flux_results:
        asm = summarize(res["assembly_samples"])
        sps = summarize(res["solve_spsolve_samples"])
        res_sps = summarize(res["residual_spsolve_samples"])
        row = f"{res['free_dofs']}, {asm['mean']:.6e}, {asm['min']:.6e}, {asm['max']:.6e}"
        if args.warmup:
            asm_warm = summarize(res["assembly_warmup_samples"])
            row += f", {asm_warm['mean']:.6e}, {asm_warm['min']:.6e}, {asm_warm['max']:.6e}"
        if args.breakdown:
            kern = summarize(res["kernel_samples"])
            back = summarize(res["backend_samples"])
            row += f", {kern['mean']:.6e}, {back['mean']:.6e}"
        if args.kernel_breakdown:
            sym = summarize(res["kernel_sym_samples"])
            bdb = summarize(res["kernel_bdb_samples"])
            row += f", {sym['mean']:.6e}, {bdb['mean']:.6e}"
        row += f", {sps['mean']:.6e}, {sps['min']:.6e}, {sps['max']:.6e}, {res_sps['median']:.6e}"
        if not args.no_cg:
            cg = summarize(res["solve_cg_samples"])
            it = summarize(res["cg_iters_samples"])
            res_cg = summarize(res["residual_cg_samples"])
            row += f", {cg['mean']:.6e}, {cg['min']:.6e}, {cg['max']:.6e}, {it['median']:.1f}, {res_cg['median']:.6e}"
        if args.petsc:
            petsc = summarize(res["solve_petsc_samples"])
            res_petsc = summarize(res["residual_petsc_samples"])
            row += f", {petsc['mean']:.6e}, {petsc['min']:.6e}, {petsc['max']:.6e}, {res_petsc['median']:.6e}"
        if args.petsc_shell and res.get("petsc_shell"):
            if args.petsc_shell_precon in ("both", "none"):
                shell = res["petsc_shell"].get("none")
                if shell is not None:
                    shell_stats = summarize(shell["solve"])
                    shell_iters = summarize(shell["iters"])
                    shell_res = summarize(shell["residual"])
                    row += (
                        f", {shell_stats['mean']:.6e}, {shell_stats['min']:.6e}, {shell_stats['max']:.6e},"
                        f" {shell_iters['median']:.1f}, {shell_res['median']:.6e}"
                    )
            if args.petsc_shell_precon in ("both", "diag0"):
                shell = res["petsc_shell"].get("diag0")
                if shell is not None:
                    shell_stats = summarize(shell["solve"])
                    shell_iters = summarize(shell["iters"])
                    shell_res = summarize(shell["residual"])
                    row += (
                        f", {shell_stats['mean']:.6e}, {shell_stats['min']:.6e}, {shell_stats['max']:.6e},"
                        f" {shell_iters['median']:.1f}, {shell_res['median']:.6e}"
                    )

        sk = sk_map.get(res["free_dofs"])
        if sk is not None:
            sk_asm = summarize(sk["assembly_samples"])
            sk_sol = summarize(sk["solve_samples"])
            sk_res = summarize(sk["residual_solve_samples"])
            row += (
                f", {sk_asm['mean']:.6e}, {sk_asm['min']:.6e}, {sk_asm['max']:.6e}, "
                f"{sk_sol['mean']:.6e}, {sk_sol['min']:.6e}, {sk_sol['max']:.6e}, {sk_res['median']:.6e}"
            )

        print(row)

    # Plot (same structure as before): left = assembly, right = solve.
    out_path = Path(args.plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Sort by DOF for nicer plots
    flux_sorted = sorted(flux_results, key=lambda r: r["free_dofs"])
    x_flux = np.array([r["free_dofs"] for r in flux_sorted], dtype=float)

    # --- Assembly plot: mean line + min-max band ---
    asm_medians = np.array([np.median(r["assembly_samples"]) for r in flux_sorted])
    asm_mins = np.array([np.min(r["assembly_samples"]) for r in flux_sorted])
    asm_maxs = np.array([np.max(r["assembly_samples"]) for r in flux_sorted])

    ax0.plot(x_flux, asm_medians, "o-", label="fluxfem assembly (median)")
    ax0.fill_between(x_flux, asm_mins, asm_maxs, alpha=0.2)

    if skfem_results:
        sk_sorted = sorted(skfem_results, key=lambda r: r["free_dofs"])
        x_sk = np.array([r["free_dofs"] for r in sk_sorted], dtype=float)
        sk_asm_medians = np.array([np.median(r["assembly_samples"]) for r in sk_sorted])
        sk_asm_mins = np.array([np.min(r["assembly_samples"]) for r in sk_sorted])
        sk_asm_maxs = np.array([np.max(r["assembly_samples"]) for r in sk_sorted])
        ax0.plot(x_sk, sk_asm_medians, "d-.", label="scikit-fem assembly (median)")
        ax0.fill_between(x_sk, sk_asm_mins, sk_asm_maxs, alpha=0.2)

    ax0.set_xlabel("free DOFs")
    ax0.set_ylabel("Assembly time [s]")
    ax0.grid(True, alpha=0.3)
    ax0.legend()
    ax0.set_title("Assembly time vs DOFs (mean with min–max band)")

    # --- Solve plot: mean line + min-max band ---
    sps_median = np.array([np.median(r["solve_spsolve_samples"]) for r in flux_sorted])
    sps_mins = np.array([np.min(r["solve_spsolve_samples"]) for r in flux_sorted])
    sps_maxs = np.array([np.max(r["solve_spsolve_samples"]) for r in flux_sorted])

    ax1.plot(x_flux, sps_median, "o-", label="fluxfem spsolve (median)")
    ax1.fill_between(x_flux, sps_mins, sps_maxs, alpha=0.2)

    if not args.no_cg:
        cg_median = np.array([np.nanmedian(r["solve_cg_samples"]) for r in flux_sorted])
        cg_mins = np.array([np.nanmin(r["solve_cg_samples"]) for r in flux_sorted])
        cg_maxs = np.array([np.nanmax(r["solve_cg_samples"]) for r in flux_sorted])
        ax1.plot(x_flux, cg_median, "s--", label="fluxfem cg (median)")
        ax1.fill_between(x_flux, cg_mins, cg_maxs, alpha=0.2)

    if args.petsc and any(np.any(np.isfinite(r["solve_petsc_samples"])) for r in flux_sorted):
        petsc_median = np.array([np.nanmedian(r["solve_petsc_samples"]) for r in flux_sorted])
        petsc_mins = np.array([np.nanmin(r["solve_petsc_samples"]) for r in flux_sorted])
        petsc_maxs = np.array([np.nanmax(r["solve_petsc_samples"]) for r in flux_sorted])
        ax1.plot(x_flux, petsc_median, "^-.", label="fluxfem petsc (median)")
        ax1.fill_between(x_flux, petsc_mins, petsc_maxs, alpha=0.2)
    if args.petsc_shell and any(r.get("petsc_shell") for r in flux_sorted):
        if args.petsc_shell_precon in ("both", "none"):
            shell_median = np.array([np.nanmedian(r["petsc_shell"]["none"]["solve"]) for r in flux_sorted])
            shell_mins = np.array([np.nanmin(r["petsc_shell"]["none"]["solve"]) for r in flux_sorted])
            shell_maxs = np.array([np.nanmax(r["petsc_shell"]["none"]["solve"]) for r in flux_sorted])
            ax1.plot(x_flux, shell_median, "x--", label="petsc shell (none)")
            ax1.fill_between(x_flux, shell_mins, shell_maxs, alpha=0.2)
        if args.petsc_shell_precon in ("both", "diag0"):
            shell_median = np.array([np.nanmedian(r["petsc_shell"]["diag0"]["solve"]) for r in flux_sorted])
            shell_mins = np.array([np.nanmin(r["petsc_shell"]["diag0"]["solve"]) for r in flux_sorted])
            shell_maxs = np.array([np.nanmax(r["petsc_shell"]["diag0"]["solve"]) for r in flux_sorted])
            ax1.plot(x_flux, shell_median, "v--", label="petsc shell (diag0)")
            ax1.fill_between(x_flux, shell_mins, shell_maxs, alpha=0.2)

    if skfem_results:
        sk_sorted = sorted(skfem_results, key=lambda r: r["free_dofs"])
        x_sk = np.array([r["free_dofs"] for r in sk_sorted], dtype=float)
        sk_medians = np.array([np.median(r["solve_samples"]) for r in sk_sorted])
        sk_mins = np.array([np.min(r["solve_samples"]) for r in sk_sorted])
        sk_maxs = np.array([np.max(r["solve_samples"]) for r in sk_sorted])
        ax1.plot(x_sk, sk_medians, "d-.", label="scikit-fem solve (median)")
        ax1.fill_between(x_sk, sk_mins, sk_maxs, alpha=0.2)

    ax1.set_xlabel("free DOFs")
    ax1.set_ylabel("Solve time [s]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title("Solve time vs DOFs (mean with min–max band)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"\nSaved plot to {out_path}")

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    def _to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jsonable(v) for v in obj]
        return obj

    payload = {
        "backend": jax.default_backend(),
        "sizes": args.sizes,
        "ny_mult": args.ny_mult,
        "nz_mult": args.nz_mult,
        "lx": args.lx,
        "ly": args.ly,
        "lz": args.lz,
        "fx": args.fx,
        "fy": args.fy,
        "fz": args.fz,
        "traction": args.traction,
        "E": args.E,
        "nu": args.nu,
        "intorder": args.intorder,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "breakdown": args.breakdown,
        "kernel_breakdown": args.kernel_breakdown,
        "cg_tol": args.cg_tol,
        "cg_maxiter": args.cg_maxiter,
        "cg_impl": args.cg_impl,
        "cg_precon": args.cg_precon,
        "cg_matvec": args.cg_matvec,
        "no_cg": args.no_cg,
        "spsolve_impl": args.spsolve_impl,
        "petsc": args.petsc,
        "petsc_ksp": args.petsc_ksp,
        "petsc_pc": args.petsc_pc,
        "petsc_shell": args.petsc_shell,
        "petsc_shell_ksp": args.petsc_shell_ksp,
        "petsc_shell_pc": args.petsc_shell_pc,
        "petsc_shell_precon": args.petsc_shell_precon,
        "petsc_shell_rtol": args.petsc_shell_rtol,
        "petsc_shell_atol": args.petsc_shell_atol,
        "petsc_shell_maxiter": args.petsc_shell_maxiter,
        "flux": _to_jsonable(flux_results),
        "skfem": _to_jsonable(skfem_results),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {out_json}")

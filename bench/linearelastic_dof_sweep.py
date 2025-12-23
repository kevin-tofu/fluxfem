#!/usr/bin/env python3
"""
Benchmark linear-elastic assembly/solve time vs DOF.

Compares:
- fluxfem assembly time (includes first-call JIT compile in the samples)
- fluxfem solve time with SciPy `spsolve`
- (optional) fluxfem solve time with in-house `cg_solve`
- scikit-fem assembly + `solve` (if installed)

USAGE
-----
Basic (default: includes CG benchmark):
  PYTHONPATH=src python bench/linearelastic_dof_sweep.py

Disable CG benchmark (skip cg_solve timing + plotting + CSV column):
  PYTHONPATH=src python bench/linearelastic_dof_sweep.py --no-cg

Custom sweep and settings:
  PYTHONPATH=src python bench/linearelastic_dof_sweep.py --sizes 8 12 16 20 --repeats 5 --intorder 2

Environment variable equivalents (optional):
  SIZES="8,12,16,20" NY_MULT=1.0 NZ_MULT=1.0 REPEATS=5 INTORDER=2 PLOT="solve_benchmark.png"

Notes
-----
- This script enables JAX x64 by default (jax_enable_x64=True).
- scikit-fem comparison is skipped if scikit-fem is not installed.
- IMPORTANT: For fluxfem assembly, the first measured sample includes JIT compile + execute.
  We report the distribution over `--repeats` samples (min/mean/max), not just the mean.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fluxfem.tools.timer import SectionTimer
from fluxfem import (
    FluxSparseMatrix,
    StructuredHexBox,
    SurfaceMesh,
    cg_solve,
    cg_solve_jax,
    constant_body_force_vector_form,
    lame_parameters,
    isotropic_3d_D,
    linear_elasticity_form,
    make_hex_space,
    make_sparsity_pattern,
    LinearSolver,
    tag_axis_minmax_facets,
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
        default=os.environ.get("PLOT", "result/bench/bench_linearelastic_dof_sweep/solve_benchmark.png"),
        help="Output PNG for timing plot.",
    )
    p.add_argument(
        "--json",
        type=str,
        default="result/bench/bench_linearelastic_dof_sweep/results.json",
        help="Output JSON path for results",
    )
    p.add_argument("--backends", type=str, default="cpu", help="Comma-separated backends to run (cpu,gpu)")
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--compare", action="store_true", help="Generate CPU/GPU comparison plot from JSON results")
    p.add_argument(
        "--compare-out",
        type=str,
        default="result/bench/bench_linearelastic_dof_sweep/compare_cpu_gpu.png",
        help="Output PNG path for CPU/GPU comparison plot",
    )
    return p.parse_args()


def make_structured_mesh(n: int, ny_mult: float, nz_mult: float, lx: float, ly: float, lz: float):
    ny = max(1, int(round(n * ny_mult)))
    nz = max(1, int(round(n * nz_mult)))
    mesh = StructuredHexBox(nx=n, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
    return mesh, ny, nz


def compute_dirichlet_dofs(mesh) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
        dof_per_node=3,
    )
    dir_vals = np.zeros(len(dir_dofs), dtype=float)
    return dir_dofs, dir_vals


def condense_flux_dirichlet(K, F, dir_dofs, dir_vals):
    K_csr = K.to_csr()
    dir_arr = np.asarray(dir_dofs, dtype=int)
    dir_vals_arr = np.asarray(dir_vals, dtype=float)
    mask = np.ones(K_csr.shape[0], dtype=bool)
    mask[dir_arr] = False
    free = np.nonzero(mask)[0]
    if dir_arr.size > 0 and np.any(dir_vals_arr):
        F_free = np.asarray(F, dtype=float)[free] - K_csr[free][:, dir_arr] @ dir_vals_arr
    else:
        F_free = np.asarray(F, dtype=float)[free]
    K_ff = K_csr[free][:, free]
    return K_ff, F_free, free


def make_block_jacobi_preconditioner(K_cg):
    """Build 3x3 block Jacobi preconditioner callable for FluxSparseMatrix or BCOO."""
    n = K_cg.n_dofs if hasattr(K_cg, "n_dofs") else int(K_cg.shape[0])
    if n % 3 != 0:
        raise ValueError("block_jacobi assumes 3 DOFs per node.")
    try:
        rows = np.asarray(K_cg.pattern.rows) if hasattr(K_cg, "pattern") else np.asarray(K_cg.indices[:, 0])
        cols = np.asarray(K_cg.pattern.cols) if hasattr(K_cg, "pattern") else np.asarray(K_cg.indices[:, 1])
        data = np.asarray(K_cg.data)
    except Exception as exc:
        raise ValueError("Unsupported matrix type for block_jacobi preconditioner") from exc

    block_rows = rows // 3
    block_cols = cols // 3
    lr = rows % 3
    lc = cols % 3
    mask = block_rows == block_cols
    block_rows = block_rows[mask]
    lr = lr[mask]
    lc = lc[mask]
    data = data[mask]
    n_block = n // 3
    blocks = np.zeros((n_block, 3, 3), dtype=data.dtype)
    np.add.at(blocks, (block_rows, lr, lc), data)
    blocks = blocks + 1e-12 * np.eye(3)[None, :, :]
    inv_blocks = jnp.asarray(np.linalg.inv(blocks))

    def precon(r):
        rb = r.reshape((n_block, 3))
        zb = jnp.einsum("bij,bj->bi", inv_blocks, rb)
        return zb.reshape((-1,))

    return precon


def _residual_error(K_ff, F_free, u) -> float:
    rhs = np.asarray(F_free, dtype=float)
    res = K_ff @ np.asarray(u, dtype=float) - rhs
    denom = np.linalg.norm(rhs)
    return float(np.linalg.norm(res) / (denom if denom > 0 else 1.0))


def time_flux_cg_samples(
    K_ff,
    F_free,
    repeats: int,
    tol: float,
    maxiter: int,
    cg_impl: str,
    cg_matvec: str,
    cg_precon: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      times: (repeats,) seconds
      iters: (repeats,) iteration counts (ints as float array)
    Note:
      First call includes any JAX first-call overhead (compile) for cg_solve path.
    """
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")

    coo = K_ff.tocoo()
    if cg_matvec == "flux":
        K_cg = FluxSparseMatrix.from_bilinear(
            (
                jnp.asarray(coo.row, dtype=jnp.int32),
                jnp.asarray(coo.col, dtype=jnp.int32),
                jnp.asarray(coo.data),
                K_ff.shape[0],
            )
        )
    else:
        try:
            from jax.experimental import sparse as jsparse  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("jax.experimental.sparse is required for --cg-matvec bcoo") from exc
        idx = jnp.stack(
            [
                jnp.asarray(coo.row, dtype=jnp.int32),
                jnp.asarray(coo.col, dtype=jnp.int32),
            ],
            axis=1,
        )
        K_cg = jsparse.BCOO((jnp.asarray(coo.data), idx), shape=(K_ff.shape[0], K_ff.shape[0]))
    b = jnp.asarray(F_free)

    times = []
    iters = []
    residual_errors = []

    cg_fn = cg_solve if cg_impl == "custom" else cg_solve_jax
    timer = SectionTimer()

    if cg_precon == "block_jacobi":
        precon = make_block_jacobi_preconditioner(K_cg)
    else:
        precon = "jacobi"

    for _ in range(repeats):
        with timer.section("solve_cg"):
            # Use Jacobi preconditioning to reduce iterations (especially on CPU).
            u, info = cg_fn(K_cg, b, tol=tol, maxiter=maxiter, preconditioner=precon)
            jax.block_until_ready(u)
        times.append(timer.last("solve_cg"))
        iters.append(int(info.get("iters", 0)))
        residual_errors.append(_residual_error(K_ff, F_free, u))

    return (
        np.asarray(times, dtype=float),
        np.asarray(iters, dtype=float),
        np.asarray(residual_errors, dtype=float),
    )


def time_spsolve_samples(
    K_full,
    F_full,
    dirichlet,
    K_ff,
    F_free,
    free,
    repeats: int,
    backend: str,
    spsolve_impl: str,
) -> tuple[np.ndarray, np.ndarray]:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    times = []
    residual_errors = []
    if backend == "gpu":
        solver = LinearSolver(method="spdirect_solve_gpu")
    elif spsolve_impl == "jax":
        solver = LinearSolver(method="spsolve_jax")
    else:
        solver = LinearSolver(method="spsolve")
    timer = SectionTimer()
    for _ in range(repeats):
        try:
            with timer.section("solve_spsolve"):
                u = solver.solve(K_full, F_full, dirichlet=dirichlet, n_total=K_full.n_dofs)[0]
        except Exception as exc:
            if backend == "gpu":
                print(f"[bench] GPU spsolve failed ({exc}); recording NaN and skipping remaining repeats.")
                times.extend([float("nan")] * (repeats - len(times)))
                residual_errors.extend([float("nan")] * (repeats - len(residual_errors)))
                break
            raise
        times.append(timer.last("solve_spsolve"))
        residual_errors.append(_residual_error(K_ff, F_free, u[free]))
    return np.asarray(times, dtype=float), np.asarray(residual_errors, dtype=float)


def summarize(samples: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(samples)),
        "mean": float(np.mean(samples)),
        "max": float(np.max(samples)),
        "median": float(np.median(samples)),
    }


def assemble_fluxfem_case(n: int, args, dtype):
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    mesh, ny, nz = make_structured_mesh(n, args.ny_mult, args.nz_mult, args.lx, args.ly, args.lz)
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=dtype),
        conn=jnp.asarray(mesh.conn, dtype=mesh.conn.dtype),
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )
    space = make_hex_space(mesh, dim=3, intorder=args.intorder)
    dir_dofs, dir_vals = compute_dirichlet_dofs(mesh)
    pattern = make_sparsity_pattern(space, with_idx=False)

    facets, tags = tag_axis_minmax_facets(mesh, axis=0, dirichlet_tag=1, neumann_tag=2)
    neumann_facets = facets[np.asarray(tags) == 2]
    traction_surf = (
        SurfaceMesh.from_hex_mesh(mesh, neumann_facets)
        if abs(args.traction) > 0 and neumann_facets.size > 0
        else None
    )

    D = isotropic_3d_D(args.E, args.nu)
    f_body = jnp.array([args.fx, args.fy, args.fz], dtype=dtype)

    assemble_K = jax.jit(
        lambda: space.assemble_bilinear_form(
            linear_elasticity_form,
            params=D,
            pattern=pattern,
        )
    )
    assemble_F = jax.jit(
        lambda: space.assemble_linear_form(
            constant_body_force_vector_form, params=f_body, sparse=False
        )
    )

    # IMPORTANT: do NOT warm up here.
    # We include first-call JIT compile cost in the samples.

    assembly_times = []
    last_K = None
    last_F = None

    timer = SectionTimer()
    for _ in range(args.repeats):
        with timer.section("assemble_flux"):
            K_tmp = assemble_K()
            jax.block_until_ready(K_tmp.data)

            F_tmp = assemble_F()
            jax.block_until_ready(F_tmp)

        if traction_surf is not None:
            F_tmp = traction_surf.assemble_load(
                load=np.array([args.traction, 0.0, 0.0], dtype=float),
                dim=3,
                n_total_nodes=mesh.n_nodes,
                F0=F_tmp,
            )

        assembly_times.append(timer.last("assemble_flux"))
        last_K = K_tmp
        last_F = np.asarray(F_tmp, dtype=float)

    if last_K is None or last_F is None:
        raise RuntimeError("Internal error: no assembly samples collected.")

    K_ff, F_free, free = condense_flux_dirichlet(last_K, last_F, dir_dofs, dir_vals)

    backend = jax.default_backend()
    if backend == "gpu" and not args.gpu_spsolve:
        solve_sps_times = np.full((args.repeats,), np.nan, dtype=float)
        residual_sps = np.full((args.repeats,), np.nan, dtype=float)
    else:
        solve_sps_times, residual_sps = time_spsolve_samples(
            last_K,
            last_F,
            (dir_dofs, dir_vals),
            K_ff,
            F_free,
            free,
            args.repeats,
            backend,
            args.spsolve_impl,
        )

    if args.no_cg:
        solve_cg_times = np.full((args.repeats,), np.nan, dtype=float)
        cg_iters = np.full((args.repeats,), np.nan, dtype=float)
    else:
        solve_cg_times, cg_iters, residual_cg = time_flux_cg_samples(
            K_ff, F_free, args.repeats, args.cg_tol, args.cg_maxiter, args.cg_impl, args.cg_matvec, args.cg_precon
        )
    if args.no_cg:
        residual_cg = np.full((args.repeats,), np.nan, dtype=float)

    return {
        "n": n,
        "ny": ny,
        "nz": nz,
        "dofs_total": int(space.n_dofs),
        "free_dofs": int(K_ff.shape[0]),
        "assembly_samples": np.asarray(assembly_times, dtype=float),
        "solve_spsolve_samples": solve_sps_times,
        "residual_spsolve_samples": residual_sps,
        "solve_cg_samples": solve_cg_times,
        "residual_cg_samples": residual_cg,
        "cg_iters_samples": cg_iters,
    }


def assemble_skfem_case(n: int, ny: int, nz: int, args):
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
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

    # Assembly samples (includes any first-call overhead on the first repetition).
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
        "assembly_samples": np.asarray(assembly_times, dtype=float),
        "solve_samples": np.asarray(solve_times, dtype=float),
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
                "--plot",
                _make_backend_path(args.plot, backend),
                "--json",
                _make_backend_path(args.json, backend),
            ]
            if args.gpu_spsolve:
                cmd.append("--gpu-spsolve")
            if args.no_cg:
                cmd.append("--no-cg")
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
        sps_stats = summarize(flux_res["solve_spsolve_samples"])
        residual_sps_stats = summarize(flux_res["residual_spsolve_samples"])

        msg = (
            f"fluxfem: dof={flux_res['free_dofs']}, "
            f"assembly mean={asm_stats['mean']:.3e}s [min={asm_stats['min']:.3e}, max={asm_stats['max']:.3e}], "
            f"spsolve mean={sps_stats['mean']:.3e}s [min={sps_stats['min']:.3e}, max={sps_stats['max']:.3e}]"
            f", residual_spsolve med={residual_sps_stats['median']:.3e}"
        )
        if not args.no_cg:
            cg_stats = summarize(flux_res["solve_cg_samples"])
            it_stats = summarize(flux_res["cg_iters_samples"])
            residual_cg_stats = summarize(flux_res["residual_cg_samples"])
            msg += (
                f", cg mean={cg_stats['mean']:.3e}s [min={cg_stats['min']:.3e}, max={cg_stats['max']:.3e}]"
                f" (iters median~{it_stats['median']:.1f}, residual_cg med={residual_cg_stats['median']:.3e})"
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
    header = "\nDOF (free), flux_asm_mean[s], flux_asm_min[s], flux_asm_max[s], flux_sps_mean[s], flux_sps_min[s], flux_sps_max[s], flux_res_sps_median"
    if not args.no_cg:
        header += ", flux_cg_mean[s], flux_cg_min[s], flux_cg_max[s], flux_cg_iters_median, flux_res_cg_median"
    if skfem_results:
        header += ", sk_asm_mean[s], sk_asm_min[s], sk_asm_max[s], sk_solve_mean[s], sk_solve_min[s], sk_solve_max[s], sk_res_sps_median"
    print(header)

    sk_map = {r["free_dofs"]: r for r in skfem_results} if skfem_results else {}

    for res in flux_results:
        asm = summarize(res["assembly_samples"])
        sps = summarize(res["solve_spsolve_samples"])
        res_sps = summarize(res["residual_spsolve_samples"])
        row = (
            f"{res['free_dofs']}, "
            f"{asm['mean']:.6e}, {asm['min']:.6e}, {asm['max']:.6e}, "
            f"{sps['mean']:.6e}, {sps['min']:.6e}, {sps['max']:.6e}, {res_sps['median']:.6e}"
        )
        if not args.no_cg:
            cg = summarize(res["solve_cg_samples"])
            it = summarize(res["cg_iters_samples"])
            res_cg = summarize(res["residual_cg_samples"])
            row += f", {cg['mean']:.6e}, {cg['min']:.6e}, {cg['max']:.6e}, {it['median']:.1f}, {res_cg['median']:.6e}"

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
        "cg_tol": args.cg_tol,
        "cg_maxiter": args.cg_maxiter,
        "cg_impl": args.cg_impl,
        "cg_precon": args.cg_precon,
        "cg_matvec": args.cg_matvec,
        "no_cg": args.no_cg,
        "spsolve_impl": args.spsolve_impl,
        "flux": _to_jsonable(flux_results),
        "skfem": _to_jsonable(skfem_results),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {out_json}")

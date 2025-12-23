#!/usr/bin/env python3
"""
MMS heat equation benchmark with FluxFEM only (CPU/GPU compare).

Solves: u_t - kappa * Laplacian(u) = 0 on a unit cube with Dirichlet BCs.
Exact solution: sin(pi x) sin(pi y) sin(pi z) * exp(-lambda t),
lambda = 3 * pi^2 * kappa.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff
from fluxfem.tools.timer import SectionTimer


def parse_args():
    p = argparse.ArgumentParser(description="MMS heat equation benchmark (FluxFEM only).")
    p.add_argument("--n", type=int, default=None, help="Uniform mesh size (sets nx=ny=nz).")
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--nz", type=int, default=16)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--nsteps", type=int, default=20)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument(
        "--mode",
        choices=["single", "multi", "phase", "phase_sum"],
        default="single",
        help="Exact solution mode: single, multi, phase, or phase_sum (sum of phase-shifted modes).",
    )
    p.add_argument("--phase", type=float, default=0.0, help="Phase shift (radians) for phase mode.")
    p.add_argument(
        "--phase-shifts",
        type=str,
        default="0.0,0.7,1.4",
        help="Comma-separated phase shifts (radians) for phase_sum.",
    )
    p.add_argument(
        "--phase-weights",
        type=str,
        default="1.0,0.6,0.3",
        help="Comma-separated weights for phase_sum (same length as phase-shifts).",
    )
    p.add_argument("--save-vtu", action="store_true", help="Write VTU per step.")
    p.add_argument("--save-every", type=int, default=1, help="VTU output cadence in steps.")
    p.add_argument(
        "--vtu-dir",
        type=str,
        default="result/tutorials/heat_mms_timeseries/vtu",
        help="Directory for VTU output files.",
    )
    p.add_argument("--backends", type=str, default="cpu", help="Comma-separated backends to run (cpu,gpu).")
    p.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--json",
        type=str,
        default="result/tutorials/heat_mms_timeseries/results.json",
        help="Output JSON path (backend suffix is added automatically).",
    )
    p.add_argument(
        "--plot",
        type=str,
        default="result/tutorials/heat_mms_timeseries/compare_cpu_gpu.png",
        help="Output PNG path for CPU/GPU comparison.",
    )
    p.add_argument(
        "--pyvista-preview",
        action="store_true",
        default=True,
        help="Save a PyVista preview PNG from the last VTU (if pyvista is installed).",
    )
    p.add_argument(
        "--no-pyvista-preview",
        action="store_false",
        dest="pyvista_preview",
        help="Disable PyVista preview output.",
    )
    return p.parse_args()


def _make_backend_path(path: str, backend: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_{backend}{ext}"


def _parse_csv_floats(text: str) -> list[float]:
    if not text:
        return []
    return [float(tok) for tok in text.split(",") if tok.strip()]


def exact_u(
    coords: np.ndarray,
    t: float,
    kappa: float,
    mode: str,
    phase: float,
    phase_shifts: list[float],
    phase_weights: list[float],
) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    if mode == "multi":
        u1 = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        u2 = np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) * np.sin(3.0 * np.pi * z)
        lam1 = 3.0 * (np.pi ** 2) * kappa
        lam2 = (2.0 ** 2 + 1.0 ** 2 + 3.0 ** 2) * (np.pi ** 2) * kappa
        return u1 * np.exp(-lam1 * t) + 0.3 * u2 * np.exp(-lam2 * t)
    if mode == "phase":
        lam = 3.0 * (np.pi ** 2) * kappa
        return (
            np.sin(np.pi * x + phase)
            * np.sin(np.pi * y)
            * np.sin(np.pi * z)
            * np.exp(-lam * t)
        )
    if mode == "phase_sum":
        lam = 3.0 * (np.pi ** 2) * kappa
        shifts = phase_shifts if phase_shifts else [0.0]
        weights = phase_weights if phase_weights else [1.0]
        if len(weights) != len(shifts):
            raise ValueError("phase-weights must match phase-shifts length.")
        base = np.zeros_like(x)
        for w, ph in zip(weights, shifts):
            base += w * np.sin(np.pi * x + ph) * np.sin(np.pi * y) * np.sin(np.pi * z)
        return base * np.exp(-lam * t)
    lam = 3.0 * (np.pi ** 2) * kappa
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.exp(-lam * t)


def make_mesh(args):
    return ff.StructuredHexBox(
        nx=args.nx, ny=args.ny, nz=args.nz, lx=1.0, ly=1.0, lz=1.0
    ).build()


def make_space(mesh, intorder: int):
    return ff.make_hex_space(mesh, dim=1, intorder=intorder)


def assemble_weak_forms(
    space, kappa: float
) -> tuple[ff.FluxSparseMatrix, ff.FluxSparseMatrix]:
    stiffness = space.assemble_bilinear_form(ff.diffusion_form, params=kappa)
    mass = space.assemble_mass_matrix()
    return stiffness, mass


def run_backend(args, backend: str) -> dict:
    jax.config.update("jax_enable_x64", True)
    timer = SectionTimer()
    mesh = make_mesh(args)
    space = make_space(mesh, args.intorder)

    with timer.section("assemble"):
        K, M = assemble_weak_forms(space, args.kappa)
        jax.block_until_ready(K.data)
        jax.block_until_ready(M.data)
    assemble_s = timer.last("assemble")

    K_csr = K.to_csr()
    M_csr = M.to_csr()

    # FEA core: apply Dirichlet BCs and reduce the system to free DOFs.
    coords = np.asarray(mesh.coords)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox_pred = ff.bbox_predicate(mins, maxs, tol=1e-8)
    dir_dofs = mesh.boundary_dofs_where(bbox_pred, components="x")
    n_dofs = int(space.n_dofs)
    free = ff.free_dofs(n_dofs, dir_dofs)
    K_ff = K_csr[free][:, free]
    K_fd = K_csr[free][:, dir_dofs]
    M_ff = M_csr[free][:, free]
    M_fd = M_csr[free][:, dir_dofs]
    A_ff = (M_ff / args.dt) + K_ff
    A_fd = (M_fd / args.dt) + K_fd

    phase_shifts = _parse_csv_floats(args.phase_shifts)
    phase_weights = _parse_csv_floats(args.phase_weights)
    u0_nodal = exact_u(
        coords, 0.0, args.kappa, args.mode, args.phase,
        phase_shifts, phase_weights
    )
    norm0 = float(np.sqrt(u0_nodal @ (M_csr @ u0_nodal)))
    rhs0 = M_csr @ u0_nodal
    rhs0_free = rhs0[free] - M_fd @ u0_nodal[dir_dofs]
    u0_free = ff.spdirect_solve_cpu(M_ff, rhs0_free)
    u_dir0 = u0_nodal[dir_dofs]
    u = ff.expand_dirichlet_solution(u0_free, free, dir_dofs, u_dir0, n_total=n_dofs)
    u_free = u[free]

    solve_times = []
    errors = []
    errors_t0 = []
    errors_abs = []
    times = []

    vtu_dir = Path(args.vtu_dir)
    if args.save_vtu:
        if vtu_dir.exists():
            for path in vtu_dir.glob("*.vtu"):
                path.unlink()
            for path in vtu_dir.glob("preview_*.png"):
                path.unlink()
        vtu_dir.mkdir(parents=True, exist_ok=True)
    saved_vtus = []
    val_min = float("inf")
    val_max = float("-inf")

    for step in range(1, args.nsteps + 1):
        t_n = (step - 1) * args.dt
        t_np1 = step * args.dt
        u_dir_n = exact_u(
            coords[dir_dofs], t_n, args.kappa, args.mode, args.phase,
            phase_shifts, phase_weights
        )
        u_dir_np1 = exact_u(
            coords[dir_dofs], t_np1, args.kappa, args.mode, args.phase,
            phase_shifts, phase_weights
        )

        # Implicit Euler: (M/dt + K) u^{n+1} = (M/dt) u^n with Dirichlet coupling.
        rhs = (M_ff @ u_free) / args.dt + (M_fd @ u_dir_n) / args.dt - A_fd @ u_dir_np1

        with timer.section("solve_step"):
            if backend == "gpu":
                u_free = ff.spdirect_solve_jax(A_ff, rhs)
            else:
                u_free = ff.spdirect_solve_cpu(A_ff, rhs)
        solve_dt = timer.last("solve_step")
        solve_times.append(solve_dt)

        u = ff.expand_dirichlet_solution(u_free, free, dir_dofs, u_dir_np1, n_total=n_dofs)

        u_ex = exact_u(coords, t_np1, args.kappa, args.mode, args.phase, phase_shifts, phase_weights)
        e = u - u_ex
        err = float(np.sqrt(e @ (M_csr @ e)))
        norm = float(np.sqrt(u_ex @ (M_csr @ u_ex)))
        rel_err = err / (norm if norm > 0 else 1.0)
        rel_err_t0 = err / (norm0 if norm0 > 0 else 1.0)
        errors.append(rel_err)
        errors_t0.append(rel_err_t0)
        errors_abs.append(err)
        times.append(t_np1)

        if args.save_vtu and (step % max(args.save_every, 1) == 0 or step == args.nsteps):
            out_vtu = vtu_dir / f"step_{step:04d}_{backend}.vtu"
            ff.write_vtu(mesh, str(out_vtu), point_data={"u": u, "u_exact": u_ex})
            saved_vtus.append(out_vtu)
            val_min = min(val_min, float(u.min()), float(u_ex.min()))
            val_max = max(val_max, float(u.max()), float(u_ex.max()))

    if args.save_vtu and args.pyvista_preview and saved_vtus:
        clim = (val_min, val_max)
        for out_vtu in saved_vtus:
            step = int(out_vtu.stem.split("_")[1])
            frame_idx = max(step - 1, 0)
            t_val = step * args.dt
            out_png = out_vtu.with_name(f"preview_{frame_idx:04d}_{backend}.png")
            save_pyvista_preview(
                str(out_vtu), out_png, clim=clim, field="u",
                title=f"Solution t={t_val:.3f}"
            )
            out_png_ex = out_vtu.with_name(
                f"preview_{frame_idx:04d}_exact_{backend}.png"
            )
            save_pyvista_preview(
                str(out_vtu), out_png_ex, clim=clim,
                field="u_exact", title=f"Exact t={t_val:.3f}"
            )
            out_png_cat = out_vtu.with_name(
                f"preview_concat_{frame_idx:04d}_{backend}.png"
            )
            save_pyvista_preview_pair(
                str(out_vtu),
                out_png_cat,
                clim=clim,
                t_val=t_val,
            )

    return {
        "backend": backend,
        "dofs": int(n_dofs),
        "assemble_s": float(assemble_s),
        "solve_s": [float(x) for x in solve_times],
        "times": [float(x) for x in times],
        "rel_error": [float(x) for x in errors],
        "rel_error_t0": [float(x) for x in errors_t0],
        "abs_error": [float(x) for x in errors_abs],
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "kappa": args.kappa,
        "mode": args.mode,
        "phase": args.phase,
        "phase_shifts": phase_shifts,
        "phase_weights": phase_weights,
        "dt": args.dt,
        "nsteps": args.nsteps,
        "save_vtu": bool(args.save_vtu),
        "save_every": int(args.save_every),
        "vtu_dir": str(vtu_dir) if args.save_vtu else "",
    }


def main():
    args = parse_args()
    if args.n is not None:
        args.nx = args.n
        args.ny = args.n
        args.nz = args.n
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not args.single_backend and (len(backends) > 1 or (backends and backends[0] != jax.default_backend())):
        for backend in backends:
            env = os.environ.copy()
            env["JAX_PLATFORM_NAME"] = backend
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--single-backend",
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--nz",
                str(args.nz),
                "--kappa",
                str(args.kappa),
                "--dt",
                str(args.dt),
                "--nsteps",
                str(args.nsteps),
                "--intorder",
                str(args.intorder),
                "--json",
                _make_backend_path(args.json, backend),
            ]
            if args.save_vtu:
                cmd += [
                    "--save-vtu",
                    "--save-every",
                    str(args.save_every),
                    "--vtu-dir",
                    args.vtu_dir,
                ]
            if args.pyvista_preview:
                cmd.append("--pyvista-preview")
            else:
                cmd.append("--no-pyvista-preview")
            proc = subprocess.run(cmd, env=env, check=False)
            if proc.returncode != 0:
                if backend == "gpu":
                    print("[heat_mms] GPU backend unavailable; skipping GPU run.")
                    continue
                raise SystemExit(proc.returncode)

        cpu_json = Path(_make_backend_path(args.json, "cpu"))
        gpu_json = Path(_make_backend_path(args.json, "gpu"))
        if cpu_json.exists() and gpu_json.exists():
            cpu = json.loads(cpu_json.read_text())
            gpu = json.loads(gpu_json.read_text())
            plot_compare(cpu, gpu, args.plot)
        return

    backend = jax.default_backend()
    result = run_backend(args, backend)
    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(f"Saved results to {out_json}")

    plot_path = Path("result/tutorials/heat_mms_timeseries") / f"mms_{backend}.png"
    plot_backend(result, plot_path)


def plot_backend(result: dict, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = result["times"]
    rel = result.get("rel_error_t0", result["rel_error"])
    abs_err = result.get("abs_error", [])

    out_base = Path(out_path)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    fig_err, ax_err = plt.subplots(1, 1, figsize=(6.4, 4.0))
    ax_err.plot(times, rel, "o-", label="relative (t0)")
    if abs_err:
        ax_err.plot(times, abs_err, "s--", label="absolute")
    ax_err.set_xlabel("time")
    ax_err.set_ylabel("error")
    ax_err.set_title(f"MMS error vs time ({result['backend']})")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()
    out_err = out_base.with_name(out_base.stem + "_error" + out_base.suffix)
    fig_err.tight_layout()
    fig_err.savefig(out_err, dpi=180)
    plt.close(fig_err)
    print(f"Saved error plot to {out_err}")

    fig_sol, ax_sol = plt.subplots(1, 1, figsize=(6.4, 4.0))
    ax_sol.plot(result["solve_s"], "o-")
    ax_sol.set_xlabel("step")
    ax_sol.set_ylabel("solve time [s]")
    ax_sol.set_title(f"Solve time per step ({result['backend']})")
    ax_sol.grid(True, alpha=0.3)
    out_sol = out_base.with_name(out_base.stem + "_solve" + out_base.suffix)
    fig_sol.tight_layout()
    fig_sol.savefig(out_sol, dpi=180)
    plt.close(fig_sol)
    print(f"Saved solve plot to {out_sol}")


def save_pyvista_preview(
    vtu_path: str,
    out_path: Path,
    *,
    clim=None,
    field: str = "u",
    title: str | None = None,
) -> None:
    try:
        import pyvista as pv
    except Exception:
        print("[heat_mms] pyvista not available; skipping preview.")
        return

    pv.OFF_SCREEN = True
    mesh = pv.read(vtu_path)
    if "u" not in mesh.point_data and "u_exact" not in mesh.point_data:
        print("[heat_mms] no scalar field found; skipping preview.")
        return
    pl = pv.Plotter(off_screen=True)
    if title:
        pl.add_text(title, position="upper_left", font_size=10)
    sliced = mesh.slice_orthogonal()
    if hasattr(sliced, "combine"):
        sliced = sliced.combine()
    if hasattr(sliced, "point_data"):
        scalars = field if field in sliced.point_data else "u"
    else:
        if len(sliced) == 0:
            print("[heat_mms] no slice blocks; skipping preview.")
            return
        sliced = sliced[0]
        scalars = field if field in sliced.point_data else "u"
    if clim is None:
        arr = np.asarray(sliced.point_data[scalars])
        clim = (float(arr.min()), float(arr.max()))
    pl.add_mesh(sliced, scalars=scalars, cmap="viridis", clim=clim)
    pl.add_axes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.show(screenshot=str(out_path))
    print(f"Saved PyVista preview to {out_path}")


def save_pyvista_preview_pair(
    vtu_path: str,
    out_path: Path,
    *,
    clim=None,
    t_val: float,
) -> None:
    try:
        import pyvista as pv
    except Exception:
        print("[heat_mms] pyvista not available; skipping preview.")
        return

    pv.OFF_SCREEN = True
    mesh = pv.read(vtu_path)
    if "u" not in mesh.point_data or "u_exact" not in mesh.point_data:
        print("[heat_mms] missing u/u_exact; skipping preview concat.")
        return

    slices = mesh.slice_orthogonal()
    if hasattr(slices, "combine"):
        slices = slices.combine()
    if not hasattr(slices, "point_data"):
        if len(slices) == 0:
            print("[heat_mms] no slice blocks; skipping preview concat.")
            return
        slices = slices[0]

    pl = pv.Plotter(off_screen=True, shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_text(f"Solution t={t_val:.3f}", position="upper_left", font_size=10)
    pl.add_mesh(slices, scalars="u", cmap="viridis", clim=clim)
    pl.add_axes()

    pl.subplot(0, 1)
    pl.add_text(f"Exact t={t_val:.3f}", position="upper_left", font_size=10)
    pl.add_mesh(slices, scalars="u_exact", cmap="viridis", clim=clim)
    pl.add_axes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.show(screenshot=str(out_path))
    print(f"Saved PyVista preview to {out_path}")


def plot_compare(cpu: dict, gpu: dict, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpu_err = cpu.get("rel_error_t0", cpu["rel_error"])
    gpu_err = gpu.get("rel_error_t0", gpu["rel_error"])

    out_base = Path(out_path)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    fig_err, ax_err = plt.subplots(1, 1, figsize=(6.4, 4.0))
    ax_err.plot(cpu["times"], cpu_err, "o-", label="CPU")
    ax_err.plot(gpu["times"], gpu_err, "s--", label="GPU")
    ax_err.set_xlabel("time")
    ax_err.set_ylabel("relative L2 error")
    ax_err.set_title("MMS error vs time (t0 normalized)")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()
    out_err = out_base.with_name(out_base.stem + "_error" + out_base.suffix)
    fig_err.tight_layout()
    fig_err.savefig(out_err, dpi=180)
    plt.close(fig_err)
    print(f"Saved comparison error plot to {out_err}")

    fig_sol, ax_sol = plt.subplots(1, 1, figsize=(6.4, 4.0))
    ax_sol.plot(cpu["solve_s"], "o-", label="CPU")
    ax_sol.plot(gpu["solve_s"], "s--", label="GPU")
    ax_sol.set_xlabel("step")
    ax_sol.set_ylabel("solve time [s]")
    ax_sol.set_title("Solve time per step")
    ax_sol.grid(True, alpha=0.3)
    ax_sol.legend()
    out_sol = out_base.with_name(out_base.stem + "_solve" + out_base.suffix)
    fig_sol.tight_layout()
    fig_sol.savefig(out_sol, dpi=180)
    plt.close(fig_sol)
    print(f"Saved comparison solve plot to {out_sol}")


if __name__ == "__main__":
    main()

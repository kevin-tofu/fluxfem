#!/usr/bin/env python
"""Render a sphere-on-box release sequence as cross-section JPG frames.

This visualization-oriented tutorial starts from a pressed contact state:

- a deformable sphere sits on a deformable box under gravity,
- an additional top pressing load is present for the initial state,
- the pressing load is removed at t=0,
- the deformation relaxes with a simple mass-damping-stiffness field model.

The script writes JPG frames of the center cross-section.  It is intended for
README/release visuals and is not a full nonlinear 3D contact solve.

Run from the repository root:

    PYTHONPATH=src python tutorials/contact/sphere_box_release_section_jpg.py
    PYTHONPATH=src python tutorials/contact/sphere_box_release_section_jpg.py --renderer pyvista
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fluxfem")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
import numpy as np

import fluxfem as ff


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _release_response(times: np.ndarray, *, delta0: float, delta_eq: float, omega: float, zeta: float) -> np.ndarray:
    wd = omega * np.sqrt(max(1.0 - zeta * zeta, 1.0e-12))
    a = delta0 - delta_eq
    return delta_eq + a * np.exp(-zeta * omega * times) * (
        np.cos(wd * times) + (zeta * omega / wd) * np.sin(wd * times)
    )


def _sphere_section_points(radius: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return radius * np.cos(theta), radius * np.sin(theta)


def _box_grid(width: float, height: float, nx: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-0.5 * width, 0.5 * width, nx)
    zs = np.linspace(-height, 0.0, nz)
    return np.meshgrid(xs, zs)


def _contact_half_width(radius: float, indentation: float) -> float:
    hertz_like = np.sqrt(max(radius * max(indentation, 0.0), 1.0e-12))
    return float(np.clip(hertz_like, 0.10 * radius, 0.62 * radius))


def _solve_section_kkt(
    x_top: np.ndarray,
    *,
    radius: float,
    indentation: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    n_top = int(x_top.size)
    dx = float(np.mean(np.diff(x_top))) if n_top > 1 else 1.0
    support = np.abs(x_top) < 0.98 * radius
    x_c = x_top[support]
    if x_c.size == 0:
        raise ValueError("box top grid does not intersect the sphere footprint.")

    box_ground = 22.0 * dx * np.ones((n_top,), dtype=float)
    box_smooth = 2.2 / max(dx, 1.0e-12)
    K_box = np.diag(box_ground + 2.0 * box_smooth)
    for i in range(n_top - 1):
        K_box[i, i + 1] -= box_smooth
        K_box[i + 1, i] -= box_smooth
    K_box[0, 0] -= box_smooth
    K_box[-1, -1] -= box_smooth

    sphere_stiffness = 0.42 * float(np.sum(box_ground[support]))
    K = np.zeros((n_top + 1, n_top + 1), dtype=float)
    K[:n_top, :n_top] = K_box
    K[-1, -1] = sphere_stiffness

    force = np.zeros((n_top + 1,), dtype=float)
    force[-1] = -sphere_stiffness * float(indentation)

    profile = radius - np.sqrt(np.maximum(radius * radius - x_c * x_c, 0.0))
    G = np.zeros((x_c.size, n_top + 1), dtype=float)
    contact_top_ids = np.flatnonzero(support)
    G[np.arange(x_c.size), contact_top_ids] = -1.0
    G[:, -1] = 1.0

    result = ff.solve_unilateral_contact_active_set_kkt(
        K,
        force,
        G,
        profile,
        gap_tol=1e-9,
        lambda_tol=1e-9,
        maxiter=80,
    )
    w_top = result.displacement[:n_top]
    sphere_shift = float(result.displacement[-1])
    pressure = np.zeros((n_top,), dtype=float)
    pressure[contact_top_ids] = np.asarray(result.lambda_n, dtype=float)
    active_top = np.zeros((n_top,), dtype=bool)
    active_top[contact_top_ids] = np.asarray(result.active_mask, dtype=bool)
    gap_top = np.full((n_top,), np.nan, dtype=float)
    gap_top[contact_top_ids] = np.asarray(result.gap, dtype=float)
    return w_top, sphere_shift, pressure, active_top, gap_top


def _deformation_fields(
    x_box: np.ndarray,
    z_box: np.ndarray,
    x_sphere: np.ndarray,
    z_sphere0: np.ndarray,
    *,
    indentation: float,
    indentation0: float,
    radius: float,
    box_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_top = np.asarray(x_box[-1], dtype=float)
    w_top, sphere_shift, pressure_top, _active_top, _gap_top = _solve_section_kkt(
        x_top,
        radius=radius,
        indentation=indentation,
    )
    w_interp = np.interp(x_box, x_top, w_top)
    patch = np.exp(-(x_box / (0.42 * radius)) ** 2)
    depth = np.exp(z_box / (0.35 * box_height))
    box_w = w_interp * depth
    box_u = 0.018 * indentation * (x_box / radius) * patch * depth

    sphere_x = x_sphere
    sphere_z = radius + sphere_shift + z_sphere0

    pressure = np.tile(pressure_top, (x_box.shape[0], 1))
    return box_u, box_w, sphere_x, sphere_z, pressure, _gap_top


def _colored_lines(x: np.ndarray, z: np.ndarray, values: np.ndarray) -> LineCollection:
    points = np.column_stack([x, z])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, array=values[:-1], cmap="viridis", linewidth=3.0)
    return lc


def _contact_patch_mismatch(
    x_box: np.ndarray,
    z_box: np.ndarray,
    box_u: np.ndarray,
    box_w: np.ndarray,
    sphere_x: np.ndarray,
    sphere_z: np.ndarray,
    z_sphere0: np.ndarray,
    pressure_top: np.ndarray,
    *,
    radius: float,
    indentation: float,
) -> float:
    _ = indentation
    active_contact = np.asarray(pressure_top, dtype=float) > 1e-10
    if not np.any(active_contact):
        return 0.0
    box_top_x = x_box[-1] + box_u[-1]
    box_top_z = z_box[-1] + box_w[-1]
    sphere_center_z = float(np.mean(sphere_z - z_sphere0))
    lower_at_top = sphere_center_z - np.sqrt(np.maximum(radius * radius - box_top_x * box_top_x, 0.0))
    return float(np.max(np.abs(lower_at_top[active_contact] - box_top_z[active_contact])))


def _render_frame(
    path: Path,
    *,
    frame: int,
    time: float,
    indentation: float,
    indentation0: float,
    radius: float,
    box_width: float,
    box_height: float,
    x_box: np.ndarray,
    z_box: np.ndarray,
    x_sphere0: np.ndarray,
    z_sphere0: np.ndarray,
) -> dict[str, float]:
    box_u, box_w, sphere_x, sphere_z, pressure, gap_top = _deformation_fields(
        x_box,
        z_box,
        x_sphere0,
        z_sphere0,
        indentation=indentation,
        indentation0=indentation0,
        radius=radius,
        box_height=box_height,
    )
    disp_box = np.sqrt(box_u * box_u + box_w * box_w)
    sphere_disp = np.sqrt((sphere_x - x_sphere0) ** 2 + (sphere_z - (radius + z_sphere0)) ** 2)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax.set_facecolor("#f8fafc")
    ax.add_patch(Rectangle((-0.5 * box_width, -box_height), box_width, box_height, facecolor="#e5e7eb", edgecolor="#334155", lw=1.0))

    # Draw deformed box grid on the center section.
    for i in range(0, x_box.shape[0], 3):
        ax.plot(x_box[i] + box_u[i], z_box[i] + box_w[i], color="#94a3b8", lw=0.7, alpha=0.8)
    for j in range(0, x_box.shape[1], 4):
        ax.plot(x_box[:, j] + box_u[:, j], z_box[:, j] + box_w[:, j], color="#94a3b8", lw=0.7, alpha=0.8)

    lc = _colored_lines(sphere_x, sphere_z, sphere_disp)
    ax.add_collection(lc)
    sphere_center_z = float(np.mean(sphere_z - z_sphere0))
    ax.add_patch(Circle((0.0, sphere_center_z), radius, fill=False, ls="--", lw=0.8, ec="#64748b", alpha=0.8))

    contact_x = x_box[-1]
    contact_z = z_box[-1] + box_w[-1]
    contact_color = pressure[-1] / max(float(np.max(pressure)), 1.0e-12)
    ax.scatter(contact_x, contact_z, c=contact_color, cmap="magma", s=18.0, edgecolors="none", label="contact pressure")

    arrow_x = x_box[::4, ::5]
    arrow_z = z_box[::4, ::5]
    arrow_u = box_u[::4, ::5]
    arrow_w = box_w[::4, ::5]
    ax.quiver(
        arrow_x,
        arrow_z,
        5.0 * arrow_u,
        5.0 * arrow_w,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#0f766e",
        width=0.003,
        alpha=0.85,
    )

    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array(np.concatenate([disp_box.ravel(), sphere_disp.ravel()]))
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("displacement magnitude")

    ax.text(
        0.02,
        0.96,
        f"t = {time:.3f} s\nreleased top load\nindentation = {indentation:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
    )
    ax.set_title("Sphere-on-box release: center-section displacement")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5 * box_width - 0.1, 0.5 * box_width + 0.1)
    ax.set_ylim(-box_height - 0.15, 2.25 * radius)
    ax.grid(True, color="#cbd5e1", lw=0.5, alpha=0.45)
    fig.savefig(path, dpi=180, format="jpg")
    plt.close(fig)

    return {
        "frame": int(frame),
        "renderer": "matplotlib",
        "time": float(time),
        "indentation": float(indentation),
        "max_box_displacement": float(np.max(disp_box)),
        "max_sphere_displacement": float(np.max(sphere_disp)),
        "max_contact_pressure_proxy": float(np.max(pressure)),
        "section_contact_gap": float(np.nanmin(gap_top)),
        "active_contact_nodes": int(np.count_nonzero(pressure[-1] > 1e-10)),
        "max_contact_patch_mismatch": _contact_patch_mismatch(
            x_box,
            z_box,
            box_u,
            box_w,
            sphere_x,
            sphere_z,
            z_sphere0,
            pressure[-1],
            radius=radius,
            indentation=indentation,
        ),
    }


def _polyline(points: np.ndarray):
    import pyvista as pv

    line = pv.PolyData(points)
    line.lines = np.concatenate([[points.shape[0]], np.arange(points.shape[0])]).astype(np.int64)
    return line


def _render_frame_pyvista(
    path: Path,
    *,
    frame: int,
    time: float,
    indentation: float,
    indentation0: float,
    radius: float,
    box_width: float,
    box_height: float,
    x_box: np.ndarray,
    z_box: np.ndarray,
    x_sphere0: np.ndarray,
    z_sphere0: np.ndarray,
) -> dict[str, float]:
    import pyvista as pv

    pv.OFF_SCREEN = True
    box_u, box_w, sphere_x, sphere_z, pressure, gap_top = _deformation_fields(
        x_box,
        z_box,
        x_sphere0,
        z_sphere0,
        indentation=indentation,
        indentation0=indentation0,
        radius=radius,
        box_height=box_height,
    )
    disp_box = np.sqrt(box_u * box_u + box_w * box_w)
    sphere_disp = np.sqrt((sphere_x - x_sphere0) ** 2 + (sphere_z - (radius + z_sphere0)) ** 2)

    x_def = x_box + box_u
    z_def = z_box + box_w
    y_def = np.zeros_like(x_def)
    box_grid = pv.StructuredGrid(x_def, y_def, z_def)
    box_grid.point_data["displacement_magnitude"] = disp_box.ravel(order="F")
    box_grid.point_data["contact_pressure_proxy"] = pressure.ravel(order="F")

    sphere_points = np.column_stack([sphere_x, np.zeros_like(sphere_x), sphere_z])
    sphere_line = _polyline(sphere_points)
    sphere_line.point_data["displacement_magnitude"] = sphere_disp

    contact_points = np.column_stack([x_def[-1], np.zeros_like(x_def[-1]), z_def[-1]])
    contact_cloud = pv.PolyData(contact_points)
    contact_cloud.point_data["contact_pressure_proxy"] = pressure[-1]

    path.parent.mkdir(parents=True, exist_ok=True)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 900))
    pl.set_background("white")
    pl.add_text(
        f"sphere-box release section\n"
        f"t = {time:.3f} s, indentation = {indentation:.3f}\n"
        "top pressing load removed; gravity indentation remains",
        position="upper_left",
        font_size=11,
        color="black",
    )
    pl.add_mesh(
        box_grid,
        scalars="displacement_magnitude",
        cmap="viridis",
        show_edges=True,
        edge_color="#94a3b8",
        line_width=0.5,
        scalar_bar_args={"title": "displacement magnitude", "color": "black"},
    )
    pl.add_mesh(
        sphere_line,
        scalars="displacement_magnitude",
        cmap="viridis",
        render_lines_as_tubes=True,
        line_width=8,
        show_scalar_bar=False,
    )
    pl.add_mesh(
        contact_cloud,
        scalars="contact_pressure_proxy",
        cmap="magma",
        point_size=11,
        render_points_as_spheres=True,
        show_scalar_bar=False,
    )
    pl.add_axes(line_width=2, labels_off=True)
    pl.view_xz()
    pl.camera.zoom(1.28)
    pl.show(screenshot=str(path))
    pl.close()

    return {
        "frame": int(frame),
        "renderer": "pyvista",
        "time": float(time),
        "indentation": float(indentation),
        "max_box_displacement": float(np.max(disp_box)),
        "max_sphere_displacement": float(np.max(sphere_disp)),
        "max_contact_pressure_proxy": float(np.max(pressure)),
        "section_contact_gap": float(np.nanmin(gap_top)),
        "active_contact_nodes": int(np.count_nonzero(pressure[-1] > 1e-10)),
        "max_contact_patch_mismatch": _contact_patch_mismatch(
            x_box,
            z_box,
            box_u,
            box_w,
            sphere_x,
            sphere_z,
            z_sphere0,
            pressure[-1],
            radius=radius,
            indentation=indentation,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "sphere_box_release_section_jpg",
    )
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--duration", type=float, default=0.8)
    parser.add_argument("--radius", type=float, default=0.55)
    parser.add_argument("--box-width", type=float, default=3.0)
    parser.add_argument("--box-height", type=float, default=0.65)
    parser.add_argument("--initial-indentation", type=float, default=0.18)
    parser.add_argument("--gravity-indentation", type=float, default=0.045)
    parser.add_argument("--omega", type=float, default=18.0)
    parser.add_argument("--zeta", type=float, default=0.18)
    parser.add_argument(
        "--renderer",
        choices=("matplotlib", "pyvista", "both"),
        default="matplotlib",
        help="JPG renderer. pyvista writes sphere_box_release_pyvista_####.jpg.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("sphere_box_release_*.jpg"):
        old.unlink()

    times = np.linspace(0.0, args.duration, args.frames)
    indentation = _release_response(
        times,
        delta0=args.initial_indentation,
        delta_eq=args.gravity_indentation,
        omega=args.omega,
        zeta=args.zeta,
    )
    indentation = np.maximum(indentation, args.gravity_indentation * 0.35)

    x_box, z_box = _box_grid(args.box_width, args.box_height, nx=80, nz=24)
    x_sphere0, z_sphere0 = _sphere_section_points(args.radius, n=260)

    frames = []
    metrics = []
    for i, (time, delta) in enumerate(zip(times, indentation)):
        if args.renderer in ("matplotlib", "both"):
            frame = args.out_dir / f"sphere_box_release_{i:04d}.jpg"
            metrics.append(
                _render_frame(
                    frame,
                    frame=i,
                    time=float(time),
                    indentation=float(delta),
                    indentation0=args.initial_indentation,
                    radius=args.radius,
                    box_width=args.box_width,
                    box_height=args.box_height,
                    x_box=x_box,
                    z_box=z_box,
                    x_sphere0=x_sphere0,
                    z_sphere0=z_sphere0,
                )
            )
            frames.append(str(frame))
        if args.renderer in ("pyvista", "both"):
            frame = args.out_dir / f"sphere_box_release_pyvista_{i:04d}.jpg"
            metrics.append(
                _render_frame_pyvista(
                    frame,
                    frame=i,
                    time=float(time),
                    indentation=float(delta),
                    indentation0=args.initial_indentation,
                    radius=args.radius,
                    box_width=args.box_width,
                    box_height=args.box_height,
                    x_box=x_box,
                    z_box=z_box,
                    x_sphere0=x_sphere0,
                    z_sphere0=z_sphere0,
                )
            )
            frames.append(str(frame))

    summary = {
        "frames": frames,
        "frame_count": len(frames),
        "renderer": args.renderer,
        "duration": float(args.duration),
        "model": "Visualization-oriented K-M-C mass-damping-stiffness release surrogate.",
        "initial_state": "Sphere on a box under gravity plus an additional top pressing load.",
        "release": "The top pressing load is set to zero at t=0; gravity indentation remains.",
        "metrics": metrics,
    }
    summary_path = args.out_dir / "sphere_box_release_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"JPG frames written: {args.out_dir}")
    print(f"First frame: {frames[0]}")
    print(f"Last frame: {frames[-1]}")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()

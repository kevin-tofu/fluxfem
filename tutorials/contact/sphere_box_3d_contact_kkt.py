#!/usr/bin/env python
"""Solve a 3D sphere-box contact sequence with FluxFEM active-set KKT.

This is a real 3D volume-FE contact demo:

- a hex box is clamped on its bottom face,
- a voxelized hex sphere sits on the box,
- linear-elastic stiffness is assembled for both bodies,
- unilateral normal contact is solved with ``ff.solve_unilateral_contact_active_set_kkt``,
- VTU/PVD and optional PyVista center-section JPG frames are written.

The contact law is active-set KKT for ``gap >= 0, lambda >= 0, gap*lambda = 0``.
The sequence is quasi-static: the extra pressing load is ramped down to zero
while gravity remains.  It is not yet an implicit dynamic Newmark simulation.

Run from the repository root:

    PYTHONPATH=src python tutorials/contact/sphere_box_3d_contact_kkt.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff

jax.config.update("jax_enable_x64", True)


def _compact_hex_mesh(coords: np.ndarray, conn: np.ndarray) -> ff.HexMesh:
    used = np.unique(conn.reshape(-1))
    remap = -np.ones((coords.shape[0],), dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return ff.HexMesh(coords=jnp.asarray(coords[used], dtype=jnp.float64), conn=jnp.asarray(remap[conn], dtype=jnp.int32))


def _voxel_sphere_mesh(*, radius: float, n: int, center: tuple[float, float, float]) -> ff.HexMesh:
    xs = np.linspace(-radius, radius, n + 1)
    ys = np.linspace(-radius, radius, n + 1)
    zs = np.linspace(-radius, radius, n + 1)
    coords = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)

    def node_id(i: int, j: int, k: int) -> int:
        return k * (n + 1) * (n + 1) + j * (n + 1) + i

    conn = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cell_center = np.array(
                    [
                        0.5 * (xs[i] + xs[i + 1]),
                        0.5 * (ys[j] + ys[j + 1]),
                        0.5 * (zs[k] + zs[k + 1]),
                    ]
                )
                if np.linalg.norm(cell_center) <= radius:
                    conn.append(
                        [
                            node_id(i, j, k),
                            node_id(i + 1, j, k),
                            node_id(i + 1, j + 1, k),
                            node_id(i, j + 1, k),
                            node_id(i, j, k + 1),
                            node_id(i + 1, j, k + 1),
                            node_id(i + 1, j + 1, k + 1),
                            node_id(i, j + 1, k + 1),
                        ]
                    )
    if not conn:
        raise ValueError("sphere mesh has no cells; increase n or radius.")
    coords = coords + np.asarray(center, dtype=float)
    return _compact_hex_mesh(coords, np.asarray(conn, dtype=np.int32))


def _sphere_point_cloud(radius: float, *, surface_count: int, interior_shells: int) -> np.ndarray:
    points = [[0.0, 0.0, 0.0]]
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for shell in range(1, int(interior_shells) + 2):
        r = radius * shell / float(interior_shells + 1)
        count = max(18, int(surface_count * (r / radius) ** 2))
        for i in range(count):
            z = 1.0 - 2.0 * (i + 0.5) / count
            rho = np.sqrt(max(1.0 - z * z, 0.0))
            theta = golden * i
            points.append([r * rho * np.cos(theta), r * rho * np.sin(theta), r * z])
    return np.asarray(points, dtype=float)


def _pyvista_tet_sphere_mesh(
    *,
    radius: float,
    resolution: int,
    center: tuple[float, float, float],
    interior_shells: int = 5,
) -> ff.TetMesh:
    import pyvista as pv

    surface_count = max(64, int(8 * resolution * resolution))
    points = _sphere_point_cloud(radius, surface_count=surface_count, interior_shells=interior_shells)
    volume = pv.PolyData(points).delaunay_3d(alpha=2.5 * radius)
    cells = np.asarray(volume.cells, dtype=np.int32).reshape(-1, 5)
    tet_mask = cells[:, 0] == 4
    conn = cells[tet_mask, 1:5]
    if conn.size == 0:
        raise RuntimeError("PyVista did not generate tetrahedral sphere cells.")
    coords_local = np.asarray(volume.points, dtype=float)
    centers = np.mean(coords_local[conn], axis=1)
    keep = np.linalg.norm(centers, axis=1) <= 1.001 * radius
    conn = conn[keep]
    coords = coords_local + np.asarray(center, dtype=float)
    return ff.TetMesh(coords=jnp.asarray(coords, dtype=jnp.float64), conn=jnp.asarray(conn, dtype=jnp.int32))


def _gmsh_tet_sphere_mesh(
    *,
    radius: float,
    center: tuple[float, float, float],
    mesh_size: float,
    work_dir: Path,
    use_mmg: bool = False,
) -> ff.TetMesh:
    gmsh_exe = shutil.which("gmsh")
    if gmsh_exe is None:
        raise RuntimeError("gmsh executable was not found; install gmsh or use --sphere-mesh tet.")
    work_dir.mkdir(parents=True, exist_ok=True)
    geo = work_dir / "sphere.geo"
    msh = work_dir / "sphere.msh"
    geo.write_text(
        "\n".join(
            [
                'SetFactory("OpenCASCADE");',
                f"Sphere(1) = {{0, 0, 0, {radius:.16e}, -Pi/2, Pi/2, 2*Pi}};",
                "Physical Volume(1) = {1};",
                f"Mesh.CharacteristicLengthMin = {0.55 * mesh_size:.16e};",
                f"Mesh.CharacteristicLengthMax = {mesh_size:.16e};",
                "Mesh.Algorithm3D = 10;",
                "Mesh.Optimize = 1;",
                "Mesh.OptimizeNetgen = 1;",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    subprocess.run(
        [gmsh_exe, "-3", "-format", "msh2", "-o", str(msh), str(geo)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if use_mmg:
        mmg_exe = shutil.which("mmg3d")
        if mmg_exe is None:
            raise RuntimeError("mmg3d executable was not found; run without --mmg3d-remesh.")
        try:
            import meshio
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("meshio is required for mmg3d remeshing conversion.") from exc
        mesh = meshio.read(msh)
        medit_in = work_dir / "sphere.mesh"
        medit_out = work_dir / "sphere_mmg.mesh"
        remeshed_msh = work_dir / "sphere_mmg.msh"
        meshio.write(medit_in, mesh)
        subprocess.run(
            [
                mmg_exe,
                "-in",
                str(medit_in),
                "-out",
                str(medit_out),
                "-hsiz",
                f"{mesh_size:.16e}",
                "-hausd",
                f"{0.18 * mesh_size:.16e}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        remeshed = meshio.read(medit_out)
        if "tetra" not in remeshed.cells_dict:
            raise RuntimeError("mmg3d output does not contain tetra cells.")
        coords = np.asarray(remeshed.points[:, :3], dtype=float) + np.asarray(center, dtype=float)
        conn = np.asarray(remeshed.cells_dict["tetra"], dtype=np.int32)
        return ff.TetMesh(coords=jnp.asarray(coords, dtype=jnp.float64), conn=jnp.asarray(conn, dtype=jnp.int32))
    mesh, _facets, _tags = ff.load_gmsh_tet_mesh(str(msh))
    coords = np.asarray(mesh.coords, dtype=float) + np.asarray(center, dtype=float)
    return ff.TetMesh(coords=jnp.asarray(coords, dtype=jnp.float64), conn=mesh.conn)


def _block_diag_dense(a, b) -> np.ndarray:
    a_np = np.asarray(a.to_dense() if hasattr(a, "to_dense") else a, dtype=float)
    b_np = np.asarray(b.to_dense() if hasattr(b, "to_dense") else b, dtype=float)
    out = np.zeros((a_np.shape[0] + b_np.shape[0], a_np.shape[1] + b_np.shape[1]), dtype=float)
    out[: a_np.shape[0], : a_np.shape[1]] = a_np
    out[a_np.shape[0] :, a_np.shape[1] :] = b_np
    return out


def _space_for_mesh(mesh):
    if isinstance(mesh, ff.HexMesh):
        return ff.make_hex_space(mesh, dim=3, intorder=2)
    if isinstance(mesh, ff.TetMesh):
        return ff.make_tet_space(mesh, dim=3, intorder=2)
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def _nearest_box_top_pairs(box_coords: np.ndarray, sphere_coords: np.ndarray, *, radius: float) -> tuple[np.ndarray, np.ndarray]:
    box_top = np.flatnonzero(np.isclose(box_coords[:, 2], box_coords[:, 2].max()))
    sphere_bottom = np.flatnonzero(
        (sphere_coords[:, 2] <= sphere_coords[:, 2].min() + 0.42 * (2.0 * radius / max(np.cbrt(sphere_coords.shape[0]), 1.0)))
        & (np.linalg.norm(sphere_coords[:, :2], axis=1) <= 0.82 * radius)
    )
    if sphere_bottom.size == 0:
        sphere_bottom = np.array([int(np.argmin(sphere_coords[:, 2]))], dtype=np.int32)
    xy_top = box_coords[box_top, :2]
    box_ids = []
    for sid in sphere_bottom:
        d2 = np.sum((xy_top - sphere_coords[int(sid), :2]) ** 2, axis=1)
        box_ids.append(int(box_top[int(np.argmin(d2))]))
    return np.asarray(box_ids, dtype=np.int32), np.asarray(sphere_bottom, dtype=np.int32)


def _contact_gap_system(
    box_coords: np.ndarray,
    sphere_coords: np.ndarray,
    box_nodes: np.ndarray,
    sphere_nodes: np.ndarray,
    *,
    n_box_dofs: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_contacts = int(sphere_nodes.size)
    n_total = n_box_dofs + 3 * sphere_coords.shape[0]
    G = np.zeros((n_contacts, n_total), dtype=float)
    gap0 = np.zeros((n_contacts,), dtype=float)
    for row, (b, s) in enumerate(zip(box_nodes, sphere_nodes)):
        G[row, 3 * int(b) + 2] = -1.0
        G[row, n_box_dofs + 3 * int(s) + 2] = 1.0
        gap0[row] = float(sphere_coords[int(s), 2] - box_coords[int(b), 2])
    return G, gap0


def _write_pvd(path: Path, files: list[Path], times: list[float]) -> None:
    with path.open("w", encoding="ascii") as io:
        io.write('<?xml version="1.0"?>\n')
        io.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        io.write("  <Collection>\n")
        for file, time in zip(files, times):
            io.write(f'    <DataSet timestep="{time:.8e}" group="" part="0" file="{file.name}"/>\n')
        io.write("  </Collection>\n")
        io.write("</VTKFile>\n")


def _combined_cell_data(box_mesh, sphere_mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    box_coords = np.asarray(box_mesh.coords, dtype=float)
    sphere_coords = np.asarray(sphere_mesh.coords, dtype=float)
    box_conn = np.asarray(box_mesh.conn, dtype=np.int32)
    sphere_conn = np.asarray(sphere_mesh.conn, dtype=np.int32)
    coords = np.vstack([box_coords, sphere_coords])
    box_cells = np.column_stack([np.full(box_conn.shape[0], 8, dtype=np.int32), box_conn])
    if isinstance(sphere_mesh, ff.TetMesh):
        sphere_cells = np.column_stack(
            [np.full(sphere_conn.shape[0], 4, dtype=np.int32), sphere_conn + box_coords.shape[0]]
        )
        sphere_types = np.full(sphere_conn.shape[0], 10, dtype=np.uint8)
    elif isinstance(sphere_mesh, ff.HexMesh):
        sphere_cells = np.column_stack(
            [np.full(sphere_conn.shape[0], 8, dtype=np.int32), sphere_conn + box_coords.shape[0]]
        )
        sphere_types = np.full(sphere_conn.shape[0], 12, dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported sphere mesh type: {type(sphere_mesh)}")
    cells = np.concatenate([box_cells.reshape(-1), sphere_cells.reshape(-1)]).astype(np.int64)
    celltypes = np.concatenate([np.full(box_conn.shape[0], 12, dtype=np.uint8), sphere_types])
    body_id = np.concatenate([np.zeros(box_conn.shape[0]), np.ones(sphere_conn.shape[0])]).astype(float)
    return coords, cells, celltypes, body_id


def _write_mixed_vtu(
    path: Path,
    *,
    points: np.ndarray,
    cells: np.ndarray,
    celltypes: np.ndarray,
    point_data: dict[str, np.ndarray],
    cell_data: dict[str, np.ndarray],
) -> None:
    import pyvista as pv

    grid = pv.UnstructuredGrid(cells, celltypes, points)
    for name, values in point_data.items():
        grid.point_data[name] = np.asarray(values)
    for name, values in cell_data.items():
        grid.cell_data[name] = np.asarray(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(path))


def _write_section_jpg(
    vtu: Path,
    jpg: Path,
    *,
    title: str,
    clim: tuple[float, float] | None = None,
) -> bool:
    try:
        import pyvista as pv
    except Exception as exc:
        print(f"PyVista is not available; skipping JPG: {exc}")
        return False
    pv.OFF_SCREEN = True
    mesh = pv.read(str(vtu))
    section = mesh.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0))
    if section.n_points == 0:
        return False
    jpg.parent.mkdir(parents=True, exist_ok=True)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 950))
    pl.set_background("white")
    pl.add_text(title, position="upper_left", font_size=11, color="black")
    pl.add_mesh(
        section,
        scalars="displacement_magnitude",
        cmap="viridis",
        clim=clim,
        show_edges=True,
        edge_color="#b8c2cc",
        line_width=0.25,
        scalar_bar_args={"title": "displacement magnitude", "color": "black"},
    )
    pl.view_xz()
    pl.camera.zoom(1.45)
    pl.show(screenshot=str(jpg))
    pl.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "sphere_box_3d_contact_kkt")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--box-nx", type=int, default=14)
    parser.add_argument("--box-ny", type=int, default=14)
    parser.add_argument("--box-nz", type=int, default=4)
    parser.add_argument("--sphere-n", type=int, default=7)
    parser.add_argument("--sphere-mesh", choices=("gmsh", "tet", "voxel"), default="gmsh")
    parser.add_argument("--sphere-resolution", type=int, default=24)
    parser.add_argument("--sphere-interior-shells", type=int, default=5)
    parser.add_argument("--sphere-mesh-size", type=float, default=0.085)
    parser.add_argument("--mmg3d-remesh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radius", type=float, default=0.48)
    parser.add_argument("--initial-clearance", type=float, default=-0.025)
    parser.add_argument("--gravity-load", type=float, default=0.004)
    parser.add_argument("--press-load", type=float, default=0.014)
    parser.add_argument("--box-modulus", type=float, default=20.0)
    parser.add_argument("--sphere-modulus", type=float, default=28.0)
    parser.add_argument("--lateral-stabilization", type=float, default=1.0e-4)
    parser.add_argument("--deformed-scale", type=float, default=1.0)
    parser.add_argument("--section-jpg", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("sphere_box_3d_contact_*.vtu"):
        old.unlink()
    old_pvd = args.out_dir / "sphere_box_3d_contact_series.pvd"
    if old_pvd.exists():
        old_pvd.unlink()
    jpg_dir = args.out_dir / "section_jpg"
    if jpg_dir.exists():
        for old in jpg_dir.glob("sphere_box_3d_contact_section_*.jpg"):
            old.unlink()

    box_mesh = ff.StructuredHexBox(
        nx=args.box_nx,
        ny=args.box_ny,
        nz=args.box_nz,
        lx=2.4,
        ly=2.4,
        lz=0.55,
        origin=(-1.2, -1.2, -0.55),
    ).build()
    sphere_center = (0.0, 0.0, args.radius + args.initial_clearance)
    if args.sphere_mesh == "gmsh":
        sphere_mesh = _gmsh_tet_sphere_mesh(
            radius=args.radius,
            center=sphere_center,
            mesh_size=args.sphere_mesh_size,
            work_dir=args.out_dir / "_mesh_work",
            use_mmg=bool(args.mmg3d_remesh),
        )
    elif args.sphere_mesh == "tet":
        sphere_mesh = _pyvista_tet_sphere_mesh(
            radius=args.radius,
            resolution=args.sphere_resolution,
            center=sphere_center,
            interior_shells=args.sphere_interior_shells,
        )
    else:
        sphere_mesh = _voxel_sphere_mesh(radius=args.radius, n=args.sphere_n, center=sphere_center)
    box_space = ff.make_hex_space(box_mesh, dim=3, intorder=2)
    sphere_space = _space_for_mesh(sphere_mesh)
    D_box = ff.isotropic_3d_D(args.box_modulus, 0.30)
    D_sphere = ff.isotropic_3d_D(args.sphere_modulus, 0.28)
    K_box = box_space.assemble(ff.linear_elasticity_form, params=D_box)
    K_sphere = sphere_space.assemble(ff.linear_elasticity_form, params=D_sphere)
    K = _block_diag_dense(K_box, K_sphere)
    # Remove free rigid-body null modes without visually affecting the contact solution.
    K += 1.0e-8 * max(float(np.max(np.abs(K))), 1.0) * np.eye(K.shape[0])

    box_coords = np.asarray(box_mesh.coords, dtype=float)
    sphere_coords = np.asarray(sphere_mesh.coords, dtype=float)
    n_box_dofs = int(box_space.n_dofs)
    box_nodes, sphere_nodes = _nearest_box_top_pairs(box_coords, sphere_coords, radius=args.radius)
    G, gap0 = _contact_gap_system(box_coords, sphere_coords, box_nodes, sphere_nodes, n_box_dofs=n_box_dofs)

    bottom = np.flatnonzero(np.isclose(box_coords[:, 2], box_coords[:, 2].min()))
    fixed_dofs = np.asarray([3 * int(n) + d for n in bottom for d in range(3)], dtype=np.int32)
    sphere_dofs_z = n_box_dofs + 3 * np.arange(sphere_coords.shape[0]) + 2
    sphere_dofs_xy = np.concatenate(
        [
            n_box_dofs + 3 * np.arange(sphere_coords.shape[0]),
            n_box_dofs + 3 * np.arange(sphere_coords.shape[0]) + 1,
        ]
    )
    k_ref = max(float(np.max(np.abs(K))), 1.0)
    K[sphere_dofs_xy, sphere_dofs_xy] += float(args.lateral_stabilization) * k_ref
    base_force = np.zeros((K.shape[0],), dtype=float)
    base_force[sphere_dofs_z] = -float(args.gravity_load) / float(sphere_coords.shape[0])
    press_force = np.zeros_like(base_force)
    top_sphere = np.flatnonzero(sphere_coords[:, 2] >= np.percentile(sphere_coords[:, 2], 88.0))
    press_force[n_box_dofs + 3 * top_sphere + 2] = -float(args.press_load) / float(max(top_sphere.size, 1))

    ref_coords, cells, celltypes, body_id = _combined_cell_data(box_mesh, sphere_mesh)
    vtus: list[Path] = []
    jpgs: list[Path] = []
    metrics = []
    frame_times: list[float] = []
    active = None
    times = np.linspace(0.0, 1.0, int(args.frames))
    for step, time in enumerate(times):
        release = 1.0 - float(time)
        force = base_force + release * press_force
        result = ff.solve_unilateral_contact_active_set_kkt(
            K,
            force,
            G,
            gap0,
            fixed_dofs=fixed_dofs,
            initial_active=active,
            maxiter=80,
            gap_tol=1.0e-9,
            lambda_tol=1.0e-9,
            config=ff.ContactKKTSolveConfig(backend="numpy"),
        )
        active = result.active_mask
        u = np.asarray(result.displacement, dtype=float).reshape(-1, 3)
        deformed = ref_coords + float(args.deformed_scale) * u
        disp_mag = np.linalg.norm(u, axis=1)
        contact_lambda = np.zeros((ref_coords.shape[0],), dtype=float)
        contact_gap = np.full((ref_coords.shape[0],), np.nan, dtype=float)
        sphere_global = box_coords.shape[0] + sphere_nodes
        contact_lambda[sphere_global] = result.lambda_n
        contact_gap[sphere_global] = result.gap
        vtu = args.out_dir / f"sphere_box_3d_contact_{step:04d}.vtu"
        _write_mixed_vtu(
            vtu,
            points=deformed,
            cells=cells,
            celltypes=celltypes,
            point_data={
                "displacement": u,
                "displacement_magnitude": disp_mag,
                "contact_lambda": contact_lambda,
                "contact_gap": contact_gap,
            },
            cell_data={"body_id": body_id},
        )
        vtus.append(vtu)
        frame_times.append(float(time))
        metrics.append(
            {
                "step": int(step),
                "time": float(time),
                "release_factor": float(release),
                "converged": bool(result.converged),
                "iters": int(result.iters),
                "active_contacts": int(np.count_nonzero(result.active_mask)),
                "min_gap": float(np.min(result.gap)) if result.gap.size else 0.0,
                "max_lambda": float(np.max(result.lambda_n)) if result.lambda_n.size else 0.0,
                "max_displacement": float(np.max(disp_mag)),
                "sphere_max_displacement": float(np.max(disp_mag[box_coords.shape[0] :])),
                "box_max_displacement": float(np.max(disp_mag[: box_coords.shape[0]])),
            }
        )

    global_disp_max = float(max((m["max_displacement"] for m in metrics), default=0.0))
    if args.section_jpg:
        for step, (time, vtu) in enumerate(zip(frame_times, vtus)):
            jpg = args.out_dir / "section_jpg" / f"sphere_box_3d_contact_section_{step:04d}.jpg"
            if _write_section_jpg(
                vtu,
                jpg,
                title=f"3D sphere-box active-set KKT  t={time:.3f}",
                clim=(0.0, global_disp_max if global_disp_max > 0.0 else 1.0),
            ):
                jpgs.append(jpg)

    pvd = args.out_dir / "sphere_box_3d_contact_series.pvd"
    _write_pvd(pvd, vtus, list(map(float, times)))
    summary = {
        "model": "3D volume-FE linear elastic sphere-box unilateral contact solved by FluxFEM active-set KKT.",
        "note": "The sequence is quasi-static preload release, not dynamic Newmark yet.",
        "pvd": str(pvd),
        "first_vtu": str(vtus[0]),
        "final_vtu": str(vtus[-1]),
        "section_jpg_dir": str((args.out_dir / "section_jpg")) if jpgs else None,
        "first_section_jpg": str(jpgs[0]) if jpgs else None,
        "final_section_jpg": str(jpgs[-1]) if jpgs else None,
        "n_box_nodes": int(box_coords.shape[0]),
        "n_sphere_nodes": int(sphere_coords.shape[0]),
        "sphere_mesh": str(args.sphere_mesh),
        "deformed_scale": float(args.deformed_scale),
        "lateral_stabilization": float(args.lateral_stabilization),
        "n_contact_candidates": int(G.shape[0]),
        "section_color_limits": [0.0, global_disp_max],
        "metrics": metrics,
    }
    summary_path = args.out_dir / "sphere_box_3d_contact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PVD written: {pvd}")
    print(f"Final VTU: {vtus[-1]}")
    if jpgs:
        print(f"Section JPGs: {jpgs[0]} ... {jpgs[-1]}")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()

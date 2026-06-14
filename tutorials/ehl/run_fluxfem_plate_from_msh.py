#!/usr/bin/env python3
"""EHL plate setup from Gmsh mesh for FluxFEM.

This script reads `tutorials/ehl/data/ehl_point_contact.msh`, extracts the
`PLATE` volume and its tagged boundary facets (`TOP`, `BOTTOM`, `SIDES`), then
prepares FluxFEM objects for analysis.

Modes:
- default: load/check only (fast)
- --assemble: build FE space and assemble linear elasticity matrix/RHS
- --solve: additionally solve one static linear step (BOTTOM fixed, pressure on TOP)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare FluxFEM EHL model from .msh")
    p.add_argument(
        "--mesh",
        type=Path,
        default=Path("tutorials/ehl/data/ehl_point_contact.msh"),
        help="Input Gmsh mesh (.msh)",
    )
    p.add_argument("--intorder", type=int, default=2, help="Tet integration order")
    p.add_argument("--E", type=float, default=210_000.0, help="Young's modulus [MPa]")
    p.add_argument("--nu", type=float, default=0.30, help="Poisson ratio [-]")
    p.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Uniform normal pressure on TOP [MPa], applied along -z",
    )
    p.add_argument(
        "--assemble",
        action="store_true",
        help="Build space and assemble K/F for linear elasticity",
    )
    p.add_argument(
        "--solve",
        action="store_true",
        help="Solve static linear system (implies --assemble)",
    )
    p.add_argument(
        "--output-vtu",
        type=Path,
        default=Path("tutorials/ehl/data/ehl_plate_static.vtu"),
        help="Output VTU path when --solve is used",
    )
    return p.parse_args()


def _physical_name_to_tag(msh: meshio.Mesh) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, vals in msh.field_data.items():
        if len(vals) < 1:
            continue
        out[name] = int(vals[0])
    return out


def _require_tag(tag_map: dict[str, int], name: str) -> int:
    if name not in tag_map:
        names = ", ".join(sorted(tag_map.keys()))
        raise KeyError(f"Physical name '{name}' not found. Available: {names}")
    return int(tag_map[name])


def _extract_plate_domain(msh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    if "tetra" not in msh.cells_dict:
        raise ValueError("Input mesh has no tetra cells")
    if "triangle" not in msh.cells_dict:
        raise ValueError("Input mesh has no triangle boundary facets")

    if "gmsh:physical" not in msh.cell_data_dict:
        raise ValueError("Input mesh has no gmsh physical tags")

    phys_map = _physical_name_to_tag(msh)
    plate_tag = _require_tag(phys_map, "PLATE")
    top_tag = _require_tag(phys_map, "TOP")
    bottom_tag = _require_tag(phys_map, "BOTTOM")
    sides_tag = _require_tag(phys_map, "SIDES")

    tet_all = np.asarray(msh.cells_dict["tetra"], dtype=np.int64)
    tri_all = np.asarray(msh.cells_dict["triangle"], dtype=np.int64)

    tet_phys = np.asarray(msh.cell_data_dict["gmsh:physical"].get("tetra"), dtype=np.int64)
    tri_phys = np.asarray(msh.cell_data_dict["gmsh:physical"].get("triangle"), dtype=np.int64)

    if tet_phys.shape[0] != tet_all.shape[0]:
        raise ValueError("Mismatch between tetra cells and tetra physical tags")
    if tri_phys.shape[0] != tri_all.shape[0]:
        raise ValueError("Mismatch between triangle cells and triangle physical tags")

    plate_mask = np.abs(tet_phys) == plate_tag
    plate_conn_old = tet_all[plate_mask]
    if plate_conn_old.size == 0:
        raise ValueError("No tetra cells found for physical volume 'PLATE'")

    used_nodes = np.unique(plate_conn_old.reshape(-1))
    old_to_new = -np.ones(msh.points.shape[0], dtype=np.int64)
    old_to_new[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)

    coords = np.asarray(msh.points[used_nodes, :3], dtype=np.float64)
    plate_conn = old_to_new[plate_conn_old]

    def remap_surface_by_tag(tag: int) -> np.ndarray:
        sel = tri_all[np.abs(tri_phys) == tag]
        if sel.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        remapped = old_to_new[sel]
        keep = np.all(remapped >= 0, axis=1)
        return remapped[keep]

    surfaces = {
        "TOP": remap_surface_by_tag(top_tag),
        "BOTTOM": remap_surface_by_tag(bottom_tag),
        "SIDES": remap_surface_by_tag(sides_tag),
    }

    return coords, plate_conn, surfaces, phys_map


def _print_mesh_summary(coords: np.ndarray, conn: np.ndarray, surfaces: dict[str, np.ndarray]) -> None:
    n_nodes = int(coords.shape[0])
    n_tets = int(conn.shape[0])
    n_top = int(surfaces["TOP"].shape[0])
    n_bottom = int(surfaces["BOTTOM"].shape[0])
    n_sides = int(surfaces["SIDES"].shape[0])

    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)

    print(f"plate nodes: {n_nodes}")
    print(f"plate tets : {n_tets}")
    print(f"TOP tris   : {n_top}")
    print(f"BOTTOM tris: {n_bottom}")
    print(f"SIDES tris : {n_sides}")
    print(
        "bbox       : "
        f"x=[{xyz_min[0]:.6f}, {xyz_max[0]:.6f}] "
        f"y=[{xyz_min[1]:.6f}, {xyz_max[1]:.6f}] "
        f"z=[{xyz_min[2]:.6f}, {xyz_max[2]:.6f}]"
    )


def main() -> None:
    args = parse_args()
    if args.solve:
        args.assemble = True

    msh = meshio.read(str(args.mesh))
    coords, conn, surfaces, phys_map = _extract_plate_domain(msh)

    print("loaded:", args.mesh)
    print("physical tags:", {k: int(v) for k, v in sorted(phys_map.items())})
    _print_mesh_summary(coords, conn, surfaces)

    if not args.assemble:
        print("mode       : load/check only (no assembly)")
        return

    mesh = ff.TetMesh(coords=jnp.asarray(coords), conn=jnp.asarray(conn))
    space = ff.make_tet_space(mesh, dim=3, intorder=args.intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    D = ff.isotropic_3d_D(args.E, args.nu)
    K = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=V, trial=U),
        ff.linear_elasticity_form,
        D,
    )

    traction_vec = np.array([0.0, 0.0, -float(args.pressure)], dtype=float)
    top_surface = ff.make_surface_from_facets(coords, surfaces["TOP"])
    F = top_surface.assemble_linear_form_on_space(
        space,
        ff.vector_surface_load_form,
        params=traction_vec,
    )
    F = jnp.asarray(F, dtype=jnp.float64)

    bottom_nodes = np.unique(surfaces["BOTTOM"].reshape(-1))
    dir_dofs = mesh.node_dofs(bottom_nodes, components="xyz", dof_per_node=3)

    print(f"assemble   : dofs={space.n_dofs}, dirichlet_dofs={dir_dofs.size}")

    if not args.solve:
        print("mode       : assembled (no solve)")
        return

    solver = ff.LinearSolver(method="spsolve")
    u, _ = solver.solve(
        K,
        F,
        dirichlet=ff.DirichletBC(dir_dofs, None),
        dirichlet_mode="condense",
    )

    u_nodes = np.asarray(u).reshape(-1, 3)
    top_nodes = np.unique(surfaces["TOP"].reshape(-1))
    uz_top = u_nodes[top_nodes, 2]

    print(f"solve      : ||u||_inf={np.abs(u_nodes).max():.6e}")
    print(f"            uz_top_min={uz_top.min():.6e}, uz_top_max={uz_top.max():.6e}")

    args.output_vtu.parent.mkdir(parents=True, exist_ok=True)
    ff.write_elastic_vtu(mesh, space, u, str(args.output_vtu), compute_j=False, deformed_scale=1.0)
    print("vtu written:", args.output_vtu)


if __name__ == "__main__":
    main()

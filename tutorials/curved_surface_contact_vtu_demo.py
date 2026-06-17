#!/usr/bin/env python
"""Generate a curved-surface contact VTU time series for release visuals.

This is a visualization-oriented demo, not a nonlinear contact solve.  It writes
a pair of curved hex bodies with fields that make the contact/ROM ingredients
visible in ParaView:

- displacement: warp-by-vector field
- gap: signed local separation, negative where contact is active
- contact_pressure: penalty-style visual contact pressure
- active_contact: 1 on active interface nodes
- cb_retained: retained contact/support DOF marker
- mortar_weight: interface weighting marker

Run from the repository root:

    PYTHONPATH=src python tutorials/curved_surface_contact_vtu_demo.py

Open the generated .pvd file in ParaView and apply Warp By Vector using
`displacement`.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _surface_shape(x: np.ndarray, y: np.ndarray, *, length: float, width: float) -> np.ndarray:
    xc = (x - 0.52 * length) / (0.28 * length)
    yc = y / (0.36 * width)
    bump = 0.070 * np.exp(-(xc * xc + yc * yc))
    wave = 0.018 * np.sin(2.0 * np.pi * x / length) * np.cos(np.pi * y / width)
    return bump + wave


def _contact_shape(x: np.ndarray, y: np.ndarray, *, length: float, width: float) -> np.ndarray:
    xc = (x - 0.54 * length) / (0.24 * length)
    yc = y / (0.30 * width)
    return np.exp(-(xc * xc + yc * yc))


def _build_curved_pair(
    *,
    nx: int,
    ny: int,
    nz: int,
    length: float,
    width: float,
    thickness: float,
    clearance: float,
) -> tuple[ff.HexMesh, np.ndarray, np.ndarray, np.ndarray]:
    bottom = ff.StructuredHexBox(
        nx=nx,
        ny=ny,
        nz=nz,
        lx=length,
        ly=width,
        lz=thickness,
        origin=(0.0, -0.5 * width, 0.0),
    ).build()
    top = ff.StructuredHexBox(
        nx=nx,
        ny=ny,
        nz=nz,
        lx=length,
        ly=width,
        lz=thickness,
        origin=(0.0, -0.5 * width, 0.0),
    ).build()

    bottom_coords = np.asarray(bottom.coords, dtype=float).copy()
    top_coords = np.asarray(top.coords, dtype=float).copy()

    shape_bottom = _surface_shape(bottom_coords[:, 0], bottom_coords[:, 1], length=length, width=width)
    shape_top = _surface_shape(top_coords[:, 0], top_coords[:, 1], length=length, width=width)
    bottom_s = np.clip(bottom_coords[:, 2] / thickness, 0.0, 1.0)
    top_s = np.clip(top_coords[:, 2] / thickness, 0.0, 1.0)

    bottom_coords[:, 2] = bottom_s * (thickness + shape_bottom)
    top_lower = thickness + shape_top + clearance + 0.012 * np.cos(2.0 * np.pi * top_coords[:, 0] / length)
    top_coords[:, 2] = top_lower + top_s * thickness

    coords = np.vstack([bottom_coords, top_coords])
    bottom_conn = np.asarray(bottom.conn, dtype=np.int32)
    top_conn = np.asarray(top.conn, dtype=np.int32) + bottom_coords.shape[0]
    conn = np.vstack([bottom_conn, top_conn])

    body_id = np.concatenate(
        [
            np.zeros(bottom_conn.shape[0], dtype=np.float32),
            np.ones(top_conn.shape[0], dtype=np.float32),
        ]
    )
    node_body = np.concatenate(
        [
            np.zeros(bottom_coords.shape[0], dtype=np.int32),
            np.ones(top_coords.shape[0], dtype=np.int32),
        ]
    )
    local_s = np.concatenate([bottom_s, top_s])
    mesh = ff.HexMesh(coords=jnp.asarray(coords), conn=jnp.asarray(conn))
    return mesh, body_id, node_body, local_s


def _point_fields(
    mesh: ff.HexMesh,
    node_body: np.ndarray,
    local_s: np.ndarray,
    *,
    step: int,
    nsteps: int,
    length: float,
    width: float,
    thickness: float,
    clearance: float,
) -> dict[str, np.ndarray]:
    coords = np.asarray(mesh.coords, dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    t = 0.0 if nsteps <= 1 else step / float(nsteps - 1)
    approach = 0.155 * _smoothstep(t)
    contact_shape = _contact_shape(x, y, length=length, width=width)
    mode_shape = np.sin(np.pi * x / length) * np.cos(np.pi * y / width)

    is_bottom = node_body == 0
    is_top = ~is_bottom
    interface_weight = np.where(is_bottom, local_s, 1.0 - local_s)
    support_weight = np.where(x < 0.07 * length, 1.0, 0.0)
    cb_retained = ((interface_weight > 0.92) | (support_weight > 0.5)).astype(np.float32)

    bottom_w = -0.010 * _smoothstep(t) * contact_shape * np.where(is_bottom, local_s, 0.0)
    top_w = (-approach - 0.024 * _smoothstep(t) * contact_shape) * np.where(is_top, 1.0 - 0.18 * local_s, 0.0)
    lateral = 0.012 * math.sin(np.pi * t) * mode_shape * interface_weight
    displacement = np.zeros((coords.shape[0], 3), dtype=np.float32)
    displacement[:, 0] = np.where(is_top, 0.25 * lateral, -0.08 * lateral)
    displacement[:, 1] = np.where(is_top, -0.10 * lateral, 0.04 * lateral)
    displacement[:, 2] = bottom_w + top_w

    nominal_gap = clearance + 0.012 * np.cos(2.0 * np.pi * x / length)
    gap = nominal_gap - approach - 0.014 * _smoothstep(t) * contact_shape
    pressure = 2800.0 * np.maximum(-gap, 0.0) * contact_shape
    pressure_vis = pressure * interface_weight
    active_contact = ((pressure > 1.0) & (interface_weight > 0.90)).astype(np.float32)
    mortar_weight = interface_weight * contact_shape

    return {
        "displacement": displacement,
        "gap": gap.astype(np.float32),
        "contact_pressure": pressure_vis.astype(np.float32),
        "active_contact": active_contact.astype(np.float32),
        "cb_retained": cb_retained.astype(np.float32),
        "mortar_weight": mortar_weight.astype(np.float32),
        "reduced_mode_1": mode_shape.astype(np.float32),
    }


def _write_pvd(path: Path, files: list[Path], times: list[float]) -> None:
    with path.open("w", encoding="ascii") as io:
        io.write('<?xml version="1.0"?>\n')
        io.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        io.write("  <Collection>\n")
        for file, time in zip(files, times):
            io.write(f'    <DataSet timestep="{time:.8e}" group="" part="0" file="{file.name}"/>\n')
        io.write("  </Collection>\n")
        io.write("</VTKFile>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("result/tutorials/curved_surface_contact_vtu_demo"))
    parser.add_argument("--nsteps", type=int, default=18)
    parser.add_argument("--nx", type=int, default=28)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=2)
    args = parser.parse_args()

    length = 4.0
    width = 1.6
    thickness = 0.22
    clearance = 0.115

    mesh, body_id, node_body, local_s = _build_curved_pair(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        length=length,
        width=width,
        thickness=thickness,
        clearance=clearance,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("curved_contact_*.vtu"):
        old.unlink()
    old_pvd = args.out_dir / "curved_contact_series.pvd"
    if old_pvd.exists():
        old_pvd.unlink()

    vtus: list[Path] = []
    times: list[float] = []
    for step in range(args.nsteps):
        fields = _point_fields(
            mesh,
            node_body,
            local_s,
            step=step,
            nsteps=args.nsteps,
            length=length,
            width=width,
            thickness=thickness,
            clearance=clearance,
        )
        vtu = args.out_dir / f"curved_contact_{step:04d}.vtu"
        ff.write_vtu(
            mesh,
            str(vtu),
            point_data=fields,
            cell_data={"body_id": body_id},
        )
        vtus.append(vtu)
        times.append(0.0 if args.nsteps <= 1 else step / float(args.nsteps - 1))

    pvd = args.out_dir / "curved_contact_series.pvd"
    _write_pvd(pvd, vtus, times)
    print(f"VTU series written: {pvd}")
    print("ParaView: open the .pvd, color by contact_pressure or active_contact, then Warp By Vector(displacement).")


if __name__ == "__main__":
    main()

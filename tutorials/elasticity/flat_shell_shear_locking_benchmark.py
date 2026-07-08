#!/usr/bin/env python3
"""
Compare Q4 Reissner-Mindlin shell shear modes on a thin cantilever benchmark.

The benchmark mirrors ``mindlin_plate_shear_locking_benchmark.py`` but uses the
6-DOF shell formulation. It can run on flat 2D coordinates or on a tilted planar
3D shell. For tilted cases, the tip load and reported displacement are measured
in the shell local transverse direction, so the result should match the flat
case up to numerical roundoff.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def euler_bernoulli_tip_deflection(*, load: float, length: float, E: float, width: float, thickness: float) -> float:
    I = float(width) * float(thickness) ** 3 / 12.0
    return float(load) * float(length) ** 3 / (3.0 * float(E) * I)


def _shell_coords(coords2: np.ndarray, tilt_z: float) -> np.ndarray:
    if float(tilt_z) == 0.0:
        return coords2
    angle = np.arctan(float(tilt_z))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.column_stack([c * coords2[:, 0], coords2[:, 1], s * coords2[:, 0]])


def solve_shell_tip(
    *,
    shear_mode: str,
    nx: int,
    ny: int,
    length: float,
    width: float,
    thickness: float,
    E: float,
    nu: float,
    tip_load: float,
    tilt_z: float = 0.0,
) -> tuple[float, dict[str, object]]:
    coords2, conn = ff.structured_plate_grid(nx=nx, ny=ny, length_x=length, length_y=width)
    coords = _shell_coords(coords2, tilt_z)
    section = ff.ShellSection(E=E, nu=nu, thickness=thickness, shear_mode=shear_mode)
    K = ff.assemble_shell_stiffness(coords, conn, section, format="csr")

    R, _local = ff.shell_element_frame(coords[conn[0]])
    local_normal_global = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)

    left_nodes = np.flatnonzero(np.isclose(coords2[:, 0], 0.0))
    right_nodes = np.flatnonzero(np.isclose(coords2[:, 0], length))
    forces = np.tile(local_normal_global[None, :] * (float(tip_load) / right_nodes.size), (right_nodes.size, 1))
    F = ff.assemble_flat_shell_point_loads(coords.shape[0], right_nodes, forces=forces)
    fixed = ff.shell_node_dofs(left_nodes)
    u, _info = ff.LinearSolver(method="spsolve").solve(K, F, dirichlet=ff.DirichletBC(fixed, 0.0), dirichlet_mode="condense")

    u_nodes = np.asarray(u).reshape(-1, 6)
    local_u = (R @ u_nodes[:, :3].T).T
    tip_node = right_nodes[np.argmin(np.abs(coords2[right_nodes, 1] - 0.5 * width))]
    return float(local_u[tip_node, 2]), {
        "coords": coords,
        "conn": conn,
        "u": np.asarray(u),
        "right_nodes": right_nodes,
        "local_frame": R,
    }


def run_benchmark(
    *,
    nx: int = 12,
    ny: int = 2,
    length: float = 2.0,
    width: float = 0.3,
    thickness: float = 0.02,
    E: float = 210.0e9,
    nu: float = 0.3,
    tip_load: float = -100.0,
    tilt_z: float = 0.0,
    modes: tuple[str, ...] = ("full", "reduced", "mitc4"),
) -> dict[str, object]:
    theory = euler_bernoulli_tip_deflection(load=tip_load, length=length, E=E, width=width, thickness=thickness)
    mode_results: dict[str, dict[str, object]] = {}
    for mode in modes:
        tip, data = solve_shell_tip(
            shear_mode=mode,
            nx=nx,
            ny=ny,
            length=length,
            width=width,
            thickness=thickness,
            E=E,
            nu=nu,
            tip_load=tip_load,
            tilt_z=tilt_z,
        )
        mode_results[mode] = {
            "tip": tip,
            "rel_error": abs(tip - theory) / abs(theory),
            "data": data,
        }
    return {
        "theory": theory,
        "modes": mode_results,
        "length_to_thickness": float(length) / float(thickness),
        "tilt_z": float(tilt_z),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare Q4 Reissner-Mindlin shell shear modes on a thin cantilever.")
    p.add_argument("--nx", type=int, default=12)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--length", type=float, default=2.0)
    p.add_argument("--width", type=float, default=0.3)
    p.add_argument("--thickness", type=float, default=0.02)
    p.add_argument("--E", type=float, default=210.0e9)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--tip-load", type=float, default=-100.0)
    p.add_argument("--tilt-z", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    result = run_benchmark(
        nx=args.nx,
        ny=args.ny,
        length=args.length,
        width=args.width,
        thickness=args.thickness,
        E=args.E,
        nu=args.nu,
        tip_load=args.tip_load,
        tilt_z=args.tilt_z,
    )
    print(f"L/t:                  {result['length_to_thickness']:.1f}")
    print(f"tilt_z:               {result['tilt_z']:.6g}")
    print(f"EB theory tip un:     {result['theory']:.6e}")
    for mode, mode_result in result["modes"].items():
        print(f"{mode:>7s} tip un:       {mode_result['tip']:.6e}  rel.err={mode_result['rel_error']:.3e}")
    print("note: displacements are reported in the shell local transverse direction.")


if __name__ == "__main__":
    main()

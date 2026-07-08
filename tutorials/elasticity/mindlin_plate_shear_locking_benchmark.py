#!/usr/bin/env python3
"""
Compare Q4 Mindlin plate shear modes on a thin cantilever benchmark.

The reference is the Euler-Bernoulli narrow-beam estimate

    w_tip = P L^3 / (3 E I),   I = width * thickness^3 / 12.

This is a locking sanity check, not a plate-theory exact solution. Full 2x2
transverse shear integration is expected to be too stiff on a thin coarse mesh,
while selective reduced integration and the MITC4-style assumed shear option
should stay near the bending-dominated estimate.
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


def solve_plate_tip(
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
) -> tuple[float, dict[str, object]]:
    coords, conn = ff.structured_plate_grid(nx=nx, ny=ny, length_x=length, length_y=width)
    section = ff.PlateSection(E=E, nu=nu, thickness=thickness, shear_mode=shear_mode)
    K = ff.assemble_mindlin_plate_stiffness(coords, conn, section, format="csr")

    left_nodes = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    right_nodes = np.flatnonzero(np.isclose(coords[:, 0], length))
    F = ff.assemble_mindlin_plate_point_loads(
        coords.shape[0],
        right_nodes,
        forces=np.full(right_nodes.size, float(tip_load) / right_nodes.size),
    )
    fixed = ff.plate_node_dofs(left_nodes)
    u, _info = ff.LinearSolver(method="spsolve").solve(K, F, dirichlet=ff.DirichletBC(fixed, 0.0), dirichlet_mode="condense")

    w = np.asarray(u)[0::3]
    tip_node = right_nodes[np.argmin(np.abs(coords[right_nodes, 1] - 0.5 * width))]
    return float(w[tip_node]), {"coords": coords, "conn": conn, "u": np.asarray(u), "right_nodes": right_nodes}


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
    modes: tuple[str, ...] = ("full", "reduced", "mitc4"),
) -> dict[str, object]:
    theory = euler_bernoulli_tip_deflection(load=tip_load, length=length, E=E, width=width, thickness=thickness)
    mode_results: dict[str, dict[str, object]] = {}
    for mode in modes:
        tip, data = solve_plate_tip(
            shear_mode=mode,
            nx=nx,
            ny=ny,
            length=length,
            width=width,
            thickness=thickness,
            E=E,
            nu=nu,
            tip_load=tip_load,
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
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare Q4 Mindlin plate shear modes on a thin cantilever.")
    p.add_argument("--nx", type=int, default=12)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--length", type=float, default=2.0)
    p.add_argument("--width", type=float, default=0.3)
    p.add_argument("--thickness", type=float, default=0.02)
    p.add_argument("--E", type=float, default=210.0e9)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--tip-load", type=float, default=-100.0)
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
    )
    print(f"L/t:                  {result['length_to_thickness']:.1f}")
    print(f"EB theory tip w:      {result['theory']:.6e}")
    for mode, mode_result in result["modes"].items():
        print(f"{mode:>7s} tip w:        {mode_result['tip']:.6e}  rel.err={mode_result['rel_error']:.3e}")
    print("note: full shear integration is intentionally reported to expose thin-plate shear locking.")


if __name__ == "__main__":
    main()

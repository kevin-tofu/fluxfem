from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff

jax.config.update("jax_enable_x64", True)


def _extension_dirichlet(space, axial_strain: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(space.mesh.coords)
    dofs = np.arange(space.n_dofs, dtype=int)
    vals = np.zeros(space.n_dofs, dtype=float)
    for node_id, (x, _y, _z) in enumerate(coords):
        vals[3 * node_id + 0] = axial_strain * x
    return dofs, vals


def run(args: argparse.Namespace) -> dict[str, Path]:
    mesh = ff.StructuredHexBox(
        nx=args.nx,
        ny=1,
        nz=1,
        lx=args.length,
        ly=args.width,
        lz=args.width,
    ).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    material = ff.J2Plasticity(
        E=args.young,
        nu=args.nu,
        yield_stress=args.yield_stress,
        hardening_modulus=args.hardening,
    )
    dirichlet = _extension_dirichlet(space, args.axial_strain)

    u, state, history = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        dirichlet=dirichlet,
        n_steps=args.steps,
    )
    cell_data = ff.make_j2_cell_data(space, u, state, material)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vtu_path = out_dir / "j2_uniaxial_tension.vtu"
    csv_path = out_dir / "j2_uniaxial_tension_history.csv"
    ff.write_j2_vtu(mesh, space, u, state, material, str(vtu_path), deformed_scale=args.deformed_scale)

    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "load_factor",
                "converged",
                "committed",
                "max_p_eq",
                "avg_sigma_xx",
                "avg_sigma_vm",
            ],
        )
        writer.writeheader()
        for i, step in enumerate(history, start=1):
            step_cell = ff.make_j2_cell_data(space, step.u, step.state, material)
            writer.writerow(
                {
                    "step": i,
                    "load_factor": step.load_factor,
                    "converged": int(step.converged),
                    "committed": int(step.committed),
                    "max_p_eq": float(step.max_equivalent_plastic_strain),
                    "avg_sigma_xx": float(np.mean(step_cell["j2_sigma_xx"])),
                    "avg_sigma_vm": float(np.mean(step_cell["j2_sigma_vm"])),
                }
            )

    print(f"wrote {vtu_path}")
    print(f"wrote {csv_path}")
    print(f"final max p_eq: {float(np.max(cell_data['j2_p_eq'])):.6e}")
    print(f"final avg sigma_xx: {float(np.mean(cell_data['j2_sigma_xx'])):.6e}")
    return {"vtu": vtu_path, "csv": csv_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-strain J2 uniaxial-tension VTU demo.")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=0.2)
    parser.add_argument("--young", type=float, default=210_000.0)
    parser.add_argument("--nu", type=float, default=0.30)
    parser.add_argument("--yield-stress", type=float, default=50.0)
    parser.add_argument("--hardening", type=float, default=100.0)
    parser.add_argument("--axial-strain", type=float, default=5.0e-3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--deformed-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="tutorials/nonlinear/results")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

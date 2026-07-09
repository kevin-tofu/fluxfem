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
import fluxfem.helpers_wf as h_wf

jax.config.update("jax_enable_x64", True)


def _extension_dirichlet(space, axial_strain: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(space.mesh.coords)
    dofs = np.arange(space.n_dofs, dtype=int)
    vals = np.zeros(space.n_dofs, dtype=float)
    for node_id, (x, _y, _z) in enumerate(coords):
        vals[3 * node_id + 0] = axial_strain * x
    return dofs, vals


def _mixed_extension_dirichlet(space, axial_strain: float) -> tuple[np.ndarray, np.ndarray]:
    mesh = space.mesh
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())
    left = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1.0e-8),
        components="xyz",
    ).dofs
    right_x = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmax, atol=1.0e-8),
        components=[0],
        dof_per_node=3,
    ).dofs
    dofs = np.concatenate([left, right_x])
    vals = np.concatenate([np.zeros(len(left), dtype=float), np.full(len(right_x), axial_strain * xmax, dtype=float)])
    return dofs, vals


def _left_clamp(space) -> tuple[np.ndarray, np.ndarray]:
    mesh = space.mesh
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    dofs = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1.0e-8),
        components="xyz",
    ).dofs
    return dofs, np.zeros(len(dofs), dtype=float)


def _right_face_traction(space, traction_x: float) -> jnp.ndarray:
    mesh = space.mesh
    coords = np.asarray(mesh.coords)
    xmax = float(coords[:, 0].max())
    facets = np.asarray(mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1.0e-8)))
    surface = ff.make_surface_from_facets(coords, facets)
    form = ff.LinearForm.surface(lambda v, p: (v | p) * h_wf.ds())
    return jnp.asarray(
        surface.assemble_linear_form_on_space(space, form, params=np.array([traction_x, 0.0, 0.0], dtype=float)),
        dtype=jnp.float64,
    )


def _boundary_conditions(space, args: argparse.Namespace):
    if args.bc_mode == "full":
        return _extension_dirichlet(space, args.axial_strain), None
    if args.bc_mode == "mixed":
        return _mixed_extension_dirichlet(space, args.axial_strain), None
    if args.bc_mode == "force":
        return _left_clamp(space), _right_face_traction(space, args.traction)
    raise ValueError(f"unknown bc mode: {args.bc_mode}")


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
    dirichlet, base_external = _boundary_conditions(space, args)

    u, state, history = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        dirichlet=dirichlet,
        base_external_vector=base_external,
        n_steps=args.steps,
        tol=args.tol,
        atol=args.atol,
        maxiter=args.maxiter,
        line_search=args.line_search,
    )
    cell_data = ff.make_j2_cell_data(space, u, state, material)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"j2_uniaxial_tension_{args.bc_mode}"
    vtu_path = out_dir / f"{stem}.vtu"
    csv_path = out_dir / f"{stem}_history.csv"
    iter_csv_path = out_dir / f"{stem}_newton.csv"
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

    with iter_csv_path.open("w", newline="", encoding="ascii") as f:
        fieldnames = [
            "step",
            "load_factor",
            "iter",
            "res_inf",
            "res_two",
            "rel_residual",
            "alpha",
            "step_norm",
            "linear_iters",
            "linear_converged",
            "linear_residual",
            "nan_detected",
            "initial_residual_time",
            "initial_jacobian_time",
            "rhs_time",
            "linear_time",
            "eval_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, step in enumerate(history, start=1):
            for rec in step.iter_history:
                writer.writerow(
                    {
                        "step": i,
                        "load_factor": step.load_factor,
                        "iter": rec.get("iter"),
                        "res_inf": rec.get("res_inf"),
                        "res_two": rec.get("res_two"),
                        "rel_residual": rec.get("rel_residual"),
                        "alpha": rec.get("alpha"),
                        "step_norm": rec.get("step_norm"),
                        "linear_iters": rec.get("linear_iters"),
                        "linear_converged": rec.get("linear_converged"),
                        "linear_residual": rec.get("linear_residual"),
                        "nan_detected": rec.get("nan_detected"),
                        "initial_residual_time": rec.get("initial_residual_time"),
                        "initial_jacobian_time": rec.get("initial_jacobian_time"),
                        "rhs_time": rec.get("rhs_time"),
                        "linear_time": rec.get("linear_wall_time"),
                        "eval_time": rec.get("eval_time"),
                    }
                )

    print(f"wrote {vtu_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {iter_csv_path}")
    print(f"bc mode: {args.bc_mode}")
    print(f"final max p_eq: {float(np.max(cell_data['j2_p_eq'])):.6e}")
    print(f"final avg sigma_xx: {float(np.mean(cell_data['j2_sigma_xx'])):.6e}")
    return {"vtu": vtu_path, "csv": csv_path, "newton_csv": iter_csv_path}


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
    parser.add_argument("--traction", type=float, default=10.0)
    parser.add_argument("--bc-mode", choices=("full", "mixed", "force"), default="full")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-7)
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--deformed-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="tutorials/nonlinear/results")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

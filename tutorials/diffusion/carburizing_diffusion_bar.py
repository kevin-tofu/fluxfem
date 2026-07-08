#!/usr/bin/env python3
"""
Transient carburizing diffusion in a 1D-like steel bar.

The model is Fickian carbon diffusion with a fixed surface carbon potential:

  dc/dt = D * Laplacian(c)  in Omega
  c = c_surface             on x = 0
  -D grad(c) . n = 0        on the other faces

For short times before the diffusion front reaches x = L, the profile is close
to the semi-infinite analytical solution

  c(x, t) = c_initial + (c_surface - c_initial) * erfc(x / (2 sqrt(D t))).

This is a minimal heat-treatment example: it uses the same diffusion operator
as heat conduction, but interprets the scalar field as carbon content.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff  # noqa: E402
import fluxfem.helpers_wf as h_wf  # noqa: E402


def default_output_path(filename: str) -> str:
    return os.path.join("tutorials", "diffusion", "results", filename)


def parse_args():
    p = argparse.ArgumentParser(description="Transient carburizing diffusion in a bar.")
    p.add_argument("--nx", type=int, default=80, help="Elements through the case depth.")
    p.add_argument("--ny", type=int, default=1, help="Elements in y.")
    p.add_argument("--nz", type=int, default=1, help="Elements in z.")
    p.add_argument("--lx", type=float, default=1.0, help="Bar depth.")
    p.add_argument("--ly", type=float, default=0.04, help="Bar width.")
    p.add_argument("--lz", type=float, default=0.04, help="Bar thickness.")
    p.add_argument("--intorder", type=int, default=2, help="Quadrature order.")
    p.add_argument("--diffusivity", type=float, default=1.0e-3, help="Carbon diffusivity D.")
    p.add_argument("--dt", type=float, default=0.1, help="Time step.")
    p.add_argument("--steps", type=int, default=120, help="Number of implicit Euler steps.")
    p.add_argument("--c-initial", type=float, default=0.20, help="Initial carbon content.")
    p.add_argument("--c-surface", type=float, default=0.90, help="Surface carbon content.")
    p.add_argument("--case-threshold", type=float, default=0.40, help="Case-depth concentration threshold.")
    p.add_argument(
        "--output-vtu",
        type=str,
        default=default_output_path("carburizing_diffusion_bar.vtu"),
        help="Final VTU output path.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=default_output_path("carburizing_diffusion_profile.csv"),
        help="Final x-profile CSV output path.",
    )
    return p.parse_args()


def average_profile_by_x(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_unique = np.unique(coords[:, 0])
    c_avg = np.empty_like(x_unique)
    for i, x in enumerate(x_unique):
        c_avg[i] = float(np.mean(values[np.isclose(coords[:, 0], x, atol=1e-12)]))
    return x_unique, c_avg


def semi_infinite_profile(
    x: np.ndarray,
    *,
    time: float,
    diffusivity: float,
    c_initial: float,
    c_surface: float,
) -> np.ndarray:
    if time <= 0.0:
        out = np.full_like(x, c_initial, dtype=float)
        out[np.isclose(x, 0.0)] = c_surface
        return out
    denom = 2.0 * math.sqrt(diffusivity * time)
    return np.array(
        [c_initial + (c_surface - c_initial) * math.erfc(float(xi) / denom) for xi in x],
        dtype=float,
    )


def estimate_case_depth(x: np.ndarray, c: np.ndarray, threshold: float) -> float | None:
    if threshold >= c[0]:
        return 0.0
    below = np.flatnonzero(c < threshold)
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(x[0])
    x0, x1 = float(x[i - 1]), float(x[i])
    c0, c1 = float(c[i - 1]), float(c[i])
    if c1 == c0:
        return x1
    return x0 + (threshold - c0) * (x1 - x0) / (c1 - c0)


def write_profile_csv(
    path: str,
    x: np.ndarray,
    c_fem: np.ndarray,
    c_erfc: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="ascii") as fp:
        writer = csv.writer(fp)
        writer.writerow(["x", "carbon_fem", "carbon_semi_infinite_erfc"])
        writer.writerows(zip(x, c_fem, c_erfc))


def run_carburizing(
    *,
    nx: int = 80,
    ny: int = 1,
    nz: int = 1,
    lx: float = 1.0,
    ly: float = 0.04,
    lz: float = 0.04,
    intorder: int = 2,
    diffusivity: float = 1.0e-3,
    dt: float = 0.1,
    steps: int = 120,
    c_initial: float = 0.20,
    c_surface: float = 0.90,
    case_threshold: float = 0.40,
    output_vtu: str | None = None,
    output_csv: str | None = None,
) -> dict[str, object]:
    mesh = ff.StructuredHexBox(
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
    ).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    mass = np.asarray(space.assemble_mass_matrix().to_dense(), dtype=float)
    stiffness_form = ff.BilinearForm.volume(
        lambda u, v, p: h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
    )
    stiffness = np.asarray(
        ff.assemble_bilinear_form(
            ff.BilinearSpaces(test=V, trial=U),
            stiffness_form,
            ff.Params(),
        ).to_dense(),
        dtype=float,
    )

    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    surface_bc = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-10),
        values=c_surface,
        components=[0],
        dof_per_node=1,
    )

    c = np.full(space.n_dofs, c_initial, dtype=float)
    c[surface_bc.dofs] = c_surface

    lhs = mass + dt * diffusivity * stiffness
    solver = ff.LinearSolver(method="spsolve")
    for _step in range(steps):
        rhs = mass @ c
        c, _info = solver.solve(
            lhs,
            rhs,
            dirichlet=surface_bc,
            dirichlet_mode="condense",
        )
        c = np.asarray(c, dtype=float)

    final_time = steps * dt
    x_profile, c_profile = average_profile_by_x(coords, c)
    c_erfc = semi_infinite_profile(
        x_profile - xmin,
        time=final_time,
        diffusivity=diffusivity,
        c_initial=c_initial,
        c_surface=c_surface,
    )
    l2_err = float(np.linalg.norm(c_profile - c_erfc) / math.sqrt(c_profile.size))
    case_depth = estimate_case_depth(x_profile - xmin, c_profile, case_threshold)

    if output_vtu:
        os.makedirs(os.path.dirname(output_vtu) or ".", exist_ok=True)
        ff.write_vtu(mesh, output_vtu, point_data={"carbon": c})
    if output_csv:
        write_profile_csv(output_csv, x_profile, c_profile, c_erfc)

    return {
        "mesh": mesh,
        "space": space,
        "carbon": c,
        "x_profile": x_profile,
        "carbon_profile": c_profile,
        "semi_infinite_profile": c_erfc,
        "time": final_time,
        "diffusivity": diffusivity,
        "c_initial": c_initial,
        "c_surface": c_surface,
        "case_threshold": case_threshold,
        "case_depth": case_depth,
        "rms_error": l2_err,
        "output_vtu": output_vtu,
        "output_csv": output_csv,
    }


def main():
    args = parse_args()
    result = run_carburizing(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        intorder=args.intorder,
        diffusivity=args.diffusivity,
        dt=args.dt,
        steps=args.steps,
        c_initial=args.c_initial,
        c_surface=args.c_surface,
        case_threshold=args.case_threshold,
        output_vtu=args.output_vtu,
        output_csv=args.output_csv,
    )

    case_msg = "not reached" if result["case_depth"] is None else f"{result['case_depth']:.6e}"
    print(
        f"carburizing solved: dofs={result['space'].n_dofs}, time={result['time']:.6e}, "
        f"D={result['diffusivity']:.6e}"
    )
    print(f"surface={result['c_surface']:.6e}, core={result['carbon_profile'][-1]:.6e}")
    print(f"case depth at c={result['case_threshold']:.6e}: {case_msg}")
    print(f"semi-infinite profile RMS error={result['rms_error']:.6e}")
    print(f"wrote VTU: {args.output_vtu}")
    print(f"wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare shell and solid cantilever deflection against an Euler-Bernoulli beam estimate.

This is a sanity benchmark, not an exact shell/solid theorem. A narrow plate is
loaded by a total transverse tip force. The reference estimate is

    w_tip = P L^3 / (3 E I),   I = width * thickness^3 / 12.

The shell model uses Q4 Reissner-Mindlin shell elements. The solid model uses a
structured hex mesh of the same dimensions with the tip force distributed over
the free-end face.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import scipy.sparse as sp
import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def _as_csr(matrix):
    if sp.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "to_csr"):
        return matrix.to_csr()
    if hasattr(matrix, "toarray"):
        return sp.csr_matrix(matrix.toarray())
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def euler_bernoulli_tip_deflection(*, load: float, length: float, E: float, width: float, thickness: float) -> float:
    I = float(width) * float(thickness) ** 3 / 12.0
    return float(load) * float(length) ** 3 / (3.0 * float(E) * I)


def solve_shell_tip(
    *,
    nx: int,
    ny: int,
    length: float,
    width: float,
    thickness: float,
    E: float,
    nu: float,
    tip_load_z: float,
) -> tuple[float, dict[str, object]]:
    coords, conn = ff.structured_plate_grid(nx=nx, ny=ny, length_x=length, length_y=width)
    section = ff.ShellSection(E=E, nu=nu, thickness=thickness)
    K = ff.assemble_shell_stiffness(coords, conn, section, format="csr")

    left_nodes = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    right_nodes = np.flatnonzero(np.isclose(coords[:, 0], length))
    forces = np.zeros((right_nodes.size, 3), dtype=float)
    forces[:, 2] = float(tip_load_z) / right_nodes.size
    F = ff.assemble_flat_shell_point_loads(coords.shape[0], right_nodes, forces=forces)
    fixed = ff.shell_node_dofs(left_nodes)
    u, _info = ff.LinearSolver(method="spsolve").solve(K, F, dirichlet=ff.DirichletBC(fixed, 0.0), dirichlet_mode="condense")

    uz = np.asarray(u)[2::6]
    tip_node = right_nodes[np.argmin(np.abs(coords[right_nodes, 1] - 0.5 * width))]
    return float(uz[tip_node]), {"coords": coords, "conn": conn, "u": np.asarray(u), "right_nodes": right_nodes}


def solve_solid_tip(
    *,
    nx: int,
    ny: int,
    nz: int,
    length: float,
    width: float,
    thickness: float,
    E: float,
    nu: float,
    tip_load_z: float,
) -> tuple[float, dict[str, object]]:
    mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=length, ly=width, lz=thickness).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    K = _as_csr(space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(E, nu)))
    coords = np.asarray(mesh.coords, dtype=float)

    right_nodes = np.flatnonzero(np.isclose(coords[:, 0], length, atol=1.0e-10))
    F = np.zeros((space.n_dofs,), dtype=float)
    for node in right_nodes:
        F[3 * int(node) + 2] += float(tip_load_z) / right_nodes.size

    fixed = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0, atol=1.0e-10),
        components="xyz",
        dof_per_node=3,
    )
    u, _info = ff.LinearSolver(method="spsolve").solve(K, F, dirichlet=ff.DirichletBC(fixed, 0.0), dirichlet_mode="condense")

    u_nodes = np.asarray(u).reshape(-1, 3)
    target = np.array([length, 0.5 * width, 0.5 * thickness], dtype=float)
    tip_node = right_nodes[np.argmin(np.linalg.norm(coords[right_nodes] - target[None, :], axis=1))]
    return float(u_nodes[tip_node, 2]), {"mesh": mesh, "space": space, "u": np.asarray(u), "right_nodes": right_nodes}


def run_benchmark(
    *,
    shell_nx: int = 16,
    shell_ny: int = 2,
    solid_nx: int = 16,
    solid_ny: int = 2,
    solid_nz: int = 2,
    length: float = 2.0,
    width: float = 0.3,
    thickness: float = 0.06,
    E: float = 210.0e9,
    nu: float = 0.3,
    tip_load_z: float = -1000.0,
) -> dict[str, object]:
    theory = euler_bernoulli_tip_deflection(load=tip_load_z, length=length, E=E, width=width, thickness=thickness)
    shell_tip, shell = solve_shell_tip(
        nx=shell_nx,
        ny=shell_ny,
        length=length,
        width=width,
        thickness=thickness,
        E=E,
        nu=nu,
        tip_load_z=tip_load_z,
    )
    solid_tip, solid = solve_solid_tip(
        nx=solid_nx,
        ny=solid_ny,
        nz=solid_nz,
        length=length,
        width=width,
        thickness=thickness,
        E=E,
        nu=nu,
        tip_load_z=tip_load_z,
    )
    return {
        "theory": theory,
        "shell_tip": shell_tip,
        "solid_tip": solid_tip,
        "shell_rel_error": abs(shell_tip - theory) / abs(theory),
        "solid_rel_error": abs(solid_tip - theory) / abs(theory),
        "shell_to_solid_rel_diff": abs(shell_tip - solid_tip) / max(abs(solid_tip), 1.0e-30),
        "shell": shell,
        "solid": solid,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare shell and solid cantilever against an Euler-Bernoulli estimate.")
    p.add_argument("--shell-nx", type=int, default=16)
    p.add_argument("--shell-ny", type=int, default=2)
    p.add_argument("--solid-nx", type=int, default=16)
    p.add_argument("--solid-ny", type=int, default=2)
    p.add_argument("--solid-nz", type=int, default=2)
    p.add_argument("--length", type=float, default=2.0)
    p.add_argument("--width", type=float, default=0.3)
    p.add_argument("--thickness", type=float, default=0.06)
    p.add_argument("--E", type=float, default=210.0e9)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--tip-load-z", type=float, default=-1000.0)
    return p.parse_args()


def main():
    args = parse_args()
    result = run_benchmark(
        shell_nx=args.shell_nx,
        shell_ny=args.shell_ny,
        solid_nx=args.solid_nx,
        solid_ny=args.solid_ny,
        solid_nz=args.solid_nz,
        length=args.length,
        width=args.width,
        thickness=args.thickness,
        E=args.E,
        nu=args.nu,
        tip_load_z=args.tip_load_z,
    )
    print(f"EB theory tip uz:      {result['theory']:.6e}")
    print(f"shell tip uz:          {result['shell_tip']:.6e}  rel.err={result['shell_rel_error']:.3e}")
    print(f"solid tip uz:          {result['solid_tip']:.6e}  rel.err={result['solid_rel_error']:.3e}")
    print(f"shell-solid rel.diff:  {result['shell_to_solid_rel_diff']:.3e}")
    print("note: low-order solid hex bending is intentionally reported separately; coarse meshes are typically too stiff.")


if __name__ == "__main__":
    main()

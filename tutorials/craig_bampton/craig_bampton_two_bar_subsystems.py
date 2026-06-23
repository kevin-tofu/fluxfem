#!/usr/bin/env python
"""Two larger FE subsystems coupled after Craig-Bampton reduction.

This tutorial builds two scalar hex bars as separate subsystems.  Each subsystem
has its own sparse stiffness/mass matrices and its own Craig-Bampton basis.  The
right face of part A is tied to the left face of part B through retained
interface DOFs, then the reduced coupled system is compared with the full-order
KKT reference solve.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_two_bar_subsystems.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx-a", type=int, default=24, help="Number of hex elements in subsystem A.")
    parser.add_argument("--nx-b", type=int, default=24, help="Number of hex elements in subsystem B.")
    parser.add_argument("--n-modes", type=int, default=6, help="Fixed-interface modes per subsystem.")
    return parser.parse_args()


def make_bar(nx: int, *, stiffness_scale: float):
    mesh = ff.StructuredHexBox(nx=nx, ny=1, nz=1, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    stiffness = space.assemble(ff.diffusion_form, params=stiffness_scale)
    mass = space.assemble_mass_matrix()
    coords = np.asarray(mesh.coords, dtype=float)
    return mesh, coords, stiffness, mass


def face_nodes(coords: np.ndarray, x_value: float) -> np.ndarray:
    nodes = np.flatnonzero(np.isclose(coords[:, 0], x_value, atol=1.0e-12))
    order = np.lexsort((coords[nodes, 2], coords[nodes, 1]))
    return nodes[order].astype(np.int32)


def full_reference_solution(k_a, k_b, f_a: np.ndarray, f_b: np.ndarray, fixed_a: np.ndarray, interface_a: np.ndarray, interface_b: np.ndarray) -> np.ndarray:
    k_a_dense = np.asarray(k_a.to_dense(), dtype=float)
    k_b_dense = np.asarray(k_b.to_dense(), dtype=float)
    n_a = k_a_dense.shape[0]
    n_b = k_b_dense.shape[0]
    stiffness = np.block(
        [
            [k_a_dense, np.zeros((n_a, n_b), dtype=float)],
            [np.zeros((n_b, n_a), dtype=float), k_b_dense],
        ]
    )
    force = np.concatenate([f_a, f_b])
    c = np.zeros((interface_a.size, n_a + n_b), dtype=float)
    for row, (a_dof, b_dof) in enumerate(zip(interface_a, interface_b)):
        c[row, int(a_dof)] = 1.0
        c[row, n_a + int(b_dof)] = -1.0
    fixed = jnp.asarray(fixed_a, dtype=jnp.int32)
    return np.asarray(ff.LinearConstraintSystem(jnp.asarray(c)).solve(jnp.asarray(stiffness), jnp.asarray(force), fixed_dofs=fixed))


def main() -> None:
    args = parse_args()

    mesh_a, coords_a, k_a, m_a = make_bar(args.nx_a, stiffness_scale=1.0)
    mesh_b, coords_b, k_b, m_b = make_bar(args.nx_b, stiffness_scale=0.7)
    n_a = int(mesh_a.n_nodes)
    n_b = int(mesh_b.n_nodes)

    left_a = face_nodes(coords_a, float(coords_a[:, 0].min()))
    right_a = face_nodes(coords_a, float(coords_a[:, 0].max()))
    left_b = face_nodes(coords_b, float(coords_b[:, 0].min()))
    right_b = face_nodes(coords_b, float(coords_b[:, 0].max()))

    f_a = np.zeros((n_a,), dtype=float)
    f_b = np.zeros((n_b,), dtype=float)
    f_b[right_b] = 1.0 / right_b.size

    builder = ff.ReducedCoupledSystemBuilder.from_structural(
        "part_a",
        k_a,
        jnp.asarray(f_a),
        mass=m_a,
        value_dim=1,
        n_nodes=n_a,
    )
    builder.register_structural(
        "part_b",
        k_b,
        jnp.asarray(f_b),
        mass=m_b,
        value_dim=1,
        n_nodes=n_b,
    )
    builder.retain_dof_group("part_a", "support", left_a)
    builder.retain_dof_group("part_a", "interface", right_a)
    builder.retain_dof_group("part_b", "interface", left_b)
    builder.retain_dof_group("part_b", "load", right_b)

    cb_a = builder.reduce_field("part_a", retained_groups=["support", "interface"], n_modes=args.n_modes)
    cb_b = builder.reduce_field("part_b", retained_groups=["interface", "load"], n_modes=args.n_modes)
    builder.tie_retained_groups("part_a:interface", "part_b:interface")
    system = builder.build()

    fixed_reduced = system.reduced_dofs_from_full("part_a", left_a)
    q = system.solve(fixed_dofs=fixed_reduced)
    u_rom = np.asarray(system.expand(q), dtype=float)
    u_full = full_reference_solution(k_a, k_b, f_a, f_b, left_a, right_a, left_b)

    rel_error = np.linalg.norm(u_rom - u_full) / max(np.linalg.norm(u_full), 1.0e-30)
    interface_jump = np.linalg.norm(u_rom[right_a] - u_rom[n_a + left_b])
    load_mean = float(np.mean(u_rom[n_a + right_b]))
    mr_a = np.asarray(cb_a.project_matrix(m_a), dtype=float)
    mr_b = np.asarray(cb_b.project_matrix(m_b), dtype=float)

    print("two-subsystem Craig-Bampton ROM")
    print(f"full DOFs:             {n_a + n_b}")
    print(f"part A reduced DOFs:   {cb_a.n_reduced} / {cb_a.n_full}")
    print(f"part B reduced DOFs:   {cb_b.n_reduced} / {cb_b.n_full}")
    print(f"coupled reduced DOFs:  {system.n_dofs}")
    print(f"tie constraints:       {right_a.size}")
    print(f"Mred A/B shapes:       {mr_a.shape} / {mr_b.shape}")
    print(f"constraint norm:       {float(jnp.linalg.norm(system.constraints.residual(q))):.3e}")
    print(f"interface jump:        {interface_jump:.3e}")
    print(f"relative full error:   {rel_error:.3e}")
    print(f"mean load-face value:  {load_mean:.8e}")


if __name__ == "__main__":
    main()

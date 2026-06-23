#!/usr/bin/env python3
"""Geometric nonlinear workpiece with an RBE3-style fixture and local process force."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=0.25)
    parser.add_argument("--lz", type=float, default=0.25)
    parser.add_argument("--E", type=float, default=250.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--force", type=float, default=-0.02)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--output-vtu", type=str, default="")
    return parser.parse_args()


def _nodes_where(coords: np.ndarray, predicate) -> np.ndarray:
    mask = predicate(coords)
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int32)


def main():
    args = parse_args()
    dtype = jnp.float64

    mesh = ff.StructuredHexBox(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
    ).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    coords = np.asarray(mesh.coords, dtype=float)

    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())
    ymax = float(coords[:, 1].max())
    zmin = float(coords[:, 2].min())
    zmax = float(coords[:, 2].max())

    dirichlet = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin),
        components="xyz",
    )

    fixture_nodes = _nodes_where(
        coords,
        lambda pts: np.isclose(pts[:, 0], xmax) & np.isclose(pts[:, 2], zmin),
    )
    if fixture_nodes.size < 2:
        raise RuntimeError("fixture patch needs at least two nodes; increase ny/nz.")
    fixture_dofs = ff.vector_dofs_from_nodes(fixture_nodes, dim=3).reshape(-1, 3)[:, [1, 2]]
    fixture_patch = ff.RBE3Patch(
        jnp.asarray(fixture_dofs, dtype=jnp.int32),
        jnp.ones((fixture_nodes.size,), dtype=dtype),
    )

    tool_nodes = _nodes_where(
        coords,
        lambda pts: (
            np.isclose(pts[:, 0], xmax)
            & np.isclose(pts[:, 1], ymax)
            & np.isclose(pts[:, 2], zmax)
        ),
    )
    if tool_nodes.size != 1:
        raise RuntimeError("expected one tool node at the upper free corner.")
    tool_dof_y = int(tool_nodes[0]) * 3 + 1

    lam, mu = ff.lame_parameters(args.E, args.nu)
    problem = ff.NonlinearConstrainedProblem(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params={"lam": lam, "mu": mu},
        dirichlet=dirichlet,
        dtype=dtype,
        jacobian_pattern=ff.make_sparsity_pattern(space, with_idx=True),
    )
    problem.add_rbe3_patch_constraint(fixture_patch, rhs=jnp.zeros((2,), dtype=dtype))
    problem.add_local_force([tool_dof_y], [args.force])

    result = problem.solve(tol=args.tol, atol=args.atol, maxiter=args.maxiter)
    u = np.asarray(result.u, dtype=float)
    u_nodes = u.reshape(-1, 3)
    constraint_residual = np.asarray(problem.constraint_system().residual(result.u), dtype=float)
    tool_disp = u_nodes[int(tool_nodes[0])]
    max_disp = float(np.linalg.norm(u_nodes, axis=1).max())

    print(f"converged: {result.info.converged}")
    print(f"iterations: {result.info.iters}")
    print(f"residual_inf: {result.info.residual_norm:.6e}")
    print(f"max_displacement: {max_disp:.6e}")
    print(f"tool_node: {int(tool_nodes[0])}")
    print(f"tool_displacement: [{tool_disp[0]:.6e}, {tool_disp[1]:.6e}, {tool_disp[2]:.6e}]")
    print(f"constraint_residual_inf: {np.linalg.norm(constraint_residual, ord=np.inf):.6e}")
    print(f"multipliers: {np.asarray(result.multipliers, dtype=float)}")

    if args.output_vtu:
        out = Path(args.output_vtu)
        out.parent.mkdir(parents=True, exist_ok=True)
        point_data = ff.make_elastic_point_data(mesh, space, result.u, compute_j=True, deformed_scale=1.0)
        fixture_marker = np.zeros((coords.shape[0],), dtype=float)
        fixture_marker[fixture_nodes] = 1.0
        tool_marker = np.zeros((coords.shape[0],), dtype=float)
        tool_marker[tool_nodes] = 1.0
        dirichlet_marker = np.zeros((coords.shape[0],), dtype=float)
        dirichlet_marker[np.unique(np.asarray(dirichlet.dofs, dtype=int) // 3)] = 1.0
        process_force = np.zeros((coords.shape[0], 3), dtype=float)
        process_force[int(tool_nodes[0]), 1] = float(args.force)
        point_data.update(
            {
                "fixture_marker": fixture_marker,
                "tool_marker": tool_marker,
                "dirichlet_marker": dirichlet_marker,
                "process_force": process_force,
            }
        )
        ff.write_vtu(mesh, str(out), point_data=point_data)
        print(f"VTU written to {out}")

    if not result.info.converged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

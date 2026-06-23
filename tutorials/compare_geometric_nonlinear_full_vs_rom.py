#!/usr/bin/env python3
"""Compare geometric nonlinear full FEM and nonlinear Galerkin ROM."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


class DenseBasis:
    def __init__(self, basis):
        self.basis = jnp.asarray(basis, dtype=jnp.float64)

    @property
    def n_full(self):
        return int(self.basis.shape[0])

    @property
    def n_reduced(self):
        return int(self.basis.shape[1])

    def expand(self, q):
        return self.basis @ q

    def project_vector(self, vector):
        return self.basis.T @ jnp.asarray(vector)

    def project_matrix(self, matrix):
        dense = matrix.to_dense() if hasattr(matrix, "to_dense") else matrix
        return self.basis.T @ jnp.asarray(dense) @ self.basis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=0.25)
    parser.add_argument("--lz", type=float, default=0.25)
    parser.add_argument("--E", type=float, default=250.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--force", type=float, default=-0.001)
    parser.add_argument("--basis", choices=("complete", "tip-y"), default="complete")
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def _tool_node(coords: np.ndarray) -> int:
    xmax = float(coords[:, 0].max())
    ymax = float(coords[:, 1].max())
    zmax = float(coords[:, 2].max())
    ids = np.flatnonzero(
        np.isclose(coords[:, 0], xmax)
        & np.isclose(coords[:, 1], ymax)
        & np.isclose(coords[:, 2], zmax)
    )
    if ids.size != 1:
        raise RuntimeError("expected one tool node at the upper free corner.")
    return int(ids[0])


def _make_basis(kind: str, space, tool_node: int) -> DenseBasis:
    eye = jnp.eye(space.n_dofs, dtype=jnp.float64)
    if kind == "complete":
        return DenseBasis(eye)
    if kind == "tip-y":
        return DenseBasis(eye[:, [tool_node * 3 + 1]])
    raise ValueError(f"unknown basis kind: {kind}")


def _solve_full(space, residual_form, params, force, dirichlet, tol: float, maxiter: int):
    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=residual_form,
        params=params,
        base_external_vector=force,
        dirichlet=dirichlet,
        dtype=jnp.float64,
    )
    runner = ff.NewtonSolveRunner(
        analysis,
        ff.NewtonLoopConfig(tol=tol, atol=tol, maxiter=maxiter, linear_solver="spsolve"),
    )
    u, history = runner.run(u0=jnp.zeros(space.n_dofs, dtype=jnp.float64), newton_callback=lambda _cb: None)
    return u, history[-1].info


def _solve_rom(space, residual_form, params, force, basis, dirichlet, tol: float, maxiter: int):
    model = ff.NonlinearReducedFEModel(
        space=space,
        residual_form=residual_form,
        params=params,
        basis=basis,
        external_vector=force,
    )
    q, info = model.as_problem("body").solve(
        jnp.zeros(basis.n_reduced, dtype=jnp.float64),
        fixed_dofs=dirichlet.dofs if basis.n_reduced == space.n_dofs else None,
        fixed_values=jnp.zeros_like(jnp.asarray(dirichlet.dofs, dtype=jnp.float64)) if basis.n_reduced == space.n_dofs else None,
        tol=tol,
        atol=tol,
        maxiter=maxiter,
    )
    return model.expand(q), info


def _write_outputs(output_dir: str, mesh, space, full_u, rom_u):
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ff.write_elastic_vtu(mesh, space, full_u, str(out / "full.vtu"), compute_j=True, deformed_scale=1.0)
    ff.write_elastic_vtu(mesh, space, rom_u, str(out / "rom.vtu"), compute_j=True, deformed_scale=1.0)
    ff.write_elastic_vtu(mesh, space, full_u - rom_u, str(out / "error.vtu"), compute_j=False, deformed_scale=1.0)
    print(f"VTU written to {out}")


def main():
    args = parse_args()
    mesh = ff.StructuredHexBox(nx=args.nx, ny=args.ny, nz=args.nz, lx=args.lx, ly=args.ly, lz=args.lz).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    tool_node = _tool_node(coords)
    force = jnp.zeros(space.n_dofs, dtype=jnp.float64).at[tool_node * 3 + 1].set(args.force)
    lam, mu = ff.lame_parameters(args.E, args.nu)
    params = {"lam": lam, "mu": mu}
    dirichlet = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin),
        components="xyz",
    )
    basis = _make_basis(args.basis, space, tool_node)

    full_u, full_info = _solve_full(
        space,
        ff.neo_hookean_residual_form,
        params,
        force,
        dirichlet,
        args.tol,
        args.maxiter,
    )
    rom_u, rom_info = _solve_rom(
        space,
        ff.neo_hookean_residual_form,
        params,
        force,
        basis,
        dirichlet,
        args.tol,
        args.maxiter,
    )

    full_nodes = np.asarray(full_u).reshape(-1, 3)
    rom_nodes = np.asarray(rom_u).reshape(-1, 3)
    error = full_u - rom_u
    error_inf = float(jnp.linalg.norm(error, ord=jnp.inf))
    full_inf = float(jnp.linalg.norm(full_u, ord=jnp.inf))
    rel_inf = error_inf / max(full_inf, 1.0e-30)

    print("geometric nonlinear full FEM vs ROM")
    print(f"basis: {args.basis}")
    print(f"n_full: {space.n_dofs}")
    print(f"n_reduced: {basis.n_reduced}")
    print(f"full_converged: {full_info.converged}")
    print(f"rom_converged: {rom_info.converged}")
    print(f"full_iters: {full_info.iters}")
    print(f"rom_iters: {rom_info.iters}")
    print(f"full_tool_uy: {full_nodes[tool_node, 1]:.6e}")
    print(f"rom_tool_uy: {rom_nodes[tool_node, 1]:.6e}")
    print(f"error_inf: {error_inf:.6e}")
    print(f"relative_error_inf: {rel_inf:.6e}")
    _write_outputs(args.output_dir, mesh, space, full_u, rom_u)

    if not full_info.converged or not rom_info.converged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

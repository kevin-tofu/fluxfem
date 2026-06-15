#!/usr/bin/env python
"""Name-based Craig-Bampton ROM builder with a remote RBE3 fixture.

This is the compact API-first pattern:

1. register a structural field,
2. reduce it with Craig-Bampton,
3. append a named remote point,
4. connect the remote point to a structural patch with RBE3,
5. solve either preload or prescribed-reference variants.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton_reduced_coupled_builder.py
"""

from __future__ import annotations

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


def structural_matrices() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_nodes = 5
    n_dofs = 3 * n_nodes
    stiffness = 9.0 * jnp.eye(n_dofs, dtype=jnp.float64)
    stiffness = stiffness + 0.35 * jnp.eye(n_dofs, k=1, dtype=jnp.float64)
    stiffness = stiffness + 0.35 * jnp.eye(n_dofs, k=-1, dtype=jnp.float64)
    mass = jnp.eye(n_dofs, dtype=jnp.float64)
    force = jnp.zeros((n_dofs,), dtype=jnp.float64).at[14].set(-0.25)
    return stiffness, mass, force


def make_system(*, boundary: str) -> tuple[ff.ReducedCoupledSystem, np.ndarray, np.ndarray]:
    stiffness, mass, force = structural_matrices()
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.4, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    fixed_full = ff.vector_dofs_from_nodes(np.array([0]), dim=3).reshape(-1)
    patch_nodes = np.array([2, 3, 4], dtype=int)
    retained = ff.retained_dofs_from_node_sets(patch_nodes, dim=3, extra_dofs=np.concatenate([fixed_full, [14]]))

    builder = ff.ReducedCoupledSystemBuilder.from_structural(
        "workpiece",
        stiffness,
        force,
        mass=mass,
        value_dim=3,
        n_nodes=coords.shape[0],
    )
    builder.reduce_field(
        "workpiece",
        retained_dofs=retained,
        n_modes=1,
        method="craig_bampton",
    )
    builder.add_rbe3_fixture_from_nodes(
        "fixture",
        body="workpiece",
        ref_point=np.array([0.65, 0.65, 0.0]),
        coords=coords,
        nodes=patch_nodes,
        weights=np.array([0.25, 0.35, 0.40]),
        include_rotation=True,
    )
    if boundary == "preload":
        builder.add_remote_preload(
            "fixture",
            stiffness=18.0,
            direction=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            target_displacement=0.02,
        )

    system = builder.build()
    fixed = system.reduced_dofs_from_full("workpiece", fixed_full)
    values = np.zeros((fixed.size,), dtype=float)
    if boundary == "dirichlet":
        ref = system.resolve_block_dofs("fixture", local_dofs=np.arange(6))
        fixed = np.concatenate([fixed, ref])
        values = np.concatenate([values, np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0])])
    return system, fixed, values


def run(boundary: str) -> None:
    system, fixed, values = make_system(boundary=boundary)
    q = system.solve(fixed_dofs=fixed, fixed_values=values)
    u = np.asarray(system.expand(q))
    print(f"[{boundary}] reduced DOFs:       {system.basis.n_reduced}")
    print(f"[{boundary}] full augmented DOFs:{u.size}")
    print(f"[{boundary}] fixture uz:         {float(q[system.field('fixture').offset + 2]): .8e}")
    print(f"[{boundary}] constraint norm:    {float(jnp.linalg.norm(system.constraints.residual(q))): .3e}")


def main() -> None:
    run("preload")
    run("dirichlet")


if __name__ == "__main__":
    main()

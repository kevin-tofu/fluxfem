#!/usr/bin/env python
"""Name-based multi-field Craig-Bampton coupled system.

This tutorial builds two structural subsystems, names their retained support and
interface groups, reduces both with Craig-Bampton, and ties the named interface.
The interface groups are created from surface objects, which is the same entry
point used for contact-candidate surfaces:

    part_a:interface - part_b:interface = 0

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton_multifield_builder.py
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


def subsystem_matrices() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k_a = jnp.array(
        [
            [6.0, -2.0, 0.0],
            [-2.0, 5.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    k_b = jnp.array(
        [
            [5.0, -1.5, 0.0],
            [-1.5, 4.5, -1.0],
            [0.0, -1.0, 3.5],
        ],
        dtype=jnp.float64,
    )
    f_a = jnp.array([0.0, 0.2, 0.0], dtype=jnp.float64)
    f_b = jnp.array([0.0, -0.1, 0.3], dtype=jnp.float64)
    return k_a, k_b, f_a, f_b


def full_reference_solution(k_a, k_b, f_a, f_b) -> jnp.ndarray:
    k = jnp.zeros((6, 6), dtype=jnp.float64)
    k = k.at[:3, :3].set(k_a)
    k = k.at[3:, 3:].set(k_b)
    f = jnp.concatenate([f_a, f_b])
    c = jnp.array([[0.0, 0.0, 1.0, -1.0, 0.0, 0.0]], dtype=jnp.float64)
    return ff.LinearConstraintSystem(c).solve(k, f, fixed_dofs=jnp.array([0], dtype=jnp.int32))


def build_reduced_system() -> ff.ReducedCoupledSystem:
    k_a, k_b, f_a, f_b = subsystem_matrices()
    mass_a = jnp.eye(3, dtype=jnp.float64)
    mass_b = jnp.eye(3, dtype=jnp.float64)

    builder = ff.ReducedCoupledSystemBuilder.from_structural("part_a", k_a, f_a, mass=mass_a)
    builder.register_structural("part_b", k_b, f_b, mass=mass_b)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    interface_a = ff.make_surface_from_facets(coords, np.array([[2]], dtype=np.int32))
    interface_b = ff.make_surface_from_facets(coords, np.array([[0]], dtype=np.int32))
    builder.retain_node_set("part_a", "support", np.array([0]))
    builder.retain_surface_nodes("part_a", "interface", interface_a)
    builder.retain_surface_nodes("part_b", "interface", interface_b)
    builder.retain_node_set("part_b", "free_end", np.array([2]))
    builder.reduce_field("part_a", retained_groups=["support", "interface"], n_modes=1)
    builder.reduce_field("part_b", retained_groups=["interface", "free_end"], n_modes=1)
    builder.register_contact_pair(
        slave="part_a:interface",
        master="part_b:interface",
        name="interface_candidate",
        normal=[1.0],
        penalty=100.0,
        search_radius=0.1,
    )
    builder.tie_retained_groups("part_a:interface", "part_b:interface")
    return builder.build()


def main() -> None:
    k_a, k_b, f_a, f_b = subsystem_matrices()
    system = build_reduced_system()
    fixed = system.reduced_dofs_from_full("part_a", jnp.array([0], dtype=jnp.int32))
    q = system.solve(fixed_dofs=fixed)
    u = system.expand(q)
    u_ref = full_reference_solution(k_a, k_b, f_a, f_b)

    print("multi-field CB builder demo")
    print(f"fields:              {list(system.fields)}")
    print(f"reduced DOFs:         {system.n_dofs}")
    print(f"contact pairs:        {[pair.name for pair in system.contact_pairs]}")
    print(f"contact params:       {system.contact_pair('interface_candidate').params}")
    pair_dofs = system.contact_pair_dofs("interface_candidate")
    print(f"slave reduced DOFs:   {pair_dofs.slave_reduced_dofs.tolist()}")
    print(f"master reduced DOFs:  {pair_dofs.master_reduced_dofs.tolist()}")
    print(f"constraint norm:      {float(jnp.linalg.norm(system.constraints.residual(q))):.3e}")
    print(f"full/reference error: {float(jnp.linalg.norm(u - u_ref)):.3e}")
    print(f"interface values:     part_a[2]={float(u[2]):.8e}, part_b[0]={float(u[3]):.8e}")


if __name__ == "__main__":
    main()

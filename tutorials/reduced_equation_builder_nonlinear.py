#!/usr/bin/env python
"""Residual-first reduced equation builder with two CB-reduced subsystems.

This tutorial keeps the structural CB basis separate from the equation
definition.  Each subsystem contributes a nonlinear reduced residual, and a
coupling residual ties the retained interface coordinates with a nonlinear
spring.  The global Jacobian is obtained by autodiff from the assembled reduced
residual.

Run from the repository root:

    PYTHONPATH=src python tutorials/reduced_equation_builder_nonlinear.py
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


def make_cb(stiffness_scale: float) -> tuple[ff.CraigBamptonBasis, jnp.ndarray]:
    stiffness = stiffness_scale * jnp.array(
        [
            [6.0, -2.0, 0.0, 0.0],
            [-2.0, 5.0, -1.5, 0.0],
            [0.0, -1.5, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(4, dtype=jnp.float64)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0, 3], dtype=jnp.int32), n_modes=1)
    return cb, stiffness


def main() -> None:
    cb_a, k_a = make_cb(1.0)
    cb_b, k_b = make_cb(0.7)
    builder = ff.ReducedEquationBuilder()
    builder.register_field("part_a", basis=cb_a)
    builder.register_field("part_b", basis=cb_b)

    force_a = jnp.array([0.0, 0.1, 0.0, 0.0], dtype=jnp.float64)
    force_b = jnp.array([0.0, 0.0, -0.05, 0.2], dtype=jnp.float64)

    def reduced_internal(cb, stiffness, force, q):
        u = cb.expand(q)
        full_residual = stiffness @ u + 0.05 * u**3 - force
        return cb.project_vector(full_residual)

    builder.add_field_residual("part_a", lambda qa: reduced_internal(cb_a, k_a, force_a, qa))
    builder.add_field_residual("part_b", lambda qb: reduced_internal(cb_b, k_b, force_b, qb))

    def nonlinear_interface_spring(qa, qb):
        gap = qa[1] - qb[0]
        traction = 40.0 * gap + 5.0 * gap**3
        ra = jnp.zeros_like(qa).at[1].add(traction)
        rb = jnp.zeros_like(qb).at[0].add(-traction)
        return {"part_a": ra, "part_b": rb}

    builder.add_coupling_residual(("part_a", "part_b"), nonlinear_interface_spring)
    problem = builder.build()

    q0 = jnp.zeros((problem.n_dofs,), dtype=jnp.float64)
    q, info = ff.solve_reduced_equation(problem, q0, tol=1.0e-12, maxiter=12)
    residual = problem.residual(q)
    jacobian = problem.jacobian(q)
    fields = problem.split(q)
    expanded = problem.expand(q)

    print("reduced equation builder nonlinear demo")
    print(f"global reduced DOFs: {problem.n_dofs}")
    print(f"part_a reduced DOFs: {cb_a.n_reduced}, part_b reduced DOFs: {cb_b.n_reduced}")
    print(f"converged:           {info.converged}")
    print(f"newton iterations:   {info.iters}")
    print(f"residual norm:       {float(jnp.linalg.norm(residual)):.3e}")
    print(f"jacobian shape:      {tuple(jacobian.shape)}")
    print(f"part_a q:            {np.asarray(fields['part_a'])}")
    print(f"part_b q:            {np.asarray(fields['part_b'])}")
    print(f"part_a full u:       {np.asarray(expanded['part_a'])}")
    print(f"part_b full u:       {np.asarray(expanded['part_b'])}")


if __name__ == "__main__":
    main()

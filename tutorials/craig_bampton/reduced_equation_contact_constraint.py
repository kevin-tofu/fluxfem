#!/usr/bin/env python
"""Craig-Bampton contact constraint through ReducedEquationBuilder.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton/reduced_equation_contact_constraint.py
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


class CBPlaneContactConstraint:
    fields = ("body",)

    def __init__(self, cb: ff.CraigBamptonBasis, contact: ff.PlanePenaltyContact):
        self.cb = cb
        self.contact = contact

    def residual(self, q):
        u = self.cb.expand(q)
        return self.cb.project_vector(self.contact.residual(u))


def main() -> None:
    stiffness = jnp.array(
        [
            [6.0, -2.0, 0.0, 0.0],
            [-2.0, 5.0, -1.5, 0.0],
            [0.0, -1.5, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(4, dtype=jnp.float64)
    force = jnp.array([-1.0, 0.1, 0.0, 0.0], dtype=jnp.float64)

    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0], dtype=jnp.int32), n_modes=3)
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=4,
            dofs=jnp.array([[0]], dtype=jnp.int32),
            normals=jnp.array([[1.0]], dtype=jnp.float64),
            gaps0=jnp.array([0.01], dtype=jnp.float64),
        ),
        penalty=50.0,
    )

    builder = ff.ReducedEquationBuilder()
    builder.register_field("body", basis=cb)
    builder.add_field_residual("body", lambda q: cb.project_vector(stiffness @ cb.expand(q) - force))
    builder.add_constraint(CBPlaneContactConstraint(cb, contact))
    problem = builder.build()

    q, info = problem.solve(problem.zeros(dtype=jnp.float64), tol=1.0e-12, maxiter=12)
    u = cb.expand(q)
    residual = problem.residual(q)
    gaps = contact.gaps(u)

    print("reduced equation contact constraint demo")
    print(f"full DOFs:           {cb.n_full}")
    print(f"reduced DOFs:        {cb.n_reduced}")
    print(f"converged:           {info.converged}")
    print(f"newton iterations:   {info.iters}")
    print(f"residual norm:       {float(jnp.linalg.norm(residual)):.3e}")
    print(f"displacement:        {np.asarray(u)}")
    print(f"contact gap:         {float(gaps[0]):.6e}")
    print(f"active contacts:     {int(contact.active_count(u))}")


if __name__ == "__main__":
    main()

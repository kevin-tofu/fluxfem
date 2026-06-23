from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def main():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]]),
            gaps0=jnp.array([0.1]),
            n_dofs=1,
        ),
        penalty=10.0,
    )

    def structural(u):
        return jnp.array([u[0] + 0.2])

    def residual_from_state(contact_state):
        return ff.compose_residuals(structural, contact.residual_with_state(contact_state))

    def solve_fn(residual_fn, x0):
        x = x0
        for _ in range(4):
            J = jax.jacrev(residual_fn)(x)
            x = x + jnp.linalg.solve(J, -residual_fn(x))
        return x, {"residual_norm": float(jnp.linalg.norm(residual_fn(x)))}

    u0 = jnp.array([0.0])
    state0 = contact.state_from_displacement(u0)
    u, info = ff.active_contact_fixed_point_solve(
        u0,
        state0,
        residual_from_state,
        solve_fn,
        contact.state_from_displacement,
        max_active_updates=4,
    )

    print("Active contact outer-loop demo")
    print(f"converged: {info.converged}, outer iters: {info.iters}")
    print(f"solution: {np.asarray(u)}")
    print(f"gap: {np.asarray(info.contact_state.gaps)}")
    print(f"active: {np.asarray(info.contact_state.active)}")
    print(f"outer active changes: {[record.active_changed for record in info.records]}")


if __name__ == "__main__":
    main()

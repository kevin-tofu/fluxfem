#!/usr/bin/env python
"""Craig-Bampton reduced Newmark step with an explicit active-contact loop.

This intentionally uses a tiny one-dimensional model so the callback contract is
visible: the active set is updated outside the differentiated residual, while
the residual/Jacobian used by Newton remains pure JAX code.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_active_contact_newmark.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class PlaneContactState:
    active: jnp.ndarray

    def changed(self, other: "PlaneContactState") -> jnp.ndarray:
        return jnp.any(self.active != other.active)


def main() -> None:
    stiffness = jnp.array(
        [
            [8.0, -2.0, 0.0, 0.0],
            [-2.0, 7.0, -1.0, 0.0],
            [0.0, -1.0, 6.0, -1.5],
            [0.0, 0.0, -1.5, 5.0],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(4, dtype=jnp.float64)
    damping = 0.02 * mass
    force = jnp.array([-25.0, 0.1, 0.0, 0.0], dtype=jnp.float64)
    gap0 = 0.001
    penalty = 80.0

    # Retain the contact DOF explicitly and include all internal modes.  With all
    # modes kept, the ROM reproduces the full-order step up to numerical error.
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0]), n_modes=3)
    reduced_mass = cb.project_matrix(mass)
    reduced_damping = cb.project_matrix(damping)
    reduced_force = cb.project_vector(force)

    q0 = jnp.zeros((cb.n_reduced,), dtype=jnp.float64)
    state = ff.NewmarkState(q=q0, qd=jnp.zeros_like(q0), qdd=jnp.zeros_like(q0))
    config = ff.NewmarkConfig(dt=0.05, tol=1.0e-10, atol=1.0e-12, maxiter=12)

    def update_contact_state(q):
        u = cb.expand(q)
        return PlaneContactState(active=jnp.asarray([gap0 + u[0] < 0.0]))

    def internal_force_from_contact_state(contact_state):
        active_scale = jnp.where(contact_state.active[0], penalty, 0.0)

        def full_internal(u):
            contact_force = jnp.zeros_like(u).at[0].set(active_scale * (gap0 + u[0]))
            return stiffness @ u + contact_force

        return cb.reduced_residual(full_internal)

    next_state, info = ff.active_contact_newmark_step(
        reduced_mass,
        reduced_damping,
        internal_force_from_contact_state,
        reduced_force,
        state,
        config,
        initial_contact_state=update_contact_state(q0),
        update_contact_state=update_contact_state,
        max_active_updates=6,
    )
    u_next = cb.expand(next_state.q)

    print("converged:      ", info.converged)
    print("outer updates:  ", info.iters)
    print("active contact: ", bool(info.contact_state.active[0]))
    print("reduced DOFs:   ", cb.n_reduced)
    print("full u_next:    ", np.asarray(u_next))
    print("contact gap:    ", float(gap0 + u_next[0]))


if __name__ == "__main__":
    main()

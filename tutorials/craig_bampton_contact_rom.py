from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def make_chain_stiffness(n_dofs: int, spring: float, ground: float) -> jnp.ndarray:
    """Small SPD spring-chain stiffness used as a compact ROM demo model."""
    K = np.zeros((n_dofs, n_dofs), dtype=np.float32)
    for i in range(n_dofs - 1):
        K[i, i] += spring
        K[i + 1, i + 1] += spring
        K[i, i + 1] -= spring
        K[i + 1, i] -= spring
    K += ground * np.eye(n_dofs, dtype=np.float32)
    return jnp.asarray(K)


def main():
    n_full = 6
    K = make_chain_stiffness(n_full, spring=80.0, ground=5.0)
    M = jnp.eye(n_full)

    retained = jnp.array([n_full - 1])
    cb = ff.make_craig_bampton_basis(K, M, retained_dofs=retained, n_modes=2)
    Mr = cb.project_matrix(M)
    Cr = 0.02 * Mr

    structural_full = lambda u: K @ u
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=n_full,
            dofs=jnp.array([[n_full - 1]]),
            normals=jnp.array([[1.0]]),
            gaps0=jnp.array([0.015]),
        ),
        penalty=1_000.0,
        smoothing=1e-3,
    )
    full_residual = ff.compose_residuals(structural_full, contact.residual)
    reduced_internal = ff.reduced_residual_from_full(cb, full_residual)

    def reduced_force(t: float):
        f_full = jnp.zeros((n_full,), dtype=jnp.float32)
        ramp = min(t / 0.05, 1.0)
        f_full = f_full.at[n_full - 1].set(-1.5 * ramp)
        return cb.project_vector(f_full)

    q0 = jnp.zeros((cb.n_reduced,), dtype=jnp.float32)
    f0 = reduced_force(0.0)
    qdd0 = jnp.linalg.solve(Mr, f0 - reduced_internal(q0) - Cr @ jnp.zeros_like(q0))
    state0 = ff.NewmarkState(q=q0, qd=jnp.zeros_like(q0), qdd=qdd0, t=0.0)
    config = ff.NewmarkConfig(dt=0.001, tol=1e-7, atol=2e-3, maxiter=25)

    final_state, states, infos = ff.integrate_newmark(
        Mr,
        Cr,
        reduced_internal,
        reduced_force,
        state0,
        config,
        n_steps=100,
    )

    if not all(info.converged for info in infos):
        failed = next(i for i, info in enumerate(infos, start=1) if not info.converged)
        raise RuntimeError(f"Newmark step {failed} did not converge: {infos[failed - 1]}")

    u_final = cb.expand(final_state.q)
    contact_state = contact.state_from_displacement(u_final)
    contact_gap = float(contact_state.gaps[0])
    print("Craig-Bampton contact ROM demo")
    print(f"full DOFs: {n_full}, reduced DOFs: {cb.n_reduced}")
    print(f"steps: {len(infos)}, final time: {final_state.t:.4f}")
    print(f"final retained displacement: {float(u_final[-1]):.6e}")
    print(f"final contact gap: {contact_gap:.6e}")
    print(f"active contact points: {int(jnp.count_nonzero(contact_state.active))}")


if __name__ == "__main__":
    main()

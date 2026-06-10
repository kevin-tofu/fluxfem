from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def make_vector_chain_stiffness(n_nodes: int, dim: int, *, spring: float, ground: float) -> jnp.ndarray:
    n_dofs = n_nodes * dim
    K = ground * np.eye(n_dofs, dtype=np.float32)
    for node in range(n_nodes - 1):
        for d in range(dim):
            a = node * dim + d
            b = (node + 1) * dim + d
            K[a, a] += spring
            K[b, b] += spring
            K[a, b] -= spring
            K[b, a] -= spring
    return jnp.asarray(K)


def main():
    dim = 2
    n_nodes = 4
    n_full = n_nodes * dim
    contact_node = n_nodes - 1
    contact_dofs = jnp.array([[contact_node * dim, contact_node * dim + 1]])

    K = make_vector_chain_stiffness(n_nodes, dim, spring=30.0, ground=2.0)
    M = jnp.eye(n_full, dtype=jnp.float32)
    retained = ff.vector_dofs_from_nodes(jnp.array([contact_node]), dim)
    cb = ff.make_craig_bampton_basis(K, M, retained_dofs=retained, n_modes=2)
    Mr = cb.project_matrix(M)
    Cr = 0.02 * Mr

    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=n_full,
            dofs=contact_dofs,
            normals=jnp.array([[0.0, 1.0]], dtype=jnp.float32),
            gaps0=jnp.array([0.02], dtype=jnp.float32),
        ),
        penalty=400.0,
    )

    config = ff.NewmarkConfig(dt=0.02, tol=1e-8, atol=5e-5, maxiter=25)
    q = jnp.zeros((cb.n_reduced,), dtype=jnp.float32)
    state = ff.NewmarkState(q=q, qd=jnp.zeros_like(q), qdd=jnp.zeros_like(q), t=0.0)
    previous_u = cb.expand(q)
    friction_manager = ff.TangentialPenaltyFrictionManager(
        mu=0.35,
        tangential_penalty=80.0,
        previous_displacement=previous_u,
    )

    def force_at(t: float):
        ramp = min(t / 0.08, 1.0)
        f = jnp.zeros((n_full,), dtype=jnp.float32)
        f = f.at[contact_dofs[0, 0]].set(0.85 * ramp)
        f = f.at[contact_dofs[0, 1]].set(-1.4 * ramp)
        return cb.project_vector(f)

    infos = []
    for _ in range(12):
        frozen_snapshot = friction_manager.snapshot(contact, previous_u)

        def full_internal(u):
            return ff.compose_residuals(lambda x: K @ x, frozen_snapshot.residual())(u)

        reduced_internal = ff.reduced_residual_from_full(cb, full_internal)
        next_state, info = ff.newmark_step(
            Mr,
            Cr,
            reduced_internal,
            force_at(float(state.t) + float(config.dt)),
            state,
            config,
        )
        if not info.converged:
            raise RuntimeError(f"friction Newmark step failed: {info}")
        u_next = cb.expand(next_state.q)
        friction_manager = friction_manager.advance(contact, u_next)
        previous_u = u_next
        state = next_state
        infos.append(info)

    u_final = cb.expand(state.q)
    print("CB friction history ROM demo")
    print(f"full DOFs: {n_full}, reduced DOFs: {cb.n_reduced}")
    print(f"steps: {len(infos)}, final time: {state.t:.4f}")
    print(f"contact displacement: {np.asarray(u_final[contact_dofs[0]])}")
    print(f"gap: {float(contact.gaps(u_final)[0]):.6e}, pressure: {float(contact.pressure(u_final)[0]):.6e}")
    history = friction_manager.history
    print(f"friction force: {np.asarray(history.friction_force[0])}")
    print(f"slip norm: {float(ff.slip_norm(history)):.6e}, stick count: {int(ff.stick_count(history))}")


if __name__ == "__main__":
    main()

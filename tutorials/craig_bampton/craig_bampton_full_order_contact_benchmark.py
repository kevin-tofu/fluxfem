from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def make_problem():
    stiffness = jnp.array(
        [
            [8.0, -2.0, 0.0, 0.0, 0.0],
            [-2.0, 7.0, -1.5, 0.0, 0.0],
            [0.0, -1.5, 6.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 5.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(5, dtype=jnp.float32)
    damping = 0.03 * mass
    external_force = jnp.array([-0.30, 0.08, -0.04, 0.02, 0.01], dtype=jnp.float32)
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=5,
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]], dtype=jnp.float32),
            gaps0=jnp.array([-0.025], dtype=jnp.float32),
        ),
        penalty=40.0,
    )
    config = ff.NewmarkConfig(dt=0.4, tol=1e-9, atol=1e-6, maxiter=25)
    return stiffness, mass, damping, external_force, contact, config


def solve_full_order(stiffness, mass, damping, external_force, contact, config):
    state0 = ff.NewmarkState(
        q=jnp.zeros(stiffness.shape[0], dtype=jnp.float32),
        qd=jnp.zeros(stiffness.shape[0], dtype=jnp.float32),
        qdd=jnp.zeros(stiffness.shape[0], dtype=jnp.float32),
    )

    def internal_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def internal(u):
            return stiffness @ u + contact_residual(u)

        return internal

    return ff.active_contact_newmark_step(
        mass,
        damping,
        internal_from_snapshot,
        external_force,
        state0,
        config,
        initial_contact_state=ff.ContactUpdateSnapshot.from_contact(contact, state0.q),
        update_contact_state=lambda u: ff.ContactUpdateSnapshot.from_contact(contact, u),
        max_active_updates=6,
    )


def solve_rom(stiffness, mass, damping, external_force, contact, config, n_modes: int):
    cb = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=jnp.array([0]),
        n_modes=n_modes,
    )
    q0 = jnp.zeros(cb.n_reduced, dtype=jnp.float32)
    state0 = ff.NewmarkState(q=q0, qd=jnp.zeros_like(q0), qdd=jnp.zeros_like(q0))

    def internal_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def full_internal(u):
            return stiffness @ u + contact_residual(u)

        return ff.reduced_residual_from_full(cb, full_internal)

    state, info = ff.active_contact_newmark_step(
        cb.project_matrix(mass),
        cb.project_matrix(damping),
        internal_from_snapshot,
        cb.project_vector(external_force),
        state0,
        config,
        initial_contact_state=ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q0)),
        update_contact_state=lambda q: ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q)),
        max_active_updates=6,
    )
    return cb, state, info


def main():
    stiffness, mass, damping, external_force, contact, config = make_problem()
    full_state, full_info = solve_full_order(stiffness, mass, damping, external_force, contact, config)
    if not full_info.converged:
        raise RuntimeError(f"full-order solve did not converge: {full_info.stop_reason}")

    full_u = full_state.q
    full_norm = float(jnp.linalg.norm(full_u))
    print("CB ROM vs full-order active contact benchmark")
    print(f"full displacement: {np.asarray(full_u)}")
    print(f"full active: {bool(full_info.contact_state.active_state.active[0])}")
    print()
    print("n_modes  n_red  abs_error      rel_error      active  outer  inner")
    print("-------  -----  ------------  ------------  ------  -----  -----")
    for n_modes in range(5):
        cb, state, info = solve_rom(stiffness, mass, damping, external_force, contact, config, n_modes)
        if not info.converged:
            raise RuntimeError(f"ROM solve failed for n_modes={n_modes}: {info.stop_reason}")
        u_rom = cb.expand(state.q)
        abs_err = float(jnp.linalg.norm(u_rom - full_u))
        rel_err = abs_err / max(full_norm, np.finfo(np.float32).eps)
        active = bool(info.contact_state.active_state.active[0])
        print(
            f"{n_modes:7d}  {cb.n_reduced:5d}  {abs_err:12.6e}  "
            f"{rel_err:12.6e}  {str(active):>6s}  {info.iters:5d}  {len(info.step_infos):5d}"
        )


if __name__ == "__main__":
    main()

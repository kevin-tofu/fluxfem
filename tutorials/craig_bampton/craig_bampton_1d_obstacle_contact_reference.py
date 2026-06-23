from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def make_problem():
    stiffness = jnp.array(
        [
            [6.0, -2.0, 0.0, 0.0],
            [-2.0, 5.0, -1.5, 0.0],
            [0.0, -1.5, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    force = jnp.array([-1.0, 0.1, 0.0, 0.0], dtype=jnp.float32)
    gap0 = 0.01
    penalty = 50.0
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=4,
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]], dtype=jnp.float32),
            gaps0=jnp.array([gap0], dtype=jnp.float32),
        ),
        penalty=penalty,
    )
    return stiffness, force, gap0, penalty, contact


def closed_form_reference(stiffness, force, gap0: float, penalty: float):
    e0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=stiffness.dtype)
    inactive = jnp.linalg.solve(stiffness, force)
    if float(gap0 + inactive[0]) >= 0.0:
        return inactive, False
    active = jnp.linalg.solve(
        stiffness + penalty * jnp.outer(e0, e0),
        force - penalty * gap0 * e0,
    )
    return active, True


def solve_static(residual_from_state, update_state, x0, initial_state):
    def newton_solve(residual_fn, x_init):
        x = x_init
        for _ in range(8):
            residual = residual_fn(x)
            jacobian = jax.jacrev(residual_fn)(x)
            x = x + jnp.linalg.solve(jacobian, -residual)
        return x, {"residual_norm": float(jnp.linalg.norm(residual_fn(x)))}

    return ff.active_contact_fixed_point_solve(
        x0,
        initial_state,
        residual_from_state,
        newton_solve,
        update_state,
        max_active_updates=6,
    )


def solve_full_order(stiffness, force, contact):
    def residual_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def residual(u):
            return stiffness @ u + contact_residual(u) - force

        return residual

    u0 = jnp.zeros(stiffness.shape[0], dtype=stiffness.dtype)
    return solve_static(
        residual_from_snapshot,
        lambda u: ff.ContactUpdateSnapshot.from_contact(contact, u),
        u0,
        ff.ContactUpdateSnapshot.from_contact(contact, u0),
    )


def solve_rom(stiffness, force, contact, n_modes: int):
    mass = jnp.eye(stiffness.shape[0], dtype=stiffness.dtype)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0]), n_modes=n_modes)
    q0 = jnp.zeros(cb.n_reduced, dtype=stiffness.dtype)

    def residual_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def full_residual(u):
            return stiffness @ u + contact_residual(u) - force

        return ff.reduced_residual_from_full(cb, full_residual)

    q, info = solve_static(
        residual_from_snapshot,
        lambda q_next: ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q_next)),
        q0,
        ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q0)),
    )
    return cb, q, info


def main():
    stiffness, force, gap0, penalty, contact = make_problem()
    reference, reference_active = closed_form_reference(stiffness, force, gap0, penalty)
    full_u, full_info = solve_full_order(stiffness, force, contact)
    if not full_info.converged:
        raise RuntimeError(f"full-order static solve did not converge: {full_info.stop_reason}")

    print("CB ROM 1D obstacle penalty-contact reference")
    print(f"closed-form active: {reference_active}")
    print(f"closed-form displacement: {np.asarray(reference)}")
    print(f"full-order displacement: {np.asarray(full_u)}")
    print(f"full-order abs error: {float(jnp.linalg.norm(full_u - reference)):.6e}")
    print()
    print("n_modes  n_red  abs_error      active  outer")
    print("-------  -----  ------------  ------  -----")
    for n_modes in range(4):
        cb, q, info = solve_rom(stiffness, force, contact, n_modes)
        if not info.converged:
            raise RuntimeError(f"ROM static solve failed for n_modes={n_modes}: {info.stop_reason}")
        u_rom = cb.expand(q)
        abs_err = float(jnp.linalg.norm(u_rom - reference))
        active = bool(info.contact_state.active_state.active[0])
        print(f"{n_modes:7d}  {cb.n_reduced:5d}  {abs_err:12.6e}  {str(active):>6s}  {info.iters:5d}")


if __name__ == "__main__":
    main()

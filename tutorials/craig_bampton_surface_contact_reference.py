from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def make_vector_spring_matrix(n_nodes: int, dim: int, edges: list[tuple[int, int]], *, spring: float, ground: float):
    n_dofs = n_nodes * dim
    stiffness = ground * np.eye(n_dofs, dtype=np.float32)
    for a, b in edges:
        for d in range(dim):
            ia = a * dim + d
            ib = b * dim + d
            stiffness[ia, ia] += spring
            stiffness[ib, ib] += spring
            stiffness[ia, ib] -= spring
            stiffness[ib, ia] -= spring
    return jnp.asarray(stiffness)


def make_problem():
    dim = 2
    coords = np.array(
        [
            [0.0, 0.04],
            [1.0, 0.04],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.25, 0.35],
            [0.75, 0.35],
        ],
        dtype=np.float32,
    )
    n_nodes = coords.shape[0]
    slave = ff.make_surface_from_facets(coords, np.array([[0, 1]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[2, 3]], dtype=np.int32))
    stiffness = make_vector_spring_matrix(
        n_nodes,
        dim,
        [(0, 4), (1, 5), (4, 5), (4, 2), (5, 3), (2, 3)],
        spring=18.0,
        ground=1.0,
    )
    mass = jnp.eye(n_nodes * dim, dtype=jnp.float32)
    damping = 0.02 * mass
    external_force = jnp.zeros(n_nodes * dim, dtype=jnp.float32)
    external_force = external_force.at[0 * dim + 1].set(-2.0)
    external_force = external_force.at[1 * dim + 1].set(-2.0)
    external_force = external_force.at[0 * dim + 0].set(0.12)
    contact = ff.SurfaceQuadraturePenaltyContact(
        ff.surface_quadrature_contact_kinematics_from_surfaces(
            slave,
            master,
            dim=dim,
            n_total_nodes=n_nodes,
            normal=jnp.array([0.0, 1.0], dtype=jnp.float32),
            quadrature_rule="vertices",
        ),
        penalty=180.0,
    )
    config = ff.NewmarkConfig(dt=0.5, tol=1e-8, atol=2e-5, maxiter=30)
    contact_surface = ff.make_surface_from_facets(coords, np.array([[0, 1], [2, 3]], dtype=np.int32))
    retained = ff.retained_dofs_from_surface(contact_surface, dim)
    return stiffness, mass, damping, external_force, contact, config, retained


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
        max_active_updates=8,
    )


def solve_rom(stiffness, mass, damping, external_force, contact, config, retained, n_modes: int):
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=n_modes)
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
        max_active_updates=8,
    )
    return cb, state, info


def main():
    stiffness, mass, damping, external_force, contact, config, retained = make_problem()
    full_state, full_info = solve_full_order(stiffness, mass, damping, external_force, contact, config)
    if not full_info.converged:
        raise RuntimeError(f"full-order surface contact solve did not converge: {full_info.stop_reason}")
    full_u = full_state.q
    full_norm = float(jnp.linalg.norm(full_u))
    print("CB ROM surface-quadrature contact reference")
    print(f"full active count: {int(contact.active_count(full_u))}")
    print(f"full gaps: {np.asarray(contact.gaps(full_u))}")
    print()
    print("n_modes  n_red  abs_error      rel_error      active  outer  inner")
    print("-------  -----  ------------  ------------  ------  -----  -----")
    for n_modes in range(5):
        cb, state, info = solve_rom(stiffness, mass, damping, external_force, contact, config, retained, n_modes)
        if not info.converged:
            raise RuntimeError(f"ROM solve failed for n_modes={n_modes}: {info.stop_reason}")
        u_rom = cb.expand(state.q)
        abs_err = float(jnp.linalg.norm(u_rom - full_u))
        rel_err = abs_err / max(full_norm, np.finfo(np.float32).eps)
        print(
            f"{n_modes:7d}  {cb.n_reduced:5d}  {abs_err:12.6e}  "
            f"{rel_err:12.6e}  {int(contact.active_count(u_rom)):6d}  {info.iters:5d}  {len(info.step_infos):5d}"
        )


if __name__ == "__main__":
    main()

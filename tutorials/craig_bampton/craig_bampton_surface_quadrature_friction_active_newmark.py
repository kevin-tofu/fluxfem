from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def make_vector_spring_matrix(
    n_nodes: int,
    dim: int,
    edges: list[tuple[int, int]],
    *,
    spring: float,
    ground: float,
) -> jnp.ndarray:
    n_dofs = n_nodes * dim
    K = ground * np.eye(n_dofs, dtype=np.float32)
    for a, b in edges:
        for d in range(dim):
            ia = a * dim + d
            ib = b * dim + d
            K[ia, ia] += spring
            K[ib, ib] += spring
            K[ia, ib] -= spring
            K[ib, ia] -= spring
    return jnp.asarray(K)


def main():
    dim = 2
    coords = np.array(
        [
            [0.0, 0.04],  # slave surface
            [1.0, 0.04],
            [0.0, 0.0],  # master surface
            [1.0, 0.0],
            [0.25, 0.35],  # interior/modalized nodes
            [0.75, 0.35],
        ],
        dtype=np.float32,
    )
    n_nodes = coords.shape[0]
    n_full = n_nodes * dim
    slave = ff.make_surface_from_facets(coords, np.array([[0, 1]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[2, 3]], dtype=np.int32))

    edges = [(0, 4), (1, 5), (4, 5), (4, 2), (5, 3), (2, 3)]
    K = make_vector_spring_matrix(n_nodes, dim, edges, spring=18.0, ground=1.0)
    M = jnp.eye(n_full, dtype=jnp.float32)

    contact_nodes = np.unique(np.concatenate([slave.conn.reshape(-1), master.conn.reshape(-1)]))
    retained = ff.vector_dofs_from_nodes(contact_nodes, dim)
    cb = ff.make_craig_bampton_basis(K, M, retained_dofs=retained, n_modes=2)

    q0 = jnp.zeros((cb.n_reduced,), dtype=jnp.float32)
    u0 = cb.expand(q0)
    dynamics = ff.ReducedContactDynamics(
        cb=cb,
        stiffness=K,
        mass=M,
        damping=0.02 * M,
        search_manager=ff.make_surface_quadrature_contact_search_manager(
            slave,
            master,
            dim=dim,
            n_total_nodes=n_nodes,
            search_radius=0.08,
            skin=0.04,
            penalty=180.0,
            normal=jnp.array([0.0, 1.0], dtype=jnp.float32),
            quadrature_rule="vertices",
            cell_size=1.0,
        ),
        friction_manager=ff.TangentialPenaltyFrictionManager(
            mu=0.25,
            tangential_penalty=35.0,
            previous_displacement=u0,
        ),
    )

    state = ff.NewmarkState(
        q=q0,
        qd=jnp.zeros_like(q0),
        qdd=jnp.zeros_like(q0),
        t=0.0,
    )
    config = ff.NewmarkConfig(dt=0.5, tol=1e-8, atol=2e-5, maxiter=30)

    f_full = jnp.zeros((n_full,), dtype=jnp.float32)
    f_full = f_full.at[0 * dim + 1].set(-2.0)
    f_full = f_full.at[1 * dim + 1].set(-2.0)
    f_full = f_full.at[0 * dim + 0].set(0.20)
    f_full = f_full.at[1 * dim + 0].set(0.08)

    next_state, info = dynamics.active_newmark_step(
        f_full,
        state,
        config,
        max_active_updates=8,
    )
    if not info.converged:
        raise RuntimeError(f"quadrature friction active Newmark step did not converge: {info.stop_reason}")

    u_final = cb.expand(next_state.q)
    snapshot = info.contact_state
    history = dynamics.friction_manager.history
    contact = snapshot.contact

    print("CB surface-quadrature friction active Newmark demo")
    print(f"full DOFs: {n_full}, reduced DOFs: {cb.n_reduced}, retained DOFs: {cb.n_retained}")
    print(f"quadrature points: {contact.kinematics.gaps0.size}")
    print(f"outer active updates: {info.iters}, inner Newton solves: {len(info.step_infos)}")
    print(f"gaps: {np.asarray(contact.gaps(u_final))}")
    print(f"active count: {int(contact.active_count(u_final))}")
    print(f"contact force norm: {float(contact.force_norm(u_final)):.6e}")
    print(f"friction forces: {np.asarray(history.friction_force)}")
    print(f"slip norm: {float(ff.slip_norm(history)):.6e}, stick count: {int(ff.stick_count(history))}")
    print(f"quadrature weights: {np.asarray(contact.kinematics.quadrature_weights)}")
    print(f"cached master facet ids: {np.asarray(contact.kinematics.master_facet_ids)}")


if __name__ == "__main__":
    main()

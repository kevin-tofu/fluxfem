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
    dim = 3
    coords = np.array(
        [
            [0.25, 0.25, 0.04],  # slave node
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.45],  # interior/modalized nodes
            [0.75, 0.75, 0.45],
        ],
        dtype=np.float32,
    )
    n_nodes = coords.shape[0]
    n_full = n_nodes * dim
    slave = ff.make_surface_from_facets(coords, np.array([[0]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[1, 2, 3, 4]], dtype=np.int32))

    edges = [(0, 5), (5, 6), (6, 3), (1, 5), (2, 6), (4, 5)]
    K = make_vector_spring_matrix(n_nodes, dim, edges, spring=12.0, ground=1.5)
    M = jnp.eye(n_full, dtype=jnp.float32)

    contact_nodes = np.unique(np.concatenate([slave.conn.reshape(-1), master.conn.reshape(-1)]))
    retained = ff.vector_dofs_from_nodes(contact_nodes, dim)
    cb = ff.make_craig_bampton_basis(K, M, retained_dofs=retained, n_modes=3)

    q0 = jnp.zeros((cb.n_reduced,), dtype=jnp.float32)
    u0 = cb.expand(q0)
    dynamics = ff.ReducedContactDynamics(
        cb=cb,
        stiffness=K,
        mass=M,
        damping=0.01 * M,
        search_manager=ff.make_node_surface_contact_search_manager(
            slave,
            master,
            dim=dim,
            n_total_nodes=n_nodes,
            search_radius=0.12,
            skin=0.05,
            penalty=250.0,
            cell_size=1.0,
        ),
        friction_manager=ff.TangentialPenaltyFrictionManager(
            mu=0.25,
            tangential_penalty=45.0,
            previous_displacement=u0,
        ),
    )

    state = ff.NewmarkState(
        q=q0,
        qd=jnp.zeros_like(q0),
        qdd=jnp.zeros_like(q0),
        t=0.0,
    )
    config = ff.NewmarkConfig(dt=1.0, tol=1e-8, atol=1e-5, maxiter=25)

    f_full = jnp.zeros((n_full,), dtype=jnp.float32)
    f_full = f_full.at[0 * dim + 0].set(0.35)
    f_full = f_full.at[0 * dim + 2].set(-1.35)

    next_state, info = dynamics.active_newmark_step(
        f_full,
        state,
        config,
        max_active_updates=8,
    )
    if not info.converged:
        raise RuntimeError(f"frictional active Newmark step did not converge: {info.stop_reason}")

    u_final = cb.expand(next_state.q)
    snapshot = info.contact_state
    history = dynamics.friction_manager.history
    slave_u = u_final[snapshot.contact.kinematics.slave_dofs[0]]

    print("CB node-surface friction active Newmark demo")
    print(f"full DOFs: {n_full}, reduced DOFs: {cb.n_reduced}, retained DOFs: {cb.n_retained}")
    print(f"outer active updates: {info.iters}, inner Newton solves: {len(info.step_infos)}")
    print(f"final slave displacement: {np.asarray(slave_u)}")
    print(f"final gap: {float(snapshot.active_state.gaps[0]):.6e}")
    print(f"active: {bool(snapshot.active_state.active[0])}")
    print(f"contact force norm: {float(snapshot.contact.force_norm(u_final)):.6e}")
    print(f"friction force: {np.asarray(history.friction_force[0])}")
    print(f"slip norm: {float(ff.slip_norm(history)):.6e}, stick count: {int(ff.stick_count(history))}")
    print(f"cached master facet ids: {np.asarray(snapshot.contact.kinematics.master_facet_ids)}")


if __name__ == "__main__":
    main()

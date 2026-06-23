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
    Mr = cb.project_matrix(M)
    Cr = 0.02 * Mr

    search_manager: dict[str, ff.SurfaceQuadratureContactSearchManager] = {
        "value": ff.make_surface_quadrature_contact_search_manager(
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
        )
    }

    def build_contact(u_full: jnp.ndarray) -> ff.SurfaceQuadraturePenaltyContact:
        contact, next_manager = search_manager["value"].build_contact(u_full)
        search_manager["value"] = next_manager
        return contact

    def build_snapshot(q: jnp.ndarray) -> ff.ContactUpdateSnapshot:
        u = cb.expand(q)
        contact = build_contact(u)
        return ff.ContactUpdateSnapshot.from_contact(contact, u)

    def internal_force_from_snapshot(snapshot: ff.ContactUpdateSnapshot):
        contact_residual = snapshot.residual()

        def full_residual(u):
            return K @ u + contact_residual(u)

        return ff.reduced_residual_from_full(cb, full_residual)

    q0 = jnp.zeros((cb.n_reduced,), dtype=jnp.float32)
    state0 = ff.NewmarkState(
        q=q0,
        qd=jnp.zeros_like(q0),
        qdd=jnp.zeros_like(q0),
        t=0.0,
    )
    config = ff.NewmarkConfig(dt=0.5, tol=1e-8, atol=2e-5, maxiter=30)

    f_full = jnp.zeros((n_full,), dtype=jnp.float32)
    f_full = f_full.at[0 * dim + 1].set(-2.0)
    f_full = f_full.at[1 * dim + 1].set(-2.0)
    f_full = f_full.at[0 * dim + 0].set(0.12)
    f_reduced = cb.project_vector(f_full)

    next_state, info = ff.active_contact_newmark_step(
        Mr,
        Cr,
        internal_force_from_snapshot,
        f_reduced,
        state0,
        config,
        initial_contact_state=build_snapshot(q0),
        update_contact_state=build_snapshot,
        max_active_updates=8,
    )
    if not info.converged:
        raise RuntimeError(f"quadrature active Newmark step did not converge: {info.stop_reason}")

    u_final = cb.expand(next_state.q)
    snapshot = info.contact_state
    contact = snapshot.contact

    print("CB surface-quadrature active Newmark demo")
    print(f"full DOFs: {n_full}, reduced DOFs: {cb.n_reduced}, retained DOFs: {cb.n_retained}")
    print(f"quadrature points: {contact.kinematics.gaps0.size}")
    print(f"outer active updates: {info.iters}, inner Newton solves: {len(info.step_infos)}")
    print(f"gaps: {np.asarray(contact.gaps(u_final))}")
    print(f"active count: {int(contact.active_count(u_final))}")
    print(f"contact energy: {float(contact.penetration_energy(u_final)):.6e}")
    print(f"contact force norm: {float(contact.force_norm(u_final)):.6e}")
    print(f"quadrature weights: {np.asarray(contact.kinematics.quadrature_weights)}")
    print(f"cached master facet ids: {np.asarray(contact.kinematics.master_facet_ids)}")


if __name__ == "__main__":
    main()

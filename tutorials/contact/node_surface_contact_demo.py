from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def main():
    coords = np.array(
        [
            [0.75, 0.25, -0.05],  # slave node
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    slave = ff.make_surface_from_facets(coords, np.array([[0]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[1, 2, 3, 4]], dtype=np.int32))
    index = ff.contact_aabb_index_from_surface(
        master,
        dim=3,
        n_total_nodes=coords.shape[0],
        cell_size=1.0,
    )
    neighbors = ff.node_surface_neighbor_list_from_aabb_index(
        slave,
        index,
        dim=3,
        search_radius=0.2,
        skin=0.1,
        n_total_nodes=coords.shape[0],
    )
    candidates = neighbors.candidate_set

    kin = ff.node_surface_contact_kinematics_from_surfaces(
        slave,
        master,
        dim=3,
        n_total_nodes=coords.shape[0],
        normal=jnp.array([0.0, 0.0, 1.0]),
        candidate_facet_ids=candidates,
    )
    contact = ff.NodeSurfacePenaltyContact(kin, penalty=1_000.0, smoothing=0.0)

    u = jnp.zeros(coords.shape[0] * 3)
    residual = contact.residual(u)
    jacobian = jax.jacrev(contact.residual)(u)

    slave_force = residual[kin.slave_dofs[0]]
    master_force_sum = jnp.sum(residual[kin.master_dofs[0]], axis=0)

    print("Node-to-surface contact demo")
    print(f"slave nodes: {kin.slave_dofs.shape[0]}, master facet nodes: {kin.master_dofs.shape[1]}")
    print(f"candidate master facet ids: {np.asarray(candidates.master_facet_ids)}")
    print(f"neighbor list needs refresh: {bool(neighbors.needs_refresh(u))}")
    print(f"master weights: {np.asarray(kin.master_weights[0])}")
    print(f"gap: {float(contact.gaps(u)[0]):.6e}, active={bool(contact.active_mask(u)[0])}")
    print(f"slave force: {np.asarray(slave_force)}")
    print(f"master force sum: {np.asarray(master_force_sum)}")
    print(f"force balance sum={float(jnp.sum(residual)):.6e}, J shape={tuple(jacobian.shape)}")

    u_shift = u.at[0].set(-0.5)
    kin_updated = ff.node_surface_contact_kinematics_from_surfaces(
        slave,
        master,
        dim=3,
        n_total_nodes=coords.shape[0],
        normal=jnp.array([0.0, 0.0, 1.0]),
        displacement=u_shift,
    )
    print(f"updated weights after shifted slave: {np.asarray(kin_updated.master_weights[0])}")


if __name__ == "__main__":
    main()

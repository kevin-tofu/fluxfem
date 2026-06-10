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
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.08],
            [1.0, 0.0, 0.08],
            [0.0, 1.0, 0.08],
            [1.0, 1.0, 0.08],
        ],
        dtype=np.float32,
    )
    slave = ff.make_surface_from_facets(coords, np.array([[0, 1, 3, 2]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[4, 5, 7, 6]], dtype=np.int32))

    kin = ff.paired_contact_kinematics_from_surfaces(
        slave,
        master,
        dim=3,
        n_total_nodes=coords.shape[0],
        normal=jnp.array([0.0, 0.0, 1.0]),
    )
    contact = ff.PairedPenaltyContact(kin, penalty=500.0, smoothing=1e-4)

    u = jnp.zeros(coords.shape[0] * 3)

    residual = contact.residual(u)
    jacobian = jax.jacrev(contact.residual)(u)

    print("Paired surface contact demo")
    print(f"pairs: {kin.slave_dofs.shape[0]}")
    print(f"initial min gap: {float(jnp.min(kin.gaps0)):.6e}")
    print(f"current min gap: {float(jnp.min(contact.gaps(u))):.6e}")
    print(f"active pairs: {int(jnp.count_nonzero(contact.active_mask(u)))}")
    print(f"||R||_2={float(jnp.linalg.norm(residual)):.6e}, J shape={tuple(jacobian.shape)}")
    print(f"force balance sum={float(jnp.sum(residual)):.6e}")


if __name__ == "__main__":
    main()

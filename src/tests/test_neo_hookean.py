"""Hyperelastic residual/Jacobian sanity checks."""
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def test_neo_hookean_zero_residual_and_symmetric_jacobian():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    params = {"mu": 2.0, "lam": 3.0}
    u0 = jnp.zeros(space.n_dofs, dtype=jnp.float32)

    R = space.assemble_residual(ff.neo_hookean_residual_form, u0, params)
    assert np.allclose(np.asarray(R), 0.0, atol=1e-6)

    J = space.assemble_jacobian(ff.neo_hookean_residual_form, u0, params, sparse=False)
    assert np.allclose(np.asarray(J), np.asarray(J).T, atol=1e-8)

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff
from fluxfem.solver.bc import assemble_surface_linear_form, SurfaceFormContext, SurfaceFormField


def _scalar_surface_form(ctx: SurfaceFormContext, _params):
    # Return N for scalar dim=1; shape (n_q, n_nodes)
    return jnp.asarray(ctx.v.N)


def test_surface_linear_form_jax_quad():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    facets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    surface = ff.SurfaceMesh.from_facets(coords, facets)
    F = assemble_surface_linear_form(surface, _scalar_surface_form, None, dim=1)
    F_np = np.asarray(F)
    assert F_np.shape == (4,)
    # Unit square area = 1, linear quad integrates to 1/4 per node
    assert np.allclose(F_np, 0.25, atol=1e-6)

    # Should be differentiable w.r.t. coords
    def loss(c):
        s = ff.SurfaceMesh.from_facets(c, facets)
        F_local = assemble_surface_linear_form(s, _scalar_surface_form, None, dim=1)
        return jnp.sum(F_local)

    g = jax.grad(loss)(coords)
    assert g.shape == coords.shape


def test_surface_linear_form_jax_tri():
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    facets = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    surface = ff.SurfaceMesh.from_facets(coords, facets)
    F = assemble_surface_linear_form(surface, _scalar_surface_form, None, dim=1)
    F_np = np.asarray(F)
    assert F_np.shape == (3,)
    # Right triangle area = 0.5, each node gets 1/6
    assert np.allclose(F_np, 1.0 / 6.0, atol=1e-6)

    def loss(c):
        s = ff.SurfaceMesh.from_facets(c, facets)
        F_local = assemble_surface_linear_form(s, _scalar_surface_form, None, dim=1)
        return jnp.sum(F_local)

    g = jax.grad(loss)(coords)
    assert g.shape == coords.shape

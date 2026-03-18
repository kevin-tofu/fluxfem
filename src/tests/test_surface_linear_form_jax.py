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


def test_vector_surface_load_form_accepts_traced_callable_load():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    coords = jnp.asarray(mesh.coords, dtype=jnp.float32)
    facets = jnp.asarray(mesh.facets_on_plane(axis=0, value=1.0), dtype=jnp.int32)
    surface = ff.SurfaceMesh.from_facets(coords, facets)

    center = jnp.asarray([1.0, 0.5, 0.5], dtype=jnp.float32)
    direction = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)

    def load_fn(x_q):
        r2 = jnp.sum((x_q - center) ** 2, axis=1)
        mag = jnp.exp(-5.0 * r2)
        return mag[:, None] * direction[None, :]

    form = ff.make_vector_surface_load_form(load_fn)
    F = surface.assemble_linear_form_on_space(space, form, params=None)

    F_np = np.asarray(F)
    assert F_np.shape == (space.n_dofs,)
    assert np.linalg.norm(F_np) > 0.0

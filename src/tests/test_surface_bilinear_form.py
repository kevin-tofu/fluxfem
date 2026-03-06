"""Surface bilinear form assembly tests."""
import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def test_surface_bilinear_form_scalar():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    coords = np.asarray(mesh.coords)
    xmax = float(coords[:, 0].max())

    def on_xmax(face: np.ndarray) -> bool:
        return np.allclose(face[:, 0], xmax, atol=1e-8)

    facets = mesh.boundary_facets_where(on_xmax)
    surface = ff.SurfaceMesh.from_hex_mesh(mesh, facets)

    form = ff.BilinearForm.surface(
        lambda u, v, p: p.alpha * h_wf.outer(v, u) * h_wf.ds()
    )
    pattern = space.get_sparsity_pattern(with_idx=True)
    K_surf = surface.assemble_bilinear_form_on_space(
        space, form.get_compiled(), params=ff.Params(alpha=2.0), pattern=pattern
    )
    dense = np.asarray(K_surf.to_dense())

    face_nodes = np.unique(np.asarray(facets).reshape(-1))
    n = dense.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[face_nodes] = False

    assert np.allclose(dense, dense.T)
    assert np.any(dense[face_nodes[:, None], face_nodes] > 0.0)
    assert np.allclose(dense[np.ix_(mask, np.arange(n))], 0.0)
    assert np.allclose(dense[np.ix_(np.arange(n), mask)], 0.0)


def test_surface_bilinear_form_jax_grad_and_jit():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    coords = np.asarray(mesh.coords)
    xmax = float(coords[:, 0].max())

    def on_xmax(face: np.ndarray) -> bool:
        return np.allclose(face[:, 0], xmax, atol=1e-8)

    facets = mesh.boundary_facets_where(on_xmax)
    surface = ff.SurfaceMesh.from_hex_mesh(mesh, facets)
    pattern = space.get_sparsity_pattern(with_idx=True)
    form = ff.BilinearForm.surface(
        lambda u, v, p: p.alpha * h_wf.outer(v, u) * h_wf.ds()
    ).get_compiled()

    def loss(alpha):
        K = surface.assemble_bilinear_form_on_space(
            space,
            form,
            params=ff.Params(alpha=alpha),
            pattern=pattern,
        )
        return jnp.sum(K.data)

    val = loss(jnp.array(2.5))
    grad = jax.grad(loss)(jnp.array(2.5))
    jit_val = jax.jit(loss)(jnp.array(2.5))

    assert np.isfinite(float(val))
    assert np.isfinite(float(jit_val))
    assert np.isclose(float(grad), 1.0, rtol=1e-7, atol=1e-7)

"""Body/surface load assembly comparisons against scikit-fem."""
import numpy as np
import jax.numpy as jnp
import pytest

import fluxfem as ff


def test_rhs_body_and_surface_vs_skfem():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshHex, ElementHex1, Basis, FacetBasis, asm
    from skfem.helpers import dot
    try:
        from skfem.element import ElementVector as ElementVectorSKF  # type: ignore
    except Exception:
        ElementVectorSKF = None  # type: ignore

    mesh_ff = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space_ff = ff.make_hex_space(mesh_ff, dim=3, intorder=2)

    f_vec = np.array([0.2, -0.1, 0.05], dtype=float)
    traction_vec = np.array([0.0, 1.5, 0.0], dtype=float)

    F_body_ff = np.asarray(space_ff.assemble_linear_form(ff.vector_body_force_form, params=jnp.asarray(f_vec)))
    xmax = float(np.asarray(mesh_ff.coords)[:, 0].max())
    facets = mesh_ff.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    surf = ff.SurfaceMesh.from_hex_mesh(mesh_ff, facets)
    F_flux = surf.assemble_load(
        load=traction_vec,
        dim=3,
        n_total_nodes=mesh_ff.n_nodes,
        F0=F_body_ff,
    )

    # scikit-fem assembly
    xs = np.linspace(0.0, 1.0, 2)
    ys = np.linspace(0.0, 1.0, 2)
    zs = np.linspace(0.0, 1.0, 2)
    mesh_sf = MeshHex().init_tensor(xs, ys, zs)
    if ElementVectorSKF is not None:
        element = ElementVectorSKF(ElementHex1(), dim=3)
    else:
        try:
            element = ElementHex1() * 3
        except Exception as e:
            pytest.skip(f"Vector element not available: {e}")
    basis_sf = Basis(mesh_sf, element, intorder=2)

    @skfem.LinearForm
    def lf(v, w):
        return dot(f_vec, v)

    F_body_sf = asm(lf, basis_sf)

    # facets on x = 1.0
    facet_coords = mesh_sf.p[:, mesh_sf.facets]
    facet_centers = facet_coords.mean(axis=1)
    facet_ids = np.nonzero(np.isclose(facet_centers[0], 1.0, atol=1e-8))[0]
    fb = FacetBasis(mesh_sf, basis_sf.elem, facets=facet_ids, intorder=2)

    @skfem.LinearForm
    def tform(v, w):
        return dot(traction_vec, v)

    F_trac_sf = asm(tform, fb)
    F_sf = F_body_sf + F_trac_sf

    # reorder scikit-fem dofs to fluxfem ordering
    coords_ff = np.asarray(mesh_ff.coords)
    coords_sf = mesh_sf.p.T
    perm_nodes = []
    for c in coords_ff:
        matches = np.nonzero(np.all(np.isclose(coords_sf, c, atol=1e-8), axis=1))[0]
        assert len(matches) == 1
        perm_nodes.append(matches[0])
    perm_nodes = np.array(perm_nodes, dtype=int)

    perm_dofs = []
    for n in perm_nodes:
        perm_dofs.extend([3 * n + 0, 3 * n + 1, 3 * n + 2])
    perm_dofs = np.array(perm_dofs, dtype=int)

    F_sf_reordered = np.zeros_like(F_sf)
    for i, j in enumerate(perm_dofs):
        F_sf_reordered[i] = F_sf[j]

    np.testing.assert_allclose(F_flux, F_sf_reordered, rtol=1e-6, atol=1e-8)

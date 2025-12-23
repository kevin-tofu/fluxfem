"""Surface mesh construction and load assembly checks."""
import numpy as np
import pytest

import fluxfem as ff


def test_surface_mesh_area_and_selection():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facets = np.array([[0, 1, 2, 3]], dtype=int)
    tags = np.array([5], dtype=int)
    surf = ff.SurfaceMesh.from_facets(coords, facets, facet_tags=tags)

    areas = surf.facet_areas()
    assert np.allclose(areas, [1.0])

    surf_sel = surf.select_by_tag(5)
    assert surf_sel.conn.shape[0] == 1
    np.testing.assert_array_equal(surf_sel.conn[0], facets[0])


def test_surface_mesh_from_hex_mesh_keeps_coords():
    hex_mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=2.0, ly=1.0, lz=1.0).build()
    facets, tags = ff.tag_axis_minmax_facets(hex_mesh, axis=0, dirichlet_tag=1, neumann_tag=2)
    surf = ff.SurfaceMesh.from_hex_mesh(hex_mesh, facets, facet_tags=tags)

    assert surf.n_facets == facets.shape[0]
    np.testing.assert_array_equal(np.asarray(surf.coords), np.asarray(hex_mesh.coords))
    areas = surf.facet_areas()
    assert areas.shape == (facets.shape[0],)
    assert np.all(areas > 0)


def test_assemble_surface_load_matches_manual():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    surf = ff.SurfaceMesh.from_facets(coords, np.array([[0, 1, 2, 3]], dtype=int))
    F = surf.assemble_load(load=[1.0, 2.0, 3.0], dim=3)
    expected = np.tile(np.array([0.25, 0.5, 0.75]), 4)
    np.testing.assert_allclose(F, expected)


def test_surface_load_matches_scikit_fem():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshHex, ElementHex1, ElementVector, Basis, FacetBasis, asm

    traction = np.array([0.0, 1.5, 0.0], dtype=float)
    nx = ny = nz = 2
    lx, ly, lz = 1.0, 2.0, 1.0

    # fluxfem surface traction
    mesh_ff = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
    facets, tags = ff.tag_axis_minmax_facets(mesh_ff, axis=0, dirichlet_tag=1, neumann_tag=2)
    neumann_facets = facets[np.asarray(tags) == 2]
    surf = ff.SurfaceMesh.from_hex_mesh(mesh_ff, neumann_facets)
    F_ff = surf.assemble_load(load=traction, dim=3, n_total_nodes=mesh_ff.n_nodes)

    # scikit-fem surface traction on x=lx face
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    zs = np.linspace(0.0, lz, nz + 1)
    mesh_sf = MeshHex().init_tensor(xs, ys, zs)
    elem_vec = ElementVector(ElementHex1(), dim=3)
    basis_sf = Basis(mesh_sf, elem_vec, intorder=2)

    # select facets on x = lx
    facet_coords = mesh_sf.p[:, mesh_sf.facets]  # (3, nodes_per_facet, n_facets)
    facet_centers = facet_coords.mean(axis=1)    # (3, n_facets)
    facet_ids = np.nonzero(np.isclose(facet_centers[0], lx, atol=1e-8))[0]
    fb = FacetBasis(mesh_sf, basis_sf.elem, facets=facet_ids, intorder=2)

    @skfem.LinearForm
    def tform(v, w):
        return traction[0] * v[0] + traction[1] * v[1] + traction[2] * v[2]

    F_sf = np.asarray(asm(tform, fb)).ravel()

    # reorder scikit-fem dofs to fluxfem ordering
    coords_ff = np.asarray(mesh_ff.coords)
    coords_sf = mesh_sf.p.T
    perm = []
    for c in coords_ff:
        matches = np.nonzero(np.all(np.isclose(coords_sf, c, atol=1e-8), axis=1))[0]
        assert len(matches) == 1
        perm.append(matches[0])
    perm = np.array(perm, dtype=int)

    F_sf_reordered = np.zeros_like(F_sf)
    for i, j in enumerate(perm):
        for d in range(3):
            F_sf_reordered[3 * i + d] = F_sf[3 * j + d]

    max_diff = float(np.max(np.abs(F_ff - F_sf_reordered)))
    assert max_diff < 1e-8, f"surface traction mismatch vs scikit-fem: {max_diff}"

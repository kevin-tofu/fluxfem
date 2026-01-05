"""Surface supermesh intersection tests."""
import numpy as np

import fluxfem as ff


def _tri_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def _total_area(coords, conn):
    area = 0.0
    for tri in conn:
        a, b, c = coords[tri]
        area += _tri_area(a, b, c)
    return area


def test_supermesh_quad_quad_overlap():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.5, 1.5, 0.0],
        ],
        dtype=float,
    )
    facets_a = np.array([[0, 1, 2, 3]], dtype=int)
    facets_b = np.array([[4, 5, 6, 7]], dtype=int)

    surf_a = ff.SurfaceMesh.from_facets(coords, facets_a)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets_b)

    sm = ff.build_surface_supermesh(surf_a, surf_b, tol=1e-8)
    area = _total_area(sm.coords, sm.conn)
    assert sm.conn.shape[0] == 2
    assert np.allclose(area, 0.25, atol=1e-6)
    assert np.all(sm.source_facets_a == 0)
    assert np.all(sm.source_facets_b == 0)


def test_supermesh_tri_quad_overlap():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facets_a = np.array([[0, 1, 2]], dtype=int)
    facets_b = np.array([[0, 1, 3, 2]], dtype=int)

    surf_a = ff.SurfaceMesh.from_facets(coords, facets_a)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets_b)

    sm = ff.build_surface_supermesh(surf_a, surf_b, tol=1e-8)
    area = _total_area(sm.coords, sm.conn)
    assert np.allclose(area, 0.5, atol=1e-6)

"""Mortar coupling matrix assembly tests."""
import numpy as np

import fluxfem as ff


def _sum_matrix(mat):
    return float(np.sum(mat.data))


def _dense_matrix(mat):
    out = np.zeros(mat.shape, dtype=float)
    for r, c, v in zip(mat.rows, mat.cols, mat.data):
        out[int(r), int(c)] += float(v)
    return out


def test_mortar_overlap_area_matches():
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

    M_aa, M_ab = ff.assemble_mortar_matrices(
        sm.coords, sm.conn, sm.source_facets_a, sm.source_facets_b, surf_a, surf_b
    )

    assert np.allclose(_sum_matrix(M_aa), 0.25, atol=1e-6)
    assert np.allclose(_sum_matrix(M_ab), 0.25, atol=1e-6)


def test_mortar_full_overlap_area():
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

    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)
    sm = ff.build_surface_supermesh(surf_a, surf_b, tol=1e-8)

    M_aa, M_ab = ff.assemble_mortar_matrices(
        sm.coords, sm.conn, sm.source_facets_a, sm.source_facets_b, surf_a, surf_b
    )

    assert np.allclose(_sum_matrix(M_aa), 1.0, atol=1e-6)
    assert np.allclose(_sum_matrix(M_ab), 1.0, atol=1e-6)


def test_mortar_quad_shape_values_centroid():
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
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)

    supermesh_coords = np.array(
        [
            [0.25, 0.25, 0.0],
            [0.75, 0.25, 0.0],
            [0.25, 0.75, 0.0],
        ],
        dtype=float,
    )
    supermesh_conn = np.array([[0, 1, 2]], dtype=int)
    source_facets = np.array([0], dtype=int)

    M_aa, M_ab = ff.assemble_mortar_matrices(
        supermesh_coords,
        supermesh_conn,
        source_facets,
        source_facets,
        surf_a,
        surf_b,
    )
    dense_aa = _dense_matrix(M_aa)
    dense_ab = _dense_matrix(M_ab)

    centroid = np.mean(supermesh_coords, axis=0)
    xi = 2.0 * centroid[0] - 1.0
    eta = 2.0 * centroid[1] - 1.0
    Na = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )
    area = 0.5 * np.linalg.norm(np.cross(supermesh_coords[1] - supermesh_coords[0], supermesh_coords[2] - supermesh_coords[0]))
    expected = area * np.outer(Na, Na)
    assert np.allclose(dense_aa, expected, atol=1e-6)
    assert np.allclose(dense_ab, expected, atol=1e-6)


def test_mortar_tri6_shape_values_centroid():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    facets = np.array([[0, 1, 2, 3, 4, 5]], dtype=int)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)

    supermesh_coords = np.array(
        [
            [0.1, 0.1, 0.0],
            [0.6, 0.1, 0.0],
            [0.1, 0.6, 0.0],
        ],
        dtype=float,
    )
    supermesh_conn = np.array([[0, 1, 2]], dtype=int)
    source_facets = np.array([0], dtype=int)

    M_aa, M_ab = ff.assemble_mortar_matrices(
        supermesh_coords,
        supermesh_conn,
        source_facets,
        source_facets,
        surf_a,
        surf_b,
    )
    dense_aa = _dense_matrix(M_aa)
    dense_ab = _dense_matrix(M_ab)

    centroid = np.mean(supermesh_coords, axis=0)
    L1 = 1.0 - centroid[0] - centroid[1]
    L2 = centroid[0]
    L3 = centroid[1]
    Na = np.array(
        [
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            4.0 * L1 * L2,
            4.0 * L2 * L3,
            4.0 * L1 * L3,
        ],
        dtype=float,
    )
    area = 0.5 * np.linalg.norm(np.cross(supermesh_coords[1] - supermesh_coords[0], supermesh_coords[2] - supermesh_coords[0]))
    expected = area * np.outer(Na, Na)
    assert np.allclose(dense_aa, expected, atol=1e-6)
    assert np.allclose(dense_ab, expected, atol=1e-6)


def test_mortar_quad8_shape_values_centroid():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    facets = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)

    supermesh_coords = np.array(
        [
            [0.2, 0.3, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=float,
    )
    supermesh_conn = np.array([[0, 1, 2]], dtype=int)
    source_facets = np.array([0], dtype=int)

    M_aa, M_ab = ff.assemble_mortar_matrices(
        supermesh_coords,
        supermesh_conn,
        source_facets,
        source_facets,
        surf_a,
        surf_b,
    )
    dense_aa = _dense_matrix(M_aa)
    dense_ab = _dense_matrix(M_ab)

    centroid = np.mean(supermesh_coords, axis=0)
    xi = 2.0 * centroid[0] - 1.0
    eta = 2.0 * centroid[1] - 1.0
    Na = np.array(
        [
            -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
            -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
            -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
            -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
            0.5 * (1.0 - xi * xi) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta * eta),
            0.5 * (1.0 - xi * xi) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta * eta),
        ],
        dtype=float,
    )
    area = 0.5 * np.linalg.norm(np.cross(supermesh_coords[1] - supermesh_coords[0], supermesh_coords[2] - supermesh_coords[0]))
    expected = area * np.outer(Na, Na)
    assert np.allclose(dense_aa, expected, atol=1e-6)
    assert np.allclose(dense_ab, expected, atol=1e-6)

"""Mixed surface sym_grad evaluation for Nitsche-like traction terms."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.mesh import mortar as mortar_mod
from fluxfem.physics import operators as ops


def _nitsche_bilin(v1, v2, u1, u2, p):
    n = h_wf.normal()
    t = h_wf.dot(h_wf.sym_grad(u1), p.c)
    return h_wf.dot(v1, t * n) * h_wf.ds()


def _assemble_contact(
    coords: np.ndarray,
    facets: np.ndarray,
    *,
    elem_conn: np.ndarray | None,
    quad_order: int,
    grad_source: str,
    dof_source: str,
) -> np.ndarray:
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)
    contact = ff.ContactSurfaceSpace.from_surfaces(
        surf_a,
        surf_b,
        elem_conn_master=elem_conn,
        elem_conn_slave=elem_conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=quad_order,
    )
    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = ff.Params(c=np.ones(6, dtype=float))
    J = contact.assemble_bilinear(
        _nitsche_bilin,
        u_a,
        u_b,
        params,
        grad_source=grad_source,
        dof_source=dof_source,
        normal_source="a",
    )
    return np.asarray(J)


def _expected_surface_sym_grad(
    coords: np.ndarray,
    facets: np.ndarray,
    params: ff.Params,
    *,
    quad_order: int,
) -> np.ndarray:
    surf = ff.SurfaceMesh.from_facets(coords, facets)
    sm = ff.build_surface_supermesh(surf, surf, tol=1e-8)
    if sm.conn.shape[0] == 0:
        return np.zeros((coords.shape[0] * 3, coords.shape[0] * 3), dtype=float)

    quad_pts, quad_w = mortar_mod._tri_quadrature(quad_order)
    facet = facets[0]
    n = surf.facet_normals()[0]
    n_nodes = coords.shape[0]
    expected = np.zeros((n_nodes * 3, n_nodes * 3), dtype=float)
    for tri, fa in zip(sm.conn, sm.source_facets_a):
        if int(fa) != 0:
            continue
        a, b, c = sm.coords[tri]
        area = mortar_mod._tri_area(a, b, c)
        if area <= 0.0:
            continue
        detJ = 2.0 * area
        for (r, s), w in zip(quad_pts, quad_w):
            x_q = a + r * (b - a) + s * (c - a)
            N = mortar_mod._facet_shape_values(x_q, facet, coords, tol=1e-12)
            gradN = mortar_mod._surface_gradN(x_q, facet, coords, tol=1e-12)[None, :, :]
            field = _DummyField(gradN=gradN, dofs_per_node=3)
            B = np.asarray(ops.sym_grad(field))[0]
            t_coeff = params.c @ B
            weight = float(w * detJ)
            for a_idx in range(len(facet)):
                for d in range(3):
                    row = a_idx * 3 + d
                    expected[row, :] += weight * N[a_idx] * n[d] * t_coeff
    return expected


def test_mixed_surface_sym_grad_tet4():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2]], dtype=int)
    J = _assemble_contact(
        coords,
        facets,
        elem_conn=conn,
        quad_order=1,
        grad_source="volume",
        dof_source="volume",
    )
    params = ff.Params(c=np.ones(6, dtype=float))

    centroid = np.mean(coords[facets[0]], axis=0)
    N = _tet_shape_values(centroid, coords)
    gradN = _tet_gradN(coords)[None, :, :]
    field = _DummyField(gradN=gradN, dofs_per_node=3)
    B = np.asarray(ops.sym_grad(field))[0]
    t_coeff = params.c @ B  # (12,)
    n = np.array([0.0, 0.0, 1.0], dtype=float)
    area = 0.5
    expected = np.zeros((12, 12), dtype=float)
    for a in range(4):
        for d in range(3):
            row = a * 3 + d
            expected[row, :] = area * N[a] * n[d] * t_coeff

    n_a = coords.shape[0] * 3
    assert np.any(np.abs(expected[:, 9:12]) > 0.0)
    assert np.allclose(J[:n_a, :n_a], expected, atol=1e-6)
    assert np.allclose(J[:n_a, n_a:], 0.0, atol=1e-12)
    assert np.allclose(J[n_a:, :n_a], 0.0, atol=1e-12)
    assert np.allclose(J[n_a:, n_a:], 0.0, atol=1e-12)


def test_mixed_surface_sym_grad_tri6():
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
    J = _assemble_contact(
        coords,
        facets,
        elem_conn=None,
        quad_order=1,
        grad_source="surface",
        dof_source="surface",
    )
    params = ff.Params(c=np.ones(6, dtype=float))

    expected = _expected_surface_sym_grad(coords, facets, params, quad_order=1)

    n_a = coords.shape[0] * 3
    assert np.any(np.abs(expected[9:, :]) > 0.0)
    assert np.allclose(J[:n_a, :n_a], expected, atol=1e-6)
    assert np.allclose(J[:n_a, n_a:], 0.0, atol=1e-12)
    assert np.allclose(J[n_a:, :n_a], 0.0, atol=1e-12)
    assert np.allclose(J[n_a:, n_a:], 0.0, atol=1e-12)


def test_mixed_surface_sym_grad_quad8():
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
    J = _assemble_contact(
        coords,
        facets,
        elem_conn=None,
        quad_order=1,
        grad_source="surface",
        dof_source="surface",
    )
    params = ff.Params(c=np.ones(6, dtype=float))

    expected = _expected_surface_sym_grad(coords, facets, params, quad_order=1)

    n_a = coords.shape[0] * 3
    assert np.any(np.abs(expected[12:, :]) > 0.0)
    assert np.allclose(J[:n_a, :n_a], expected, atol=1e-6)
    assert np.allclose(J[:n_a, n_a:], 0.0, atol=1e-12)
    assert np.allclose(J[n_a:, :n_a], 0.0, atol=1e-12)
    assert np.allclose(J[n_a:, n_a:], 0.0, atol=1e-12)


class _DummyBasis:
    def __init__(self, dofs_per_node: int):
        self.dofs_per_node = dofs_per_node


class _DummyField:
    def __init__(self, gradN: np.ndarray, dofs_per_node: int):
        self.gradN = jnp.asarray(gradN)
        self.basis = _DummyBasis(dofs_per_node)


def _tet_shape_values(point: np.ndarray, elem_coords: np.ndarray) -> np.ndarray:
    M = np.stack([elem_coords[:, 0], elem_coords[:, 1], elem_coords[:, 2], np.ones(4)], axis=1)
    rhs = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    lam = np.linalg.solve(M.T, rhs)
    return lam


def _tet_gradN(elem_coords: np.ndarray) -> np.ndarray:
    M = np.stack([elem_coords[:, 0], elem_coords[:, 1], elem_coords[:, 2], np.ones(4)], axis=1)
    invM = np.linalg.inv(M)
    return invM[:3, :].T


def _tri6_shape_values(point: np.ndarray) -> np.ndarray:
    x, y = point[0], point[1]
    L1 = 1.0 - x - y
    L2 = x
    L3 = y
    return np.array(
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


def _tri6_gradN(point: np.ndarray) -> np.ndarray:
    x, y = point[0], point[1]
    L1 = 1.0 - x - y
    L2 = x
    L3 = y
    g1 = np.array([-1.0, -1.0, 0.0], dtype=float)
    g2 = np.array([1.0, 0.0, 0.0], dtype=float)
    g3 = np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array(
        [
            (4.0 * L1 - 1.0) * g1,
            (4.0 * L2 - 1.0) * g2,
            (4.0 * L3 - 1.0) * g3,
            4.0 * (L1 * g2 + L2 * g1),
            4.0 * (L2 * g3 + L3 * g2),
            4.0 * (L1 * g3 + L3 * g1),
        ],
        dtype=float,
    )


def _quad8_shape_values(point: np.ndarray) -> np.ndarray:
    x, y = point[0], point[1]
    xi = 2.0 * x - 1.0
    eta = 2.0 * y - 1.0
    return np.array(
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


def _quad8_gradN(point: np.ndarray) -> np.ndarray:
    x, y = point[0], point[1]
    xi = 2.0 * x - 1.0
    eta = 2.0 * y - 1.0
    dN_dxi = np.array(
        [
            -0.25 * (1.0 - eta) * ((1.0 - xi) - (1.0 + xi + eta)),
            -0.25 * (1.0 - eta) * ((1.0 + xi) - (1.0 - xi + eta)),
            -0.25 * (1.0 + eta) * ((1.0 + xi) - (1.0 - xi - eta)),
            -0.25 * (1.0 + eta) * ((1.0 - xi) - (1.0 + xi - eta)),
            -xi * (1.0 - eta),
            0.5 * (1.0 - eta * eta),
            -xi * (1.0 + eta),
            -0.5 * (1.0 - eta * eta),
        ],
        dtype=float,
    )
    dN_deta = np.array(
        [
            -0.25 * (1.0 - xi) * ((1.0 - eta) - (1.0 + xi + eta)),
            -0.25 * (1.0 + xi) * ((1.0 - eta) - (1.0 - xi + eta)),
            -0.25 * (1.0 + xi) * ((1.0 + eta) - (1.0 - xi - eta)),
            -0.25 * (1.0 - xi) * ((1.0 + eta) - (1.0 + xi - eta)),
            -0.5 * (1.0 - xi * xi),
            -(1.0 + xi) * eta,
            0.5 * (1.0 - xi * xi),
            -(1.0 - xi) * eta,
        ],
        dtype=float,
    )
    return np.stack([2.0 * dN_dxi, 2.0 * dN_deta, np.zeros_like(dN_dxi)], axis=1)

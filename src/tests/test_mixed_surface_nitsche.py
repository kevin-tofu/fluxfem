"""Mixed surface sym_grad evaluation for Nitsche-like traction terms.

Compares the assembled contact bilinear form against a hand-derived tet4 Jacobian.
"""
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.physics import operators as ops


def _nitsche_bilin(v1, v2, u1, u2, p):
    # Minimal Nitsche-like traction term for the contact interface.
    n = h_wf.normal()
    t = h_wf.dot(h_wf.sym_grad(u1), p.c)
    return h_wf.dot(v1, t * n) * h_wf.ds()


def _assemble_contact_bilinear(
    coords: np.ndarray,
    facets: np.ndarray,
    *,
    elem_conn: np.ndarray | None,
    quad_order: int,
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
        normal_source="a",
    )
    return np.asarray(J)


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
    J = _assemble_contact_bilinear(
        coords,
        facets,
        elem_conn=conn,
        quad_order=1,
    )
    params = ff.Params(c=np.ones(6, dtype=float))

    centroid = np.mean(coords[facets[0]], axis=0)
    N = _tet_shape_values(centroid, coords)
    basis = ff.make_tet_basis(intorder=1)
    gradN, _ = basis.spatial_grads_and_detJ(jnp.asarray(coords[conn[0]]))
    field = SimpleNamespace(
        gradN=gradN,
        basis=SimpleNamespace(dofs_per_node=3),
    )
    B = np.asarray(ops.sym_grad(field))[0]
    t_coeff = params.c @ B  # (12,)
    n = np.array([0.0, 0.0, 1.0], dtype=float)
    area = 0.5
    expected = np.zeros((12, 12), dtype=float)
    # Expected matrix uses hand-derived tet4 shape values and gradients.
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


def _tet_shape_values(point: np.ndarray, elem_coords: np.ndarray) -> np.ndarray:
    M = np.stack([elem_coords[:, 0], elem_coords[:, 1], elem_coords[:, 2], np.ones(4)], axis=1)
    rhs = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    lam = np.linalg.solve(M.T, rhs)
    return lam

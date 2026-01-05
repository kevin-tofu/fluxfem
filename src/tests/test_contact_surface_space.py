"""ContactSurfaceSpace bilinear wrapper matches mixed surface assembly."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def test_contact_surface_bilinear_wrapper():
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
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)

    contact = ff.ContactSurfaceSpace.from_surfaces(
        surf_a,
        surf_b,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )

    def bilin(v1, v2, u1, u2, p):
        ju = u1.val - u2.val
        term = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        return term * h_wf.ds()

    def res_a(v, u, p):
        u2 = ff.unknown_ref("b")
        ju = u.val - u2.val
        return (p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    def res_b(v, u, p):
        u1 = ff.unknown_ref("a")
        ju = u1.val - u.val
        return -(p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    res_form = ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    J_bilin = contact.assemble_bilinear(bilin, u_a, u_b, params)
    J_res = contact.assemble_jacobian(res_form, {"a": u_a, "b": u_b}, params)

    assert np.allclose(np.asarray(J_bilin), np.asarray(J_res), atol=1e-10)


def test_contact_surface_bilinear_tet10_mid_edge_dofs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [0.5, 0.0, 0.0],  # 4 (0-1)
            [0.5, 0.5, 0.0],  # 5 (1-2)
            [0.0, 0.5, 0.0],  # 6 (0-2)
            [0.0, 0.0, 0.5],  # 7 (0-3)
            [0.5, 0.0, 0.5],  # 8 (1-3)
            [0.0, 0.5, 0.5],  # 9 (2-3)
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=int)
    facets = np.array([[0, 1, 2]], dtype=int)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)

    contact = ff.ContactSurfaceSpace.from_surfaces(
        surf_a,
        surf_b,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )

    def bilin(v1, v2, u1, u2, p):
        ju = u1.val - u2.val
        term = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        return term * h_wf.ds()

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    J = np.asarray(contact.assemble_bilinear(bilin, u_a, u_b, params))
    n_dofs = coords.shape[0] * 3
    assert J.shape == (2 * n_dofs, 2 * n_dofs)
    # Edge-midpoint nodes on the face (4-6) should contribute.
    mid_edge_slice = slice(4 * 3, 7 * 3)  # nodes 4,5,6
    assert np.max(np.abs(J[:n_dofs, :n_dofs][mid_edge_slice, :])) > 0.0


def test_contact_surface_bilinear_hex8_face_dofs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    facets = np.array([[0, 1, 2, 3]], dtype=int)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)

    contact = ff.ContactSurfaceSpace.from_surfaces(
        surf_a,
        surf_b,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )

    def bilin(v1, v2, u1, u2, p):
        ju = u1.val - u2.val
        term = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        return term * h_wf.ds()

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    J = np.asarray(contact.assemble_bilinear(bilin, u_a, u_b, params))
    n_dofs = coords.shape[0] * 3
    assert J.shape == (2 * n_dofs, 2 * n_dofs)
    # Top face nodes (4-7) should not contribute on the z=0 interface.
    top_slice = slice(4 * 3, 8 * 3)
    assert np.max(np.abs(J[:n_dofs, :n_dofs][top_slice, :])) < 1e-8


def test_contact_surface_bilinear_hex20_edge_dofs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            [0.5, 0.0, 0.0],  # 8 (0-1)
            [1.0, 0.5, 0.0],  # 9 (1-2)
            [0.5, 1.0, 0.0],  # 10 (2-3)
            [0.0, 0.5, 0.0],  # 11 (3-0)
            [0.5, 0.0, 1.0],  # 12 (4-5)
            [1.0, 0.5, 1.0],  # 13 (5-6)
            [0.5, 1.0, 1.0],  # 14 (6-7)
            [0.0, 0.5, 1.0],  # 15 (7-4)
            [0.0, 0.0, 0.5],  # 16 (0-4)
            [1.0, 0.0, 0.5],  # 17 (1-5)
            [1.0, 1.0, 0.5],  # 18 (2-6)
            [0.0, 1.0, 0.5],  # 19 (3-7)
        ],
        dtype=float,
    )
    conn = np.array([[i for i in range(20)]], dtype=int)
    facets = np.array([[0, 1, 2, 3]], dtype=int)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)

    contact = ff.ContactSurfaceSpace.from_surfaces(
        surf_a,
        surf_b,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )

    def bilin(v1, v2, u1, u2, p):
        ju = u1.val - u2.val
        term = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        return term * h_wf.ds()

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    J = np.asarray(contact.assemble_bilinear(bilin, u_a, u_b, params))
    n_dofs = coords.shape[0] * 3
    assert J.shape == (2 * n_dofs, 2 * n_dofs)
    edge_slice = slice(8 * 3, 12 * 3)  # bottom face edge mids
    assert np.max(np.abs(J[:n_dofs, :n_dofs][edge_slice, :])) > 0.0

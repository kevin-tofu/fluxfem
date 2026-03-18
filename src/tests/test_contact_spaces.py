import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def _penalty_bilinear(v1, v2, u1, u2, p):
    ju = u1.val - u2.val
    return ((p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))) * h_wf.ds()


def test_contact_spaces_builds_pair_contact_surface_space():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf_s, elem_conn=conn, value_dim=3)

    spec = ff.ContactSpaces(master=side_m, slave=side_s, field_master="master", field_slave="slave")
    contact = spec.to_contact_surface_space(quad_order=1, backend="jax")

    assert isinstance(contact, ff.ContactSurfaceSpace)
    assert contact.field_master == "master"
    assert contact.field_slave == "slave"
    assert contact.batch_jac is None


def test_contact_spaces_preserves_explicit_batch_jac_flag():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf_s, elem_conn=conn, value_dim=3)

    contact = ff.ContactSpaces(master=side_m, slave=side_s).to_contact_surface_space(
        quad_order=1,
        backend="jax",
        batch_jac=False,
    )

    assert contact.batch_jac is False


def test_contact_spaces_matches_direct_from_sides_bilinear():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf_s, elem_conn=conn, value_dim=3)

    direct = ff.ContactSurfaceSpace.from_sides(side_m, side_s, quad_order=1, backend="jax")
    via_spec = ff.ContactSpaces(master=side_m, slave=side_s).to_contact_surface_space(
        quad_order=1,
        backend="jax",
    )

    n = coords.shape[0] * 3
    u_m = jnp.zeros(n)
    u_s = jnp.zeros(n)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    K_direct = np.asarray(direct.assemble_bilinear(_penalty_bilinear, u_m, u_s, params))
    K_spec = np.asarray(via_spec.assemble_bilinear(_penalty_bilinear, u_m, u_s, params))

    assert np.allclose(K_spec, K_direct, atol=1e-10)


def test_contact_group_spaces_matches_direct_one_to_many_bilinear():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s1 = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s2 = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s1 = ff.ContactSide.from_surfaces(surf_s1, elem_conn=conn, value_dim=3)
    side_s2 = ff.ContactSide.from_surfaces(surf_s2, elem_conn=conn, value_dim=3)

    direct = ff.OneToManyContactSurfaceSpace.from_sides(
        side_m,
        [side_s1, side_s2],
        quad_order=1,
        backend="jax",
    )
    via_spec = ff.ContactGroupSpaces(master=side_m, slaves=[side_s1, side_s2]).to_contact_surface_space(
        quad_order=1,
        backend="jax",
    )

    n = coords.shape[0] * 3
    u_m = jnp.zeros(n)
    u_s = jnp.zeros(n)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    K_direct = np.asarray(direct.assemble_bilinear(_penalty_bilinear, u_m, [u_s, u_s], params))
    K_spec = np.asarray(via_spec.assemble_bilinear(_penalty_bilinear, u_m, [u_s, u_s], params))

    assert np.allclose(K_spec, K_direct, atol=1e-10)


def test_onesided_contact_spaces_matches_direct_from_side():
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

    surf = ff.SurfaceMesh.from_facets(coords, facets)
    side = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)

    direct = ff.OneSidedContactSurfaceSpace.from_side(side, quad_order=2)
    via_spec = ff.OneSidedContactSpaces(side=side).to_contact_surface_space(quad_order=2)

    assert isinstance(via_spec, ff.OneSidedContactSurfaceSpace)
    assert np.array_equal(via_spec.facet_to_elem_slave, direct.facet_to_elem_slave)
    assert via_spec.quad_order == direct.quad_order

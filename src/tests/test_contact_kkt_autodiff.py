"""Autodiff regression tests for contact KKT assembly."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import fluxfem as ff


def _tet4_fixture():
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
    return coords, conn, facets


def _p0_multiplier(contact):
    return ff.MultiplierSpec.from_contact(contact, family="p0", side="master")


def _nodal_multiplier():
    return ff.MultiplierSpec(family="nodal")


def _dual_nodal_multiplier():
    return ff.MultiplierSpec(family="dual_nodal", side="master")


def _coupling_to_dense(coupling):
    dense = np.zeros(coupling.shape, dtype=float)
    dense[np.asarray(coupling.rows, dtype=int), np.asarray(coupling.cols, dtype=int)] += np.asarray(
        coupling.data,
        dtype=float,
    )
    return dense


def test_contact_kkt_matches_module_and_class_api():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    m_aa, m_ab = contact.assemble_contact_coupling_matrices()

    K_a = np.asarray(
        ff.assemble_contact_kkt(
            m_aa,
            m_ab,
            rho=3.0,
            multiplier=_p0_multiplier(contact),
            facet_conn_master=facets,
            backend="numpy",
            format="dense",
        )
    )
    K_b = np.asarray(contact.assemble_contact_kkt(rho=3.0, multiplier=_p0_multiplier(contact), backend="numpy", format="dense"))
    assert np.allclose(K_a, K_b, atol=1e-12)

    K_flux = contact.assemble_contact_kkt(rho=3.0, multiplier=_p0_multiplier(contact), backend="numpy")
    assert hasattr(K_flux, "to_bcoo")
    K_flux_dense = np.asarray(K_flux.to_dense())
    assert np.allclose(K_a, K_flux_dense, atol=1e-12)

    K_bcoo = contact.assemble_contact_kkt(rho=3.0, multiplier=_p0_multiplier(contact), backend="jax", format="bcoo")
    K_bcoo_dense = np.asarray(K_bcoo.todense())
    assert np.allclose(K_a, K_bcoo_dense, atol=1e-12)


def test_contact_kkt_auto_backend_prefers_jax_for_mixed_inputs():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    m_aa, m_ab = contact.assemble_contact_coupling_matrices()

    K = ff.assemble_contact_kkt(
        m_aa,
        m_ab,
        rho=jnp.array(3.0),
        multiplier=_p0_multiplier(contact),
        facet_conn_master=facets,
        format="dense",
    )

    assert isinstance(K, jax.Array)
    assert np.allclose(np.asarray(K), np.asarray(ff.assemble_contact_kkt(
        m_aa,
        m_ab,
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        facet_conn_master=facets,
        backend="numpy",
        format="dense",
    )), atol=1e-12)


def test_solve_contact_kkt_auto_backend_prefers_jax_for_mixed_inputs():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    K = contact.assemble_contact_kkt(
        rho=jnp.array(3.0),
        multiplier=_p0_multiplier(contact),
        format="dense",
    )
    rhs = np.linspace(0.2, 1.0, int(K.shape[0]))

    u = ff.solve_contact_kkt(K, rhs, diagonal_shift=1e-2)

    assert isinstance(u, jax.Array)


def test_contact_constraint_operators_match_kkt_blocks():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )

    ops = ff.assemble_multiplier(
        contact,
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        backend="numpy",
    )
    K_dense, B_a_ref, B_b_ref = contact.assemble_contact_kkt(
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        backend="numpy",
        format="dense",
        return_blocks=True,
    )

    assert np.allclose(np.asarray(ops.B_a), np.asarray(B_a_ref), atol=1e-12)
    assert np.allclose(np.asarray(ops.B_b), np.asarray(B_b_ref), atol=1e-12)
    assert np.allclose(np.asarray(ops.B), np.concatenate([np.asarray(B_a_ref), -np.asarray(B_b_ref)], axis=1), atol=1e-12)

    n_u = int(np.asarray(ops.B).shape[1])
    assert np.allclose(np.asarray(ops.Kuu), np.asarray(K_dense)[:n_u, :n_u], atol=1e-12)


def test_dual_nodal_contact_multiplier_builds_biorthogonal_blocks():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    m_aa, m_ab = contact.assemble_contact_coupling_matrices()
    M_aa = _coupling_to_dense(m_aa)
    M_ab = _coupling_to_dense(m_ab)

    ops = ff.assemble_multiplier(
        contact,
        rho=2.5,
        multiplier=_dual_nodal_multiplier(),
        backend="numpy",
    )

    assert ops.multiplier.family == "dual_nodal"
    assert np.allclose(np.asarray(ops.B_a), np.eye(M_aa.shape[0]), atol=1e-12)
    assert np.allclose(M_aa @ np.asarray(ops.B_b), M_ab, atol=1e-12)

    K_dense, B_a_ref, B_b_ref = contact.assemble_contact_kkt(
        rho=2.5,
        multiplier=_dual_nodal_multiplier(),
        backend="numpy",
        format="dense",
        return_blocks=True,
    )
    assert np.allclose(np.asarray(ops.B_a), np.asarray(B_a_ref), atol=1e-12)
    assert np.allclose(np.asarray(ops.B_b), np.asarray(B_b_ref), atol=1e-12)
    assert np.allclose(np.asarray(ops.Kuu), np.asarray(K_dense)[: ops.B.shape[1], : ops.B.shape[1]], atol=1e-12)


def test_dual_nodal_is_default_mortar_multiplier():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )

    assert ff.MultiplierSpec().family == "dual_nodal"
    assert ff.MultiplierSpec.from_contact(contact).family == "dual_nodal"

    ops_default = ff.assemble_multiplier(contact, rho=2.0, backend="numpy")
    ops_dual = ff.assemble_multiplier(contact, rho=2.0, multiplier=_dual_nodal_multiplier(), backend="numpy")
    assert ops_default.multiplier.family == "dual_nodal"
    assert np.allclose(np.asarray(ops_default.B), np.asarray(ops_dual.B), atol=1e-12)
    assert np.allclose(np.asarray(ops_default.Kuu), np.asarray(ops_dual.Kuu), atol=1e-12)

    m_aa, m_ab = contact.assemble_contact_coupling_matrices()
    K_default = ff.assemble_contact_kkt(m_aa, m_ab, rho=2.0, backend="numpy", format="dense")
    K_dual = ff.assemble_contact_kkt(
        m_aa,
        m_ab,
        rho=2.0,
        multiplier=_dual_nodal_multiplier(),
        backend="numpy",
        format="dense",
    )
    assert np.allclose(np.asarray(K_default), np.asarray(K_dual), atol=1e-12)


def test_dual_nodal_contact_kkt_sparse_formats_match_dense():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    mult = _dual_nodal_multiplier()

    K_dense = np.asarray(contact.assemble_contact_kkt(rho=1.25, multiplier=mult, backend="numpy", format="dense"))
    K_flux = contact.assemble_contact_kkt(rho=1.25, multiplier=mult, backend="numpy", format="fluxsparse")
    K_bcoo = contact.assemble_contact_kkt(rho=1.25, multiplier=mult, backend="jax", format="bcoo")

    assert np.allclose(np.asarray(K_flux.to_dense()), K_dense, atol=1e-12)
    assert np.allclose(np.asarray(K_bcoo.todense()), K_dense, atol=1e-12)


def test_coarse_mortar_rank_projection_reduces_multiplier_rows_and_matches_sparse():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    mult = ff.MultiplierSpec(family="dual_nodal", coarse_rank=1)

    ops = ff.assemble_multiplier(contact, rho=1.5, multiplier=mult, backend="numpy")
    assert ops.B.shape[0] == 1
    assert ops.B_a.shape[0] == 1
    assert ops.B_b.shape[0] == 1

    K_dense = np.asarray(contact.assemble_contact_kkt(rho=1.5, multiplier=mult, backend="numpy", format="dense"))
    K_sparse = contact.assemble_contact_kkt(rho=1.5, multiplier=mult, backend="numpy", format="fluxsparse")
    assert K_dense.shape[-1] == int(ops.B.shape[1] + 1)
    assert np.allclose(np.asarray(K_sparse.to_dense()), K_dense, atol=1e-12)


def test_coarse_dual_mortar_auto_selects_rank_without_user_rank():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.coarse_dual_mortar(max_rank=2)

    assert mult.family == "dual_nodal"
    assert mult.coarse_rank is None
    assert mult.coarse_mode == "auto"

    ops = ff.assemble_multiplier(contact, rho=1.5, multiplier=mult, backend="numpy")
    K_dense = np.asarray(contact.assemble_contact_kkt(rho=1.5, multiplier=mult, backend="numpy", format="dense"))

    assert 1 <= int(ops.B.shape[0]) <= 2
    assert K_dense.shape[-1] == int(ops.B.shape[1] + ops.B.shape[0])


def test_multiplier_factory_constructors_cover_common_mortar_choices():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )

    assert ff.MultiplierSpec.dual_mortar().family == "dual_nodal"
    assert ff.MultiplierSpec.nodal_mortar().family == "nodal"
    assert ff.MultiplierSpec.coarse_p1_mortar(basis=np.eye(4)).family == "coarse_p1"
    p0 = ff.MultiplierSpec.p0_mortar(contact)
    assert p0.family == "p0"
    assert p0.facet_conn is not None
    with pytest.raises(ValueError, match="contact or facet_conn"):
        ff.MultiplierSpec.p0_mortar()


def test_integrated_coarse_p1_mortar_projects_nodal_integral_rows():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    basis = ff.coarse_p1_basis_from_node_groups(4, [[0, 2], [1, 3]])
    fine = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.nodal_mortar(),
        backend="numpy",
    )
    coarse = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.coarse_p1_mortar(basis=basis),
        backend="numpy",
    )

    assert coarse.multiplier.family == "coarse_p1"
    assert ff.coarse_p1_basis_from_node_groups is ff.mesh.coarse_p1_basis_from_node_groups
    np.testing.assert_allclose(coarse.multiplier.coarse_basis, basis)
    assert coarse.B.shape == (2, 8)
    np.testing.assert_allclose(np.asarray(coarse.B), basis @ np.asarray(fine.B))
    np.testing.assert_allclose(np.asarray(coarse.B_a), basis @ np.asarray(fine.B_a))
    np.testing.assert_allclose(np.asarray(coarse.B_b), basis @ np.asarray(fine.B_b))

    K_dense = np.asarray(contact.assemble_contact_kkt(rho=1.25, multiplier=coarse.multiplier, backend="numpy", format="dense"))
    K_sparse = contact.assemble_contact_kkt(rho=1.25, multiplier=coarse.multiplier, backend="numpy", format="fluxsparse")
    assert K_dense.shape[-1] == int(coarse.B.shape[1] + coarse.B.shape[0])
    np.testing.assert_allclose(np.asarray(K_sparse.to_dense()), K_dense, atol=1e-12)


def test_integrated_coarse_p1_mortar_supports_vector_components():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )
    basis = ff.coarse_p1_basis_from_node_groups(4, [[0, 1, 2, 3]])
    coarse = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.coarse_p1_mortar(basis=basis, value_dim=3),
        backend="numpy",
    )

    assert coarse.B.shape == (3, 24)
    assert coarse.B_a.shape == (3, 12)
    assert coarse.B_b.shape == (3, 12)


def test_coarse_p1_basis_from_node_groups_supports_weights_and_validation():
    basis = ff.coarse_p1_basis_from_node_groups(
        5,
        [[0, 1, 3], [2, 4]],
        weights=[[2.0, 1.0, 1.0], [3.0, 1.0]],
    )
    expected = np.array(
        [
            [0.5, 0.25, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.75, 0.0, 0.25],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(basis, expected)
    unnormalized = ff.coarse_p1_basis_from_node_groups(
        3,
        [[0, 2]],
        weights=[[2.0, 3.0]],
        normalize=False,
    )
    np.testing.assert_allclose(unnormalized, np.array([[2.0, 0.0, 3.0]]))
    with pytest.raises(ValueError, match="out-of-range"):
        ff.coarse_p1_basis_from_node_groups(3, [[0, 3]])
    with pytest.raises(ValueError, match="zero weight sum"):
        ff.coarse_p1_basis_from_node_groups(3, [[0, 1]], weights=[[1.0, -1.0]])
    with pytest.raises(ValueError, match="same number of rows"):
        ff.coarse_p1_basis_from_node_groups(3, [[0], [1]], weights=[[1.0]])


def test_coarse_p1_basis_from_surface_grid_builds_bilinear_rows():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    surface = ff.make_surface_from_facets(coords, np.array([[0, 1, 3], [0, 3, 2]], dtype=int))
    basis = ff.coarse_p1_basis_from_surface_grid(surface, shape=(2, 2), axes=(0, 1))
    expected = np.eye(4, dtype=float)
    expected = expected[[0, 1, 2, 3], :]

    assert ff.coarse_p1_basis_from_surface_grid is ff.mesh.coarse_p1_basis_from_surface_grid
    np.testing.assert_allclose(basis, expected)
    np.testing.assert_allclose(np.sum(basis, axis=0), np.ones(4))

    with_center = np.vstack([coords, np.array([[0.5, 0.5, 0.0]], dtype=float)])
    basis3 = ff.coarse_p1_basis_from_surface_grid(with_center, shape=(3, 3), axes=(0, 1))
    assert basis3.shape == (9, 5)
    np.testing.assert_allclose(basis3[:, 4], np.eye(9)[4])
    np.testing.assert_allclose(np.sum(basis3, axis=0), np.ones(5))


def test_coarse_p1_basis_from_surface_grid_supports_bounds_and_validation():
    coords = np.array([[0.25, 0.25, 0.0]], dtype=float)
    basis = ff.coarse_p1_basis_from_surface_grid(
        coords,
        shape=(2, 2),
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    np.testing.assert_allclose(basis[:, 0], np.array([0.5625, 0.1875, 0.1875, 0.0625]))
    with pytest.raises(ValueError, match="outside"):
        ff.coarse_p1_basis_from_surface_grid(coords, shape=(2, 2), bounds=((0.5, 1.0), (0.5, 1.0)), clamp=False)
    with pytest.raises(ValueError, match="at least"):
        ff.coarse_p1_basis_from_surface_grid(coords, shape=(1, 2))
    with pytest.raises(ValueError, match="distinct"):
        ff.coarse_p1_basis_from_surface_grid(coords, shape=(2, 2), axes=(0, 0))


def test_integrated_coarse_p1_mortar_rejects_invalid_basis_shape():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    with pytest.raises(ValueError, match="n_coarse_nodes, n_master_nodes"):
        ff.assemble_multiplier(
            contact,
            rho=0.0,
            multiplier=ff.MultiplierSpec.coarse_p1_mortar(basis=np.ones((1, 3))),
            backend="numpy",
        )
    with pytest.raises(NotImplementedError, match="side='master'"):
        ff.MultiplierSpec.coarse_p1_mortar(basis=np.ones((1, 4)), side="slave")


def test_integrated_coarse_p0_mortar_groups_facet_integral_rows():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    fine = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.p0_mortar(contact),
        backend="numpy",
    )
    coarse = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.coarse_p0_mortar(contact, patch_ids=np.array([0, 0])),
        backend="numpy",
    )

    assert coarse.multiplier.family == "p0"
    np.testing.assert_array_equal(coarse.multiplier.coarse_patch_ids, np.array([0, 0]))
    assert coarse.B.shape[0] == 1
    np.testing.assert_allclose(np.asarray(coarse.B), np.asarray(fine.B)[[0], :] + np.asarray(fine.B)[[1], :])
    np.testing.assert_allclose(np.asarray(coarse.B_a), np.asarray(fine.B_a)[[0], :] + np.asarray(fine.B_a)[[1], :])
    np.testing.assert_allclose(np.asarray(coarse.B_b), np.asarray(fine.B_b)[[0], :] + np.asarray(fine.B_b)[[1], :])

    K_dense = np.asarray(contact.assemble_contact_kkt(rho=1.25, multiplier=coarse.multiplier, backend="numpy", format="dense"))
    K_sparse = contact.assemble_contact_kkt(rho=1.25, multiplier=coarse.multiplier, backend="numpy", format="fluxsparse")
    assert K_dense.shape[-1] == int(coarse.B.shape[1] + coarse.B.shape[0])
    np.testing.assert_allclose(np.asarray(K_sparse.to_dense()), K_dense, atol=1e-12)


def test_integrated_coarse_p0_mortar_keeps_distinct_patch_rows():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.coarse_p0_mortar(contact, patch_ids=np.array([0, 1]), value_dim=3)
    ops = ff.assemble_multiplier(contact, rho=0.0, multiplier=mult, backend="numpy")

    assert ops.B.shape == (6, 24)
    assert ops.B_a.shape == (6, 12)
    assert ops.B_b.shape == (6, 12)


def test_integrated_coarse_p0_active_and_supermesh_group_integral_rows():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )

    fine_active = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.from_contact(contact, family="p0_active", side="master", value_dim=3),
        backend="numpy",
    )
    n_active = int(np.unique(contact.source_facets_master).shape[0])
    coarse_active = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.coarse_p0_mortar(
            contact,
            patch_ids=np.zeros(n_active, dtype=int),
            family="p0_active",
            value_dim=3,
        ),
        backend="numpy",
    )
    assert coarse_active.B.shape[0] == 3
    np.testing.assert_allclose(
        np.asarray(coarse_active.B),
        np.sum(np.asarray(fine_active.B).reshape(n_active, 3, -1), axis=0),
    )

    n_tri = int(contact.supermesh_conn.shape[0])
    fine_super = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.from_contact(contact, family="p0_supermesh", side="master", value_dim=3),
        backend="numpy",
    )
    coarse_super = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=ff.MultiplierSpec.coarse_p0_mortar(
            contact,
            patch_ids=np.zeros(n_tri, dtype=int),
            family="p0_supermesh",
            value_dim=3,
        ),
        backend="numpy",
    )
    assert coarse_super.B.shape[0] == 3
    np.testing.assert_allclose(
        np.asarray(coarse_super.B),
        np.sum(np.asarray(fine_super.B).reshape(n_tri, 3, -1), axis=0),
    )


def test_integrated_coarse_p0_mortar_rejects_invalid_patch_count():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.coarse_p0_mortar(contact, patch_ids=np.array([0, 1]))
    with pytest.raises(ValueError, match="one entry per fine P0 multiplier row"):
        ff.assemble_multiplier(contact, rho=0.0, multiplier=mult, backend="numpy")


def test_coarse_mortar_explicit_projection_reduces_multiplier_rows():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    P = np.zeros((1, 4), dtype=float)
    P[0, 0] = 1.0
    mult = ff.MultiplierSpec(family="dual_nodal", coarse_projection=P)

    ops = ff.assemble_multiplier(contact, rho=1.0, multiplier=mult, backend="numpy")
    _, B_a_ref, B_b_ref = contact.assemble_contact_kkt(
        rho=1.0,
        multiplier=mult,
        backend="numpy",
        format="dense",
        return_blocks=True,
    )

    assert ops.B.shape[0] == 1
    assert np.allclose(np.asarray(ops.B_a), np.asarray(B_a_ref), atol=1e-12)
    assert np.allclose(np.asarray(ops.B_b), np.asarray(B_b_ref), atol=1e-12)


def test_dual_nodal_contact_multiplier_rejects_slave_side_until_slave_mass_is_available():
    with pytest.raises(NotImplementedError, match="side='master'"):
        ff.assemble_contact_kkt(
            ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            ),
            ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            ),
            multiplier=ff.MultiplierSpec(family="dual_nodal", side="slave"),
            format="dense",
        )


def test_augmented_lagrangian_outer_loop_solves_scalar_equality_constraint():
    stiffness = 4.0
    force = 3.0
    target = 0.25

    def solve_subproblem(x0, state):
        _ = x0
        lam = jnp.asarray(state.lambda_values)[0]
        rho = state.rho
        x = (force - lam + rho * target) / (stiffness + rho)
        return jnp.array([x]), {"rho": rho}

    result = ff.solve_augmented_lagrangian_outer_loop(
        solve_subproblem,
        jnp.array([0.0]),
        constraint_fn=lambda x: x - jnp.array([target]),
        lambda0=jnp.array([0.0]),
        rho=1.0,
        maxiter=120,
        tol=1e-6,
        lambda_tol=1e-6,
    )

    assert isinstance(result.state, ff.AugmentedLagrangianState)
    assert isinstance(result, ff.AugmentedLagrangianResult)
    assert result.converged
    assert np.allclose(np.asarray(result.solution), np.array([target]), atol=1e-6)
    assert np.allclose(np.asarray(result.state.lambda_values), np.array([force - stiffness * target]), atol=1e-6)
    assert result.constraint_norm < 1e-6
    assert result.info["rho"] == 1.0


def test_augmented_lagrangian_outer_loop_accepts_contact_operator_B_path():
    stiffness = jnp.diag(jnp.array([3.0, 2.0]))
    force = jnp.array([1.0, -0.5])
    B = jnp.array([[1.0, 0.0]])
    target = jnp.array([0.1])
    ops = ff.MultiplierContactContribution(enforcement="mortar", B=B)

    def solve_subproblem(x0, state):
        _ = x0
        lam = jnp.asarray(state.lambda_values)
        rho = state.rho
        A = stiffness + rho * (B.T @ B)
        rhs = force - B.T @ lam + rho * (B.T @ target)
        return jnp.linalg.solve(A, rhs)

    result = ff.solve_augmented_lagrangian_outer_loop(
        solve_subproblem,
        jnp.zeros(2),
        operators=ops,
        offset=target,
        rho=1.0,
        maxiter=120,
        tol=1e-6,
        lambda_tol=1e-6,
    )

    assert result.converged
    assert np.allclose(np.asarray(B @ result.solution), np.asarray(target), atol=1e-6)


def test_augmented_lagrangian_outer_loop_nonnegative_projection_clips_multiplier():
    def solve_subproblem(x0, state):
        _ = (x0, state)
        return jnp.array([-1.0])

    result = ff.solve_augmented_lagrangian_outer_loop(
        solve_subproblem,
        jnp.array([0.0]),
        constraint_fn=lambda x: x,
        lambda0=jnp.array([0.0]),
        rho=2.0,
        maxiter=2,
        projection="nonnegative",
    )

    assert np.allclose(np.asarray(result.state.lambda_values), np.array([0.0]), atol=1e-12)
    assert np.asarray(result.state.active_mask).shape == (1,)


def test_contact_operator_method_aliases_match_existing_entrypoints():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    mult = _p0_multiplier(contact)

    ops_top = ff.assemble_multiplier(
        contact,
        rho=3.0,
        multiplier=mult,
        backend="numpy",
    )
    ops_method = contact.assemble_multiplier(
        rho=3.0,
        multiplier=mult,
        backend="numpy",
    )
    with pytest.warns(DeprecationWarning, match="assemble_constraint_operators"):
        ops_alias = contact.assemble_constraint_operators(
            rho=3.0,
            multiplier=mult,
            backend="numpy",
        )

    assert np.allclose(np.asarray(ops_top.B), np.asarray(ops_method.B), atol=1e-12)
    assert np.allclose(np.asarray(ops_top.B), np.asarray(ops_alias.B), atol=1e-12)
    assert np.allclose(np.asarray(ops_top.Kuu), np.asarray(ops_alias.Kuu), atol=1e-12)


def test_contact_multiplier_object_path_is_consistent():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.from_contact(contact, family="p0", side="master")

    ops_obj = ff.assemble_multiplier(
        contact,
        rho=3.0,
        multiplier=mult,
        backend="numpy",
    )
    ops_ref = ff.assemble_multiplier(
        contact,
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        backend="numpy",
    )
    assert np.allclose(np.asarray(ops_obj.B), np.asarray(ops_ref.B), atol=1e-12)
    assert np.allclose(np.asarray(ops_obj.Kuu), np.asarray(ops_ref.Kuu), atol=1e-12)
    assert isinstance(ops_obj.multiplier, ff.MultiplierSpec)

    m_aa, m_ab = contact.assemble_contact_coupling_matrices()
    K_obj = ff.assemble_contact_kkt(
        m_aa,
        m_ab,
        rho=3.0,
        multiplier=mult,
        backend="numpy",
        format="dense",
    )
    K_str = ff.assemble_contact_kkt(
        m_aa,
        m_ab,
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        facet_conn_master=facets,
        backend="numpy",
        format="dense",
    )
    assert np.allclose(np.asarray(K_obj), np.asarray(K_str), atol=1e-12)


def test_contact_constraint_operators_default_formulation_is_multiplier():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )

    ops_default = ff.assemble_multiplier(
        contact,
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        backend="numpy",
    )
    ops_formulation = ff.assemble_multiplier(
        contact,
        formulation="multiplier",
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        backend="numpy",
    )

    assert np.allclose(np.asarray(ops_default.B), np.asarray(ops_formulation.B), atol=1e-12)
    assert np.allclose(np.asarray(ops_default.Kuu), np.asarray(ops_formulation.Kuu), atol=1e-12)
    assert ops_formulation.enforcement == "mortar"
    assert ops_formulation.formulation == "multiplier"
    assert ops_formulation.law == "one_sided_normal_frictionless"


def test_contact_p0_multiplier_vector_value_dim_expands_blocks():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.from_contact(contact, family="p0", side="master", value_dim=3)

    ops = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=mult,
        backend="numpy",
    )
    assert np.asarray(ops.B_a).shape == (3, 12)
    assert np.asarray(ops.B_b).shape == (3, 12)
    assert np.asarray(ops.B).shape == (3, 24)

    K_dense = np.asarray(
        contact.assemble_contact_kkt(
            rho=0.0,
            multiplier=mult,
            backend="numpy",
            format="dense",
        )
    )
    assert K_dense.shape == (27, 27)


def test_contact_p0_multiplier_vector_value_dim_lifts_into_coupled_system():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.from_contact(contact, family="p0", side="master", value_dim=3)
    ops = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=mult,
        backend="numpy",
    )

    builder = ff.CoupledSystemBuilder.from_structural(np.eye(24), np.zeros(24))
    builder.register_field("a", n_dofs=12, value_dim=3, n_nodes=4)
    builder.register_field("b", n_dofs=12, value_dim=3, n_nodes=4)
    builder.add_contact_mortar(ops, master="a", slave="b", value_dim=3)
    K = builder.build().to_dense()
    assert K.shape == (27, 27)


def test_contact_p0_supermesh_multiplier_tracks_supermesh_triangles():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=np.array([[0, 1, 2, 3]], dtype=int),
        elem_conn_slave=np.array([[0, 1, 2, 3]], dtype=int),
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.from_contact(contact, family="p0_supermesh", side="master", value_dim=3)
    ops = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=mult,
        backend="numpy",
    )

    n_tri = int(contact.supermesh_conn.shape[0])
    assert np.asarray(ops.B_a).shape == (3 * n_tri, 3 * contact.surface_master.n_nodes)
    assert np.asarray(ops.B_b).shape == (3 * n_tri, 3 * contact.surface_slave.n_nodes)
    assert np.asarray(ops.B).shape == (3 * n_tri, 3 * (contact.surface_master.n_nodes + contact.surface_slave.n_nodes))

    builder = ff.CoupledSystemBuilder.from_structural(np.eye(24), np.zeros(24))
    builder.register_field("a", n_dofs=12, value_dim=3, n_nodes=4)
    builder.register_field("b", n_dofs=12, value_dim=3, n_nodes=4)
    builder.add_contact_mortar(ops, master="a", slave="b", value_dim=3)
    K = builder.build().to_dense()
    assert K.shape == (24 + 3 * n_tri, 24 + 3 * n_tri)


def test_contact_p0_active_multiplier_tracks_active_master_facets():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=np.array([[0, 1, 2, 3]], dtype=int),
        elem_conn_slave=np.array([[0, 1, 2, 3]], dtype=int),
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.from_contact(contact, family="p0_active", side="master", value_dim=3)
    ops = ff.assemble_multiplier(
        contact,
        rho=0.0,
        multiplier=mult,
        backend="numpy",
    )

    n_active_facets = int(np.unique(contact.source_facets_master).shape[0])
    assert np.asarray(ops.B_a).shape == (3 * n_active_facets, 3 * contact.surface_master.n_nodes)
    assert np.asarray(ops.B_b).shape == (3 * n_active_facets, 3 * contact.surface_slave.n_nodes)

    builder = ff.CoupledSystemBuilder.from_structural(np.eye(24), np.zeros(24))
    builder.register_field("a", n_dofs=12, value_dim=3, n_nodes=4)
    builder.register_field("b", n_dofs=12, value_dim=3, n_nodes=4)
    builder.add_contact_mortar(ops, master="a", slave="b", value_dim=3)
    K = builder.build().to_dense()
    assert K.shape == (24 + 3 * n_active_facets, 24 + 3 * n_active_facets)


def test_contact_mortar_solve_variants_agree_on_fixture_workpiece_example():
    fixture_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    workpiece_coords = fixture_coords.copy()
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        fixture_coords,
        facets,
        workpiece_coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )

    low_order_smoother = np.eye(4) - np.ones((4, 4), dtype=float) / 4.0
    k_fixture = 25.0 * np.eye(4) + 30000.0 * low_order_smoother
    k_workpiece = np.eye(4) + 30000.0 * low_order_smoother
    stiffness = np.block(
        [
            [k_fixture, np.zeros((4, 4), dtype=float)],
            [np.zeros((4, 4), dtype=float), k_workpiece],
        ]
    )
    force = np.concatenate([np.zeros(4, dtype=float), np.ones(4, dtype=float)])
    multipliers = {
        "nodal": ff.MultiplierSpec.nodal_mortar(),
        "dual": ff.MultiplierSpec.dual_mortar(),
        "coarse_dual": ff.MultiplierSpec.coarse_dual_mortar(rank=1),
        "p0": ff.MultiplierSpec.p0_mortar(contact),
        "coarse_p0": ff.MultiplierSpec.coarse_p0_mortar(contact, patch_ids=np.array([0, 0], dtype=int)),
        "coarse_p1": ff.MultiplierSpec.coarse_p1_mortar(
            basis=ff.coarse_p1_basis_from_node_groups(4, [[0, 1, 2, 3]])
        ),
        "grid_p1": ff.MultiplierSpec.coarse_p1_mortar(
            basis=ff.coarse_p1_basis_from_surface_grid(contact.surface_master, shape=(2, 2), axes=(0, 1))
        ),
    }

    solutions = {}
    for name, multiplier in multipliers.items():
        ops = ff.assemble_multiplier(contact, rho=0.0, multiplier=multiplier, backend="numpy")
        B = np.asarray(ops.B, dtype=float)
        system = np.block(
            [
                [stiffness, B.T],
                [B, np.zeros((B.shape[0], B.shape[0]), dtype=float)],
            ]
        )
        rhs = np.concatenate([force, np.zeros(B.shape[0], dtype=float)])
        solutions[name] = np.linalg.solve(system, rhs)[:8]
        np.testing.assert_allclose(B @ solutions[name], np.zeros(B.shape[0]), atol=1e-10)

    reference = solutions["dual"]
    np.testing.assert_allclose(solutions["nodal"], reference, atol=1e-10)
    np.testing.assert_allclose(solutions["grid_p1"], reference, atol=1e-10)
    for name in ("coarse_dual", "p0", "coarse_p0", "coarse_p1"):
        np.testing.assert_allclose(solutions[name], reference, atol=1e-3)

    np.testing.assert_allclose(np.mean(reference[:4]), 1.0 / 26.0, atol=1e-12)
    np.testing.assert_allclose(np.mean(reference[4:]), 1.0 / 26.0, atol=1e-12)
    reference_work = float(force @ reference)
    for solution in solutions.values():
        np.testing.assert_allclose(float(force @ solution), reference_work, rtol=5e-3, atol=1e-12)


def test_contact_mortar_solve_variants_agree_on_nonmatching_fixture_workpiece_meshes():
    fixture_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    fixture_facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    workpiece_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    workpiece_facets = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        fixture_coords,
        fixture_facets,
        workpiece_coords,
        workpiece_facets,
        facet_dofs_master=fixture_facets,
        facet_dofs_slave=workpiece_facets,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    assert contact.supermesh_conn.shape[0] == 4

    fixture_smoother = np.eye(4) - np.ones((4, 4), dtype=float) / 4.0
    workpiece_smoother = np.eye(5) - np.ones((5, 5), dtype=float) / 5.0
    stiffness = np.block(
        [
            [25.0 * np.eye(4) + 100000.0 * fixture_smoother, np.zeros((4, 5), dtype=float)],
            [np.zeros((5, 4), dtype=float), np.eye(5) + 100000.0 * workpiece_smoother],
        ]
    )
    force = np.concatenate([np.zeros(4, dtype=float), np.ones(5, dtype=float)])
    multipliers = {
        "nodal": ff.MultiplierSpec.nodal_mortar(),
        "dual": ff.MultiplierSpec.dual_mortar(),
        "coarse_dual": ff.MultiplierSpec.coarse_dual_mortar(rank=1),
        "p0": ff.MultiplierSpec.p0_mortar(contact),
        "p0_active": ff.MultiplierSpec.from_contact(contact, family="p0_active", side="master"),
        "p0_supermesh": ff.MultiplierSpec.from_contact(contact, family="p0_supermesh", side="master"),
        "coarse_p0": ff.MultiplierSpec.coarse_p0_mortar(contact, patch_ids=np.array([0, 0], dtype=int)),
        "coarse_p1": ff.MultiplierSpec.coarse_p1_mortar(
            basis=ff.coarse_p1_basis_from_node_groups(4, [[0, 1, 2, 3]])
        ),
        "grid_p1": ff.MultiplierSpec.coarse_p1_mortar(
            basis=ff.coarse_p1_basis_from_surface_grid(contact.surface_master, shape=(2, 2), axes=(0, 1))
        ),
    }

    solutions = {}
    for name, multiplier in multipliers.items():
        ops = ff.assemble_multiplier(contact, rho=0.0, multiplier=multiplier, backend="numpy")
        B = np.asarray(ops.B, dtype=float)
        system = np.block(
            [
                [stiffness, B.T],
                [B, np.zeros((B.shape[0], B.shape[0]), dtype=float)],
            ]
        )
        rhs = np.concatenate([force, np.zeros(B.shape[0], dtype=float)])
        solutions[name] = np.linalg.lstsq(system, rhs, rcond=None)[0][:9]
        np.testing.assert_allclose(B @ solutions[name], np.zeros(B.shape[0]), atol=1e-9)

    reference = solutions["dual"]
    np.testing.assert_allclose(solutions["nodal"], reference, atol=1e-10)
    np.testing.assert_allclose(solutions["grid_p1"], reference, atol=1e-10)
    for name in ("coarse_dual", "p0", "p0_active", "p0_supermesh", "coarse_p0", "coarse_p1"):
        np.testing.assert_allclose(solutions[name], reference, atol=1e-4)

    reference_work = float(force @ reference)
    for solution in solutions.values():
        np.testing.assert_allclose(float(force @ solution), reference_work, rtol=2e-3, atol=1e-12)


def test_contact_penalty_operators_from_inputs():
    class _ContactStub:
        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, u, params, normal_source)
            return np.array([0.0], dtype=float)

        def assemble_jacobian(
            self,
            res_form,
            u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="numpy",
            batch_jac=None,
        ):
            _ = (res_form, u, params, normal_source, sparse, backend, batch_jac)
            return np.array([[1.0]], dtype=float)

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0]), "b": np.array([0.0])}

    ops = ff.assemble_penalty(
        _ContactStub(),
        weak_form=_dummy_res_form,
        state={"a": np.array([0.0]), "b": np.array([0.0])},
        params=object(),
    )
    assert isinstance(ops, ff.ContactOperators)
    assert ops.enforcement == "nitsche"
    assert ops.formulation == "penalty_consistent"


def test_contact_penalty_operators_accepts_weak_form_state_aliases():
    class _ContactStub:
        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, u, params, normal_source)
            return np.array([0.0], dtype=float)

        def assemble_jacobian(
            self,
            res_form,
            u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="numpy",
            batch_jac=None,
        ):
            _ = (res_form, u, params, normal_source, sparse, backend, batch_jac)
            return np.array([[1.0]], dtype=float)

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0]), "b": np.array([0.0])}

    ops = ff.assemble_penalty(
        _ContactStub(),
        weak_form=_dummy_res_form,
        state={"a": np.array([0.0]), "b": np.array([0.0])},
        params=object(),
    )
    assert ops.enforcement == "nitsche"


def test_contact_constraint_operators_keep_law_formulation_metadata():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    ops = ff.assemble_multiplier(
        contact,
        law="coulomb_like",
        formulation="augmented_lagrangian",
        rho=1.0,
        multiplier=_nodal_multiplier(),
    )
    assert isinstance(ops, ff.ContactOperators)
    assert ops.enforcement == "mortar"
    assert ops.formulation == "augmented_lagrangian"
    assert ops.law == "coulomb_like"


def test_contact_constraint_operators_accept_penalty_style_inputs_for_api_symmetry():
    class _ContactStub:
        def assemble_contact_coupling_matrices(self):
            coupling_aa = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            coupling_ab = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            return coupling_aa, coupling_ab

        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, u, params, normal_source)
            return np.array([0.3, -0.3], dtype=float)

        def assemble_jacobian(
            self,
            res_form,
            u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="numpy",
            batch_jac=None,
        ):
            _ = (res_form, u, params, normal_source, sparse, backend, batch_jac)
            return np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=float)

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0]), "b": np.array([0.0])}

    contact = _ContactStub()
    ops_base = ff.assemble_multiplier(contact, rho=1.0, multiplier=_nodal_multiplier())
    ops_with_alias_inputs = ff.assemble_multiplier(
        contact,
        rho=1.0,
        multiplier=_nodal_multiplier(),
        weak_form=_dummy_res_form,
        state={"a": np.array([0.0]), "b": np.array([0.0])},
        params=object(),
        backend="jax",
    )

    assert np.allclose(np.asarray(ops_base.B), np.asarray(ops_with_alias_inputs.B), atol=1e-12)
    assert np.allclose(np.asarray(ops_base.Kuu), np.asarray(ops_with_alias_inputs.Kuu), atol=1e-12)
    assert np.allclose(np.asarray(ops_with_alias_inputs.residual), np.array([0.3, -0.3]), atol=1e-12)
    assert np.allclose(
        np.asarray(ops_with_alias_inputs.jacobian),
        np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=float),
        atol=1e-12,
    )


def test_contact_constraint_operators_reject_partial_eval_inputs():
    class _ContactStub:
        def assemble_contact_coupling_matrices(self):
            coupling_aa = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            coupling_ab = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            return coupling_aa, coupling_ab

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0]), "b": np.array([0.0])}

    with pytest.raises(ValueError, match="must be provided together"):
        ff.assemble_multiplier(_ContactStub(), multiplier=_nodal_multiplier(), weak_form=_dummy_res_form)


def test_contact_constraint_operators_reject_penalty_formulation():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    with pytest.raises(ValueError, match="Constraint operators are multiplier-family only"):
        ff.assemble_multiplier(contact, multiplier=_nodal_multiplier(), formulation="penalty")


def test_contact_constraint_eval_grad_state_matches_fd():
    class _ContactStub:
        def assemble_contact_coupling_matrices(self):
            coupling_aa = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            coupling_ab = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            return coupling_aa, coupling_ab

        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, normal_source)
            ua = u["a"][0]
            ub = u["b"][0]
            return jnp.asarray([params["alpha"] * (ua - ub), params["alpha"] * (ub - ua)])

        def assemble_jacobian(
            self,
            res_form,
            u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="numpy",
            batch_jac=None,
        ):
            _ = (res_form, normal_source, sparse, backend, batch_jac)
            scale = params["alpha"] * (1.0 + u["a"][0])
            return jnp.asarray([[scale, -scale], [-scale, scale]])

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": jnp.asarray([0.0]), "b": jnp.asarray([0.0])}

    contact = _ContactStub()
    params = {"alpha": jnp.asarray(2.0)}

    def objective(s):
        ops = ff.assemble_multiplier(
            contact,
            rho=1.5,
            multiplier=_nodal_multiplier(),
            backend="jax",
            weak_form=_dummy_res_form,
            state={"a": jnp.asarray([s]), "b": jnp.asarray([0.5])},
            params=params,
        )
        r = jnp.asarray(ops.residual)
        j = jnp.asarray(ops.jacobian)
        return 0.5 * jnp.dot(r, r) + 0.25 * jnp.sum(j * j)

    s0 = jnp.asarray(0.3)
    g_ad = float(jax.grad(objective)(s0))
    eps = 1e-4
    g_fd = (float(objective(s0 + eps)) - float(objective(s0 - eps))) / (2.0 * eps)
    rel = abs(g_ad - g_fd) / max(1.0, abs(g_fd))
    assert rel < 5e-3


def test_contact_constraint_eval_grad_rho_matches_fd():
    class _ContactStub:
        def assemble_contact_coupling_matrices(self):
            coupling_aa = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            coupling_ab = ff.ContactCouplingMatrix(
                rows=np.array([0], dtype=int),
                cols=np.array([0], dtype=int),
                data=np.array([1.0], dtype=float),
                shape=(1, 1),
            )
            return coupling_aa, coupling_ab

        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, u, params, normal_source)
            return jnp.asarray([0.1, -0.1])

        def assemble_jacobian(
            self,
            res_form,
            u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="numpy",
            batch_jac=None,
        ):
            _ = (res_form, u, params, normal_source, sparse, backend, batch_jac)
            return jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": jnp.asarray([0.0]), "b": jnp.asarray([0.0])}

    contact = _ContactStub()

    def objective(rho):
        ops = ff.assemble_multiplier(
            contact,
            rho=rho,
            multiplier=_nodal_multiplier(),
            backend="jax",
            weak_form=_dummy_res_form,
            state={"a": jnp.asarray([0.2]), "b": jnp.asarray([0.4])},
            params={"alpha": jnp.asarray(1.0)},
        )
        kuu = jnp.asarray(ops.Kuu)
        return 0.5 * jnp.sum(kuu * kuu)

    rho0 = jnp.asarray(2.0)
    g_ad = float(jax.grad(objective)(rho0))
    eps = 1e-4
    g_fd = (float(objective(rho0 + eps)) - float(objective(rho0 - eps))) / (2.0 * eps)
    rel = abs(g_ad - g_fd) / max(1.0, abs(g_fd))
    assert rel < 5e-3


def test_contact_kkt_augmented_lagrangian_grad_rho_matches_fd():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )

    # n_u = n_master + n_slave = 8, n_lambda(p0) = n_facets = 1
    x = jnp.linspace(0.1, 0.9, 9)

    def objective(rho):
        K = contact.assemble_contact_kkt(
            rho=rho,
            multiplier=_p0_multiplier(contact),
            backend="jax",
            format="dense",
        )
        y = K @ x
        return 0.5 * jnp.dot(y, y)

    rho0 = jnp.array(2.0)
    g_ad = float(jax.grad(objective)(rho0))
    eps = 1e-5
    f_p = float(objective(rho0 + eps))
    f_m = float(objective(rho0 - eps))
    g_fd = (f_p - f_m) / (2.0 * eps)
    rel = abs(g_ad - g_fd) / max(1.0, abs(g_fd))
    assert rel < 2e-4


def test_numpy_builder_contact_mortar_sugar_matches_explicit_multiplier_choice():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    mult = ff.MultiplierSpec.nodal_mortar()
    ops = ff.assemble_multiplier(contact, rho=2.0, multiplier=mult, backend="numpy")

    direct = ff.NumpyCoupledSystemBuilder.from_structural(np.eye(8), np.zeros(8))
    direct.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    direct.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    direct.add_contact_mortar(ops, master="a", slave="b", value_dim=1)

    raw = ff.NumpyCoupledSystemBuilder.from_structural(np.eye(8), np.zeros(8))
    raw.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    raw.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    raw.add_contact(
        contact,
        master="a",
        slave="b",
        family="constraint",
        mortar="nodal",
        rho=2.0,
        value_dim=1,
    )

    assert np.allclose(np.asarray(raw.build().K_u.toarray()), np.asarray(direct.build().K_u.toarray()), atol=1e-12)


def test_solve_contact_kkt_implicit_grad_rho_matches_fd():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )

    rhs = jnp.linspace(0.2, 1.0, 9)

    def objective(rho):
        K = contact.assemble_contact_kkt(
            rho=rho,
            multiplier=_p0_multiplier(contact),
            backend="jax",
            format="dense",
        )
        u = ff.solve_contact_kkt(K, rhs, backend="jax", diagonal_shift=1e-2)
        return 0.5 * jnp.dot(u, u)

    def objective_ref(rho):
        K = contact.assemble_contact_kkt(
            rho=rho,
            multiplier=_p0_multiplier(contact),
            backend="jax",
            format="dense",
        )
        A = K + 1e-2 * jnp.eye(K.shape[0], dtype=K.dtype)
        u = jnp.linalg.solve(A, rhs)
        return 0.5 * jnp.dot(u, u)

    rho0 = jnp.array(2.0)
    g_custom = float(jax.grad(objective)(rho0))
    g_ref = float(jax.grad(objective_ref)(rho0))
    rel = abs(g_custom - g_ref) / max(1.0, abs(g_ref))
    tol = 2e-4 if jax.config.jax_enable_x64 else 2e-3
    assert rel < tol


def test_solve_contact_kkt_implicit_grad_rhs_matches_ref():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    K = contact.assemble_contact_kkt(
        rho=jnp.array(2.0),
        multiplier=_p0_multiplier(contact),
        backend="jax",
        format="dense",
    )

    def objective(rhs):
        u = ff.solve_contact_kkt(K, rhs, backend="jax", diagonal_shift=1e-2)
        return 0.5 * jnp.dot(u, u)

    def objective_ref(rhs):
        A = K + 1e-2 * jnp.eye(K.shape[0], dtype=K.dtype)
        u = jnp.linalg.solve(A, rhs)
        return 0.5 * jnp.dot(u, u)

    rhs0 = jnp.linspace(0.2, 1.0, int(K.shape[0]))
    g_custom = jax.grad(objective)(rhs0)
    g_ref = jax.grad(objective_ref)(rhs0)
    assert np.allclose(np.asarray(g_custom), np.asarray(g_ref), atol=1e-3, rtol=1e-5)


def test_solve_contact_kkt_implicit_grad_diagonal_shift_matches_ref():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    K = contact.assemble_contact_kkt(
        rho=jnp.array(2.0),
        multiplier=_p0_multiplier(contact),
        backend="jax",
        format="dense",
    )
    rhs = jnp.linspace(0.2, 1.0, int(K.shape[0]))

    def objective(shift):
        u = ff.solve_contact_kkt(K, rhs, backend="jax", diagonal_shift=shift)
        return 0.5 * jnp.dot(u, u)

    def objective_ref(shift):
        A = K + shift * jnp.eye(K.shape[0], dtype=K.dtype)
        u = jnp.linalg.solve(A, rhs)
        return 0.5 * jnp.dot(u, u)

    shift0 = jnp.array(1e-2)
    g_custom = float(jax.grad(objective)(shift0))
    g_ref = float(jax.grad(objective_ref)(shift0))
    rel = abs(g_custom - g_ref) / max(1.0, abs(g_ref))
    tol = 2e-4 if jax.config.jax_enable_x64 else 2e-3
    assert rel < tol


def test_solve_contact_kkt_sparse_gmres_grad_rhs_matches_dense_ref():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    K_sparse = contact.assemble_contact_kkt(
        rho=2.0,
        multiplier=_p0_multiplier(contact),
        backend="jax",
        format="bcoo",
    )
    K_dense = contact.assemble_contact_kkt(
        rho=2.0,
        multiplier=_p0_multiplier(contact),
        backend="jax",
        format="dense",
    )
    cfg = ff.ContactKKTSolveConfig(
        backend="jax",
        diagonal_shift=1e-2,
        jax_solver="gmres",
        jax_tol=1e-10,
        jax_maxiter=400,
    )

    def objective(rhs):
        u = ff.solve_contact_kkt(K_sparse, rhs, config=cfg)
        return 0.5 * jnp.dot(u, u)

    def objective_ref(rhs):
        A = K_dense + 1e-2 * jnp.eye(K_dense.shape[0], dtype=K_dense.dtype)
        u = jnp.linalg.solve(A, rhs)
        return 0.5 * jnp.dot(u, u)

    rhs0 = jnp.linspace(0.2, 1.0, int(K_dense.shape[0]))
    g_sparse = jax.grad(objective)(rhs0)
    g_ref = jax.grad(objective_ref)(rhs0)
    assert np.allclose(np.asarray(g_sparse), np.asarray(g_ref), atol=1e-3, rtol=1e-5)


def test_solve_contact_kkt_sparse_spsolve_matches_dense_ref():
    coords, conn, facets = _tet4_fixture()
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )
    K_dense = contact.assemble_contact_kkt(
        rho=2.0,
        multiplier=_p0_multiplier(contact),
        backend="jax",
        format="dense",
    )
    from jax.experimental import sparse as jsparse

    A_dense = K_dense + 1e-2 * jnp.eye(K_dense.shape[0], dtype=K_dense.dtype)
    K_sparse = jsparse.BCOO.fromdense(A_dense)
    rhs = jnp.linspace(0.2, 1.0, int(K_dense.shape[0]))
    cfg = ff.ContactKKTSolveConfig(backend="jax", jax_solver="spsolve", diagonal_shift=0.0)
    u_sparse = ff.solve_contact_kkt(K_sparse, rhs, config=cfg)
    u_ref = jnp.linalg.solve(A_dense, rhs)
    assert np.allclose(np.asarray(u_sparse), np.asarray(u_ref), atol=1e-5, rtol=1e-5)


def test_solve_contact_kkt_petsc_config_forwarding(monkeypatch):
    import fluxfem.solver.petsc as petsc_mod

    captured = {}

    def _stub_petsc_shell_solve(A, b, **kwargs):
        captured["A_shape"] = tuple(getattr(A, "shape", ()))
        captured["b"] = np.asarray(b)
        captured["kwargs"] = dict(kwargs)
        return np.arange(np.asarray(b).shape[0], dtype=float)

    monkeypatch.setattr(petsc_mod, "petsc_shell_solve", _stub_petsc_shell_solve)

    K = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=float)
    rhs = np.array([1.0, 2.0], dtype=float)
    cfg = ff.ContactKKTSolveConfig(
        backend="petsc4py",
        diagonal_shift=1e-3,
        petsc_ksp_type="bcgs",
        petsc_pc_type="jacobi",
        petsc_preconditioner=None,
        petsc_rtol=1e-7,
        petsc_atol=1e-9,
        petsc_max_it=123,
        petsc_options={"ksp_monitor": None},
        petsc_options_prefix="kkt_test_",
    )

    x = ff.solve_contact_kkt(K, rhs, config=cfg)
    assert np.allclose(np.asarray(x), np.array([0.0, 1.0], dtype=float))
    assert captured["A_shape"] == (2, 2)
    assert np.allclose(captured["b"], rhs)
    assert captured["kwargs"]["n_dofs"] == 2
    assert captured["kwargs"]["ksp_type"] == "bcgs"
    assert captured["kwargs"]["pc_type"] == "jacobi"
    assert captured["kwargs"]["preconditioner"] is None
    assert captured["kwargs"]["rtol"] == 1e-7
    assert captured["kwargs"]["atol"] == 1e-9
    assert captured["kwargs"]["max_it"] == 123
    assert captured["kwargs"]["options"] == {"ksp_monitor": None}
    assert captured["kwargs"]["options_prefix"] == "kkt_test_"

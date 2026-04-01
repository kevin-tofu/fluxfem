from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import fluxfem as ff


def test_build_nodal_embedding_map_nearest_neighbor():
    master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave = np.array(
        [
            [0.1, 0.0, 0.0],  # -> master 0
            [0.9, 0.0, 0.0],  # -> master 1
        ],
        dtype=float,
    )

    emb = ff.build_nodal_embedding_map(master, slave)
    assert emb.mode == "nodal"
    assert emb.shape == (2, 3)
    assert np.array_equal(np.asarray(emb.rows), np.array([0, 1], dtype=int))
    assert np.array_equal(np.asarray(emb.cols), np.array([0, 1], dtype=int))
    assert np.allclose(np.asarray(emb.data), np.array([1.0, 1.0]), atol=1e-12)


def test_assemble_embedding_constraint_matrix_numpy():
    emb = ff.EmbeddingMap(
        rows=np.array([0, 1], dtype=int),
        cols=np.array([0, 1], dtype=int),
        data=np.array([1.0, 1.0], dtype=float),
        shape=(2, 2),
        mode="nodal",
    )
    C = ff.assemble_embedding_constraint_matrix(
        emb,
        n_master_nodes=2,
        n_slave_nodes=2,
        value_dim=1,
        backend="numpy",
    )
    C_ref = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    assert np.allclose(np.asarray(C), C_ref, atol=1e-12)


def test_assemble_embedding_constraint_matrix_jax_value_dim2():
    emb = ff.EmbeddingMap(
        rows=np.array([0], dtype=int),
        cols=np.array([1], dtype=int),
        data=np.array([0.5], dtype=float),
        shape=(1, 2),
        mode="nodal",
    )
    C = ff.assemble_embedding_constraint_matrix(
        emb,
        n_master_nodes=2,
        n_slave_nodes=1,
        value_dim=2,
        backend="jax",
    )
    C_np = np.asarray(C)
    assert C_np.shape == (2, 6)
    # row 0 (x component): +0.5 at master node 1-x, -1 at slave node 0-x
    assert np.isclose(C_np[0, 2], 0.5)
    assert np.isclose(C_np[0, 4], -1.0)
    # row 1 (y component): +0.5 at master node 1-y, -1 at slave node 0-y
    assert np.isclose(C_np[1, 3], 0.5)
    assert np.isclose(C_np[1, 5], -1.0)
    # quick functional check
    u_master = jnp.array([0.0, 0.0, 2.0, 4.0])  # node0(x,y), node1(x,y)
    u_slave = jnp.array([1.0, 2.0])  # node0(x,y)
    u = jnp.concatenate([u_master, u_slave], axis=0)
    r = C @ u
    assert np.allclose(np.asarray(r), np.zeros((2,)), atol=1e-12)


def test_build_barycentric_embedding_map_tet4_linear_reproduces_shape_values():
    master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    slave = np.array([[0.2, 0.3, 0.1]], dtype=float)
    emb = ff.build_barycentric_embedding_map(master, conn, slave, tol=1e-10)
    assert emb.mode == "barycentric"
    assert emb.shape == (1, 4)
    w = np.zeros((4,), dtype=float)
    for r, c, v in zip(np.asarray(emb.rows), np.asarray(emb.cols), np.asarray(emb.data)):
        assert int(r) == 0
        w[int(c)] += float(v)
    assert np.allclose(np.sum(w), 1.0, atol=1e-10)
    # For tet4, barycentric coordinates for point (x,y,z): [1-x-y-z, x, y, z]
    w_ref = np.array([0.4, 0.2, 0.3, 0.1], dtype=float)
    assert np.allclose(w, w_ref, atol=1e-10)


def test_build_barycentric_embedding_map_raises_when_unmapped():
    master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    slave = np.array([[2.0, 2.0, 2.0]], dtype=float)
    with pytest.raises(ValueError, match="Failed to map slave point index"):
        ff.build_barycentric_embedding_map(master, conn, slave, tol=1e-10, allow_unmapped="error")


def test_build_barycentric_embedding_map_skip_returns_unmapped_ids():
    master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    slave = np.array(
        [
            [0.2, 0.3, 0.1],  # mapped
            [2.0, 2.0, 2.0],  # unmapped
        ],
        dtype=float,
    )
    emb, unmapped = ff.build_barycentric_embedding_map(
        master,
        conn,
        slave,
        tol=1e-10,
        allow_unmapped="skip",
        return_unmapped_ids=True,
    )
    assert emb.shape == (2, 4)
    assert np.array_equal(np.asarray(unmapped), np.array([1], dtype=int))


def test_build_barycentric_embedding_map_bool_allow_unmapped_deprecated():
    master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    slave = np.array([[2.0, 2.0, 2.0]], dtype=float)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        emb = ff.build_barycentric_embedding_map(master, conn, slave, allow_unmapped=True)
    assert emb.shape == (1, 4)


def test_build_barycentric_embedding_map_empty_master_conn_error_mode():
    master = np.zeros((0, 3), dtype=float)
    conn = np.zeros((0, 4), dtype=int)
    slave = np.array([[0.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="master_conn has no elements"):
        ff.build_barycentric_embedding_map(master, conn, slave, allow_unmapped="error")


def test_build_barycentric_embedding_map_from_meshes_with_plane_selector():
    master_mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    slave_mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()

    emb = ff.build_barycentric_embedding_map_from_meshes(
        master_mesh,
        slave_mesh,
        slave_facet_selector=lambda m: m.facets_on_plane(axis=2, value=0.0),
        tol=1e-8,
        allow_unmapped="error",
    )
    assert emb.mode == "barycentric"
    assert emb.shape[1] == int(np.asarray(master_mesh.coords).shape[0])
    assert emb.shape[0] == int(np.asarray(slave_mesh.coords).shape[0])
    assert emb.rows.size > 0


def test_build_barycentric_embedding_map_from_meshes_returns_global_unmapped_ids():
    master_mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    slave_mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()

    def _top_nodes(mesh):
        z = np.asarray(mesh.coords)[:, 2]
        return np.nonzero(np.isclose(z, np.max(z), atol=1e-12))[0]

    emb, unmapped = ff.build_barycentric_embedding_map_from_meshes(
        master_mesh,
        slave_mesh,
        slave_node_selector=_top_nodes,
        master_element_selector=lambda m: np.array([0], dtype=int),
        tol=1e-8,
        allow_unmapped="skip",
        return_unmapped_ids=True,
    )
    assert emb.shape[0] == int(np.asarray(slave_mesh.coords).shape[0])
    assert unmapped.ndim == 1


def test_assemble_embedding_constraint_matrix_compacts_subset_rows():
    emb = ff.EmbeddingMap(
        rows=np.array([1], dtype=int),
        cols=np.array([0], dtype=int),
        data=np.array([1.0], dtype=float),
        shape=(2, 1),
        mode="barycentric",
    )
    C = ff.assemble_embedding_constraint_matrix(
        emb,
        n_master_nodes=1,
        n_slave_nodes=2,
        value_dim=1,
        backend="numpy",
    )
    # one constrained slave node => one lambda row (compacted)
    assert C.shape == (1, 3)
    # C = [1, 0, -1] in [master0, slave0, slave1] ordering
    assert np.allclose(np.asarray(C), np.array([[1.0, 0.0, -1.0]], dtype=float), atol=1e-12)


def test_assemble_rbe2_constraint_matrix_kinematics_identity():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    C = ff.assemble_rbe2_constraint_matrix(x_ref, x_slave, backend="numpy")
    assert C.shape == (6, 12)

    u_ref = np.array([0.3, -0.2, 0.1], dtype=float)
    w_ref = np.array([0.0, 0.0, 0.5], dtype=float)
    u_s = []
    for p in x_slave:
        r = p - x_ref
        u_s.append(u_ref + np.cross(w_ref, r))
    u_s = np.asarray(u_s).reshape(-1)
    q = np.concatenate([u_ref, w_ref, u_s], axis=0)
    res = C @ q
    assert np.allclose(res, np.zeros_like(res), atol=1e-12)


def test_assemble_rbe3_constraint_matrix_rigid_motion_reproduction():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 2.0, 3.0], dtype=float)
    C = ff.assemble_rbe3_constraint_matrix(x_ref, x_slave, weights=weights, backend="numpy")
    assert C.shape == (6, 15)

    u_ref = np.array([0.3, -0.2, 0.1], dtype=float)
    w_ref = np.array([0.1, -0.05, 0.2], dtype=float)

    def rigid_u(point):
        r = point - x_ref
        return u_ref + np.cross(w_ref, r)

    u_s = np.asarray([rigid_u(p) for p in x_slave], dtype=float).reshape(-1)
    q = np.concatenate([u_ref, w_ref, u_s], axis=0)
    res = C @ q
    assert np.allclose(res, np.zeros_like(res), atol=1e-12)


def test_build_rbe3_weights_equal_and_distance():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    w_equal = ff.build_rbe3_weights(x_ref, x_slave, method="equal")
    assert np.allclose(w_equal, np.array([0.5, 0.5], dtype=float), atol=1e-12)

    w_dist = ff.build_rbe3_weights(x_ref, x_slave, method="distance", power=1.0)
    assert np.allclose(w_dist, np.array([2.0 / 3.0, 1.0 / 3.0], dtype=float), atol=1e-12)


def test_build_rbe3_weights_facet_area():
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
    surface = ff.SurfaceMesh.from_facets(coords, facets)
    w = ff.build_rbe3_weights(
        np.array([0.5, 0.5, 1.0], dtype=float),
        coords,
        method="facet_area",
        surface=surface,
    )
    assert np.allclose(w, 0.25 * np.ones((4,), dtype=float), atol=1e-12)

from __future__ import annotations

import numpy as np
import jax

import fluxfem as ff

jax.config.update("jax_enable_x64", True)


def test_structured_plate_grid_and_dofs():
    coords, conn = ff.structured_plate_grid(nx=2, ny=1, length_x=2.0, length_y=1.0)

    assert coords.shape == (6, 2)
    assert conn.tolist() == [[0, 1, 4, 3], [1, 2, 5, 4]]
    np.testing.assert_array_equal(ff.plate_node_dofs([0, 2], "wrx"), np.array([0, 1, 6, 7]))
    np.testing.assert_array_equal(ff.plate_element_dofs(conn).shape, np.array([2, 12]))


def test_mindlin_plate_element_stiffness_is_symmetric_and_preserves_rigid_modes():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
    section = ff.PlateSection(E=210.0e9, nu=0.3, thickness=0.02)
    K = ff.mindlin_plate_element_stiffness(coords, section)

    assert K.shape == (12, 12)
    np.testing.assert_allclose(K, K.T, rtol=0.0, atol=1.0e-8)

    rigid_w = np.tile(np.array([1.0, 0.0, 0.0]), 4)
    rigid_rx = np.asarray([[x, 1.0, 0.0] for x, _y in coords], dtype=float).reshape(-1)
    rigid_ry = np.asarray([[y, 0.0, 1.0] for _x, y in coords], dtype=float).reshape(-1)

    scale = float(np.max(np.abs(K)))
    for mode in (rigid_w, rigid_rx, rigid_ry):
        np.testing.assert_allclose(K @ mode, np.zeros((12,)), atol=scale * 1.0e-12)


def test_mindlin_plate_uniform_load_resultant():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
    f = ff.mindlin_plate_element_uniform_load(coords, 5.0)

    assert f.shape == (12,)
    np.testing.assert_allclose(f[0::3], np.full((4,), 2.5), atol=1.0e-12)
    np.testing.assert_allclose(f[1::3], np.zeros((4,)), atol=1.0e-12)
    np.testing.assert_allclose(f[2::3], np.zeros((4,)), atol=1.0e-12)


def test_assemble_mindlin_plate_stiffness_backends_and_load_vector():
    coords, conn = ff.structured_plate_grid(nx=1, ny=1, length_x=2.0, length_y=1.0)
    section = ff.PlateSection(E=70.0e9, nu=0.33, thickness=0.05)

    K_dense = ff.assemble_mindlin_plate_stiffness(coords, conn, section, backend="numpy")
    K_csr = ff.assemble_mindlin_plate_stiffness(coords, conn, section, backend="scipy")
    K_jax = ff.assemble_mindlin_plate_stiffness(coords, conn, section, backend="jax")
    f = ff.assemble_mindlin_plate_uniform_load(coords, conn, 7.0)

    assert K_dense.shape == (12, 12)
    np.testing.assert_allclose(K_csr.toarray(), K_dense)
    np.testing.assert_allclose(np.asarray(K_jax.to_dense()), K_dense)
    np.testing.assert_allclose(f[0::3].sum(), 14.0, atol=1.0e-12)

from __future__ import annotations

import numpy as np
import jax
import scipy.sparse as sp

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


def test_mindlin_plate_shear_modes_are_selectable():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 1.0], [0.0, 1.0]], dtype=float)
    reduced = ff.PlateSection(E=210.0e9, nu=0.3, thickness=0.01, shear_mode="reduced")
    full = ff.PlateSection(E=210.0e9, nu=0.3, thickness=0.01, shear_mode="full")
    mitc4 = ff.PlateSection(E=210.0e9, nu=0.3, thickness=0.01, shear_mode="mitc4")

    K_reduced = ff.mindlin_plate_element_stiffness(coords, reduced)
    K_full = ff.mindlin_plate_element_stiffness(coords, full)
    K_mitc4 = ff.mindlin_plate_element_stiffness(coords, mitc4)

    np.testing.assert_allclose(K_reduced, K_reduced.T, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(K_full, K_full.T, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(K_mitc4, K_mitc4.T, rtol=0.0, atol=1.0e-8)
    assert not np.allclose(K_full, K_reduced)
    assert not np.allclose(K_mitc4, K_reduced)


def test_mindlin_plate_rejects_unknown_shear_mode():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
    section = ff.PlateSection(E=210.0e9, nu=0.3, thickness=0.02, shear_mode="bogus")

    with np.testing.assert_raises(ValueError):
        ff.mindlin_plate_element_stiffness(coords, section)


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


def test_assemble_mindlin_plate_mitc4_backend_matches_dense():
    coords, conn = ff.structured_plate_grid(nx=1, ny=1, length_x=2.0, length_y=1.0)
    section = ff.PlateSection(E=70.0e9, nu=0.33, thickness=0.01, shear_mode="mitc4")

    K_dense = ff.assemble_mindlin_plate_stiffness(coords, conn, section, backend="numpy")
    K_csr = ff.assemble_mindlin_plate_stiffness(coords, conn, section, backend="scipy")
    K_jax = ff.assemble_mindlin_plate_stiffness(coords, conn, section, backend="jax")

    np.testing.assert_allclose(K_csr.toarray(), K_dense)
    np.testing.assert_allclose(np.asarray(K_jax.to_dense()), K_dense)


def test_assemble_mindlin_plate_point_loads():
    f = ff.assemble_mindlin_plate_point_loads(
        4,
        [1, 3],
        forces=[-2.0, -3.0],
        moments=[[0.5, 0.0], [0.0, -0.25]],
    )

    expected = np.zeros((12,), dtype=float)
    expected[3:6] = [-2.0, 0.5, 0.0]
    expected[9:12] = [-3.0, 0.0, -0.25]
    np.testing.assert_allclose(f, expected)


def test_flat_shell_element_stiffness_is_symmetric_and_preserves_rigid_modes():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
    section = ff.ShellSection(E=210.0e9, nu=0.3, thickness=0.02, drilling_stiffness=0.0)
    K = ff.flat_shell_element_stiffness(coords, section)

    assert K.shape == (24, 24)
    np.testing.assert_allclose(K, K.T, rtol=0.0, atol=1.0e-8)

    rigid_ux = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 4)
    rigid_uy = np.tile(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 4)
    rigid_w = np.tile(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 4)
    rigid_rz = np.asarray([[-y, x, 0.0, 0.0, 0.0, 1.0] for x, y in coords], dtype=float).reshape(-1)
    rigid_rx = np.asarray([[0.0, 0.0, y, 1.0, 0.0, 0.0] for _x, y in coords], dtype=float).reshape(-1)
    rigid_ry = np.asarray([[0.0, 0.0, -x, 0.0, 1.0, 0.0] for x, _y in coords], dtype=float).reshape(-1)

    scale = float(np.max(np.abs(K)))
    for mode in (rigid_ux, rigid_uy, rigid_w, rigid_rz, rigid_rx, rigid_ry):
        np.testing.assert_allclose(K @ mode, np.zeros((24,)), atol=scale * 1.0e-12)


def test_flat_shell_bending_block_matches_plate_stiffness():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
    shell = ff.ShellSection(E=70.0e9, nu=0.33, thickness=0.05, drilling_stiffness=0.0)
    plate = ff.PlateSection(E=shell.E, nu=shell.nu, thickness=shell.thickness, shear_correction=shell.shear_correction)
    Ks = ff.flat_shell_element_stiffness(coords, shell)
    Kp = ff.mindlin_plate_element_stiffness(coords, plate)
    P = np.zeros((12, 24), dtype=float)
    for a in range(4):
        P[3 * a + 0, 6 * a + 2] = 1.0
        P[3 * a + 1, 6 * a + 4] = -1.0
        P[3 * a + 2, 6 * a + 3] = 1.0

    np.testing.assert_allclose(P @ Ks @ P.T, Kp)


def test_flat_shell_uses_plate_shear_mode():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 1.0], [0.0, 1.0]], dtype=float)
    shell = ff.ShellSection(E=70.0e9, nu=0.33, thickness=0.01, drilling_stiffness=0.0, shear_mode="mitc4")
    plate = ff.PlateSection(E=shell.E, nu=shell.nu, thickness=shell.thickness, shear_correction=shell.shear_correction, shear_mode="mitc4")
    Ks = ff.flat_shell_element_stiffness(coords, shell)
    Kp = ff.mindlin_plate_element_stiffness(coords, plate)
    P = np.zeros((12, 24), dtype=float)
    for a in range(4):
        P[3 * a + 0, 6 * a + 2] = 1.0
        P[3 * a + 1, 6 * a + 4] = -1.0
        P[3 * a + 2, 6 * a + 3] = 1.0

    np.testing.assert_allclose(P @ Ks @ P.T, Kp)


def test_assemble_flat_shell_stiffness_backends_and_uniform_load():
    coords, conn = ff.structured_plate_grid(nx=1, ny=1, length_x=2.0, length_y=1.0)
    section = ff.ShellSection(E=70.0e9, nu=0.33, thickness=0.05)

    K_dense = ff.assemble_flat_shell_stiffness(coords, conn, section, backend="numpy")
    K_csr = ff.assemble_flat_shell_stiffness(coords, conn, section, backend="scipy")
    K_jax = ff.assemble_flat_shell_stiffness(coords, conn, section, backend="jax")
    f = ff.assemble_flat_shell_uniform_load(coords, conn, (1.0, 2.0, 3.0))

    assert K_dense.shape == (24, 24)
    np.testing.assert_allclose(K_csr.toarray(), K_dense)
    np.testing.assert_allclose(np.asarray(K_jax.to_dense()), K_dense)
    np.testing.assert_allclose(f[0::6].sum(), 2.0, atol=1.0e-12)
    np.testing.assert_allclose(f[1::6].sum(), 4.0, atol=1.0e-12)
    np.testing.assert_allclose(f[2::6].sum(), 6.0, atol=1.0e-12)


def test_shell_node_dofs_and_element_dofs():
    _coords, conn = ff.structured_plate_grid(nx=1, ny=1, length_x=1.0, length_y=1.0)

    np.testing.assert_array_equal(ff.shell_node_dofs([0, 2], "uxuzrz"), np.array([0, 2, 5, 12, 14, 17]))
    np.testing.assert_array_equal(ff.shell_element_dofs(conn).shape, np.array([1, 24]))


def test_assemble_flat_shell_point_loads():
    f = ff.assemble_flat_shell_point_loads(
        4,
        [1, 3],
        forces=[[1.0, 0.0, -2.0], [0.0, 3.0, -4.0]],
        moments=[[0.5, 0.0, 0.1], [0.0, -0.25, 0.0]],
    )

    expected = np.zeros((24,), dtype=float)
    expected[6:12] = [1.0, 0.0, -2.0, 0.5, 0.0, 0.1]
    expected[18:24] = [0.0, 3.0, -4.0, 0.0, -0.25, 0.0]
    np.testing.assert_allclose(f, expected)


def test_shell_element_stiffness_global_preserves_3d_rigid_modes():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.4],
            [2.0, 1.0, 0.6],
            [0.0, 1.0, 0.2],
        ],
        dtype=float,
    )
    section = ff.ShellSection(E=210.0e9, nu=0.3, thickness=0.02, drilling_stiffness=0.0)
    K = ff.shell_element_stiffness_global(coords, section)

    assert K.shape == (24, 24)
    np.testing.assert_allclose(K, K.T, rtol=0.0, atol=1.0e-8)
    scale = float(np.max(np.abs(K)))
    origin = coords[0]

    modes = []
    for direction in np.eye(3):
        q = np.zeros((4, 6), dtype=float)
        q[:, :3] = direction
        modes.append(q.reshape(-1))
    for omega in np.eye(3):
        q = np.zeros((4, 6), dtype=float)
        q[:, :3] = np.cross(omega[None, :], coords - origin)
        q[:, 3:6] = omega
        modes.append(q.reshape(-1))

    for mode in modes:
        np.testing.assert_allclose(K @ mode, np.zeros((24,)), atol=scale * 1.0e-12)


def test_assemble_shell_stiffness_3d_backends_and_uniform_load_resultant():
    coords2, conn = ff.structured_plate_grid(nx=1, ny=1, length_x=2.0, length_y=1.0)
    coords3 = np.column_stack([coords2[:, 0], coords2[:, 1], 0.2 * coords2[:, 0] + 0.1 * coords2[:, 1]])
    section = ff.ShellSection(E=70.0e9, nu=0.33, thickness=0.05)

    K_dense = ff.assemble_shell_stiffness(coords3, conn, section, backend="numpy")
    K_csr = ff.assemble_shell_stiffness(coords3, conn, section, backend="scipy")
    K_jax = ff.assemble_shell_stiffness(coords3, conn, section, backend="jax")
    load = np.array([1.0, 2.0, -3.0], dtype=float)
    f = np.asarray(ff.assemble_shell_uniform_load(coords3, conn, load), dtype=float)

    assert K_dense.shape == (24, 24)
    np.testing.assert_allclose(K_csr.toarray(), K_dense)
    np.testing.assert_allclose(np.asarray(K_jax.to_dense()), K_dense)
    _R, local = ff.shell_element_frame(coords3[conn[0]])
    def tri_area(a, b, c):
        ab = b - a
        ac = c - a
        return 0.5 * abs(float(ab[0] * ac[1] - ab[1] * ac[0]))

    area = tri_area(local[0], local[1], local[2]) + tri_area(local[0], local[2], local[3])
    np.testing.assert_allclose(f.reshape(-1, 6)[:, :3].sum(axis=0), load * area, atol=1.0e-12)


def test_flat_shell_node_can_tie_to_beam_root_with_coupled_builder():
    shell_coords, shell_conn = ff.structured_plate_grid(nx=1, ny=1, length_x=1.0, length_y=0.3)
    shell_section = ff.ShellSection(E=2.0e5, nu=0.3, thickness=0.03)
    shell_K = ff.assemble_flat_shell_stiffness(shell_coords, shell_conn, shell_section, format="csr")
    shell_F = np.zeros((shell_K.shape[0],), dtype=float)

    root_node = int(np.flatnonzero(np.isclose(shell_coords[:, 0], 1.0) & np.isclose(shell_coords[:, 1], 0.0))[0])
    beam_coords, beam_conn = ff.structured_beam_chain(n_elems=2, length=0.7, origin=(1.0, 0.0, 0.0))
    beam_section = ff.BeamSection(E=2.0e5, G=7.7e4, A=1.0e-2, Iy=1.0e-5, Iz=1.0e-5, J=2.0e-5)
    beam_K = ff.assemble_beam_stiffness(beam_coords, beam_conn, beam_section, format="csr")
    beam_F = ff.assemble_beam_point_load(beam_coords.shape[0], beam_coords.shape[0] - 1, force=(0.0, 0.0, -1.0))

    structural_K = sp.block_diag((shell_K, beam_K), format="csr")
    structural_F = np.concatenate([shell_F, np.asarray(beam_F, dtype=float)])
    beam_offset = shell_K.shape[0]
    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_K, structural_F)
    builder.register_field("shell", n_dofs=shell_K.shape[0], value_dim=1, offset=0)
    builder.register_field("beam", n_dofs=beam_K.shape[0], value_dim=1, offset=beam_offset)
    builder.add_dof_tie_constraint(
        master="shell",
        slave="beam",
        master_dofs=ff.shell_node_dofs([root_node]),
        slave_dofs=ff.beam_node_dofs([0]),
    )

    fixed_shell_nodes = np.flatnonzero(np.isclose(shell_coords[:, 0], 0.0))
    fixed = ff.shell_node_dofs(fixed_shell_nodes)
    u = np.asarray(
        builder.build().solve(
            format="csr",
            dirichlet_dofs=fixed,
            dirichlet_vals=np.zeros((fixed.size,), dtype=float),
        ),
        dtype=float,
    )

    shell_root = u[ff.shell_node_dofs([root_node])]
    beam_root = u[beam_offset + ff.beam_node_dofs([0])]
    beam_tip = u[beam_offset + ff.beam_node_dofs([beam_coords.shape[0] - 1])]
    np.testing.assert_allclose(shell_root, beam_root, rtol=1.0e-9, atol=1.0e-9)
    assert float(beam_tip[2]) < float(beam_root[2])

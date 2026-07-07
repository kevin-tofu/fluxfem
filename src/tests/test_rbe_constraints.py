from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import fluxfem as ff


def test_rbe2_constraint_matrix_reproduces_rigid_kinematics():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    C = ff.assemble_rbe2_constraint_matrix(x_ref, x_slave, backend="numpy")
    C_np = np.asarray(C)
    assert C_np.shape == (6, 12)

    u_ref = np.array([0.3, -0.2, 0.1], dtype=float)
    w_ref = np.array([0.0, 0.0, 0.5], dtype=float)
    u_slave = np.asarray([u_ref + np.cross(w_ref, p - x_ref) for p in x_slave], dtype=float).reshape(-1)
    q = np.concatenate([u_ref, w_ref, u_slave], axis=0)

    assert np.allclose(C_np @ q, np.zeros((6,), dtype=float), atol=1e-12)


def test_fixed_rigid_hub_constraint_matches_rbe2_with_fixed_reference():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_hub = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    hub_dofs = np.array([[2, 3, 4], [7, 8, 9]], dtype=int)
    C_hub = ff.assemble_fixed_rigid_hub_constraint_matrix(
        x_ref,
        x_hub,
        hub_dofs,
        n_structural_dofs=12,
    )
    C_rbe2 = ff.assemble_rbe2_constraint_matrix(x_ref, x_hub, backend="numpy")

    assert C_hub.shape == (6, 12)
    expected = np.zeros((6, 12), dtype=float)
    expected[:, hub_dofs.reshape(-1)] = C_rbe2[:, 6:]
    np.testing.assert_allclose(C_hub, expected, atol=1.0e-12)


def test_fixed_rigid_hub_constraint_kkt_enforces_zero_hub_dofs():
    k = sp.eye(9, format="csr")
    f = np.ones(9, dtype=float)
    x_ref = np.zeros(3, dtype=float)
    x_hub = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    hub_dofs = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    C = ff.assemble_fixed_rigid_hub_constraint_matrix(x_ref, x_hub, hub_dofs, n_structural_dofs=9)
    constraints = ff.LinearConstraintSystem(C)

    u = np.asarray(constraints.solve(k, f, solver="spsolve"), dtype=float)

    np.testing.assert_allclose(u[hub_dofs.reshape(-1)], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(u[6:], 1.0, atol=1.0e-12)


def test_rigid_hub_constraint_matrix_preserves_hub_rigid_motion():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_hub = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    hub_dofs = np.array([[0, 1, 2], [6, 7, 8]], dtype=int)
    ref_dofs = np.array([10, 11, 12, 13, 14, 15], dtype=int)
    C = ff.assemble_rigid_hub_constraint_matrix(
        x_ref,
        x_hub,
        hub_dofs,
        hub_reference_dofs=ref_dofs,
        n_total_dofs=18,
    )

    u_ref = np.array([0.3, -0.2, 0.1], dtype=float)
    w_ref = np.array([0.0, 0.0, 0.5], dtype=float)
    q = np.zeros(18, dtype=float)
    q[ref_dofs] = np.concatenate([u_ref, w_ref])
    for dofs, point in zip(hub_dofs, x_hub, strict=True):
        q[dofs] = u_ref + np.cross(w_ref, point - x_ref)

    assert C.shape == (6, 18)
    np.testing.assert_allclose(C @ q, 0.0, atol=1.0e-12)


def test_rbe3_constraint_matrix_preserves_rigid_motion():
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
    C_np = np.asarray(C)
    assert C_np.shape == (6, 15)

    u_ref = np.array([0.3, -0.2, 0.1], dtype=float)
    w_ref = np.array([0.1, -0.05, 0.2], dtype=float)
    u_slave = np.asarray([u_ref + np.cross(w_ref, p - x_ref) for p in x_slave], dtype=float).reshape(-1)
    q = np.concatenate([u_ref, w_ref, u_slave], axis=0)

    assert np.allclose(C_np @ q, np.zeros((6,), dtype=float), atol=1e-12)


def test_rbe_constraint_matrix_jax_backend_is_rejected():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array([[1.0, 0.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="numpy' only"):
        ff.assemble_rbe2_constraint_matrix(x_ref, x_slave, backend="jax")

    with pytest.raises(ValueError, match="numpy' only"):
        ff.assemble_rbe3_constraint_matrix(x_ref, x_slave, weights=np.array([1.0], dtype=float), backend="jax")


def test_builder_rbe2_remote_spring_transfers_remote_translation():
    builder = ff.NumpyCoupledSystemBuilder.from_structural(sp.eye(3, format="csr"), np.zeros((3,), dtype=float))
    builder.register_field("slave", n_dofs=3, value_dim=1, offset=0)
    builder.append_remote_point("remote", point=np.array([0.0, 0.0, 0.0], dtype=float))
    builder.add_remote_spring(
        "remote",
        translational_stiffness=np.array([20.0, 20.0, 20.0], dtype=float),
        rotational_stiffness=np.array([5.0, 5.0, 5.0], dtype=float),
        translational_target=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    builder.add_rbe2_constraint(
        master="remote",
        slave="slave",
        ref_point=np.array([0.0, 0.0, 0.0], dtype=float),
        slave_coords=np.array([[1.0, 0.0, 0.0]], dtype=float),
        rho=0.0,
        backend="numpy",
    )

    u = np.asarray(builder.build().solve(format="csr", diagonal_shift=1e-8), dtype=float)
    q_slave = u[:3]
    q_remote = u[3:9]

    assert np.allclose(q_slave, q_remote[:3], atol=1e-6)
    assert np.allclose(q_remote[:3], np.array([20.0 / 21.0, 0.0, 0.0]), atol=1e-6)


def test_builder_rbe3_constraint_preserves_rigid_mode_in_kkt_rows():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    builder = ff.NumpyCoupledSystemBuilder.from_structural(sp.csr_matrix((12, 12)), np.zeros((12,), dtype=float))
    builder.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    builder.register_field("slave", n_dofs=6, value_dim=1, offset=6)
    builder.add_rbe3_constraint(
        master="remote",
        slave="slave",
        ref_point=x_ref,
        slave_coords=x_slave,
        weights=np.array([0.25, 0.75], dtype=float),
        rho=0.0,
        backend="numpy",
    )

    K, _ = builder.build().assemble(format="csr")
    C = K.toarray()[12:, :12]

    u_ref = np.array([0.2, -0.1, 0.05], dtype=float)
    w_ref = np.array([0.0, 0.0, 0.4], dtype=float)
    u_slave = np.asarray([u_ref + np.cross(w_ref, p - x_ref) for p in x_slave], dtype=float).reshape(-1)
    q = np.concatenate([u_ref, w_ref, u_slave], axis=0)

    assert np.allclose(C @ q, np.zeros((6,), dtype=float), atol=1e-12)


def test_builder_append_dof_copy_field_adds_expected_tie_constraint():
    builder = ff.NumpyCoupledSystemBuilder.from_structural(sp.eye(5, format="csr"), np.zeros((5,), dtype=float))
    builder.register_field("workpiece", n_dofs=5, value_dim=1, offset=0)
    builder.append_dof_copy_field("patch", source="workpiece", source_dofs=np.array([1, 3], dtype=int))

    K, _ = builder.build().assemble(format="csr")
    C = K.toarray()[7:, :7]
    q = np.array([0.0, 0.4, 0.0, -0.2, 0.0, 0.4, -0.2], dtype=float)

    assert C.shape == (2, 7)
    assert np.allclose(C @ q, np.zeros((2,), dtype=float), atol=1e-12)


def test_builder_add_dof_tie_constraint_adds_selected_dof_rows_with_rhs():
    builder = ff.NumpyCoupledSystemBuilder.from_structural(sp.eye(5, format="csr"), np.zeros((5,), dtype=float))
    builder.register_field("master", n_dofs=3, value_dim=1, offset=0)
    builder.register_field("slave", n_dofs=2, value_dim=1, offset=3)
    builder.add_dof_tie_constraint(
        master="master",
        slave="slave",
        master_dofs=np.array([0, 2], dtype=int),
        slave_dofs=np.array([1, 0], dtype=int),
        rhs=np.array([0.25, -0.5], dtype=float),
    )

    K, F = builder.build().assemble(format="csr")
    C = K.toarray()[5:, :5]
    q = np.array([0.75, 0.0, -0.2, 0.3, 0.5], dtype=float)

    assert C.shape == (2, 5)
    np.testing.assert_allclose(C @ q, np.array([0.25, -0.5]))
    np.testing.assert_allclose(F[5:], np.array([0.25, -0.5]))


def test_builder_rbe2_rejects_slave_size_mismatch():
    builder = ff.NumpyCoupledSystemBuilder.from_structural(sp.eye(6, format="csr"), np.zeros((6,), dtype=float))
    builder.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    builder.append_field("slave", n_dofs=2, value_dim=1)

    with pytest.raises(ValueError, match="RBE2 slave field size must match 3 \\* n_slave_nodes"):
        builder.add_rbe2_constraint(
            master="remote",
            slave="slave",
            ref_point=np.array([0.0, 0.0, 0.0], dtype=float),
            slave_coords=np.array([[1.0, 0.0, 0.0]], dtype=float),
        )


def test_rbe3_constraint_matrix_rejects_zero_sum_weights_when_normalizing():
    with pytest.raises(ValueError, match="weights sum must be non-zero"):
        ff.assemble_rbe3_constraint_matrix(
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([[1.0, 0.0, 0.0]], dtype=float),
            weights=np.array([0.0], dtype=float),
            normalize_weights=True,
            backend="numpy",
        )

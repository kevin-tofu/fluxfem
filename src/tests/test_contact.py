import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff

def test_plane_contact_residual_scatter_and_jacobian():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=4,
            dofs=jnp.array([[0, 1], [2, 3]]),
            normals=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
            gaps0=jnp.array([0.1, -0.2]),
        ),
        penalty=10.0,
    )
    u = jnp.array([0.0, -0.05, 0.05, 0.0])
    residual = contact.residual(u)

    np.testing.assert_allclose(np.asarray(residual), np.array([0.0, 0.0, -1.5, 0.0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.gaps(u)), np.array([0.05, -0.15]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.pressure(u)), np.array([0.0, 1.5]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.active_mask(u)), np.array([False, True]))
    np.testing.assert_allclose(np.asarray(contact.penetration_energy(u)), 0.1125, atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.force_norm(u)), 1.5, atol=1e-6)
    assert int(contact.active_count(u)) == 1
    jacobian = jax.jacrev(contact.residual)(u)
    assert jacobian.shape == (4, 4)

def test_plane_contact_rejects_out_of_range_dofs():
    try:
        ff.make_unilateral_plane_contact_residual(
            n_dofs=2,
            contact_dofs=jnp.array([[0, 2]]),
            normals=jnp.array([[1.0, 0.0]]),
            gaps0=jnp.array([0.0]),
            penalty=1.0,
        )
    except ValueError as exc:
        assert "outside the full DOF range" in str(exc)
    else:
        raise AssertionError("expected out-of-range contact dofs to raise")

def test_vector_dof_helpers():
    dofs = ff.vector_dofs_from_nodes(jnp.array([2, 0]), dim=3)
    np.testing.assert_array_equal(np.asarray(dofs), np.array([6, 7, 8, 0, 1, 2]))

    class SurfaceLike:
        conn = np.array([[3, 1], [1, 2]])

    retained = ff.retained_dofs_from_surface(SurfaceLike(), dim=2)
    np.testing.assert_array_equal(np.asarray(retained), np.array([2, 3, 4, 5, 6, 7]))

def test_orthonormal_tangent_basis_for_1d_2d_and_3d_normals():
    t1 = ff.orthonormal_tangent_basis(jnp.array([[1.0], [-2.0]]))
    assert t1.shape == (2, 0, 1)

    normals2 = jnp.array([[0.0, 1.0], [1.0, 1.0]])
    t2 = ff.orthonormal_tangent_basis(normals2)
    n2 = normals2 / jnp.linalg.norm(normals2, axis=1, keepdims=True)
    np.testing.assert_allclose(np.asarray(jnp.einsum("id,itd->it", n2, t2)), np.zeros((2, 1)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(t2, axis=2)), np.ones((2, 1)), atol=1e-6)

    normals3 = jnp.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    t3 = ff.orthonormal_tangent_basis(normals3)
    n3 = normals3 / jnp.linalg.norm(normals3, axis=1, keepdims=True)
    gram = jnp.einsum("iad,ibd->iab", t3, t3)
    np.testing.assert_allclose(np.asarray(jnp.einsum("id,itd->it", n3, t3)), np.zeros((2, 2)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(gram), np.broadcast_to(np.eye(2), (2, 2, 2)), atol=1e-6)

def test_tangential_penalty_history_updates_stick_and_slip():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=2,
            dofs=jnp.array([[0, 1]]),
            normals=jnp.array([[0.0, 1.0]]),
            gaps0=jnp.array([-0.1]),
        ),
        penalty=10.0,
    )
    u_prev = jnp.array([0.0, 0.0])
    u_stick = jnp.array([0.2, 0.0])
    history_stick = ff.update_tangential_penalty_history(
        contact,
        u_stick,
        u_prev,
        None,
        mu=0.5,
        tangential_penalty=2.0,
    )

    np.testing.assert_allclose(np.asarray(history_stick.tangential_slip), np.array([[0.2]]), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(history_stick.stick), np.array([True]))
    np.testing.assert_allclose(np.asarray(history_stick.friction_force), np.array([[-0.4, 0.0]]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(ff.slip_norm(history_stick)), 0.2, atol=1e-6)
    assert int(ff.stick_count(history_stick)) == 1

    u_slip = jnp.array([1.0, 0.0])
    history_slip = ff.update_tangential_penalty_history(
        contact,
        u_slip,
        u_prev,
        None,
        mu=0.5,
        tangential_penalty=2.0,
    )
    np.testing.assert_allclose(np.asarray(history_slip.tangential_slip), np.array([[0.25]]), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(history_slip.stick), np.array([False]))
    np.testing.assert_allclose(np.asarray(history_slip.friction_force), np.array([[-0.5, 0.0]]), atol=1e-6)

def test_friction_residual_from_history_scatters_for_contact_types():
    plane = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=4,
            dofs=jnp.array([[0, 1]]),
            normals=jnp.array([[0.0, 1.0]]),
            gaps0=jnp.array([-0.1]),
        ),
        penalty=10.0,
    )
    plane_history = ff.TangentialPenaltyHistory(
        tangential_slip=jnp.array([[0.2]]),
        stick=jnp.array([True]),
        friction_force=jnp.array([[-0.4, 0.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(ff.friction_residual_from_history(plane, plane_history)),
        np.array([-0.4, 0.0, 0.0, 0.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(ff.make_friction_residual(plane, plane_history)(jnp.zeros(4))),
        np.array([-0.4, 0.0, 0.0, 0.0]),
        atol=1e-6,
    )

    paired = ff.PairedPenaltyContact(
        ff.PairedContactKinematics(
            slave_dofs=jnp.array([[0, 1]]),
            master_dofs=jnp.array([[2, 3]]),
            normals=jnp.array([[0.0, 1.0]]),
            gaps0=jnp.array([-0.1]),
            n_dofs=4,
        ),
        penalty=10.0,
    )
    np.testing.assert_allclose(
        np.asarray(ff.friction_residual_from_history(paired, plane_history)),
        np.array([-0.4, 0.0, 0.4, 0.0]),
        atol=1e-6,
    )

    node_surface = ff.NodeSurfacePenaltyContact(
        ff.NodeSurfaceContactKinematics(
            slave_dofs=jnp.array([[0, 1]]),
            master_dofs=jnp.array([[[2, 3], [4, 5]]]),
            master_weights=jnp.array([[0.25, 0.75]]),
            normals=jnp.array([[0.0, 1.0]]),
            gaps0=jnp.array([-0.1]),
            n_dofs=6,
        ),
        penalty=10.0,
    )
    np.testing.assert_allclose(
        np.asarray(ff.friction_residual_from_history(node_surface, plane_history)),
        np.array([-0.4, 0.0, 0.1, 0.0, 0.3, 0.0]),
        atol=1e-6,
    )

def test_active_contact_state_freezes_active_set():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=2,
            dofs=jnp.array([[0], [1]]),
            normals=jnp.array([[1.0], [1.0]]),
            gaps0=jnp.array([-0.1, 0.1]),
        ),
        penalty=10.0,
    )
    u0 = jnp.array([0.0, 0.0])
    state = ff.update_active_contact_state(contact, u0)
    np.testing.assert_array_equal(np.asarray(state.active), np.array([True, False]))

    frozen_residual = contact.residual_with_state(state)
    u1 = jnp.array([0.0, -0.2])
    np.testing.assert_allclose(np.asarray(contact.residual(u1)), np.array([-1.0, -1.0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(frozen_residual(u1)), np.array([-1.0, 0.0]), atol=1e-6)

    new_state = contact.state_from_displacement(u1)
    assert bool(new_state.changed(state))

def test_plane_contact_kinematics_from_surface():
    class SurfaceLike:
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        conn = np.array([[1, 2]])

    kin = ff.plane_contact_kinematics_from_surface(
        SurfaceLike(),
        dim=3,
        normal=jnp.array([1.0, 0.0, 0.0]),
        plane_offset=0.9,
    )

    np.testing.assert_array_equal(np.asarray(kin.dofs), np.array([[3, 4, 5], [6, 7, 8]]))
    np.testing.assert_allclose(np.asarray(kin.gaps0), np.array([0.1, 0.1]), atol=1e-6)

def test_paired_penalty_contact_action_reaction_and_jacobian():
    kin = ff.PairedContactKinematics(
        slave_dofs=jnp.array([[0], [1]]),
        master_dofs=jnp.array([[2], [3]]),
        normals=jnp.array([[1.0], [1.0]]),
        gaps0=jnp.array([-0.1, 0.2]),
        n_dofs=4,
    )
    contact = ff.PairedPenaltyContact(kin, penalty=10.0)
    u = jnp.zeros(4)
    residual = contact.residual(u)

    np.testing.assert_allclose(np.asarray(residual), np.array([-1.0, 0.0, 1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.active_mask(u)), np.array([True, False]))
    np.testing.assert_allclose(np.asarray(contact.penetration_energy(u)), 0.05, atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.force_norm(u)), np.sqrt(2.0), atol=1e-6)
    assert int(contact.active_count(u)) == 1
    jacobian = jax.jacrev(contact.residual)(u)
    assert jacobian.shape == (4, 4)

    state = contact.state_from_displacement(u)
    frozen = contact.residual_with_state(state)
    u_inactive_to_active = jnp.array([0.0, -0.3, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(contact.residual(u_inactive_to_active)),
        np.array([-1.0, -1.0, 1.0, 1.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(frozen(u_inactive_to_active)),
        np.array([-1.0, 0.0, 1.0, 0.0]),
        atol=1e-6,
    )

def test_paired_contact_kinematics_from_surfaces_nearest_nodes():
    class SlaveSurface:
        coords = np.array([[0.0], [1.0], [2.0]])
        conn = np.array([[0, 2]])

    class MasterSurface:
        coords = np.array([[0.1], [1.1], [2.2]])
        conn = np.array([[0, 2]])

    kin = ff.paired_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=1,
        normal=jnp.array([1.0]),
    )
    np.testing.assert_array_equal(np.asarray(kin.slave_dofs), np.array([[0], [2]]))
    np.testing.assert_array_equal(np.asarray(kin.master_dofs), np.array([[0], [2]]))
    np.testing.assert_allclose(np.asarray(kin.gaps0), np.array([-0.1, -0.2]), atol=1e-6)

def test_node_surface_penalty_contact_distributes_master_force():
    kin = ff.NodeSurfaceContactKinematics(
        slave_dofs=jnp.array([[0]]),
        master_dofs=jnp.array([[[1], [2]]]),
        master_weights=jnp.array([[0.25, 0.75]]),
        normals=jnp.array([[1.0]]),
        gaps0=jnp.array([-0.2]),
        n_dofs=3,
    )
    contact = ff.NodeSurfacePenaltyContact(kin, penalty=10.0)
    residual = contact.residual(jnp.zeros(3))

    np.testing.assert_allclose(np.asarray(residual), np.array([-2.0, 0.5, 1.5]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.sum(residual)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.penetration_energy(jnp.zeros(3))), 0.2, atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.force_norm(jnp.zeros(3))), 2.5495098, atol=1e-6)
    assert int(contact.active_count(jnp.zeros(3))) == 1
    jacobian = jax.jacrev(contact.residual)(jnp.zeros(3))
    assert jacobian.shape == (3, 3)

    state = contact.state_from_displacement(jnp.zeros(3))
    frozen = contact.residual_with_state(state)
    np.testing.assert_allclose(np.asarray(frozen(jnp.zeros(3))), np.asarray(residual), atol=1e-6)

def test_node_surface_contact_kinematics_from_surfaces():
    class SlaveSurface:
        coords = np.array([[0.5], [2.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = np.array([[0.0], [1.0], [3.0], [4.0]])
        conn = np.array([[0, 1], [2, 3]])

    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=1,
        normal=jnp.array([1.0]),
    )
    np.testing.assert_array_equal(np.asarray(kin.slave_dofs), np.array([[0]]))
    np.testing.assert_array_equal(np.asarray(kin.master_dofs), np.array([[[0], [1]]]))
    np.testing.assert_allclose(np.asarray(kin.master_weights), np.array([[0.5, 0.5]]))
    np.testing.assert_allclose(np.asarray(kin.gaps0), np.array([0.0]), atol=1e-6)

def test_node_surface_contact_selects_closest_projected_facet_not_centroid():
    class SlaveSurface:
        coords = np.array(
            [
                [1.0, 0.1],
                [0.0, 0.0],
                [10.0, 0.0],
                [1.2, 1.0],
                [1.2, 2.0],
            ]
        )
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
    )

    np.testing.assert_array_equal(np.asarray(kin.master_dofs), np.array([[[2, 3], [4, 5]]]))
    np.testing.assert_allclose(np.asarray(kin.master_weights), np.array([[0.9, 0.1]]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.gaps0), np.array([0.1]), atol=1e-6)

def test_node_surface_contact_auto_normal_updates_from_deformed_facet():
    class SlaveSurface:
        coords = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2]])

    u = jnp.zeros(6)
    u = u.at[5].set(1.0)  # rotate master line from horizontal to diagonal
    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        displacement=u,
    )

    expected_normal = np.array([[-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]])
    np.testing.assert_allclose(np.asarray(kin.normals), expected_normal, atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.master_weights), np.array([[0.5, 0.5]]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.gaps(u)), np.array([1.0 / np.sqrt(2.0)]), atol=1e-6)

def test_surface_quadrature_contact_centroid_residual_and_force_balance():
    class SlaveSurface:
        coords = np.array([[0.0, -0.1], [1.0, -0.1], [0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3]])

    kin = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="centroid",
    )
    contact = ff.SurfaceQuadraturePenaltyContact(kin, penalty=10.0)
    residual = contact.residual(jnp.zeros(8))

    np.testing.assert_array_equal(np.asarray(kin.slave_dofs), np.array([[[0, 1], [2, 3]]]))
    np.testing.assert_allclose(np.asarray(kin.slave_weights), np.array([[0.5, 0.5]]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.master_weights), np.array([[0.5, 0.5]]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.gaps0), np.array([-0.1]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(contact.penetration_energy(jnp.zeros(8))), 0.05, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(residual),
        np.array([0.0, -0.5, 0.0, -0.5, 0.0, 0.5, 0.0, 0.5]),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(jnp.sum(residual)), 0.0, atol=1e-6)
    assert jax.jacrev(contact.residual)(jnp.zeros(8)).shape == (8, 8)

def test_surface_quadrature_contact_quad_vertices_rule_has_four_points():
    class SlaveSurface:
        coords = np.array(
            [
                [0.0, 0.0, -0.1],
                [1.0, 0.0, -0.1],
                [1.0, 1.0, -0.1],
                [0.0, 1.0, -0.1],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        conn = np.array([[0, 1, 2, 3]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[4, 5, 6, 7]])

    kin = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=3,
        normal=jnp.array([0.0, 0.0, 1.0]),
        quadrature_rule="vertices",
    )
    contact = ff.SurfaceQuadraturePenaltyContact(kin, penalty=20.0)
    residual = contact.residual(jnp.zeros(24))

    assert kin.slave_dofs.shape == (4, 4, 3)
    np.testing.assert_allclose(np.asarray(kin.quadrature_weights), 0.25 * np.ones(4), atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.gaps0), -0.1 * np.ones(4), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.sum(residual.reshape(-1, 3), axis=0)), np.zeros(3), atol=1e-6)
    assert int(contact.active_count(jnp.zeros(24))) == 4

def test_surface_quadrature_contact_matches_independent_weighted_penalty_form():
    class SlaveSurface:
        coords = np.array([[0.0, -0.05], [1.0, -0.05], [0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3]])

    kin = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="vertices",
    )
    contact = ff.SurfaceQuadraturePenaltyContact(kin, penalty=12.0)
    u = jnp.array(
        [
            0.02,
            0.0,
            -0.01,
            0.08,
            0.03,
            0.0,
            -0.02,
            -0.01,
        ],
        dtype=jnp.float32,
    )

    n_dofs = int(kin.n_dofs)
    displacement = np.asarray(u)
    normal = np.asarray(kin.normals)
    gaps0 = np.asarray(kin.gaps0)
    quadrature_weights = np.asarray(kin.quadrature_weights)
    slave_dofs = np.asarray(kin.slave_dofs)
    master_dofs = np.asarray(kin.master_dofs)
    slave_weights = np.asarray(kin.slave_weights)
    master_weights = np.asarray(kin.master_weights)

    reference_gaps = []
    contact_rows = []
    for q in range(gaps0.shape[0]):
        row = np.zeros(n_dofs)
        for a, weight in enumerate(slave_weights[q]):
            row[slave_dofs[q, a]] += weight * normal[q]
        for a, weight in enumerate(master_weights[q]):
            row[master_dofs[q, a]] -= weight * normal[q]
        contact_rows.append(row)
        reference_gaps.append(gaps0[q] + row @ displacement)

    reference_gaps = np.asarray(reference_gaps)
    contact_rows = np.asarray(contact_rows)
    active = reference_gaps < 0.0
    reference_residual = np.zeros(n_dofs)
    reference_jacobian = np.zeros((n_dofs, n_dofs))
    for q in range(gaps0.shape[0]):
        if active[q]:
            scale = float(contact.penalty) * quadrature_weights[q]
            reference_residual += scale * reference_gaps[q] * contact_rows[q]
            reference_jacobian += scale * np.outer(contact_rows[q], contact_rows[q])

    np.testing.assert_allclose(np.asarray(contact.gaps(u)), reference_gaps, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(contact.active_mask(u)), active)
    np.testing.assert_allclose(np.asarray(contact.residual(u)), reference_residual, atol=1e-6)
    np.testing.assert_allclose(np.asarray(jax.jacrev(contact.residual)(u)), reference_jacobian, atol=1e-6)

def test_node_surface_contact_quad_shape_weights():
    class SlaveSurface:
        coords = np.array(
            [
                [0.75, 0.25, -0.05],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2, 3, 4]])

    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=3,
        normal=jnp.array([0.0, 0.0, 1.0]),
    )
    expected = np.array([[0.1875, 0.5625, 0.1875, 0.0625]])
    np.testing.assert_allclose(np.asarray(kin.master_weights), expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.gaps0), np.array([-0.05]), atol=1e-6)

def test_node_surface_kinematics_updates_weights_from_displacement():
    class SlaveSurface:
        coords = np.array(
            [
                [0.25, 0.25, -0.05],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2, 3, 4]])

    u = jnp.zeros(15)
    u = u.at[0].set(0.5)  # move slave from x=0.25 to x=0.75 for pairing/weights
    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=3,
        normal=jnp.array([0.0, 0.0, 1.0]),
        displacement=u,
    )
    expected = np.array([[0.1875, 0.5625, 0.1875, 0.0625]])
    np.testing.assert_allclose(np.asarray(kin.master_weights), expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(kin.gaps(u)), np.array([-0.05]), atol=1e-6)

def test_node_surface_contact_search_cache_freezes_facet_pairing():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    kin0 = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
    )
    cache = kin0.search_cache()

    u = jnp.zeros(10)
    u = u.at[0].set(2.25)  # closest facet would switch without the cache
    kin_cached = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        displacement=u,
        search_cache=cache,
    )
    kin_researched = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        displacement=u,
    )

    np.testing.assert_array_equal(np.asarray(kin0.master_facet_ids), np.array([0]))
    np.testing.assert_array_equal(np.asarray(cache.master_facet_ids), np.array([0]))
    np.testing.assert_array_equal(np.asarray(kin_cached.master_facet_ids), np.array([0]))
    np.testing.assert_array_equal(np.asarray(kin_researched.master_facet_ids), np.array([1]))
    np.testing.assert_array_equal(np.asarray(kin_cached.master_dofs), np.array([[[2, 3], [4, 5]]]))
    np.testing.assert_array_equal(np.asarray(kin_researched.master_dofs), np.array([[[6, 7], [8, 9]]]))

def test_node_surface_candidate_set_prunes_closest_facet_search():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        candidate_facet_ids=ff.ContactCandidateSet(jnp.array([1])),
    )

    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([1]))
    np.testing.assert_array_equal(np.asarray(kin.master_dofs), np.array([[[6, 7], [8, 9]]]))

def test_node_surface_per_contact_candidate_set_prunes_each_slave_node():
    class SlaveSurface:
        coords = np.array(
            [
                [0.25, -0.05],
                [2.75, -0.05],
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        conn = np.array([[0], [1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    candidates = ff.contact_candidate_set_from_per_contact([jnp.array([0]), jnp.array([1])])
    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        candidate_facet_ids=candidates,
    )

    np.testing.assert_array_equal(np.asarray(candidates.contact_offsets), np.array([0, 1, 2]))
    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([0, 1]))

def test_contact_candidate_set_from_bounding_boxes_prunes_far_facets():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    candidates = ff.contact_candidate_set_from_bounding_boxes(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        search_radius=0.2,
    )
    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        candidate_facet_ids=candidates,
    )

    np.testing.assert_array_equal(np.asarray(candidates.master_facet_ids), np.array([0]))
    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([0]))

def test_contact_candidate_set_from_bounding_boxes_uses_displacement():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    u = jnp.zeros(10)
    u = u.at[0].set(5.0)
    candidates = ff.contact_candidate_set_from_bounding_boxes(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        search_radius=0.2,
        displacement=u,
    )

    np.testing.assert_array_equal(np.asarray(candidates.master_facet_ids), np.array([1]))

def test_node_surface_candidate_set_from_bounding_boxes_builds_per_node_candidates():
    class SlaveSurface:
        coords = np.array(
            [
                [0.25, -0.05],
                [5.75, -0.05],
                [0.0, 0.0],
                [1.0, 0.0],
                [5.0, 0.0],
                [6.0, 0.0],
            ]
        )
        conn = np.array([[0], [1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    candidates = ff.node_surface_candidate_set_from_bounding_boxes(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        search_radius=0.2,
    )
    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        candidate_facet_ids=candidates,
    )

    np.testing.assert_array_equal(np.asarray(candidates.contact_offsets), np.array([0, 1, 2]))
    np.testing.assert_array_equal(np.asarray(candidates.master_facet_ids), np.array([0, 1]))
    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([0, 1]))

def test_node_surface_candidate_set_from_bounding_boxes_uses_displacement():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    u = jnp.zeros(10)
    u = u.at[0].set(5.0)
    candidates = ff.node_surface_candidate_set_from_bounding_boxes(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        search_radius=0.2,
        displacement=u,
    )

    np.testing.assert_array_equal(np.asarray(candidates.master_facet_ids), np.array([1]))

def test_contact_aabb_index_query_and_node_surface_candidates():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [5.75, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0], [1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    index = ff.contact_aabb_index_from_surface(MasterSurface(), dim=2, cell_size=1.0)
    near_left = index.query_box(np.array([0.2, -0.2]), np.array([0.3, 0.2]))
    candidates = ff.node_surface_candidate_set_from_aabb_index(
        SlaveSurface(),
        index,
        dim=2,
        search_radius=0.2,
    )
    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        candidate_facet_ids=candidates,
    )

    np.testing.assert_array_equal(near_left, np.array([0]))
    np.testing.assert_array_equal(np.asarray(candidates.master_facet_ids), np.array([0, 1]))
    np.testing.assert_array_equal(np.asarray(candidates.contact_offsets), np.array([0, 1, 2]))
    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([0, 1]))

def test_contact_aabb_index_from_surface_uses_displacement():
    class SlaveSurface:
        coords = np.array([[5.25, -0.05], [0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2]])

    u = jnp.zeros(6)
    u = u.at[2].set(5.0)
    u = u.at[4].set(5.0)
    index = ff.contact_aabb_index_from_surface(
        MasterSurface(),
        dim=2,
        displacement=u,
        cell_size=1.0,
    )
    candidates = ff.node_surface_candidate_set_from_aabb_index(
        SlaveSurface(),
        index,
        dim=2,
        search_radius=0.2,
    )

    np.testing.assert_array_equal(np.asarray(candidates.master_facet_ids), np.array([0]))

def test_node_surface_neighbor_list_refresh_policy_and_rebuild():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    u0 = jnp.zeros(10)
    neighbors0 = ff.node_surface_neighbor_list_from_bounding_boxes(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        search_radius=0.2,
        skin=1.0,
        displacement=u0,
    )
    u_small = u0.at[0].set(0.4)
    u_large = u0.at[0].set(5.0)

    assert not bool(neighbors0.needs_refresh(u_small))
    assert bool(neighbors0.needs_refresh(u_large))
    np.testing.assert_allclose(np.asarray(neighbors0.max_drift(u_small)), 0.4, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(neighbors0.candidate_set.master_facet_ids), np.array([0]))

    neighbors1 = ff.node_surface_neighbor_list_from_bounding_boxes(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        search_radius=0.2,
        skin=1.0,
        displacement=u_large,
    )
    kin = ff.node_surface_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        displacement=u_large,
        candidate_facet_ids=neighbors1.candidate_set,
    )

    assert not bool(neighbors1.needs_refresh(u_large))
    np.testing.assert_array_equal(np.asarray(neighbors1.candidate_set.master_facet_ids), np.array([1]))
    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([1]))

def test_node_surface_neighbor_list_from_aabb_index_refresh_policy():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    u0 = jnp.zeros(10)
    index = ff.contact_aabb_index_from_surface(MasterSurface(), dim=2, n_total_nodes=5, displacement=u0, cell_size=1.0)
    neighbors = ff.node_surface_neighbor_list_from_aabb_index(
        SlaveSurface(),
        index,
        dim=2,
        search_radius=0.2,
        skin=1.0,
        n_total_nodes=5,
        displacement=u0,
    )

    assert not bool(neighbors.needs_refresh(u0.at[0].set(0.4)))
    assert bool(neighbors.needs_refresh(u0.at[0].set(5.0)))
    np.testing.assert_array_equal(np.asarray(neighbors.candidate_set.master_facet_ids), np.array([0]))

def test_node_surface_contact_search_manager_builds_and_refreshes_contact():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2], [3, 4]])

    manager0 = ff.make_node_surface_contact_search_manager(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        n_total_nodes=5,
        search_radius=0.2,
        skin=1.0,
        penalty=10.0,
        normal=jnp.array([0.0, 1.0]),
        cell_size=1.0,
    )
    u0 = jnp.zeros(10)
    contact0, manager1 = manager0.build_contact(u0)
    contact1, manager2 = manager1.build_contact(u0.at[0].set(0.4))
    contact2, manager3 = manager2.with_search_cache(None).build_contact(u0.at[0].set(5.0))

    assert isinstance(contact0, ff.NodeSurfacePenaltyContact)
    assert manager1.index is not None
    assert manager1.neighbor_list is not None
    assert manager1.search_cache is not None
    np.testing.assert_array_equal(np.asarray(contact0.kinematics.master_facet_ids), np.array([0]))
    np.testing.assert_array_equal(np.asarray(contact1.kinematics.master_facet_ids), np.array([0]))
    np.testing.assert_array_equal(np.asarray(contact2.kinematics.master_facet_ids), np.array([1]))
    assert manager3.search_cache is not None

def test_contact_update_snapshot_detects_kinematics_changes():
    class SlaveSurface:
        coords = np.array(
            [
                [0.25, 0.25, -0.05],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2, 3, 4]])

    def build_contact(u):
        kin = ff.node_surface_contact_kinematics_from_surfaces(
            SlaveSurface(),
            MasterSurface(),
            dim=3,
            normal=jnp.array([0.0, 0.0, 1.0]),
            displacement=u,
        )
        return ff.NodeSurfacePenaltyContact(kin, penalty=10.0)

    u0 = jnp.zeros(15)
    u1 = u0.at[0].set(0.5)
    snapshot0 = ff.ContactUpdateSnapshot.from_contact(build_contact(u0), u0)
    snapshot1 = ff.ContactUpdateSnapshot.from_contact(build_contact(u1), u1)

    assert bool(snapshot1.changed(snapshot0))
    residual = snapshot1.residual()
    assert residual(u1).shape == (15,)

def test_surface_quadrature_contact_search_cache_freezes_facet_pairing():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.75, -0.05], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    kin0 = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="vertices",
    )
    cache = ff.contact_search_cache_from_kinematics(kin0)

    u = jnp.zeros(12)
    u = u.at[0].set(2.25)
    u = u.at[2].set(2.25)
    kin_cached = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        displacement=u,
        quadrature_rule="vertices",
        search_cache=cache,
    )
    kin_researched = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        displacement=u,
        quadrature_rule="vertices",
    )

    np.testing.assert_array_equal(np.asarray(kin0.master_facet_ids), np.array([0, 0]))
    np.testing.assert_array_equal(np.asarray(kin_cached.master_facet_ids), np.array([0, 0]))
    np.testing.assert_array_equal(np.asarray(kin_researched.master_facet_ids), np.array([1, 1]))
    np.testing.assert_array_equal(np.asarray(kin_cached.slave_facet_ids), np.array([0, 0]))

def test_surface_quadrature_candidate_set_prunes_closest_facet_search():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.75, -0.05], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    kin = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="vertices",
        candidate_facet_ids=jnp.array([1]),
    )

    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([1, 1]))
    np.testing.assert_array_equal(np.asarray(kin.master_dofs), np.array([[[8, 9], [10, 11]], [[8, 9], [10, 11]]]))

def test_surface_quadrature_per_contact_candidate_set_prunes_each_point():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [2.75, -0.05], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    candidates = ff.contact_candidate_set_from_per_contact([jnp.array([0]), jnp.array([1])])
    kin = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="vertices",
        candidate_facet_ids=candidates,
    )

    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([0, 1]))

def test_surface_quadrature_candidate_set_from_aabb_index_prunes_each_point():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [2.75, -0.05], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    index = ff.contact_aabb_index_from_surface(MasterSurface(), dim=2, cell_size=1.0)
    candidates = ff.surface_quadrature_candidate_set_from_aabb_index(
        SlaveSurface(),
        index,
        dim=2,
        search_radius=0.2,
        quadrature_rule="vertices",
    )
    kin = ff.surface_quadrature_contact_kinematics_from_surfaces(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="vertices",
        candidate_facet_ids=candidates,
    )

    np.testing.assert_array_equal(np.asarray(candidates.contact_offsets), np.array([0, 1, 2]))
    np.testing.assert_array_equal(np.asarray(kin.master_facet_ids), np.array([0, 1]))

def test_surface_quadrature_neighbor_list_from_aabb_index_refresh_policy():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.75, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    u0 = jnp.zeros(12)
    index = ff.contact_aabb_index_from_surface(MasterSurface(), dim=2, n_total_nodes=6, displacement=u0, cell_size=1.0)
    neighbors = ff.surface_quadrature_neighbor_list_from_aabb_index(
        SlaveSurface(),
        index,
        dim=2,
        search_radius=0.2,
        skin=1.0,
        n_total_nodes=6,
        displacement=u0,
        quadrature_rule="vertices",
    )

    assert not bool(neighbors.needs_refresh(u0.at[0].set(0.4)))
    assert bool(neighbors.needs_refresh(u0.at[0].set(5.0)))
    np.testing.assert_array_equal(np.asarray(neighbors.candidate_set.master_facet_ids), np.array([0, 0]))

def test_surface_quadrature_contact_search_manager_builds_and_refreshes_contact():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.75, -0.05], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3], [4, 5]])

    manager0 = ff.make_surface_quadrature_contact_search_manager(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        n_total_nodes=6,
        search_radius=0.2,
        skin=1.0,
        penalty=10.0,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="vertices",
        cell_size=1.0,
    )
    u0 = jnp.zeros(12)
    contact0, manager1 = manager0.build_contact(u0)
    contact1, manager2 = manager1.build_contact(u0.at[0].set(0.4))
    contact2, manager3 = manager2.with_search_cache(None).build_contact(u0.at[0].set(5.0).at[2].set(5.0))

    assert isinstance(contact0, ff.SurfaceQuadraturePenaltyContact)
    assert manager1.index is not None
    assert manager1.neighbor_list is not None
    assert manager1.search_cache is not None
    np.testing.assert_array_equal(np.asarray(contact0.kinematics.master_facet_ids), np.array([0, 0]))
    np.testing.assert_array_equal(np.asarray(contact1.kinematics.master_facet_ids), np.array([0, 0]))
    np.testing.assert_array_equal(np.asarray(contact2.kinematics.master_facet_ids), np.array([1, 1]))
    assert manager3.search_cache is not None

def test_contact_search_inputs_validate_shape_and_ids():
    cache = ff.ContactSearchCache(master_facet_ids=jnp.array([0, 1]))

    class SlaveSurface:
        coords = np.array([[0.0, -0.1], [0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2]])

    with pytest.raises(ValueError, match="shape"):
        ff.node_surface_contact_kinematics_from_surfaces(
            SlaveSurface(),
            MasterSurface(),
            dim=2,
            normal=jnp.array([0.0, 1.0]),
            search_cache=cache,
        )

    with pytest.raises(ValueError, match="invalid"):
        ff.node_surface_contact_kinematics_from_surfaces(
            SlaveSurface(),
            MasterSurface(),
            dim=2,
            normal=jnp.array([0.0, 1.0]),
            search_cache=ff.ContactSearchCache(master_facet_ids=jnp.array([1])),
        )

    with pytest.raises(ValueError, match="candidate_facet_ids"):
        ff.node_surface_contact_kinematics_from_surfaces(
            SlaveSurface(),
            MasterSurface(),
            dim=2,
            normal=jnp.array([0.0, 1.0]),
            candidate_facet_ids=jnp.array([1]),
        )

    with pytest.raises(ValueError, match="no contact candidate"):
        ff.contact_candidate_set_from_bounding_boxes(
            SlaveSurface(),
            MasterSurface(),
            dim=2,
            search_radius=0.01,
        )

    with pytest.raises(ValueError, match="slave node"):
        ff.node_surface_candidate_set_from_bounding_boxes(
            SlaveSurface(),
            MasterSurface(),
            dim=2,
            search_radius=0.01,
        )

    with pytest.raises(ValueError, match="contact_offsets"):
        ff.node_surface_contact_kinematics_from_surfaces(
            SlaveSurface(),
            MasterSurface(),
            dim=2,
            normal=jnp.array([0.0, 1.0]),
            candidate_facet_ids=ff.contact_candidate_set_from_per_contact([jnp.array([0]), jnp.array([0])]),
        )

    with pytest.raises(ValueError, match="skin"):
        ff.node_surface_neighbor_list_from_bounding_boxes(
            SlaveSurface(),
            MasterSurface(),
            dim=2,
            search_radius=0.1,
            skin=-0.1,
        )

    with pytest.raises(ValueError, match="cell_size"):
        ff.contact_aabb_index_from_surface(
            MasterSurface(),
            dim=2,
            cell_size=0.0,
        )

    index = ff.contact_aabb_index_from_surface(MasterSurface(), dim=2, cell_size=1.0)
    with pytest.raises(ValueError, match="quadrature point"):
        ff.surface_quadrature_candidate_set_from_aabb_index(
            SlaveSurface(),
            index,
            dim=2,
            search_radius=0.01,
            quadrature_rule="vertices",
        )

def test_contact_update_snapshot_carries_history_without_changed_comparison():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]]),
            gaps0=jnp.array([-0.1]),
            n_dofs=1,
        ),
        penalty=10.0,
    )
    u = jnp.array([0.0])
    snapshot0 = ff.ContactUpdateSnapshot.from_contact(
        contact,
        u,
        history={"stick": jnp.array([True]), "slip": jnp.array([0.0])},
    )
    snapshot1 = snapshot0.with_history({"stick": jnp.array([False]), "slip": jnp.array([0.2])})

    assert snapshot0.history["stick"].shape == (1,)
    assert snapshot1.history["slip"].shape == (1,)
    assert not bool(snapshot1.changed(snapshot0))
    np.testing.assert_allclose(np.asarray(snapshot1.residual()(u)), np.array([-1.0]), atol=1e-6)

def test_tangential_penalty_friction_manager_snapshots_and_advances_history():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            dofs=jnp.array([[0, 1]]),
            normals=jnp.array([[0.0, 1.0]]),
            gaps0=jnp.array([-0.1]),
            n_dofs=2,
        ),
        penalty=10.0,
    )
    u0 = jnp.array([0.0, 0.0])
    u1 = jnp.array([0.2, 0.0])
    manager0 = ff.TangentialPenaltyFrictionManager(
        mu=0.5,
        tangential_penalty=4.0,
        previous_displacement=u0,
    )
    snapshot0 = manager0.snapshot(contact, u0)
    manager1 = manager0.advance(contact, u1)
    snapshot1 = manager1.snapshot(contact, u1)

    assert snapshot0.history is None
    assert snapshot1.history is not None
    np.testing.assert_allclose(np.asarray(snapshot0.residual()(u0)), np.array([0.0, -1.0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(snapshot1.history.friction_force), np.array([[-0.5, 0.0]]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(snapshot1.residual()(u1)), np.array([-0.5, -1.0]), atol=1e-6)
    assert int(ff.stick_count(snapshot1.history)) == 0

def test_tangential_penalty_friction_manager_snapshot_and_advance():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            dofs=jnp.array([[0, 1]]),
            normals=jnp.array([[0.0, 1.0]]),
            gaps0=jnp.array([-0.1]),
            n_dofs=2,
        ),
        penalty=10.0,
    )
    manager0 = ff.TangentialPenaltyFrictionManager(
        mu=0.5,
        tangential_penalty=4.0,
        previous_displacement=jnp.zeros(2),
    )
    snapshot, manager1 = manager0.snapshot_and_advance(contact, jnp.array([0.2, 0.0]))

    assert snapshot.history is manager1.history
    assert manager1.history is not None
    with pytest.raises(ValueError, match="mu"):
        ff.TangentialPenaltyFrictionManager(
            mu=-0.1,
            tangential_penalty=4.0,
            previous_displacement=jnp.zeros(2),
        )

def test_node_surface_search_and_friction_managers_compose_workflow():
    class SlaveSurface:
        coords = np.array([[0.25, -0.05], [0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[1, 2]])

    search_manager = ff.make_node_surface_contact_search_manager(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        n_total_nodes=3,
        search_radius=0.2,
        skin=0.1,
        penalty=10.0,
        normal=jnp.array([0.0, 1.0]),
        cell_size=1.0,
    )
    u0 = jnp.zeros(6)
    contact0, search_manager = search_manager.build_contact(u0)
    friction_manager = ff.TangentialPenaltyFrictionManager(
        mu=0.5,
        tangential_penalty=4.0,
        previous_displacement=u0,
    )
    friction_manager = friction_manager.advance(contact0, u0.at[0].set(0.2))
    contact1, search_manager = search_manager.build_contact(u0.at[0].set(0.2))
    snapshot = friction_manager.snapshot(contact1, u0.at[0].set(0.2))

    assert search_manager.search_cache is not None
    assert snapshot.history is not None
    np.testing.assert_allclose(
        np.asarray(snapshot.residual()(u0.at[0].set(0.2))),
        np.array([-0.25, -0.5, 0.1375, 0.275, 0.1125, 0.225]),
        atol=1e-6,
    )

def test_surface_quadrature_search_and_friction_managers_compose_workflow():
    class SlaveSurface:
        coords = np.array([[0.0, -0.05], [1.0, -0.05], [0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0, 1]])

    class MasterSurface:
        coords = SlaveSurface.coords
        conn = np.array([[2, 3]])

    search_manager = ff.make_surface_quadrature_contact_search_manager(
        SlaveSurface(),
        MasterSurface(),
        dim=2,
        n_total_nodes=4,
        search_radius=0.2,
        skin=0.1,
        penalty=10.0,
        normal=jnp.array([0.0, 1.0]),
        quadrature_rule="vertices",
        cell_size=1.0,
    )
    u0 = jnp.zeros(8)
    u1 = u0.at[0].set(0.2).at[2].set(0.2)
    contact0, search_manager = search_manager.build_contact(u0)
    friction_manager = ff.TangentialPenaltyFrictionManager(
        mu=0.5,
        tangential_penalty=4.0,
        previous_displacement=u0,
    )
    friction_manager = friction_manager.advance(contact0, u1)
    snapshot = friction_manager.snapshot(contact0, u1)

    assert search_manager.search_cache is not None
    assert snapshot.history is not None
    np.testing.assert_allclose(np.asarray(contact0.kinematics.quadrature_weights), np.array([0.5, 0.5]), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(snapshot.residual()(u1)),
        np.array([-0.125, -0.25, -0.125, -0.25, 0.125, 0.25, 0.125, 0.25]),
        atol=1e-6,
    )

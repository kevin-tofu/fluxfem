"""Autodiff regression tests for contact KKT assembly."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

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
            multiplier_space="p0",
            facet_conn_master=facets,
            backend="numpy",
            format="dense",
        )
    )
    K_b = np.asarray(contact.assemble_contact_kkt(rho=3.0, multiplier_space="p0", backend="numpy", format="dense"))
    assert np.allclose(K_a, K_b, atol=1e-12)

    K_flux = contact.assemble_contact_kkt(rho=3.0, multiplier_space="p0", backend="numpy")
    assert hasattr(K_flux, "to_bcoo")
    K_flux_dense = np.asarray(K_flux.to_dense())
    assert np.allclose(K_a, K_flux_dense, atol=1e-12)

    K_bcoo = contact.assemble_contact_kkt(rho=3.0, multiplier_space="p0", backend="jax", format="bcoo")
    K_bcoo_dense = np.asarray(K_bcoo.todense())
    assert np.allclose(K_a, K_bcoo_dense, atol=1e-12)


def test_contact_operators_mortar_matches_kkt_blocks():
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

    ops = ff.assemble_contact_operators(
        contact,
        method="mortar",
        rho=3.0,
        multiplier_space="p0",
        backend="numpy",
    )
    K_dense, B_a_ref, B_b_ref = contact.assemble_contact_kkt(
        rho=3.0,
        multiplier_space="p0",
        backend="numpy",
        format="dense",
        return_blocks=True,
    )

    assert np.allclose(np.asarray(ops.B_a), np.asarray(B_a_ref), atol=1e-12)
    assert np.allclose(np.asarray(ops.B_b), np.asarray(B_b_ref), atol=1e-12)
    assert np.allclose(np.asarray(ops.B), np.concatenate([np.asarray(B_a_ref), -np.asarray(B_b_ref)], axis=1), atol=1e-12)

    n_u = int(np.asarray(ops.B).shape[1])
    assert np.allclose(np.asarray(ops.Kuu), np.asarray(K_dense)[:n_u, :n_u], atol=1e-12)


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
            multiplier_space="p0",
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
            multiplier_space="p0",
            backend="jax",
            format="dense",
        )
        u = ff.solve_contact_kkt(K, rhs, backend="jax", diagonal_shift=1e-2)
        return 0.5 * jnp.dot(u, u)

    def objective_ref(rho):
        K = contact.assemble_contact_kkt(
            rho=rho,
            multiplier_space="p0",
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
    assert rel < 2e-4

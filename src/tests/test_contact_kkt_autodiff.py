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
    return ff.ContactMultiplierSpace.from_contact(contact, family="p0", side="master")


def _nodal_multiplier():
    return ff.ContactMultiplierSpace(family="nodal")


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

    ops = ff.assemble_contact_constraint_operators(
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
    mult = ff.ContactMultiplierSpace.from_contact(contact, family="p0", side="master")

    ops_obj = ff.assemble_contact_constraint_operators(
        contact,
        rho=3.0,
        multiplier=mult,
        backend="numpy",
    )
    ops_ref = ff.assemble_contact_constraint_operators(
        contact,
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        backend="numpy",
    )
    assert np.allclose(np.asarray(ops_obj.B), np.asarray(ops_ref.B), atol=1e-12)
    assert np.allclose(np.asarray(ops_obj.Kuu), np.asarray(ops_ref.Kuu), atol=1e-12)
    assert isinstance(ops_obj.multiplier, ff.ContactMultiplierSpace)

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

    ops_default = ff.assemble_contact_constraint_operators(
        contact,
        rho=3.0,
        multiplier=_p0_multiplier(contact),
        backend="numpy",
    )
    ops_formulation = ff.assemble_contact_constraint_operators(
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

    ops = ff.assemble_contact_penalty_operators(
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

    ops = ff.assemble_contact_penalty_operators(
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
    ops = ff.assemble_contact_constraint_operators(
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
    ops_base = ff.assemble_contact_constraint_operators(contact, rho=1.0, multiplier=_nodal_multiplier())
    ops_with_alias_inputs = ff.assemble_contact_constraint_operators(
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
        ff.assemble_contact_constraint_operators(_ContactStub(), multiplier=_nodal_multiplier(), weak_form=_dummy_res_form)


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
        ff.assemble_contact_constraint_operators(contact, multiplier=_nodal_multiplier(), formulation="penalty")


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
        ops = ff.assemble_contact_constraint_operators(
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
        ops = ff.assemble_contact_constraint_operators(
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
    assert rel < 2e-4


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
    assert rel < 2e-4


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

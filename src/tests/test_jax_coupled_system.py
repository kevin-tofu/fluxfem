from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import einsum as wf_einsum, param_ref, test_ref as wf_test_ref, unknown_ref
from fluxfem.mesh.contact import compile_tagged_pair_nitsche_penalty_residual

jax.config.update("jax_enable_x64", True)


def test_jax_coupled_system_remote_spring_compliance_grad():
    load = jnp.array([2.0, 0.0, 0.0])

    def compliance(k):
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.zeros((0, 0)), jnp.zeros((0,)))
        builder.append_remote_point(
            "remote",
            point=jnp.array([0.0, 0.0, 0.0]),
            include_rotation=False,
            F_block=load,
        )
        builder.add_remote_spring(
            "remote",
            translational_stiffness=jnp.array([k, k, k]),
            translational_target=jnp.zeros((3,)),
        )
        system = builder.build()
        u = system.solve()
        return system.compliance(load, u=u)

    k0 = jnp.array(5.0)
    comp = compliance(k0)
    grad = jax.grad(compliance)(k0)

    assert np.allclose(float(comp), 4.0 / 5.0, atol=1e-12)
    assert np.allclose(float(grad), -4.0 / 25.0, atol=1e-12)


def test_coupled_factory_selects_backend_specific_types():
    sys_np = ff.make_coupled_system(np.eye(1), np.zeros((1,), dtype=float), backend="numpy")
    sys_jax = ff.make_coupled_system(jnp.eye(1), jnp.zeros((1,)), backend="jax")
    bld_np = ff.make_coupled_system_builder(np.eye(1), np.zeros((1,), dtype=float), backend="numpy")
    bld_jax = ff.make_coupled_system_builder(jnp.eye(1), jnp.zeros((1,)), backend="jax")

    assert isinstance(sys_np, ff.NumpyCoupledSystem)
    assert isinstance(sys_jax, ff.CoupledSystem)
    assert isinstance(bld_np, ff.NumpyCoupledSystemBuilder)
    assert isinstance(bld_jax, ff.CoupledSystemBuilder)


def test_explicit_jax_coupled_factory_returns_jax_types():
    sys_jax = ff.make_jax_coupled_system(jnp.eye(1), jnp.zeros((1,)))
    bld_jax = ff.make_jax_coupled_system_builder(jnp.eye(1), jnp.zeros((1,)))

    assert isinstance(sys_jax, ff.CoupledSystem)
    assert isinstance(bld_jax, ff.CoupledSystemBuilder)


def test_explicit_numpy_coupled_factory_returns_numpy_types():
    sys_np = ff.make_numpy_coupled_system(np.eye(1), np.zeros((1,), dtype=float))
    bld_np = ff.make_numpy_coupled_system_builder(np.eye(1), np.zeros((1,), dtype=float))

    assert isinstance(sys_np, ff.NumpyCoupledSystem)
    assert isinstance(bld_np, ff.NumpyCoupledSystemBuilder)


def test_coupled_create_selects_backend_specific_types():
    sys_np = ff.CoupledSystem.create(np.eye(1), np.zeros((1,), dtype=float), backend="numpy")
    sys_jax = ff.CoupledSystem.create(jnp.eye(1), jnp.zeros((1,)), backend="jax")
    bld_np = ff.CoupledSystemBuilder.create(np.eye(1), np.zeros((1,), dtype=float), backend="numpy")
    bld_jax = ff.CoupledSystemBuilder.create(jnp.eye(1), jnp.zeros((1,)), backend="jax")

    assert isinstance(sys_np, ff.NumpyCoupledSystem)
    assert isinstance(sys_jax, ff.CoupledSystem)
    assert isinstance(bld_np, ff.NumpyCoupledSystemBuilder)
    assert isinstance(bld_jax, ff.CoupledSystemBuilder)


def test_jax_builder_add_constraint_spec_rbe2_matches_direct_call():
    x_ref = jnp.array([0.0, 0.0, 0.0])
    x_slave = jnp.array([[1.0, 0.0, 0.0]])

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(9), jnp.zeros((9,)))
    direct.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    direct.register_field("slave", n_dofs=3, value_dim=1, offset=6)
    direct.add_rbe2_constraint(master="remote", slave="slave", ref_point=x_ref, slave_coords=x_slave)

    via_spec = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(9), jnp.zeros((9,)))
    via_spec.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    via_spec.register_field("slave", n_dofs=3, value_dim=1, offset=6)
    via_spec.add_constraint(
        ff.ConstraintSpec(
            kind="rbe2",
            master="remote",
            slave="slave",
            ref_point=np.asarray(x_ref),
            slave_coords=np.asarray(x_slave),
        )
    )

    assert np.allclose(np.asarray(via_spec.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)


def test_jax_builder_add_constraint_spec_embedding_matches_direct_call():
    emb = ff.EmbeddingMap(
        rows=np.array([0], dtype=int),
        cols=np.array([1], dtype=int),
        data=np.array([0.5], dtype=float),
        shape=(1, 2),
        mode="nodal",
    )

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(6), jnp.zeros((6,)))
    direct.register_field("master", n_dofs=4, value_dim=2, n_nodes=2, offset=0)
    direct.register_field("slave", n_dofs=2, value_dim=2, n_nodes=1, offset=4)
    direct.add_embedding_constraint(emb, master="master", slave="slave", value_dim=2)

    via_spec = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(6), jnp.zeros((6,)))
    via_spec.register_field("master", n_dofs=4, value_dim=2, n_nodes=2, offset=0)
    via_spec.register_field("slave", n_dofs=2, value_dim=2, n_nodes=1, offset=4)
    via_spec.add_constraint(
        ff.ConstraintSpec(
            kind="embedding",
            master="master",
            slave="slave",
            embedding=emb,
            value_dim=2,
        )
    )

    assert np.allclose(np.asarray(via_spec.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)


def test_jax_builder_add_constraint_spec_contact_matches_direct_call():
    class _Ops:
        enforcement = "nitsche"
        jacobian = jnp.array([[1.0, -1.0], [-1.0, 1.0]], dtype=jnp.float64)
        residual = jnp.array([2.0, -2.0], dtype=jnp.float64)

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(2), jnp.zeros((2,)))
    direct.register_field("a", n_dofs=1, value_dim=1, offset=0)
    direct.register_field("b", n_dofs=1, value_dim=1, offset=1)
    direct.add_contact_nitsche(_Ops(), master="a", slave="b")

    via_spec = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(2), jnp.zeros((2,)))
    via_spec.register_field("a", n_dofs=1, value_dim=1, offset=0)
    via_spec.register_field("b", n_dofs=1, value_dim=1, offset=1)
    via_spec.add_constraint(
        ff.ConstraintSpec(
            kind="contact",
            master="a",
            slave="b",
            contact_obj=_Ops(),
            enforcement="nitsche",
        )
    )

    assert np.allclose(np.asarray(via_spec.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)
    assert np.allclose(np.asarray(via_spec.build().F_u), np.asarray(direct.build().F_u), atol=1e-12)


def test_jax_builder_add_constraint_spec_rejects_unsupported_contact_modes():
    builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(2), jnp.zeros((2,)))
    builder.register_field("a", n_dofs=1, value_dim=1, offset=0)
    builder.register_field("b", n_dofs=1, value_dim=1, offset=1)
    with pytest.raises(NotImplementedError, match="accepts explicit mortar operators"):
        builder.add_constraint(
            ff.ConstraintSpec(
                kind="contact",
                master="a",
                slave="b",
                contact_obj=object(),
                enforcement="mortar",
            )
        )


def test_jax_builder_resolve_dirichlet_specs_mixed_inputs():
    builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8), jnp.zeros((8,)))
    builder.register_field("a", n_dofs=6, value_dim=2, n_nodes=3, offset=0)
    builder.register_field("b", n_dofs=2, value_dim=1, n_nodes=2, offset=6)

    dofs, vals = builder.resolve_dirichlet(
        [
            ff.DirichletSpec(field="a", nodes=[0, 2], components=[0], value=0.0),
            ff.DirichletSpec(field="a", local_dofs=[3], value=1.25),
            ff.DirichletSpec(field="b", local_dofs=[1], value=-2.0),
        ]
    )
    assert np.array_equal(dofs, np.array([0, 3, 4, 7], dtype=int))
    assert np.allclose(vals, np.array([0.0, 1.25, 0.0, -2.0], dtype=float), atol=1e-12)


def test_jax_builder_solve_with_dirichlet_specs():
    K = jnp.eye(4)
    F = jnp.array([2.0, 0.0, 0.0, -3.0])
    builder = ff.JAXCoupledSystemBuilder.from_structural(K, F)
    builder.register_field("master", n_dofs=2, value_dim=1, offset=0)
    builder.register_field("slave", n_dofs=2, value_dim=1, offset=2)

    u = np.asarray(
        builder.solve(
            dirichlet_specs=[
                ff.DirichletSpec(field="master", nodes=[0], value=0.5),
                ff.DirichletSpec(field="slave", local_dofs=[1], value=-1.0),
            ]
        ),
        dtype=float,
    )
    assert np.allclose(u[[0, 3]], np.array([0.5, -1.0]), atol=1e-12)


def test_jax_coupled_system_remote_spring_solve_matches_numpy_builder():
    k = 7.0
    load = np.array([3.0, 0.0, 0.0], dtype=float)

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.zeros((0, 0)), jnp.zeros((0,)))
    jax_builder.append_remote_point(
        "remote",
        point=jnp.array([0.0, 0.0, 0.0]),
        include_rotation=False,
        F_block=jnp.asarray(load),
    )
    jax_builder.add_remote_spring(
        "remote",
        translational_stiffness=jnp.array([k, k, k]),
        translational_target=jnp.zeros((3,)),
    )
    u_jax = np.asarray(jax_builder.build().solve(), dtype=float)

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float))
    np_builder.append_remote_point(
        "remote",
        point=np.array([0.0, 0.0, 0.0], dtype=float),
        include_rotation=False,
        F_block=load,
    )
    np_builder.add_remote_spring(
        "remote",
        translational_stiffness=np.array([k, k, k], dtype=float),
        translational_target=np.zeros((3,), dtype=float),
    )
    u_np = np.asarray(np_builder.build().solve(format="csr"), dtype=float)

    assert np.allclose(u_jax, u_np, atol=1e-12)


def test_jax_coupled_system_cg_matches_dense():
    K = jnp.diag(jnp.array([4.0, 5.0, 6.0]))
    F = jnp.array([1.0, -2.0, 3.0])
    system = ff.JAXCoupledSystem.from_structural(K, F)

    u_dense = np.asarray(np.linalg.solve(np.asarray(system.to_dense()), np.asarray(F)), dtype=float)
    u_cg = np.asarray(system.solve(solver="cg", tol=1e-10, maxiter=50), dtype=float)

    assert np.allclose(u_cg, u_dense, atol=1e-10)


def test_jax_coupled_system_sparse_dirichlet_cg_matches_dense_and_grad():
    K = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    F = jnp.array([1.0, 2.0])
    dir_dofs = np.array([0], dtype=int)

    def objective(v):
        system = ff.JAXCoupledSystem.from_structural(K, F)
        u = system.solve(
            dirichlet_dofs=dir_dofs,
            dirichlet_vals=jnp.array([v]),
            tol=1e-12,
            maxiter=50,
        )
        return jnp.dot(u, u)

    system = ff.JAXCoupledSystem.from_structural(K, F)
    K_ref, F_ref = ff.enforce_dirichlet_dense(
        np.asarray(system.to_dense(), dtype=float),
        np.asarray(F, dtype=float),
        dir_dofs,
        np.array([0.25], dtype=float),
    )
    u_dense = np.asarray(np.linalg.solve(K_ref, F_ref), dtype=float)
    u_cg = np.asarray(
        system.solve(
            dirichlet_dofs=dir_dofs,
            dirichlet_vals=jnp.array([0.25]),
            tol=1e-12,
            maxiter=50,
        ),
        dtype=float,
    )
    grad = float(jax.grad(objective)(jnp.array(0.25)))
    eps = 1.0e-6
    fd = float((objective(0.25 + eps) - objective(0.25 - eps)) / (2.0 * eps))

    assert np.allclose(u_cg, u_dense, atol=1e-10)
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_jax_coupled_system_rejects_dense_solver():
    system = ff.JAXCoupledSystem.from_structural(jnp.eye(2), jnp.ones((2,)))

    with pytest.raises(ValueError, match="only supports solver='cg'"):
        system.solve(solver="dense")


def test_jax_coupled_system_constraint_matrix_matches_numpy_builder():
    K_u = np.eye(3, dtype=float)
    F_u = np.array([0.0, 1.0, 2.0], dtype=float)
    C = np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    np_builder.register_field("master", n_dofs=1, value_dim=1)
    np_builder.register_field("slave", n_dofs=2, value_dim=1)
    np_builder.add_constraint_matrix(C, master="master", slave="slave", value_dim=1, rho=0.0)
    u_np = np.asarray(np_builder.build().solve(format="csr", diagonal_shift=1e-8), dtype=float)

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.asarray(K_u), jnp.asarray(F_u))
    jax_builder.register_field("master", n_dofs=1, value_dim=1)
    jax_builder.register_field("slave", n_dofs=2, value_dim=1)
    jax_builder.add_constraint_matrix(C, master="master", slave="slave", value_dim=1, rho=0.0)
    u_jax = np.asarray(jax_builder.build().solve(diagonal_shift=1e-8), dtype=float)

    assert np.allclose(u_jax, u_np, atol=1e-8)


def test_jax_coupled_system_embedding_constraint_matches_numpy_builder():
    emb = ff.EmbeddingMap(
        rows=np.array([0], dtype=int),
        cols=np.array([1], dtype=int),
        data=np.array([0.5], dtype=float),
        shape=(1, 2),
        mode="nodal",
    )

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(np.eye(6), np.zeros((6,), dtype=float))
    np_builder.register_field("master", n_dofs=4, value_dim=2, n_nodes=2, offset=0)
    np_builder.register_field("slave", n_dofs=2, value_dim=2, n_nodes=1, offset=4)
    np_builder.add_embedding_constraint(emb, master="master", slave="slave", value_dim=2, backend="jax")
    K_np, F_np = np_builder.build().assemble(format="dense")

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(6), jnp.zeros((6,)))
    jax_builder.register_field("master", n_dofs=4, value_dim=2, n_nodes=2, offset=0)
    jax_builder.register_field("slave", n_dofs=2, value_dim=2, n_nodes=1, offset=4)
    jax_builder.add_embedding_constraint(emb, master="master", slave="slave", value_dim=2)
    system_jax = jax_builder.build()

    assert np.allclose(np.asarray(system_jax.K_u.to_dense()), np.asarray(K_np), atol=1e-12)
    assert np.allclose(np.asarray(system_jax.F_u), np.asarray(F_np), atol=1e-12)


def test_jax_coupled_system_contact_nitsche_matches_numpy_builder():
    K_u = np.diag([10.0, 20.0, 30.0, 40.0]).astype(float)
    F_u = np.zeros((4,), dtype=float)

    class _Ops:
        enforcement = "nitsche"
        jacobian = np.array(
            [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, 2.0, 0.0, -2.0],
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0, 2.0],
            ],
            dtype=float,
        )
        residual = np.array([1.0, 2.0, -1.0, -2.0], dtype=float)

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    np_builder.register_field("master", n_dofs=2, value_dim=1, offset=0)
    np_builder.register_field("slave", n_dofs=2, value_dim=1, offset=2)
    np_builder.add_contact_nitsche(_Ops(), master="master", slave="slave")
    K_np, F_np = np_builder.build().assemble(format="dense")

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.asarray(K_u), jnp.asarray(F_u))
    jax_builder.register_field("master", n_dofs=2, value_dim=1, offset=0)
    jax_builder.register_field("slave", n_dofs=2, value_dim=1, offset=2)
    jax_builder.add_contact_nitsche(_Ops(), master="master", slave="slave")
    system_jax = jax_builder.build()

    assert np.allclose(np.asarray(system_jax.K_u.to_dense()), np.asarray(K_np), atol=1e-12)
    assert np.allclose(np.asarray(system_jax.F_u), np.asarray(F_np), atol=1e-12)


def test_jax_builder_add_contact_accepts_raw_contact_penalty_form():
    class _RawContact:
        def assemble_contact_constraint_operators(self, **kwargs):
            _ = kwargs
            raise RuntimeError("not used in this test")

        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, u, params, normal_source)
            return jnp.array([1.0, -1.0], dtype=jnp.float64)

        def assemble_jacobian(
            self,
            res_form,
            u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="jax",
            batch_jac=None,
        ):
            _ = (res_form, u, params, normal_source, sparse, backend, batch_jac)
            return jnp.array([[2.0, -2.0], [-2.0, 2.0]], dtype=jnp.float64)

    def _wf(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": jnp.array([0.0]), "b": jnp.array([0.0])}

    builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.diag(jnp.array([10.0, 30.0])), jnp.zeros((2,)))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)
    builder.add_contact(
        _RawContact(),
        master="a",
        slave="b",
        weak_form=_wf,
        state={"a": jnp.array([0.0]), "b": jnp.array([0.0])},
        params=object(),
        value_dim=1,
    )
    system = builder.build()
    assert np.allclose(np.asarray(system.K_u.to_dense()), np.array([[12.0, -2.0], [-2.0, 32.0]]), atol=1e-12)
    assert np.allclose(np.asarray(system.F_u), np.array([-1.0, 1.0]), atol=1e-12)


def test_jax_builder_add_contact_raw_penalty_path_is_differentiable():
    load = jnp.array([1.0, 0.0], dtype=jnp.float64)

    class _RawContact:
        def assemble_contact_constraint_operators(self, **kwargs):
            _ = kwargs
            raise RuntimeError("not used in this test")

        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, u, params, normal_source)
            return jnp.zeros((2,), dtype=jnp.float64)

        def assemble_jacobian(
            self,
            res_form,
            u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="jax",
            batch_jac=None,
        ):
            _ = (res_form, u, normal_source, sparse, backend, batch_jac)
            k = params.k
            return jnp.array([[k, -k], [-k, k]], dtype=jnp.float64)

    def _wf(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": jnp.array([0.0]), "b": jnp.array([0.0])}

    def compliance(k):
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(2), load)
        builder.register_field("a", n_dofs=1, value_dim=1)
        builder.register_field("b", n_dofs=1, value_dim=1)
        builder.add_contact(
            _RawContact(),
            master="a",
            slave="b",
            weak_form=_wf,
            state={"a": jnp.array([0.0]), "b": jnp.array([0.0])},
            params=ff.Params(k=k),
            value_dim=1,
        )
        u = builder.build().solve()
        return jnp.dot(load, u)

    k0 = jnp.array(3.0, dtype=jnp.float64)
    grad = jax.grad(compliance)(k0)
    eps = 1e-5
    fd = (float(compliance(k0 + eps)) - float(compliance(k0 - eps))) / (2.0 * eps)
    assert np.allclose(float(grad), fd, rtol=1e-4, atol=1e-5)


def test_jax_builder_add_contact_nitsche_with_actual_contact_space_is_differentiable():
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
        backend="jax",
    )

    v1 = wf_test_ref("a")
    v2 = wf_test_ref("b")
    u1 = unknown_ref("a")
    u2 = unknown_ref("b")
    p = param_ref()
    n = h_wf.normal()
    jump = u1.val - u2.val
    t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
    t_v1 = h_wf.traction(v1, n, p)
    t_v2 = h_wf.traction(v2, n, p)
    expr_a = ((p.alpha * p.inv_h) * h_wf.dot(v1, jump) - h_wf.dot(v1, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v1, jump)) * h_wf.ds()
    expr_b = (-(p.alpha * p.inv_h) * h_wf.dot(v2, jump) + h_wf.dot(v2, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v2, jump)) * h_wf.ds()
    res_form = compile_tagged_pair_nitsche_penalty_residual({"a": expr_a, "b": expr_b}, backend="jax")

    lam, mu = ff.lame_parameters(210e9, 0.3)
    params_base = ff.Params(alpha=20.0 * (10000.0 * mu + lam), inv_h=1.0, lam=lam, mu=mu)
    state0 = {"a": jnp.zeros((12,), dtype=jnp.float64), "b": jnp.zeros((12,), dtype=jnp.float64)}

    def objective(alpha_scale):
        params = ff.Params(
            alpha=params_base.alpha * alpha_scale,
            inv_h=params_base.inv_h,
            lam=params_base.lam,
            mu=params_base.mu,
        )
        jac = contact.assemble_jacobian(
            res_form,
            state0,
            params,
            sparse=False,
            backend="jax",
        )
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(24, dtype=jnp.float64), jnp.zeros((24,), dtype=jnp.float64))
        builder.register_field("a", n_dofs=12, value_dim=3, n_nodes=4, offset=0)
        builder.register_field("b", n_dofs=12, value_dim=3, n_nodes=4, offset=12)
        builder.add_contact_nitsche(jac, master="a", slave="b", value_dim=3)
        system = builder.build()
        K = system.K_u.to_dense()
        return 0.5 * jnp.sum(K * K)

    s0 = jnp.array(1.0, dtype=jnp.float64)
    grad = jax.grad(objective)(s0)
    eps = 1e-5
    fd = (float(objective(s0 + eps)) - float(objective(s0 - eps))) / (2.0 * eps)
    assert np.allclose(float(grad), fd, rtol=5e-4, atol=1e-5)


def test_jax_builder_add_contact_raw_actual_contact_space_is_differentiable():
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
        backend="jax",
    )

    v1 = wf_test_ref("a")
    v2 = wf_test_ref("b")
    u1 = unknown_ref("a")
    u2 = unknown_ref("b")
    p = param_ref()
    n = h_wf.normal()
    jump = u1.val - u2.val
    t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
    t_v1 = h_wf.traction(v1, n, p)
    t_v2 = h_wf.traction(v2, n, p)
    expr_a = ((p.alpha * p.inv_h) * h_wf.dot(v1, jump) - h_wf.dot(v1, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v1, jump)) * h_wf.ds()
    expr_b = (-(p.alpha * p.inv_h) * h_wf.dot(v2, jump) + h_wf.dot(v2, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v2, jump)) * h_wf.ds()
    res_form = compile_tagged_pair_nitsche_penalty_residual({"a": expr_a, "b": expr_b}, backend="jax")

    lam, mu = ff.lame_parameters(210e9, 0.3)
    params_base = ff.Params(alpha=20.0 * (10000.0 * mu + lam), inv_h=1.0, lam=lam, mu=mu)
    state0 = {
        "a": jnp.linspace(0.0, 0.11, 12, dtype=jnp.float64),
        "b": jnp.linspace(0.02, 0.13, 12, dtype=jnp.float64),
    }

    def objective(alpha_scale):
        params = ff.Params(
            alpha=params_base.alpha * alpha_scale,
            inv_h=params_base.inv_h,
            lam=params_base.lam,
            mu=params_base.mu,
        )
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(24, dtype=jnp.float64), jnp.zeros((24,), dtype=jnp.float64))
        builder.register_field("a", n_dofs=12, value_dim=3, n_nodes=4, offset=0)
        builder.register_field("b", n_dofs=12, value_dim=3, n_nodes=4, offset=12)
        builder.add_contact(
            contact,
            master="a",
            slave="b",
            weak_form=res_form,
            state=state0,
            params=params,
            value_dim=3,
        )
        system = builder.build()
        K = system.K_u.to_dense()
        return 0.5 * (jnp.sum(K * K) + jnp.sum(system.F_u * system.F_u))

    s0 = jnp.array(1.0, dtype=jnp.float64)
    grad = jax.grad(objective)(s0)
    eps = 1e-5
    fd = (float(objective(s0 + eps)) - float(objective(s0 - eps))) / (2.0 * eps)
    assert np.allclose(float(grad), fd, rtol=5e-4, atol=1e-5)


def test_actual_contact_space_sparse_jax_jacobian_grad_matches_dense():
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
        backend="jax",
    )

    v1 = wf_test_ref("a")
    v2 = wf_test_ref("b")
    u1 = unknown_ref("a")
    u2 = unknown_ref("b")
    p = param_ref()
    n = h_wf.normal()
    jump = u1.val - u2.val
    t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
    t_v1 = h_wf.traction(v1, n, p)
    t_v2 = h_wf.traction(v2, n, p)
    expr_a = ((p.alpha * p.inv_h) * h_wf.dot(v1, jump) - h_wf.dot(v1, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v1, jump)) * h_wf.ds()
    expr_b = (-(p.alpha * p.inv_h) * h_wf.dot(v2, jump) + h_wf.dot(v2, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v2, jump)) * h_wf.ds()
    res_form = compile_tagged_pair_nitsche_penalty_residual({"a": expr_a, "b": expr_b}, backend="jax")

    lam, mu = ff.lame_parameters(210e9, 0.3)
    params_base = ff.Params(alpha=20.0 * (10000.0 * mu + lam), inv_h=1.0, lam=lam, mu=mu)
    state0 = {
        "a": jnp.linspace(0.0, 0.11, 12, dtype=jnp.float64),
        "b": jnp.linspace(0.02, 0.13, 12, dtype=jnp.float64),
    }

    def objective_dense(alpha_scale):
        params = ff.Params(alpha=params_base.alpha * alpha_scale, inv_h=params_base.inv_h, lam=params_base.lam, mu=params_base.mu)
        jac = contact.assemble_jacobian(res_form, state0, params, sparse=False, backend="jax")
        return 0.5 * jnp.sum(jac * jac)

    def objective_sparse(alpha_scale):
        params = ff.Params(alpha=params_base.alpha * alpha_scale, inv_h=params_base.inv_h, lam=params_base.lam, mu=params_base.mu)
        jac = contact.assemble_jacobian(res_form, state0, params, sparse=True, backend="jax")
        dense = jac.to_dense()
        return 0.5 * jnp.sum(dense * dense)

    s0 = jnp.array(1.0, dtype=jnp.float64)
    g_dense = float(jax.grad(objective_dense)(s0))
    g_sparse = float(jax.grad(objective_sparse)(s0))
    assert np.allclose(g_sparse, g_dense, rtol=5e-4, atol=1e-5)


def test_jax_builder_add_contact_mortar_with_actual_contact_space_matches_explicit_ops():
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
        backend="jax",
    )
    mult = ff.MultiplierSpec(family="nodal")
    ops = ff.assemble_multiplier(contact, rho=2.0, multiplier=mult, backend="jax")

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
    direct.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    direct.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    direct.add_contact_mortar(ops, master="a", slave="b", value_dim=1)

    raw = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
    raw.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    raw.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    raw.add_contact(
        contact,
        master="a",
        slave="b",
        family="constraint",
        multiplier=mult,
        rho=2.0,
        value_dim=1,
    )

    assert np.allclose(np.asarray(raw.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)
    assert np.allclose(np.asarray(raw.build().F_u), np.asarray(direct.build().F_u), atol=1e-12)


def test_jax_builder_contact_mortar_sugar_matches_explicit_multiplier_choice():
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
        backend="jax",
    )
    mult = ff.MultiplierSpec.nodal_mortar()
    ops = ff.assemble_multiplier(contact, rho=2.0, multiplier=mult, backend="jax")

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
    direct.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    direct.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    direct.add_contact_mortar(ops, master="a", slave="b", value_dim=1)

    raw = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
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

    assert np.allclose(np.asarray(raw.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)


def test_jax_builder_contact_coarse_dual_mortar_sugar_reduces_lambda_rows():
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
        backend="jax",
    )

    raw = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
    raw.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    raw.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    raw.add_contact(
        contact,
        master="a",
        slave="b",
        family="constraint",
        mortar="coarse_dual",
        mortar_max_rank=2,
        rho=2.0,
        value_dim=1,
    )

    system = raw.build()
    assert system.K_u.shape[0] < 16
    assert system.K_u.shape[0] > 8


def test_jax_builder_contact_mortar_constraint_spec_matches_direct_call():
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
        backend="jax",
    )
    mult = ff.MultiplierSpec(family="nodal")
    ops = ff.assemble_multiplier(contact, rho=1.5, multiplier=mult, backend="jax")

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
    direct.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    direct.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    direct.add_contact_mortar(ops, master="a", slave="b", value_dim=1)

    via_spec = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
    via_spec.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    via_spec.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    via_spec.add_constraint(
        ff.ConstraintSpec(
            kind="contact",
            master="a",
            slave="b",
            contact_obj=ops,
            enforcement="mortar",
            value_dim=1,
        )
    )

    assert np.allclose(np.asarray(via_spec.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)


def test_jax_contact_mortar_rho_grad_matches_fd():
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
        backend="jax",
    )
    mult = ff.MultiplierSpec(family="nodal")

    def objective(rho):
        ops = ff.assemble_multiplier(contact, rho=rho, multiplier=mult, backend="jax")
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(8, dtype=jnp.float64), jnp.zeros((8,), dtype=jnp.float64))
        builder.register_field("a", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
        builder.register_field("b", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
        builder.add_contact_mortar(ops, master="a", slave="b", value_dim=1)
        system = builder.build()
        K = system.K_u.to_dense()
        return 0.5 * jnp.sum(K * K)

    rho0 = jnp.array(2.0, dtype=jnp.float64)
    grad = jax.grad(objective)(rho0)
    eps = 1e-5
    fd = (float(objective(rho0 + eps)) - float(objective(rho0 - eps))) / (2.0 * eps)
    assert np.allclose(float(grad), fd, rtol=5e-4, atol=1e-5)


@pytest.mark.parametrize("family", ["p0_active", "p0_supermesh"])
def test_jax_contact_mortar_special_multiplier_families_match_raw_path(family):
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
        backend="jax",
    )
    mult = ff.MultiplierSpec.from_contact(contact, family=family, side="master", value_dim=3)
    ops = ff.assemble_multiplier(contact, rho=0.5, multiplier=mult, backend="jax")

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(24, dtype=jnp.float64), jnp.zeros((24,), dtype=jnp.float64))
    direct.register_field("a", n_dofs=12, value_dim=3, n_nodes=4, offset=0)
    direct.register_field("b", n_dofs=12, value_dim=3, n_nodes=4, offset=12)
    direct.add_contact_mortar(ops, master="a", slave="b", value_dim=3)

    raw = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(24, dtype=jnp.float64), jnp.zeros((24,), dtype=jnp.float64))
    raw.register_field("a", n_dofs=12, value_dim=3, n_nodes=4, offset=0)
    raw.register_field("b", n_dofs=12, value_dim=3, n_nodes=4, offset=12)
    raw.add_contact(
        contact,
        master="a",
        slave="b",
        family="constraint",
        multiplier=mult,
        rho=0.5,
        value_dim=3,
    )

    assert np.allclose(np.asarray(raw.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)
    assert np.allclose(np.asarray(raw.build().F_u), np.asarray(direct.build().F_u), atol=1e-12)


@pytest.mark.parametrize("family", ["p0_active", "p0_supermesh"])
def test_jax_contact_mortar_special_multiplier_families_constraint_spec_matches_direct(family):
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
        backend="jax",
    )
    mult = ff.MultiplierSpec.from_contact(contact, family=family, side="master", value_dim=3)
    ops = ff.assemble_multiplier(contact, rho=0.25, multiplier=mult, backend="jax")

    direct = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(24, dtype=jnp.float64), jnp.zeros((24,), dtype=jnp.float64))
    direct.register_field("a", n_dofs=12, value_dim=3, n_nodes=4, offset=0)
    direct.register_field("b", n_dofs=12, value_dim=3, n_nodes=4, offset=12)
    direct.add_contact_mortar(ops, master="a", slave="b", value_dim=3)

    via_spec = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(24, dtype=jnp.float64), jnp.zeros((24,), dtype=jnp.float64))
    via_spec.register_field("a", n_dofs=12, value_dim=3, n_nodes=4, offset=0)
    via_spec.register_field("b", n_dofs=12, value_dim=3, n_nodes=4, offset=12)
    via_spec.add_constraint(
        ff.ConstraintSpec(
            kind="contact",
            master="a",
            slave="b",
            contact_obj=ops,
            enforcement="mortar",
            value_dim=3,
        )
    )

    assert np.allclose(np.asarray(via_spec.build().K_u.to_dense()), np.asarray(direct.build().K_u.to_dense()), atol=1e-12)


def test_jax_coupled_system_remote_spring_with_constraint_is_differentiable():
    load = jnp.array([1.5, 0.0, 0.0, 0.0])
    C = jnp.array([[1.0, -1.0, 0.0, 0.0]], dtype=load.dtype)

    def compliance(k):
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(2), load[:2])
        builder.register_field("slave", n_dofs=2, value_dim=1, offset=0)
        builder.append_field("remote", n_dofs=2, value_dim=1, F_block=load[2:])
        builder.add_dof_spring("remote", local_dofs=[0], stiffness=k, reference_value=0.0)
        builder.add_constraint_matrix_dof(C, master="slave", slave="remote", rho=0.0)
        system = builder.build()
        u = system.solve(diagonal_shift=1e-8)
        load_full = jnp.concatenate([load[:2], load[2:], jnp.zeros((1,), dtype=load.dtype)])
        return jnp.dot(load_full, u)

    k0 = jnp.array(4.0)
    grad = jax.grad(compliance)(k0)
    eps = 1e-5
    fd = (float(compliance(k0 + eps)) - float(compliance(k0 - eps))) / (2.0 * eps)
    assert np.allclose(float(grad), fd, rtol=1e-4, atol=1e-5)


def test_jax_coupled_system_rbe2_matches_numpy_builder():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    K_u = np.eye(12, dtype=float)
    F_u = np.zeros((12,), dtype=float)

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    np_builder.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    np_builder.register_field("slave", n_dofs=6, value_dim=1, offset=6)
    np_builder.add_rbe2_constraint(
        master="remote",
        slave="slave",
        ref_point=x_ref,
        slave_coords=x_slave,
        rho=0.0,
        backend="numpy",
    )
    K_np, F_np = np_builder.build().assemble(format="dense")

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.asarray(K_u), jnp.asarray(F_u))
    jax_builder.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    jax_builder.register_field("slave", n_dofs=6, value_dim=1, offset=6)
    jax_builder.add_rbe2_constraint(
        master="remote",
        slave="slave",
        ref_point=jnp.asarray(x_ref),
        slave_coords=jnp.asarray(x_slave),
        rho=0.0,
    )
    system_jax = jax_builder.build()

    assert np.allclose(np.asarray(system_jax.K_u.to_dense()), np.asarray(K_np), atol=1e-12)
    assert np.allclose(np.asarray(system_jax.F_u), np.asarray(F_np), atol=1e-12)


def test_jax_coupled_system_remote_spring_with_rbe2_is_differentiable():
    x_ref = jnp.array([0.0, 0.0, 0.0])
    x_slave = jnp.array([[1.0, 0.0, 0.0]])
    load = jnp.array([1.0, 0.0, 0.0])

    def compliance(k):
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(3), load)
        builder.register_field("slave", n_dofs=3, value_dim=1, offset=0)
        builder.append_remote_point("remote", point=x_ref)
        builder.add_remote_spring(
            "remote",
            translational_stiffness=jnp.array([k, k, k]),
            rotational_stiffness=jnp.array([5.0, 5.0, 5.0]),
            translational_target=jnp.array([1.0, 0.0, 0.0]),
        )
        builder.add_rbe2_constraint(
            master="remote",
            slave="slave",
            ref_point=x_ref,
            slave_coords=x_slave,
            rho=0.0,
        )
        system = builder.build()
        u = system.solve(diagonal_shift=1e-8)
        load_full = jnp.concatenate([load, jnp.zeros((system.n_u - load.shape[0],), dtype=load.dtype)])
        return jnp.dot(load_full, u)

    k0 = jnp.array(20.0)
    grad = jax.grad(compliance)(k0)
    eps = 1e-4
    fd = (float(compliance(k0 + eps)) - float(compliance(k0 - eps))) / (2.0 * eps)
    assert np.allclose(float(grad), fd, rtol=5e-4, atol=1e-5)


def test_jax_coupled_system_rbe3_matches_numpy_builder():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([0.25, 0.75], dtype=float)
    K_u = np.eye(12, dtype=float)
    F_u = np.zeros((12,), dtype=float)

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    np_builder.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    np_builder.register_field("slave", n_dofs=6, value_dim=1, offset=6)
    np_builder.add_rbe3_constraint(
        master="remote",
        slave="slave",
        ref_point=x_ref,
        slave_coords=x_slave,
        weights=weights,
        rho=0.0,
        backend="numpy",
    )
    K_np, F_np = np_builder.build().assemble(format="dense")

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.asarray(K_u), jnp.asarray(F_u))
    jax_builder.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    jax_builder.register_field("slave", n_dofs=6, value_dim=1, offset=6)
    jax_builder.add_rbe3_constraint(
        master="remote",
        slave="slave",
        ref_point=jnp.asarray(x_ref),
        slave_coords=jnp.asarray(x_slave),
        weights=jnp.asarray(weights),
        rho=0.0,
    )
    system_jax = jax_builder.build()

    assert np.allclose(np.asarray(system_jax.K_u.to_dense()), np.asarray(K_np), atol=1e-12)
    assert np.allclose(np.asarray(system_jax.F_u), np.asarray(F_np), atol=1e-12)


def test_jax_coupled_system_add_dof_tie_constraint_matches_numpy_builder():
    K_u = np.eye(5, dtype=float)
    F_u = np.zeros((5,), dtype=float)

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    np_builder.register_field("master", n_dofs=3, value_dim=1, offset=0)
    np_builder.register_field("slave", n_dofs=2, value_dim=1, offset=3)
    np_builder.add_dof_tie_constraint(
        master="master",
        slave="slave",
        master_dofs=np.array([0, 2], dtype=int),
        slave_dofs=np.array([1, 0], dtype=int),
        rhs=np.array([0.25, -0.5], dtype=float),
    )
    K_np, F_np = np_builder.build().assemble(format="dense")

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.asarray(K_u), jnp.asarray(F_u))
    jax_builder.register_field("master", n_dofs=3, value_dim=1, offset=0)
    jax_builder.register_field("slave", n_dofs=2, value_dim=1, offset=3)
    jax_builder.add_dof_tie_constraint(
        master="master",
        slave="slave",
        master_dofs=jnp.array([0, 2], dtype=jnp.int32),
        slave_dofs=jnp.array([1, 0], dtype=jnp.int32),
        rhs=jnp.array([0.25, -0.5]),
    )
    system_jax = jax_builder.build()

    assert np.allclose(np.asarray(system_jax.K_u.to_dense()), np.asarray(K_np), atol=1e-12)
    assert np.allclose(np.asarray(system_jax.F_u), np.asarray(F_np), atol=1e-12)


def test_jax_coupled_system_distributed_coupling_matches_numpy_builder():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    weights = np.array([0.25, 0.35, 0.40], dtype=float)
    K_u = np.eye(9, dtype=float)
    F_u = np.zeros((9,), dtype=float)

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    np_builder.register_field("workpiece", n_dofs=9, value_dim=1, offset=0)
    np_builder.add_distributed_coupling(
        source="workpiece",
        source_dofs=np.arange(9, dtype=int),
        remote="remote",
        point=x_ref,
        slave_coords=x_slave,
        weights=weights,
        backend="numpy",
    )
    K_np, F_np = np_builder.build().assemble(format="dense")

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.asarray(K_u), jnp.asarray(F_u))
    jax_builder.register_field("workpiece", n_dofs=9, value_dim=1, offset=0)
    copy_name = jax_builder.add_distributed_coupling(
        source="workpiece",
        source_dofs=jnp.arange(9, dtype=jnp.int32),
        remote="remote",
        point=jnp.asarray(x_ref),
        slave_coords=jnp.asarray(x_slave),
        weights=jnp.asarray(weights),
    )
    system_jax = jax_builder.build()

    assert copy_name == "remote_distributed_patch"
    assert np.asarray(system_jax.K_u.to_dense()).shape == np.asarray(K_np).shape
    assert np.asarray(system_jax.F_u).shape == np.asarray(F_np).shape


def test_jax_coupled_system_distributed_coupling_rejects_degenerate_remote_patch():
    builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(3), jnp.zeros((3,)))
    builder.register_field("workpiece", n_dofs=3, value_dim=1, offset=0)

    with pytest.raises(ValueError, match="rank"):
        builder.add_distributed_coupling(
            source="workpiece",
            source_dofs=jnp.arange(3, dtype=jnp.int32),
            remote="remote",
            point=jnp.zeros((3,)),
            slave_coords=jnp.array([[1.0, 0.0, 0.0]]),
        )


def test_jax_coupled_system_bolt_preload_matches_numpy_builder():
    K_u = np.zeros((6, 6), dtype=float)
    F_u = np.zeros((6,), dtype=float)

    np_builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    np_builder.register_field("bolt", n_dofs=6, value_dim=1, offset=0)
    np_builder.add_bolt_preload(
        "bolt",
        stiffness=12.0,
        direction=np.array([2.0, 0.0, 0.0], dtype=float),
        target_displacement=0.25,
    )
    K_np, F_np = np_builder.build().assemble(format="dense")

    jax_builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.zeros((6, 6), dtype=jnp.float64), jnp.zeros((6,), dtype=jnp.float64))
    jax_builder.register_field("bolt", n_dofs=6, value_dim=1, offset=0)
    jax_builder.add_bolt_preload(
        "bolt",
        stiffness=12.0,
        direction=jnp.array([2.0, 0.0, 0.0]),
        target_displacement=0.25,
    )
    system_jax = jax_builder.build()

    assert np.allclose(np.asarray(system_jax.K_u.to_dense()), np.asarray(K_np), atol=1e-12)
    assert np.allclose(np.asarray(system_jax.F_u), np.asarray(F_np), atol=1e-12)


def test_jax_coupled_system_remote_spring_with_rbe3_is_differentiable():
    x_ref = jnp.array([0.0, 0.0, 0.0])
    x_slave = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    weights = jnp.array([0.4, 0.6])
    load = jnp.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])

    def compliance(k):
        builder = ff.JAXCoupledSystemBuilder.from_structural(jnp.eye(6), load)
        builder.register_field("slave", n_dofs=6, value_dim=1, offset=0)
        builder.append_remote_point("remote", point=x_ref)
        builder.add_remote_spring(
            "remote",
            translational_stiffness=jnp.array([k, k, k]),
            rotational_stiffness=jnp.array([1e3, 1e3, 1e3]),
            translational_target=jnp.zeros((3,)),
            rotational_target=jnp.zeros((3,)),
        )
        builder.add_rbe3_constraint(
            master="remote",
            slave="slave",
            ref_point=x_ref,
            slave_coords=x_slave,
            weights=weights,
            rho=0.0,
        )
        system = builder.build()
        u = system.solve(diagonal_shift=1e-8)
        load_full = jnp.concatenate([load, jnp.zeros((system.n_u - load.shape[0],), dtype=load.dtype)])
        return jnp.dot(load_full, u)

    k0 = jnp.array(7.0)
    grad = jax.grad(compliance)(k0)
    eps = 1e-4
    fd = (float(compliance(k0 + eps)) - float(compliance(k0 - eps))) / (2.0 * eps)
    assert np.allclose(float(grad), fd, rtol=5e-4, atol=1e-5)

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import fluxfem as ff


def test_coupled_system_add_contact_nitsche_lifts_blocks():
    # Structural block: 4 dofs
    K_u = sp.diags([10.0, 20.0, 30.0, 40.0], format="csr")
    F_u = np.zeros(4, dtype=float)
    system = ff.CoupledSystem.from_structural(K_u, F_u)

    # Interface ordering: [master node 0, slave node 0], value_dim=1
    J_if = np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=float)
    r_if = np.array([1.0, -1.0], dtype=float)
    system.add_contact_nitsche(
        J_if,
        residual=r_if,
        n_master_nodes=1,
        n_slave_nodes=1,
        master_offset=0,
        slave_offset=2,
        value_dim=1,
    )

    K, F = system.assemble(format="csr")
    Kd = K.toarray()
    expected = np.diag([10.0, 20.0, 30.0, 40.0])
    expected[0, 0] += 2.0
    expected[0, 2] += -2.0
    expected[2, 0] += -2.0
    expected[2, 2] += 2.0
    assert np.allclose(Kd, expected, atol=1e-12)
    # residual_sign default is -1: F += -P^T r = [-1, 0, 1, 0]
    assert np.allclose(F, np.array([-1.0, 0.0, 1.0, 0.0]), atol=1e-12)


def test_coupled_system_builder_register_and_add_nitsche():
    K_u = sp.diags([10.0, 20.0, 30.0, 40.0], format="csr")
    F_u = np.zeros(4, dtype=float)
    builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
    builder.register_field("master", offset=0, n_dofs=2, value_dim=1, n_nodes=2)
    builder.register_field("slave", offset=2, n_dofs=2, value_dim=1, n_nodes=2)

    class _Ops:
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

    builder.add_contact_nitsche(_Ops(), master="master", slave="slave")
    K, F = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 0], 11.0, atol=1e-12)
    assert np.allclose(Kd[1, 1], 22.0, atol=1e-12)
    assert np.allclose(Kd[2, 2], 31.0, atol=1e-12)
    assert np.allclose(Kd[3, 3], 42.0, atol=1e-12)
    assert np.allclose(F, np.array([-1.0, -2.0, 1.0, 2.0]), atol=1e-12)


def test_coupled_system_builder_auto_offset_register_space():
    class _Space:
        def __init__(self, n_dofs):
            self.n_dofs = n_dofs

    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 20.0, 30.0, 40.0], format="csr"), np.zeros(4))
    builder.register_space("a", _Space(2), value_dim=1)
    builder.register_space("b", _Space(2), value_dim=1)

    J_if = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 2.0, 0.0, -2.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -2.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    builder.add_contact_nitsche(J_if, master="a", slave="b", residual=np.zeros(4))
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], -1.0, atol=1e-12)
    assert np.allclose(Kd[1, 3], -2.0, atol=1e-12)


def test_coupled_system_builder_auto_offset_register_field():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 20.0, 30.0, 40.0], format="csr"), np.zeros(4))
    builder.register_field("a", n_dofs=2, value_dim=1)
    builder.register_field("b", n_dofs=2, value_dim=1)

    J_if = np.array(
        [
            [3.0, 0.0, -3.0, 0.0],
            [0.0, 4.0, 0.0, -4.0],
            [-3.0, 0.0, 3.0, 0.0],
            [0.0, -4.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    builder.add_contact_nitsche(J_if, master="a", slave="b", residual=np.zeros(4))
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], -3.0, atol=1e-12)
    assert np.allclose(Kd[1, 3], -4.0, atol=1e-12)


def test_coupled_system_builder_register_blocks_mixed_entries():
    class _Space:
        def __init__(self, n_dofs):
            self.n_dofs = n_dofs

    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(4, format="csr"), np.zeros(4))
    builder.register_blocks(
        [
            ("a", _Space(1)),
            ("b", _Space(1), {"value_dim": 1}),
            {"name": "c", "n_dofs": 1, "value_dim": 1},
            {"name": "d", "space": _Space(1), "value_dim": 1},
        ]
    )
    J_if = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    builder.add_contact_nitsche(J_if, master="a", slave="b", residual=np.zeros(2))
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 1], -1.0, atol=1e-12)


def test_coupled_system_builder_field_name_suggestion():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(2, format="csr"), np.zeros(2))
    builder.register_field("master", n_dofs=1, value_dim=1)
    builder.register_field("slave", n_dofs=1, value_dim=1)
    with pytest.raises(ValueError, match="Did you mean 'master'\\?"):
        builder.add_contact_nitsche(np.eye(2), master="maste", slave="slave", residual=np.zeros(2))


def test_coupled_system_builder_add_contact_mortar_from_operators():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

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

    class _Ops:
        pass

    ops = _Ops()
    ops.coupling_aa = coupling_aa
    ops.coupling_ab = coupling_ab
    ops.facet_conn_master = np.array([[0]], dtype=int)

    builder.add_contact_mortar(ops, master="a", slave="b", multiplier_space="nodal", rho=0.0)
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert Kd.shape == (3, 3)
    # Structural diagonal
    assert np.allclose(Kd[0, 0], 10.0, atol=1e-12)
    assert np.allclose(Kd[1, 1], 30.0, atol=1e-12)
    # Lifted mortar coupling (+/- to lambda dof)
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)
    assert np.allclose(Kd[2, 0], 1.0, atol=1e-12)
    assert np.allclose(Kd[2, 1], -1.0, atol=1e-12)


def test_coupled_system_builder_add_contact_mortar_uses_ops_defaults():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

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
    ops = ff.ContactOperators(
        enforcement="mortar",
        coupling_aa=coupling_aa,
        coupling_ab=coupling_ab,
        facet_conn_master=np.array([[0]], dtype=int),
        rho=2.0,
        multiplier_space="nodal",
    )
    builder.add_contact_mortar(ops, master="a", slave="b")
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 0], 12.0, atol=1e-12)
    assert np.allclose(Kd[1, 1], 32.0, atol=1e-12)
    assert np.allclose(Kd[0, 1], -2.0, atol=1e-12)


def test_coupled_system_builder_add_contact_unified_nitsche():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 20.0, 30.0, 40.0], format="csr"), np.zeros(4))
    builder.register_field("master", n_dofs=2, value_dim=1)
    builder.register_field("slave", n_dofs=2, value_dim=1)

    ops = ff.ContactOperators(
        enforcement="nitsche",
        jacobian=np.array(
            [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, 2.0, 0.0, -2.0],
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0, 2.0],
            ],
            dtype=float,
        ),
        residual=np.array([1.0, 2.0, -1.0, -2.0], dtype=float),
    )
    builder.add_contact(ops, master="master", slave="slave")
    K, F = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], -1.0, atol=1e-12)
    assert np.allclose(Kd[1, 3], -2.0, atol=1e-12)
    assert np.allclose(F, np.array([-1.0, -2.0, 1.0, 2.0]), atol=1e-12)


def test_coupled_system_builder_add_contact_unified_mortar():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

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
    ops = ff.ContactOperators(
        enforcement="mortar",
        coupling_aa=coupling_aa,
        coupling_ab=coupling_ab,
        facet_conn_master=np.array([[0]], dtype=int),
        rho=0.0,
        multiplier_space="nodal",
    )
    builder.add_contact(ops, master="a", slave="b")
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)


def test_coupled_system_builder_add_contact_accepts_enforcement_alias():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 20.0, 30.0, 40.0], format="csr"), np.zeros(4))
    builder.register_field("master", n_dofs=2, value_dim=1)
    builder.register_field("slave", n_dofs=2, value_dim=1)

    ops = ff.ContactOperators(
        enforcement="nitsche",
        jacobian=np.array(
            [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, 2.0, 0.0, -2.0],
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0, 2.0],
            ],
            dtype=float,
        ),
        residual=np.array([1.0, 2.0, -1.0, -2.0], dtype=float),
    )
    builder.add_contact(ops, master="master", slave="slave", enforcement="nitsche")
    K, F = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], -1.0, atol=1e-12)
    assert np.allclose(Kd[1, 3], -2.0, atol=1e-12)
    assert np.allclose(F, np.array([-1.0, -2.0, 1.0, 2.0]), atol=1e-12)


def test_coupled_system_builder_add_contact_rejects_unknown_enforcement():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(2, format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)
    with pytest.raises(ValueError, match="enforcement must be"):
        builder.add_contact(np.eye(2), master="a", slave="b", enforcement="foobar")


def test_coupled_system_builder_add_contact_routes_by_formulation_multiplier():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

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
    ops = ff.ContactOperators(
        enforcement="mortar",
        formulation="multiplier",
        coupling_aa=coupling_aa,
        coupling_ab=coupling_ab,
        facet_conn_master=np.array([[0]], dtype=int),
        rho=0.0,
        multiplier_space="nodal",
    )
    builder.add_contact(ops, master="a", slave="b", formulation="multiplier")
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)


def test_coupled_system_builder_add_contact_rejects_formulation_enforcement_mismatch():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(2, format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)
    with pytest.raises(ValueError, match="formulation suggests nitsche"):
        builder.add_contact(np.eye(2), master="a", slave="b", enforcement="mortar", formulation="penalty")


def test_coupled_system_builder_add_contact_accepts_raw_contact_penalty_form():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

    class _RawContact:
        # Marker so builder treats this as a contact-space object.
        def assemble_contact_constraint_operators(self, **kwargs):
            _ = kwargs
            raise RuntimeError("not used in this test")

        def assemble_residual(self, res_form, u, params, *, normal_source="master"):
            _ = (res_form, u, params, normal_source)
            return np.array([1.0, -1.0], dtype=float)

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

    def _wf(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0]), "b": np.array([0.0])}

    builder.add_contact(
        _RawContact(),
        master="a",
        slave="b",
        weak_form=_wf,
        state={"a": np.array([0.0]), "b": np.array([0.0])},
        params=object(),
        value_dim=1,
    )
    K, F = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 1], -2.0, atol=1e-12)
    assert np.allclose(F, np.array([-1.0, 1.0]), atol=1e-12)

from __future__ import annotations

import warnings
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

    builder.add_contact_mortar(ops, master="a", slave="b", multiplier=ff.MultiplierSpec(family="nodal"), rho=0.0)
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
        multiplier=ff.MultiplierSpec(family="nodal"),
    )
    builder.add_contact_mortar(ops, master="a", slave="b")
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 0], 12.0, atol=1e-12)
    assert np.allclose(Kd[1, 1], 32.0, atol=1e-12)
    assert np.allclose(Kd[0, 1], -2.0, atol=1e-12)


def test_coupled_system_builder_add_contact_mortar_uses_multiplier_object_default():
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
        multiplier=ff.MultiplierSpec(family="nodal"),
    )
    builder.add_contact_mortar(ops, master="a", slave="b")
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 0], 12.0, atol=1e-12)
    assert np.allclose(Kd[1, 1], 32.0, atol=1e-12)
    assert np.allclose(Kd[0, 1], -2.0, atol=1e-12)


def test_coupled_system_builder_add_contact_mortar_multiple_contacts_different_lambda_sizes():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], format="csr"), np.zeros(6))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)
    builder.register_field("c", n_dofs=2, value_dim=1)
    builder.register_field("d", n_dofs=2, value_dim=1)

    coupling_aa_1 = ff.ContactCouplingMatrix(
        rows=np.array([0], dtype=int),
        cols=np.array([0], dtype=int),
        data=np.array([1.0], dtype=float),
        shape=(1, 1),
    )
    coupling_ab_1 = ff.ContactCouplingMatrix(
        rows=np.array([0], dtype=int),
        cols=np.array([0], dtype=int),
        data=np.array([1.0], dtype=float),
        shape=(1, 1),
    )
    ops1 = ff.ContactOperators(
        enforcement="mortar",
        coupling_aa=coupling_aa_1,
        coupling_ab=coupling_ab_1,
        facet_conn_master=np.array([[0]], dtype=int),
        rho=0.0,
        multiplier=ff.MultiplierSpec(family="nodal"),
    )
    builder.add_contact_mortar(ops1, master="a", slave="b")

    coupling_aa_2 = ff.ContactCouplingMatrix(
        rows=np.array([0, 1], dtype=int),
        cols=np.array([0, 1], dtype=int),
        data=np.array([1.0, 1.0], dtype=float),
        shape=(2, 2),
    )
    coupling_ab_2 = ff.ContactCouplingMatrix(
        rows=np.array([0, 1], dtype=int),
        cols=np.array([0, 1], dtype=int),
        data=np.array([1.0, 1.0], dtype=float),
        shape=(2, 2),
    )
    ops2 = ff.ContactOperators(
        enforcement="mortar",
        coupling_aa=coupling_aa_2,
        coupling_ab=coupling_ab_2,
        facet_conn_master=np.array([[0], [1]], dtype=int),
        rho=0.0,
        multiplier=ff.MultiplierSpec(family="nodal"),
    )
    builder.add_contact_mortar(ops2, master="c", slave="d")

    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()

    # n_u=6, lambda blocks: 1 + 2
    assert Kd.shape == (9, 9)

    # First contact (a,b) couples into first lambda dof at index 6.
    assert np.allclose(Kd[0, 6], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 6], -1.0, atol=1e-12)
    assert np.allclose(Kd[6, 0], 1.0, atol=1e-12)
    assert np.allclose(Kd[6, 1], -1.0, atol=1e-12)

    # Second contact (c,d) couples into lambda dofs at indices 7,8.
    assert np.allclose(Kd[2, 7], 1.0, atol=1e-12)
    assert np.allclose(Kd[4, 7], -1.0, atol=1e-12)
    assert np.allclose(Kd[3, 8], 1.0, atol=1e-12)
    assert np.allclose(Kd[5, 8], -1.0, atol=1e-12)

    # Independent lambda blocks: no cross coupling between old/new lambda.
    assert np.allclose(Kd[6, 7], 0.0, atol=1e-12)
    assert np.allclose(Kd[7, 6], 0.0, atol=1e-12)


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
        multiplier=ff.MultiplierSpec(family="nodal"),
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


def test_coupled_system_builder_add_contact_accepts_family_constraint_alias():
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
        multiplier=ff.MultiplierSpec(family="nodal"),
    )
    builder.add_contact(ops, master="a", slave="b", family="constraint")
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)


def test_coupled_system_builder_add_contact_rejects_family_enforcement_conflict():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(2, format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)
    with pytest.raises(ValueError, match="family conflicts with enforcement"):
        builder.add_contact(np.eye(2), master="a", slave="b", family="constraint", enforcement="nitsche")


def test_coupled_system_builder_add_contact_rejects_family_formulation_conflict():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(2, format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)
    with pytest.raises(ValueError, match="family='constraint' conflicts"):
        builder.add_contact(np.eye(2), master="a", slave="b", family="constraint", formulation="penalty")


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
        multiplier=ff.MultiplierSpec(family="nodal"),
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

    with pytest.warns(DeprecationWarning, match="raw contact interface"):
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


def test_coupled_system_builder_add_contact_constraint_family_consumes_eval_inputs():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)
    calls = {"residual": 0, "jacobian": 0}

    class _RawContact:
        def assemble_contact_constraint_operators(self, **kwargs):
            _ = kwargs
            raise RuntimeError("not used in this test")

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
            calls["residual"] += 1
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
            calls["jacobian"] += 1
            return np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=float)

    def _wf(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0]), "b": np.array([0.0])}

    with pytest.warns(DeprecationWarning, match="raw contact interface"):
        builder.add_contact(
            _RawContact(),
            master="a",
            slave="b",
            family="constraint",
            multiplier=ff.MultiplierSpec(family="nodal"),
            weak_form=_wf,
            state={"a": np.array([0.0]), "b": np.array([0.0])},
            params=object(),
            value_dim=1,
        )
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)
    assert calls["residual"] == 1
    assert calls["jacobian"] == 1


def test_coupled_system_builder_add_constraint_matrix_unified_kkt():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

    C = np.array([[1.0, -1.0]], dtype=float)
    builder.add_constraint_matrix(C, master="a", slave="b", value_dim=1, rho=0.0)
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)


def test_coupled_system_builder_add_embedding_constraint_unified_kkt():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

    emb = ff.EmbeddingMap(
        rows=np.array([0], dtype=int),
        cols=np.array([0], dtype=int),
        data=np.array([1.0], dtype=float),
        shape=(1, 1),
        mode="nodal",
    )
    builder.add_embedding_constraint(emb, master="a", slave="b", value_dim=1, rho=0.0)
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)


def test_coupled_system_builder_add_embedding_constraint_subset_rows_compacted():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 20.0, 30.0], format="csr"), np.zeros(3))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=2, value_dim=1)

    emb = ff.EmbeddingMap(
        rows=np.array([1], dtype=int),
        cols=np.array([0], dtype=int),
        data=np.array([1.0], dtype=float),
        shape=(2, 1),
        mode="barycentric",
    )
    builder.add_embedding_constraint(emb, master="a", slave="b", value_dim=1, rho=0.0)
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    # n_u=3, n_lambda=1 -> shape 4x4
    assert Kd.shape == (4, 4)
    # tie only b[1] to a[0] => [1, 0, -1] in contact ordering
    assert np.allclose(Kd[0, 3], 1.0, atol=1e-12)
    assert np.allclose(Kd[2, 3], -1.0, atol=1e-12)


def test_embedding_plane_selector_to_builder_solve_flow():
    master_mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    slave_mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    n_m = int(np.asarray(master_mesh.coords).shape[0])
    n_s = int(np.asarray(slave_mesh.coords).shape[0])

    emb = ff.build_barycentric_embedding_map_from_meshes(
        master_mesh,
        slave_mesh,
        slave_facet_selector=lambda m: m.facets_on_plane(axis=2, value=0.0),
        tol=1e-8,
        allow_unmapped="error",
    )

    K_u = sp.eye(n_m + n_s, format="csr")
    F_u = np.zeros((n_m + n_s,), dtype=float)
    # Apply load only on slave side so constraint effect is visible.
    F_u[n_m:] = 1.0

    builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
    builder.register_field("master", n_dofs=n_m, value_dim=1)
    builder.register_field("slave", n_dofs=n_s, value_dim=1)
    builder.add_embedding_constraint(emb, master="master", slave="slave", value_dim=1, rho=0.0)

    u = builder.build().solve(format="csr", diagonal_shift=1e-8)

    C = ff.assemble_embedding_constraint_matrix(
        emb,
        n_master_nodes=n_m,
        n_slave_nodes=n_s,
        value_dim=1,
        backend="numpy",
    )
    r = C @ np.asarray(u[: n_m + n_s], dtype=float)
    assert np.allclose(r, np.zeros_like(r), atol=1e-6)


def test_rbe2_like_constraint_matrix_flow_in_tests():
    # Scalar analogue of RBE2:
    # slave nodes (1,2) are rigidly tied to master reference node (0):
    # u_s1 - u_m0 = 0, u_s2 - u_m0 = 0
    n_m = 1
    n_s = 2
    K_u = sp.eye(n_m + n_s, format="csr")
    F_u = np.array([0.0, 1.0, 2.0], dtype=float)  # load only on slave side

    builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
    builder.register_field("master", n_dofs=n_m, value_dim=1)
    builder.register_field("slave", n_dofs=n_s, value_dim=1)

    C = np.array(
        [
            [-1.0, 1.0, 0.0],  # u_s1 - u_m0
            [-1.0, 0.0, 1.0],  # u_s2 - u_m0
        ],
        dtype=float,
    )
    builder.add_constraint_matrix(C, master="master", slave="slave", value_dim=1, rho=0.0)
    u = builder.build().solve(format="csr", diagonal_shift=1e-8)

    u_m0 = float(u[0])
    u_s1 = float(u[1])
    u_s2 = float(u[2])
    assert np.allclose([u_s1, u_s2], [u_m0, u_m0], atol=1e-6)


def test_rbe2_constraint_matrix_flow_with_ref_rotation_dofs():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    C = ff.assemble_rbe2_constraint_matrix(x_ref, x_slave, backend="numpy")

    n_ref = 6
    n_slave = 3 * x_slave.shape[0]
    K_u = sp.eye(n_ref + n_slave, format="csr")
    F_u = np.zeros((n_ref + n_slave,), dtype=float)

    builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
    # value_dim=1 so add_constraint_matrix expects direct DOF counts.
    builder.register_field("ref", n_dofs=n_ref, value_dim=1)
    builder.register_field("slave", n_dofs=n_slave, value_dim=1)
    builder.add_constraint_matrix(C, master="ref", slave="slave", value_dim=1, rho=0.0)
    u = np.asarray(builder.build().solve(format="csr", diagonal_shift=1e-8), dtype=float)

    q = u[: n_ref + n_slave]
    r = C @ q
    assert np.allclose(r, np.zeros_like(r), atol=1e-6)


def test_builder_resolve_block_dofs_from_nodes_components():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(6, format="csr"), np.zeros(6))
    builder.register_field("u", n_dofs=6, value_dim=2, n_nodes=3, offset=0)

    dofs = builder.resolve_block_dofs("u", nodes=[0, 2], components=[1])
    assert np.array_equal(dofs, np.array([1, 5], dtype=int))

    dofs_all = builder.resolve_block_dofs("u", nodes=[1])
    assert np.array_equal(dofs_all, np.array([2, 3], dtype=int))


def test_builder_resolve_dirichlet_specs_mixed_inputs():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(8, format="csr"), np.zeros(8))
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


def test_builder_solve_with_dirichlet_specs():
    K = sp.eye(4, format="csr")
    F = np.array([2.0, 0.0, 0.0, -3.0], dtype=float)
    builder = ff.CoupledSystemBuilder.from_structural(K, F)
    builder.register_field("master", n_dofs=2, value_dim=1, offset=0)
    builder.register_field("slave", n_dofs=2, value_dim=1, offset=2)

    u = np.asarray(
        builder.solve(
            dirichlet_specs=[
                ff.DirichletSpec(field="master", nodes=[0], value=0.5),
                ff.DirichletSpec(field="slave", local_dofs=[1], value=-1.0),
            ],
            format="csr",
        ),
        dtype=float,
    )
    # Identity system with Dirichlet enforcement => constrained values should match exactly.
    assert np.allclose(u[[0, 3]], np.array([0.5, -1.0]), atol=1e-12)


def test_builder_resolve_dirichlet_rejects_dict_specs():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(2, format="csr"), np.zeros(2))
    builder.register_field("u", n_dofs=2, value_dim=1, n_nodes=2, offset=0)
    with pytest.raises(TypeError, match="DirichletSpec"):
        builder.resolve_dirichlet([{"field": "u", "nodes": [0], "value": 0.0}])  # type: ignore[list-item]


def test_dirichlet_spec_requires_single_selector_style():
    with pytest.raises(ValueError, match="exactly one of nodes or local_dofs"):
        ff.DirichletSpec(field="u", value=0.0)
    with pytest.raises(ValueError, match="exactly one of nodes or local_dofs"):
        ff.DirichletSpec(field="u", nodes=[0], local_dofs=[0], value=0.0)


def test_builder_add_constraint_with_matrix_spec():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

    spec = ff.ConstraintSpec(
        kind="matrix",
        master="a",
        slave="b",
        C=np.array([[1.0, -1.0]], dtype=float),
        value_dim=1,
        rho=0.0,
    )
    builder.add_constraint(spec)
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)


def test_builder_add_constraint_with_embedding_spec():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 30.0], format="csr"), np.zeros(2))
    builder.register_field("a", n_dofs=1, value_dim=1)
    builder.register_field("b", n_dofs=1, value_dim=1)

    emb = ff.EmbeddingMap(
        rows=np.array([0], dtype=int),
        cols=np.array([0], dtype=int),
        data=np.array([1.0], dtype=float),
        shape=(1, 1),
        mode="nodal",
    )
    spec = ff.ConstraintSpec(
        kind="embedding",
        master="a",
        slave="b",
        embedding=emb,
        value_dim=1,
        rho=0.0,
        backend="numpy",
    )
    builder.add_constraint(spec)
    K, _ = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd[0, 2], 1.0, atol=1e-12)
    assert np.allclose(Kd[1, 2], -1.0, atol=1e-12)


def test_builder_append_field_extends_structural_system():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(2, format="csr"), np.array([1.0, 2.0], dtype=float))
    builder.register_field("base", n_dofs=2, value_dim=1, offset=0)
    builder.append_field("remote", n_dofs=3, value_dim=1, F_block=np.array([3.0, 4.0, 5.0], dtype=float))

    remote = builder._get_block("remote")
    assert remote.offset == 2
    assert remote.n_dofs == 3
    assert builder.system.K_u.shape == (5, 5)
    assert np.allclose(builder.system.F_u, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float), atol=1e-12)


def test_builder_append_remote_point_registers_6dof_field():
    builder = ff.CoupledSystemBuilder.from_structural(sp.csr_matrix((0, 0)), np.zeros((0,), dtype=float))
    builder.append_remote_point("remote", point=np.array([1.0, 2.0, 3.0], dtype=float))

    remote = builder._get_block("remote")
    assert remote.offset == 0
    assert remote.n_dofs == 6
    assert np.allclose(remote.point, np.array([1.0, 2.0, 3.0], dtype=float), atol=1e-12)


def test_builder_add_field_matrix_and_dof_spring():
    builder = ff.CoupledSystemBuilder.from_structural(sp.csr_matrix((0, 0)), np.zeros((0,), dtype=float))
    builder.append_field("remote", n_dofs=2, value_dim=1)
    builder.add_field_matrix(
        "remote",
        np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float),
        F_local=np.array([0.5, -0.5], dtype=float),
    )
    builder.add_dof_spring("remote", local_dofs=[1], stiffness=3.0, reference_value=2.0)

    K, F = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(Kd, np.array([[2.0, -1.0], [-1.0, 5.0]], dtype=float), atol=1e-12)
    assert np.allclose(F, np.array([0.5, 5.5], dtype=float), atol=1e-12)


def test_builder_add_remote_spring_for_6dof_remote_point():
    builder = ff.CoupledSystemBuilder.from_structural(sp.csr_matrix((0, 0)), np.zeros((0,), dtype=float))
    builder.append_remote_point("remote", point=np.array([0.0, 0.0, 0.0], dtype=float))
    builder.add_remote_spring(
        "remote",
        translational_stiffness=np.array([10.0, 20.0, 30.0], dtype=float),
        rotational_stiffness=np.array([4.0, 5.0, 6.0], dtype=float),
        translational_target=np.array([1.0, 0.0, -1.0], dtype=float),
        rotational_target=np.array([0.5, 0.0, 0.0], dtype=float),
    )

    K, F = builder.build().assemble(format="csr")
    Kd = K.toarray()
    assert np.allclose(np.diag(Kd), np.array([10.0, 20.0, 30.0, 4.0, 5.0, 6.0], dtype=float), atol=1e-12)
    assert np.allclose(F, np.array([10.0, 0.0, -30.0, 2.0, 0.0, 0.0], dtype=float), atol=1e-12)


def test_builder_remote_rbe2_spring_flow():
    K_u = sp.eye(3, format="csr")
    F_u = np.zeros((3,), dtype=float)

    builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
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

    # The remote spring is balanced by the slave structural stiffness through the
    # RBE2 constraint, so the target displacement is approached but not matched
    # exactly for finite spring stiffness.
    assert np.allclose(q_slave, q_remote[:3], atol=1e-6)
    assert np.allclose(q_remote[:3], np.array([20.0 / 21.0, 0.0, 0.0]), atol=1e-6)


def test_builder_add_constraint_with_rbe2_spec():
    builder = ff.CoupledSystemBuilder.from_structural(sp.eye(3, format="csr"), np.zeros((3,), dtype=float))
    builder.register_field("slave", n_dofs=3, value_dim=1, offset=0)
    builder.append_field("remote", n_dofs=6, value_dim=1)
    spec = ff.ConstraintSpec(
        kind="rbe2",
        master="remote",
        slave="slave",
        ref_point=np.array([0.0, 0.0, 0.0], dtype=float),
        slave_coords=np.array([[1.0, 0.0, 0.0]], dtype=float),
        rho=0.0,
    )
    builder.add_constraint(spec)
    K, _ = builder.build().assemble(format="csr")
    assert K.shape == (12, 12)


def test_builder_add_rbe3_constraint_preserves_rigid_motion():
    x_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    x_slave = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    builder = ff.CoupledSystemBuilder.from_structural(sp.csr_matrix((12, 12)), np.zeros((12,), dtype=float))
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
    Kd = K.toarray()
    C = Kd[12:, :12]

    u_ref = np.array([0.2, -0.1, 0.05], dtype=float)
    w_ref = np.array([0.0, 0.0, 0.4], dtype=float)
    u_s = np.asarray([u_ref + np.cross(w_ref, p - x_ref) for p in x_slave], dtype=float).reshape(-1)
    q = np.concatenate([u_ref, w_ref, u_s], axis=0)
    assert np.allclose(C @ q, np.zeros((6,), dtype=float), atol=1e-12)


def test_builder_allows_multiple_rbe3_constraints():
    builder = ff.CoupledSystemBuilder.from_structural(sp.csr_matrix((0, 0)), np.zeros((0,), dtype=float))
    builder.append_field("remote_a", n_dofs=6, value_dim=1)
    builder.append_field("slave_a", n_dofs=6, value_dim=1)
    builder.append_field("remote_b", n_dofs=6, value_dim=1)
    builder.append_field("slave_b", n_dofs=6, value_dim=1)

    builder.add_rbe3_constraint(
        master="remote_a",
        slave="slave_a",
        ref_point=np.array([0.0, 0.0, 0.0], dtype=float),
        slave_coords=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
        rho=0.0,
    )
    builder.add_rbe3_constraint(
        master="remote_b",
        slave="slave_b",
        ref_point=np.array([2.0, 0.0, 0.0], dtype=float),
        slave_coords=np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=float),
        weights=np.array([2.0, 1.0], dtype=float),
        rho=0.0,
    )

    K, _ = builder.build().assemble(format="csr")
    # 24 structural dofs + 6 + 6 lambda rows from two RBE3 constraints
    assert K.shape == (36, 36)


def test_builder_add_constraint_with_rbe3_spec():
    builder = ff.CoupledSystemBuilder.from_structural(sp.csr_matrix((12, 12)), np.zeros((12,), dtype=float))
    builder.register_field("remote", n_dofs=6, value_dim=1, offset=0)
    builder.register_field("slave", n_dofs=6, value_dim=1, offset=6)
    spec = ff.ConstraintSpec(
        kind="rbe3",
        master="remote",
        slave="slave",
        ref_point=np.array([0.0, 0.0, 0.0], dtype=float),
        slave_coords=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
        rho=0.0,
    )
    builder.add_constraint(spec)
    K, _ = builder.build().assemble(format="csr")
    assert K.shape == (18, 18)


def test_constraint_spec_rejects_missing_payload():
    with pytest.raises(ValueError, match="requires C"):
        ff.ConstraintSpec(kind="matrix", master="a", slave="b")
    with pytest.raises(ValueError, match="requires ref_point and slave_coords"):
        ff.ConstraintSpec(kind="rbe2", master="a", slave="b")
    with pytest.raises(ValueError, match="requires ref_point and slave_coords"):
        ff.ConstraintSpec(kind="rbe3", master="a", slave="b")


def test_coupled_system_builder_add_contact_prefers_explicit_contribution_without_warning():
    builder = ff.CoupledSystemBuilder.from_structural(sp.diags([10.0, 20.0, 30.0, 40.0], format="csr"), np.zeros(4))
    builder.register_field("master", n_dofs=2, value_dim=1)
    builder.register_field("slave", n_dofs=2, value_dim=1)

    ops = ff.PenaltyContactContribution(
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

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        builder.add_contact(ops, master="master", slave="slave")
    assert len(record) == 0

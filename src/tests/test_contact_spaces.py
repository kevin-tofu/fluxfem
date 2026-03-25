import numpy as np
import jax.numpy as jnp
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def _penalty_bilinear(v1, v2, u1, u2, p):
    ju = u1.val - u2.val
    return ((p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))) * h_wf.ds()


def test_contact_spaces_builds_pair_contact_surface_space():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf_s, elem_conn=conn, value_dim=3)

    spec = ff.ContactSpaces(master=side_m, slave=side_s, field_master="master", field_slave="slave")
    with pytest.warns(DeprecationWarning, match="to_contact_surface_space"):
        contact = spec.to_contact_surface_space(quad_order=1, backend="jax")

    assert isinstance(contact, ff.ContactSurfaceSpace)
    assert contact.field_master == "master"
    assert contact.field_slave == "slave"
    assert contact.batch_jac is None


def test_contact_spaces_preserves_explicit_batch_jac_flag():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf_s, elem_conn=conn, value_dim=3)

    with pytest.warns(DeprecationWarning, match="to_contact_surface_space"):
        contact = ff.ContactSpaces(master=side_m, slave=side_s).to_contact_surface_space(
            quad_order=1,
            backend="jax",
            batch_jac=False,
        )

    assert contact.batch_jac is False


def test_contact_pair_spec_prepare_alias_builds_same_contact_surface_space():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSideSpec.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSideSpec.from_surfaces(surf_s, elem_conn=conn, value_dim=3)

    prepared = ff.ContactPairSpec(master=side_m, slave=side_s).prepare(
        quad_order=1,
        backend="jax",
    )

    assert isinstance(prepared, ff.PreparedContactInterface)
    assert isinstance(prepared, ff.ContactSurfaceSpace)
    assert prepared.field_master == "a"
    assert prepared.field_slave == "b"


def test_contact_spaces_matches_direct_from_sides_bilinear():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf_s, elem_conn=conn, value_dim=3)

    direct = ff.ContactSurfaceSpace.from_sides(side_m, side_s, quad_order=1, backend="jax")
    with pytest.warns(DeprecationWarning, match="to_contact_surface_space"):
        via_spec = ff.ContactSpaces(master=side_m, slave=side_s).to_contact_surface_space(
            quad_order=1,
            backend="jax",
        )

    n = coords.shape[0] * 3
    u_m = jnp.zeros(n)
    u_s = jnp.zeros(n)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    K_direct = np.asarray(direct.assemble_bilinear(_penalty_bilinear, u_m, u_s, params))
    K_spec = np.asarray(via_spec.assemble_bilinear(_penalty_bilinear, u_m, u_s, params))

    assert np.allclose(K_spec, K_direct, atol=1e-10)


def test_contact_group_spaces_matches_direct_one_to_many_bilinear():
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

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s1 = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s2 = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s1 = ff.ContactSide.from_surfaces(surf_s1, elem_conn=conn, value_dim=3)
    side_s2 = ff.ContactSide.from_surfaces(surf_s2, elem_conn=conn, value_dim=3)

    direct = ff.OneToManyContactSurfaceSpace.from_sides(
        side_m,
        [side_s1, side_s2],
        quad_order=1,
        backend="jax",
    )
    with pytest.warns(DeprecationWarning, match="to_contact_surface_space"):
        via_spec = ff.ContactGroupSpaces(master=side_m, slaves=[side_s1, side_s2]).to_contact_surface_space(
            quad_order=1,
            backend="jax",
        )

    n = coords.shape[0] * 3
    u_m = jnp.zeros(n)
    u_s = jnp.zeros(n)
    params = ff.Params(alpha=10.0, inv_h=1.0)

    K_direct = np.asarray(direct.assemble_bilinear(_penalty_bilinear, u_m, [u_s, u_s], params))
    K_spec = np.asarray(via_spec.assemble_bilinear(_penalty_bilinear, u_m, [u_s, u_s], params))

    assert np.allclose(K_spec, K_direct, atol=1e-10)


def test_onesided_contact_spaces_matches_direct_from_side():
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

    surf = ff.SurfaceMesh.from_facets(coords, facets)
    side = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)

    direct = ff.OneSidedContactSurfaceSpace.from_side(side, quad_order=2)
    with pytest.warns(DeprecationWarning, match="to_contact_surface_space"):
        via_spec = ff.OneSidedContactSpaces(side=side).to_contact_surface_space(quad_order=2)

    assert isinstance(via_spec, ff.OneSidedContactSurfaceSpace)
    assert np.array_equal(via_spec.facet_to_elem_slave, direct.facet_to_elem_slave)
    assert via_spec.quad_order == direct.quad_order


def test_new_contact_aliases_are_publicly_exported():
    assert ff.ContactPairSpec is ff.ContactSpaces
    assert ff.ContactGroupSpec is ff.ContactGroupSpaces
    assert ff.OneSidedContactSpec is ff.OneSidedContactSpaces
    assert ff.ContactSideSpec is ff.ContactSide
    assert ff.PreparedContactInterface is ff.ContactSurfaceSpace
    assert ff.PreparedOneToManyContactInterface is ff.OneToManyContactSurfaceSpace
    assert ff.PreparedOneSidedContactInterface is ff.OneSidedContactSurfaceSpace
    assert ff.MultiplierSpec is ff.ContactMultiplierSpace


def test_contact_state_pair_initialize_and_update():
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
    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf_s, elem_conn=conn, value_dim=3)
    prepared = ff.ContactPairSpec(master=side_m, slave=side_s).prepare(quad_order=1, backend="jax")

    state0 = prepared.initialize_state(metadata={"source": "test"})
    state1 = prepared.update_state(
        state={"a": np.zeros(coords.shape[0] * 3), "b": np.zeros(coords.shape[0] * 3)},
        contact_state=state0,
        geometry="current",
        active_set="update",
    )

    assert isinstance(state0, ff.ContactState)
    assert state0.interface_kind == "pair"
    assert state0.iteration == 0
    assert state0.metadata["source"] == "test"
    assert state1.iteration == 1
    assert state1.geometry == "current"
    assert state1.active_set == "update"
    assert state1.field_summary == {"a": (12,), "b": (12,)}


def test_contact_state_one_to_many_and_one_sided_initialize():
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
    surf = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)
    side_s1 = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)
    side_s2 = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)

    otm = ff.ContactGroupSpec(master=side_m, slaves=[side_s1, side_s2]).prepare(quad_order=1, backend="jax")
    one_sided = ff.OneSidedContactSpec(side=side_s1).prepare(quad_order=2)

    otm_state = otm.initialize_state()
    one_sided_state = one_sided.initialize_state()

    assert isinstance(otm_state, ff.ContactState)
    assert otm_state.interface_kind == "one_to_many"
    assert otm_state.field_summary["master"] == (12,)
    assert otm_state.field_summary["slave"] == (12, 12)
    assert isinstance(one_sided_state, ff.ContactState)
    assert one_sided_state.interface_kind == "one_sided"


def test_contact_contribution_types_are_explicit_and_compatible():
    ops_penalty = ff.PenaltyContactContribution(enforcement="nitsche", residual=None, jacobian=None)
    ops_multiplier = ff.MultiplierContactContribution(enforcement="mortar", B=None, Kuu=None)

    assert isinstance(ops_penalty, ff.PenaltyContactContribution)
    assert isinstance(ops_penalty, ff.ContactOperators)
    assert ops_penalty.enforcement == "nitsche"
    assert isinstance(ops_multiplier, ff.MultiplierContactContribution)
    assert isinstance(ops_multiplier, ff.ContactOperators)
    assert ops_multiplier.enforcement == "mortar"


def test_onesided_penalty_top_level_alias_matches_legacy_tutorial_path():
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
    surface = ff.SurfaceMesh.from_facets(coords, facets)
    side = ff.ContactSideSpec.from_surfaces(surface, elem_conn=conn, value_dim=3)
    contact = ff.OneSidedContactSpec(
        side=side,
        surface_master=surface,
        elem_conn_master=conn,
    ).prepare(quad_order=2)
    u_master = np.linspace(0.0, 1.0, coords.shape[0] * 3, dtype=float)
    lam, mu = ff.lame_parameters(210e9, 0.3)
    params = ff.Params(alpha=10.0, lam=float(lam), mu=float(mu))

    class _OneSidedAdapter:
        def __init__(self, cs, u_master_vec):
            self._cs = cs
            self._u_master = u_master_vec
            self._cached = None

        @staticmethod
        def _u_hat(x):
            x = np.asarray(x, dtype=float)
            return np.stack([x[:, 0], -0.5 * x[:, 1], 0.25 * x[:, 2]], axis=1)

        def _assemble(self, params_in):
            if self._cached is None:
                self._cached = self._cs.assemble_bilinear(
                    self._u_hat,
                    params_in,
                    u_master=self._u_master,
                )
            return self._cached

        def assemble_residual(self, _res_form, _u, params_in, *, normal_source="master"):
            _ = normal_source
            _K, f = self._assemble(params_in)
            return np.asarray(f)

        def assemble_jacobian(
            self,
            _res_form,
            _u,
            params_in,
            *,
            normal_source="master",
            sparse=False,
            backend="numpy",
            batch_jac=None,
        ):
            _ = (normal_source, sparse, backend, batch_jac)
            K, _f = self._assemble(params_in)
            return np.asarray(K)

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0])}

    adapter = _OneSidedAdapter(contact, u_master)
    state = {"a": np.zeros(int(contact.surface_slave.n_nodes * contact.value_dim), dtype=float)}

    ops_new = ff.assemble_penalty(adapter, weak_form=_dummy_res_form, state=state, params=params, backend="jax")
    ops_old = ff.assemble_contact_penalty_operators(adapter, weak_form=_dummy_res_form, state=state, params=params, backend="jax")

    assert np.allclose(np.asarray(ops_new.jacobian), np.asarray(ops_old.jacobian))
    assert np.allclose(np.asarray(ops_new.residual), np.asarray(ops_old.residual))


def test_contact_assemble_contact_operators_routes_penalty_alias():
    class _PenaltyStub:
        def assemble_residual(self, _res_form, u, params, *, normal_source="master"):
            _ = normal_source
            return params.k * jnp.asarray(u["a"]) - jnp.asarray([params.f])

        def assemble_jacobian(
            self,
            _res_form,
            _u,
            params,
            *,
            normal_source="master",
            sparse=False,
            backend="jax",
            batch_jac=None,
        ):
            _ = (normal_source, sparse, backend, batch_jac)
            return jnp.asarray([[params.k]])

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": jnp.asarray([0.0])}

    state = {"a": jnp.asarray([0.0])}
    params = ff.Params(k=4.0, f=2.0, alpha=7.0)

    ops_new = ff.assemble_contact_operators(
        _PenaltyStub(),
        enforcement="penalty",
        weak_form=_dummy_res_form,
        state=state,
        params=params,
    )
    ops_old = ff.assemble_contact_penalty_operators(
        _PenaltyStub(),
        weak_form=_dummy_res_form,
        state=state,
        params=params,
        backend="jax",
    )

    assert ops_new.enforcement == "nitsche"
    assert np.allclose(np.asarray(ops_new.jacobian), np.asarray(ops_old.jacobian))
    assert np.allclose(np.asarray(ops_new.residual), np.asarray(ops_old.residual))


def test_contact_new_assemble_aliases_match_existing_names():
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
    surf = ff.SurfaceMesh.from_facets(coords, facets)
    side_m = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)
    side_s = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)
    contact = ff.ContactPairSpec(master=side_m, slave=side_s).prepare(quad_order=1, backend="jax")
    lm = ff.MultiplierSpec.from_contact(contact, family="p0", side="master", value_dim=3)

    with pytest.warns(DeprecationWarning, match="assemble_constraint_operators"):
        ops_old = contact.assemble_constraint_operators(rho=1.0, multiplier=lm, backend="numpy")
    ops_new = contact.assemble_multiplier(rho=1.0, multiplier=lm, backend="numpy")

    assert type(ops_old) is type(ops_new)
    assert ops_old.enforcement == ops_new.enforcement == "mortar"

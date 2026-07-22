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


def test_contact_contribution_diagnostics_report_penalty_energy_and_mortar_residual():
    ops_penalty = ff.PenaltyContactContribution(
        enforcement="nitsche",
        jacobian=np.array([[2.0, 0.5], [0.5, 4.0]], dtype=float),
    )
    assert np.isclose(ops_penalty.penalty_energy(np.array([1.0, 2.0])), 10.0)

    ops_multiplier = ff.MultiplierContactContribution(
        enforcement="mortar",
        B=np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]], dtype=float),
        rho=3.0,
    )
    u = (np.array([2.0, -1.0]), np.array([0.5, 1.0]))

    np.testing.assert_allclose(ops_multiplier.constraint_residual(u), np.array([1.5, -2.0]))
    assert np.isclose(ops_multiplier.constraint_residual_norm(u), 2.5)
    assert np.isclose(ops_multiplier.augmentation_energy(u), 9.375)


def test_contact_constraint_diagnostics_report_row_norms_and_rank_deficiency():
    B = np.array(
        [
            [1.0, -1.0, 0.0],
            [2.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    diag = ff.contact_constraint_matrix_diagnostics(B, max_singular_values=None)

    assert isinstance(diag, ff.ContactConstraintDiagnostics)
    assert diag.n_rows == 3
    assert diag.n_cols == 3
    assert diag.zero_row_count == 1
    assert diag.estimated_rank == 1
    assert diag.rank_deficiency == 2
    assert diag.singular_value_count == 3
    assert np.isclose(diag.row_norm_min, 0.0)
    assert np.isclose(diag.row_norm_max, np.sqrt(8.0))

    ops = ff.MultiplierContactContribution(enforcement="mortar", B=B)
    assert ops.constraint_diagnostics().estimated_rank == diag.estimated_rank


def test_contact_constraint_quality_reports_pass_warn_and_fail():
    B_good = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    report_good = ff.assess_contact_constraint_quality(B_good)

    assert isinstance(report_good, ff.ContactConstraintQualityReport)
    assert report_good.status == "pass"
    assert report_good.passed
    assert report_good.issues == ()

    B_bad = np.array(
        [
            [1.0, -1.0, 0.0],
            [2.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    report_fail = ff.assess_contact_constraint_quality(B_bad)

    assert report_fail.status == "fail"
    assert not report_fail.passed
    assert {issue.check for issue in report_fail.failures} == {"zero_rows", "rank_deficiency"}
    assert all(issue.hint for issue in report_fail.failures)
    assert "coarser multiplier" in next(issue.hint for issue in report_fail.failures if issue.check == "rank_deficiency")
    assert report_fail.warnings == ()

    diag = ff.contact_constraint_matrix_diagnostics(B_bad)
    report_warn = ff.assess_contact_constraint_quality(
        diag,
        zero_row_severity="warn",
        rank_deficiency_severity="warn",
    )

    assert report_warn.status == "warn"
    assert report_warn.passed
    assert {issue.check for issue in report_warn.warnings} == {"zero_rows", "rank_deficiency"}
    assert report_warn.failures == ()

    ops = ff.MultiplierContactContribution(enforcement="mortar", B=B_bad)
    assert ops.constraint_quality().status == "fail"


def test_contact_constraint_quality_optional_condition_and_row_norm_checks():
    B = np.array([[1.0, 0.0], [0.0, 1.0e-6]], dtype=float)

    report = ff.assess_contact_constraint_quality(
        B,
        max_condition_number=10.0,
        min_row_norm=1.0e-3,
    )

    assert report.status == "warn"
    assert report.passed
    assert {issue.check for issue in report.warnings} == {"condition_number", "row_norm_min"}
    assert all(issue.hint for issue in report.warnings)
    assert "block-scaled" in next(issue.hint for issue in report.warnings if issue.check == "condition_number")
    assert "overlap patches" in next(issue.hint for issue in report.warnings if issue.check == "row_norm_min")


def test_contact_constraint_quality_validates_policy_arguments():
    B = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="zero_row_severity"):
        ff.assess_contact_constraint_quality(B, zero_row_severity="error")
    with pytest.raises(ValueError, match="max_zero_rows"):
        ff.assess_contact_constraint_quality(B, max_zero_rows=-1)
    with pytest.raises(ValueError, match="max_condition_number"):
        ff.assess_contact_constraint_quality(B, max_condition_number=0.0)
    with pytest.raises(ValueError, match="min_row_norm"):
        ff.assess_contact_constraint_quality(B, min_row_norm=-1.0)


def test_algebraic_qr_mortar_selects_independent_supermesh_rows():
    master = ff.StructuredHexBox(nx=2, ny=2, nz=1, lx=1.0, ly=1.0, lz=0.25).build()
    slave = ff.StructuredHexBox(
        nx=3,
        ny=3,
        nz=1,
        lx=1.0,
        ly=1.0,
        lz=0.25,
        origin=(0.0, 0.0, -0.25),
    ).build()
    master_space = ff.make_hex_space(master, dim=3)
    slave_space = ff.make_hex_space(slave, dim=3)
    contact = ff.ContactSurfaceSpace.from_sides(
        ff.ContactSide.from_facets(master, master.facets_on_plane(axis=2, value=0.0), master_space),
        ff.ContactSide.from_facets(slave, slave.facets_on_plane(axis=2, value=0.0), slave_space),
        quad_order=2,
        backend="numpy",
        normal_sign=-1.0,
    )

    fine = contact.assemble_multiplier(
        rho=0.0,
        multiplier=ff.MultiplierSpec.from_contact(contact, family="p0_supermesh", side="master", value_dim=3),
        backend="numpy",
    )
    reduced = contact.assemble_multiplier(
        rho=0.0,
        multiplier=ff.MultiplierSpec.algebraic_qr_mortar(
            contact,
            family="p0_supermesh",
            side="master",
            value_dim=3,
        ),
        backend="numpy",
    )
    fine_diag = fine.constraint_diagnostics()
    reduced_diag = reduced.constraint_diagnostics()

    assert reduced.B.shape[0] == fine_diag.estimated_rank
    assert reduced.B.shape[0] < fine.B.shape[0]
    assert fine_diag.rank_deficiency > 0
    assert reduced_diag.rank_deficiency == 0
    assert reduced.constraint_quality().status == "pass"


def test_mortar_constraint_l2_scaling_normalizes_nonzero_rows():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
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

    unscaled = contact.assemble_multiplier(
        rho=0.0,
        multiplier=ff.MultiplierSpec.p0_mortar(contact),
        backend="numpy",
    )
    scaled = contact.assemble_multiplier(
        rho=0.0,
        multiplier=ff.MultiplierSpec.p0_mortar(contact, constraint_scaling="l2"),
        backend="numpy",
    )
    row_norms = np.linalg.norm(np.asarray(scaled.B, dtype=float), axis=1)

    assert scaled.diagnostics["constraint_scaling"] == "l2"
    assert unscaled.diagnostics["constraint_scaling"] == "none"
    np.testing.assert_allclose(row_norms[row_norms > 0.0], np.ones(np.count_nonzero(row_norms > 0.0)))


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

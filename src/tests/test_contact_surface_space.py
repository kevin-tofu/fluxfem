"""ContactSurfaceSpace bilinear wrapper matches mixed surface assembly."""
import numpy as np
import jax.numpy as jnp
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import einsum as wf_einsum
from fluxfem.mesh.contact import (
    compile_tagged_pair_nitsche_penalty_residual,
    make_tagged_pair_nitsche_penalty_bilinear,
)
from fluxfem.mesh.contact_interface import _build_mixed_surface_context


def _penalty_bilinear(v1, v2, u1, u2, p):
    ju = u1.val - u2.val
    term = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
    return term * h_wf.ds()


def _direct_trial_layout_residual(ctx, u_elem, p):
    ua = jnp.asarray(u_elem["a"])
    ub = jnp.asarray(u_elem["b"])
    scale = p.alpha * p.inv_h
    n_q = int(ctx.x_q.shape[0])
    return {
        "a": scale * jnp.broadcast_to(ua - ub, (n_q, ua.shape[0])),
        "b": scale * jnp.broadcast_to(ub - ua, (n_q, ub.shape[0])),
    }


def _make_contact(coords, facets, conn, *, value_dim=3, quad_order=1):
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)
    return ff.ContactSurfaceSpace.from_surfaces(
        surf_a,
        surf_b,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=value_dim,
        value_dim_slave=value_dim,
        quad_order=quad_order,
    )


def _make_one_to_many_contact(coords, facets, conn, *, value_dim=3, quad_order=1):
    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s = ff.SurfaceMesh.from_facets(coords, facets)
    return ff.OneToManyContactSurfaceSpace.from_surfaces(
        surf_m,
        [surf_s],
        elem_conn_master=conn,
        elem_conn_slaves=[conn],
        value_dim_master=value_dim,
        value_dim_slave=value_dim,
        quad_order=quad_order,
    )


def _penalty_params():
    return ff.Params(alpha=10.0, inv_h=1.0)


def _tagged_fastpath_params():
    return ff.Params(alpha=10.0, inv_h=1.0, lam=1.0, mu=1.0)


def test_contact_surface_bilinear_wrapper():
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
    contact = _make_contact(coords, facets, conn)

    def res_a(v, u, p):
        u2 = ff.unknown_ref("b")
        ju = u.val - u2.val
        return (p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    def res_b(v, u, p):
        u1 = ff.unknown_ref("a")
        ju = u1.val - u.val
        return -(p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    res_form = ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = _penalty_params()

    J_bilin = contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params)
    J_res = contact.assemble_jacobian(res_form, {"a": u_a, "b": u_b}, params)

    assert np.allclose(np.asarray(J_bilin), np.asarray(J_res), atol=1e-10)


def test_contact_surface_compile_bilinear_returns_reusable_residual_form():
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
    contact = _make_contact(coords, facets, conn)

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = _penalty_params()

    compiled = contact.compile_bilinear(_penalty_bilinear)
    J_compiled = contact.assemble_jacobian(compiled, {"a": u_a, "b": u_b}, params)
    J_direct = contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params)

    assert np.allclose(np.asarray(J_compiled), np.asarray(J_direct), atol=1e-10)


def test_contact_surface_assemble_bilinear_form_accepts_compiled_bilinear():
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
    contact = _make_contact(coords, facets, conn)
    params = _penalty_params()

    compiled = contact.compile_bilinear(_penalty_bilinear)
    J_compiled = contact.assemble_bilinear_form(compiled, params, sparse=False)
    J_direct = contact.assemble_bilinear_form(_penalty_bilinear, params, sparse=False)

    assert np.allclose(np.asarray(J_compiled), np.asarray(J_direct), atol=1e-10)


def test_contact_surface_assemble_bilinear_form_accepts_bilinearform_contact_compiled():
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
    contact = _make_contact(coords, facets, conn)
    params = _penalty_params()

    compiled = ff.BilinearForm.contact(_penalty_bilinear).get_compiled()
    J_compiled = contact.assemble_bilinear_form(compiled, params, sparse=False)
    J_direct = contact.assemble_bilinear_form(_penalty_bilinear, params, sparse=False)

    assert np.allclose(np.asarray(J_compiled), np.asarray(J_direct), atol=1e-10)


def test_contact_surface_compile_bilinear_matches_bilinearform_contact_compiled():
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
    contact = _make_contact(coords, facets, conn)
    params = _penalty_params()

    compiled_from_contact = contact.compile_bilinear(_penalty_bilinear)
    compiled_from_form = ff.BilinearForm.contact(_penalty_bilinear).get_compiled()

    J_contact = contact.assemble_bilinear_form(compiled_from_contact, params, sparse=False)
    J_form = contact.assemble_bilinear_form(compiled_from_form, params, sparse=False)

    assert np.allclose(np.asarray(J_contact), np.asarray(J_form), atol=1e-10)


def test_contact_surface_role_compiled_space_aliases_propagate_to_residual_form():
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
    contact = _make_contact(coords, facets, conn)

    compiled = ff.BilinearForm.contact(_penalty_bilinear).get_compiled()
    compiled._ff_contact_test_space_by_role = {"a": "Va", "b": "Vb"}
    compiled._ff_contact_unknown_space_by_role = {"a": "Ua", "b": "Ub"}

    res_form = contact.compile_bilinear(compiled, use_cache=False)

    assert getattr(res_form, "_test_space_by_target", None) == {"a": "Va", "b": "Vb"}
    assert getattr(res_form, "_unknown_space_by_target", None) == {"a": "Ua", "b": "Ub"}


def test_mixed_surface_context_uses_distinct_test_and_trial_field_objects():
    ctx = _build_mixed_surface_context(
        field_a="a",
        field_b="b",
        test_space_key_a="Va",
        test_space_key_b="Vb",
        unknown_space_key_a="Ua",
        unknown_space_key_b="Ub",
        test_Na=np.ones((1, 3), dtype=float),
        test_Nb=np.ones((1, 3), dtype=float),
        trial_Na=np.ones((1, 3), dtype=float),
        trial_Nb=np.ones((1, 3), dtype=float),
        test_gradNa=np.ones((1, 3, 3), dtype=float),
        test_gradNb=np.ones((1, 3, 3), dtype=float),
        trial_gradNa=np.ones((1, 3, 3), dtype=float),
        trial_gradNb=np.ones((1, 3, 3), dtype=float),
        test_value_dim_a=3,
        test_value_dim_b=3,
        trial_value_dim_a=3,
        trial_value_dim_b=3,
        x_q=np.zeros((1, 3), dtype=float),
        w=np.ones((1,), dtype=float),
        detJ=np.ones((1,), dtype=float),
        normal_q=np.array([[0.0, 0.0, 1.0]], dtype=float),
    )

    pair_a = ctx.bindings["a"]
    pair_b = ctx.bindings["b"]

    assert pair_a.test is not pair_a.trial
    assert pair_a.unknown is pair_a.trial
    assert pair_b.test is not pair_b.trial
    assert pair_b.unknown is pair_b.trial


def test_mixed_surface_context_accepts_distinct_test_and_trial_basis_tables():
    test_Na = np.array([[1.0, 2.0, 3.0]], dtype=float)
    trial_Na = np.array([[4.0, 5.0, 6.0]], dtype=float)
    ctx = _build_mixed_surface_context(
        field_a="a",
        field_b="b",
        test_space_key_a="Va",
        test_space_key_b="Vb",
        unknown_space_key_a="Ua",
        unknown_space_key_b="Ub",
        test_Na=test_Na,
        test_Nb=np.ones((1, 3), dtype=float),
        trial_Na=trial_Na,
        trial_Nb=np.ones((1, 3), dtype=float),
        test_gradNa=np.ones((1, 3, 3), dtype=float),
        test_gradNb=np.ones((1, 3, 3), dtype=float),
        trial_gradNa=np.full((1, 3, 3), 2.0, dtype=float),
        trial_gradNb=np.ones((1, 3, 3), dtype=float),
        test_value_dim_a=3,
        test_value_dim_b=3,
        trial_value_dim_a=3,
        trial_value_dim_b=3,
        x_q=np.zeros((1, 3), dtype=float),
        w=np.ones((1,), dtype=float),
        detJ=np.ones((1,), dtype=float),
        normal_q=np.array([[0.0, 0.0, 1.0]], dtype=float),
    )

    pair_a = ctx.bindings["a"]
    assert np.allclose(pair_a.test.N, test_Na)
    assert np.allclose(pair_a.trial.N, trial_Na)
    assert np.allclose(pair_a.unknown.gradN, np.full((1, 3, 3), 2.0, dtype=float))


def test_contact_surface_distinct_trial_p0_layout_permutes_columns():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    facets = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)
    trial_facet_dofs = np.array(
        [
            [3, 4, 5],
            [0, 1, 2],
        ],
        dtype=int,
    )

    contact_base = ff.ContactSurfaceSpace.from_surfaces(
        ff.SurfaceMesh.from_facets(coords, facets),
        ff.SurfaceMesh.from_facets(coords, facets),
        value_dim_master=3,
        value_dim_slave=3,
        space_mode_master="p0",
        space_mode_slave="p0",
        quad_order=1,
    )
    contact_trial = ff.ContactSurfaceSpace.from_surfaces(
        ff.SurfaceMesh.from_facets(coords, facets),
        ff.SurfaceMesh.from_facets(coords, facets),
        value_dim_master=3,
        value_dim_slave=3,
        space_mode_master="p0",
        space_mode_slave="p0",
        trial_space_mode_master="p0",
        trial_space_mode_slave="p0",
        trial_facet_dofs_master=trial_facet_dofs,
        trial_facet_dofs_slave=trial_facet_dofs,
        quad_order=1,
    )

    u_a = np.zeros(6, dtype=float)
    u_b = np.zeros(6, dtype=float)
    params = _penalty_params()
    base = np.asarray(contact_base.assemble_jacobian(_direct_trial_layout_residual, {"a": u_a, "b": u_b}, params))
    trial = np.asarray(contact_trial.assemble_jacobian(_direct_trial_layout_residual, {"a": u_a, "b": u_b}, params))

    perm_master = np.array([3, 4, 5, 0, 1, 2], dtype=int)
    perm_slave = 6 + perm_master
    perm = np.concatenate([perm_master, perm_slave], axis=0)
    expected = np.zeros_like(base)
    expected[:, perm] = base

    assert np.allclose(trial, expected, atol=1e-10)

def test_contact_surface_assemble_bilinear_form_matches_zero_state_bilinear():
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
    contact = _make_contact(coords, facets, conn)

    u_a = np.zeros(coords.shape[0] * 3, dtype=float)
    u_b = np.zeros(coords.shape[0] * 3, dtype=float)
    params = _penalty_params()

    J_form = contact.assemble_bilinear_form(_penalty_bilinear, params, sparse=False)
    J_state = contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params, sparse=False)

    assert np.allclose(np.asarray(J_form), np.asarray(J_state), atol=1e-10)


def test_one_to_many_contact_surface_assemble_bilinear_form_matches_zero_state_bilinear():
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
    contact = _make_one_to_many_contact(coords, facets, conn)

    u_m = np.zeros(coords.shape[0] * 3, dtype=float)
    u_s = [np.zeros(coords.shape[0] * 3, dtype=float)]
    params = _penalty_params()

    J_form = contact.assemble_bilinear_form(_penalty_bilinear, params, sparse=False)
    J_state = contact.assemble_bilinear(_penalty_bilinear, u_m, u_s, params, sparse=False)

    assert np.allclose(np.asarray(J_form), np.asarray(J_state), atol=1e-10)


def test_one_to_many_contact_surface_assemble_bilinear_form_accepts_compiled_bilinear():
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
    contact = _make_one_to_many_contact(coords, facets, conn)
    params = _penalty_params()

    compiled = contact.compile_bilinear(_penalty_bilinear)
    J_compiled = contact.assemble_bilinear_form(compiled, params, sparse=False)
    J_direct = contact.assemble_bilinear_form(_penalty_bilinear, params, sparse=False)

    assert np.allclose(np.asarray(J_compiled), np.asarray(J_direct), atol=1e-10)


def test_one_to_many_contact_surface_compile_bilinear_matches_bilinearform_contact_compiled():
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
    contact = _make_one_to_many_contact(coords, facets, conn)
    params = _penalty_params()

    compiled_from_contact = contact.compile_bilinear(_penalty_bilinear)
    compiled_from_form = ff.BilinearForm.contact(_penalty_bilinear).get_compiled()

    J_contact = contact.assemble_bilinear_form(compiled_from_contact, params, sparse=False)
    J_form = contact.assemble_bilinear_form(compiled_from_form, params, sparse=False)

    assert np.allclose(np.asarray(J_contact), np.asarray(J_form), atol=1e-10)


def test_one_to_many_contact_surface_role_compiled_space_aliases_propagate_to_residual_form():
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
    contact = _make_one_to_many_contact(coords, facets, conn)

    compiled = ff.BilinearForm.contact(_penalty_bilinear).get_compiled()
    compiled._ff_contact_test_space_by_role = {"a": "Vm", "b": "Vs"}
    compiled._ff_contact_unknown_space_by_role = {"a": "Um", "b": "Us"}

    res_form = contact.compile_bilinear(compiled, use_cache=False)

    assert getattr(res_form, "_test_space_by_target", None) == {"master": "Vm", "slave": "Vs"}
    assert getattr(res_form, "_unknown_space_by_target", None) == {"master": "Um", "slave": "Us"}


def test_contact_surface_assemble_bilinear_reuses_compiled_form(monkeypatch):
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
    contact = _make_contact(coords, facets, conn)
    params = _penalty_params()
    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)

    calls = {"n": 0}
    orig = ff.compile_mixed_surface_residual

    def _count_compile(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr("fluxfem.core.weakform.compile_mixed_surface_residual", _count_compile)

    _ = contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params)
    _ = contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params)

    assert calls["n"] == 1


def test_one_to_many_contact_surface_assemble_bilinear_reuses_compiled_form(monkeypatch):
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
    contact = _make_one_to_many_contact(coords, facets, conn)
    params = _penalty_params()
    u_m = jnp.zeros(coords.shape[0] * 3)
    u_s = [jnp.zeros(coords.shape[0] * 3)]

    calls = {"n": 0}
    orig = ff.compile_mixed_surface_residual

    def _count_compile(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr("fluxfem.core.weakform.compile_mixed_surface_residual", _count_compile)

    _ = contact.assemble_bilinear(_penalty_bilinear, u_m, u_s, params)
    _ = contact.assemble_bilinear(_penalty_bilinear, u_m, u_s, params)

    assert calls["n"] == 1


def test_contact_surface_bilinear_tag_is_propagated_to_compiled_form():
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
    contact = _make_contact(coords, facets, conn)

    tagged = make_tagged_pair_nitsche_penalty_bilinear(_penalty_bilinear)
    assert getattr(tagged, "_ff_contact_formulation", None) == "pair_nitsche_penalty"
    assert getattr(tagged, "_ff_contact_backend_fastpath", None) == "numpy_local_kernel"

    captured = {}
    orig = contact.assemble_jacobian

    def _capture(res_form, u, params, **kwargs):
        captured["formulation"] = getattr(res_form, "_ff_contact_formulation", None)
        captured["fastpath"] = getattr(res_form, "_ff_contact_backend_fastpath", None)
        return orig(res_form, u, params, **kwargs)

    contact.assemble_jacobian = _capture  # type: ignore[method-assign]
    try:
        u_a = jnp.zeros(coords.shape[0] * 3)
        u_b = jnp.zeros(coords.shape[0] * 3)
        _ = contact.assemble_bilinear(tagged, u_a, u_b, _penalty_params())
    finally:
        contact.assemble_jacobian = orig  # type: ignore[method-assign]

    assert captured["formulation"] == "pair_nitsche_penalty"
    assert captured["fastpath"] == "numpy_local_kernel"


def test_contact_surface_tagged_numpy_fastpath_skips_mixed_surface_compile(monkeypatch):
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
    contact = _make_contact(coords, facets, conn)
    contact.backend = "numpy"

    def _fail_compile(*_args, **_kwargs):
        raise AssertionError("compile_mixed_surface_residual_numpy should not run for tagged NumPy fast path")

    monkeypatch.setattr("fluxfem.core.weakform.compile_mixed_surface_residual_numpy", _fail_compile)
    tagged = make_tagged_pair_nitsche_penalty_bilinear(_penalty_bilinear)
    u_a = np.zeros(coords.shape[0] * 3, dtype=float)
    u_b = np.zeros(coords.shape[0] * 3, dtype=float)

    jac = contact.assemble_bilinear(tagged, u_a, u_b, _tagged_fastpath_params(), sparse=True)
    assert hasattr(jac, "to_dense")


def test_contact_surface_tagged_numpy_fastpath_skips_expression_build(monkeypatch):
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
    contact = _make_contact(coords, facets, conn)
    contact.backend = "numpy"

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("Tagged NumPy fast path should skip weak-form expression construction")

    tagged = make_tagged_pair_nitsche_penalty_bilinear(_should_not_run)
    u_a = np.zeros(coords.shape[0] * 3, dtype=float)
    u_b = np.zeros(coords.shape[0] * 3, dtype=float)

    jac = contact.assemble_bilinear(tagged, u_a, u_b, _tagged_fastpath_params(), sparse=True)
    assert hasattr(jac, "to_dense")


def test_contact_surface_tagged_residual_form_enables_direct_numpy_fastpath(monkeypatch):
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
    contact = _make_contact(coords, facets, conn)
    contact.backend = "numpy"

    def res_a(v, u, p):
        u2 = ff.unknown_ref("b")
        ju = u.val - u2.val
        return (p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    def res_b(v, u, p):
        u1 = ff.unknown_ref("a")
        ju = u1.val - u.val
        return -(p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    tagged = compile_tagged_pair_nitsche_penalty_residual(
        {"a": res_a, "b": res_b},
        backend="numpy",
    )

    def _fail_generic(*_args, **_kwargs):
        raise AssertionError("_compute_mixed_surface_local_jacobian should not run for tagged direct fast path")

    monkeypatch.setattr(
        "fluxfem.mesh.contact_interface._compute_mixed_surface_local_jacobian",
        _fail_generic,
    )

    u = {"a": np.zeros(coords.shape[0] * 3, dtype=float), "b": np.zeros(coords.shape[0] * 3, dtype=float)}
    jac = contact.assemble_jacobian(
        tagged,
        u,
        _tagged_fastpath_params(),
        backend="numpy",
        sparse=True,
    )

    assert hasattr(jac, "to_dense")


def test_contact_surface_tagged_jax_batch_fastpath_skips_jacrev(monkeypatch):
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
    contact = _make_contact(coords, facets, conn)
    contact.backend = "jax"
    contact.batch_jac = True

    def res_a(v, u, p):
        n = h_wf.normal()
        u_b = ff.unknown_ref("b", space="B")
        ju = u.val - u_b.val
        t_u = 0.5 * (h_wf.traction(u, n, p) + h_wf.traction(u_b, n, p))
        t_v = h_wf.traction(v, n, p)
        penalty = (p.alpha * p.inv_h) * h_wf.dot(v, ju)
        traction = -h_wf.dot(v, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v, ju)
        return (penalty + traction) * h_wf.ds()

    def res_b(v, u, p):
        n = h_wf.normal()
        u_a = ff.unknown_ref("a", space="A")
        ju = u_a.val - u.val
        t_u = 0.5 * (h_wf.traction(u_a, n, p) + h_wf.traction(u, n, p))
        t_v = h_wf.traction(v, n, p)
        penalty = -(p.alpha * p.inv_h) * h_wf.dot(v, ju)
        traction = h_wf.dot(v, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v, ju)
        return (penalty + traction) * h_wf.ds()

    tagged = compile_tagged_pair_nitsche_penalty_residual(
        {
            "a": ff.bind_mixed_residual("a", res_a, space="A"),
            "b": ff.bind_mixed_residual("b", res_b, space="B"),
        },
        backend="jax",
    )

    def _fail_jacrev(*_args, **_kwargs):
        raise AssertionError("tagged JAX batch fast path should skip jax.jacrev")

    monkeypatch.setattr("fluxfem.mesh.contact_interface.jax.jacrev", _fail_jacrev)

    u = {"a": jnp.zeros(coords.shape[0] * 3), "b": jnp.zeros(coords.shape[0] * 3)}
    jac = contact.assemble_jacobian(
        tagged,
        u,
        ff.Params(alpha=10.0, inv_h=1.0, lam=1.0, mu=1.0, use_penalty=1.0, use_traction=1.0),
        backend="jax",
        sparse=True,
        batch_jac=True,
    )

    assert hasattr(jac, "to_dense")


def test_contact_surface_jacobian_uses_precomputed_supermesh_geometry(monkeypatch):
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
    contact = _make_contact(coords, facets, conn)

    def _fail_prepare(*_args, **_kwargs):
        raise AssertionError("precomputed supermesh geometry should bypass per-triangle geometry preparation")

    monkeypatch.setattr(
        "fluxfem.mesh.contact_interface._prepare_supermesh_jacobian_triangle_geometry",
        _fail_prepare,
    )

    def res_a(v, u, p):
        u2 = ff.unknown_ref("b")
        ju = u.val - u2.val
        return (p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    def res_b(v, u, p):
        u1 = ff.unknown_ref("a")
        ju = u1.val - u.val
        return -(p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    res_form = ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})
    u = {"a": jnp.zeros(coords.shape[0] * 3), "b": jnp.zeros(coords.shape[0] * 3)}
    jac = contact.assemble_jacobian(res_form, u, _penalty_params(), sparse=True, backend="jax")

    assert hasattr(jac, "to_dense")


def test_contact_surface_bilinear_tet10_mid_edge_dofs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [0.5, 0.0, 0.0],  # 4 (0-1)
            [0.5, 0.5, 0.0],  # 5 (1-2)
            [0.0, 0.5, 0.0],  # 6 (0-2)
            [0.0, 0.0, 0.5],  # 7 (0-3)
            [0.5, 0.0, 0.5],  # 8 (1-3)
            [0.0, 0.5, 0.5],  # 9 (2-3)
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=int)
    facets = np.array([[0, 1, 2]], dtype=int)
    contact = _make_contact(coords, facets, conn)

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = _penalty_params()

    J = np.asarray(contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params))
    n_dofs = coords.shape[0] * 3
    assert J.shape == (2 * n_dofs, 2 * n_dofs)
    # Edge-midpoint nodes on the face (4-6) should contribute.
    mid_edge_slice = slice(4 * 3, 7 * 3)  # nodes 4,5,6
    assert np.max(np.abs(J[:n_dofs, :n_dofs][mid_edge_slice, :])) > 0.0


def test_contact_surface_bilinear_hex8_face_dofs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    facets = np.array([[0, 1, 2, 3]], dtype=int)
    contact = _make_contact(coords, facets, conn)

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = _penalty_params()

    J = np.asarray(contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params))
    n_dofs = coords.shape[0] * 3
    assert J.shape == (2 * n_dofs, 2 * n_dofs)
    # Top face nodes (4-7) should not contribute on the z=0 interface.
    top_slice = slice(4 * 3, 8 * 3)
    assert np.max(np.abs(J[:n_dofs, :n_dofs][top_slice, :])) < 1e-8


def test_contact_surface_bilinear_hex20_edge_dofs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            [0.5, 0.0, 0.0],  # 8 (0-1)
            [1.0, 0.5, 0.0],  # 9 (1-2)
            [0.5, 1.0, 0.0],  # 10 (2-3)
            [0.0, 0.5, 0.0],  # 11 (3-0)
            [0.5, 0.0, 1.0],  # 12 (4-5)
            [1.0, 0.5, 1.0],  # 13 (5-6)
            [0.5, 1.0, 1.0],  # 14 (6-7)
            [0.0, 0.5, 1.0],  # 15 (7-4)
            [0.0, 0.0, 0.5],  # 16 (0-4)
            [1.0, 0.0, 0.5],  # 17 (1-5)
            [1.0, 1.0, 0.5],  # 18 (2-6)
            [0.0, 1.0, 0.5],  # 19 (3-7)
        ],
        dtype=float,
    )
    conn = np.array([[i for i in range(20)]], dtype=int)
    facets = np.array([[0, 1, 2, 3]], dtype=int)
    contact = _make_contact(coords, facets, conn)

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = _penalty_params()

    J = np.asarray(contact.assemble_bilinear(_penalty_bilinear, u_a, u_b, params))
    n_dofs = coords.shape[0] * 3
    assert J.shape == (2 * n_dofs, 2 * n_dofs)
    edge_slice = slice(8 * 3, 12 * 3)  # bottom face edge mids
    assert np.max(np.abs(J[:n_dofs, :n_dofs][edge_slice, :])) > 0.0

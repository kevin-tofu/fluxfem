"""Mixed weak-form assembly tests."""
import numpy as np
import jax.numpy as jnp
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.mixed_weakform import (
    MixedResidualForm,
    assemble_mixed_jacobian_wf,
    assemble_mixed_residual_wf,
)


def _make_mixed_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = ff.MixedSpaces(
        {
            "u": ff.NamedSpace("u", space),
            "p": ff.NamedSpace("p", space),
        }
    ).to_fe_space()
    return mixed


def _mixed_residuals():
    def res_u(v, u, p):
        p_ref = ff.unknown_ref("p")
        return (v * (u.val + p.alpha * p_ref.val)) * h_wf.dOmega()

    def res_p(q, p_field, p):
        u_ref = ff.unknown_ref("u")
        return (q * (p_field.val + p.beta * u_ref.val)) * h_wf.dOmega()

    return {"u": res_u, "p": res_p}


def test_mixed_residual_consistency():
    """Mixed weak-form residual is consistent across wrapper types."""
    mixed = _make_mixed_space()
    params = {"alpha": 1.5, "beta": -0.3}
    rng = np.random.default_rng(0)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    residuals = _mixed_residuals()
    R_dict = assemble_mixed_residual_wf(mixed, residuals, u_vec, params)
    R_form = assemble_mixed_residual_wf(mixed, MixedResidualForm(residuals), u_vec, params)
    R_compiled = assemble_mixed_residual_wf(
        mixed, ff.MixedWeakForm(residuals=residuals), u_vec, params
    )

    assert np.allclose(np.asarray(R_dict), np.asarray(R_form))
    assert np.allclose(np.asarray(R_dict), np.asarray(R_compiled))


def test_mixed_jacobian_consistency():
    """Mixed weak-form Jacobian is consistent across wrapper types."""
    mixed = _make_mixed_space()
    params = {"alpha": 0.7, "beta": 1.2}
    rng = np.random.default_rng(1)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    residuals = _mixed_residuals()
    J_dict = assemble_mixed_jacobian_wf(mixed, residuals, u_vec, params)
    J_form = assemble_mixed_jacobian_wf(
        mixed, MixedResidualForm(residuals), u_vec, params
    )
    J_compiled = assemble_mixed_jacobian_wf(
        mixed, ff.MixedWeakForm(residuals=residuals), u_vec, params
    )

    assert np.allclose(np.asarray(J_dict.to_dense()), np.asarray(J_form.to_dense()), atol=1e-6)
    assert np.allclose(np.asarray(J_dict.to_dense()), np.asarray(J_compiled.to_dense()), atol=1e-6)


def test_mixed_residual_form_wrapper_matches_compile_then_assemble_style():
    mixed = _make_mixed_space()
    params = {"alpha": 1.2, "beta": -0.7}
    rng = np.random.default_rng(11)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    residuals = _mixed_residuals()
    compiled = ff.ResidualForm.mixed(residuals).get_compiled()

    direct = mixed.assemble_residual(compiled, u_vec, params)
    wrapped = mixed.assemble_residual(ff.ResidualForm.mixed(residuals), u_vec, params)

    assert np.allclose(np.asarray(direct), np.asarray(wrapped))


def test_mixed_jacobian_form_wrapper_matches_compile_then_assemble_style():
    mixed = _make_mixed_space()
    params = {"alpha": -0.2, "beta": 0.4}
    rng = np.random.default_rng(12)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    residuals = _mixed_residuals()
    compiled = ff.ResidualForm.mixed(residuals).get_compiled()

    direct = mixed.assemble_jacobian(compiled, u_vec, params)
    wrapped = mixed.assemble_jacobian(ff.ResidualForm.mixed(residuals), u_vec, params)

    assert np.allclose(np.asarray(direct.to_dense()), np.asarray(wrapped.to_dense()), atol=1e-6)


def test_mixed_residual_matches_single_field():
    """Mixed residual for one field matches single-field assembly."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = ff.MixedSpaces(
        {
            "u": ff.NamedSpace("u", space),
            "t": ff.NamedSpace("t", space),
        }
    ).to_fe_space()

    def res_u(v, u, p):
        return p.kappa * h_wf.gaction(v, h_wf.grad(u)) * h_wf.dOmega()

    def res_t(v, _u, _p):
        return (v * 0.0) * h_wf.dOmega()

    params = ff.Params(kappa=2.0)
    residuals = {"u": res_u, "t": res_t}

    rng = np.random.default_rng(2)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))
    u_field = u_vec[mixed.field_slices["u"]]

    mixed_form = MixedResidualForm(residuals)
    R_mixed = assemble_mixed_residual_wf(mixed, mixed_form, u_vec, params)
    R_u = np.asarray(R_mixed[mixed.field_slices["u"]])

    single_form = ff.ResidualForm.volume(res_u)
    R_single = space.assemble_residual(single_form.get_compiled(), u_field, params)

    assert np.allclose(R_u, np.asarray(R_single))


def test_mixed_residual_explicit_space_refs_match_default_resolution():
    """Explicit space= refs resolve through ctx.spaces and match default mixed refs."""
    mixed = _make_mixed_space()
    params = {"alpha": 0.25, "beta": -0.5}
    rng = np.random.default_rng(3)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    residuals_default = _mixed_residuals()

    def res_u(v, u, p):
        p_ref = ff.unknown_ref("pressure", space="p")
        return (v * (u.val + p.alpha * p_ref.val)) * h_wf.dOmega()

    def res_p(q, p_field, p):
        u_ref = ff.unknown_ref("disp", space="u")
        return (q * (p_field.val + p.beta * u_ref.val)) * h_wf.dOmega()

    residuals_space = {"u": res_u, "p": res_p}

    r_default = assemble_mixed_residual_wf(mixed, residuals_default, u_vec, params)
    r_space = assemble_mixed_residual_wf(mixed, residuals_space, u_vec, params)

    assert np.allclose(np.asarray(r_default), np.asarray(r_space))


def test_mixed_space_key_mapping_decouples_field_name_from_space_key():
    """Field bindings can keep field names while resolving refs through distinct space keys."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = ff.MixedSpaces(
        {
            "disp": ff.NamedSpace("V", space),
            "press": ff.NamedSpace("Q", space),
        }
    ).to_fe_space()

    params = {"alpha": 0.5, "beta": -0.2}
    rng = np.random.default_rng(4)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    def res_disp(v, u, p):
        press = ff.unknown_ref("pressure", space="Q")
        return (v * (u.val + p.alpha * press.val)) * h_wf.dOmega()

    def res_press(q, press, p):
        disp = ff.unknown_ref("displacement", space="V")
        return (q * (press.val + p.beta * disp.val)) * h_wf.dOmega()

    residuals = {"disp": res_disp, "press": res_press}
    result = assemble_mixed_residual_wf(mixed, residuals, u_vec, params)

    assert np.asarray(result).shape == (mixed.n_dofs,)


def test_mixed_residual_binding_decouples_residual_label_from_target_field():
    """Residual labels can differ from assembled target fields via explicit bindings."""
    mixed = _make_mixed_space()
    params = {"alpha": 1.0, "beta": -0.25}
    rng = np.random.default_rng(5)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    default = _mixed_residuals()
    bound = ff.make_mixed_residuals(
        momentum=ff.bind_mixed_residual("u", default["u"], space="u"),
        continuity=ff.bind_mixed_residual("p", default["p"], space="p"),
    )

    r_default = assemble_mixed_residual_wf(mixed, default, u_vec, params)
    r_bound = assemble_mixed_residual_wf(mixed, bound, u_vec, params)

    assert np.allclose(np.asarray(r_default), np.asarray(r_bound))


def test_mixed_form_context_uses_bindings():
    """Mixed contexts expose bindings directly."""
    mixed = _make_mixed_space()
    ctx = mixed.build_form_contexts()
    assert "u" in ctx.bindings
    assert not hasattr(ctx, "fields")


def test_mixed_spaces_builds_mixed_fespace_with_named_space_keys():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    spec = ff.MixedSpaces(
        {
            "disp": ff.NamedSpace("V", scalar),
            "press": ff.NamedSpace("Q", scalar),
        }
    )
    mixed = spec.to_fe_space()

    assert mixed.field_names == ("disp", "press")
    assert mixed.space_key_by_field["disp"] == "V"
    assert mixed.space_key_by_field["press"] == "Q"


def test_mixed_spaces_support_explicit_space_refs():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = ff.MixedSpaces(
        {
            "disp": ff.NamedSpace("V", scalar),
            "press": ff.NamedSpace("Q", scalar),
        }
    ).to_fe_space()

    params = {"alpha": 0.5, "beta": -0.2}
    rng = np.random.default_rng(6)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    def res_disp(v, u, p):
        press = ff.unknown_ref("pressure", space="Q")
        return (v * (u.val + p.alpha * press.val)) * h_wf.dOmega()

    def res_press(q, press, p):
        disp = ff.unknown_ref("displacement", space="V")
        return (q * (press.val + p.beta * disp.val)) * h_wf.dOmega()

    result = assemble_mixed_residual_wf(mixed, {"disp": res_disp, "press": res_press}, u_vec, params)
    assert np.asarray(result).shape == (mixed.n_dofs,)


def test_mixed_spaces_accept_residual_spaces_with_role_aliases():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = ff.MixedSpaces(
        {
            "disp": ff.ResidualSpaces(
                test=ff.NamedSpace("V", scalar),
                unknown=ff.NamedSpace("U", scalar),
            ),
            "press": ff.ResidualSpaces(
                test=ff.NamedSpace("Qv", scalar),
                unknown=ff.NamedSpace("Q", scalar),
            ),
        }
    ).to_fe_space()

    ctx = mixed.build_form_contexts()
    assert mixed.space_key_by_field["disp"] == "U"
    assert mixed.space_key_by_field["press"] == "Q"
    assert "V" in ctx.spaces
    assert "U" in ctx.spaces
    assert "Qv" in ctx.spaces
    assert "Q" in ctx.spaces

    params = {"alpha": 0.5, "beta": -0.2}
    rng = np.random.default_rng(7)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    def res_disp(v, u, p):
        press = ff.unknown_ref("pressure", space="Q")
        return (v * (u.val + p.alpha * press.val)) * h_wf.dOmega()

    def res_press(q, press, p):
        disp = ff.unknown_ref("displacement", space="V")
        return (q * (press.val + p.beta * disp.val)) * h_wf.dOmega()

    result = assemble_mixed_residual_wf(mixed, {"disp": res_disp, "press": res_press}, u_vec, params)
    assert np.asarray(result).shape == (mixed.n_dofs,)


def test_mixed_spaces_reject_distinct_residual_spaces_per_field():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar_a = ff.make_hex_space(mesh, dim=1, intorder=2)
    scalar_b = ff.make_hex_space(mesh, dim=1, intorder=2)

    with pytest.raises(ValueError, match="one unknown vector and one residual block per field"):
        ff.MixedSpaces(
            {
                "disp": ff.ResidualSpaces(
                    test=ff.NamedSpace("V", scalar_a),
                    unknown=ff.NamedSpace("U", scalar_b),
                ),
            }
        )


def test_mixed_role_spaces_preserve_distinct_role_layouts():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    with pytest.warns(UserWarning, match="experimental"):
        mixed = ff.MixedRoleSpaces(
            {
                "u": ff.ResidualSpaces(
                    test=ff.NamedSpace("V", vector),
                    unknown=ff.NamedSpace("U", scalar),
                ),
            }
        ).to_fe_space()

    assert mixed.n_residual_dofs == vector.n_dofs
    assert mixed.n_unknown_dofs == scalar.n_dofs
    assert mixed.n_residual_ldofs == vector.n_ldofs
    assert mixed.n_unknown_ldofs == scalar.n_ldofs

    ctx = mixed.build_form_contexts()
    assert ctx.bindings["u"].test.value_dim == 3
    assert not hasattr(ctx.bindings["u"].unknown, "value_dim")
    assert "V" in ctx.spaces
    assert "U" in ctx.spaces


def test_mixed_role_spaces_assemble_residual_with_distinct_role_layouts():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    u = {"u": jnp.linspace(0.0, 1.0, mixed.n_unknown_dofs, dtype=jnp.float32)}

    def residual(ctx, u_elem, _params):
        _ = u_elem
        return {
            "u": jnp.ones((ctx.w.shape[0], mixed.n_residual_ldofs), dtype=jnp.float32),
        }

    F = mixed.assemble_residual(residual, u, params=None)

    assert np.asarray(F).shape == (mixed.n_residual_dofs,)


def test_mixed_role_spaces_assemble_jacobian_with_distinct_role_layouts():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    u = {"u": jnp.linspace(0.0, 1.0, mixed.n_unknown_dofs, dtype=jnp.float32)}

    def residual(_ctx, u_elem, _params):
        q = 8
        u_local = u_elem["u"]
        return {
            "u": jnp.broadcast_to(jnp.repeat(u_local[:, None], 3, axis=0).T, (q, mixed.n_residual_ldofs)),
        }

    J = mixed.assemble_jacobian(residual, u, params=None)

    assert J.shape == (mixed.n_residual_dofs, mixed.n_unknown_dofs)


def test_mixed_role_spaces_assemble_compiled_residual_with_space_binding():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    u = {"u": jnp.linspace(0.0, 1.0, mixed.n_unknown_dofs, dtype=jnp.float32)}

    def res_u(v, u_field, _p):
        _ = u_field
        return h_wf.dot(v, (1.0, 0.0, 0.0)) * h_wf.dOmega()

    residual = ff.compile_mixed_residual(
        ff.make_mixed_residuals(u=ff.bind_mixed_residual("u", res_u, space="V"))
    )
    F = mixed.assemble_residual(residual, u, params=None)

    assert np.asarray(F).shape == (mixed.n_residual_dofs,)


def test_mixed_role_spaces_assemble_compiled_jacobian_with_space_binding():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    u = {"u": jnp.linspace(0.0, 1.0, mixed.n_unknown_dofs, dtype=jnp.float32)}

    def res_u(v, u_field, p):
        return (1.0 + p.alpha * u_field.val) * v.val * h_wf.dOmega()

    residual = ff.compile_mixed_residual(
        ff.make_mixed_residuals(u=ff.bind_mixed_residual("u", res_u, space="V"))
    )
    J = mixed.assemble_jacobian(residual, u, params=ff.Params(alpha=0.5))

    assert J.shape == (mixed.n_residual_dofs, mixed.n_unknown_dofs)


def test_mixed_role_spaces_reject_compiled_surface_residuals():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    u = {"u": jnp.linspace(0.0, 1.0, mixed.n_unknown_dofs, dtype=jnp.float32)}

    def res_u(v, _u_field, _p):
        return h_wf.dot(v, (1.0, 0.0, 0.0)) * h_wf.ds()

    residual = ff.compile_mixed_surface_residual(
        ff.make_mixed_residuals(u=ff.bind_mixed_residual("u", res_u, space="V"))
    )

    with pytest.raises(NotImplementedError, match="volume-only"):
        mixed.assemble_residual(residual, u, params=None)

    with pytest.raises(NotImplementedError, match="volume-only"):
        mixed.assemble_jacobian(residual, u, params=None)


def test_mixed_role_spaces_reject_block_system_helpers():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    with pytest.raises(NotImplementedError, match="build_block_system"):
        mixed.build_block_system(diag={"u": np.eye(mixed.n_unknown_dofs)})


def test_mixed_role_block_system_splits_unknown_and_residual_layouts():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    K = np.eye(mixed.n_unknown_dofs)
    R = np.arange(mixed.n_residual_dofs, dtype=float)
    system = mixed.build_role_block_system(K, R)

    u = {"u": jnp.linspace(0.0, 1.0, mixed.n_unknown_dofs, dtype=jnp.float32)}
    u_vec = system.join_unknown(u)
    r_fields = system.split_residual(R)

    assert np.asarray(u_vec).shape == (mixed.n_unknown_dofs,)
    assert np.asarray(r_fields["u"]).shape == (mixed.n_residual_dofs,)
    assert np.allclose(np.asarray(system.split_unknown(u_vec)["u"]), np.asarray(u["u"]))


def test_mixed_role_block_system_free_unknown_dofs_follow_unknown_dirichlet():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    bc = ff.DirichletBC(np.array([0, 2], dtype=int), np.array([0.0, 0.0], dtype=float))
    system = mixed.build_role_block_system(np.eye(mixed.n_unknown_dofs), np.zeros(mixed.n_residual_dofs), unknown_dirichlet=bc)

    assert np.array_equal(
        system.free_unknown_dofs,
        np.setdiff1d(np.arange(mixed.n_unknown_dofs, dtype=int), np.array([0, 2], dtype=int)),
    )


def test_build_mixed_role_block_system_builds_rectangular_flux_operator():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    block = np.zeros((mixed.n_residual_dofs, mixed.n_unknown_dofs))
    block[0, 0] = 2.0
    block[-1, -1] = -1.0

    system = ff.build_mixed_role_block_system(
        mixed,
        blocks={("u", "u"): block},
        residual={"u": np.ones(mixed.n_residual_dofs)},
        format="flux",
    )

    assert system.K.shape == (mixed.n_residual_dofs, mixed.n_unknown_dofs)
    assert np.allclose(np.asarray(system.R), np.ones(mixed.n_residual_dofs))


def test_build_mixed_role_block_system_accepts_nested_blocks_and_dense_format():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    block = np.eye(mixed.n_unknown_dofs)
    block = np.pad(block, ((0, mixed.n_residual_dofs - mixed.n_unknown_dofs), (0, 0)))
    residual = np.arange(mixed.n_residual_dofs, dtype=float)

    system = ff.build_mixed_role_block_system(
        mixed,
        blocks={"u": {"u": block}},
        residual=residual,
        format="dense",
    )

    assert np.asarray(system.K).shape == (mixed.n_residual_dofs, mixed.n_unknown_dofs)
    assert np.allclose(np.asarray(system.R), residual)


def test_mixed_role_block_system_condense_unknown_flux_operator():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    K = np.zeros((mixed.n_residual_dofs, mixed.n_unknown_dofs))
    K[0, 0] = 2.0
    K[1, 1] = 3.0
    R = np.ones(mixed.n_residual_dofs)
    system = ff.build_mixed_role_block_system(
        mixed,
        blocks={("u", "u"): K},
        residual=R,
        unknown_dirichlet=ff.DirichletBC(np.array([0]), np.array([5.0])),
        format="flux",
    )

    condensed = system.condense_unknown()

    assert condensed.K.shape == (mixed.n_residual_dofs, mixed.n_unknown_dofs - 1)
    assert np.allclose(np.asarray(condensed.R[:2]), np.array([1.0 - 10.0, 1.0]))
    assert np.array_equal(condensed.free_unknown_dofs, np.arange(1, mixed.n_unknown_dofs, dtype=int))
    full = condensed.expand_unknown(np.zeros(condensed.free_unknown_dofs.size))
    assert full[0] == 5.0


def test_mixed_role_block_system_condense_unknown_dense_matrix():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    scalar = ff.make_hex_space(mesh, dim=1, intorder=2)
    vector = ff.make_hex_space(mesh, dim=3, intorder=2)

    mixed = ff.MixedRoleSpaces(
        {
            "u": ff.ResidualSpaces(
                test=ff.NamedSpace("V", vector),
                unknown=ff.NamedSpace("U", scalar),
            ),
        }
    ).to_fe_space()

    K = np.zeros((mixed.n_residual_dofs, mixed.n_unknown_dofs))
    K[2, 0] = -4.0
    K[2, 2] = 7.0
    R = np.arange(mixed.n_residual_dofs, dtype=float)
    system = ff.build_mixed_role_block_system(
        mixed,
        blocks={("u", "u"): K},
        residual=R,
        format="dense",
    )

    condensed = system.condense_unknown(ff.DirichletBC(np.array([2]), np.array([2.0])))

    assert np.asarray(condensed.K).shape == (mixed.n_residual_dofs, mixed.n_unknown_dofs - 1)
    assert np.isclose(np.asarray(condensed.R)[2], R[2] - 14.0)

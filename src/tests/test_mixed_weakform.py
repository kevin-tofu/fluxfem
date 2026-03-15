"""Mixed weak-form assembly tests."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.mixed_space import MixedFESpace
from fluxfem.core.mixed_weakform import (
    MixedResidualForm,
    assemble_mixed_jacobian_wf,
    assemble_mixed_residual_wf,
)


def _make_mixed_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = MixedFESpace({"u": space, "p": space})
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


def test_mixed_residual_matches_single_field():
    """Mixed residual for one field matches single-field assembly."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = MixedFESpace({"u": space, "t": space})

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
    mixed = MixedFESpace(
        {"disp": space, "press": space},
        field_to_space_key={"disp": "V", "press": "Q"},
    )

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

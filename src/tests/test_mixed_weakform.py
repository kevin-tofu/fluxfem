"""Mixed weak-form assembly tests."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.mixed_space import MixedFESpace
from fluxfem.mixed_weakform import (
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
    J_dict = assemble_mixed_jacobian_wf(mixed, residuals, u_vec, params, sparse=False)
    J_form = assemble_mixed_jacobian_wf(
        mixed, MixedResidualForm(residuals), u_vec, params, sparse=False
    )
    J_compiled = assemble_mixed_jacobian_wf(
        mixed, ff.MixedWeakForm(residuals=residuals), u_vec, params, sparse=False
    )

    assert np.allclose(np.asarray(J_dict), np.asarray(J_form), atol=1e-6)
    assert np.allclose(np.asarray(J_dict), np.asarray(J_compiled), atol=1e-6)


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

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

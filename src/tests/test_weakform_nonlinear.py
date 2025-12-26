"""Nonlinear weak-form residual tests."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def test_weakform_nonlinear_residual_matches_tensor():
    """Weak-form residual matches a tensor-based reference for u^2."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    def tensor_residual(ctx: ff.FormContext, u_elem: jnp.ndarray, _p) -> jnp.ndarray:
        u_q = ctx.trial.eval(u_elem)  # (n_q,)
        return ctx.test.N * (u_q[:, None] ** 2)

    wf_residual = ff.ResidualForm.volume(
        lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
    )

    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(space.n_dofs))

    R_tensor = space.assemble_residual(tensor_residual, u, params=None)
    R_wf = space.assemble_residual(wf_residual.get_compiled(), u, params=None)

    assert np.allclose(np.asarray(R_tensor), np.asarray(R_wf))


def test_weakform_nonlinear_jacobian_matches_tensor():
    """Weak-form Jacobian matches a tensor-based reference for u^2."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    def tensor_residual(ctx: ff.FormContext, u_elem: jnp.ndarray, _p) -> jnp.ndarray:
        u_q = ctx.trial.eval(u_elem)
        return ctx.test.N * (u_q[:, None] ** 2)

    wf_residual = ff.ResidualForm.volume(
        lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
    )

    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.standard_normal(space.n_dofs))

    J_tensor = space.assemble_jacobian(
        tensor_residual, u, params=None, sparse=False
    )
    J_wf = space.assemble_jacobian(
        wf_residual.get_compiled(), u, params=None, sparse=False
    )

    assert np.allclose(np.asarray(J_tensor), np.asarray(J_wf))


def test_weakform_neo_hookean_residual_matches_tensor():
    """Weak-form Neo-Hookean residual matches tensor-based reference."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)

    params = {"mu": 2.0, "lam": 3.0}

    def neo_hookean_wf(v, u, p):
        F = h_wf.I(3) + u.grad
        C = h_wf.ddot(F, F)
        C_inv = h_wf.inv(C)
        logJ = h_wf.log(h_wf.det(F))
        S = p.mu * (h_wf.I(3) - C_inv) + p.lam * logJ * C_inv
        P = h_wf.matmul(F, S)
        # P = F @ S.T
        return h_wf.gaction(v, P) * h_wf.dOmega()

    wf_residual = ff.ResidualForm.volume(neo_hookean_wf)

    u0 = jnp.zeros(space.n_dofs)

    R_tensor = space.assemble_residual(ff.neo_hookean_residual_form, u0, params)
    R_wf = space.assemble_residual(wf_residual.get_compiled(), u0, params)

    assert np.allclose(np.asarray(R_tensor), np.asarray(R_wf), atol=1e-6)
    assert np.allclose(np.asarray(R_wf), 0.0, atol=1e-8)

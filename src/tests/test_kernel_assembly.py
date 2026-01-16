"""Kernel-based assembly tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def _make_space():
    mesh = ff.StructuredHexBox(nx=2, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    return ff.make_hex_space(mesh, dim=1, intorder=2)


@pytest.mark.parametrize("n_chunks", [None, 2])
def test_bilinear_kernel_matches_form(n_chunks):
    space = _make_space()
    kappa = 1.0
    ker = ff.make_element_bilinear_kernel(ff.diffusion_form, kappa, jit=True)
    K_kernel = space.assemble_bilinear_form(
        ff.diffusion_form, kappa, kernel=ker, n_chunks=n_chunks
    )
    K_default = space.assemble_bilinear_form(
        ff.diffusion_form, kappa, n_chunks=n_chunks
    )
    assert np.allclose(
        np.asarray(K_kernel.to_dense()), np.asarray(K_default.to_dense())
    )


@pytest.mark.parametrize("n_chunks", [None, 2])
def test_linear_kernel_matches_form(n_chunks):
    space = _make_space()

    def _linear_kernel(ctx):
        integrand = ff.scalar_body_force_form(ctx, 2.0)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    ker = jax.jit(_linear_kernel)
    F_kernel = space.assemble_linear_form(
        ff.scalar_body_force_form, 2.0, kernel=ker, n_chunks=n_chunks
    )
    F_default = space.assemble_linear_form(
        ff.scalar_body_force_form, 2.0, n_chunks=n_chunks
    )
    assert np.allclose(np.asarray(F_kernel), np.asarray(F_default))


@pytest.mark.parametrize("n_chunks", [None, 2])
def test_residual_kernel_matches_form(n_chunks):
    space = _make_space()
    u = jnp.arange(space.n_dofs, dtype=jnp.float64)

    def simple_residual(ctx, u_elem, _params):
        return jnp.broadcast_to(u_elem, (ctx.w.shape[0], u_elem.shape[0]))

    ker = ff.make_element_residual_kernel(simple_residual, params=None)
    R_kernel = space.assemble_residual(
        simple_residual, u, params=None, kernel=ker, n_chunks=n_chunks
    )
    R_default = space.assemble_residual(
        simple_residual, u, params=None, n_chunks=n_chunks
    )
    assert np.allclose(np.asarray(R_kernel), np.asarray(R_default))


@pytest.mark.parametrize("n_chunks", [None, 2])
def test_jacobian_kernel_matches_form(n_chunks):
    space = _make_space()
    u = jnp.arange(space.n_dofs, dtype=jnp.float64)

    def simple_residual(ctx, u_elem, _params):
        return jnp.broadcast_to(u_elem, (ctx.w.shape[0], u_elem.shape[0]))

    ker = ff.make_element_jacobian_kernel(simple_residual, params=None)
    J_kernel = space.assemble_jacobian(
        simple_residual, u, params=None, kernel=ker, n_chunks=n_chunks, sparse=False
    )
    J_default = space.assemble_jacobian(
        simple_residual, u, params=None, n_chunks=n_chunks, sparse=False
    )
    assert np.allclose(np.asarray(J_kernel), np.asarray(J_default))


def test_bilinear_kernel_with_compiled_dsl():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form_wf = ff.BilinearForm.volume(
        lambda u, v, p: p.kappa * (v.grad @ u.grad) * h_wf.dOmega()
    ).get_compiled()
    params = ff.Params(kappa=1.0)
    ker = ff.make_element_bilinear_kernel(form_wf, params, jit=True)

    K_kernel = space.assemble_bilinear_form(form_wf, params, kernel=ker)
    K_default = space.assemble_bilinear_form(form_wf, params)
    assert np.allclose(
        np.asarray(K_kernel.to_dense()), np.asarray(K_default.to_dense())
    )


def test_neo_hookean_kernel_matches_default():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    params = {"mu": 2.0, "lam": 3.0}
    u0 = jnp.zeros(space.n_dofs)

    res_ker = ff.make_element_residual_kernel(ff.neo_hookean_residual_form, params)
    jac_ker = ff.make_element_jacobian_kernel(ff.neo_hookean_residual_form, params)

    R_kernel = space.assemble_residual(ff.neo_hookean_residual_form, u0, params, kernel=res_ker)
    R_default = space.assemble_residual(ff.neo_hookean_residual_form, u0, params)
    assert np.allclose(np.asarray(R_kernel), np.asarray(R_default), atol=1e-6)

    J_kernel = space.assemble_jacobian(
        ff.neo_hookean_residual_form, u0, params, kernel=jac_ker, sparse=False
    )
    J_default = space.assemble_jacobian(
        ff.neo_hookean_residual_form, u0, params, sparse=False
    )
    assert np.allclose(np.asarray(J_kernel), np.asarray(J_default), atol=1e-6)


def test_make_element_kernel_dispatch():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    u0 = jnp.zeros(space.n_dofs)

    ker_bilin = ff.make_element_kernel(ff.diffusion_form, 1.0, kind="bilinear")
    K = space.assemble_bilinear_form(ff.diffusion_form, 1.0, kernel=ker_bilin)
    K_ref = space.assemble_bilinear_form(ff.diffusion_form, 1.0)
    assert np.allclose(np.asarray(K.to_dense()), np.asarray(K_ref.to_dense()))

    def simple_residual(ctx, u_elem, _params):
        return jnp.broadcast_to(u_elem, (ctx.w.shape[0], u_elem.shape[0]))

    ker_res = ff.make_element_kernel(simple_residual, None, kind="residual")
    R = space.assemble_residual(simple_residual, u0, None, kernel=ker_res)
    R_ref = space.assemble_residual(simple_residual, u0, None)
    assert np.allclose(np.asarray(R), np.asarray(R_ref))

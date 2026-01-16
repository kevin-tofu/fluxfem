"""BatchedAssembler tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff


def _make_space():
    mesh = ff.StructuredHexBox(nx=2, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    return ff.make_hex_space(mesh, dim=1, intorder=2)


def test_bilinear_with_kernel_matches_space():
    space = _make_space()
    kappa = 1.0
    ker = ff.make_element_bilinear_kernel(ff.diffusion_form, kappa, jit=True)
    batch = space.make_batched_assembler()
    K_kernel = batch.assemble_bilinear_with_kernel(ker)
    K_default = space.assemble_bilinear_form(ff.diffusion_form, kappa)
    assert np.allclose(
        np.asarray(K_kernel.to_dense()), np.asarray(K_default.to_dense())
    )


def test_linear_with_kernel_matches_space():
    space = _make_space()

    def linear_kernel(ctx):
        integrand = ff.scalar_body_force_form(ctx, 2.0)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    ker = jax.jit(linear_kernel)
    batch = space.make_batched_assembler()
    F_kernel = batch.assemble_linear_with_kernel(ker)
    F_default = space.assemble_linear_form(ff.scalar_body_force_form, 2.0)
    assert np.allclose(np.asarray(F_kernel), np.asarray(F_default))


@pytest.mark.parametrize("n_active", [0, 1, 2])
def test_mask_zeros_out_tail(n_active):
    space = _make_space()
    batch = space.make_batched_assembler()
    mask = batch.make_mask(n_active)
    ker = ff.make_element_bilinear_kernel(ff.diffusion_form, 1.0, jit=True)
    K_masked = batch.assemble_bilinear_with_kernel(ker, mask=mask).to_dense()
    K_sliced = batch.slice(n_active).assemble_bilinear_with_kernel(ker).to_dense()
    assert np.allclose(np.asarray(K_masked), np.asarray(K_sliced))


def test_residual_jacobian_with_kernel_matches_space():
    space = _make_space()
    u = jnp.arange(space.n_dofs, dtype=jnp.float64)

    def simple_residual(ctx, u_elem, _params):
        return jnp.broadcast_to(u_elem, (ctx.w.shape[0], u_elem.shape[0]))

    ker_r = ff.make_element_residual_kernel(simple_residual, params=None)
    ker_j = ff.make_element_jacobian_kernel(simple_residual, params=None)
    batch = space.make_batched_assembler()

    R_kernel = batch.assemble_residual_with_kernel(ker_r, u)
    R_default = space.assemble_residual(simple_residual, u, params=None)
    assert np.allclose(np.asarray(R_kernel), np.asarray(R_default))

    J_kernel = batch.assemble_jacobian_with_kernel(ker_j, u, sparse=False)
    J_default = space.assemble_jacobian(simple_residual, u, params=None, sparse=False)
    assert np.allclose(np.asarray(J_kernel), np.asarray(J_default))


@pytest.mark.parametrize("n_chunks", [None, 2])
def test_bilinear_kernel_with_chunks_matches_space(n_chunks):
    space = _make_space()
    kappa = 1.0
    ker = ff.make_element_bilinear_kernel(ff.diffusion_form, kappa, jit=True)
    batch = space.make_batched_assembler()
    K_kernel = batch.assemble_bilinear_with_kernel(ker)
    K_space = space.assemble_bilinear_form(
        ff.diffusion_form, kappa, n_chunks=n_chunks
    )
    assert np.allclose(
        np.asarray(K_kernel.to_dense()), np.asarray(K_space.to_dense())
    )


def test_bilinear_kernel_with_pattern_matches_space():
    space = _make_space()
    kappa = 1.0
    ker = ff.make_element_bilinear_kernel(ff.diffusion_form, kappa, jit=True)
    pattern = space.get_sparsity_pattern(with_idx=True)
    batch = space.make_batched_assembler(pattern=pattern)
    K_kernel = batch.assemble_bilinear_with_kernel(ker)
    K_space = space.assemble_bilinear_form(
        ff.diffusion_form, kappa, pattern=pattern
    )
    assert np.allclose(
        np.asarray(K_kernel.to_dense()), np.asarray(K_space.to_dense())
    )

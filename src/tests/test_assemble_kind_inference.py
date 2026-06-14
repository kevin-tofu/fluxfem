"""assemble() kind inference tests."""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def _make_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    return ff.make_hex_space(mesh, dim=1, intorder=2)


def test_assemble_infers_kind_from_compiled_form():
    space = _make_space()
    form = ff.BilinearForm.volume(
        lambda u, v, p: p.kappa * (v.grad @ u.grad) * h_wf.dOmega()
    )
    compiled = form.get_compiled()
    params = ff.Params(kappa=1.0)

    K_compiled = space.assemble(compiled, params=params)
    K_ref = space.assemble(form, params=params)

    assert np.allclose(
        np.asarray(K_compiled.to_dense()), np.asarray(K_ref.to_dense())
    )


def test_assemble_infers_kind_from_tagged_kernel():
    space = _make_space()

    @ff.kernel(kind="bilinear", domain="volume")
    def tagged_kernel(ctx, kappa):
        return kappa * jnp.einsum(
            "qia,qja->qij",
            ctx.test.gradN,
            ctx.trial.gradN,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Raw kernel has no _ff_kind metadata",
            category=UserWarning,
        )
        K_tagged = space.assemble(tagged_kernel, params=1.0)

    K_ref = space.assemble(ff.diffusion_form, params=1.0)
    assert np.allclose(np.asarray(K_tagged.to_dense()), np.asarray(K_ref.to_dense()))


def test_assemble_warns_on_untagged_kernel():
    space = _make_space()

    def untagged_kernel(ctx, kappa):
        return kappa * jnp.einsum(
            "qia,qja->qij",
            ctx.test.gradN,
            ctx.trial.gradN,
        )

    with pytest.warns(UserWarning, match="no _ff_kind metadata"):
        space.assemble(untagged_kernel, params=1.0, kind="bilinear")

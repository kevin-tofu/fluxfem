"""Two-body coupling block test for mixed weak-form assembly."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.mixed_space import MixedFESpace
from fluxfem.core.mixed_weakform import assemble_mixed_jacobian_wf


def test_mixed_two_body_coupling_blocks():
    """Mixed Jacobian exposes two trial/test pairs via coupling blocks."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = MixedFESpace({"a": space, "b": space})

    def res_a(v, u, p):
        u_b = ff.unknown_ref("b")
        return (v * (u.val + p.kappa * u_b.val)) * h_wf.dOmega()

    def res_b(w, u_b, p):
        u_a = ff.unknown_ref("a")
        return (w * (u_b.val + p.kappa * u_a.val)) * h_wf.dOmega()

    params = ff.Params(kappa=2.5)
    rng = np.random.default_rng(3)
    u_vec = jnp.asarray(rng.standard_normal(mixed.n_dofs))

    J_mixed = assemble_mixed_jacobian_wf(
        mixed, {"a": res_a, "b": res_b}, u_vec, params, sparse=False
    )

    def res_mass(v, u, _p):
        return (v * u.val) * h_wf.dOmega()

    u_zero = jnp.zeros(space.n_dofs)
    M = space.assemble_jacobian(
        ff.ResidualForm.volume(res_mass).get_compiled(), u_zero, params, sparse=False
    )
    M = np.asarray(M)

    a_slice = mixed.field_slices["a"]
    b_slice = mixed.field_slices["b"]
    J = np.asarray(J_mixed)

    assert np.allclose(J[a_slice, a_slice], M, atol=1e-6)
    assert np.allclose(J[a_slice, b_slice], params.kappa * M, atol=1e-6)
    assert np.allclose(J[b_slice, a_slice], params.kappa * M, atol=1e-6)
    assert np.allclose(J[b_slice, b_slice], M, atol=1e-6)

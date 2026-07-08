from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def _one_hex_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    return ff.make_hex_space(mesh, dim=3, intorder=2)


def test_make_j2_quadrature_state_matches_space_quadrature_shape():
    space = _one_hex_space()
    state = ff.make_j2_quadrature_state(space, dtype=jnp.float64)

    assert isinstance(state, ff.J2PlasticityQuadratureState)
    assert state.plastic_strain.shape == (1, 8, 6)
    assert state.equivalent_plastic_strain.shape == (1, 8)
    np.testing.assert_allclose(np.asarray(state.plastic_strain), 0.0)
    np.testing.assert_allclose(np.asarray(state.equivalent_plastic_strain), 0.0)


def test_j2_residual_matches_linear_elasticity_while_elastic():
    space = _one_hex_space()
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=1.0e12, hardening_modulus=0.0)
    state = ff.make_j2_quadrature_state(space, dtype=jnp.float64)
    params = {"material": material, "state": state}

    rng = np.random.default_rng(7)
    u = jnp.asarray(1.0e-5 * rng.standard_normal(space.n_dofs), dtype=jnp.float64)

    R_j2 = space.assemble_residual(ff.j2_plasticity_residual_form, u, params)
    K = space.assemble_bilinear_form(ff.linear_elasticity_form, ff.isotropic_3d_D(material.E, material.nu))
    R_linear = K.to_dense() @ u

    np.testing.assert_allclose(np.asarray(R_j2), np.asarray(R_linear), rtol=1.0e-10, atol=1.0e-10)


def test_update_j2_quadrature_state_commits_plastic_history():
    space = _one_hex_space()
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=50.0, hardening_modulus=100.0)
    state0 = ff.make_j2_quadrature_state(space, dtype=jnp.float64)

    u = jnp.zeros(space.n_dofs, dtype=jnp.float64)
    coords = np.asarray(space.mesh.coords)
    for node_id, (x, _y, _z) in enumerate(coords):
        u = u.at[3 * node_id + 0].set(5.0e-3 * x)

    state1 = ff.update_j2_quadrature_state(space, u, state0, material)

    assert state1.plastic_strain.shape == state0.plastic_strain.shape
    assert state1.equivalent_plastic_strain.shape == state0.equivalent_plastic_strain.shape
    assert float(jnp.max(state1.equivalent_plastic_strain)) > 0.0
    np.testing.assert_allclose(np.asarray(ff.voigt_trace(state1.plastic_strain[0])), 0.0, atol=1.0e-12)


def test_j2_residual_form_uses_frozen_state_without_committing():
    space = _one_hex_space()
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=50.0, hardening_modulus=100.0)
    state0 = ff.make_j2_quadrature_state(space, dtype=jnp.float64)
    params0 = {"material": material, "state": state0}

    u = jnp.zeros(space.n_dofs, dtype=jnp.float64)
    coords = np.asarray(space.mesh.coords)
    for node_id, (x, _y, _z) in enumerate(coords):
        u = u.at[3 * node_id + 0].set(5.0e-3 * x)

    _ = space.assemble_residual(ff.j2_plasticity_residual_form, u, params0)

    np.testing.assert_allclose(np.asarray(state0.plastic_strain), 0.0)
    np.testing.assert_allclose(np.asarray(state0.equivalent_plastic_strain), 0.0)

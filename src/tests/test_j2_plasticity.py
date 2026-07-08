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


def test_j2_return_mapping_is_elastic_below_yield():
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=250.0, hardening_modulus=1_000.0)
    state = ff.make_j2_plasticity_state()
    strain = jnp.array([1.0e-4, -2.0e-5, 0.0, 3.0e-5, 0.0, 0.0], dtype=jnp.float64)

    stress, next_state = ff.j2_return_mapping(strain, state, material)

    np.testing.assert_allclose(np.asarray(stress), np.asarray(ff.isotropic_3d_D(material.E, material.nu) @ strain), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(next_state.plastic_strain), np.zeros((6,)), atol=1.0e-14)
    assert float(next_state.equivalent_plastic_strain) == 0.0


def test_j2_return_mapping_keeps_hydrostatic_response_elastic():
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=10.0, hardening_modulus=0.0)
    state = ff.make_j2_plasticity_state()
    strain = jnp.array([2.0e-3, 2.0e-3, 2.0e-3, 0.0, 0.0, 0.0], dtype=jnp.float64)

    stress, next_state = ff.j2_return_mapping(strain, state, material)

    np.testing.assert_allclose(float(ff.von_mises_stress_voigt(stress)), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(next_state.plastic_strain), np.zeros((6,)), atol=1.0e-14)
    assert float(next_state.equivalent_plastic_strain) == 0.0


def test_j2_return_mapping_returns_pure_shear_to_yield_surface():
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=120.0, hardening_modulus=500.0)
    state = ff.make_j2_plasticity_state()
    strain = jnp.array([0.0, 0.0, 0.0, 6.0e-3, 0.0, 0.0], dtype=jnp.float64)

    stress, next_state = ff.j2_return_mapping(strain, state, material)
    yield_value = ff.j2_yield_function(stress, next_state, material)

    np.testing.assert_allclose(float(yield_value), 0.0, atol=1.0e-9)
    assert float(next_state.equivalent_plastic_strain) > 0.0
    np.testing.assert_allclose(float(ff.voigt_trace(next_state.plastic_strain)), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(float(ff.von_mises_stress_voigt(stress)), material.yield_stress + material.hardening_modulus * float(next_state.equivalent_plastic_strain), rtol=1.0e-12)


def test_j2_return_mapping_unload_is_elastic_after_plastic_step():
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=120.0, hardening_modulus=500.0)
    state0 = ff.make_j2_plasticity_state()
    strain1 = jnp.array([0.0, 0.0, 0.0, 6.0e-3, 0.0, 0.0], dtype=jnp.float64)
    stress1, state1 = ff.j2_return_mapping(strain1, state0, material)

    strain2 = jnp.array([0.0, 0.0, 0.0, 5.8e-3, 0.0, 0.0], dtype=jnp.float64)
    stress2, state2 = ff.j2_return_mapping(strain2, state1, material)

    elastic_delta = ff.isotropic_3d_D(material.E, material.nu) @ (strain2 - strain1)
    np.testing.assert_allclose(np.asarray(stress2 - stress1), np.asarray(elastic_delta), rtol=1.0e-12, atol=1.0e-10)
    np.testing.assert_allclose(float(state2.equivalent_plastic_strain), float(state1.equivalent_plastic_strain), atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(state2.plastic_strain), np.asarray(state1.plastic_strain), atol=1.0e-14)


def test_j2_return_mapping_is_jittable_with_state_pytree():
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=120.0, hardening_modulus=500.0)
    state = ff.make_j2_plasticity_state()
    strain = jnp.array([0.0, 0.0, 0.0, 6.0e-3, 0.0, 0.0], dtype=jnp.float64)

    update = jax.jit(ff.j2_return_mapping)
    stress, next_state = update(strain, state, material)

    assert stress.shape == (6,)
    assert next_state.plastic_strain.shape == (6,)
    assert float(next_state.equivalent_plastic_strain) > 0.0

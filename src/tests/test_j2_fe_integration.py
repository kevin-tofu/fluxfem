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


def _homogeneous_extension_dirichlet(space, axial_strain: float):
    coords = np.asarray(space.mesh.coords)
    dofs = np.arange(space.n_dofs, dtype=int)
    vals = np.zeros(space.n_dofs, dtype=float)
    for node_id, (x, _y, _z) in enumerate(coords):
        vals[3 * node_id + 0] = axial_strain * x
    return dofs, vals


def test_solve_j2_plasticity_load_steps_commits_only_after_each_converged_step():
    space = _one_hex_space()
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=50.0, hardening_modulus=100.0)
    dirichlet = _homogeneous_extension_dirichlet(space, axial_strain=5.0e-3)

    u, state, history = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        dirichlet=dirichlet,
        n_steps=2,
    )

    assert len(history) == 2
    assert all(step.converged for step in history)
    assert all(step.committed for step in history)
    assert history[0].max_equivalent_plastic_strain <= history[1].max_equivalent_plastic_strain
    assert float(jnp.max(state.equivalent_plastic_strain)) > 0.0
    np.testing.assert_allclose(np.asarray(state.equivalent_plastic_strain), np.asarray(history[-1].trial_state.equivalent_plastic_strain))
    np.testing.assert_allclose(np.asarray(u), dirichlet[1])


def test_solve_j2_plasticity_load_steps_can_skip_commit():
    space = _one_hex_space()
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=50.0, hardening_modulus=100.0)
    state0 = ff.make_j2_quadrature_state(space, dtype=jnp.float64)
    dirichlet = _homogeneous_extension_dirichlet(space, axial_strain=5.0e-3)

    _u, state, history = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        initial_state=state0,
        dirichlet=dirichlet,
        n_steps=1,
        commit_on_converged=False,
    )

    assert len(history) == 1
    assert history[0].converged
    assert not history[0].committed
    assert history[0].max_equivalent_plastic_strain > 0.0
    np.testing.assert_allclose(np.asarray(state.equivalent_plastic_strain), 0.0)
    np.testing.assert_allclose(np.asarray(state.plastic_strain), 0.0)


def test_j2_uniaxial_extension_matches_material_point_reference():
    space = _one_hex_space()
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=50.0, hardening_modulus=100.0)
    axial_strain = 5.0e-3
    dirichlet = _homogeneous_extension_dirichlet(space, axial_strain=axial_strain)

    u, state, history = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        dirichlet=dirichlet,
        n_steps=1,
    )

    stress_q = ff.evaluate_j2_quadrature_stress(space, u, history[0].state, material)
    strain_q = ff.evaluate_j2_quadrature_strain(space, u)
    strain_ref = jnp.array([axial_strain, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    stress_ref, state_ref = ff.j2_return_mapping(strain_ref, ff.make_j2_plasticity_state(dtype=jnp.float64), material)

    np.testing.assert_allclose(np.asarray(strain_q), np.broadcast_to(np.asarray(strain_ref), strain_q.shape), atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(stress_q), np.broadcast_to(np.asarray(stress_ref), stress_q.shape), rtol=1.0e-12, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(state.equivalent_plastic_strain), float(state_ref.equivalent_plastic_strain), rtol=1.0e-12)


def test_j2_unload_after_committed_extension_has_elastic_stress_increment():
    space = _one_hex_space()
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=50.0, hardening_modulus=100.0)
    dirichlet_load = _homogeneous_extension_dirichlet(space, axial_strain=5.0e-3)
    u_load, state_load, _history_load = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        dirichlet=dirichlet_load,
        n_steps=1,
    )

    stress_load = ff.evaluate_j2_quadrature_stress(space, u_load, state_load, material)
    dirichlet_unload = _homogeneous_extension_dirichlet(space, axial_strain=4.8e-3)
    u_unload, state_unload, history_unload = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        initial_state=state_load,
        u0=u_load,
        dirichlet=dirichlet_unload,
        n_steps=1,
    )
    stress_unload = ff.evaluate_j2_quadrature_stress(space, u_unload, state_load, material)

    deps = jnp.array([-2.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    elastic_delta = ff.isotropic_3d_D(material.E, material.nu) @ deps

    assert history_unload[0].converged
    np.testing.assert_allclose(
        np.asarray(stress_unload - stress_load),
        np.broadcast_to(np.asarray(elastic_delta), stress_unload.shape),
        rtol=1.0e-12,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(np.asarray(state_unload.equivalent_plastic_strain), np.asarray(state_load.equivalent_plastic_strain), atol=1.0e-14)


def test_make_j2_cell_data_and_write_vtu(tmp_path):
    space = _one_hex_space()
    mesh = space.mesh
    material = ff.J2Plasticity(E=210_000.0, nu=0.30, yield_stress=50.0, hardening_modulus=100.0)
    dirichlet = _homogeneous_extension_dirichlet(space, axial_strain=5.0e-3)
    u, state, _history = ff.solve_j2_plasticity_load_steps(
        space,
        material,
        dirichlet=dirichlet,
        n_steps=1,
    )

    cell_data = ff.make_j2_cell_data(space, u, state, material)

    assert set(cell_data) >= {"j2_p_eq", "j2_sigma_vm", "j2_sigma_xx", "j2_stress_voigt"}
    assert cell_data["j2_p_eq"].shape == (mesh.conn.shape[0],)
    assert cell_data["j2_stress_voigt"].shape == (mesh.conn.shape[0], 6)
    assert float(cell_data["j2_p_eq"].max()) > 0.0
    assert float(cell_data["j2_sigma_vm"].max()) > 0.0

    out = tmp_path / "j2_uniaxial.vtu"
    ff.write_j2_vtu(mesh, space, u, state, material, str(out))

    text = out.read_text(encoding="ascii")
    assert 'Name="displacement"' in text
    assert 'Name="j2_p_eq"' in text
    assert 'Name="j2_sigma_vm"' in text
    assert 'Name="j2_stress_voigt"' in text

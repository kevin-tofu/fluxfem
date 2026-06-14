import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def _condensed_bar_matrices(nx: int = 4):
    mesh = ff.StructuredHexBox(nx=nx, ny=1, nz=1, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    stiffness = space.assemble(ff.diffusion_form, params=1.0)
    mass = space.assemble_mass_matrix()

    coords = np.asarray(mesh.coords, dtype=float)
    left = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], coords[:, 0].min(), atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    right = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], coords[:, 0].max(), atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    free = np.asarray(ff.free_dofs(space.n_dofs, left), dtype=int)
    retained = np.flatnonzero(np.isin(free, np.asarray(right, dtype=int))).astype(int)
    return stiffness, mass, free, retained


def test_craig_bampton_dense_static_modes_satisfy_partitioned_equilibrium():
    stiffness = np.array(
        [
            [4.0, -1.0, 0.0, -0.5],
            [-1.0, 3.0, -0.4, 0.0],
            [0.0, -0.4, 2.6, -0.8],
            [-0.5, 0.0, -0.8, 2.8],
        ],
        dtype=float,
    )
    mass = np.diag([2.0, 1.5, 1.2, 1.0])
    retained = np.array([0, 3], dtype=int)

    cb = ff.make_craig_bampton_basis(stiffness, mass, retained, n_modes=1)

    internal = np.asarray(cb.internal_dofs)
    k_ii = stiffness[np.ix_(internal, internal)]
    k_ir = stiffness[np.ix_(internal, retained)]
    psi = np.asarray(cb.basis)[np.ix_(internal, np.arange(retained.size))]

    np.testing.assert_allclose(k_ii @ psi + k_ir, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(cb.basis)[retained, : retained.size], np.eye(2), atol=1.0e-12)
    assert cb.n_reduced == retained.size + 1


def test_craig_bampton_reduced_residual_keeps_jax_autodiff_path():
    stiffness = jnp.array(
        [
            [3.0, -1.0, 0.0],
            [-1.0, 2.5, -0.5],
            [0.0, -0.5, 1.8],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(3, dtype=jnp.float64)
    cb = ff.make_craig_bampton_basis(stiffness, mass, jnp.array([0], dtype=jnp.int32), n_modes=1)

    def residual(u):
        return stiffness @ u + 0.2 * u**3

    q = jnp.array([0.1, -0.03], dtype=jnp.float64)
    jac = cb.reduced_jacobian(residual)(q)
    u = cb.expand(q)
    expected = cb.basis.T @ (stiffness + jnp.diag(0.6 * u**2)) @ cb.basis

    np.testing.assert_allclose(np.asarray(jac), np.asarray(expected), rtol=1.0e-10, atol=1.0e-12)


def test_craig_bampton_sparse_fluxfem_matrices_match_dense_eigenvalues():
    pytest.importorskip("scipy")

    stiffness, mass, free, retained = _condensed_bar_matrices(nx=4)
    k_free = stiffness.to_csr()[free, :][:, free]
    m_free = mass.to_csr()[free, :][:, free]

    cb_sparse = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained,
        n_modes=2,
        constraint_solver="spsolve",
        modal_solver="eigsh",
    )
    cb_dense = ff.make_craig_bampton_basis(
        k_free.toarray(),
        m_free.toarray(),
        retained,
        n_modes=2,
        constraint_solver="dense",
        modal_solver="dense",
    )

    np.testing.assert_allclose(np.asarray(cb_sparse.eigenvalues), np.asarray(cb_dense.eigenvalues), rtol=1.0e-8)
    np.testing.assert_allclose(
        np.asarray(cb_sparse.basis)[retained, : retained.size],
        np.eye(retained.size),
        atol=1.0e-12,
    )
    assert cb_sparse.n_full == free.size
    assert cb_sparse.n_modes == 2


def test_craig_bampton_top_level_exports_are_available():
    assert ff.CraigBamptonBasis is not None
    assert ff.make_craig_bampton_basis is ff.solver.make_craig_bampton_basis


def test_craig_bampton_cg_and_subspace_solvers_match_dense():
    stiffness = jnp.array(
        [
            [6.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 7.0, -1.5, 0.0, 0.0],
            [0.0, -1.5, 8.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 7.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(5, dtype=jnp.float64)
    retained = jnp.array([0, 4], dtype=jnp.int32)

    dense = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=2)
    iterative = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=retained,
        n_modes=2,
        constraint_solver="cg",
        modal_solver="subspace",
        modal_linear_solver="cg",
        modal_oversample=1,
        modal_maxiter=60,
        modal_tol=1.0e-10,
        cg_tol=1.0e-11,
        cg_maxiter=100,
    )

    np.testing.assert_allclose(np.asarray(iterative.eigenvalues), np.asarray(dense.eigenvalues), rtol=2.0e-8)
    np.testing.assert_allclose(
        np.asarray(iterative.basis @ iterative.basis.T),
        np.asarray(dense.basis @ dense.basis.T),
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_linear_constraint_fixture_projection_matches_full_kkt():
    stiffness = jnp.array(
        [
            [8.0, -2.0, 0.0, 0.0],
            [-2.0, 7.0, -1.0, 0.0],
            [0.0, -1.0, 6.0, -1.5],
            [0.0, 0.0, -1.5, 5.0],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(4, dtype=jnp.float64)
    fixture = ff.ReferencePointFixture(
        "preload",
        ff.RBE3Patch(
            dofs=jnp.array([[1], [2]], dtype=jnp.int32),
            weights=jnp.ones((2,), dtype=jnp.float64),
        ),
        reference_dofs=jnp.array([4], dtype=jnp.int32),
        stiffness=12.0,
        target_displacement=0.03,
    )
    constraints = ff.linear_constraint_system_from_reference_fixtures(
        [fixture],
        n_structural_dofs=4,
        total_dofs=5,
    )
    preload_k, preload_f = ff.assemble_reference_fixture_preload([fixture], total_dofs=5)
    external_f = jnp.array([0.0, 0.0, 0.0, -0.4, 0.0], dtype=jnp.float64)
    k_full = jnp.zeros((5, 5), dtype=jnp.float64).at[:4, :4].set(stiffness) + preload_k
    f_full = external_f + preload_f
    full_u = constraints.solve(k_full, f_full, fixed_dofs=jnp.array([0]), solver="dense")

    retained = jnp.unique(jnp.concatenate([jnp.array([0, 3], dtype=jnp.int32), fixture.retained_dofs]))
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=1)
    reduced_constraints = constraints.project(cb, n_extra_dofs=1)
    k_rom = jnp.zeros((cb.n_reduced + 1, cb.n_reduced + 1), dtype=jnp.float64)
    k_rom = k_rom.at[: cb.n_reduced, : cb.n_reduced].set(cb.project_matrix(stiffness))
    k_rom = k_rom.at[cb.n_reduced, cb.n_reduced].set(preload_k[4, 4])
    f_rom = jnp.concatenate([cb.project_vector(external_f[:4]), preload_f[4:]])
    q_rom = reduced_constraints.solve(k_rom, f_rom, fixed_dofs=jnp.array([0]), solver="dense")

    np.testing.assert_allclose(np.asarray(reduced_constraints.expand(q_rom)), np.asarray(full_u), rtol=1.0e-10)
    np.testing.assert_allclose(np.asarray(reduced_constraints.residual(q_rom)), np.zeros(1), atol=1.0e-10)


def test_linear_constraint_kkt_supports_nonzero_fixed_values():
    stiffness = jnp.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    force = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
    constraints = ff.LinearConstraintSystem(jnp.array([[0.0, 1.0, -1.0]], dtype=jnp.float64))

    u = constraints.solve(
        stiffness,
        force,
        fixed_dofs=jnp.array([0], dtype=jnp.int32),
        fixed_values=jnp.array([0.25], dtype=jnp.float64),
    )

    free = np.array([1, 2])
    fixed = np.array([0])
    k_ff = np.asarray(stiffness)[np.ix_(free, free)]
    k_fc = np.asarray(stiffness)[np.ix_(free, fixed)]
    c_f = np.asarray(constraints.matrix)[:, free]
    c_c = np.asarray(constraints.matrix)[:, fixed]
    rhs = np.concatenate(
        [
            np.asarray(force)[free] - k_fc[:, 0] * 0.25,
            np.asarray(constraints.rhs) - c_c[:, 0] * 0.25,
        ]
    )
    lhs = np.block([[k_ff, c_f.T], [c_f, np.zeros((1, 1))]])
    expected = np.zeros(3)
    expected[0] = 0.25
    expected[free] = np.linalg.solve(lhs, rhs)[:2]

    np.testing.assert_allclose(np.asarray(u), expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(constraints.residual(u)), np.zeros(1), atol=1.0e-12)


def test_newmark_and_active_contact_callbacks_are_autodiff_friendly():
    mass = jnp.array([[2.0]], dtype=jnp.float64)
    stiffness = jnp.array([[8.0]], dtype=jnp.float64)
    state = ff.NewmarkState(
        q=jnp.array([0.1], dtype=jnp.float64),
        qd=jnp.array([0.0], dtype=jnp.float64),
        qdd=jnp.array([-0.4], dtype=jnp.float64),
    )
    config = ff.NewmarkConfig(dt=0.05, tol=1.0e-10, atol=1.0e-12, maxiter=8)

    def internal_force(q):
        return stiffness @ q + 0.1 * q**3

    next_state, info = ff.newmark_step(
        mass,
        None,
        internal_force,
        jnp.array([0.0], dtype=jnp.float64),
        state,
        config,
    )
    residual = ff.make_newmark_effective_residual(mass, None, internal_force, jnp.zeros(1), state, config)
    jac = jax.jacrev(residual)(next_state.q)

    assert info.converged
    assert jac.shape == (1, 1)
    np.testing.assert_allclose(np.asarray(residual(next_state.q)), np.zeros(1), atol=1.0e-10)

    class ActiveState:
        def __init__(self, active: bool):
            self.active = jnp.asarray([active])

        def changed(self, other):
            return jnp.any(self.active != other.active)

    def residual_from_contact_state(active_state):
        penalty = jnp.where(active_state.active[0], 10.0, 0.0)

        def residual_fn(x):
            return jnp.array([(1.0 + penalty) * x[0] - 1.0])

        return residual_fn

    def solve_fn(residual_fn, x0):
        jac_fn = jax.jacrev(residual_fn)
        x = x0 + jnp.linalg.solve(jac_fn(x0), -residual_fn(x0))
        return x, {"ok": True}

    def update_state(x):
        return ActiveState(bool(x[0] > 0.05))

    x, active_info = ff.active_contact_fixed_point_solve(
        jnp.array([0.0], dtype=jnp.float64),
        ActiveState(False),
        residual_from_contact_state,
        solve_fn,
        update_state,
        max_active_updates=4,
    )

    assert active_info.converged
    assert active_info.iters == 2
    np.testing.assert_allclose(np.asarray(x), np.array([1.0 / 11.0]), rtol=1.0e-10)

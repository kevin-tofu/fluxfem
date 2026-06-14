import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def _flux_sparse_from_dense(matrix):
    matrix_np = np.asarray(matrix)
    rows, cols = np.nonzero(matrix_np)
    data = matrix_np[rows, cols]
    return ff.FluxSparseMatrix(
        ff.SparsityPattern(
            rows=jnp.asarray(rows, dtype=jnp.int32),
            cols=jnp.asarray(cols, dtype=jnp.int32),
            n_dofs=matrix_np.shape[0],
        ),
        jnp.asarray(data, dtype=jnp.asarray(matrix).dtype),
    )


def test_craig_bampton_basis_keeps_retained_dofs_physical():
    stiffness = jnp.array(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 4.0],
        ]
    )
    mass = jnp.eye(4)

    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0, 3]), n_modes=1)

    assert cb.basis.shape == (4, 3)
    np.testing.assert_allclose(np.asarray(cb.basis[jnp.array([0, 3]), :2]), np.eye(2))
    np.testing.assert_allclose(np.asarray(cb.basis[jnp.array([0, 3]), 2]), np.zeros(2))
    assert cb.n_retained == 2
    assert cb.n_modes == 1

def test_reduced_residual_keeps_autodiff_chain_rule():
    stiffness = jnp.array(
        [
            [5.0, -1.0, 0.0, 0.0],
            [-1.0, 5.0, -1.0, 0.0],
            [0.0, -1.0, 5.0, -1.0],
            [0.0, 0.0, -1.0, 5.0],
        ]
    )
    mass = jnp.eye(4)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0, 3]), n_modes=1)

    def full_residual(u):
        return stiffness @ u + 0.2 * u**3

    q = jnp.array([0.1, -0.2, 0.3])
    reduced_residual = ff.reduced_residual_from_full(cb, full_residual)
    reduced_jacobian = ff.reduced_jacobian_from_full(cb, full_residual)

    u = cb.expand(q)
    full_jacobian = jax.jacrev(full_residual)(u)
    expected = cb.basis.T @ full_jacobian @ cb.basis

    np.testing.assert_allclose(np.asarray(reduced_residual(q)), np.asarray(cb.basis.T @ full_residual(u)))
    np.testing.assert_allclose(np.asarray(reduced_jacobian(q)), np.asarray(expected), rtol=1e-6, atol=1e-6)

def test_craig_bampton_basis_all_retained_is_identity():
    stiffness = jnp.eye(3)
    mass = jnp.eye(3)

    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0, 1, 2]), n_modes=2)

    np.testing.assert_allclose(np.asarray(cb.basis), np.eye(3))
    assert cb.n_retained == 3
    assert cb.n_modes == 0

def test_craig_bampton_cg_constraint_solver_matches_dense():
    stiffness = jnp.array(
        [
            [6.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 7.0, -1.5, 0.0, 0.0],
            [0.0, -1.5, 8.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 7.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(5, dtype=jnp.float32)
    retained = jnp.array([0, 4])
    dense = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=2)
    cg = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=retained,
        n_modes=2,
        constraint_solver="cg",
        cg_tol=1e-8,
        cg_maxiter=50,
    )

    np.testing.assert_allclose(
        np.asarray(cg.basis[:, : dense.n_retained]),
        np.asarray(dense.basis[:, : dense.n_retained]),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(cg.basis[:, dense.n_retained :]),
        np.asarray(dense.basis[:, dense.n_retained :]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(cg.eigenvalues), np.asarray(dense.eigenvalues), rtol=1e-6, atol=1e-6)

def test_craig_bampton_accepts_custom_constraint_solver():
    stiffness = jnp.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(3, dtype=jnp.float32)
    calls = {"count": 0}

    def solver(k_ii, rhs):
        calls["count"] += 1
        return jnp.linalg.solve(k_ii, rhs)

    custom = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=jnp.array([0, 2]),
        n_modes=0,
        constraint_solver=solver,
    )
    dense = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0, 2]), n_modes=0)

    assert calls["count"] == 1
    np.testing.assert_allclose(np.asarray(custom.basis), np.asarray(dense.basis), rtol=1e-6, atol=1e-6)

def test_fixed_interface_subspace_modes_match_dense_eigenvalues():
    stiffness = jnp.array(
        [
            [7.0, -1.0, 0.0, 0.0],
            [-1.0, 6.0, -1.0, 0.0],
            [0.0, -1.0, 5.0, -1.0],
            [0.0, 0.0, -1.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.diag(jnp.array([1.0, 1.1, 0.9, 1.2], dtype=jnp.float32))
    dense_modes, dense_eigvals = ff.fixed_interface_modes(stiffness, mass, n_modes=2)
    subspace_modes, subspace_eigvals = ff.fixed_interface_modes(
        stiffness,
        mass,
        n_modes=2,
        solver="subspace",
        modal_linear_solver="cg",
        modal_oversample=2,
        modal_maxiter=40,
        modal_tol=1e-8,
        cg_tol=1e-9,
        cg_maxiter=80,
    )

    np.testing.assert_allclose(
        np.asarray(subspace_eigvals),
        np.asarray(dense_eigvals),
        rtol=2e-5,
        atol=2e-5,
    )
    overlap = np.abs(np.asarray(dense_modes.T @ mass @ subspace_modes))
    np.testing.assert_allclose(overlap, np.eye(2), rtol=2e-5, atol=2e-5)

def test_craig_bampton_subspace_modal_solver_matches_dense():
    stiffness = jnp.array(
        [
            [6.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 7.0, -1.5, 0.0, 0.0],
            [0.0, -1.5, 8.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 7.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(5, dtype=jnp.float32)
    retained = jnp.array([0, 4])
    dense = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=2)
    subspace = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=retained,
        n_modes=2,
        constraint_solver="cg",
        modal_solver="subspace",
        modal_linear_solver="cg",
        modal_oversample=1,
        modal_maxiter=50,
        modal_tol=1e-8,
        cg_tol=1e-9,
        cg_maxiter=100,
    )

    np.testing.assert_allclose(
        np.asarray(subspace.eigenvalues),
        np.asarray(dense.eigenvalues),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(subspace.basis @ subspace.basis.T),
        np.asarray(dense.basis @ dense.basis.T),
        rtol=3e-5,
        atol=3e-5,
    )

def test_craig_bampton_accepts_custom_modal_solver():
    stiffness = jnp.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(3, dtype=jnp.float32)
    calls = {"count": 0}

    def modal_solver(k_ii, m_ii, n_modes):
        calls["count"] += 1
        return ff.fixed_interface_modes(k_ii, m_ii, n_modes)

    custom = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=jnp.array([0]),
        n_modes=1,
        modal_solver=modal_solver,
    )
    dense = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0]), n_modes=1)

    assert calls["count"] == 1
    np.testing.assert_allclose(
        np.asarray(custom.eigenvalues),
        np.asarray(dense.eigenvalues),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(custom.basis), np.asarray(dense.basis), rtol=1e-6, atol=1e-6)

def test_fixed_interface_eigsh_modes_match_dense_eigenvalues():
    stiffness = jnp.array(
        [
            [9.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 8.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 7.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 6.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 5.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.diag(jnp.array([1.0, 1.1, 0.9, 1.2, 1.0, 0.8], dtype=jnp.float32))
    dense_modes, dense_eigvals = ff.fixed_interface_modes(stiffness, mass, n_modes=2)
    eigsh_modes, eigsh_eigvals = ff.fixed_interface_modes(
        stiffness,
        mass,
        n_modes=2,
        solver="eigsh",
        modal_tol=1e-9,
        modal_maxiter=200,
    )

    np.testing.assert_allclose(
        np.asarray(eigsh_eigvals),
        np.asarray(dense_eigvals),
        rtol=2e-5,
        atol=2e-5,
    )
    overlap = np.abs(np.asarray(dense_modes.T @ mass @ eigsh_modes))
    np.testing.assert_allclose(overlap, np.eye(2), rtol=3e-5, atol=3e-5)

def test_craig_bampton_eigsh_modal_solver_matches_dense():
    stiffness = jnp.array(
        [
            [7.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 8.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 9.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 8.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 7.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 6.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(6, dtype=jnp.float32)
    retained = jnp.array([0, 5])
    dense = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=2)
    eigsh = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=retained,
        n_modes=2,
        constraint_solver="cg",
        modal_solver="eigsh",
        modal_tol=1e-9,
        modal_maxiter=200,
        cg_tol=1e-9,
        cg_maxiter=100,
    )

    np.testing.assert_allclose(
        np.asarray(eigsh.eigenvalues),
        np.asarray(dense.eigenvalues),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(eigsh.basis @ eigsh.basis.T),
        np.asarray(dense.basis @ dense.basis.T),
        rtol=3e-5,
        atol=3e-5,
    )

def test_craig_bampton_flux_sparse_blocks_match_dense():
    stiffness = jnp.array(
        [
            [7.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 8.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 9.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 8.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 7.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 6.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(6, dtype=jnp.float32)
    retained = jnp.array([0, 5])
    dense = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=2)
    sparse = ff.make_craig_bampton_basis(
        _flux_sparse_from_dense(stiffness),
        _flux_sparse_from_dense(mass),
        retained_dofs=retained,
        n_modes=2,
        constraint_solver="spsolve",
        modal_solver="eigsh",
        modal_tol=1e-9,
        modal_maxiter=200,
    )

    np.testing.assert_allclose(
        np.asarray(sparse.eigenvalues),
        np.asarray(dense.eigenvalues),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sparse.basis @ sparse.basis.T),
        np.asarray(dense.basis @ dense.basis.T),
        rtol=3e-5,
        atol=3e-5,
    )

def test_craig_bampton_assembled_sparse_fe_matrices_match_dense():
    mesh = ff.StructuredHexBox(nx=3, ny=1, nz=1, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    stiffness = space.assemble_bilinear_form(ff.diffusion_form, params=1.0)
    mass = space.assemble_mass_matrix()
    coords = np.asarray(mesh.coords)
    retained_nodes = np.flatnonzero(
        np.isclose(coords[:, 0], coords[:, 0].min()) | np.isclose(coords[:, 0], coords[:, 0].max())
    )
    retained = ff.vector_dofs_from_nodes(jnp.asarray(retained_nodes, dtype=jnp.int32), dim=1)
    dense = ff.make_craig_bampton_basis(
        stiffness.to_dense(),
        mass.to_dense(),
        retained_dofs=retained,
        n_modes=2,
    )
    sparse = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=retained,
        n_modes=2,
        constraint_solver="spsolve",
        modal_solver="eigsh",
        modal_tol=1e-9,
        modal_maxiter=300,
    )

    np.testing.assert_allclose(
        np.asarray(sparse.eigenvalues),
        np.asarray(dense.eigenvalues),
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sparse.basis @ sparse.basis.T),
        np.asarray(dense.basis @ dense.basis.T),
        rtol=4e-5,
        atol=4e-5,
    )

def test_linear_constraint_system_projects_explicit_reference_dof_through_cb_rom():
    stiffness = jnp.array(
        [
            [8.0, -2.0, 0.0, 0.0],
            [-2.0, 7.0, -1.0, 0.0],
            [0.0, -1.0, 6.0, -1.5],
            [0.0, 0.0, -1.5, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(4, dtype=jnp.float32)
    cb = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=jnp.array([0, 3]),
        n_modes=2,
    )

    preload_stiffness = jnp.asarray(12.0, dtype=jnp.float32)
    target = jnp.asarray(0.03, dtype=jnp.float32)
    k_aug = jnp.zeros((5, 5), dtype=jnp.float32).at[:4, :4].set(stiffness)
    k_aug = k_aug.at[4, 4].set(preload_stiffness)
    f_aug = jnp.array([0.0, 0.0, 0.0, -0.4, preload_stiffness * target], dtype=jnp.float32)

    c_aug = jnp.array([[0.0, -0.5, -0.5, 0.0, 1.0]], dtype=jnp.float32)
    constraints = ff.LinearConstraintSystem(c_aug)
    full_u = constraints.solve(k_aug, f_aug, fixed_dofs=jnp.array([0]))

    k_rom = jnp.zeros((cb.n_reduced + 1, cb.n_reduced + 1), dtype=jnp.float32)
    k_rom = k_rom.at[: cb.n_reduced, : cb.n_reduced].set(cb.project_matrix(stiffness))
    k_rom = k_rom.at[cb.n_reduced, cb.n_reduced].set(preload_stiffness)
    f_rom = jnp.concatenate([cb.project_vector(f_aug[:4]), f_aug[4:]])
    reduced_constraints = constraints.project(cb, n_extra_dofs=1)
    q_aug = reduced_constraints.solve(k_rom, f_rom, fixed_dofs=jnp.array([0]))
    rom_u = reduced_constraints.expand(q_aug)

    np.testing.assert_allclose(np.asarray(rom_u), np.asarray(full_u), rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(np.asarray(constraints.residual(full_u)), np.zeros(1), atol=2e-6)
    np.testing.assert_allclose(np.asarray(reduced_constraints.residual(q_aug)), np.zeros(1), atol=2e-6)

def test_reference_point_fixture_builds_rbe3_mpc_and_preload():
    patch = ff.RBE3Patch(
        dofs=jnp.array([[1], [2]], dtype=jnp.int32),
        weights=jnp.array([2.0, 2.0], dtype=jnp.float32),
    )
    fixture = ff.ReferencePointFixture(
        "clamp",
        patch,
        reference_dofs=jnp.array([4], dtype=jnp.int32),
        direction=jnp.array([1.0], dtype=jnp.float32),
        stiffness=12.0,
        target_displacement=0.03,
    )

    constraints = ff.linear_constraint_system_from_reference_fixtures(
        [fixture],
        n_structural_dofs=4,
        total_dofs=5,
    )
    preload_k, preload_f = ff.assemble_reference_fixture_preload([fixture], total_dofs=5)
    sparse_preload_k, sparse_preload_f = ff.assemble_reference_fixture_preload([fixture], total_dofs=5, sparse=True)

    expected_c = jnp.array([[0.0, -0.5, -0.5, 0.0, 1.0]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(constraints.matrix), np.asarray(expected_c))
    np.testing.assert_allclose(np.asarray(preload_k[4, 4]), np.asarray(12.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(preload_f[4]), np.asarray(0.36), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sparse_preload_k.toarray(), np.asarray(preload_k), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sparse_preload_f), np.asarray(preload_f), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(fixture.retained_dofs), np.array([1, 2], dtype=np.int32))

def test_reference_point_fixture_cb_rom_matches_full_kkt():
    stiffness = jnp.array(
        [
            [8.0, -2.0, 0.0, 0.0],
            [-2.0, 7.0, -1.0, 0.0],
            [0.0, -1.0, 6.0, -1.5],
            [0.0, 0.0, -1.5, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(4, dtype=jnp.float32)
    fixture = ff.ReferencePointFixture(
        "clamp",
        ff.RBE3Patch(
            dofs=jnp.array([[1], [2]], dtype=jnp.int32),
            weights=jnp.ones((2,), dtype=jnp.float32),
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
    external_f = jnp.array([0.0, 0.0, 0.0, -0.4, 0.0], dtype=jnp.float32)
    k_full = jnp.zeros((5, 5), dtype=jnp.float32).at[:4, :4].set(stiffness) + preload_k
    f_full = external_f + preload_f
    full_u = constraints.solve(k_full, f_full, fixed_dofs=jnp.array([0]))

    retained = jnp.unique(jnp.concatenate([jnp.array([0, 3], dtype=jnp.int32), fixture.retained_dofs]))
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=1)
    reduced_constraints = constraints.project(cb, n_extra_dofs=1)
    k_rom = jnp.zeros((cb.n_reduced + 1, cb.n_reduced + 1), dtype=jnp.float32)
    k_rom = k_rom.at[: cb.n_reduced, : cb.n_reduced].set(cb.project_matrix(stiffness))
    k_rom = k_rom.at[cb.n_reduced, cb.n_reduced].set(preload_k[4, 4])
    f_rom = jnp.concatenate([cb.project_vector(external_f[:4]), preload_f[4:]])
    q_rom = reduced_constraints.solve(k_rom, f_rom, fixed_dofs=jnp.array([0]))

    np.testing.assert_allclose(np.asarray(reduced_constraints.expand(q_rom)), np.asarray(full_u), rtol=4e-5, atol=4e-5)

def test_linear_constraint_kkt_sparse_solver_matches_dense():
    import scipy.sparse as sp

    stiffness = jnp.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    force = jnp.array([1.0, 0.0, -0.5], dtype=jnp.float32)
    constraints = ff.LinearConstraintSystem(jnp.array([[0.0, 1.0, -1.0]], dtype=jnp.float32))

    dense = constraints.solve(stiffness, force, fixed_dofs=jnp.array([0]), solver="dense")
    sparse = constraints.solve(sp.csr_matrix(np.asarray(stiffness)), force, fixed_dofs=jnp.array([0]), solver="spsolve")

    np.testing.assert_allclose(np.asarray(sparse), np.asarray(dense), rtol=1e-6, atol=1e-6)

def test_reduced_linear_constraint_residual_is_autodiff_friendly():
    stiffness = jnp.array(
        [
            [5.0, -1.0, 0.0],
            [-1.0, 6.0, -1.0],
            [0.0, -1.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    cb = ff.make_craig_bampton_basis(
        stiffness,
        jnp.eye(3, dtype=jnp.float32),
        retained_dofs=jnp.array([0, 2]),
        n_modes=1,
    )
    constraints = ff.LinearConstraintSystem(
        jnp.array([[0.0, -1.0, 0.0, 1.0]], dtype=jnp.float32),
        rhs=jnp.array([0.02], dtype=jnp.float32),
    )
    reduced_constraints = constraints.project(cb, n_extra_dofs=1)
    q = jnp.array([0.1, -0.2, 0.05, 0.03], dtype=jnp.float32)

    jacobian = jax.jacrev(reduced_constraints.residual)(q)
    expanded = reduced_constraints.expand(q)

    np.testing.assert_allclose(
        np.asarray(reduced_constraints.residual(q)),
        np.asarray(constraints.residual(expanded)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.asarray(reduced_constraints.matrix),
        rtol=1e-6,
        atol=1e-6,
    )

def test_newmark_step_solves_linear_reduced_dynamics():
    mass = jnp.array([[2.0]])
    damping = None
    stiffness = jnp.array([[8.0]])
    state = ff.NewmarkState(
        q=jnp.array([0.1]),
        qd=jnp.array([0.0]),
        qdd=jnp.array([-0.4]),
        t=0.0,
    )
    config = ff.NewmarkConfig(dt=0.05, tol=1e-8, atol=1e-5, maxiter=8)

    def internal_force(q):
        return stiffness @ q

    next_state, info = ff.newmark_step(
        mass, damping, internal_force, external_force=jnp.array([0.0]), state=state, config=config
    )

    residual_fn = ff.make_newmark_effective_residual(
        mass, damping, internal_force, jnp.array([0.0]), state, config
    )
    assert info.converged
    assert np.linalg.norm(np.asarray(residual_fn(next_state.q))) < 1e-5

def test_newmark_effective_residual_is_autodiff_friendly():
    mass = jnp.eye(2)
    damping = 0.1 * jnp.eye(2)
    state = ff.NewmarkState(
        q=jnp.array([0.2, -0.1]),
        qd=jnp.array([0.0, 0.3]),
        qdd=jnp.array([0.1, -0.2]),
    )
    config = ff.NewmarkConfig(dt=0.1)

    def internal_force(q):
        return jnp.array([q[0] ** 3 + q[1], q[1] ** 3 - q[0]])

    residual_fn = ff.make_newmark_effective_residual(
        mass, damping, internal_force, jnp.array([1.0, -0.5]), state, config
    )
    q_next = jnp.array([0.25, -0.05])
    jacobian = jax.jacrev(residual_fn)(q_next)

    assert jacobian.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(jacobian)))

def test_contact_residual_composes_with_cb_reduction():
    stiffness = jnp.diag(jnp.array([4.0, 5.0, 6.0]))
    mass = jnp.eye(3)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0]), n_modes=1)

    def structural(u):
        return stiffness @ u

    contact = ff.make_unilateral_plane_contact_residual(
        n_dofs=3,
        contact_dofs=jnp.array([[0]]),
        normals=jnp.array([[1.0]]),
        gaps0=jnp.array([-0.1]),
        penalty=20.0,
        smoothing=1e-3,
    )
    full_residual = ff.compose_residuals(structural, contact)
    reduced_residual = ff.reduced_residual_from_full(cb, full_residual)
    jacobian = jax.jacrev(reduced_residual)(jnp.array([0.0, 0.1]))

    assert reduced_residual(jnp.array([0.0, 0.1])).shape == (2,)
    assert jacobian.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(jacobian)))

def test_frozen_contact_friction_rom_objective_is_autodiff_friendly():
    stiffness = jnp.array(
        [
            [8.0, -1.0, 0.0, 0.0],
            [-1.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, -1.0],
            [0.0, 0.0, -1.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(4, dtype=jnp.float32)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0, 1]), n_modes=1)
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=4,
            dofs=jnp.array([[0, 1]]),
            normals=jnp.array([[0.0, 1.0]], dtype=jnp.float32),
            gaps0=jnp.array([-0.06], dtype=jnp.float32),
        ),
        penalty=25.0,
    )

    q_ref = jnp.array([0.08, -0.02, 0.1], dtype=jnp.float32)
    u_ref = cb.expand(q_ref)
    manager = ff.TangentialPenaltyFrictionManager(
        mu=0.4,
        tangential_penalty=6.0,
        previous_displacement=jnp.zeros(4, dtype=jnp.float32),
    ).advance(contact, u_ref)
    snapshot = manager.snapshot(contact, u_ref)

    def full_residual(u):
        return stiffness @ u + snapshot.residual()(u)

    reduced_residual = ff.reduced_residual_from_full(cb, full_residual)

    def objective(q):
        r = reduced_residual(q)
        return 0.5 * jnp.dot(r, r)

    grad = jax.grad(objective)(q_ref)
    reduced_tangent = cb.basis.T @ jax.jacrev(full_residual)(u_ref) @ cb.basis
    expected_grad = reduced_tangent.T @ reduced_residual(q_ref)
    finite_diff_grad0 = (objective(q_ref.at[0].add(1e-3)) - objective(q_ref.at[0].add(-1e-3))) / 2e-3

    assert snapshot.history is not None
    assert bool(snapshot.active_state.active[0])
    assert grad.shape == q_ref.shape
    assert np.all(np.isfinite(np.asarray(grad)))
    np.testing.assert_allclose(np.asarray(grad), np.asarray(expected_grad), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(grad[0]), np.asarray(finite_diff_grad0), rtol=2e-3, atol=2e-3)

def test_active_contact_fixed_point_solve_updates_active_set():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]]),
            gaps0=jnp.array([0.1]),
            n_dofs=1,
        ),
        penalty=10.0,
    )
    initial_contact_state = contact.state_from_displacement(jnp.array([0.0]))

    def structural(u):
        return jnp.array([u[0] + 0.2])

    def residual_from_state(contact_state):
        return ff.compose_residuals(structural, contact.residual_with_state(contact_state))

    def solve_fn(residual_fn, x0):
        def jac(x):
            return jax.jacrev(residual_fn)(x)

        x = x0
        for _ in range(4):
            r = residual_fn(x)
            dx = jnp.linalg.solve(jac(x), -r)
            x = x + dx
        return x, {"residual_norm": float(jnp.linalg.norm(residual_fn(x)))}

    solution, info = ff.active_contact_fixed_point_solve(
        jnp.array([0.0]),
        initial_contact_state,
        residual_from_state,
        solve_fn,
        contact.state_from_displacement,
        max_active_updates=4,
    )

    assert info.converged
    assert info.iters == 2
    np.testing.assert_array_equal(np.asarray(info.contact_state.active), np.array([True]))
    np.testing.assert_allclose(np.asarray(solution), np.array([-1.2 / 11.0]), atol=1e-6)

def test_active_contact_newmark_step_updates_active_set():
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]]),
            gaps0=jnp.array([0.01]),
            n_dofs=1,
        ),
        penalty=10.0,
    )
    initial_contact_state = contact.state_from_displacement(jnp.array([0.0]))
    state = ff.NewmarkState(
        q=jnp.array([0.0]),
        qd=jnp.array([0.0]),
        qdd=jnp.array([0.0]),
        t=0.0,
    )
    config = ff.NewmarkConfig(dt=1.0, tol=1e-10, atol=1e-5, maxiter=8)

    def structural(q):
        return jnp.array([q[0] + 0.2])

    def internal_force_from_state(contact_state):
        return ff.compose_residuals(structural, contact.residual_with_state(contact_state))

    next_state, info = ff.active_contact_newmark_step(
        jnp.eye(1),
        None,
        internal_force_from_state,
        external_force=jnp.array([0.0]),
        state=state,
        config=config,
        initial_contact_state=initial_contact_state,
        update_contact_state=contact.state_from_displacement,
        max_active_updates=4,
    )

    assert info.converged
    assert info.iters == 2
    assert len(info.step_infos) == 2
    assert all(step.converged for step in info.step_infos)
    np.testing.assert_array_equal(np.asarray(info.contact_state.active), np.array([True]))
    np.testing.assert_allclose(np.asarray(next_state.q), np.array([-0.02]), atol=1e-6)

def test_reduced_contact_dynamics_facade_runs_search_friction_newmark_step():
    coords = np.array([[0.25, 0.04], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    slave = ff.make_surface_from_facets(coords, np.array([[0]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[1, 2]], dtype=np.int32))
    n_full = 6
    stiffness = 2.0 * jnp.eye(n_full, dtype=jnp.float32)
    mass = jnp.eye(n_full, dtype=jnp.float32)
    cb = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=ff.vector_dofs_from_nodes(jnp.array([0, 1, 2]), dim=2),
        n_modes=1,
    )
    search_manager = ff.make_node_surface_contact_search_manager(
        slave,
        master,
        dim=2,
        n_total_nodes=3,
        search_radius=0.2,
        skin=0.1,
        penalty=20.0,
        normal=jnp.array([0.0, 1.0], dtype=jnp.float32),
        cell_size=1.0,
    )
    dynamics = ff.ReducedContactDynamics(
        cb=cb,
        stiffness=stiffness,
        mass=mass,
        damping=0.02 * mass,
        search_manager=search_manager,
        friction_manager=ff.TangentialPenaltyFrictionManager(
            mu=0.4,
            tangential_penalty=5.0,
            previous_displacement=jnp.zeros(n_full, dtype=jnp.float32),
        ),
    )
    state = ff.NewmarkState(
        q=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
        qd=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
        qdd=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
    )
    external_force = jnp.zeros(n_full, dtype=jnp.float32).at[0].set(0.2).at[1].set(-1.0)

    next_state, info = dynamics.active_newmark_step(
        external_force,
        state,
        ff.NewmarkConfig(dt=1.0, tol=1e-8, atol=1e-5, maxiter=20),
        max_active_updates=6,
    )

    assert info.converged
    assert isinstance(info.contact_state, ff.FrictionalContactUpdateSnapshot)
    assert dynamics.friction_manager.history is not None
    assert dynamics.search_manager.search_cache is not None
    assert int(info.contact_state.contact.active_count(dynamics.expand(next_state.q))) == 1

def test_reduced_contact_dynamics_validates_manager_protocols():
    stiffness = jnp.eye(2, dtype=jnp.float32)
    mass = jnp.eye(2, dtype=jnp.float32)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0]), n_modes=1)

    class SearchManager:
        def build_contact(self, displacement):
            raise AssertionError("not used")

    class BadFrictionManager:
        def snapshot(self, contact, u):
            raise AssertionError("not used")

    with np.testing.assert_raises_regex(TypeError, "search_manager"):
        ff.ReducedContactDynamics(
            cb=cb,
            stiffness=stiffness,
            mass=mass,
            damping=None,
            search_manager=object(),
        )

    with np.testing.assert_raises_regex(TypeError, "friction_manager"):
        ff.ReducedContactDynamics(
            cb=cb,
            stiffness=stiffness,
            mass=mass,
            damping=None,
            search_manager=SearchManager(),
            friction_manager=BadFrictionManager(),
        )

def test_reduced_contact_dynamics_matches_manual_callbacks():
    coords = np.array([[0.25, 0.04], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    slave = ff.make_surface_from_facets(coords, np.array([[0]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[1, 2]], dtype=np.int32))
    n_full = 6
    stiffness = 2.0 * jnp.eye(n_full, dtype=jnp.float32)
    mass = jnp.eye(n_full, dtype=jnp.float32)
    damping = 0.02 * mass
    cb = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=ff.vector_dofs_from_nodes(jnp.array([0, 1, 2]), dim=2),
        n_modes=1,
    )
    state = ff.NewmarkState(
        q=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
        qd=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
        qdd=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
    )
    config = ff.NewmarkConfig(dt=1.0, tol=1e-8, atol=1e-5, maxiter=20)
    external_force = jnp.zeros(n_full, dtype=jnp.float32).at[0].set(0.2).at[1].set(-1.0)

    def make_search_manager():
        return ff.make_node_surface_contact_search_manager(
            slave,
            master,
            dim=2,
            n_total_nodes=3,
            search_radius=0.2,
            skin=0.1,
            penalty=20.0,
            normal=jnp.array([0.0, 1.0], dtype=jnp.float32),
            cell_size=1.0,
        )

    def make_friction_manager():
        return ff.TangentialPenaltyFrictionManager(
            mu=0.4,
            tangential_penalty=5.0,
            previous_displacement=jnp.zeros(n_full, dtype=jnp.float32),
        )

    dynamics = ff.ReducedContactDynamics(
        cb=cb,
        stiffness=stiffness,
        mass=mass,
        damping=damping,
        search_manager=make_search_manager(),
        friction_manager=make_friction_manager(),
    )
    facade_state, facade_info = dynamics.active_newmark_step(
        external_force,
        state,
        config,
        max_active_updates=6,
    )

    search_manager = {"value": make_search_manager()}
    friction_manager = {"value": make_friction_manager()}

    def build_contact(u_full):
        contact, next_manager = search_manager["value"].build_contact(u_full)
        search_manager["value"] = next_manager
        return contact

    def build_snapshot(q):
        u_full = cb.expand(q)
        return friction_manager["value"].snapshot(build_contact(u_full), u_full)

    def internal_force_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def full_residual(u):
            return stiffness @ u + contact_residual(u)

        return ff.reduced_residual_from_full(cb, full_residual)

    manual_state, manual_info = ff.active_contact_newmark_step(
        cb.project_matrix(mass),
        cb.project_matrix(damping),
        internal_force_from_snapshot,
        cb.project_vector(external_force),
        state,
        config,
        initial_contact_state=build_snapshot(state.q),
        update_contact_state=build_snapshot,
        max_active_updates=6,
    )
    friction_manager["value"] = friction_manager["value"].advance(
        manual_info.contact_state.contact,
        cb.expand(manual_state.q),
    )

    assert facade_info.converged
    assert manual_info.converged
    np.testing.assert_allclose(np.asarray(facade_state.q), np.asarray(manual_state.q), atol=1e-6)
    np.testing.assert_allclose(np.asarray(facade_state.qd), np.asarray(manual_state.qd), atol=1e-6)
    np.testing.assert_allclose(np.asarray(facade_state.qdd), np.asarray(manual_state.qdd), atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(facade_info.contact_state.active_state.active),
        np.asarray(manual_info.contact_state.active_state.active),
    )
    np.testing.assert_allclose(
        np.asarray(dynamics.friction_manager.history.friction_force),
        np.asarray(friction_manager["value"].history.friction_force),
        atol=1e-6,
    )

def test_reduced_contact_dynamics_surface_quadrature_friction_matches_manual_callbacks():
    coords = np.array([[0.0, -0.04], [1.0, -0.04], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    slave = ff.make_surface_from_facets(coords, np.array([[0, 1]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[2, 3]], dtype=np.int32))
    n_full = 8
    stiffness = 3.0 * jnp.eye(n_full, dtype=jnp.float32)
    mass = jnp.eye(n_full, dtype=jnp.float32)
    damping = 0.01 * mass
    cb = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=ff.vector_dofs_from_nodes(jnp.array([0, 1, 2, 3]), dim=2),
        n_modes=2,
    )
    state = ff.NewmarkState(
        q=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
        qd=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
        qdd=jnp.zeros(cb.n_reduced, dtype=jnp.float32),
    )
    config = ff.NewmarkConfig(dt=0.5, tol=1e-8, atol=1e-5, maxiter=24)
    external_force = jnp.zeros(n_full, dtype=jnp.float32)
    external_force = external_force.at[0].set(0.25).at[2].set(0.10)
    external_force = external_force.at[1].set(-0.8).at[3].set(-0.3)

    def make_search_manager():
        return ff.make_surface_quadrature_contact_search_manager(
            slave,
            master,
            dim=2,
            n_total_nodes=4,
            search_radius=0.2,
            skin=0.1,
            penalty=25.0,
            normal=jnp.array([0.0, 1.0], dtype=jnp.float32),
            quadrature_rule="vertices",
            cell_size=1.0,
        )

    def make_friction_manager():
        return ff.TangentialPenaltyFrictionManager(
            mu=0.45,
            tangential_penalty=6.0,
            previous_displacement=jnp.zeros(n_full, dtype=jnp.float32),
        )

    dynamics = ff.ReducedContactDynamics(
        cb=cb,
        stiffness=stiffness,
        mass=mass,
        damping=damping,
        search_manager=make_search_manager(),
        friction_manager=make_friction_manager(),
    )
    facade_state, facade_info = dynamics.active_newmark_step(
        external_force,
        state,
        config,
        max_active_updates=6,
    )

    search_manager = {"value": make_search_manager()}
    friction_manager = {"value": make_friction_manager()}

    def build_contact(u_full):
        contact, next_manager = search_manager["value"].build_contact(u_full)
        search_manager["value"] = next_manager
        return contact

    def build_snapshot(q):
        u_full = cb.expand(q)
        return friction_manager["value"].snapshot(build_contact(u_full), u_full)

    def internal_force_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def full_residual(u):
            return stiffness @ u + contact_residual(u)

        return ff.reduced_residual_from_full(cb, full_residual)

    manual_state, manual_info = ff.active_contact_newmark_step(
        cb.project_matrix(mass),
        cb.project_matrix(damping),
        internal_force_from_snapshot,
        cb.project_vector(external_force),
        state,
        config,
        initial_contact_state=build_snapshot(state.q),
        update_contact_state=build_snapshot,
        max_active_updates=6,
    )
    friction_manager["value"] = friction_manager["value"].advance(
        manual_info.contact_state.contact,
        cb.expand(manual_state.q),
    )

    assert facade_info.converged
    assert manual_info.converged
    assert isinstance(facade_info.contact_state.contact, ff.SurfaceQuadraturePenaltyContact)
    np.testing.assert_allclose(np.asarray(facade_state.q), np.asarray(manual_state.q), atol=1e-6)
    np.testing.assert_allclose(np.asarray(facade_state.qd), np.asarray(manual_state.qd), atol=1e-6)
    np.testing.assert_allclose(np.asarray(facade_state.qdd), np.asarray(manual_state.qdd), atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(facade_info.contact_state.active_state.active),
        np.asarray(manual_info.contact_state.active_state.active),
    )
    np.testing.assert_array_equal(
        np.asarray(facade_info.contact_state.contact.kinematics.master_facet_ids),
        np.asarray(manual_info.contact_state.contact.kinematics.master_facet_ids),
    )
    np.testing.assert_allclose(
        np.asarray(dynamics.friction_manager.history.friction_force),
        np.asarray(friction_manager["value"].history.friction_force),
        atol=1e-6,
    )
    assert dynamics.search_manager.search_cache is not None

def test_cb_rom_contact_step_matches_full_order_when_all_internal_modes_kept():
    stiffness = jnp.array(
        [
            [8.0, -2.0, 0.0, 0.0, 0.0],
            [-2.0, 7.0, -1.5, 0.0, 0.0],
            [0.0, -1.5, 6.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 5.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(5, dtype=jnp.float32)
    damping = 0.03 * mass
    external_force = jnp.array([-0.30, 0.08, -0.04, 0.02, 0.01], dtype=jnp.float32)
    config = ff.NewmarkConfig(dt=0.4, tol=1e-9, atol=1e-6, maxiter=25)
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=5,
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]], dtype=jnp.float32),
            gaps0=jnp.array([-0.025], dtype=jnp.float32),
        ),
        penalty=40.0,
    )
    full_state0 = ff.NewmarkState(
        q=jnp.zeros(5, dtype=jnp.float32),
        qd=jnp.zeros(5, dtype=jnp.float32),
        qdd=jnp.zeros(5, dtype=jnp.float32),
    )

    def full_internal_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def full_internal(u):
            return stiffness @ u + contact_residual(u)

        return full_internal

    full_state, full_info = ff.active_contact_newmark_step(
        mass,
        damping,
        full_internal_from_snapshot,
        external_force,
        full_state0,
        config,
        initial_contact_state=ff.ContactUpdateSnapshot.from_contact(contact, full_state0.q),
        update_contact_state=lambda u: ff.ContactUpdateSnapshot.from_contact(contact, u),
        max_active_updates=6,
    )
    assert full_info.converged

    def solve_rom(n_modes: int):
        cb = ff.make_craig_bampton_basis(
            stiffness,
            mass,
            retained_dofs=jnp.array([0]),
            n_modes=n_modes,
        )
        q0 = jnp.zeros(cb.n_reduced, dtype=jnp.float32)
        state0 = ff.NewmarkState(q=q0, qd=jnp.zeros_like(q0), qdd=jnp.zeros_like(q0))

        def rom_internal_from_snapshot(snapshot):
            contact_residual = snapshot.residual()

            def full_internal(u):
                return stiffness @ u + contact_residual(u)

            return ff.reduced_residual_from_full(cb, full_internal)

        state, info = ff.active_contact_newmark_step(
            cb.project_matrix(mass),
            cb.project_matrix(damping),
            rom_internal_from_snapshot,
            cb.project_vector(external_force),
            state0,
            config,
            initial_contact_state=ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q0)),
            update_contact_state=lambda q: ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q)),
            max_active_updates=6,
        )
        assert info.converged
        return cb.expand(state.q)

    u_rom0 = solve_rom(0)
    u_rom1 = solve_rom(1)
    u_rom_all = solve_rom(4)
    err0 = jnp.linalg.norm(u_rom0 - full_state.q)
    err1 = jnp.linalg.norm(u_rom1 - full_state.q)
    err_all = jnp.linalg.norm(u_rom_all - full_state.q)

    assert float(err_all) < 2e-5
    assert float(err1) < float(err0)
    np.testing.assert_allclose(np.asarray(u_rom_all), np.asarray(full_state.q), atol=2e-5)

def test_cb_rom_surface_quadrature_contact_matches_full_order_with_all_modes():
    dim = 2
    coords = np.array(
        [
            [0.0, 0.04],
            [1.0, 0.04],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.25, 0.35],
            [0.75, 0.35],
        ],
        dtype=np.float32,
    )
    n_nodes = coords.shape[0]
    n_full = n_nodes * dim
    slave = ff.make_surface_from_facets(coords, np.array([[0, 1]], dtype=np.int32))
    master = ff.make_surface_from_facets(coords, np.array([[2, 3]], dtype=np.int32))
    stiffness = 1.0 * np.eye(n_full, dtype=np.float32)
    for a, b in [(0, 4), (1, 5), (4, 5), (4, 2), (5, 3), (2, 3)]:
        for d in range(dim):
            ia = a * dim + d
            ib = b * dim + d
            stiffness[ia, ia] += 18.0
            stiffness[ib, ib] += 18.0
            stiffness[ia, ib] -= 18.0
            stiffness[ib, ia] -= 18.0
    stiffness = jnp.asarray(stiffness)
    mass = jnp.eye(n_full, dtype=jnp.float32)
    damping = 0.02 * mass
    external_force = jnp.zeros(n_full, dtype=jnp.float32)
    external_force = external_force.at[0 * dim + 1].set(-2.0)
    external_force = external_force.at[1 * dim + 1].set(-2.0)
    external_force = external_force.at[0 * dim + 0].set(0.12)
    contact = ff.SurfaceQuadraturePenaltyContact(
        ff.surface_quadrature_contact_kinematics_from_surfaces(
            slave,
            master,
            dim=dim,
            n_total_nodes=n_nodes,
            normal=jnp.array([0.0, 1.0], dtype=jnp.float32),
            quadrature_rule="vertices",
        ),
        penalty=180.0,
    )
    config = ff.NewmarkConfig(dt=0.5, tol=1e-8, atol=2e-5, maxiter=30)
    full_state0 = ff.NewmarkState(
        q=jnp.zeros(n_full, dtype=jnp.float32),
        qd=jnp.zeros(n_full, dtype=jnp.float32),
        qdd=jnp.zeros(n_full, dtype=jnp.float32),
    )

    def full_internal_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def full_internal(u):
            return stiffness @ u + contact_residual(u)

        return full_internal

    full_state, full_info = ff.active_contact_newmark_step(
        mass,
        damping,
        full_internal_from_snapshot,
        external_force,
        full_state0,
        config,
        initial_contact_state=ff.ContactUpdateSnapshot.from_contact(contact, full_state0.q),
        update_contact_state=lambda u: ff.ContactUpdateSnapshot.from_contact(contact, u),
        max_active_updates=8,
    )
    assert full_info.converged

    retained = ff.retained_dofs_from_surface(
        ff.make_surface_from_facets(coords, np.array([[0, 1], [2, 3]], dtype=np.int32)),
        dim,
    )

    def solve_rom(n_modes: int):
        cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=n_modes)
        q0 = jnp.zeros(cb.n_reduced, dtype=jnp.float32)
        state0 = ff.NewmarkState(q=q0, qd=jnp.zeros_like(q0), qdd=jnp.zeros_like(q0))

        def rom_internal_from_snapshot(snapshot):
            contact_residual = snapshot.residual()

            def full_internal(u):
                return stiffness @ u + contact_residual(u)

            return ff.reduced_residual_from_full(cb, full_internal)

        state, info = ff.active_contact_newmark_step(
            cb.project_matrix(mass),
            cb.project_matrix(damping),
            rom_internal_from_snapshot,
            cb.project_vector(external_force),
            state0,
            config,
            initial_contact_state=ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q0)),
            update_contact_state=lambda q: ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q)),
            max_active_updates=8,
        )
        assert info.converged
        return cb.expand(state.q)

    u_rom0 = solve_rom(0)
    u_rom2 = solve_rom(2)
    u_rom_all = solve_rom(4)
    err0 = jnp.linalg.norm(u_rom0 - full_state.q)
    err2 = jnp.linalg.norm(u_rom2 - full_state.q)
    err_all = jnp.linalg.norm(u_rom_all - full_state.q)

    assert int(contact.active_count(full_state.q)) == 2
    assert float(err_all) < 2e-5
    assert float(err2) < float(err0)
    np.testing.assert_allclose(np.asarray(u_rom_all), np.asarray(full_state.q), atol=2e-5)

def test_1d_obstacle_penalty_contact_matches_closed_form_reference():
    stiffness = jnp.array(
        [
            [6.0, -2.0, 0.0, 0.0],
            [-2.0, 5.0, -1.5, 0.0],
            [0.0, -1.5, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    force = jnp.array([-1.0, 0.1, 0.0, 0.0], dtype=jnp.float32)
    gap0 = 0.01
    penalty = 50.0
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=4,
            dofs=jnp.array([[0]]),
            normals=jnp.array([[1.0]], dtype=jnp.float32),
            gaps0=jnp.array([gap0], dtype=jnp.float32),
        ),
        penalty=penalty,
    )
    e0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    inactive_reference = jnp.linalg.solve(stiffness, force)
    active_reference = jnp.linalg.solve(
        stiffness + penalty * jnp.outer(e0, e0),
        force - penalty * gap0 * e0,
    )
    assert float(gap0 + inactive_reference[0]) < 0.0
    assert float(gap0 + active_reference[0]) < 0.0

    def solve_static(residual_from_state, update_state, x0, initial_state):
        def newton_solve(residual_fn, x_init):
            x = x_init
            for _ in range(8):
                residual = residual_fn(x)
                jacobian = jax.jacrev(residual_fn)(x)
                x = x + jnp.linalg.solve(jacobian, -residual)
            return x, {"residual_norm": float(jnp.linalg.norm(residual_fn(x)))}

        return ff.active_contact_fixed_point_solve(
            x0,
            initial_state,
            residual_from_state,
            newton_solve,
            update_state,
            max_active_updates=6,
        )

    def full_residual_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def residual(u):
            return stiffness @ u + contact_residual(u) - force

        return residual

    initial_full = ff.ContactUpdateSnapshot.from_contact(contact, jnp.zeros(4, dtype=jnp.float32))
    full_u, full_info = solve_static(
        full_residual_from_snapshot,
        lambda u: ff.ContactUpdateSnapshot.from_contact(contact, u),
        jnp.zeros(4, dtype=jnp.float32),
        initial_full,
    )

    mass = jnp.eye(4, dtype=jnp.float32)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0]), n_modes=3)
    q0 = jnp.zeros(cb.n_reduced, dtype=jnp.float32)

    def rom_residual_from_snapshot(snapshot):
        contact_residual = snapshot.residual()

        def full_residual(u):
            return stiffness @ u + contact_residual(u) - force

        return ff.reduced_residual_from_full(cb, full_residual)

    rom_q, rom_info = solve_static(
        rom_residual_from_snapshot,
        lambda q: ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q)),
        q0,
        ff.ContactUpdateSnapshot.from_contact(contact, cb.expand(q0)),
    )
    rom_u = cb.expand(rom_q)

    assert full_info.converged
    assert rom_info.converged
    np.testing.assert_allclose(np.asarray(full_u), np.asarray(active_reference), atol=2e-6)
    np.testing.assert_allclose(np.asarray(rom_u), np.asarray(active_reference), atol=2e-6)
    np.testing.assert_array_equal(np.asarray(full_info.contact_state.active_state.active), np.array([True]))
    np.testing.assert_array_equal(np.asarray(rom_info.contact_state.active_state.active), np.array([True]))

def test_reduced_dynamic_contact_friction_newmark_objective_is_autodiff_friendly():
    stiffness = jnp.array(
        [
            [8.0, -1.0, 0.0, 0.0],
            [-1.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, -1.0],
            [0.0, 0.0, -1.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(4, dtype=jnp.float32)
    damping = 0.05 * mass
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0, 1]), n_modes=1)
    contact = ff.PlanePenaltyContact(
        ff.ContactKinematics(
            n_dofs=4,
            dofs=jnp.array([[0, 1]]),
            normals=jnp.array([[0.0, 1.0]], dtype=jnp.float32),
            gaps0=jnp.array([-0.05], dtype=jnp.float32),
        ),
        penalty=30.0,
    )
    q_prev = jnp.array([0.04, -0.01, 0.03], dtype=jnp.float32)
    q_next = jnp.array([0.07, -0.02, 0.05], dtype=jnp.float32)
    state = ff.NewmarkState(
        q=q_prev,
        qd=jnp.array([0.01, -0.02, 0.03], dtype=jnp.float32),
        qdd=jnp.array([0.0, 0.01, -0.01], dtype=jnp.float32),
    )
    config = ff.NewmarkConfig(dt=0.2, tol=1e-9, atol=1e-7, maxiter=12)
    friction_manager = ff.TangentialPenaltyFrictionManager(
        mu=0.35,
        tangential_penalty=8.0,
        previous_displacement=cb.expand(q_prev),
    ).advance(contact, cb.expand(q_next))
    snapshot = friction_manager.snapshot(contact, cb.expand(q_next))

    def full_internal(u):
        return stiffness @ u + snapshot.residual()(u)

    reduced_internal = ff.reduced_residual_from_full(cb, full_internal)
    reduced_external = cb.project_vector(jnp.array([0.2, -0.3, 0.05, 0.0], dtype=jnp.float32))
    effective_residual = ff.make_newmark_effective_residual(
        cb.project_matrix(mass),
        cb.project_matrix(damping),
        reduced_internal,
        reduced_external,
        state,
        config,
    )

    def objective(q):
        r = effective_residual(q)
        return 0.5 * jnp.dot(r, r)

    grad = jax.grad(objective)(q_next)
    jac = jax.jacrev(effective_residual)(q_next)
    expected_grad = jac.T @ effective_residual(q_next)
    finite_diff_grad0 = (objective(q_next.at[0].add(1e-3)) - objective(q_next.at[0].add(-1e-3))) / 2e-3

    assert snapshot.history is not None
    assert bool(snapshot.active_state.active[0])
    assert grad.shape == q_next.shape
    assert np.all(np.isfinite(np.asarray(grad)))
    np.testing.assert_allclose(np.asarray(grad), np.asarray(expected_grad), rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(np.asarray(grad[0]), np.asarray(finite_diff_grad0), rtol=3e-3, atol=3e-2)

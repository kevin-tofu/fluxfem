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


def test_craig_bampton_project_matrix_keeps_sparse_input_sparse():
    sp = pytest.importorskip("scipy.sparse")

    stiffness = np.array(
        [
            [5.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -0.5, 0.0],
            [0.0, -0.5, 3.5, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ],
        dtype=float,
    )
    mass = np.eye(4, dtype=float)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=np.array([0, 3]), n_modes=1)
    sparse = sp.csr_matrix(stiffness)

    class SparseOnlyMatrix:
        shape = sparse.shape
        dtype = sparse.dtype

        def to_csr(self):
            return sparse

        def toarray(self):  # pragma: no cover - should never be reached
            raise AssertionError("project_matrix should not densify sparse inputs")

        def __array__(self):  # pragma: no cover - should never be reached
            raise AssertionError("project_matrix should not convert sparse inputs through __array__")

    expected = cb.basis.T @ jnp.asarray(stiffness) @ cb.basis
    np.testing.assert_allclose(np.asarray(cb.project_matrix(sparse)), np.asarray(expected), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(cb.project_matrix(SparseOnlyMatrix())), np.asarray(expected), atol=1.0e-12)


def test_craig_bampton_project_operator_matvec_matches_projected_matrix():
    sp = pytest.importorskip("scipy.sparse")

    stiffness = np.array(
        [
            [5.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -0.5, 0.0],
            [0.0, -0.5, 3.5, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ],
        dtype=float,
    )
    mass = np.eye(4, dtype=float)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=np.array([0, 3]), n_modes=1)
    q = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float64)
    k_red = cb.project_matrix(stiffness)
    sparse = sp.csr_matrix(stiffness)

    class SparseOnlyOperator:
        shape = sparse.shape

        def matvec(self, vector):
            return sparse @ np.asarray(vector)

        def to_csr(self):  # pragma: no cover - should never be reached
            raise AssertionError("project_operator should prefer matvec over sparse conversion")

        def toarray(self):  # pragma: no cover - should never be reached
            raise AssertionError("project_operator should use matvec for operator inputs")

        def __array__(self):  # pragma: no cover - should never be reached
            raise AssertionError("project_operator should not densify operator inputs")

    sparse_op = cb.project_operator(sparse)
    matvec_op = cb.project_operator(SparseOnlyOperator())
    callable_op = cb.project_operator(lambda vector: jnp.asarray(stiffness) @ vector)

    np.testing.assert_allclose(np.asarray(sparse_op.matvec(q)), np.asarray(k_red @ q), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(matvec_op @ q), np.asarray(k_red @ q), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(callable_op.matvec(q)), np.asarray(k_red @ q), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(sparse_op.to_dense()), np.asarray(k_red), atol=1.0e-12)
    assert sparse_op.shape == (cb.n_reduced, cb.n_reduced)
    assert ff.ProjectedReducedOperator is ff.solver.ProjectedReducedOperator


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


def test_rbe3_remote_fixture_rotation_matches_coupled_system_matrix():
    ref = jnp.array([0.2, -0.1, 0.0], dtype=jnp.float64)
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    weights = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64)
    slave_dofs = jnp.arange(9, dtype=jnp.int32).reshape(3, 3)
    fixture = ff.RBE3RemoteFixture(
        "remote",
        ref_point=ref,
        slave_coords=coords,
        slave_dofs=slave_dofs,
        weights=weights,
        include_rotation=True,
        reference_dofs=jnp.arange(9, 15, dtype=jnp.int32),
    )

    expected_local = ff.assemble_rbe3_constraint_matrix(
        np.asarray(ref),
        np.asarray(coords),
        weights=np.asarray(weights),
    )
    np.testing.assert_allclose(np.asarray(fixture.local_constraint_matrix()), expected_local, atol=1.0e-12)

    c = np.asarray(fixture.constraint_matrix(n_structural_dofs=9, total_dofs=15))
    order = np.concatenate([np.arange(9, 15), np.arange(9)])
    np.testing.assert_allclose(c[:, order], expected_local, atol=1.0e-12)


@pytest.mark.parametrize("include_rotation", [False, True])
def test_rbe3_remote_fixture_cb_projection_matches_full_kkt(include_rotation):
    n_struct = 12
    stiffness = 8.0 * jnp.eye(n_struct, dtype=jnp.float64)
    stiffness = stiffness + 0.4 * jnp.eye(n_struct, k=1, dtype=jnp.float64)
    stiffness = stiffness + 0.4 * jnp.eye(n_struct, k=-1, dtype=jnp.float64)
    mass = jnp.eye(n_struct, dtype=jnp.float64)
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    n_ref = 6 if include_rotation else 3
    direction = jnp.zeros((n_ref,), dtype=jnp.float64).at[2].set(1.0)
    if include_rotation:
        direction = direction.at[4].set(0.25)
    fixture = ff.RBE3RemoteFixture(
        "remote",
        ref_point=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        slave_coords=coords,
        slave_dofs=jnp.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=jnp.int32),
        weights=jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64),
        include_rotation=include_rotation,
        reference_dofs=n_struct + jnp.arange(n_ref, dtype=jnp.int32),
        direction=direction,
        stiffness=11.0,
        target_displacement=0.02,
    )
    constraints = ff.linear_constraint_system_from_reference_fixtures(
        [fixture],
        n_structural_dofs=n_struct,
        total_dofs=n_struct + n_ref,
    )
    k_pre, f_pre = ff.assemble_reference_fixture_preload([fixture], total_dofs=n_struct + n_ref)
    k_full = jnp.zeros((n_struct + n_ref, n_struct + n_ref), dtype=jnp.float64).at[:n_struct, :n_struct].set(stiffness) + k_pre
    force = jnp.zeros((n_struct + n_ref,), dtype=jnp.float64).at[11].set(-0.1) + f_pre
    full_u = constraints.solve(k_full, force, fixed_dofs=jnp.array([0, 1, 2], dtype=jnp.int32))

    retained = jnp.unique(jnp.concatenate([jnp.array([0, 1, 2, 11], dtype=jnp.int32), fixture.retained_dofs]))
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=retained, n_modes=2)
    reduced_constraints = constraints.project(cb, n_extra_dofs=n_ref)
    k_rom = jnp.zeros((cb.n_reduced + n_ref, cb.n_reduced + n_ref), dtype=jnp.float64)
    k_rom = k_rom.at[: cb.n_reduced, : cb.n_reduced].set(cb.project_matrix(stiffness))
    k_rom = k_rom.at[cb.n_reduced :, cb.n_reduced :].set(k_pre[n_struct:, n_struct:])
    f_rom = jnp.concatenate([cb.project_vector(force[:n_struct]), force[n_struct:]])
    fixed_rom = jnp.array([0, 1, 2], dtype=jnp.int32)
    q_rom = reduced_constraints.solve(k_rom, f_rom, fixed_dofs=fixed_rom)

    np.testing.assert_allclose(np.asarray(reduced_constraints.expand(q_rom)), np.asarray(full_u), rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(reduced_constraints.residual(q_rom)), np.zeros(n_ref), atol=1.0e-10)


def test_reduced_coupled_system_builder_matches_manual_rbe3_projection():
    stiffness = jnp.array(
        [
            [10.0, -1.0, 0.0, -0.2, 0.0, 0.0],
            [-1.0, 9.0, -0.3, 0.0, -0.2, 0.0],
            [0.0, -0.3, 8.0, 0.0, 0.0, -0.2],
            [-0.2, 0.0, 0.0, 7.0, -0.4, 0.0],
            [0.0, -0.2, 0.0, -0.4, 7.5, -0.4],
            [0.0, 0.0, -0.2, 0.0, -0.4, 8.0],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(6, dtype=jnp.float64)
    force = jnp.zeros((6,), dtype=jnp.float64).at[5].set(-0.2)
    retained = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)
    slave_dofs = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    slave_coords = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64)

    builder = ff.ReducedCoupledSystemBuilder.from_structural("workpiece", stiffness, force, mass=mass)
    cb = builder.reduce_field("workpiece", retained_dofs=retained, n_modes=0)
    builder.add_rbe3_fixture(
        "fixture",
        body="workpiece",
        ref_point=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        slave_coords=slave_coords,
        slave_dofs=slave_dofs,
        weights=jnp.ones((1,), dtype=jnp.float64),
        include_rotation=False,
        translational_stiffness=jnp.array([0.0, 0.0, 13.0], dtype=jnp.float64),
        translational_target=jnp.array([0.0, 0.0, 0.01], dtype=jnp.float64),
    )
    system = builder.build()
    q = system.solve(fixed_dofs=jnp.array([0, 1, 2], dtype=jnp.int32))

    fixture = ff.RBE3RemoteFixture(
        "fixture",
        ref_point=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        slave_coords=slave_coords,
        slave_dofs=slave_dofs,
        include_rotation=False,
        reference_dofs=6 + jnp.arange(3, dtype=jnp.int32),
        direction=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        stiffness=13.0,
        target_displacement=0.01,
    )
    constraints = ff.linear_constraint_system_from_reference_fixtures([fixture], n_structural_dofs=6, total_dofs=9)
    k_pre, f_pre = ff.assemble_reference_fixture_preload([fixture], total_dofs=9)
    reduced_constraints = constraints.project(cb, n_extra_dofs=3)
    k_manual = jnp.zeros((cb.n_reduced + 3, cb.n_reduced + 3), dtype=jnp.float64)
    k_manual = k_manual.at[: cb.n_reduced, : cb.n_reduced].set(cb.project_matrix(stiffness))
    k_manual = k_manual.at[cb.n_reduced :, cb.n_reduced :].set(k_pre[6:, 6:])
    f_manual = jnp.concatenate([cb.project_vector(force), f_pre[6:]])
    q_manual = reduced_constraints.solve(k_manual, f_manual, fixed_dofs=jnp.array([0, 1, 2], dtype=jnp.int32))

    np.testing.assert_allclose(np.asarray(q), np.asarray(q_manual), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(system.expand(q)), np.asarray(reduced_constraints.expand(q_manual)), atol=1.0e-12)


def test_reduced_coupled_system_builder_ties_multiple_reduced_fields():
    k_a = jnp.array(
        [
            [6.0, -2.0, 0.0],
            [-2.0, 5.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    k_b = jnp.array(
        [
            [5.0, -1.5, 0.0],
            [-1.5, 4.5, -1.0],
            [0.0, -1.0, 3.5],
        ],
        dtype=jnp.float64,
    )
    m_a = jnp.eye(3, dtype=jnp.float64)
    m_b = jnp.eye(3, dtype=jnp.float64)
    f_a = jnp.array([0.0, 0.2, 0.0], dtype=jnp.float64)
    f_b = jnp.array([0.0, -0.1, 0.3], dtype=jnp.float64)

    builder = ff.ReducedCoupledSystemBuilder.from_structural("part_a", k_a, f_a, mass=m_a)
    builder.register_structural("part_b", k_b, f_b, mass=m_b)
    cb_a = builder.reduce_field("part_a", retained_dofs=jnp.array([0, 1, 2], dtype=jnp.int32), n_modes=0)
    cb_b = builder.reduce_field("part_b", retained_dofs=jnp.array([0, 1, 2], dtype=jnp.int32), n_modes=0)
    builder.add_dof_tie_constraint(
        master="part_a",
        slave="part_b",
        master_dofs=jnp.array([2], dtype=jnp.int32),
        slave_dofs=jnp.array([0], dtype=jnp.int32),
    )
    system = builder.build()
    q = system.solve(fixed_dofs=system.reduced_dofs_from_full("part_a", jnp.array([0], dtype=jnp.int32)))
    u = np.asarray(system.expand(q))

    k_full = np.zeros((6, 6), dtype=float)
    k_full[:3, :3] = np.asarray(k_a)
    k_full[3:, 3:] = np.asarray(k_b)
    f_full = np.concatenate([np.asarray(f_a), np.asarray(f_b)])
    c_full = jnp.array([[0.0, 0.0, 1.0, -1.0, 0.0, 0.0]], dtype=jnp.float64)
    full = ff.LinearConstraintSystem(c_full).solve(
        jnp.asarray(k_full),
        jnp.asarray(f_full),
        fixed_dofs=jnp.array([0], dtype=jnp.int32),
    )

    assert system.primary_field == "part_a"
    assert set(system.bases) == {"part_a", "part_b"}
    assert system.field("part_b").offset == cb_a.n_reduced
    np.testing.assert_allclose(np.asarray(system.constraints.residual(q)), np.zeros(1), atol=1.0e-12)
    np.testing.assert_allclose(u, np.asarray(full), rtol=1.0e-12, atol=1.0e-12)


def test_reduced_coupled_system_builder_ties_named_retained_groups():
    k_a = jnp.array(
        [
            [6.0, -2.0, 0.0],
            [-2.0, 5.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    k_b = jnp.array(
        [
            [5.0, -1.5, 0.0],
            [-1.5, 4.5, -1.0],
            [0.0, -1.0, 3.5],
        ],
        dtype=jnp.float64,
    )
    f_a = jnp.array([0.0, 0.2, 0.0], dtype=jnp.float64)
    f_b = jnp.array([0.0, -0.1, 0.3], dtype=jnp.float64)

    builder = ff.ReducedCoupledSystemBuilder.from_structural("part_a", k_a, f_a, value_dim=1)
    builder.register_structural("part_b", k_b, f_b, value_dim=1)
    builder.retain_node_set("part_a", "support", np.array([0]))
    builder.retain_node_set("part_a", "interface", np.array([2]))
    builder.retain_node_set("part_b", "interface", np.array([0]))
    builder.retain_node_set("part_b", "free_end", np.array([2]))
    builder.reduce_field("part_a", retained_groups=["support", "interface"], n_modes=1)
    builder.reduce_field("part_b", retained_groups=["interface", "free_end"], n_modes=1)
    builder.tie_retained_groups("part_a:interface", "part_b:interface")
    system = builder.build()

    fixed = system.reduced_dofs_from_full("part_a", builder.retained_group_dofs("part_a:support"))
    q = system.solve(fixed_dofs=fixed)
    u = np.asarray(system.expand(q))

    k_full = np.zeros((6, 6), dtype=float)
    k_full[:3, :3] = np.asarray(k_a)
    k_full[3:, 3:] = np.asarray(k_b)
    f_full = np.concatenate([np.asarray(f_a), np.asarray(f_b)])
    c_full = jnp.array([[0.0, 0.0, 1.0, -1.0, 0.0, 0.0]], dtype=jnp.float64)
    full = ff.LinearConstraintSystem(c_full).solve(
        jnp.asarray(k_full),
        jnp.asarray(f_full),
        fixed_dofs=jnp.array([0], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(builder.retained_group_dofs("part_b:interface"), np.array([0], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(system.constraints.residual(q)), np.zeros(1), atol=1.0e-12)
    np.testing.assert_allclose(u, np.asarray(full), rtol=1.0e-12, atol=1.0e-12)


def test_reduced_coupled_system_builder_retains_surface_nodes():
    stiffness = jnp.eye(8, dtype=jnp.float64)
    mass = jnp.eye(8, dtype=jnp.float64)
    force = jnp.zeros((8,), dtype=jnp.float64)
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    surface = ff.make_surface_from_facets(coords, np.array([[1, 3], [3, 2]], dtype=np.int32))

    builder = ff.ReducedCoupledSystemBuilder.from_structural("body", stiffness, force, mass=mass, value_dim=2)
    retained = builder.retain_surface_nodes("body", "contact", surface, components=[1])
    cb = builder.reduce_field("body", retained_groups=["contact"], n_modes=1)

    np.testing.assert_array_equal(retained, np.array([3, 5, 7], dtype=np.int32))
    np.testing.assert_array_equal(builder.retained_group_dofs("body:contact"), retained)
    np.testing.assert_array_equal(np.asarray(cb.retained_dofs), retained)


def test_reduced_coupled_system_builder_records_contact_pair_metadata():
    stiffness = jnp.eye(8, dtype=jnp.float64)
    mass = jnp.eye(8, dtype=jnp.float64)
    force = jnp.zeros((8,), dtype=jnp.float64)
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    slave_surface = ff.make_surface_from_facets(coords, np.array([[0, 1]], dtype=np.int32))
    master_surface = ff.make_surface_from_facets(coords, np.array([[2, 3]], dtype=np.int32))

    builder = ff.ReducedCoupledSystemBuilder.from_structural("slave", stiffness, force, mass=mass, value_dim=2)
    builder.register_structural("master", stiffness, force, mass=mass, value_dim=2)
    builder.retain_surface("slave", "contact", slave_surface)
    builder.retain_surface("master", "contact", master_surface)
    builder.reduce_field("slave", retained_groups=["contact"], n_modes=1)
    builder.reduce_field("master", retained_groups=["contact"], n_modes=1)
    pair = builder.register_contact_pair(
        slave="slave:contact",
        master="master:contact",
        name="candidate",
        normal=[0.0, 1.0],
        penalty=25.0,
        search_radius=0.2,
    )
    system = builder.build()

    assert ff.ReducedContactPair is ff.solver.ReducedContactPair
    assert ff.ReducedContactPairAdapter is ff.solver.ReducedContactPairAdapter
    assert ff.ReducedContactPairDofs is ff.solver.ReducedContactPairDofs
    assert pair.name == "candidate"
    assert system.contact_pair("candidate") == system.contact_pairs[0]
    assert system.contact_pairs[0].slave_field == "slave"
    assert system.contact_pairs[0].master_group == "contact"
    assert system.contact_pairs[0].enforcement == "external"
    assert system.contact_pairs[0].params == {"normal": [0.0, 1.0], "penalty": 25.0, "search_radius": 0.2}
    np.testing.assert_array_equal(system.retained_group_dofs("slave:contact"), np.array([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(system.reduced_group_dofs("slave:contact"), np.array([0, 1, 2, 3], dtype=np.int32))
    pair_dofs = system.contact_pair_dofs("candidate")
    np.testing.assert_array_equal(pair_dofs.slave_full_dofs, np.array([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(pair_dofs.master_full_dofs, np.array([4, 5, 6, 7], dtype=np.int32))
    np.testing.assert_array_equal(pair_dofs.slave_reduced_dofs, np.array([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(pair_dofs.master_reduced_dofs, np.array([5, 6, 7, 8], dtype=np.int32))
    adapter = system.contact_pair_adapter("candidate")
    assert adapter.pair == pair_dofs.pair
    assert adapter.dofs.pair == pair_dofs.pair
    assert adapter.slave == "slave:contact"
    assert adapter.master == "master:contact"
    assert adapter.slave_field == "slave"
    assert adapter.master_field == "master"
    assert adapter.slave_group == "contact"
    assert adapter.master_group == "contact"
    assert adapter.params == {"normal": [0.0, 1.0], "penalty": 25.0, "search_radius": 0.2}
    assert adapter.param("normal") == [0.0, 1.0]
    assert adapter.param("missing", 3) == 3
    assert adapter.mortar_kwargs() == {}
    np.testing.assert_array_equal(adapter.slave_full_dofs, pair_dofs.slave_full_dofs)
    np.testing.assert_array_equal(adapter.master_full_dofs, pair_dofs.master_full_dofs)
    np.testing.assert_array_equal(adapter.slave_reduced_dofs, pair_dofs.slave_reduced_dofs)
    np.testing.assert_array_equal(adapter.master_reduced_dofs, pair_dofs.master_reduced_dofs)


def test_cb_remote_fixture_utilities_are_top_level_api():
    np.testing.assert_array_equal(
        ff.vector_dofs_from_nodes(np.array([2, 4]), dim=3),
        np.array([6, 7, 8, 12, 13, 14], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        ff.retained_dofs_from_node_sets(np.array([2]), np.array([4]), dim=3, extra_dofs=np.array([1])),
        np.array([1, 6, 7, 8, 12, 13, 14], dtype=np.int32),
    )
    np.testing.assert_allclose(
        ff.remote_reference_direction([0.0, 0.0, 1.0], include_rotation=True),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    rank = ff.validate_rbe3_remote_reference_rank(
        np.array([0.0, 0.0, 0.0]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        include_rotation=True,
    )
    assert rank == 6


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

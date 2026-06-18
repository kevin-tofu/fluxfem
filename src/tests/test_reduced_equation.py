import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def test_reduced_equation_builder_composes_field_and_coupling_residuals():
    builder = ff.ReducedEquationBuilder()
    builder.register_field("a", n_dofs=2)
    builder.register_field("b", n_dofs=1)
    builder.add_field_residual("a", lambda qa: jnp.array([qa[0] ** 2, 2.0 * qa[1]], dtype=qa.dtype))
    builder.add_field_residual("b", lambda qb: jnp.array([3.0 * qb[0]], dtype=qb.dtype))

    def spring(qa, qb):
        gap = qa[0] - qb[0]
        return {
            "a": jnp.array([5.0 * gap, 0.0], dtype=qa.dtype),
            "b": jnp.array([-5.0 * gap], dtype=qa.dtype),
        }

    builder.add_coupling_residual(("a", "b"), spring)
    problem = builder.build()
    q = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float64)

    residual = problem.residual(q)
    expected = jnp.array(
        [
            0.2**2 + 5.0 * (0.2 - 0.05),
            2.0 * -0.1,
            3.0 * 0.05 - 5.0 * (0.2 - 0.05),
        ],
        dtype=jnp.float64,
    )

    np.testing.assert_allclose(np.asarray(residual), np.asarray(expected), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(problem.jacobian(q)), np.asarray(jax.jacrev(problem.residual)(q)), atol=1.0e-12)
    np.testing.assert_array_equal(np.asarray(problem.field_vector(q, "a")), np.array([0.2, -0.1]))


def test_reduced_equation_coupling_accepts_tuple_and_flat_outputs():
    tuple_builder = ff.ReducedEquationBuilder()
    tuple_builder.register_field("a", n_dofs=1)
    tuple_builder.register_field("b", n_dofs=2)
    tuple_builder.add_coupling_residual(
        ("a", "b"),
        lambda qa, qb: (jnp.array([qa[0] + qb[0]]), jnp.array([qb[0] - qa[0], qb[1]])),
    )
    tuple_problem = tuple_builder.build()

    flat_builder = ff.ReducedEquationBuilder()
    flat_builder.register_field("a", n_dofs=1)
    flat_builder.register_field("b", n_dofs=2)
    flat_builder.add_coupling_residual(
        ("a", "b"),
        lambda qa, qb: jnp.array([qa[0] + qb[0], qb[0] - qa[0], qb[1]], dtype=qa.dtype),
    )
    flat_problem = flat_builder.build()

    q = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(tuple_problem.residual(q)), np.array([3.0, 1.0, 3.0]), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(flat_problem.residual(q)), np.array([3.0, 1.0, 3.0]), atol=1.0e-12)


def test_reduced_equation_field_can_use_craig_bampton_basis_and_autodiff():
    stiffness = jnp.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 3.0, -0.5],
            [0.0, -0.5, 2.0],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(3, dtype=jnp.float64)
    cb = ff.make_craig_bampton_basis(stiffness, mass, retained_dofs=jnp.array([0], dtype=jnp.int32), n_modes=1)

    builder = ff.ReducedEquationBuilder()
    field = builder.register_field("body", basis=cb)

    def full_residual(u):
        return stiffness @ u + 0.1 * u**3

    builder.add_field_residual("body", lambda q: cb.project_vector(full_residual(cb.expand(q))))
    problem = builder.build()
    q = jnp.array([0.2, -0.03], dtype=jnp.float64)

    np.testing.assert_array_equal((field.offset, field.n_dofs), (0, cb.n_reduced))
    np.testing.assert_allclose(np.asarray(problem.expand_field(q, "body")), np.asarray(cb.expand(q)), atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(problem.residual(q)),
        np.asarray(cb.project_vector(full_residual(cb.expand(q)))),
        atol=1.0e-12,
    )
    jac = problem.jacobian(q)
    expected = cb.basis.T @ (stiffness + jnp.diag(0.3 * cb.expand(q) ** 2)) @ cb.basis
    np.testing.assert_allclose(np.asarray(jac), np.asarray(expected), rtol=1.0e-10, atol=1.0e-12)


def test_reduced_equation_exports_are_available_from_public_api():
    assert ff.ReducedEquationBuilder is ff.solver.ReducedEquationBuilder
    assert ff.ReducedEquationProblem is ff.solver.ReducedEquationProblem
    assert ff.ReducedEquationField is ff.solver.ReducedEquationField
    assert ff.solve_reduced_equation is ff.solver.solve_reduced_equation
    assert ff.reduced_equation_newmark_step is ff.solver.reduced_equation_newmark_step


def test_solve_reduced_equation_converges_for_nonlinear_residual():
    builder = ff.ReducedEquationBuilder()
    builder.register_field("x", n_dofs=2)
    builder.add_field_residual(
        "x",
        lambda q: jnp.array([q[0] ** 2 + q[1] - 1.0, q[0] + q[1] ** 2 - 1.0], dtype=q.dtype),
    )
    problem = builder.build()

    q, info = ff.solve_reduced_equation(problem, jnp.array([0.6, 0.6], dtype=jnp.float64), tol=1.0e-12)

    assert info.converged
    assert info.iters > 0
    np.testing.assert_allclose(np.asarray(problem.residual(q)), np.zeros(2), atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(q), np.array([0.61803399, 0.61803399]), rtol=1.0e-7)


def test_solve_reduced_equation_honors_fixed_dofs_and_problem_method():
    builder = ff.ReducedEquationBuilder()
    builder.register_field("x", n_dofs=2)
    builder.add_field_residual("x", lambda q: jnp.array([q[0], q[1] - 2.0], dtype=q.dtype))
    problem = builder.build()

    q, info = problem.solve(
        jnp.array([0.5, 0.0], dtype=jnp.float64),
        fixed_dofs=jnp.array([0], dtype=jnp.int32),
        fixed_values=jnp.array([0.5], dtype=jnp.float64),
        tol=1.0e-12,
    )

    assert info.converged
    np.testing.assert_allclose(np.asarray(q), np.array([0.5, 2.0]), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(problem.residual(q)[1:]), np.zeros(1), atol=1.0e-12)


def test_solve_reduced_equation_reports_maxiter_without_convergence():
    builder = ff.ReducedEquationBuilder()
    builder.register_field("x", n_dofs=1)
    builder.add_field_residual("x", lambda q: jnp.array([jnp.exp(q[0]) + 1.0], dtype=q.dtype))
    problem = builder.build()

    _q, info = ff.solve_reduced_equation(problem, jnp.array([1.0], dtype=jnp.float64), maxiter=2)

    assert not info.converged
    assert info.iters == 2
    assert info.stop_reason == "maxiter"


def test_reduced_equation_newmark_step_matches_existing_linear_newmark():
    stiffness = jnp.array([[6.0, -1.0], [-1.0, 4.0]], dtype=jnp.float64)
    mass = jnp.array([[2.0, 0.1], [0.1, 1.5]], dtype=jnp.float64)
    damping = 0.03 * mass
    external_force = jnp.array([0.3, -0.2], dtype=jnp.float64)
    state = ff.NewmarkState(
        q=jnp.array([0.1, -0.05], dtype=jnp.float64),
        qd=jnp.array([0.02, 0.01], dtype=jnp.float64),
        qdd=jnp.array([0.0, 0.0], dtype=jnp.float64),
        t=0.2,
    )
    config = ff.NewmarkConfig(dt=0.05, tol=1.0e-12, atol=1.0e-13, maxiter=8)

    builder = ff.ReducedEquationBuilder()
    builder.register_field("u", n_dofs=2)
    builder.add_field_residual("u", lambda q: stiffness @ q)
    problem = builder.build()

    state_re, info_re = ff.reduced_equation_newmark_step(problem, mass, damping, external_force, state, config)
    state_ref, info_ref = ff.newmark_step(mass, damping, lambda q: stiffness @ q, external_force, state, config)

    assert info_re.converged
    assert info_ref.converged
    np.testing.assert_allclose(np.asarray(state_re.q), np.asarray(state_ref.q), rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(state_re.qd), np.asarray(state_ref.qd), rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(state_re.qdd), np.asarray(state_ref.qdd), rtol=1.0e-11, atol=1.0e-12)


def test_reduced_equation_newmark_step_supports_force_callback_and_fixed_dofs():
    stiffness = jnp.diag(jnp.array([4.0, 5.0], dtype=jnp.float64))
    mass = jnp.eye(2, dtype=jnp.float64)
    state = ff.NewmarkState(
        q=jnp.array([0.25, 0.0], dtype=jnp.float64),
        qd=jnp.zeros(2, dtype=jnp.float64),
        qdd=jnp.zeros(2, dtype=jnp.float64),
        t=0.0,
    )
    config = ff.NewmarkConfig(dt=0.1, tol=1.0e-12, atol=1.0e-13, maxiter=8)

    builder = ff.ReducedEquationBuilder()
    builder.register_field("u", n_dofs=2)
    builder.add_field_residual("u", lambda q: stiffness @ q)
    problem = builder.build()

    next_state, info = ff.reduced_equation_newmark_step(
        problem,
        mass,
        None,
        lambda t: jnp.array([0.0, 1.0 + t], dtype=jnp.float64),
        state,
        config,
        fixed_dofs=jnp.array([0], dtype=jnp.int32),
        fixed_values=jnp.array([0.25], dtype=jnp.float64),
    )

    assert info.converged
    effective_residual = ff.make_reduced_equation_newmark_residual(
        problem,
        mass,
        None,
        lambda t: jnp.array([0.0, 1.0 + t], dtype=jnp.float64),
        state,
        config,
    )
    np.testing.assert_allclose(np.asarray(next_state.q[0]), np.array(0.25), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(effective_residual(next_state.q)[1]), np.array(0.0), atol=1.0e-12)

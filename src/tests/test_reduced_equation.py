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

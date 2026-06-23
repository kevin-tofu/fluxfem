import numpy as np
import jax
import jax.numpy as jnp
import pytest

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def nonlinear_diffusion_residual(ctx, u_elem, kappa0):
    u_q = jnp.einsum("qa,a->q", ctx.trial.N, u_elem)
    k = kappa0 * (1.0 + u_q**2)
    grad_u = jnp.einsum("qai,a->qi", ctx.trial.gradN, u_elem)
    return jnp.einsum("q,qai,qi->qa", k, ctx.test.gradN, grad_u)


def linear_diffusion_residual(ctx, u_elem, kappa0):
    grad_u = jnp.einsum("qai,a->qi", ctx.trial.gradN, u_elem)
    return kappa0 * jnp.einsum("qai,qi->qa", ctx.test.gradN, grad_u)


def test_nonlinear_constrained_problem_enforces_rbe3_patch_constraint():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    dtype = jnp.float64
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0),
        components=[0],
        dof_per_node=1,
    )

    patch = ff.RBE3Patch(
        dofs=jnp.array([[5], [7]], dtype=jnp.int32),
        weights=jnp.array([0.25, 0.75], dtype=dtype),
    )
    target = jnp.array([0.08], dtype=dtype)
    problem = ff.NonlinearConstrainedProblem(
        space=space,
        residual_form=nonlinear_diffusion_residual,
        params=1.0,
        dirichlet=(np.asarray(dir_dofs, dtype=int), np.zeros(len(dir_dofs))),
        dtype=dtype,
    )
    problem.add_rbe3_patch_constraint(patch, rhs=target)
    problem.add_local_force([7], [0.02])

    result = problem.solve(tol=1.0e-10, atol=1.0e-10, maxiter=20)

    assert result.info.converged
    u = np.asarray(result.u)
    weighted = 0.25 * u[5] + 0.75 * u[7]
    np.testing.assert_allclose(weighted, float(target[0]), atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(problem.constraint_system().residual(result.u)), np.zeros(1), atol=1.0e-10)
    assert result.multipliers.shape == (1,)


def test_solve_nonlinear_constrained_kkt_accepts_explicit_constraint_system():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    dtype = jnp.float64
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0),
        components=[0],
        dof_per_node=1,
    )
    matrix = jnp.zeros((1, space.n_dofs), dtype=dtype).at[0, 7].set(1.0)
    constraints = ff.LinearConstraintSystem(matrix, jnp.array([0.05], dtype=dtype))

    result = ff.solve_nonlinear_constrained_kkt(
        space,
        linear_diffusion_residual,
        jnp.zeros(space.n_dofs, dtype=dtype),
        1.0,
        constraints=constraints,
        dirichlet=(np.asarray(dir_dofs, dtype=int), np.zeros(len(dir_dofs))),
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=20,
    )

    assert result.info.converged
    np.testing.assert_allclose(np.asarray(result.u)[7], 0.05, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(constraints.residual(result.u)), np.zeros(1), atol=1.0e-10)


def test_nonlinear_constrained_problem_accepts_newton_loop_config():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    dtype = jnp.float64
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0),
        components=[0],
        dof_per_node=1,
    )
    matrix = jnp.zeros((1, space.n_dofs), dtype=dtype).at[0, 7].set(1.0)
    problem = ff.NonlinearConstrainedProblem(
        space=space,
        residual_form=linear_diffusion_residual,
        params=1.0,
        dirichlet=(np.asarray(dir_dofs, dtype=int), np.zeros(len(dir_dofs))),
        constraints=[ff.LinearConstraintSystem(matrix, jnp.array([0.03], dtype=dtype))],
        dtype=dtype,
    )

    result = problem.solve(config=ff.NewtonLoopConfig(tol=1.0e-10, atol=1.0e-10, maxiter=20))

    assert result.info.converged
    np.testing.assert_allclose(np.asarray(problem.constraint_system().residual(result.u)), np.zeros(1), atol=1.0e-10)
    assert result.info.tol == 1.0e-10


def test_nonlinear_constrained_problem_rejects_unsupported_newton_loop_config():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    problem = ff.NonlinearConstrainedProblem(
        space=space,
        residual_form=linear_diffusion_residual,
        params=1.0,
    )

    with pytest.raises(NotImplementedError, match="line_search"):
        problem.solve(config=ff.NewtonLoopConfig(line_search=True))
    with pytest.raises(NotImplementedError, match="linear_solver"):
        problem.solve(config=ff.NewtonLoopConfig(linear_solver="cg"))


def test_nonlinear_constrained_problem_load_steps_match_single_step_linear_solution():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    dtype = jnp.float64
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0),
        components=[0],
        dof_per_node=1,
    )
    matrix = jnp.zeros((1, space.n_dofs), dtype=dtype).at[0, 7].set(1.0)
    external = jnp.zeros((space.n_dofs,), dtype=dtype).at[5].set(0.02)
    problem = ff.NonlinearConstrainedProblem(
        space=space,
        residual_form=linear_diffusion_residual,
        params=1.0,
        dirichlet=(np.asarray(dir_dofs, dtype=int), np.zeros(len(dir_dofs))),
        external_vector=external,
        constraints=[ff.LinearConstraintSystem(matrix, jnp.array([0.03], dtype=dtype))],
        dtype=dtype,
    )

    single = problem.solve(config=ff.NewtonLoopConfig(tol=1.0e-10, atol=1.0e-10, maxiter=20))
    stepped = problem.solve(config=ff.NewtonLoopConfig(tol=1.0e-10, atol=1.0e-10, maxiter=20, n_steps=3))

    assert stepped.info.converged
    assert stepped.load_factors == (1.0 / 3.0, 2.0 / 3.0, 1.0)
    assert len(stepped.step_infos) == 3
    np.testing.assert_allclose(np.asarray(stepped.u), np.asarray(single.u), atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(stepped.multipliers), np.asarray(single.multipliers), atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(problem.constraint_system().residual(stepped.u)), np.zeros(1), atol=1.0e-10)


def test_nonlinear_constrained_problem_gmres_matches_direct_kkt_solution():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    dtype = jnp.float64
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0),
        components=[0],
        dof_per_node=1,
    )
    matrix = jnp.zeros((1, space.n_dofs), dtype=dtype).at[0, 7].set(1.0)
    external = jnp.zeros((space.n_dofs,), dtype=dtype).at[5].set(0.015)
    problem = ff.NonlinearConstrainedProblem(
        space=space,
        residual_form=linear_diffusion_residual,
        params=1.0,
        dirichlet=(np.asarray(dir_dofs, dtype=int), np.zeros(len(dir_dofs))),
        external_vector=external,
        constraints=[ff.LinearConstraintSystem(matrix, jnp.array([0.04], dtype=dtype))],
        dtype=dtype,
    )

    direct = problem.solve(config=ff.NewtonLoopConfig(tol=1.0e-10, atol=1.0e-10, maxiter=20))
    gmres = problem.solve(
        config=ff.NewtonLoopConfig(
            tol=1.0e-10,
            atol=1.0e-10,
            maxiter=20,
            linear_solver="gmres",
            linear_tol=1.0e-12,
            linear_maxiter=100,
        )
    )

    assert gmres.info.converged
    assert gmres.info.linear_converged
    assert gmres.info.linear_iters is not None
    assert gmres.info.linear_residual is not None
    np.testing.assert_allclose(np.asarray(gmres.u), np.asarray(direct.u), atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(gmres.multipliers), np.asarray(direct.multipliers), atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(problem.constraint_system().residual(gmres.u)), np.zeros(1), atol=1.0e-10)

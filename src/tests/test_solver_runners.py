"""Solver runner integration checks (linear/nonlinear)."""
import numpy as np
import pytest
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_linear_solve_runner_matches_direct():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    D = ff.isotropic_3d_D(100.0, 0.3)
    f_vec = jnp.array([0.0, 0.0, -1.0])
    F_ext = np.asarray(jnp.tile(f_vec, space.n_dofs // 3 + 1)[: space.n_dofs])

    # clamp x=min
    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    dir_vals = np.zeros(len(dir_dofs))

    analysis = ff.LinearAnalysis(
        space=space,
        bilinear_form=ff.linear_elasticity_form,
        params=D,
        base_rhs_vector=F_ext,
        dirichlet=(dir_dofs, dir_vals),
    )
    cfg = ff.LinearSolveConfig(method="spsolve")
    runner = ff.LinearSolveRunner(analysis, cfg)
    u, history = runner.run()
    assert history[-1].info.converged

    # simple consistency: clamp dofs remain zero
    u_np = np.asarray(u)
    np.testing.assert_allclose(u_np[dir_dofs], 0.0, atol=1e-10)


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_linear_solve_runner_petsc_shell():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    D = ff.isotropic_3d_D(100.0, 0.3)
    f_vec = jnp.array([0.0, 0.0, -1.0])
    F_ext = np.asarray(jnp.tile(f_vec, space.n_dofs // 3 + 1)[: space.n_dofs])

    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    dir_vals = np.zeros(len(dir_dofs))

    analysis = ff.LinearAnalysis(
        space=space,
        bilinear_form=ff.linear_elasticity_form,
        params=D,
        base_rhs_vector=F_ext,
        dirichlet=(dir_dofs, dir_vals),
    )
    cfg = ff.LinearSolveConfig(
        method="petsc_shell",
        tol=1e-8,
        maxiter=200,
        preconditioner="diag0",
        ksp_type="cg",
    )
    runner = ff.LinearSolveRunner(analysis, cfg)
    u, history = runner.run()
    assert history[-1].info.converged
    u_np = np.asarray(u)
    np.testing.assert_allclose(u_np[dir_dofs], 0.0, atol=1e-10)


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_linear_solve_runner_petsc_shell_config_overrides_options():
    import petsc4py
    petsc4py.init([])
    from petsc4py import PETSc

    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    D = ff.isotropic_3d_D(100.0, 0.3)
    f_vec = jnp.array([0.0, 0.0, -1.0])
    F_ext = np.asarray(jnp.tile(f_vec, space.n_dofs // 3 + 1)[: space.n_dofs])

    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    dir_vals = np.zeros(len(dir_dofs))

    analysis = ff.LinearAnalysis(
        space=space,
        bilinear_form=ff.linear_elasticity_form,
        params=D,
        base_rhs_vector=F_ext,
        dirichlet=(dir_dofs, dir_vals),
    )

    opts = PETSc.Options()
    opts["fluxfem_ksp_max_it"] = "1"
    try:
        cfg = ff.LinearSolveConfig(
            method="petsc_shell",
            tol=1e-8,
            maxiter=200,
            preconditioner="diag0",
            ksp_type="cg",
            ksp_max_it=200,
        )
        runner = ff.LinearSolveRunner(analysis, cfg)
        u, history = runner.run()
        assert history[-1].info.converged
        assert history[-1].info.linear_iters is None or history[-1].info.linear_iters > 1
        u_np = np.asarray(u)
        np.testing.assert_allclose(u_np[dir_dofs], 0.0, atol=1e-10)
    finally:
        try:
            del opts["fluxfem_ksp_max_it"]
        except Exception:
            pass


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_linear_solve_runner_petsc_shell_max_it_limit():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    D = ff.isotropic_3d_D(100.0, 0.3)
    f_vec = jnp.array([0.0, 0.0, -1.0])
    F_ext = np.asarray(jnp.tile(f_vec, space.n_dofs // 3 + 1)[: space.n_dofs])

    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    dir_vals = np.zeros(len(dir_dofs))

    analysis = ff.LinearAnalysis(
        space=space,
        bilinear_form=ff.linear_elasticity_form,
        params=D,
        base_rhs_vector=F_ext,
        dirichlet=(dir_dofs, dir_vals),
    )
    cfg = ff.LinearSolveConfig(
        method="petsc_shell",
        tol=1e-12,
        maxiter=1,
        preconditioner="diag0",
        ksp_type="cg",
        ksp_max_it=1,
    )
    runner = ff.LinearSolveRunner(analysis, cfg)
    _u, history = runner.run()
    assert history[-1].info.converged is False


def test_newton_solve_runner_trivial_zero_load():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    # promote to float64 to avoid mixed-dtype scatter warnings
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=jnp.float64),
        conn=mesh.conn,
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    E = 10.0
    nu = 0.3
    lam, mu = ff.lame_parameters(E, nu)
    params = {"mu": mu, "lam": lam}

    # clamp x=min
    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    dir_vals = np.zeros(len(dir_dofs))

    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params=params,
        base_external_vector=jnp.zeros(space.n_dofs, dtype=jnp.float64),
        dirichlet=(dir_dofs, dir_vals),
        dtype=jnp.float64,
    )
    cfg = ff.NewtonLoopConfig(maxiter=5, n_steps=1)
    runner = ff.NewtonSolveRunner(analysis, cfg)
    u, history = runner.run(u0=jnp.zeros(space.n_dofs))
    assert history[-1].info.converged
    np.testing.assert_allclose(np.asarray(u), 0.0, atol=1e-12)


def test_solve_nonlinear_wrapper_trivial_zero_load():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=jnp.float64),
        conn=mesh.conn,
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    lam, mu = ff.lame_parameters(10.0, 0.3)
    params = {"mu": mu, "lam": lam}
    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    dir_vals = np.zeros(len(dir_dofs))

    u, history = ff.solve_nonlinear(
        space,
        ff.neo_hookean_residual_form,
        params,
        base_external_vector=jnp.zeros(space.n_dofs, dtype=jnp.float64),
        dirichlet=(dir_dofs, dir_vals),
        dtype=jnp.float64,
        maxiter=5,
        n_steps=1,
        u0=jnp.zeros(space.n_dofs),
    )
    assert history[-1].info.converged
    np.testing.assert_allclose(np.asarray(u), 0.0, atol=1e-12)


@pytest.mark.skipif(not ff.petsc_is_available(), reason="petsc4py is required for PETSc shell tests")
def test_newton_solve_runner_petsc_shell_small_load():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    mesh = mesh.__class__(
        coords=jnp.asarray(mesh.coords, dtype=jnp.float64),
        conn=mesh.conn,
        cell_tags=getattr(mesh, "cell_tags", None),
        node_tags=getattr(mesh, "node_tags", None),
    )
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    lam, mu = ff.lame_parameters(10.0, 0.3)
    params = {"mu": mu, "lam": lam}

    xmin = float(np.asarray(mesh.coords)[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
    )
    dir_vals = np.zeros(len(dir_dofs))

    F = jnp.zeros(space.n_dofs, dtype=jnp.float64)
    F = F.at[-1].set(1e-4)

    analysis = ff.NonlinearAnalysis(
        space=space,
        residual_form=ff.neo_hookean_residual_form,
        params=params,
        base_external_vector=F,
        dirichlet=(dir_dofs, dir_vals),
        dtype=jnp.float64,
    )
    cfg = ff.NewtonLoopConfig(
        maxiter=3,
        n_steps=1,
        tol=1e-4,
        linear_solver="petsc_shell",
        linear_preconditioner=None,
        petsc_ksp_type="preonly",
        petsc_pc_type="ilu",
        petsc_use_pmat=True,
    )
    runner = ff.NewtonSolveRunner(analysis, cfg)
    u, history = runner.run(u0=jnp.zeros(space.n_dofs, dtype=jnp.float64))

    assert history[-1].info.converged
    assert history[-1].info.linear_iters == 1
    assert history[-1].info.linear_converged is True
    assert history[-1].info.linear_solve_time is not None
    assert float(np.linalg.norm(np.asarray(u))) > 0.0

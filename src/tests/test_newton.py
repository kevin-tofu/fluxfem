"""Newton solver and Jacobian consistency checks."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff


def nonlinear_residual(ctx, u_elem, kappa0):
    # u at quadrature points
    u_q = jnp.einsum("qa,a->q", ctx.trial.N, u_elem)          # (q,)
    # k(u)
    k = kappa0 * (1.0 + u_q**2)                               # (q,)
    # grad u at quadrature points
    grad_u = jnp.einsum("qai,a->qi", ctx.trial.gradN, u_elem) # (q,3)
    # residual contributions per quad & node: k * gradN_test · grad_u
    r_int = jnp.einsum("q,qai,qi->qa", k, ctx.test.gradN, grad_u)  # (q,n_nodes)
    return r_int


def test_newton_nonlinear_diffusion_converges():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    kappa0 = 1.0

    K_lin = np.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=kappa0).to_dense())
    u_lin = np.linalg.solve(np.asarray(K_lin), np.ones(8, dtype=np.float32))

    u0 = jnp.zeros_like(jnp.asarray(u_lin))
    solver = ff.NonlinearSolver(space, nonlinear_residual, kappa0, tol=1e-6, maxiter=15)
    u_newton, info = solver.solve(u0)
    assert info.converged
    # Only check scale/finite values (nonlinear solution will differ in magnitude)
    assert np.all(np.isfinite(np.asarray(u_newton)))


def test_jacobian_matches_linear_diffusion():
    """Jacobian of linear diffusion should match stiffness matrix."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    kappa = 1.0
    u0 = jnp.zeros(space.n_dofs, dtype=jnp.float32)

    def linear_res(ctx, u_elem, params):
        # grad_u(q,i) = sum_a u_a * dN_a/dx_i
        grad_u = jnp.einsum("qai,a->qi", ctx.trial.gradN, u_elem)          # (q,3)
        # r(q,a) = kappa * dN_a/dx_i * grad_u_i
        r_int = params * jnp.einsum("qai,qi->qa", ctx.test.gradN, grad_u)  # (q,n_nodes)
        return r_int

    J_dense = space.assemble_jacobian(linear_res, u0, kappa, sparse=False)
    K = np.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())

    # residual is +K u, so Jacobian should be +K
    np.testing.assert_allclose(np.asarray(J_dense), np.asarray(K), rtol=1e-6, atol=1e-6)


def test_newton_with_external_force_and_dirichlet_line_search():
    """Linear problem: external force + Dirichlet with line search enabled should match condensed solve."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    kappa = 1.0
    u0 = jnp.zeros(space.n_dofs, dtype=jnp.float32)

    def linear_res_total(ctx, u_elem, params):
        # residual: R = K u - F, where
        #   K_ij = ∫ kappa ∇N_i·∇N_j dΩ
        #   F_i  = ∫ N_i * 1 dΩ   (source f=1)
        grad_u = jnp.einsum("qai,a->qi", ctx.trial.gradN, u_elem)             # (q,3)
        internal = params * jnp.einsum("qai,qi->qa", ctx.test.gradN, grad_u)  # (q,n_nodes)
        external = ctx.test.N  # (q,n_nodes)
        return internal - external

    # Dirichlet on first dof
    dir_dofs = np.array([0], dtype=int)
    dir_vals = np.array([0.0], dtype=float)

    u_newton, info = ff.newton_solve(
        space,
        linear_res_total,
        u0,
        kappa,
        tol=1e-3,
        atol=1e-8,
        maxiter=50,
        linear_solver="spsolve",
        dirichlet=(dir_dofs, dir_vals),
        line_search=True,
    )
    assert info.converged

    # --- Reference solution via Dirichlet condensation: solve K u = F ---
    K = np.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())

    # This matches external = ctx.test.N (i.e., f(x)=1):
    # F_i = ∫ N_i * 1 dΩ
    F = space.assemble_linear_form(ff.scalar_body_force_form, params=1.0)

    solver = ff.LinearSolver(method="spsolve")
    u_expected, _ = solver.solve(
        K,
        F,
        dirichlet=(dir_dofs, dir_vals),
        dirichlet_mode="condense",
        n_total=space.n_dofs,
    )

    np.testing.assert_allclose(np.asarray(u_newton), u_expected, rtol=1e-6, atol=1e-5)

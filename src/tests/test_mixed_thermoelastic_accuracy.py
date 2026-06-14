import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import einsum


def _solve_thermoelastic_bar(nx: int):
    mesh = ff.StructuredHexBox(
        nx=nx,
        ny=1,
        nz=1,
        lx=1.0,
        ly=0.1,
        lz=0.1,
    ).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    mixed = ff.MixedSpaces(
        {
            "u": ff.NamedSpace("U", space),
            "T": ff.NamedSpace("T", space),
        }
    ).to_fe_space()

    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())
    left_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1.0e-8),
        components="x",
    )
    right_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmax, atol=1.0e-8),
        components="x",
    )
    bc = mixed.make_dirichlet(
        u=(left_dofs, None),
        T=(np.unique(np.concatenate([left_dofs, right_dofs])), None),
    )

    def res_T(v, T, p):
        return (p.kappa * h_wf.gaction(v, h_wf.grad(T)) - v * p.q) * h_wf.dOmega()

    def res_u(v, u, p):
        T_ref = ff.unknown_ref("T", space="T")
        e_x = einsum("q,i->qi", T_ref.val, p.ex)
        return (
            p.E * h_wf.gaction(v, h_wf.grad(u))
            - p.E * p.alpha * h_wf.gaction(v, e_x)
        ) * h_wf.dOmega()

    residuals = ff.make_mixed_residuals(
        u=ff.bind_mixed_residual("u", res_u, space="U"),
        T=ff.bind_mixed_residual("T", res_T, space="T"),
    )
    params = ff.Params(
        kappa=1.0,
        q=1.0,
        E=1.0,
        alpha=1.0e-3,
        ex=jnp.asarray([1.0, 0.0, 0.0]),
    )
    problem = ff.MixedProblem(
        mixed,
        residuals,
        params=params,
        pattern=mixed.get_sparsity_pattern(with_idx=True),
    )
    u0 = jnp.zeros(mixed.n_dofs)
    K = problem.assemble_jacobian(u0)
    R0 = problem.assemble_residual(u0)
    sol, _ = ff.LinearSolver(method="spsolve").solve(
        K,
        -R0,
        dirichlet=bc.as_dirichlet_bc(),
        dirichlet_mode="condense",
    )
    solution_fields = mixed.unpack_fields(sol)
    residual = np.asarray(problem.assemble_residual(sol), dtype=float)
    free = bc.free_dofs(mixed.n_dofs)

    u_nodes = np.asarray(solution_fields["u"], dtype=float)
    x_coords = coords[:, 0]
    u_tip = float(np.max(u_nodes[np.isclose(x_coords, xmax, atol=1.0e-8)]))
    u_tip_theory = 1.0e-3 / 12.0
    rel_err = abs(u_tip - u_tip_theory) / abs(u_tip_theory)
    free_res_norm = float(np.linalg.norm(residual[free]))
    return rel_err, free_res_norm


def test_mixed_thermoelastic_bar_refines_toward_theory():
    coarse_err, coarse_res = _solve_thermoelastic_bar(2)
    fine_err, fine_res = _solve_thermoelastic_bar(8)

    assert coarse_res < 1.0e-10
    assert fine_res < 1.0e-10
    assert fine_err < coarse_err
    assert fine_err < 2.0e-2

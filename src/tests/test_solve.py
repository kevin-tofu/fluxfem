"""Linear elasticity solve comparisons against scikit-fem."""
import numpy as np
import jax.numpy as jnp
import pytest
import jax
jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def test_elasticity_displacement_matches_scikit_fem():
    """Compare displacements: Dirichlet at x=0, natural at x=L, body force fx=1."""
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshHex, ElementHex1, Basis, asm
    from skfem.models.elasticity import linear_elasticity
    from skfem.helpers import dot
    try:
        from skfem.element import ElementVector as ElementVectorSKF  # type: ignore
    except Exception:
        ElementVectorSKF = None  # type: ignore

    E = 10.0
    nu = 0.25
    lam, mu = ff.lame_parameters(E, nu)
    D = ff.isotropic_3d_D(E, nu)
    f_vec = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)

    n_xyz = 2  # 3^3=27 nodes → 81 dof
    mesh_ff = ff.StructuredHexBox(nx=n_xyz, ny=n_xyz, nz=n_xyz, lx=1.0, ly=1.0, lz=1.0).build()
    space_ff = ff.make_hex_space(mesh_ff, dim=3, intorder=2)

    K_flux = np.asarray(space_ff.assemble(ff.linear_elasticity_form, params=D).to_dense())
    F_flux = np.asarray(space_ff.assemble(ff.vector_body_force_form, params=f_vec))

    coords = np.asarray(mesh_ff.coords)
    xmin = coords[:, 0].min()
    dir_dofs = mesh_ff.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
        dof_per_node=3,
    )
    dir_vals = np.zeros(len(dir_dofs))

    solver = ff.LinearSolver(method="spsolve")
    u_flux, _ = solver.solve(
        K_flux,
        F_flux,
        dirichlet=(dir_dofs, dir_vals),
        dirichlet_mode="enforce",
    )

    # LinearSolveRunner path (should match u_flux)
    analysis = ff.LinearAnalysis(
        space=space_ff,
        bilinear_form=ff.linear_elasticity_form,
        params=D,
        base_rhs_vector=F_flux,
        dirichlet=(dir_dofs, dir_vals),
    )
    cfg = ff.LinearSolveConfig(method="spsolve")
    runner = ff.LinearSolveRunner(analysis, cfg)
    u_runner, history = runner.run()
    assert history[-1].info.converged
    np.testing.assert_allclose(np.asarray(u_runner), u_flux, rtol=1e-6, atol=1e-8)

    # assemble with scikit-fem
    xs = np.linspace(0.0, 1.0, n_xyz + 1)
    ys = np.linspace(0.0, 1.0, n_xyz + 1)
    zs = np.linspace(0.0, 1.0, n_xyz + 1)
    mesh_sf = MeshHex().init_tensor(xs, ys, zs)
    if ElementVectorSKF is not None:
        element = ElementVectorSKF(ElementHex1(), dim=3)
    else:
        try:
            element = ElementHex1() * 3
        except Exception as e:
            pytest.skip(f"Vector element not available: {e}")
    basis_sf = Basis(mesh_sf, element, intorder=2)

    K_sf = asm(linear_elasticity(lam, mu), basis_sf).toarray()

    @skfem.LinearForm
    def lf(v, w):
        return dot(f_vec, v)

    F_sf = asm(lf, basis_sf)

    coords_sf = mesh_sf.p.T
    perm_nodes = []
    for c in coords:
        matches = np.nonzero(np.all(np.isclose(coords_sf, c, atol=1e-8), axis=1))[0]
        assert len(matches) == 1, "node mapping ambiguous"
        perm_nodes.append(matches[0])
    perm_nodes = np.array(perm_nodes, dtype=int)

    perm_dofs = []
    for n in perm_nodes:
        perm_dofs.extend([3 * n + 0, 3 * n + 1, 3 * n + 2])
    perm_dofs = np.array(perm_dofs, dtype=int)

    K_sf_reordered = K_sf[np.ix_(perm_dofs, perm_dofs)]
    F_sf_reordered = F_sf[perm_dofs]

    # Dirichlet DOFs in reordered system (same node indices as fluxfem)
    u_sf, _ = solver.solve(
        K_sf_reordered,
        F_sf_reordered,
        dirichlet=(dir_dofs, dir_vals),
        dirichlet_mode="enforce",
    )

    max_diff = float(np.max(np.abs(u_flux - u_sf)))
    print("elasticity solve max |u_flux - u_sf|:", max_diff)
    print("u_flux first 6 dof:", u_flux[:6])
    print("u_sf   first 6 dof:", u_sf[:6])
    assert max_diff < 1e-5, f"u mismatch vs scikit-fem: {max_diff}"

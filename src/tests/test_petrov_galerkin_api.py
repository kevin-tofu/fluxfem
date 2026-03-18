import numpy as np
import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def test_assemble_bilinear_form_named_spaces_matches_standard_galerkin_for_same_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    def form(u, v, p):
        return p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()

    compiled = ff.compile_bilinear(form)
    A_std = space.assemble_bilinear_form(compiled, ff.Params(kappa=2.0))
    A_pg = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=ff.NamedSpace("V", space), trial=ff.NamedSpace("U", space)),
        compiled,
        ff.Params(kappa=2.0),
    )

    assert isinstance(A_pg, ff.FluxSparseMatrix)
    assert np.asarray(A_pg.to_dense()).shape == (space.n_dofs, space.n_dofs)
    assert np.allclose(np.asarray(A_pg.to_dense()), np.asarray(A_std.to_dense()))


def test_assemble_bilinear_form_named_spaces_uses_named_trial_test_spaces():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    V_space = ff.make_hex_space(mesh, dim=1, intorder=2)
    U_space = ff.make_hex_space(mesh, dim=1, intorder=2)

    u = ff.trial_ref(space="U")
    v = ff.test_ref(space="V")
    p = ff.param_ref()
    expr = p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
    compiled = ff.compile_bilinear(expr)

    A = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=ff.NamedSpace("V", V_space), trial=ff.NamedSpace("U", U_space)),
        compiled,
        ff.Params(kappa=1.5),
    )

    assert A.shape == (V_space.n_dofs, U_space.n_dofs)
    assert A.meta == {"test_space": "V", "trial_space": "U"}
    assert np.asarray(A.to_dense()).shape == (V_space.n_dofs, U_space.n_dofs)


def test_assemble_bilinear_form_named_spaces_numpy_backend_matches_jax():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    V_space = ff.make_hex_space(mesh, dim=1, intorder=2)
    U_space = ff.make_hex_space(mesh, dim=1, intorder=2)

    u = ff.trial_ref(space="U")
    v = ff.test_ref(space="V")
    p = ff.param_ref()
    expr = p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
    compiled = ff.compile_bilinear(expr)

    A_jax = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=ff.NamedSpace("V", V_space), trial=ff.NamedSpace("U", U_space)),
        compiled,
        ff.Params(kappa=1.5),
        backend="jax",
    )
    A_np = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=ff.NamedSpace("V", V_space), trial=ff.NamedSpace("U", U_space)),
        compiled,
        ff.Params(kappa=1.5),
        backend="numpy",
    )

    assert A_np.shape == (V_space.n_dofs, U_space.n_dofs)
    assert np.allclose(np.asarray(A_np.to_dense()), np.asarray(A_jax.to_dense()), atol=1e-6)


def test_assemble_jacobian_named_spaces_matches_standard_galerkin_for_same_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    residual = ff.ResidualForm.volume(
        lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
    )
    rng = np.random.default_rng(0)
    u0 = rng.standard_normal(space.n_dofs)
    J_std = space.assemble_jacobian(residual.get_compiled(), u0, params=None)
    J_named = ff.assemble_jacobian(
        ff.JacobianSpaces(test=ff.NamedSpace("V", space), trial=ff.NamedSpace("U", space)),
        residual.get_compiled(),
        u0,
        params=None,
    )

    assert isinstance(J_named, ff.FluxSparseMatrix)
    assert np.allclose(np.asarray(J_named.to_dense()), np.asarray(J_std.to_dense()))

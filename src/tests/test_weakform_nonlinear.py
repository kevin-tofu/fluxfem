"""Nonlinear weak-form residual tests."""
import numpy as np
import jax.numpy as jnp
import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def test_weakform_nonlinear_residual_matches_tensor():
    """Weak-form residual matches a tensor-based reference for u^2."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    def tensor_residual(ctx: ff.FormContext, u_elem: jnp.ndarray, _p) -> jnp.ndarray:
        u_q = ctx.trial.eval(u_elem)  # (n_q,)
        return ctx.test.N * (u_q[:, None] ** 2)

    wf_residual = ff.ResidualForm.volume(
        lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
    )

    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(space.n_dofs))

    R_tensor = space.assemble_residual(tensor_residual, u, params=None)
    R_wf = space.assemble_residual(wf_residual.get_compiled(), u, params=None)

    assert np.allclose(np.asarray(R_tensor), np.asarray(R_wf))


def test_weakform_nonlinear_jacobian_matches_tensor():
    """Weak-form Jacobian matches a tensor-based reference for u^2."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    def tensor_residual(ctx: ff.FormContext, u_elem: jnp.ndarray, _p) -> jnp.ndarray:
        u_q = ctx.trial.eval(u_elem)
        return ctx.test.N * (u_q[:, None] ** 2)

    wf_residual = ff.ResidualForm.volume(
        lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
    )

    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.standard_normal(space.n_dofs))

    J_tensor = space.assemble_jacobian(tensor_residual, u, params=None).to_dense()
    J_wf = space.assemble_jacobian(wf_residual.get_compiled(), u, params=None).to_dense()

    assert np.allclose(np.asarray(J_tensor), np.asarray(J_wf))


def test_space_assemble_residual_accepts_residual_form_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.ResidualForm.volume(
        lambda v, u, p: (v * (u.val + p.alpha)) * h_wf.dOmega()
    )
    u = jnp.zeros(space.n_dofs)
    params = ff.Params(alpha=2.0)

    direct = space.assemble_residual(form, u, params)
    compiled = space.assemble_residual(form.get_compiled(), u, params)

    assert np.allclose(np.asarray(direct), np.asarray(compiled))


def test_space_assemble_jacobian_accepts_residual_form_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.ResidualForm.volume(
        lambda v, u, p: (v * (u.val + p.alpha)) * h_wf.dOmega()
    )
    u = jnp.zeros(space.n_dofs)
    params = ff.Params(alpha=2.0)

    direct = space.assemble_jacobian(form, u, params)
    compiled = space.assemble_jacobian(form.get_compiled(), u, params)

    assert np.allclose(np.asarray(direct.to_dense()), np.asarray(compiled.to_dense()))


def test_residual_spaces_matches_single_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    wf_residual = ff.ResidualForm.volume(
        lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
    )

    rng = np.random.default_rng(2)
    u = jnp.asarray(rng.standard_normal(space.n_dofs))

    R_ref = space.assemble_residual(wf_residual.get_compiled(), u, params=None)
    R_named = ff.assemble_residual(
        ff.ResidualSpaces(test=ff.NamedSpace("V", space), unknown=ff.NamedSpace("U", space)),
        wf_residual.get_compiled(),
        u,
        params=None,
    )

    assert np.allclose(np.asarray(R_named), np.asarray(R_ref))


def test_jacobian_spaces_matches_single_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    wf_residual = ff.ResidualForm.volume(
        lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
    )

    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.standard_normal(space.n_dofs))

    J_ref = space.assemble_jacobian(wf_residual.get_compiled(), u, params=None)
    J_named = ff.assemble_jacobian(
        ff.JacobianSpaces(test=ff.NamedSpace("V", space), trial=ff.NamedSpace("U", space)),
        wf_residual.get_compiled(),
        u,
        params=None,
    )

    if hasattr(J_named, "shape"):
        assert J_named.shape == (space.n_dofs, space.n_dofs)
    else:
        assert np.asarray(J_named.to_dense()).shape == (space.n_dofs, space.n_dofs)
    assert np.allclose(np.asarray(J_named.to_dense()), np.asarray(J_ref.to_dense()))


def test_named_residual_spaces_accept_residual_form_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    form = ff.ResidualForm.volume(
        lambda v, u, p: (v * (u.val + p.alpha)) * h_wf.dOmega()
    )
    u = jnp.zeros(space.n_dofs)
    params = ff.Params(alpha=3.0)

    spaces = ff.ResidualSpaces(test=ff.NamedSpace("V", space), unknown=ff.NamedSpace("U", space))
    direct = ff.assemble_residual(spaces, form, u, params)
    compiled = ff.assemble_residual(spaces, form.get_compiled(), u, params)

    assert np.allclose(np.asarray(direct), np.asarray(compiled))


def test_named_jacobian_spaces_accept_residual_form_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    form = ff.ResidualForm.volume(
        lambda v, u, p: (v * (u.val + p.alpha)) * h_wf.dOmega()
    )
    u = jnp.zeros(space.n_dofs)
    params = ff.Params(alpha=3.0)

    spaces = ff.JacobianSpaces(test=ff.NamedSpace("V", space), trial=ff.NamedSpace("U", space))
    direct = ff.assemble_jacobian(spaces, form, u, params)
    compiled = ff.assemble_jacobian(spaces, form.get_compiled(), u, params)

    assert np.allclose(np.asarray(direct.to_dense()), np.asarray(compiled.to_dense()))


def test_distinct_residual_spaces_numpy_backend_matches_jax():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    test_space = ff.make_hex_space(mesh, dim=1, intorder=2)
    unknown_space = ff.make_hex_space(mesh, dim=1, intorder=2)

    wf_residual = ff.ResidualForm.volume(
        lambda v, u, p: (v * (p.alpha * u.val)) * h_wf.dOmega()
    )

    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.standard_normal(unknown_space.n_dofs))
    spaces = ff.ResidualSpaces(
        test=ff.NamedSpace("V", test_space),
        unknown=ff.NamedSpace("U", unknown_space),
    )

    r_jax = ff.assemble_residual(spaces, wf_residual.get_compiled(), u, ff.Params(alpha=2.0), backend="jax")
    r_np = ff.assemble_residual(spaces, wf_residual.get_compiled(), u, ff.Params(alpha=2.0), backend="numpy")

    assert np.asarray(r_jax).shape == (test_space.n_dofs,)
    assert np.asarray(r_np).shape == (test_space.n_dofs,)
    assert np.allclose(np.asarray(r_np), np.asarray(r_jax))


def test_weakform_neo_hookean_residual_matches_tensor():
    """Weak-form Neo-Hookean residual matches tensor-based reference."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)

    params = {"mu": 2.0, "lam": 3.0}

    def neo_hookean_wf(v, u, p):
        F = h_wf.I(3) + u.grad
        C = h_wf.ddot(F, F)
        C_inv = h_wf.inv(C)
        logJ = h_wf.log(h_wf.det(F))
        S = p.mu * (h_wf.I(3) - C_inv) + p.lam * logJ * C_inv
        P = h_wf.matmul_std(F, S)
        # P = F @ S.T
        return h_wf.gaction(v, P) * h_wf.dOmega()

    wf_residual = ff.ResidualForm.volume(neo_hookean_wf)

    u0 = jnp.zeros(space.n_dofs)

    R_tensor = space.assemble_residual(ff.neo_hookean_residual_form, u0, params)
    R_wf = space.assemble_residual(wf_residual.get_compiled(), u0, params)

    assert np.allclose(np.asarray(R_tensor), np.asarray(R_wf), atol=1e-6)
    assert np.allclose(np.asarray(R_wf), 0.0, atol=1e-8)

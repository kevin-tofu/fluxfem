"""Linear weak-form expression tests."""
import numpy as np
import pytest
import fluxfem as ff
import fluxfem.helpers_wf as h_wf
import fluxfem.helpers_ts as h_ts


def test_weakform_mass_matches():
    """u*v expression matches mass matrix assembly for scalar space."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.BilinearForm.volume(lambda u, v, _p: h_wf.outer(v, u) * h_wf.dOmega())
    K_expr = space.assemble_bilinear_form(form.get_compiled(), params=0.0).to_dense()
    K_mass = space.assemble_mass_matrix().to_dense()

    assert np.allclose(np.asarray(K_expr), np.asarray(K_mass))


def test_bilinearform_volume_single_space_accepts_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.BilinearForm.volume(lambda u, v, p: p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega())
    params = ff.Params(kappa=2.0)

    direct = space.assemble_bilinear_form(form, params).to_dense()
    compiled = space.assemble_bilinear_form(form.get_compiled(), params).to_dense()

    assert np.allclose(np.asarray(direct), np.asarray(compiled))


def test_weakform_diffusion_matches():
    """grad(v)·grad(u) expression matches diffusion_form."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.BilinearForm.volume(
        lambda u, v, kappa: kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
    )
    K_expr = space.assemble_bilinear_form(form.get_compiled(), params=2.0).to_dense()
    K_ref = space.assemble_bilinear_form(ff.diffusion_form, params=2.0).to_dense()

    assert np.allclose(np.asarray(K_expr), np.asarray(K_ref))


def test_weakform_diffusion_operator_matches():
    """Operator version matches diffusion_form."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    params = ff.Params(kappa=2.0)
    form = ff.BilinearForm.volume(
        lambda u, v, p: p.kappa * (v.grad @ u.grad) * h_wf.dOmega()
    )
    K_expr = space.assemble_bilinear_form(form.get_compiled(), params=params).to_dense()
    K_ref = space.assemble_bilinear_form(ff.diffusion_form, params=params.kappa).to_dense()

    assert np.allclose(np.asarray(K_expr), np.asarray(K_ref))


def test_weakform_linear_elasticity_matches():
    """sym_grad-based expression matches linear_elasticity_form."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)

    D = ff.isotropic_3d_D(210_000.0, 0.3)
    form = ff.BilinearForm.volume(
        lambda u, v, D: h_wf.ddot(h_wf.sym_grad(v), h_wf.matmul_std(D, h_wf.sym_grad(u)))
        * h_wf.dOmega()
    )
    K_expr = space.assemble_bilinear_form(form.get_compiled(), params=D).to_dense()
    K_ref = space.assemble_bilinear_form(ff.linear_elasticity_form, params=D).to_dense()

    assert np.allclose(np.asarray(K_expr), np.asarray(K_ref))


def test_weakform_ddot_linear_elasticity_matches():
    """ddot(sym_grad(v), D @ sym_grad(u)) matches linear_elasticity_form."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)

    D = ff.isotropic_3d_D(210_000.0, 0.3)
    form = ff.BilinearForm.volume(
        lambda u, v, D: h_wf.ddot(h_wf.sym_grad(v), h_wf.matmul_std(D, h_wf.sym_grad(u)))
        * h_wf.dOmega()
    )
    K_expr = space.assemble_bilinear_form(form.get_compiled(), params=D).to_dense()
    K_ref = space.assemble_bilinear_form(ff.linear_elasticity_form, params=D).to_dense()

    assert np.allclose(np.asarray(K_expr), np.asarray(K_ref))


def test_weakform_multi_term_matches():
    """Multiple terms with Params match manual combination."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    params = ff.Params(rho=3.0, kappa=2.0)
    form = ff.BilinearForm.volume(
        lambda u, v, p: (
            p.rho * h_wf.outer(v, u)
            + p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u))
        ) * h_wf.dOmega()
    )
    K_expr = space.assemble_bilinear_form(form.get_compiled(), params=params).to_dense()
    K_ref = params.rho * space.assemble_mass_matrix().to_dense()
    K_ref += space.assemble_bilinear_form(ff.diffusion_form, params=params.kappa).to_dense()

    assert np.allclose(np.asarray(K_expr), np.asarray(K_ref))


def test_weakform_multi_term_operator_matches():
    """Operator version with Params matches manual combination."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    params = ff.Params(rho=3.0, kappa=2.0)
    form = ff.BilinearForm.volume(
        lambda u, v, p: (
            p.rho * h_wf.outer(v, u) + p.kappa * (v.grad @ u.grad)
        ) * h_wf.dOmega()
    )
    K_expr = space.assemble_bilinear_form(form.get_compiled(), params=params).to_dense()
    K_ref = params.rho * space.assemble_mass_matrix().to_dense()
    K_ref += space.assemble_bilinear_form(ff.diffusion_form, params=params.kappa).to_dense()

    assert np.allclose(np.asarray(K_expr), np.asarray(K_ref))


def test_weakform_vector_uv_raises():
    """u*v on vector fields should raise to avoid ambiguous meaning."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)

    form = ff.BilinearForm.volume(lambda u, v, _p: h_wf.outer(v, u) * h_wf.dOmega())
    with pytest.raises(ValueError, match="scalar fields"):
        space.assemble_bilinear_form(form.get_compiled(), params=0.0)


def test_linearform_volume_body_force_matches():
    """LinearForm.volume(action) matches scalar_body_force_form."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.LinearForm.volume(lambda v, p: (v * p) * h_wf.dOmega())
    F_expr = space.assemble_linear_form(form.get_compiled(), params=2.0)
    F_ref = space.assemble_linear_form(ff.scalar_body_force_form, params=2.0)

    assert np.allclose(np.asarray(F_expr), np.asarray(F_ref))


def test_linearform_volume_single_space_accepts_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.LinearForm.volume(lambda v, p: (v * p) * h_wf.dOmega())

    direct = space.assemble_linear_form(form, params=2.0)
    compiled = space.assemble_linear_form(form.get_compiled(), params=2.0)

    assert np.allclose(np.asarray(direct), np.asarray(compiled))


def test_linearform_named_spaces_matches_single_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    form = ff.LinearForm.volume(lambda v, p: (v * p) * h_wf.dOmega())
    F_named = ff.assemble_linear_form(
        ff.LinearSpaces(test=ff.NamedSpace("V", space)),
        form.get_compiled(),
        params=2.0,
    )
    F_ref = space.assemble_linear_form(ff.scalar_body_force_form, params=2.0)

    assert np.allclose(np.asarray(F_named), np.asarray(F_ref))


def test_bilinearform_named_spaces_accept_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    form = ff.BilinearForm.volume(lambda u, v, p: p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega())
    params = ff.Params(kappa=2.0)

    spaces = ff.BilinearSpaces(test=ff.NamedSpace("V", space), trial=ff.NamedSpace("U", space))
    direct = ff.assemble_bilinear_form(spaces, form, params).to_dense()
    compiled = ff.assemble_bilinear_form(spaces, form.get_compiled(), params).to_dense()

    assert np.allclose(np.asarray(direct), np.asarray(compiled))


def test_linearform_named_spaces_accept_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    form = ff.LinearForm.volume(lambda v, p: (v * p) * h_wf.dOmega())

    spaces = ff.LinearSpaces(test=ff.NamedSpace("V", space))
    direct = ff.assemble_linear_form(spaces, form, params=2.0)
    compiled = ff.assemble_linear_form(spaces, form.get_compiled(), params=2.0)

    assert np.allclose(np.asarray(direct), np.asarray(compiled))


def test_linearform_surface_matches_tensor():
    """LinearForm.surface matches tensor-based surface traction form."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    coords = np.asarray(mesh.coords)
    xmax = float(coords[:, 0].max())

    def on_xmax(face: np.ndarray) -> bool:
        return np.allclose(face[:, 0], xmax, atol=1e-8)

    facets = mesh.boundary_facets_where(on_xmax)
    surface = ff.SurfaceMesh.from_hex_mesh(mesh, facets)

    traction = np.array([1.0, 0.0, 0.0], dtype=float)

    def traction_form(ctx: ff.SurfaceFormContext, t: np.ndarray) -> np.ndarray:
        return h_ts.dot(ctx.v, t)

    surface_form = ff.LinearForm.surface(lambda v, p: h_wf.dot(v, p) * h_wf.ds())
    F_tensor = surface.assemble_linear_form_on_space(
        space, traction_form, params=traction
    )
    F_wf = surface.assemble_linear_form_on_space(
        space, surface_form.get_compiled(), params=traction
    )

    assert np.allclose(np.asarray(F_tensor), np.asarray(F_wf))


def test_linearform_surface_on_space_accepts_wrapper():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    coords = np.asarray(mesh.coords)
    xmax = float(coords[:, 0].max())

    def on_xmax(face: np.ndarray) -> bool:
        return np.allclose(face[:, 0], xmax, atol=1e-8)

    facets = mesh.boundary_facets_where(on_xmax)
    surface = ff.SurfaceMesh.from_hex_mesh(mesh, facets)
    traction = np.array([1.0, 0.0, 0.0], dtype=float)
    surface_form = ff.LinearForm.surface(lambda v, p: h_wf.dot(v, p) * h_wf.ds())

    direct = surface.assemble_linear_form_on_space(space, surface_form, params=traction)
    compiled = surface.assemble_linear_form_on_space(space, surface_form.get_compiled(), params=traction)

    assert np.allclose(np.asarray(direct), np.asarray(compiled))


def test_linearform_surface_domain_matches_surface_entrypoint():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    V = ff.NamedSpace("V", space)
    coords = np.asarray(mesh.coords)
    xmax = float(coords[:, 0].max())

    def on_xmax(face: np.ndarray) -> bool:
        return np.allclose(face[:, 0], xmax, atol=1e-8)

    facets = mesh.boundary_facets_where(on_xmax)
    surface = ff.SurfaceMesh.from_hex_mesh(mesh, facets)

    surface_form = ff.LinearForm.surface(
        lambda v, p: h_wf.dot(v, p.pressure * h_wf.normal()) * h_wf.ds()
    )
    params = ff.Params(pressure=1.0)

    F_surface = surface.assemble_linear_form_on_space(space, surface_form.get_compiled(), params=params)
    F_domain = ff.assemble_linear_form(
        ff.LinearSpaces(test=V),
        surface_form.get_compiled(),
        params,
        domain=surface,
    )

    assert np.allclose(np.asarray(F_surface), np.asarray(F_domain))

"""Mixed surface weak-form assembly on supermesh for two-body contact (penalty)."""
import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def _two_square_surfaces():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.5, 1.5, 0.0],
        ],
        dtype=float,
    )
    facets_a = np.array([[0, 1, 2, 3]], dtype=int)
    facets_b = np.array([[4, 5, 6, 7]], dtype=int)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets_a)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets_b)
    return coords, surf_a, surf_b


def _assemble_surface_jacobian(contact, res_form, u_a, u_b):
    return ff.assemble_mixed_surface_jacobian(
        contact.supermesh_coords,
        contact.supermesh_conn,
        contact.source_facets_master,
        contact.source_facets_slave,
        contact.surface_master,
        contact.surface_slave,
        res_form,
        u_a,
        u_b,
        params={},
        grad_source="surface",
        dof_source="surface",
        sparse=False,
    )


def test_mixed_surface_penalty_matches_mortar():
    coords, surf_a, surf_b = _two_square_surfaces()
    contact_ab = ff.ContactSurfaceSpace.from_surfaces(surf_a, surf_b, tol=1e-8)

    def res_a(v, u, _p):
        # Penalty coupling uses the opposing side's unknown.
        u_b = ff.unknown_ref("b")
        return (v * (u.val - u_b.val)) * h_wf.ds()

    def res_b(v, u, _p):
        u_a = ff.unknown_ref("a")
        return (v * (u.val - u_a.val)) * h_wf.ds()

    res_form = ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})
    rng = np.random.default_rng(0)
    u_a = jnp.asarray(rng.standard_normal(surf_a.n_nodes))
    u_b = jnp.asarray(rng.standard_normal(surf_b.n_nodes))

    J = _assemble_surface_jacobian(contact_ab, res_form, u_a, u_b)
    J = np.asarray(J)

    # Mortar matrices provide an independent assembly path for the same coupling.
    M_aa, M_ab = contact_ab.assemble_mortar_matrices()
    contact_ba = ff.ContactSurfaceSpace.from_surfaces(surf_b, surf_a, tol=1e-8)
    M_bb, M_ba = contact_ba.assemble_mortar_matrices()

    n_a = surf_a.n_nodes
    n_b = surf_b.n_nodes
    assert np.allclose(J[:n_a, :n_a], _coo_to_dense(M_aa, (n_a, n_a)), atol=1e-6)
    assert np.allclose(J[:n_a, n_a:], -_coo_to_dense(M_ab, (n_a, n_b)), atol=1e-6)
    assert np.allclose(J[n_a:, :n_a], -_coo_to_dense(M_ba, (n_b, n_a)), atol=1e-6)
    assert np.allclose(J[n_a:, n_a:], _coo_to_dense(M_bb, (n_b, n_b)), atol=1e-6)


def test_mixed_surface_supports_p0_multiplier():
    coords, surf_a, surf_b = _two_square_surfaces()
    sm = ff.build_surface_supermesh(surf_a, surf_b, tol=1e-8)

    def res_a(v, u, _p):
        u_b = ff.unknown_ref("b")
        return (v * (u.val - u_b.val)) * h_wf.ds()

    def res_b(v, u, _p):
        return (v * u.val) * h_wf.ds()

    res_form = ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})
    u_a = jnp.zeros((surf_a.n_nodes,))
    u_b = jnp.zeros((surf_b.n_facets,))  # P0: one dof per slave facet

    R = ff.assemble_mixed_surface_residual(
        sm.coords,
        sm.conn,
        sm.source_facets_a,
        sm.source_facets_b,
        surf_a,
        surf_b,
        res_form,
        u_a,
        u_b,
        params={},
        space_mode_a="nodal",
        space_mode_b="p0",
        dof_source="surface",
        grad_source="surface",
        quad_order=1,
        tol=1e-8,
    )
    J = ff.assemble_mixed_surface_jacobian(
        sm.coords,
        sm.conn,
        sm.source_facets_a,
        sm.source_facets_b,
        surf_a,
        surf_b,
        res_form,
        u_a,
        u_b,
        params={},
        space_mode_a="nodal",
        space_mode_b="p0",
        dof_source="surface",
        grad_source="surface",
        sparse=False,
        quad_order=1,
        tol=1e-8,
    )

    R = np.asarray(R)
    J = np.asarray(J)
    n_total = surf_a.n_nodes + surf_b.n_facets
    assert R.shape == (n_total,)
    assert J.shape == (n_total, n_total)
    assert np.count_nonzero(np.abs(J)) > 0


def test_mixed_surface_projection_supermesh_parity(monkeypatch):
    coords, surf_a, surf_b = _two_square_surfaces()
    sm = ff.build_surface_supermesh(surf_a, surf_b, tol=1e-8)

    def res_a(v, u, _p):
        u_b = ff.unknown_ref("b")
        return (v * (u.val - u_b.val)) * h_wf.ds()

    def res_b(v, u, _p):
        u_a = ff.unknown_ref("a")
        return (v * (u.val - u_a.val)) * h_wf.ds()

    res_form = ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})
    rng = np.random.default_rng(1)
    u_a = jnp.asarray(rng.standard_normal(surf_a.n_nodes))
    u_b = jnp.asarray(rng.standard_normal(surf_b.n_nodes))

    common = dict(
        supermesh_coords=sm.coords,
        supermesh_conn=sm.conn,
        source_facets_a=sm.source_facets_a,
        source_facets_b=sm.source_facets_b,
        surface_a=surf_a,
        surface_b=surf_b,
        res_form=res_form,
        u_a=u_a,
        u_b=u_b,
        params={},
        space_mode_a="nodal",
        space_mode_b="nodal",
        dof_source="surface",
        grad_source="surface",
        quad_order=1,
        tol=1e-8,
    )

    monkeypatch.setenv("FLUXFEM_MORTAR_MODE", "supermesh")
    r_super = np.asarray(ff.assemble_mixed_surface_residual(**common))
    j_super = np.asarray(ff.assemble_mixed_surface_jacobian(**common, sparse=False))

    monkeypatch.setenv("FLUXFEM_MORTAR_MODE", "projection")
    r_proj = np.asarray(ff.assemble_mixed_surface_residual(**common))
    j_proj = np.asarray(ff.assemble_mixed_surface_jacobian(**common, sparse=False))

    assert np.allclose(r_super, r_proj, atol=1e-6)
    assert np.allclose(j_super, j_proj, atol=1e-6)


def _coo_to_dense(mat, shape):
    out = np.zeros(shape, dtype=float)
    for r, c, v in zip(mat.rows, mat.cols, mat.data):
        out[int(r), int(c)] += float(v)
    return out

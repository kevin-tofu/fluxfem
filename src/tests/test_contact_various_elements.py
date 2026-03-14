import numpy as np
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import (
    compile_mixed_surface_residual,
    einsum as wf_einsum,
    param_ref,
    test_ref as wf_test_ref,
    unknown_ref,
)


def _build_meshes(elem_top: str, elem_bot: str):
    if elem_top == "tet":
        box_top = ff.StructuredTetTensorBox(
            nx=2, ny=2, nz=1, lx=2.0, ly=2.0, lz=1.0, origin=(0.0, 0.0, 0.0)
        )
        space_top = ff.make_tet_space(box_top.build(), dim=3)
    elif elem_top == "hex":
        box_top = ff.StructuredHexBox(
            nx=2, ny=2, nz=1, lx=2.0, ly=2.0, lz=1.0, origin=(0.0, 0.0, 0.0), order=1
        )
        space_top = ff.make_hex_space(box_top.build(), dim=3)
    else:
        raise ValueError(f"unsupported element: {elem_top}")

    if elem_bot == "tet":
        box_bot = ff.StructuredTetTensorBox(
            nx=2, ny=2, nz=1, lx=1.0, ly=1.0, lz=0.5, origin=(0.5, 0.5, -0.5)
        )
        space_bot = ff.make_tet_space(box_bot.build(), dim=3)
    elif elem_bot == "hex":
        box_bot = ff.StructuredHexBox(
            nx=2, ny=2, nz=1, lx=1.0, ly=1.0, lz=0.5, origin=(0.5, 0.5, -0.5), order=1
        )
        space_bot = ff.make_hex_space(box_bot.build(), dim=3)
    else:
        raise ValueError(f"unsupported element: {elem_bot}")

    mesh_top = space_top.mesh
    mesh_bot = space_bot.mesh
    return box_top, box_bot, mesh_top, mesh_bot, space_top, space_bot


def _contact_facets(box_top, box_bot, mesh_top, mesh_bot):
    contact_facets_bot = mesh_bot.facets_on_plane(axis=2, value=0.0)
    x0, y0, _ = box_bot.origin
    x1 = x0 + box_bot.lx
    y1 = y0 + box_bot.ly
    dx_top = box_top.lx / box_top.nx
    dy_top = box_top.ly / box_top.ny
    pad = 2.0 * min(dx_top, dy_top)
    contact_facets_top = mesh_top.facets_on_plane_box(
        axis=2,
        value=0.0,
        x=(x0 - pad, x1 + pad),
        y=(y0 - pad, y1 + pad),
        mode="centroid",
    )
    return contact_facets_top, contact_facets_bot


def _contact_params(box_top, box_bot, E=210e9, nu=0.3):
    lam, mu = ff.lame_parameters(E, nu)
    dx_top = box_top.lx / box_top.nx
    dy_top = box_top.ly / box_top.ny
    dz_top = box_top.lz / box_top.nz
    dx_bot = box_bot.lx / box_bot.nx
    dy_bot = box_bot.ly / box_bot.ny
    dz_bot = box_bot.lz / box_bot.nz
    h = min(dx_top, dy_top, dz_top, dx_bot, dy_bot, dz_bot)
    alpha = 20.0 * (10000.0 * mu + lam)
    return ff.Params(alpha=float(alpha), inv_h=float(1.0 / h), lam=float(lam), mu=float(mu))


def _nitsche_bilinear(v1, v2, u1, u2, p):
    n = h_wf.normal()
    ju = u1.val - u2.val
    t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
    t_v1 = h_wf.traction(v1, n, p)
    t_v2 = h_wf.traction(v2, n, p)
    penalty = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
    traction = -h_wf.dot(v1, t_u) + h_wf.dot(v2, t_u)
    traction -= 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
    traction -= 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
    return (penalty + traction) * h_wf.ds()


def _nitsche_residuals_numpy():
    v1 = wf_test_ref("a")
    v2 = wf_test_ref("b")
    u1 = unknown_ref("a")
    u2 = unknown_ref("b")
    p = param_ref()
    n = h_wf.normal()
    ju = u1.val - u2.val
    t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
    t_v1 = h_wf.traction(v1, n, p)
    t_v2 = h_wf.traction(v2, n, p)
    penalty_a = (p.alpha * p.inv_h) * h_wf.dot(v1, ju)
    penalty_b = -(p.alpha * p.inv_h) * h_wf.dot(v2, ju)
    traction_a = -h_wf.dot(v1, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
    traction_b = h_wf.dot(v2, t_u) - 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
    expr_a = (penalty_a + traction_a) * h_wf.ds()
    expr_b = (penalty_b + traction_b) * h_wf.ds()
    return compile_mixed_surface_residual({"a": expr_a, "b": expr_b})


def _zeros_u(space):
    return np.zeros(space.n_dofs, dtype=float)


def _u_hat_fn(x_q: np.ndarray) -> np.ndarray:
    return np.zeros((x_q.shape[0], 3), dtype=float)


def _assert_finite(arr, name: str):
    assert np.isfinite(np.asarray(arr)).all(), f"{name} contains non-finite values"


def test_contact_two_sided_tet_hex_numpy_not_implemented():
    for elem in ("tet", "hex"):
        box_top, box_bot, mesh_top, mesh_bot, space_top, space_bot = _build_meshes(elem, elem)
        contact_facets_top, contact_facets_bot = _contact_facets(
            box_top, box_bot, mesh_top, mesh_bot
        )
        side_top = ff.ContactSide.from_facets(mesh_top, contact_facets_top, space_top)
        side_bot = ff.ContactSide.from_facets(mesh_bot, contact_facets_bot, space_bot)
        contact = ff.ContactSurfaceSpace.from_sides(
            side_top,
            side_bot,
            quad_order=1,
            backend="numpy",
            batch_jac=False,
        )
        params = _contact_params(box_top, box_bot)
        with pytest.raises(NotImplementedError, match="backend='numpy'"):
            contact.assemble_bilinear(
                _nitsche_bilinear, (_zeros_u(space_top), _zeros_u(space_bot)), params, sparse=False
            )


def test_contact_onesided_tet_hex_numpy():
    for elem in ("tet", "hex"):
        box_top, box_bot, mesh_top, mesh_bot, space_top, _space_bot = _build_meshes(elem, elem)
        contact_facets_top, _contact_facets_bot = _contact_facets(
            box_top, box_bot, mesh_top, mesh_bot
        )
        side_top = ff.ContactSide.from_facets(mesh_top, contact_facets_top, space_top)
        contact_space = ff.OneSidedContactSurfaceSpace.from_side(
            side_top,
            quad_order=1,
        )
        params = _contact_params(box_top, box_bot)
        K, f = contact_space.assemble_bilinear(_u_hat_fn, params)
        K = np.asarray(K)
        f = np.asarray(f)
        assert K.shape[0] == K.shape[1]
        assert f.shape[0] == K.shape[0]
        _assert_finite(K, f"onesided_{elem}_K")
        _assert_finite(f, f"onesided_{elem}_f")


def test_contact_two_sided_hex_tet_numpy():
    box_top, box_bot, mesh_top, mesh_bot, space_top, space_bot = _build_meshes("hex", "tet")
    contact_facets_top, contact_facets_bot = _contact_facets(
        box_top, box_bot, mesh_top, mesh_bot
    )
    side_top = ff.ContactSide.from_facets(mesh_top, contact_facets_top, space_top)
    side_bot = ff.ContactSide.from_facets(mesh_bot, contact_facets_bot, space_bot)
    contact = ff.ContactSurfaceSpace.from_sides(
        side_top,
        side_bot,
        quad_order=1,
        backend="jax",
        batch_jac=False,
    )
    params = _contact_params(box_top, box_bot)
    res_form = _nitsche_residuals_numpy()
    K = contact.assemble_jacobian(
        res_form,
        (_zeros_u(space_top), _zeros_u(space_bot)),
        params,
        sparse=False,
        backend="jax",
        batch_jac=False,
    )
    K = np.asarray(K)
    assert K.shape[0] == K.shape[1]
    _assert_finite(K, "two_sided_hex_tet_K")


def test_contact_two_sided_hex_batch_jac_matches_nonbatch():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, origin=(0.0, 0.0, 0.0), order=1).build()
    space = ff.make_hex_space(mesh, dim=3)
    conn = np.asarray(mesh.conn, dtype=int)
    facets = np.array([[int(conn[0, i]) for i in (0, 1, 2, 3)]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        np.asarray(mesh.coords, dtype=float),
        facets,
        np.asarray(mesh.coords, dtype=float),
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=1,
        backend="jax",
        batch_jac=True,
    )
    params = ff.Params(alpha=10.0, inv_h=1.0, lam=0.0, mu=0.0, use_penalty=1.0, use_traction=0.0)

    def res_a(v, u, p):
        u_b = ff.unknown_ref("b", space="B")
        return ((p.alpha * p.inv_h) * h_wf.dot(v, u.val - u_b.val)) * h_wf.ds()

    def res_b(v, u, p):
        u_a = ff.unknown_ref("a", space="A")
        return (-(p.alpha * p.inv_h) * h_wf.dot(v, u_a.val - u.val)) * h_wf.ds()

    res_form = compile_mixed_surface_residual(
        {
            "a": ff.bind_mixed_residual("a", res_a, space="A"),
            "b": ff.bind_mixed_residual("b", res_b, space="B"),
        }
    )
    u = np.zeros(space.n_dofs, dtype=float)
    K_batch = np.asarray(
        contact.assemble_jacobian(
            res_form,
            (u, u),
            params,
            sparse=False,
            backend="jax",
            batch_jac=True,
        )
    )
    K_ref = np.asarray(
        contact.assemble_jacobian(
            res_form,
            (u, u),
            params,
            sparse=False,
            backend="jax",
            batch_jac=False,
        )
    )
    assert np.allclose(K_batch, K_ref, atol=1e-10)


def test_contact_two_sided_hex_batch_jac_sparse_matches_dense():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, origin=(0.0, 0.0, 0.0), order=1).build()
    space = ff.make_hex_space(mesh, dim=3)
    conn = np.asarray(mesh.conn, dtype=int)
    facets = np.array([[int(conn[0, i]) for i in (0, 1, 2, 3)]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        np.asarray(mesh.coords, dtype=float),
        facets,
        np.asarray(mesh.coords, dtype=float),
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=4,
        backend="jax",
        batch_jac=True,
    )
    params = ff.Params(alpha=10.0, inv_h=1.0, lam=0.0, mu=0.0, use_penalty=1.0, use_traction=0.0)

    def res_a(v, u, p):
        u_b = ff.unknown_ref("b", space="B")
        return ((p.alpha * p.inv_h) * h_wf.dot(v, u.val - u_b.val)) * h_wf.ds()

    def res_b(v, u, p):
        u_a = ff.unknown_ref("a", space="A")
        return (-(p.alpha * p.inv_h) * h_wf.dot(v, u_a.val - u.val)) * h_wf.ds()

    res_form = compile_mixed_surface_residual(
        {
            "a": ff.bind_mixed_residual("a", res_a, space="A"),
            "b": ff.bind_mixed_residual("b", res_b, space="B"),
        }
    )
    u = np.zeros(space.n_dofs, dtype=float)
    K_dense = np.asarray(
        contact.assemble_jacobian(
            res_form,
            (u, u),
            params,
            sparse=False,
            backend="jax",
            batch_jac=True,
        )
    )
    K_sparse = contact.assemble_jacobian(
        res_form,
        (u, u),
        params,
        sparse=True,
        backend="jax",
        batch_jac=True,
    )
    K_sparse_dense = np.asarray(K_sparse.to_dense()) if hasattr(K_sparse, "to_dense") else np.asarray(K_sparse)
    assert np.allclose(K_sparse_dense, K_dense, atol=1e-10)

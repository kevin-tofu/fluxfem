import numpy as np

import jax
import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import einsum as wf_einsum


def _mesh_spacing(box):
    return box.lx / box.nx, box.ly / box.ny, box.lz / box.nz


def _build_contact_spaces():
    box_top = ff.StructuredTetTensorBox(
        nx=1, ny=1, nz=1, lx=2.0, ly=2.0, lz=1.0, origin=(0.0, 0.0, 0.0)
    )
    box_bot = ff.StructuredTetTensorBox(
        nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=0.5, origin=(0.5, 0.5, -0.5)
    )
    mesh_top = box_top.build()
    mesh_bot = box_bot.build()
    space_top = ff.make_tet_space(mesh_top, dim=3)
    space_bot = ff.make_tet_space(mesh_bot, dim=3)

    contact_facets_bot = mesh_bot.facets_on_plane(axis=2, value=0.0)
    x0, y0, _ = box_bot.origin
    x1 = x0 + box_bot.lx
    y1 = y0 + box_bot.ly
    dx_top, dy_top, _ = _mesh_spacing(box_top)
    pad = 2.0 * min(dx_top, dy_top)
    contact_facets_top = mesh_top.facets_on_plane_box(
        axis=2,
        value=0.0,
        x=(x0 - pad, x1 + pad),
        y=(y0 - pad, y1 + pad),
        mode="centroid",
    )

    side_top = ff.ContactSide.from_facets(mesh_top, contact_facets_top, space_top)
    side_bot = ff.ContactSide.from_facets(mesh_bot, contact_facets_bot, space_bot)
    return side_top, side_bot, box_top, box_bot


def test_contact_numpy_block_fd_matches_jax():
    jax.config.update("jax_enable_x64", True)
    side_top, side_bot, box_top, box_bot = _build_contact_spaces()

    contact_np = ff.ContactSurfaceSpace.from_sides(
        side_top,
        side_bot,
        quad_order=1,
        backend="numpy",
        fd_mode="forward",
        fd_eps=1e-6,
        fd_block_size=4,
        batch_jac=False,
    )
    contact_jax = ff.ContactSurfaceSpace.from_sides(
        side_top,
        side_bot,
        quad_order=1,
        backend="jax",
        batch_jac=False,
    )

    E, nu = 210e9, 0.3
    lam, mu = ff.lame_parameters(E, nu)
    dx_top, dy_top, dz_top = _mesh_spacing(box_top)
    dx_bot, dy_bot, dz_bot = _mesh_spacing(box_bot)
    h = min(dx_top, dy_top, dz_top, dx_bot, dy_bot, dz_bot)
    alpha = 20.0 * (10000.0 * mu + lam)
    params = ff.Params(alpha=float(alpha), inv_h=float(1.0 / h), lam=float(lam), mu=float(mu))

    def bilin(v1, v2, u1, u2, p):
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

    u_top0 = np.zeros(side_top.space.n_dofs)
    u_bot0 = np.zeros(side_bot.space.n_dofs)

    K_np = contact_np.assemble_bilinear(bilin, (u_top0, u_bot0), params, sparse=False)
    K_jax = contact_jax.assemble_bilinear(bilin, (u_top0, u_bot0), params, sparse=False)

    K_np = np.asarray(K_np)
    K_jax = np.asarray(K_jax)
    diff = K_np - K_jax
    max_err = float(np.max(np.abs(diff))) if diff.size else 0.0
    if diff.size:
        max_idx = int(np.argmax(np.abs(diff)))
        i_max, j_max = np.unravel_index(max_idx, diff.shape)
    else:
        i_max, j_max = 0, 0
    max_ref = float(np.max(np.abs(K_jax))) if K_jax.size else 0.0
    ref = float(np.quantile(np.abs(K_jax), 0.95)) if K_jax.size else 0.0
    ref = max(ref, 1.0)
    rel_err = max_err / max(1.0, max_ref)
    assert np.isfinite(max_err)
    assert rel_err < 1e-6
    abs_tol = 1e-6 * ref + 1e-6
    if max_err >= abs_tol:
        jax_val = float(K_jax[i_max, j_max]) if K_jax.size else 0.0
        np_val = float(K_np[i_max, j_max]) if K_np.size else 0.0
        abs_err = float(np.abs(np_val - jax_val))
        denom = max(1e-12, abs(jax_val))
        rel_loc = abs_err / denom
        msg = (
            "contact numpy block fd mismatch: "
            f"idx=({i_max},{j_max}) "
            f"np={np_val:.6e} jax={jax_val:.6e} "
            f"abs_err={abs_err:.6e} rel_err={rel_loc:.6e} "
            f"ref={ref:.6e} abs_tol={abs_tol:.6e} "
            f"max_abs_np={np.max(np.abs(K_np)):.6e} "
            f"max_abs_jax={np.max(np.abs(K_jax)):.6e}"
        )
        raise AssertionError(msg)

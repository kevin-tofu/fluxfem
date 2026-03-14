"""Integration-style contact solve on asymmetric box resolutions."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def _translate_mesh(mesh, shift_xyz):
    shift = np.asarray(shift_xyz, dtype=float)
    coords = np.asarray(mesh.coords, dtype=float) + shift[None, :]
    return mesh.__class__(
        coords=jnp.asarray(coords),
        conn=mesh.conn,
        cell_tags=mesh.cell_tags,
        node_tags=mesh.node_tags,
    )


def _penalty_residual_form():
    def res_a(v, u, p):
        u_b = ff.unknown_ref("b")
        ju = u.val - u_b.val
        return (p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    def res_b(v, u, p):
        u_a = ff.unknown_ref("a")
        ju = u_a.val - u.val
        return -(p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    return ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})


def test_contact_solve_box_12x12x12_vs_2x2x2():
    # Master (upper block): finer mesh
    top_raw = ff.StructuredHexBox(nx=12, ny=12, nz=12, lx=2.0, ly=2.0, lz=2.0).build()
    top_mesh = _translate_mesh(top_raw, (-1.0, -1.0, 0.61))
    top_space = ff.make_hex_space(top_mesh, dim=1, intorder=2)

    # Slave (support): coarser mesh
    support_raw = ff.StructuredHexBox(nx=2, ny=2, nz=2, lx=0.6, ly=0.6, lz=0.6).build()
    support_mesh = _translate_mesh(support_raw, (-0.3, -0.3, 0.0))
    support_space = ff.make_hex_space(support_mesh, dim=1, intorder=2)

    bilinear = ff.BilinearForm.volume(
        lambda u, v, k: k * (v.grad @ u.grad) * h_wf.dOmega()
    )
    kappa = 1.0
    K_top = top_space.assemble_bilinear_form(bilinear.get_compiled(), params=kappa)
    K_support = support_space.assemble_bilinear_form(bilinear.get_compiled(), params=kappa)
    F_top = np.asarray(top_space.assemble_linear_form(ff.scalar_body_force_form, params=-1.0))
    F_support = np.asarray(support_space.assemble_linear_form(ff.scalar_body_force_form, params=0.0))

    z_top_min = float(np.asarray(top_mesh.coords)[:, 2].min())
    z_support_max = float(np.asarray(support_mesh.coords)[:, 2].max())
    master_facets = np.asarray(top_mesh.facets_on_plane(axis=2, value=z_top_min, tol=1e-8), dtype=int)
    slave_facets = np.asarray(support_mesh.facets_on_plane(axis=2, value=z_support_max, tol=1e-8), dtype=int)
    assert master_facets.shape[0] > 0
    assert slave_facets.shape[0] > 0

    contact = ff.ContactSurfaceSpace.from_facets(
        np.asarray(top_mesh.coords),
        master_facets,
        np.asarray(support_mesh.coords),
        slave_facets,
        elem_conn_master=np.asarray(top_mesh.conn, dtype=int),
        elem_conn_slave=np.asarray(support_mesh.conn, dtype=int),
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
        backend="numpy",
    )
    ops = ff.assemble_contact_penalty_operators(
        contact,
        weak_form=_penalty_residual_form(),
        state={"a": jnp.zeros(int(top_space.n_dofs)), "b": jnp.zeros(int(support_space.n_dofs))},
        params=ff.Params(alpha=40.0, inv_h=1.0),
        backend="jax",
    )

    K_u = sp.block_diag((K_top.to_csr(), K_support.to_csr()), format="csr")
    F_u = np.concatenate([F_top, F_support], axis=0)
    builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
    builder.register_space("top", top_space, value_dim=1)
    builder.register_space("support", support_space, value_dim=1)
    builder.add_contact(ops, master="top", slave="support", value_dim=1)
    system = builder.build()

    z_support_min = float(np.asarray(support_mesh.coords)[:, 2].min())
    support_bottom_dofs = ff.DirichletBC.from_boundary_dofs(
        support_mesh,
        lambda pts, z=z_support_min: np.isclose(pts[:, 2], z, atol=1e-8),
        components="x",
    ).dofs
    dir_dofs = int(top_space.n_dofs) + np.asarray(support_bottom_dofs, dtype=int)
    sol = np.asarray(
        system.solve(
            dirichlet_dofs=dir_dofs,
            dirichlet_vals=0.0,
            format="csr",
            diagonal_shift=1.0e-8,
        )
    )
    K_csr, F = system.assemble(format="csr")
    K_coo = K_csr.tocoo()
    K_flux = ff.FluxSparseMatrix(K_coo.row, K_coo.col, K_coo.data, K_csr.shape[0])
    K_bc, F_bc = ff.enforce_dirichlet_sparse(K_flux, F, dir_dofs, 0.0)
    K_bc = K_bc + 1.0e-8 * sp.eye(K_bc.shape[0], format="csr")
    residual = np.asarray(K_bc @ sol - F_bc, dtype=float)
    free = np.setdiff1d(np.arange(sol.shape[0], dtype=int), np.asarray(dir_dofs, dtype=int))

    u_top = sol[: int(top_space.n_dofs)]
    assert np.all(np.isfinite(sol))
    assert np.linalg.norm(u_top) > 0.0
    assert float(np.mean(u_top)) < 0.0
    assert float(np.linalg.norm(residual[free])) < 1.0e-7

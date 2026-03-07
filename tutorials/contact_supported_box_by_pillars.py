#!/usr/bin/env python3
"""
Large box supported by multiple small boxes (pillars) via penalty contact.

- Top (large) box receives gravity load.
- Several small support boxes are merged as one disconnected support mesh.
- Bottom faces of support boxes are fixed with Dirichlet BC.
- Interface coupling is assembled with ContactSurfaceSpace + penalty residual/Jacobian.
- CoupledSystem lifts interface operators into structural DOFs and solves.
"""

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


def _merge_disconnected_hex_meshes(meshes):
    coords_all = []
    conn_all = []
    offset = 0
    for m in meshes:
        c = np.asarray(m.coords, dtype=float)
        e = np.asarray(m.conn, dtype=int)
        coords_all.append(c)
        conn_all.append(e + offset)
        offset += c.shape[0]
    coords = jnp.asarray(np.vstack(coords_all))
    conn = jnp.asarray(np.vstack(conn_all))
    return ff.HexMesh(coords=coords, conn=conn)


def _facet_dofs(facets: np.ndarray, *, dim: int) -> np.ndarray:
    facets = np.asarray(facets, dtype=int)
    out = []
    for f in facets:
        dofs = []
        for node in f:
            for d in range(dim):
                dofs.append(dim * int(node) + d)
        out.append(dofs)
    return np.asarray(out, dtype=int)


def _nitsche_residual_form():
    def res_a(v, u, p):
        u2 = ff.unknown_ref("b")
        ju = u.val - u2.val
        return (p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    def res_b(v, u, p):
        u1 = ff.unknown_ref("a")
        ju = u1.val - u.val
        return -(p.alpha * p.inv_h) * h_wf.dot(v, ju) * h_wf.ds()

    return ff.compile_mixed_surface_residual({"a": res_a, "b": res_b})


def main():
    # Material
    E, nu = 210_000.0, 0.3
    D = ff.isotropic_3d_D(E, nu)

    # Gravity on top box only
    density = 7.8e-3
    g = 9.81
    body_force_top = np.array([0.0, 0.0, -density * g], dtype=float)

    # Top box
    top_raw = ff.StructuredHexBox(nx=6, ny=6, nz=2, lx=2.0, ly=2.0, lz=1.0).build()
    top_mesh = _translate_mesh(top_raw, (-1.0, -1.0, 1.05))
    top_space = ff.make_hex_space(top_mesh, dim=1, intorder=2)

    # Four support boxes ("pillars"), merged as one disconnected mesh
    support_raw = ff.StructuredHexBox(nx=2, ny=2, nz=2, lx=0.6, ly=0.6, lz=1.0).build()
    support_shifts = [
        (-0.8, -0.8, 0.0),
        (0.2, -0.8, 0.0),
        (-0.8, 0.2, 0.0),
        (0.2, 0.2, 0.0),
    ]
    support_meshes = [_translate_mesh(support_raw, s) for s in support_shifts]
    support_mesh = _merge_disconnected_hex_meshes(support_meshes)
    support_space = ff.make_hex_space(support_mesh, dim=1, intorder=2)

    # Volume stiffness (kept separate in this assembly-focused tutorial)
    bilinear = ff.BilinearForm.volume(
        lambda u, v, k: k * (v.grad @ u.grad)
        * h_wf.dOmega()
    )
    # Use scalar stiffness for this coupled tutorial path (value_dim=1).
    kappa = float(D[0, 0])
    K_top = top_space.assemble_bilinear_form(bilinear.get_compiled(), params=kappa)
    K_support = support_space.assemble_bilinear_form(bilinear.get_compiled(), params=kappa)

    # Gravity force (top only)
    F_top = np.asarray(top_space.assemble_linear_form(ff.scalar_body_force_form, params=body_force_top[2]))
    F_support = np.asarray(support_space.assemble_linear_form(ff.scalar_body_force_form, params=0.0))

    # Contact facets
    z_top_min = float(np.asarray(top_mesh.coords)[:, 2].min())
    z_support_max = float(np.asarray(support_mesh.coords)[:, 2].max())
    master_facets = np.asarray(top_mesh.facets_on_plane(axis=2, value=z_top_min, tol=1e-8), dtype=int)
    slave_facets = np.asarray(support_mesh.facets_on_plane(axis=2, value=z_support_max, tol=1e-8), dtype=int)

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

    # Penalty-family contact operators
    res_form = _nitsche_residual_form()
    u_if = {
        "a": jnp.zeros(int(top_space.n_dofs)),
        "b": jnp.zeros(int(support_space.n_dofs)),
    }
    params_if = ff.Params(alpha=50.0, inv_h=1.0)
    ops = ff.assemble_contact_penalty_operators(
        contact,
        weak_form=res_form,
        state=u_if,
        params=params_if,
        normal_source="master",
        backend="numpy",
    )

    # Build structural block and coupled solve
    K_u = sp.block_diag((K_top.to_csr(), K_support.to_csr()), format="csr")
    F_u = np.concatenate([F_top, F_support], axis=0)
    builder = ff.CoupledSystemBuilder.from_structural(K_u, F_u)
    builder.register_space("top", top_space, value_dim=1)
    builder.register_space("support", support_space, value_dim=1)
    builder.add_contact(
        ops,
        master="top",
        slave="support",
        value_dim=1,
    )
    system = builder.build()

    # Dirichlet: fix support bottom
    z_support_min = float(np.asarray(support_mesh.coords)[:, 2].min())
    support_bottom_dofs = ff.DirichletBC.from_boundary_dofs(
        support_mesh,
        lambda pts, z=z_support_min: np.isclose(pts[:, 2], z, atol=1e-8),
        components="x",
    ).dofs
    dir_dofs = int(top_space.n_dofs) + np.asarray(support_bottom_dofs, dtype=int)
    diagonal_shift = 1.0e-8
    sol = system.solve(
        dirichlet_dofs=dir_dofs,
        dirichlet_vals=0.0,
        format="csr",
        diagonal_shift=diagonal_shift,
    )

    u = np.asarray(sol[: int(top_space.n_dofs + support_space.n_dofs)])
    u_top = u[: int(top_space.n_dofs)]
    print("solved: large box + multiple support boxes")
    print(f"top dofs: {top_space.n_dofs}, support dofs: {support_space.n_dofs}")
    print(f"K_top nnz: {K_top.nnz}, K_support nnz: {K_support.nnz}")
    print(f"contact method: nitsche, interface dofs: {int(np.asarray(ops.residual).shape[0])}")
    print(f"support bottom Dirichlet dofs: {np.asarray(support_bottom_dofs).shape[0]}")
    print(f"diagonal shift: {diagonal_shift:.1e}")
    print(f"top displacement range: [{float(np.min(u_top)):.6e}, {float(np.max(u_top)):.6e}]")


if __name__ == "__main__":
    main()

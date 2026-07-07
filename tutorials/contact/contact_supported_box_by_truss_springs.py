#!/usr/bin/env python3
"""
Large 3D box supported by contact pads backed by truss-equivalent springs.

This keeps the contact side on the stable surface-penalty path while replacing
fully meshed support pillars with bottom-pad z springs whose stiffness comes
from a truss/bar section:

    k_column = E A / L

The support pads still provide contact facets; their bottom z DOFs are grounded
by the equivalent truss-column stiffness while bottom x/y are fixed.

Run from the repository root:

    PYTHONPATH=src python tutorials/contact/contact_supported_box_by_truss_springs.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.mesh.contact import compile_tagged_pair_nitsche_penalty_residual


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
    for mesh in meshes:
        coords = np.asarray(mesh.coords, dtype=float)
        conn = np.asarray(mesh.conn, dtype=int)
        coords_all.append(coords)
        conn_all.append(conn + offset)
        offset += coords.shape[0]
    return ff.HexMesh(coords=jnp.asarray(np.vstack(coords_all)), conn=jnp.asarray(np.vstack(conn_all)))


def _penalty_residual_form():
    def res_a(v, u, p):
        u_b = ff.unknown_ref("b", space="B")
        jump = u.val - u_b.val
        return (p.alpha * p.inv_h) * h_wf.dot(v, jump) * h_wf.ds()

    def res_b(v, u, p):
        u_a = ff.unknown_ref("a", space="A")
        jump = u_a.val - u.val
        return -(p.alpha * p.inv_h) * h_wf.dot(v, jump) * h_wf.ds()

    return compile_tagged_pair_nitsche_penalty_residual(
        {
            "a": ff.bind_mixed_residual("a", res_a, space="A"),
            "b": ff.bind_mixed_residual("b", res_b, space="B"),
        },
        backend="jax",
    )


def build_contact_supported_box_by_truss_springs(
    *,
    truss_E: float = 210_000.0,
    truss_A: float = 0.08,
    truss_length: float = 1.0,
    gravity: float = -0.08,
):
    top_raw = ff.StructuredHexBox(nx=6, ny=6, nz=2, lx=2.0, ly=2.0, lz=1.0).build()
    top_mesh = _translate_mesh(top_raw, (-1.0, -1.0, 1.05))
    top_space = ff.make_hex_space(top_mesh, dim=3, intorder=2)

    pad_raw = ff.StructuredHexBox(nx=2, ny=2, nz=1, lx=0.6, ly=0.6, lz=0.25).build()
    pad_shifts = [(-0.3, -0.3, 0.80)]
    support_mesh = _merge_disconnected_hex_meshes([_translate_mesh(pad_raw, shift) for shift in pad_shifts])
    support_space = ff.make_hex_space(support_mesh, dim=3, intorder=2)

    material = ff.isotropic_3d_D(210_000.0, 0.30)
    K_top = top_space.assemble(ff.linear_elasticity_form, params=material).to_csr()
    K_support = support_space.assemble(ff.linear_elasticity_form, params=material).to_csr()
    F_top = np.asarray(top_space.assemble(ff.vector_body_force_form, params=np.array([0.0, 0.0, float(gravity)])))
    F_support = np.zeros((support_space.n_dofs,), dtype=float)

    support_coords = np.asarray(support_mesh.coords, dtype=float)
    z_support_min = float(np.min(support_coords[:, 2]))
    bottom_z_dofs = ff.DirichletBC.from_boundary_dofs(
        support_mesh,
        lambda pts, z=z_support_min: np.isclose(pts[:, 2], z, atol=1.0e-8),
        components="z",
    ).dofs
    bottom_xy_dofs = ff.DirichletBC.from_boundary_dofs(
        support_mesh,
        lambda pts, z=z_support_min: np.isclose(pts[:, 2], z, atol=1.0e-8),
        components="xy",
    ).dofs

    truss_section = ff.TrussSection(E=truss_E, A=truss_A)
    truss_column_stiffness = truss_section.E * truss_section.A / float(truss_length)
    spring_per_bottom_dof = truss_column_stiffness / int(np.asarray(bottom_z_dofs).size)
    K_support = K_support + ff.assemble_dof_spring(
        int(support_space.n_dofs),
        bottom_z_dofs,
        spring_per_bottom_dof,
        format="csr",
    )

    z_top_min = float(np.asarray(top_mesh.coords)[:, 2].min())
    z_support_max = float(np.max(support_coords[:, 2]))
    master_facets = np.asarray(top_mesh.facets_on_plane(axis=2, value=z_top_min, tol=1.0e-8), dtype=int)
    slave_facets = np.asarray(support_mesh.facets_on_plane(axis=2, value=z_support_max, tol=1.0e-8), dtype=int)
    master_side = ff.ContactSideSpec.from_facets(top_mesh, master_facets, top_space)
    slave_side = ff.ContactSideSpec.from_facets(support_mesh, slave_facets, support_space)
    contact = ff.ContactPairSpec(master=master_side, slave=slave_side).prepare(
        quad_order=1,
        backend="jax",
    )
    ops = ff.assemble_contact_operators(
        contact,
        enforcement="penalty",
        weak_form=_penalty_residual_form(),
        state={"a": jnp.zeros(int(top_space.n_dofs)), "b": jnp.zeros(int(support_space.n_dofs))},
        params=ff.Params(alpha=50.0, inv_h=1.0),
        normal_source="master",
        backend="jax",
    )

    K_u = sp.block_diag((K_top, K_support), format="csr")
    F_u = np.concatenate([F_top, F_support], axis=0)
    builder = ff.NumpyCoupledSystemBuilder.from_structural(K_u, F_u)
    builder.register_space("top", top_space, value_dim=3)
    builder.register_space("support", support_space, value_dim=3)
    builder.add_contact(ops, master="top", slave="support", value_dim=3)

    return {
        "system": builder.build(),
        "top_space": top_space,
        "support_space": support_space,
        "top_mesh": top_mesh,
        "support_mesh": support_mesh,
        "fixed_dofs": int(top_space.n_dofs) + np.asarray(bottom_xy_dofs, dtype=int),
        "bottom_z_dofs": np.asarray(bottom_z_dofs, dtype=int),
        "truss_column_stiffness": float(truss_column_stiffness),
        "spring_per_bottom_dof": float(spring_per_bottom_dof),
    }


def main():
    model = build_contact_supported_box_by_truss_springs()
    system = model["system"]
    diagonal_shift = 1.0e-8
    fixed = model["fixed_dofs"]
    sol = np.asarray(
        system.solve(
            format="csr",
            dirichlet_dofs=fixed,
            dirichlet_vals=np.zeros((fixed.size,), dtype=float),
            diagonal_shift=diagonal_shift,
        ),
        dtype=float,
    )

    n_top = int(model["top_space"].n_dofs)
    u_top = sol[:n_top].reshape(-1, 3)
    u_support = sol[n_top : n_top + int(model["support_space"].n_dofs)]
    bottom_uz = u_support[model["bottom_z_dofs"]]

    print("solved: contact box + truss-equivalent support springs")
    print(f"top dofs: {model['top_space'].n_dofs}, support dofs: {model['support_space'].n_dofs}")
    print(f"truss column stiffness: {model['truss_column_stiffness']:.6e}")
    print(f"spring per bottom support dof: {model['spring_per_bottom_dof']:.6e}")
    print(f"diagonal shift: {diagonal_shift:.1e}")
    print(f"top uz range: [{float(np.min(u_top[:, 2])):.6e}, {float(np.max(u_top[:, 2])):.6e}]")
    print(f"support bottom mean uz: {float(np.mean(bottom_uz)):.6e}")


if __name__ == "__main__":
    main()

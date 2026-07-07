#!/usr/bin/env python3
"""
3D solid block coupled to a 3D truss/bar through an RBE3 face average.

The solid right face is copied into an auxiliary field and reduced to a 6-DOF
remote point by an RBE3 constraint. The remote translational DOFs are tied to
the truss root translations with ``add_dof_tie_constraint``.

Run from the repository root:

    PYTHONPATH=src python tutorials/elasticity/solid_truss_rbe3_coupling.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import scipy.sparse as sp
import jax

jax.config.update("jax_enable_x64", True)

import fluxfem as ff


def _as_csr(matrix):
    if sp.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "to_csr"):
        return matrix.to_csr()
    if hasattr(matrix, "toarray"):
        return sp.csr_matrix(matrix.toarray())
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def build_solid_truss_coupling(
    *,
    solid_nx: int = 3,
    solid_ny: int = 2,
    solid_nz: int = 2,
    solid_length: float = 1.0,
    solid_width: float = 0.3,
    solid_height: float = 0.3,
    truss_elems: int = 4,
    truss_length: float = 1.2,
    tip_load_x: float = 25.0,
):
    solid_mesh = ff.StructuredHexBox(
        nx=solid_nx,
        ny=solid_ny,
        nz=solid_nz,
        lx=solid_length,
        ly=solid_width,
        lz=solid_height,
    ).build()
    solid_space = ff.make_hex_space(solid_mesh, dim=3, intorder=2)
    solid_K = _as_csr(solid_space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(2.0e5, 0.30)))
    solid_F = np.zeros((solid_space.n_dofs,), dtype=float)

    solid_coords = np.asarray(solid_mesh.coords, dtype=float)
    x_min = float(np.min(solid_coords[:, 0]))
    x_max = float(np.max(solid_coords[:, 0]))
    face_nodes = np.flatnonzero(np.isclose(solid_coords[:, 0], x_max, atol=1.0e-10)).astype(int)
    face_coords = solid_coords[face_nodes]
    face_dofs = ff.vector_dofs_from_nodes(face_nodes, dim=3)
    x_ref = face_coords.mean(axis=0)

    truss_coords, truss_conn = ff.structured_truss_chain(
        n_elems=truss_elems,
        length=truss_length,
        origin=x_ref,
        direction=(1.0, 0.0, 0.0),
    )
    truss_section = ff.TrussSection(E=2.0e5, A=1.0e-2)
    truss_K = ff.assemble_truss_stiffness(truss_coords, truss_conn, truss_section, format="csr")
    truss_tip = truss_coords.shape[0] - 1
    truss_F = ff.assemble_truss_point_load(truss_coords.shape[0], truss_tip, force=(tip_load_x, 0.0, 0.0))

    structural_K = sp.block_diag((solid_K, truss_K), format="csr")
    structural_F = np.concatenate([solid_F, np.asarray(truss_F, dtype=float)])
    truss_offset = solid_space.n_dofs

    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_K, structural_F)
    builder.register_field("solid", n_dofs=solid_space.n_dofs, value_dim=1, offset=0)
    builder.register_field("truss", n_dofs=truss_K.shape[0], value_dim=1, offset=truss_offset)

    weights = ff.build_rbe3_weights(x_ref, face_coords, method="equal")
    builder.add_distributed_coupling(
        source="solid",
        source_dofs=face_dofs,
        remote="interface_remote",
        point=x_ref,
        slave_coords=face_coords,
        weights=weights,
        backend="numpy",
    )
    builder.add_dof_tie_constraint(
        master="interface_remote",
        slave="truss",
        master_dofs=np.arange(3),
        slave_dofs=ff.truss_node_dofs([0]),
    )

    fixed_solid = solid_mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_min, atol=1.0e-10),
        components="xyz",
        dof_per_node=3,
    )
    truss_lateral = truss_offset + ff.truss_node_dofs(np.arange(1, truss_coords.shape[0]), "yz")
    fixed_dofs = np.unique(np.concatenate([fixed_solid, truss_lateral]))

    return {
        "builder": builder,
        "system": builder.build(),
        "solid_mesh": solid_mesh,
        "solid_space": solid_space,
        "solid_coords": solid_coords,
        "truss_coords": truss_coords,
        "truss_conn": truss_conn,
        "face_nodes": face_nodes,
        "face_dofs": face_dofs,
        "fixed_dofs": fixed_dofs,
        "remote_dofs": builder.resolve_block_dofs("interface_remote", local_dofs=np.arange(6)),
        "truss_root_dofs": truss_offset + ff.truss_node_dofs([0]),
        "truss_tip_dofs": truss_offset + ff.truss_node_dofs([truss_tip]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Couple a 3D solid face to a 3D truss root with an RBE3 remote point.")
    p.add_argument("--truss-elems", type=int, default=4, help="Number of truss/bar elements.")
    p.add_argument("--tip-load-x", type=float, default=25.0, help="Truss tip load in global x.")
    return p.parse_args()


def main():
    args = parse_args()
    model = build_solid_truss_coupling(truss_elems=args.truss_elems, tip_load_x=args.tip_load_x)
    system = model["system"]
    fixed = model["fixed_dofs"]

    u_all = np.asarray(
        system.solve(
            format="csr",
            dirichlet_dofs=fixed,
            dirichlet_vals=np.zeros((fixed.size,), dtype=float),
        ),
        dtype=float,
    )

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False

    remote_q = u_all[model["remote_dofs"]]
    truss_root_q = u_all[model["truss_root_dofs"]]
    truss_tip_q = u_all[model["truss_tip_dofs"]]
    solid_u = u_all[: model["solid_space"].n_dofs].reshape(-1, 3)
    face_mean = solid_u[model["face_nodes"]].mean(axis=0)

    print("solid nodes:          ", model["solid_coords"].shape[0])
    print("truss nodes:          ", model["truss_coords"].shape[0])
    print("coupled unknowns:     ", K_full.shape[0])
    print("interface remote u:   ", remote_q[:3])
    print("truss root u:         ", truss_root_q)
    print("solid face mean u:    ", face_mean)
    print("truss tip u:          ", truss_tip_q)
    print("free residual norm:   ", f"{float(np.linalg.norm(residual[free])):.8e}")


if __name__ == "__main__":
    main()

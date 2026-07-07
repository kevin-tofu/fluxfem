#!/usr/bin/env python3
"""
3D solid block coupled to a 3D Euler-Bernoulli beam through an RBE3 face average.

The model uses the existing continuum weak-form assembly for the solid and the
dedicated beam helper for the line element. The right solid face is copied into
an auxiliary field, reduced to a 6-DOF remote point by an RBE3 constraint, and
the beam root DOFs are tied to that remote point.

Run from the repository root:

    PYTHONPATH=src python tutorials/elasticity/solid_beam_rbe3_coupling.py
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


def build_solid_beam_coupling(
    *,
    solid_nx: int = 3,
    solid_ny: int = 2,
    solid_nz: int = 2,
    solid_length: float = 1.0,
    solid_width: float = 0.3,
    solid_height: float = 0.3,
    beam_elems: int = 4,
    beam_length: float = 1.2,
    tip_load_z: float = -25.0,
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

    beam_coords, beam_conn = ff.structured_beam_chain(
        n_elems=beam_elems,
        length=beam_length,
        origin=x_ref,
        direction=(1.0, 0.0, 0.0),
    )
    beam_section = ff.BeamSection(E=2.0e5, G=7.7e4, A=1.0e-2, Iy=1.5e-5, Iz=1.5e-5, J=3.0e-5)
    beam_K = ff.assemble_beam_stiffness(beam_coords, beam_conn, beam_section, format="csr")
    beam_tip = beam_coords.shape[0] - 1
    beam_F = ff.assemble_beam_point_load(beam_coords.shape[0], beam_tip, force=(0.0, 0.0, tip_load_z))

    structural_K = sp.block_diag((solid_K, beam_K), format="csr")
    structural_F = np.concatenate([solid_F, np.asarray(beam_F, dtype=float)])
    beam_offset = solid_space.n_dofs

    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_K, structural_F)
    builder.register_field("solid", n_dofs=solid_space.n_dofs, value_dim=1, offset=0)
    builder.register_field("beam", n_dofs=beam_K.shape[0], value_dim=1, offset=beam_offset)

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
        slave="beam",
        master_dofs=np.arange(6),
        slave_dofs=ff.beam_node_dofs([0]),
    )

    fixed_solid = solid_mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_min, atol=1.0e-10),
        components="xyz",
        dof_per_node=3,
    )

    return {
        "builder": builder,
        "system": builder.build(),
        "solid_mesh": solid_mesh,
        "solid_space": solid_space,
        "solid_coords": solid_coords,
        "beam_coords": beam_coords,
        "beam_conn": beam_conn,
        "face_nodes": face_nodes,
        "face_dofs": face_dofs,
        "fixed_dofs": fixed_solid,
        "remote_dofs": builder.resolve_block_dofs("interface_remote", local_dofs=np.arange(6)),
        "beam_root_dofs": beam_offset + ff.beam_node_dofs([0]),
        "beam_tip_dofs": beam_offset + ff.beam_node_dofs([beam_tip]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Couple a 3D solid face to a 3D beam root with an RBE3 remote point.")
    p.add_argument("--beam-elems", type=int, default=4, help="Number of beam elements.")
    p.add_argument("--tip-load-z", type=float, default=-25.0, help="Beam tip load in global z.")
    return p.parse_args()


def main():
    args = parse_args()
    model = build_solid_beam_coupling(beam_elems=args.beam_elems, tip_load_z=args.tip_load_z)
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
    beam_root_q = u_all[model["beam_root_dofs"]]
    beam_tip_q = u_all[model["beam_tip_dofs"]]
    solid_u = u_all[: model["solid_space"].n_dofs].reshape(-1, 3)
    face_mean = solid_u[model["face_nodes"]].mean(axis=0)

    print("solid nodes:          ", model["solid_coords"].shape[0])
    print("beam nodes:           ", model["beam_coords"].shape[0])
    print("coupled unknowns:     ", K_full.shape[0])
    print("interface remote u:   ", remote_q[:3])
    print("beam root u:          ", beam_root_q[:3])
    print("solid face mean u:    ", face_mean)
    print("beam tip u:           ", beam_tip_q[:3])
    print("beam root rotation:   ", beam_root_q[3:])
    print("free residual norm:   ", f"{float(np.linalg.norm(residual[free])):.8e}")


if __name__ == "__main__":
    main()

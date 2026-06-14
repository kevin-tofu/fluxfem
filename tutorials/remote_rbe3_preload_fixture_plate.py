#!/usr/bin/env python
"""FluxFEM remote RBE3 preload fixture example for a small elastic plate.

This is the current 0.3.x-style replacement for the older hand-built
reference-point fixture examples.  It uses the public coupled-system builder:

* assemble a structural workpiece stiffness with FluxFEM,
* copy a fixture patch into an auxiliary ``support_face`` field,
* constrain the auxiliary field to a 6-DOF remote point with RBE3,
* add a preload spring on the remote point,
* solve the sparse KKT system with a fixed left edge.

Run from the repository root:

    PYTHONPATH=src python tutorials/remote_rbe3_preload_fixture_plate.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import scipy.sparse as sp

import fluxfem as ff


def vector_dofs(nodes: np.ndarray, *, dim: int = 3) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=int).reshape(-1)
    return (nodes[:, None] * dim + np.arange(dim, dtype=int)[None, :]).reshape(-1)


def main() -> None:
    mesh = ff.StructuredHexBox(nx=4, ny=2, nz=1, lx=2.0, ly=0.8, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)

    D = ff.isotropic_3d_D(100.0, 0.30)
    K = space.assemble(ff.linear_elasticity_form, params=D)
    F = np.zeros((space.n_dofs,), dtype=float)

    coords = np.asarray(mesh.coords, dtype=float)
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())

    fixed_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_min, atol=1e-10),
        components="xyz",
        dof_per_node=3,
    )
    patch_nodes = np.flatnonzero(np.isclose(coords[:, 0], x_max, atol=1e-10)).astype(int)
    patch_dofs = vector_dofs(patch_nodes, dim=3)
    patch_coords = coords[patch_nodes]
    x_ref = patch_coords.mean(axis=0) + np.array([0.15, 0.0, 0.0], dtype=float)

    builder = ff.NumpyCoupledSystemBuilder.from_structural(K, F)
    builder.register_field("workpiece", n_dofs=space.n_dofs, value_dim=1, offset=0)
    builder.append_field("support_face", n_dofs=patch_dofs.size, value_dim=1)
    builder.append_remote_point("remote", point=x_ref)

    # Tie selected workpiece DOFs to the auxiliary support-face field.
    tie = np.zeros((patch_dofs.size, space.n_dofs + patch_dofs.size), dtype=float)
    for row, dof in enumerate(patch_dofs):
        tie[row, int(dof)] = 1.0
        tie[row, space.n_dofs + row] = -1.0
    builder.add_constraint_matrix_dof(tie, master="workpiece", slave="support_face")

    weights = ff.build_rbe3_weights(x_ref, patch_coords, method="distance")
    builder.add_rbe3_constraint(
        master="remote",
        slave="support_face",
        ref_point=x_ref,
        slave_coords=patch_coords,
        weights=weights,
    )
    builder.add_remote_spring(
        "remote",
        translational_stiffness=np.array([200.0, 200.0, 200.0], dtype=float),
        rotational_stiffness=np.array([1.0e6, 1.0e6, 1.0e6], dtype=float),
        translational_target=np.array([0.0, 0.0, -0.01], dtype=float),
    )

    system = builder.build()
    u_all = np.asarray(
        system.solve(
            format="csr",
            dirichlet_dofs=fixed_dofs,
            dirichlet_vals=np.zeros((fixed_dofs.size,), dtype=float),
            diagonal_shift=1.0e-10,
        ),
        dtype=float,
    )

    workpiece_u = u_all[: space.n_dofs].reshape(-1, 3)
    remote_dofs = builder.resolve_block_dofs("remote", local_dofs=np.arange(6))
    remote_q = u_all[remote_dofs]
    patch_u = workpiece_u[patch_nodes]

    K_full, F_full = system.assemble(format="csr")
    compliance = float(np.asarray(F_full, dtype=float) @ u_all)
    residual_vec = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed_dofs] = False
    free_residual = float(np.linalg.norm(residual_vec[free]))

    print("workpiece DOFs:       ", space.n_dofs)
    print("patch nodes:          ", patch_nodes.size)
    print("coupled unknowns:     ", K_full.shape[0])
    print("remote displacement:  ", remote_q[:3])
    print("remote rotation:      ", remote_q[3:])
    print("mean patch z disp:    ", float(np.mean(patch_u[:, 2])))
    print("max workpiece |u|:    ", float(np.max(np.abs(workpiece_u))))
    print("compliance:           ", f"{compliance:.8e}")
    print("free residual norm:  ", f"{free_residual:.8e}")


if __name__ == "__main__":
    main()

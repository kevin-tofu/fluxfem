#!/usr/bin/env python
"""Remote RBE3 spring compliance example using the 0.3.x coupled-system API.

The model has two structural surface nodes, a 6-DOF remote point, and a
slave-face auxiliary field.  Structural DOFs are tied to the auxiliary field,
the auxiliary field is interpolated to the remote point by an RBE3 constraint,
and a translational spring preload is applied to the remote point.

Run from the repository root:

    PYTHONPATH=src python tutorials/remote_rbe3_spring_compliance.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import fluxfem as ff


def main() -> None:
    # Two 3D surface nodes with simple structural grounding stiffness.
    slave_coords = np.array(
        [
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    x_ref = np.array([0.0, 0.0, 0.2], dtype=float)
    n_structural = 3 * slave_coords.shape[0]

    structural_stiffness = sp.eye(n_structural, format="csr") * 50.0
    structural_force = np.zeros((n_structural,), dtype=float)

    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_stiffness, structural_force)
    builder.register_field("workpiece", n_dofs=n_structural, value_dim=1, offset=0)

    # The face field is an auxiliary copy of selected structural DOFs.
    builder.append_dof_copy_field("support_face", source="workpiece", source_dofs=np.arange(n_structural))
    builder.append_remote_point("remote", point=x_ref)

    weights = ff.build_rbe3_weights(x_ref, slave_coords, method="equal")
    builder.add_rbe3_constraint(
        master="remote",
        slave="support_face",
        ref_point=x_ref,
        slave_coords=slave_coords,
        weights=weights,
    )

    builder.add_remote_spring(
        "remote",
        translational_stiffness=np.array([10.0, 10.0, 10.0], dtype=float),
        rotational_stiffness=np.array([1.0e6, 1.0e6, 1.0e6], dtype=float),
        translational_target=np.array([0.0, 0.0, 0.02], dtype=float),
    )

    system = builder.build()
    u = np.asarray(system.solve(format="csr", diagonal_shift=1.0e-9), dtype=float)

    workpiece_u = u[:n_structural].reshape(-1, 3)
    remote_dofs = builder.resolve_block_dofs("remote", local_dofs=np.arange(6))
    remote_q = u[remote_dofs]
    _K_full, F_full = system.assemble(format="csr")
    compliance = float(np.asarray(F_full, dtype=float) @ u)

    print("remote displacement:", remote_q[:3])
    print("remote rotation:    ", remote_q[3:])
    print("surface z disp:    ", workpiece_u[:, 2])
    print("compliance:        ", f"{compliance:.8e}")


if __name__ == "__main__":
    main()

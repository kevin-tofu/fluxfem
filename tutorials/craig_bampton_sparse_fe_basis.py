#!/usr/bin/env python
"""Craig-Bampton ROM basis from current FluxFEM sparse FE matrices.

The example condenses a scalar diffusion bar by fixing the left face and
retaining the right-face DOFs as physical interface coordinates.  The internal
coordinates are represented by static constraint modes plus a small number of
fixed-interface modes.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton_sparse_fe_basis.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def main() -> None:
    mesh = ff.StructuredHexBox(nx=8, ny=1, nz=1, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    K = space.assemble(ff.diffusion_form, params=1.0)
    M = space.assemble_mass_matrix()

    coords = np.asarray(mesh.coords, dtype=float)
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    left = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_min, atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    right = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_max, atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    free = np.asarray(ff.free_dofs(space.n_dofs, left), dtype=int)
    retained = np.flatnonzero(np.isin(free, np.asarray(right, dtype=int))).astype(int)

    k_free = K.to_csr()[free, :][:, free]
    m_free = M.to_csr()[free, :][:, free]

    cb = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained_dofs=retained,
        n_modes=4,
        constraint_solver="spsolve",
        modal_solver="eigsh",
    )

    # `project_matrix` keeps SciPy/FluxFEM sparse inputs sparse during K @ Phi.
    k_red = np.asarray(cb.project_matrix(k_free), dtype=float)
    m_red = np.asarray(cb.project_matrix(m_free), dtype=float)
    k_reduced_operator = cb.project_operator(k_free)
    q_probe = jnp.linspace(-0.2, 0.2, cb.n_reduced, dtype=jnp.float64)
    operator_error = np.linalg.norm(np.asarray(k_reduced_operator.matvec(q_probe)) - k_red @ np.asarray(q_probe))
    retained_identity_error = np.linalg.norm(
        np.asarray(cb.basis)[np.ix_(retained, np.arange(retained.size))] - np.eye(retained.size)
    )

    def full_residual(u):
        return jnp.asarray(k_free.toarray()) @ u + 0.05 * u**3

    q = jnp.zeros((cb.n_reduced,), dtype=jnp.float64)
    j_red = np.asarray(cb.reduced_jacobian(full_residual)(q), dtype=float)

    print("full free DOFs:       ", cb.n_full)
    print("retained DOFs:        ", cb.n_retained)
    print("fixed-interface modes:", cb.n_modes)
    print("reduced DOFs:         ", cb.n_reduced)
    print("lowest eigenvalues:   ", np.asarray(cb.eigenvalues))
    print("K_red shape:          ", k_red.shape)
    print("M_red shape:          ", m_red.shape)
    print("K operator matvec err:", f"{operator_error:.3e}")
    print("retained identity err:", f"{retained_identity_error:.3e}")
    print("AD Jacobian shape:    ", j_red.shape)
    print("AD Jacobian symmetry: ", f"{np.linalg.norm(j_red - j_red.T):.3e}")


if __name__ == "__main__":
    main()

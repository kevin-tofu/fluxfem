import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


def _condensed_bar_matrices(nx: int = 4):
    mesh = ff.StructuredHexBox(nx=nx, ny=1, nz=1, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    stiffness = space.assemble(ff.diffusion_form, params=1.0)
    mass = space.assemble_mass_matrix()

    coords = np.asarray(mesh.coords, dtype=float)
    left = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], coords[:, 0].min(), atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    right = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], coords[:, 0].max(), atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    free = np.asarray(ff.free_dofs(space.n_dofs, left), dtype=int)
    retained = np.flatnonzero(np.isin(free, np.asarray(right, dtype=int))).astype(int)
    return stiffness, mass, free, retained


def test_craig_bampton_dense_static_modes_satisfy_partitioned_equilibrium():
    stiffness = np.array(
        [
            [4.0, -1.0, 0.0, -0.5],
            [-1.0, 3.0, -0.4, 0.0],
            [0.0, -0.4, 2.6, -0.8],
            [-0.5, 0.0, -0.8, 2.8],
        ],
        dtype=float,
    )
    mass = np.diag([2.0, 1.5, 1.2, 1.0])
    retained = np.array([0, 3], dtype=int)

    cb = ff.make_craig_bampton_basis(stiffness, mass, retained, n_modes=1)

    internal = np.asarray(cb.internal_dofs)
    k_ii = stiffness[np.ix_(internal, internal)]
    k_ir = stiffness[np.ix_(internal, retained)]
    psi = np.asarray(cb.basis)[np.ix_(internal, np.arange(retained.size))]

    np.testing.assert_allclose(k_ii @ psi + k_ir, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(cb.basis)[retained, : retained.size], np.eye(2), atol=1.0e-12)
    assert cb.n_reduced == retained.size + 1


def test_craig_bampton_reduced_residual_keeps_jax_autodiff_path():
    stiffness = jnp.array(
        [
            [3.0, -1.0, 0.0],
            [-1.0, 2.5, -0.5],
            [0.0, -0.5, 1.8],
        ],
        dtype=jnp.float64,
    )
    mass = jnp.eye(3, dtype=jnp.float64)
    cb = ff.make_craig_bampton_basis(stiffness, mass, jnp.array([0], dtype=jnp.int32), n_modes=1)

    def residual(u):
        return stiffness @ u + 0.2 * u**3

    q = jnp.array([0.1, -0.03], dtype=jnp.float64)
    jac = cb.reduced_jacobian(residual)(q)
    u = cb.expand(q)
    expected = cb.basis.T @ (stiffness + jnp.diag(0.6 * u**2)) @ cb.basis

    np.testing.assert_allclose(np.asarray(jac), np.asarray(expected), rtol=1.0e-10, atol=1.0e-12)


def test_craig_bampton_sparse_fluxfem_matrices_match_dense_eigenvalues():
    pytest.importorskip("scipy")

    stiffness, mass, free, retained = _condensed_bar_matrices(nx=4)
    k_free = stiffness.to_csr()[free, :][:, free]
    m_free = mass.to_csr()[free, :][:, free]

    cb_sparse = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained,
        n_modes=2,
        constraint_solver="spsolve",
        modal_solver="eigsh",
    )
    cb_dense = ff.make_craig_bampton_basis(
        k_free.toarray(),
        m_free.toarray(),
        retained,
        n_modes=2,
        constraint_solver="dense",
        modal_solver="dense",
    )

    np.testing.assert_allclose(np.asarray(cb_sparse.eigenvalues), np.asarray(cb_dense.eigenvalues), rtol=1.0e-8)
    np.testing.assert_allclose(
        np.asarray(cb_sparse.basis)[retained, : retained.size],
        np.eye(retained.size),
        atol=1.0e-12,
    )
    assert cb_sparse.n_full == free.size
    assert cb_sparse.n_modes == 2


def test_craig_bampton_top_level_exports_are_available():
    assert ff.CraigBamptonBasis is not None
    assert ff.make_craig_bampton_basis is ff.solver.make_craig_bampton_basis

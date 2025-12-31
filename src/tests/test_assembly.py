"""Assembly API consistency and regression checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff
from fluxfem.tools.timer import SectionTimer


def _diffusion_dense_manual(space, kappa):
    """Assemble diffusion matrix explicitly via basis grads (regression helper)."""
    mesh, basis = space.mesh, space.basis
    elem_coords = np.asarray(mesh.element_coords())  # (n_elems, n_nodes, 3)
    w = np.asarray(basis.quad_weights)               # (n_q,)
    K = np.zeros((space.n_dofs, space.n_dofs), dtype=w.dtype)

    for e, Xe in enumerate(elem_coords):
        dN_dx, detJ = basis.spatial_grads_and_detJ(jnp.asarray(Xe))
        G = jnp.einsum("qia,qja->qij", dN_dx, dN_dx)           # (n_q, n_nodes, n_nodes)
        Ke = (kappa * G * (w * detJ)[:, None, None]).sum(axis=0)
        dofs = np.asarray(space.elem_dofs[e])
        np.add.at(K, (dofs[:, None], dofs[None, :]), np.asarray(Ke))
    return K


def _mass_dense_manual(space):
    """Assemble mass matrix explicitly via basis shape functions (regression helper)."""
    mesh, basis = space.mesh, space.basis
    elem_coords = np.asarray(mesh.element_coords())  # (n_elems, n_nodes, 3)
    N_ref = np.asarray(basis.shape_functions())      # (n_q, n_nodes)
    w = np.asarray(basis.quad_weights)               # (n_q,)
    M = np.zeros((space.n_dofs, space.n_dofs), dtype=w.dtype)

    for e, Xe in enumerate(elem_coords):
        _, detJ = basis.spatial_grads_and_detJ(jnp.asarray(Xe))
        Me = np.einsum("qa,qb,q->ab", N_ref, N_ref, w * np.asarray(detJ))
        dofs = np.asarray(space.elem_dofs[e])
        np.add.at(M, (dofs[:, None], dofs[None, :]), np.asarray(Me))
    return M


@pytest.mark.parametrize("intorder", [2, 3])
def test_diffusion_matrix_agrees(intorder):
    """Form-based assembly matches manual integration."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=intorder)

    K_form = np.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense())
    K_manual = _diffusion_dense_manual(space, kappa=1.0)

    assert K_form.shape == (8, 8)
    max_diff = float(jnp.max(jnp.abs(K_form - K_manual)))
    assert max_diff < 1e-6, f"K mismatch: {max_diff}"


@pytest.mark.parametrize("intorder", [2, 3])
def test_mass_matrix_agrees(intorder):
    """Mass assembly matches manual integration."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=intorder)

    M_form = np.asarray(space.assemble_mass_matrix().to_dense())
    M_manual = _mass_dense_manual(space)

    assert M_form.shape == (8, 8)
    max_diff = float(jnp.max(jnp.abs(M_form - M_manual)))
    assert max_diff < 1e-6, f"M mismatch: {max_diff}"


@pytest.mark.parametrize("n_chunks", [None, 2, 3])
def test_linear_form_chunk_consistency(n_chunks):
    """Chunked linear form matches non-chunked assembly."""
    mesh = ff.StructuredHexBox(nx=5, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    F_ref = space.assemble_linear_form(ff.scalar_body_force_form, params=2.0)
    F_chk = space.assemble_linear_form(ff.scalar_body_force_form, params=2.0, n_chunks=n_chunks)
    assert np.allclose(np.asarray(F_ref), np.asarray(F_chk))


@pytest.mark.parametrize("n_chunks", [None, 2, 3])
def test_mass_matrix_chunk_consistency(n_chunks):
    """Chunked mass matrix matches non-chunked assembly."""
    mesh = ff.StructuredHexBox(nx=5, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    M_ref = space.assemble_mass_matrix()
    M_chk = space.assemble_mass_matrix(n_chunks=n_chunks)
    assert np.allclose(np.asarray(M_ref.to_dense()), np.asarray(M_chk.to_dense()))


@pytest.mark.parametrize("n_chunks", [None, 2, 3])
def test_bilinear_form_chunk_consistency(n_chunks):
    """Chunked bilinear form matches non-chunked assembly."""
    mesh = ff.StructuredHexBox(nx=5, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    K_ref = space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense()
    K_chk = space.assemble_bilinear_form(ff.diffusion_form, params=1.0, n_chunks=n_chunks).to_dense()
    assert np.allclose(np.asarray(K_ref), np.asarray(K_chk))

def test_sparse_bilinear_matches_dense():
    """ff.FluxSparseMatrix dense matches manual integration."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    A = space.assemble_bilinear_form(ff.diffusion_form, params=1.0)
    K_dense = np.asarray(A.to_dense())
    K_manual = _diffusion_dense_manual(space, kappa=1.0)
    assert np.allclose(K_dense, K_manual)


def test_sparse_linear_matches_dense():
    """Check sparse linear-form output matches dense."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    F_dense = np.asarray(space.assemble_linear_form(ff.scalar_body_force_form, params=2.0))

    rows, data, n = space.assemble_linear_form(ff.scalar_body_force_form, params=2.0, sparse=True)
    F_sparse = np.zeros(n, dtype=np.asarray(data).dtype)
    np.add.at(F_sparse, np.asarray(rows, dtype=int), np.asarray(data))
    assert np.allclose(F_sparse, F_dense)


def test_flux_sparse_matrix_matvec_and_dense():
    """Check matvec/dense consistency via ff.FluxSparseMatrix."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    A = space.assemble_bilinear_form(ff.diffusion_form, params=1.0)
    K_dense = np.asarray(A.to_dense())
    x = np.arange(K_dense.shape[0], dtype=np.float32)

    y_dense = K_dense @ x
    y_sparse = np.asarray(A.matvec(x))
    assert np.allclose(y_sparse, y_dense)

    K_from_A = np.asarray(A.to_dense())
    assert np.allclose(K_from_A, K_dense)


def test_assemble_returns_flux_matrix():
    """Check return_flux_matrix=True returns ff.FluxSparseMatrix."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    A = space.assemble_bilinear_form(ff.diffusion_form, params=1.0)
    assert isinstance(A, ff.FluxSparseMatrix)
    x = np.ones(A.n_dofs, dtype=np.float32)
    y1 = np.asarray(A.matvec(x))
    y2 = np.asarray(A.to_dense() @ x)
    assert np.allclose(y1, y2, atol=1e-6, rtol=1e-6)


def test_jacobian_assemble_reuses_pattern():
    """assemble_jacobian_scatter accepts a prebuilt pattern."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    pattern = ff.make_sparsity_pattern(space, with_idx=True)

    def simple_residual(ctx, u_elem, params):
        # element residual = sum_q u_elem * wJ  -> Jacobian = (Σ wJ) * I
        return jnp.broadcast_to(u_elem, (ctx.w.shape[0], u_elem.shape[0]))

    u = jnp.zeros(space.n_dofs)
    K_pattern = space.assemble_jacobian(
        simple_residual, u, params=None, pattern=pattern, sparse=False
    )
    K_default = space.assemble_jacobian(
        simple_residual, u, params=None, sparse=False
    )
    # K_pattern = assemble_jacobian_scatter(
    #     space, simple_residual, u, params=None, pattern=pattern, sparse=False
    # )
    # K_default = assemble_jacobian_scatter(
    #     space, simple_residual, u, params=None, sparse=False
    # )
    np.testing.assert_allclose(np.asarray(K_pattern), np.asarray(K_default))

    A_sparse = space.assemble_jacobian(
        simple_residual, u, params=None,
        pattern=pattern, sparse=True, return_flux_matrix=True
    )
    vec = np.ones(space.n_dofs, dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(A_sparse.matvec(vec)), np.asarray(K_default) @ vec
    )


def test_structured_hex_box_connectivity():
    """Structured grid produces expected conn when viewed as unstructured."""
    box = ff.StructuredHexBox(nx=2, ny=1, nz=1, lx=2.0, ly=1.0, lz=1.0)
    mesh = box.build()

    assert mesh.coords.shape == (12, 3)
    assert mesh.conn.shape == (2, 8)

    expected_conn = jnp.array(
        [
            [0, 1, 4, 3, 6, 7, 10, 9],   # element at i=0
            [1, 2, 5, 4, 7, 8, 11, 10],  # element at i=1
        ],
        dtype=jnp.int64,
    )
    assert jnp.array_equal(mesh.conn, expected_conn), f"conn mismatch:\n{mesh.conn}"



def test_linear_form_constant_body_force():
    """Check linear form integrates constant body force f."""
    fval = 2.0
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    F = space.assemble_linear_form(ff.scalar_body_force_form, params=fval)
    expected = np.full((8,), fval / 8.0, dtype=np.float32)  # vol=1 → ∫N_i=1/8
    assert np.allclose(np.asarray(F), expected, rtol=1e-6, atol=1e-6)


def test_functional_integrates_volume():
    """Functional of 1 integrates to volume."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=2.0, ly=3.0, lz=4.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    def one(ctx, params):
        return jnp.ones_like(ctx.w)

    val = float(space.assemble_functional(one, params=None))
    assert np.isclose(val, 24.0, rtol=1e-6, atol=1e-6)


def test_tag_axis_minmax_facets():
    """Check x=0 and x=max facets get tagged."""
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    facets, tags = ff.tag_axis_minmax_facets(mesh, axis=0, dirichlet_tag=11, neumann_tag=22)
    assert facets.shape == (2, 4)
    assert set(np.asarray(tags).tolist()) == {11, 22}
    # in structured box numbering: x=0 -> {0,2,4,6}, x=1 -> {1,3,5,7}
    faces = [set(f) for f in np.asarray(facets).tolist()]
    assert set([0, 2, 4, 6]) in faces
    assert set([1, 3, 5, 7]) in faces


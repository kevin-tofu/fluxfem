"""Mortar-style coupling parity against scikit-fem (including P0 + AL block)."""

from __future__ import annotations

import numpy as np
import pytest

import fluxfem as ff


def _tet4_coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _perm_by_coords(coords_ff: np.ndarray, doflocs_sf: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    coords_ff = np.asarray(coords_ff, dtype=float)
    doflocs_sf = np.asarray(doflocs_sf, dtype=float)
    if doflocs_sf.shape[0] == 3 and doflocs_sf.shape[1] != 3:
        doflocs_sf = doflocs_sf.T
    perm = np.empty(coords_ff.shape[0], dtype=int)
    for i, c in enumerate(coords_ff):
        matches = np.nonzero(np.all(np.isclose(doflocs_sf, c, atol=atol), axis=1))[0]
        if len(matches) != 1:
            raise RuntimeError("scalar dof mapping is ambiguous")
        perm[i] = int(matches[0])
    return perm


def _coo_to_dense(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=float)
    for r, c, v in zip(rows, cols, data):
        out[int(r), int(c)] += float(v)
    return out


def _p0_reduction_from_surface_facets(surface: ff.SurfaceMesh) -> np.ndarray:
    """Facet-wise P0 lambda basis expressed on nodal interface basis: lambda = S * lambda_nodal."""
    facets = np.asarray(surface.conn, dtype=int)
    n_facets = int(facets.shape[0])
    n_nodes = int(np.asarray(surface.coords).shape[0])
    S = np.zeros((n_facets, n_nodes), dtype=float)
    for f, nodes in enumerate(facets):
        S[f, np.asarray(nodes, dtype=int)] = 1.0
    return S


def test_contact_coupling_and_p0_augmented_lagrangian_block_match_skfem_tet4():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshTet, Basis, FacetBasis, ElementTetP1, asm
    from skfem.supermeshing import intersect, elementwise_quadrature

    coords = _tet4_coords()
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2]], dtype=int)
    quad_order = 1
    rho = 5.0

    mesh_m = ff.TetMesh(coords=coords, conn=conn)
    mesh_s = ff.TetMesh(coords=coords, conn=conn)

    def select_contact(mesh):
        return mesh.facets_on_plane(axis=2, value=0.0)

    contact = ff.OneToManyContactSurfaceSpace.from_meshes(
        master_mesh=mesh_m,
        slave_meshes=[mesh_s],
        master_facet_selector=select_contact,
        slave_facet_selectors=select_contact,
        value_dim_master=1,
        value_dim_slaves=1,
        quad_order=quad_order,
    )
    m_aa_ff, m_ab_ff = contact.assemble_contact_coupling_matrices()
    M_aa_ff = _coo_to_dense(m_aa_ff.rows, m_aa_ff.cols, m_aa_ff.data, m_aa_ff.shape)
    M_ab_ff = _coo_to_dense(m_ab_ff.rows, m_ab_ff.cols, m_ab_ff.data, m_ab_ff.shape)

    mesh_a = MeshTet(coords.T, conn.T).with_boundaries({"contact": lambda x: np.isclose(x[2], 0.0)})
    mesh_b = MeshTet(coords.T, conn.T).with_boundaries({"contact": lambda x: np.isclose(x[2], 0.0)})
    elem_s = ElementTetP1()
    m1t, orig1 = mesh_a.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m2t, orig2 = mesh_b.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m12, t1, t2 = intersect(m1t, m2t)
    try:
        quad1 = elementwise_quadrature(m1t, m12, t1, intorder=quad_order)
        quad2 = elementwise_quadrature(m2t, m12, t2, intorder=quad_order)
    except TypeError:
        quad1 = elementwise_quadrature(m1t, m12, t1)
        quad2 = elementwise_quadrature(m2t, m12, t2)

    basis_a = Basis(mesh_a, elem_s)
    basis_b = Basis(mesh_b, elem_s)
    fb_a = FacetBasis(mesh_a, elem_s, facets=orig1[t1], quadrature=quad1)
    fb_b = FacetBasis(mesh_b, elem_s, facets=orig2[t2], quadrature=quad2)
    fbasis = fb_a * fb_b

    @skfem.BilinearForm
    def mass_aa(u1, u2, v1, v2, w):
        return u1 * v1

    @skfem.BilinearForm
    def mass_ab(u1, u2, v1, v2, w):
        return u2 * v1

    A_aa_full = asm(mass_aa, fbasis).toarray()
    A_ab_full = asm(mass_ab, fbasis).toarray()
    n_a = int(fb_a.N)
    n_b = int(fb_b.N)
    M_aa_sf = A_aa_full[:n_a, :n_a]
    M_ab_sf = A_ab_full[:n_a, n_a : n_a + n_b]

    perm_a = _perm_by_coords(coords, np.asarray(basis_a.doflocs))
    perm_b = _perm_by_coords(coords, np.asarray(basis_b.doflocs))
    M_aa_sf = M_aa_sf[np.ix_(perm_a, perm_a)]
    M_ab_sf = M_ab_sf[np.ix_(perm_a, perm_b)]

    # Nodal matrices differ because FluxFEM uses centroid quadrature for coupling assembly,
    # while skfem asm here gives the exact P1 surface mass integration.
    # For P0 multipliers (facet-wise constant), both reduce to the same operator.

    # P0 multiplier reduction per facet (one lambda dof per facet).
    S = _p0_reduction_from_surface_facets(contact.contacts[0].surface_master)
    B_a_ff = S @ M_aa_ff
    B_b_ff = S @ M_ab_ff
    B_a_sf = S @ M_aa_sf
    B_b_sf = S @ M_ab_sf
    assert np.allclose(B_a_ff, B_a_sf, atol=1e-10)
    assert np.allclose(B_b_ff, B_b_sf, atol=1e-10)

    # Augmented Lagrangian linearization block:
    # [ rho*B^T B   B^T ]
    # [    B        0   ]  with B = [B_a, -B_b].
    B_ff = np.hstack([B_a_ff, -B_b_ff])
    B_sf = np.hstack([B_a_sf, -B_b_sf])
    Kuu_ff = rho * (B_ff.T @ B_ff)
    Kuu_sf = rho * (B_sf.T @ B_sf)
    Zll = np.zeros((B_ff.shape[0], B_ff.shape[0]), dtype=float)
    KKT_ff = np.block([[Kuu_ff, B_ff.T], [B_ff, Zll]])
    KKT_sf = np.block([[Kuu_sf, B_sf.T], [B_sf, Zll]])
    assert np.allclose(KKT_ff, KKT_sf, atol=1e-10)


def test_contact_p0_supermesh_operators_match_skfem_intersection_triangle_p0_tet4():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshTet, Basis, FacetBasis, ElementTetP1, ElementVector, ElementTetP0, asm
    from skfem.supermeshing import intersect, elementwise_quadrature

    coords = _tet4_coords()
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    quad_order = 1
    rho = 5.0

    facets = np.array([[0, 1, 2]], dtype=int)
    contact = ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=3,
        value_dim_slave=3,
        quad_order=quad_order,
    )
    mult = ff.ContactMultiplierSpace.from_contact(contact, family="p0_supermesh", side="master", value_dim=3)
    ops = ff.assemble_contact_constraint_operators(contact, rho=rho, multiplier=mult, backend="numpy")
    B_ff = np.asarray(ops.B, dtype=float)
    Kuu_ff = np.asarray(ops.Kuu, dtype=float)

    mesh_a = MeshTet(coords.T, conn.T).with_boundaries({"contact": lambda x: np.isclose(x[2], 0.0)})
    mesh_b = MeshTet(coords.T, conn.T).with_boundaries({"contact": lambda x: np.isclose(x[2], 0.0)})
    elem_u = ElementVector(ElementTetP1())
    m1t, orig1 = mesh_a.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m2t, orig2 = mesh_b.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m12, t1, t2 = intersect(m1t, m2t)
    try:
        quad1 = elementwise_quadrature(m1t, m12, t1, intorder=quad_order)
        quad2 = elementwise_quadrature(m2t, m12, t2, intorder=quad_order)
    except TypeError:
        quad1 = elementwise_quadrature(m1t, m12, t1)
        quad2 = elementwise_quadrature(m2t, m12, t2)

    fb_u_top = FacetBasis(mesh_a, elem_u, facets=orig1[t1], quadrature=quad1)
    fb_u_bot = FacetBasis(mesh_b, elem_u, facets=orig2[t2], quadrature=quad2)
    elem_lam = ElementVector(ElementTetP0())
    fb_lam = FacetBasis(mesh_a, elem_lam, facets=orig1[t1], quadrature=quad1)

    @skfem.BilinearForm
    def b_dot(u, v, w):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    B1_sf = asm(b_dot, fb_u_top, fb_lam).toarray()
    B2_sf = asm(b_dot, fb_u_bot, fb_lam).toarray()

    basis_a = Basis(mesh_a, ElementTetP1())
    basis_b = Basis(mesh_b, ElementTetP1())
    perm_a = _perm_by_coords(coords, np.asarray(basis_a.doflocs))
    perm_b = _perm_by_coords(coords, np.asarray(basis_b.doflocs))
    perm_u_a = np.array([3 * n + c for n in perm_a for c in range(3)], dtype=int)
    perm_u_b = np.array([3 * n + c for n in perm_b for c in range(3)], dtype=int)
    B_sf = np.hstack([B1_sf[:, perm_u_a], -B2_sf[:, perm_u_b]])

    assert B_ff.shape == B_sf.shape
    assert np.allclose(B_ff, B_sf, atol=1e-10)

    Kuu_sf = rho * (B_sf.T @ B_sf)
    assert np.allclose(Kuu_ff, Kuu_sf, atol=1e-10)

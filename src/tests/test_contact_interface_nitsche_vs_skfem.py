"""Nitsche contact-interface Jacobian parity against scikit-fem."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import einsum as wf_einsum


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


def _vector_perm_for_skfem(
    coords_ff: np.ndarray,
    scalar_doflocs: np.ndarray,
    vector_doflocs: np.ndarray,
    value_dim: int,
    *,
    atol: float = 1e-8,
) -> np.ndarray:
    scalar_doflocs = np.asarray(scalar_doflocs, dtype=float)
    if scalar_doflocs.shape[0] == 3 and scalar_doflocs.shape[1] != 3:
        scalar_doflocs = scalar_doflocs.T
    vector_doflocs = np.asarray(vector_doflocs, dtype=float)
    if vector_doflocs.shape[0] == 3 and vector_doflocs.shape[1] != 3:
        vector_doflocs = vector_doflocs.T

    perm_nodes = _perm_by_coords(np.asarray(coords_ff, dtype=float), scalar_doflocs, atol=atol)
    n_nodes = int(coords_ff.shape[0])
    if vector_doflocs.shape[0] != n_nodes * value_dim:
        raise RuntimeError("vector doflocs size mismatch")

    node_major = np.repeat(scalar_doflocs, value_dim, axis=0)
    comp_major = np.tile(scalar_doflocs, (value_dim, 1))
    if np.allclose(node_major, vector_doflocs, atol=atol):
        order = "node"
    elif np.allclose(comp_major, vector_doflocs, atol=atol):
        order = "component"
    else:
        order = "node"

    if order == "component":
        return np.array(
            [comp * n_nodes + perm_nodes[node] for node in range(n_nodes) for comp in range(value_dim)],
            dtype=int,
        )
    return np.array(
        [perm_nodes[node] * value_dim + comp for node in range(n_nodes) for comp in range(value_dim)],
        dtype=int,
    )


def test_nitsche_contact_interface_jacobian_matches_skfem_tet4():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshTet, Basis, FacetBasis, ElementTetP1, ElementVectorH1, asm
    from skfem.helpers import dot, sym_grad, mul
    from skfem.supermeshing import intersect, elementwise_quadrature
    from skfem.models.elasticity import lame_parameters, linear_stress

    coords = _tet4_coords()
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2]], dtype=int)
    alpha = 10.0
    quad_order = 2

    mesh_m = ff.TetMesh(coords=coords, conn=conn)
    mesh_s = ff.TetMesh(coords=coords, conn=conn)

    def select_contact(mesh):
        return mesh.facets_on_plane(axis=2, value=0.0)

    contact = ff.OneToManyContactSurfaceSpace.from_meshes(
        master_mesh=mesh_m,
        slave_meshes=[mesh_s],
        master_facet_selector=select_contact,
        slave_facet_selectors=select_contact,
        value_dim_master=3,
        value_dim_slaves=3,
        quad_order=quad_order,
        normal_sign=-1.0,
    )

    E, nu = 210e9, 0.3
    lam, mu = ff.lame_parameters(E, nu)

    def bilin(v1, v2, u1, u2, p):
        n = h_wf.normal()
        ju = u1.val - u2.val
        t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
        t_v1 = h_wf.traction(v1, n, p)
        t_v2 = h_wf.traction(v2, n, p)
        penalty = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        traction = -h_wf.dot(v1, t_u) + h_wf.dot(v2, t_u)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
        return (penalty + traction) * h_wf.ds()

    u0 = jnp.zeros(coords.shape[0] * 3)
    params = ff.Params(alpha=alpha, inv_h=1.0, lam=float(lam), mu=float(mu))
    K_ff = np.asarray(contact.assemble_bilinear(bilin, u0, [u0], params, sparse=False), dtype=float)

    mesh_a = MeshTet(coords.T, conn.T).with_boundaries({"contact": lambda x: np.isclose(x[2], 0.0)})
    mesh_b = MeshTet(coords.T, conn.T).with_boundaries({"contact": lambda x: np.isclose(x[2], 0.0)})
    elem_s = ElementTetP1()
    elem_v = ElementVectorH1(elem_s)
    m1t, orig1 = mesh_a.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m2t, orig2 = mesh_b.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m12, t1, t2 = intersect(m1t, m2t)
    try:
        quad1 = elementwise_quadrature(m1t, m12, t1, intorder=quad_order)
        quad2 = elementwise_quadrature(m2t, m12, t2, intorder=quad_order)
    except TypeError:
        quad1 = elementwise_quadrature(m1t, m12, t1)
        quad2 = elementwise_quadrature(m2t, m12, t2)

    basis_scalar_a = Basis(mesh_a, elem_s)
    basis_scalar_b = Basis(mesh_b, elem_s)
    basis_vec_a = Basis(mesh_a, elem_v)
    basis_vec_b = Basis(mesh_b, elem_v)
    fb_u_a = FacetBasis(mesh_a, elem_v, facets=orig1[t1], quadrature=quad1)
    fb_u_b = FacetBasis(mesh_b, elem_v, facets=orig2[t2], quadrature=quad2)
    fbasis = fb_u_a * fb_u_b

    lam_sf, mu_sf = lame_parameters(E, nu)
    C = linear_stress(lam_sf, mu_sf)

    @skfem.BilinearForm
    def bilin_sf(u1, u2, v1, v2, w):
        ju = u1 - u2
        t_u = 0.5 * (mul(C(sym_grad(u1)), w.n) + mul(C(sym_grad(u2)), w.n))
        t_v1 = mul(C(sym_grad(v1)), w.n)
        t_v2 = mul(C(sym_grad(v2)), w.n)
        return (alpha / w.h) * dot(v1 - v2, ju) - dot(v1, t_u) + dot(v2, t_u) - 0.5 * dot(t_v1, ju) - 0.5 * dot(t_v2, ju)

    K_sf = asm(bilin_sf, fbasis, h=fb_u_a.mesh_parameters()).toarray()
    perm_a = _vector_perm_for_skfem(coords, np.asarray(basis_scalar_a.doflocs), np.asarray(basis_vec_a.doflocs), 3)
    perm_b = _vector_perm_for_skfem(coords, np.asarray(basis_scalar_b.doflocs), np.asarray(basis_vec_b.doflocs), 3) + int(fb_u_a.N)
    perm = np.concatenate([perm_a, perm_b])
    K_sf = K_sf[np.ix_(perm, perm)]

    assert K_ff.shape == K_sf.shape
    diff = K_ff - K_sf
    rel_inf = float(np.linalg.norm(diff, ord=np.inf) / max(1.0, np.linalg.norm(K_sf, ord=np.inf)))
    assert rel_inf < 1e-5

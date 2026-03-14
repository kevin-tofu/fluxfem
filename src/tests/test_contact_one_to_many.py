import numpy as np
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf


def _penalty_bilinear(v1, v2, u1, u2, p):
    ju = u1.val - u2.val
    term = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
    return term * h_wf.ds()


def _params():
    return ff.Params(alpha=10.0, inv_h=1.0)


def test_one_to_many_contact_bilinear_matches_pair_sum():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2]], dtype=int)

    surf_m = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s1 = ff.SurfaceMesh.from_facets(coords, facets)
    surf_s2 = ff.SurfaceMesh.from_facets(coords, facets)

    side_m = ff.ContactSide.from_surfaces(surf_m, elem_conn=conn, value_dim=3)
    side_s1 = ff.ContactSide.from_surfaces(surf_s1, elem_conn=conn, value_dim=3)
    side_s2 = ff.ContactSide.from_surfaces(surf_s2, elem_conn=conn, value_dim=3)

    otm = ff.OneToManyContactSurfaceSpace.from_sides(
        side_m,
        [side_s1, side_s2],
        quad_order=1,
        backend="jax",
    )

    pair_1 = ff.ContactSurfaceSpace.from_sides(side_m, side_s1, quad_order=1, backend="jax")
    pair_2 = ff.ContactSurfaceSpace.from_sides(side_m, side_s2, quad_order=1, backend="jax")

    n_m = coords.shape[0] * 3
    n_s = coords.shape[0] * 3
    u_m = jnp.zeros(n_m)
    u_s1 = jnp.zeros(n_s)
    u_s2 = jnp.zeros(n_s)
    params = _params()

    K_pair_1 = np.asarray(pair_1.assemble_bilinear(_penalty_bilinear, u_m, u_s1, params))
    K_pair_2 = np.asarray(pair_2.assemble_bilinear(_penalty_bilinear, u_m, u_s2, params))
    K_otm = np.asarray(otm.assemble_bilinear(_penalty_bilinear, u_m, [u_s1, u_s2], params))

    K_ref = np.zeros((n_m + n_s + n_s, n_m + n_s + n_s), dtype=float)
    K_ref[:n_m, :n_m] += K_pair_1[:n_m, :n_m]
    K_ref[:n_m, n_m : n_m + n_s] += K_pair_1[:n_m, n_m:]
    K_ref[n_m : n_m + n_s, :n_m] += K_pair_1[n_m:, :n_m]
    K_ref[n_m : n_m + n_s, n_m : n_m + n_s] += K_pair_1[n_m:, n_m:]

    off2 = n_m + n_s
    K_ref[:n_m, :n_m] += K_pair_2[:n_m, :n_m]
    K_ref[:n_m, off2 : off2 + n_s] += K_pair_2[:n_m, n_m:]
    K_ref[off2 : off2 + n_s, :n_m] += K_pair_2[n_m:, :n_m]
    K_ref[off2 : off2 + n_s, off2 : off2 + n_s] += K_pair_2[n_m:, n_m:]

    assert K_otm.shape == K_ref.shape
    assert np.allclose(K_otm, K_ref, atol=1e-10)

    K_sparse = otm.assemble_bilinear(
        _penalty_bilinear,
        u_m,
        [u_s1, u_s2],
        params,
        sparse=True,
    )
    assert np.allclose(np.asarray(K_sparse.to_dense()), K_ref, atol=1e-10)


def test_one_to_many_from_meshes_with_selectors():
    mesh_m = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=1).build()
    mesh_s1 = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=1).build()
    mesh_s2 = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=1).build()

    def select_contact(m):
        return m.facets_on_plane(axis=2, value=0.0)

    otm = ff.OneToManyContactSurfaceSpace.from_meshes(
        master_mesh=mesh_m,
        slave_meshes=[mesh_s1, mesh_s2],
        master_facet_selector=select_contact,
        slave_facet_selectors=select_contact,
        value_dim_master=3,
        value_dim_slaves=3,
        quad_order=1,
        backend="jax",
    )

    n = mesh_m.coords.shape[0] * 3
    u = jnp.zeros(n)
    K = np.asarray(otm.assemble_bilinear(_penalty_bilinear, u, [u, u], _params()))
    assert K.shape == (n + n + n, n + n + n)

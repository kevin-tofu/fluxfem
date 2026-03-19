"""Tet mesh diffusion and scikit-fem comparison checks."""
import numpy as np
import pytest

import fluxfem as ff


def test_tet_mesh_shapes():
    mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    assert mesh.coords.shape == (8, 3)  # corners only
    assert mesh.conn.shape == (5, 4)    # 5 tets per cube


def test_tet_diffusion_small():
    mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_tet_space(mesh, dim=1, intorder=2)
    K = np.asarray(space.assemble(ff.diffusion_form, params=1.0).to_dense())
    assert K.shape == (8, 8)
    assert np.all(np.isfinite(K))


def test_tet_against_scikit_fem():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshTet, ElementTetP1, Basis, asm
    from skfem.helpers import dot, grad

    kappa = 1.0
    n_xyz = 2  # 3x3x3 = 27 nodes

    # fluxfem mesh
    mesh_ff = ff.StructuredTetBox(nx=n_xyz, ny=n_xyz, nz=n_xyz, lx=1.0, ly=1.0, lz=1.0).build()
    space_ff = ff.make_tet_space(mesh_ff, dim=1, intorder=2)
    K_ff = np.asarray(space_ff.assemble(ff.diffusion_form, params=kappa).to_dense())

    # Build scikit-fem mesh with the same 5-tet subdivision
    xs = np.linspace(0.0, 1.0, n_xyz + 1)
    ys = np.linspace(0.0, 1.0, n_xyz + 1)
    zs = np.linspace(0.0, 1.0, n_xyz + 1)
    coords_sf = []
    for k in range(n_xyz + 1):
        for j in range(n_xyz + 1):
            for i in range(n_xyz + 1):
                coords_sf.append([xs[i], ys[j], zs[k]])
    coords_sf = np.array(coords_sf).T  # (3, n_nodes)

    def node_id(i, j, k):
        return k * (n_xyz + 1) * (n_xyz + 1) + j * (n_xyz + 1) + i

    tris = []
    for k in range(n_xyz):
        for j in range(n_xyz):
            for i in range(n_xyz):
                v000 = node_id(i, j, k)
                v100 = node_id(i + 1, j, k)
                v010 = node_id(i, j + 1, k)
                v110 = node_id(i + 1, j + 1, k)
                v001 = node_id(i, j, k + 1)
                v101 = node_id(i + 1, j, k + 1)
                v011 = node_id(i, j + 1, k + 1)
                v111 = node_id(i + 1, j + 1, k + 1)
                tris.extend(
                    [
                        [v000, v100, v010, v001],
                        [v100, v110, v010, v111],
                        [v100, v010, v001, v111],
                        [v100, v001, v101, v111],
                        [v010, v001, v011, v111],
                    ]
                )
    tris = np.array(tris).T  # (4, n_elems)
    mesh_sf = MeshTet(coords_sf, tris)
    basis_sf = Basis(mesh_sf, ElementTetP1(), intorder=2)

    @skfem.BilinearForm
    def diff(u, v, w):
        return kappa * dot(grad(u), grad(v))

    K_sf = asm(diff, basis_sf).toarray()

    # Nodes are already aligned (same construction)
    max_diff = float(np.max(np.abs(K_ff - K_sf)))
    assert max_diff < 1e-6, f"K mismatch vs scikit-fem tet: {max_diff}"


def test_tet_tensor_matches_skfem_init_tensor():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshTet

    n_xyz = 2
    mesh_ff = ff.StructuredTetTensorBox(nx=n_xyz, ny=n_xyz, nz=n_xyz, lx=1.0, ly=1.0, lz=1.0).build()
    nodes = np.linspace(0.0, 1.0, n_xyz + 1)
    mesh_sf = MeshTet.init_tensor(*(3 * (nodes,)))

    coords_ff = np.asarray(mesh_ff.coords)
    coords_sf = np.asarray(mesh_sf.p).T
    tris_ff = np.asarray(mesh_ff.conn)
    tris_sf = np.asarray(mesh_sf.t).T

    assert coords_ff.shape == coords_sf.shape
    assert tris_ff.shape == tris_sf.shape
    assert np.allclose(coords_ff, coords_sf)
    def _sorted_tris(tris):
        tris_sorted = np.sort(tris, axis=1)
        order = np.lexsort(tris_sorted.T)
        return tris_sorted[order]

    assert np.array_equal(_sorted_tris(tris_ff), _sorted_tris(tris_sf))

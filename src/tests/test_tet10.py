"""Tet10 shape and diffusion assembly checks."""
import numpy as np
import pytest

import fluxfem as ff


def test_tet10_shapes():
    mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=2).build()
    # 1 cube subdivided into 5 tets; corner + edge mids on each edge (shared across tets)
    # For 1 element cube: 8 corners + 18 unique edge midpoints = 26 nodes
    assert mesh.coords.shape[0] == 26
    assert mesh.conn.shape[1] == 10


def test_tet10_diffusion_sum_energy():
    mesh = ff.StructuredTetBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=2).build()
    space = ff.make_tet10_space(mesh, dim=1, intorder=2)
    K = np.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense())
    assert K.shape[0] == K.shape[1] == mesh.conn.max() + 1
    assert np.all(np.isfinite(K))
    # compliance-like sanity check
    assert abs(K.sum()) > 0


@pytest.mark.xfail(reason="Mapping to scikit-fem ElementTetP2 not implemented")
def test_tet10_against_scikit_fem():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshTet, ElementTetP2, Basis, asm
    from skfem.helpers import dot, grad

    n_xyz = 1
    kappa = 1.0

    mesh_ff = ff.StructuredTetBox(nx=n_xyz, ny=n_xyz, nz=n_xyz, lx=1.0, ly=1.0, lz=1.0, order=2).build()
    space_ff = ff.make_tet10_space(mesh_ff, dim=1, intorder=2)
    K_ff = np.asarray(space_ff.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())

    xs = np.linspace(0.0, 1.0, n_xyz + 1)
    ys = np.linspace(0.0, 1.0, n_xyz + 1)
    zs = np.linspace(0.0, 1.0, n_xyz + 1)
    mesh_sf = MeshTet().init_tensor(xs, ys, zs)
    basis_sf = Basis(mesh_sf, ElementTetP2(), intorder=2)

    @skfem.BilinearForm
    def diff(u, v, w):
        return kappa * dot(grad(u), grad(v))

    K_sf = asm(diff, basis_sf).toarray()

    # Loose comparison: total energy (sum of entries) only
    rel = abs(K_ff.sum() - K_sf.sum()) / max(1.0, abs(K_sf.sum()))
    assert rel < 0.2

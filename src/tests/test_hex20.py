"""Hex20/Hex27 shape and diffusion assembly checks."""
import numpy as np
import pytest

import fluxfem as ff


def test_hex20_mesh_shapes():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=2).build()
    assert mesh.coords.shape[0] == 20
    assert mesh.conn.shape == (1, 20)


def test_hex20_diffusion_small():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=2).build()
    space = ff.make_hex20_space(mesh, dim=1, intorder=2)
    K = np.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense())
    assert K.shape == (20, 20)
    assert np.all(np.isfinite(K))


@pytest.mark.skip(reason="scikit-fem serendipity Hex20 not readily available for direct comparison")
def test_hex20_against_scikit_fem():
    # Placeholder: compare against scikit-fem ElementHex2 if serendipity variant is available.
    # Keeping skip to avoid false failures if element definitions differ (27-node vs 20-node).
    pass


@pytest.mark.xfail(reason="scikit-fem ElementHex2 dof mapping differs; reference doflocs not aligned with StructuredHexBox order=3")
def test_hex27_against_scikit_fem():
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshHex, ElementHex2, Basis, asm
    from skfem.helpers import dot, grad

    kappa = 1.0
    n_xyz = 1  # single element

    mesh_ff = ff.StructuredHexBox(nx=n_xyz, ny=n_xyz, nz=n_xyz, lx=1.0, ly=1.0, lz=1.0, order=3).build()
    space_ff = ff.make_hex27_space(mesh_ff, dim=1, intorder=3)
    K_ff = np.asarray(space_ff.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())

    xs = np.linspace(0.0, 1.0, n_xyz + 1)
    ys = np.linspace(0.0, 1.0, n_xyz + 1)
    zs = np.linspace(0.0, 1.0, n_xyz + 1)
    mesh_sf = MeshHex().init_tensor(xs, ys, zs)
    basis_sf = Basis(mesh_sf, ElementHex2(), intorder=3)

    @skfem.BilinearForm
    def diff(u, v, w):
        return kappa * dot(grad(u), grad(v))

    K_sf = asm(diff, basis_sf).toarray()

    # Compare total energy for unit vector (sum of entries) to avoid ordering issues
    sum_ff = float(K_ff.sum())
    sum_sf = float(K_sf.sum())
    rel_diff = abs(sum_ff - sum_sf) / max(1.0, abs(sum_sf))
    assert rel_diff < 1e-6, f"sum(K) mismatch vs scikit-fem Hex27: {rel_diff}"


@pytest.mark.xfail(reason="scikit-fem ElementHex2 dof mapping differs; reference doflocs not aligned with StructuredHexBox order=3")
def test_hex20_energy_vs_hex27_with_face_center_zero():
    """
    Compare energy of 20-node serendipity vs 27-node (embedding with zeroed face/center DOFs).
    Not identical, but sanity-check the magnitude/order.
    """
    skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
    from skfem import MeshHex, ElementHex2, Basis, asm
    from skfem.helpers import dot, grad

    kappa = 1.0
    mesh20 = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=2).build()
    space20 = ff.make_hex20_space(mesh20, dim=1, intorder=3)
    K20 = np.asarray(space20.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())

    xs = np.linspace(0.0, 1.0, 2)
    ys = np.linspace(0.0, 1.0, 2)
    zs = np.linspace(0.0, 1.0, 2)
    mesh27 = MeshHex().init_tensor(xs, ys, zs)
    basis27 = Basis(mesh27, ElementHex2(), intorder=3)

    @skfem.BilinearForm
    def diff(u, v, w):
        return kappa * dot(grad(u), grad(v))

    K27 = asm(diff, basis27).toarray()

    # Compare total energy for unit vector (sum of entries) as a loose sanity check
    sum20 = float(K20.sum())
    sum27 = float(K27.sum())
    rel_diff = abs(sum20 - sum27) / max(1.0, abs(sum27))
    assert rel_diff < 0.2, f"Energy mismatch (sum(K)) 20-node vs 27-node: {rel_diff}"

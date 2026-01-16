"""Hex20/Hex27 shape and diffusion assembly checks."""
import numpy as np
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


def test_hex8_hex20_linear_energy_matches():
    kappa = 1.0
    mesh8 = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=1).build()
    mesh20 = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=2).build()
    space8 = ff.make_hex_space(mesh8, dim=1, intorder=3)
    space20 = ff.make_hex20_space(mesh20, dim=1, intorder=3)

    K8 = np.asarray(space8.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())
    K20 = np.asarray(space20.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())

    def linear_field(coords: np.ndarray) -> np.ndarray:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        return 1.3 * x - 0.7 * y + 2.1 * z

    u8 = linear_field(np.asarray(mesh8.coords))
    u20 = linear_field(np.asarray(mesh20.coords))

    e8 = float(u8 @ K8 @ u8)
    e20 = float(u20 @ K20 @ u20)
    rel_diff = abs(e8 - e20) / max(1.0, abs(e8), abs(e20))
    assert rel_diff < 1e-6, f"Linear-field energy mismatch hex8 vs hex20: {rel_diff}"

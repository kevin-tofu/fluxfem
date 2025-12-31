"""Compare fluxfem vs scikit-fem assembly for hex/tet, P1/P2."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fluxfem as ff

try:  # ensure stable numeric comparisons for P2 tolerances
    jax.config.update("jax_enable_x64", True)
except Exception:  # pragma: no cover - defensive for older JAX
    pass

skfem = pytest.importorskip("skfem", reason="scikit-fem not installed")
from skfem import Basis, MeshHex, MeshTet, asm  # type: ignore
from skfem.helpers import dot, grad  # type: ignore
try:
    from skfem import ElementHex1, ElementHex2, ElementTetP1, ElementTetP2  # type: ignore
except Exception:  # pragma: no cover - version compatibility
    from skfem.element import (  # type: ignore
        ElementHex1,
        ElementHex2,
        ElementTetP1,
        ElementTetP2,
    )


@dataclass(frozen=True)
class Case:
    element: str
    order: int
    intorder: int


CASES = [
    pytest.param(Case("hex", 1, 2), id="hex-p1"),
    pytest.param(Case("hex", 2, 3), id="hex-p2"),
    pytest.param(Case("tet", 1, 2), id="tet-p1"),
    pytest.param(Case("tet", 2, 3), id="tet-p2"),
]


def _perm_by_coords(coords_ff: np.ndarray, doflocs_sf: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    coords_ff = np.asarray(coords_ff)
    doflocs_sf = np.asarray(doflocs_sf)
    if doflocs_sf.shape[0] == 3 and doflocs_sf.shape[1] != 3:
        doflocs_sf = doflocs_sf.T
    perm = np.empty(coords_ff.shape[0], dtype=int)
    for i, c in enumerate(coords_ff):
        matches = np.nonzero(np.all(np.isclose(doflocs_sf, c, atol=atol), axis=1))[0]
        assert len(matches) == 1, "dof mapping ambiguous"
        perm[i] = matches[0]
    return perm


def _build_case(case: Case):
    n_xyz = 1
    xs = np.linspace(0.0, 1.0, n_xyz + 1)
    ys = np.linspace(0.0, 1.0, n_xyz + 1)
    zs = np.linspace(0.0, 1.0, n_xyz + 1)

    if case.element == "hex":
        if case.order == 1:
            mesh_ff = ff.StructuredHexBox(nx=n_xyz, ny=n_xyz, nz=n_xyz).build()
            space_ff = ff.make_hex_space(mesh_ff, dim=1, intorder=case.intorder)
            mesh_sf = MeshHex().init_tensor(xs, ys, zs)
            basis_sf = Basis(mesh_sf, ElementHex1(), intorder=case.intorder)
        else:
            mesh_ff = ff.StructuredHexBox(nx=n_xyz, ny=n_xyz, nz=n_xyz, order=3).build()
            space_ff = ff.make_hex27_space(mesh_ff, dim=1, intorder=case.intorder)
            mesh_sf = MeshHex().init_tensor(xs, ys, zs)
            basis_sf = Basis(mesh_sf, ElementHex2(), intorder=case.intorder)
        perm = _perm_by_coords(np.asarray(mesh_ff.coords), np.asarray(basis_sf.doflocs))
        return space_ff, basis_sf, perm

    if case.element == "tet":
        if case.order == 1:
            mesh_ff = ff.StructuredTetBox(nx=n_xyz, ny=n_xyz, nz=n_xyz).build()
            space_ff = ff.make_tet_space(mesh_ff, dim=1, intorder=case.intorder)
            mesh_sf = MeshTet(np.asarray(mesh_ff.coords).T, np.asarray(mesh_ff.conn).T)
            basis_sf = Basis(mesh_sf, ElementTetP1(), intorder=case.intorder)
        else:
            mesh_ff = ff.StructuredTetBox(nx=n_xyz, ny=n_xyz, nz=n_xyz, order=2).build()
            space_ff = ff.make_tet10_space(mesh_ff, dim=1, intorder=case.intorder)
            mesh_sf_lin = ff.StructuredTetBox(nx=n_xyz, ny=n_xyz, nz=n_xyz).build()
            mesh_sf = MeshTet(np.asarray(mesh_sf_lin.coords).T, np.asarray(mesh_sf_lin.conn).T)
            basis_sf = Basis(mesh_sf, ElementTetP2(), intorder=case.intorder)
        perm = _perm_by_coords(np.asarray(mesh_ff.coords), np.asarray(basis_sf.doflocs))
        return space_ff, basis_sf, perm

    raise ValueError(f"Unknown element: {case.element}")


@pytest.mark.parametrize("case", CASES)
def test_bilinear_form_matches_skfem(case: Case):
    space_ff, basis_sf, perm = _build_case(case)
    kappa = 1.3

    K_ff = np.asarray(space_ff.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense())

    @skfem.BilinearForm
    def diff(u, v, w):
        return kappa * dot(grad(u), grad(v))

    K_sf = asm(diff, basis_sf).toarray()
    K_sf = K_sf[np.ix_(perm, perm)]

    tol = 1e-6 if case.order == 1 else 1e-6
    max_diff = float(np.max(np.abs(K_ff - K_sf)))
    assert max_diff < tol, f"K mismatch vs scikit-fem ({case.element} P{case.order}): {max_diff}"


@pytest.mark.parametrize("case", CASES)
def test_linear_form_matches_skfem(case: Case):
    space_ff, basis_sf, perm = _build_case(case)
    load = 2.0

    F_ff = np.asarray(space_ff.assemble_linear_form(ff.scalar_body_force_form, params=load))

    @skfem.LinearForm
    def body_force(v, w):
        return load * v

    F_sf = np.asarray(asm(body_force, basis_sf))
    F_sf = F_sf[perm]

    tol = 1e-6 if case.order == 1 else 1e-6
    max_diff = float(np.max(np.abs(F_ff - F_sf)))
    assert max_diff < tol, f"F mismatch vs scikit-fem ({case.element} P{case.order}): {max_diff}"


@pytest.mark.parametrize("case", CASES)
def test_functional_matches_skfem(case: Case):
    space_ff, basis_sf, _ = _build_case(case)

    def one(ctx, params):
        return jnp.ones_like(ctx.w)

    J_ff = float(space_ff.assemble_functional(one, params=None))

    @skfem.Functional
    def volume(w):
        return 1.0

    J_sf = float(asm(volume, basis_sf))
    tol = 1e-8 if case.order == 1 else 1e-7
    assert abs(J_ff - J_sf) < tol, f"J mismatch vs scikit-fem ({case.element} P{case.order}): {J_ff - J_sf}"

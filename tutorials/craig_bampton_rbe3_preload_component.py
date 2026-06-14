#!/usr/bin/env python
"""FluxFEM Craig-Bampton ROM for an explicit RBE3 preload fixture component.

This is the compact FluxFEM counterpart of
``skfem-Craig-Bampton-ROM/experiment-2``.  A notched tetrahedral workpiece is
loaded by explicit translational RBE3 reference fixtures:

    u_ref - weighted_average(u_patch) = 0

The full model solves the augmented workpiece/reference-point KKT system.  The
ROM solves the same KKT system after projecting only the workpiece with a
Craig-Bampton basis; reference-point DOFs and RBE3 constraints remain explicit.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton_rbe3_preload_component.py
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np
import scipy.sparse as sp

import fluxfem as ff


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]


@dataclass(frozen=True)
class Fixture:
    name: str
    nodes: np.ndarray
    weights: np.ndarray
    point: np.ndarray
    direction: np.ndarray
    stiffness: float
    target: float


@dataclass(frozen=True)
class Model:
    mesh: ff.TetMesh
    space: object
    stiffness: object
    coords: np.ndarray

    @property
    def n_dofs(self) -> int:
        return int(self.space.n_dofs)


def node_dofs(nodes: np.ndarray, *, dim: int = 3) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=int).reshape(-1)
    return (nodes[:, None] * dim + np.arange(dim, dtype=int)[None, :]).reshape(-1)


def nodes_on(coords: np.ndarray, *, x=None, y=None, z=None, tol: float = 1.0e-10) -> np.ndarray:
    mask = np.ones((coords.shape[0],), dtype=bool)
    if x is not None:
        mask &= np.isclose(coords[:, 0], float(x), atol=tol)
    if y is not None:
        mask &= np.isclose(coords[:, 1], float(y), atol=tol)
    if z is not None:
        mask &= np.isclose(coords[:, 2], float(z), atol=tol)
    return np.flatnonzero(mask).astype(int)


def nodes_in_box(coords: np.ndarray, nodes: np.ndarray, x_range, y_range, *, tol: float = 1.0e-10) -> np.ndarray:
    pts = coords[np.asarray(nodes, dtype=int)]
    keep = (
        (pts[:, 0] >= x_range[0] - tol)
        & (pts[:, 0] <= x_range[1] + tol)
        & (pts[:, 1] >= y_range[0] - tol)
        & (pts[:, 1] <= y_range[1] + tol)
    )
    return np.asarray(nodes, dtype=int)[keep]


def sorted_xy(coords: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    pts = coords[np.asarray(nodes, dtype=int)]
    return np.asarray(nodes, dtype=int)[np.lexsort((pts[:, 1], pts[:, 0]))]


def notched_tet_mesh(nx: int, ny: int, nz: int) -> ff.TetMesh:
    box = ff.StructuredTetTensorBox(nx=nx, ny=ny, nz=nz, lx=4.0, ly=3.0, lz=0.25).build()
    coords = np.asarray(box.coords, dtype=float)
    conn = np.asarray(box.conn, dtype=int)
    centroids = coords[conn].mean(axis=1)
    cutouts = (
        ((0.875, 1.375), (0.0, 0.25)),
        ((2.625, 3.125), (2.5, 3.0)),
        ((1.25, 1.625), (1.25, 1.75)),
        ((2.375, 2.75), (1.25, 1.75)),
    )
    keep = np.ones((conn.shape[0],), dtype=bool)
    for xr, yr in cutouts:
        keep &= ~(
            (centroids[:, 0] >= xr[0])
            & (centroids[:, 0] <= xr[1])
            & (centroids[:, 1] >= yr[0])
            & (centroids[:, 1] <= yr[1])
        )
    conn = conn[keep]
    used = np.unique(conn)
    remap = -np.ones((coords.shape[0],), dtype=int)
    remap[used] = np.arange(used.size, dtype=int)
    return ff.TetMesh(coords=coords[used], conn=remap[conn])


def fixture_specs() -> list[FixtureSpec]:
    return [
        FixtureSpec("bottom_left", (0.25, 0.75), (0.00, 0.50)),
        FixtureSpec("bottom_right", (3.25, 3.75), (0.00, 0.50)),
        FixtureSpec("top_left", (0.25, 0.75), (2.50, 3.00)),
        FixtureSpec("top_right", (3.25, 3.75), (2.50, 3.00)),
        FixtureSpec("left_mid", (0.00, 0.50), (1.25, 1.75)),
        FixtureSpec("right_mid", (3.50, 4.00), (1.25, 1.75)),
    ]


def build_model(nx: int, ny: int, nz: int) -> Model:
    mesh = notched_tet_mesh(nx, ny, nz)
    space = ff.make_tet_space(mesh, dim=3, intorder=2)
    stiffness = space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(1.0, 0.30))
    return Model(mesh=mesh, space=space, stiffness=stiffness, coords=np.asarray(mesh.coords, dtype=float))


def make_fixture(model: Model, spec: FixtureSpec, *, stiffness: float, target: float) -> Fixture:
    top = nodes_on(model.coords, z=0.25)
    patch_nodes = sorted_xy(model.coords, nodes_in_box(model.coords, top, spec.x_range, spec.y_range))
    if patch_nodes.size < 2:
        raise ValueError(f"{spec.name}: fixture patch has only {patch_nodes.size} top nodes.")
    weights = np.ones((patch_nodes.size,), dtype=float) / float(patch_nodes.size)
    point = np.average(model.coords[patch_nodes], axis=0, weights=weights)
    return Fixture(
        name=spec.name,
        nodes=patch_nodes,
        weights=weights,
        point=point,
        direction=np.array([0.0, 0.0, 1.0], dtype=float),
        stiffness=float(stiffness),
        target=float(target),
    )


def fixed_321_dofs(model: Model) -> np.ndarray:
    c = model.coords
    a = nodes_on(c, x=0.0, y=0.0, z=0.0)[0]
    b = nodes_on(c, x=4.0, y=0.0, z=0.0)[0]
    d = nodes_on(c, x=0.0, y=3.0, z=0.0)[0]
    return np.array([3 * a, 3 * a + 1, 3 * a + 2, 3 * b + 1, 3 * b + 2, 3 * d + 2], dtype=int)


def external_force(model: Model) -> np.ndarray:
    bottom = nodes_on(model.coords, z=0.0)
    loaded = sorted_xy(model.coords, nodes_in_box(model.coords, bottom, (1.5, 2.5), (1.0, 2.0)))
    f = np.zeros((model.n_dofs,), dtype=float)
    for node in loaded:
        f[3 * int(node) + 2] += -1.0 / float(loaded.size)
    return f


def fixture_constraint(fixture: Fixture, *, n_workpiece: int, reference_dofs: np.ndarray, total_dofs: int) -> np.ndarray:
    patch = ff.RBE3Patch(dofs=node_dofs(fixture.nodes).reshape(-1, 3), weights=fixture.weights)
    wrapper = ff.ReferencePointFixture(
        fixture.name,
        patch,
        reference_dofs=reference_dofs,
        direction=fixture.direction,
        stiffness=fixture.stiffness,
        target_displacement=fixture.target,
    )
    return np.asarray(wrapper.constraint_matrix(n_structural_dofs=n_workpiece, total_dofs=total_dofs), dtype=float)


def preload_terms(fixtures: list[Fixture], *, n_workpiece: int, active: set[str], total_dofs: int):
    k = np.zeros((total_dofs, total_dofs), dtype=float)
    f = np.zeros((total_dofs,), dtype=float)
    wrappers = []
    for i, fixture in enumerate(fixtures):
        refs = n_workpiece + 3 * i + np.arange(3, dtype=int)
        patch = ff.RBE3Patch(dofs=node_dofs(fixture.nodes).reshape(-1, 3), weights=fixture.weights)
        wrappers.append(
            ff.ReferencePointFixture(
                fixture.name,
                patch,
                refs,
                direction=fixture.direction,
                stiffness=fixture.stiffness,
                target_displacement=fixture.target,
            )
        )
    kk, ff_vec = ff.assemble_reference_fixture_preload(wrappers, total_dofs=total_dofs, active_names=active)
    k += np.asarray(kk, dtype=float)
    f += np.asarray(ff_vec, dtype=float)
    return k, f


def solve_full(model: Model, fixtures: list[Fixture], active: set[str], fixed_dofs: np.ndarray) -> np.ndarray:
    n_ref = 3 * len(fixtures)
    total = model.n_dofs + n_ref
    k = sp.block_diag((model.stiffness.to_csr(), sp.csr_matrix((n_ref, n_ref))), format="csr")
    k_pre, f = preload_terms(fixtures, n_workpiece=model.n_dofs, active=active, total_dofs=total)
    f[: model.n_dofs] += external_force(model)
    c = np.vstack(
        [
            fixture_constraint(
                fixture,
                n_workpiece=model.n_dofs,
                reference_dofs=model.n_dofs + 3 * i + np.arange(3, dtype=int),
                total_dofs=total,
            )
            for i, fixture in enumerate(fixtures)
        ]
    )
    return np.asarray(
        ff.LinearConstraintSystem(c).solve(k + sp.csr_matrix(k_pre), f, fixed_dofs=fixed_dofs, solver="spsolve")
    )


def build_cb(model: Model, fixtures: list[Fixture], fixed_dofs: np.ndarray, n_modes: int) -> ff.CraigBamptonBasis:
    force_dofs = np.flatnonzero(np.abs(external_force(model)) > 0.0)
    fixture_dofs = np.concatenate([node_dofs(f.nodes) for f in fixtures])
    retained = np.unique(np.concatenate([fixed_dofs, force_dofs, fixture_dofs])).astype(int)
    identity_mass = sp.eye(model.n_dofs, format="csr")
    return ff.make_craig_bampton_basis(
        model.stiffness.to_csr(),
        identity_mass,
        retained_dofs=retained,
        n_modes=n_modes,
        constraint_solver="spsolve",
        modal_solver="eigsh",
        modal_tol=1.0e-8,
        modal_maxiter=500,
    )


def solve_rom(model: Model, cb: ff.CraigBamptonBasis, fixtures: list[Fixture], active: set[str], fixed_dofs: np.ndarray):
    n_ref = 3 * len(fixtures)
    n_rom = cb.n_reduced
    total = n_rom + n_ref
    k = np.zeros((total, total), dtype=float)
    k[:n_rom, :n_rom] = np.asarray(cb.project_matrix(model.stiffness.to_csr()), dtype=float)
    f = np.zeros((total,), dtype=float)
    f[:n_rom] = np.asarray(cb.project_vector(external_force(model)), dtype=float)

    full_total = model.n_dofs + n_ref
    k_pre, f_pre = preload_terms(fixtures, n_workpiece=model.n_dofs, active=active, total_dofs=full_total)
    for i in range(len(fixtures)):
        full_refs = model.n_dofs + 3 * i + np.arange(3, dtype=int)
        rom_refs = n_rom + 3 * i + np.arange(3, dtype=int)
        k[np.ix_(rom_refs, rom_refs)] += k_pre[np.ix_(full_refs, full_refs)]
        f[rom_refs] += f_pre[full_refs]

    constraints = []
    for i, fixture in enumerate(fixtures):
        c_full = fixture_constraint(
            fixture,
            n_workpiece=model.n_dofs,
            reference_dofs=model.n_dofs + 3 * i + np.arange(3, dtype=int),
            total_dofs=full_total,
        )
        constraints.append(np.asarray(ff.LinearConstraintSystem(c_full).project(cb, n_extra_dofs=n_ref).matrix))
    c_rom = np.vstack(constraints)

    master_to_rom = {int(dof): i for i, dof in enumerate(np.asarray(cb.retained_dofs))}
    fixed_rom = np.array([master_to_rom[int(dof)] for dof in fixed_dofs], dtype=int)
    q_aug = ff.LinearConstraintSystem(c_rom).solve(k, f, fixed_dofs=fixed_rom, solver="dense")
    return np.concatenate([np.asarray(cb.expand(q_aug[:n_rom])), np.asarray(q_aug[n_rom:])])


def compliance_like(fixtures: list[Fixture], active: set[str], model: Model, u_aug: np.ndarray) -> float:
    _, f_pre = preload_terms(
        fixtures,
        n_workpiece=model.n_dofs,
        active=active,
        total_dofs=model.n_dofs + 3 * len(fixtures),
    )
    return float(f_pre @ u_aug)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=9)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    model = build_model(args.nx, args.ny, args.nz)
    fixtures = [make_fixture(model, spec, stiffness=25.0, target=0.030) for spec in fixture_specs()]
    active = {fixture.name for fixture in fixtures}
    fixed = fixed_321_dofs(model)

    t0 = time.perf_counter()
    cb = build_cb(model, fixtures, fixed, args.modes)
    build_seconds = time.perf_counter() - t0

    def timed(fn):
        out = fn()
        start = time.perf_counter()
        for _ in range(args.timing_repeats):
            out = fn()
        return out, (time.perf_counter() - start) / max(args.timing_repeats, 1)

    full_u, full_seconds = timed(lambda: solve_full(model, fixtures, active, fixed))
    rom_u, rom_seconds = timed(lambda: solve_rom(model, cb, fixtures, active, fixed))
    full_wp = full_u[: model.n_dofs]
    rom_wp = rom_u[: model.n_dofs]
    rel_error = np.linalg.norm(full_wp - rom_wp) / max(np.linalg.norm(full_wp), 1.0e-15)
    max_error = float(np.max(np.abs(full_wp - rom_wp)))
    full_c = compliance_like(fixtures, active, model, full_u)
    rom_c = compliance_like(fixtures, active, model, rom_u)

    print("workpiece full dofs:       ", model.n_dofs)
    print("workpiece ROM dofs:        ", cb.n_reduced)
    print("retained workpiece dofs:   ", cb.n_retained)
    print("fixed-interface modes:     ", cb.n_modes)
    print("RBE3 reference points:     ", len(fixtures))
    print("CB build seconds:          ", f"{build_seconds:.6e}")
    print("full solve ms:             ", f"{1000.0 * full_seconds:.3f}")
    print("ROM solve ms:              ", f"{1000.0 * rom_seconds:.3f}")
    print("speedup full over ROM:     ", f"{full_seconds / max(rom_seconds, 1.0e-15):.2f}x")
    print("full ||u_workpiece||:      ", f"{np.linalg.norm(full_wp):.8e}")
    print("ROM  ||u_workpiece||:      ", f"{np.linalg.norm(rom_wp):.8e}")
    print("relative displacement err: ", f"{rel_error:.3e}")
    print("max abs displacement err:  ", f"{max_error:.3e}")
    print("full compliance-like work: ", f"{full_c:.8e}")
    print("ROM  compliance-like work: ", f"{rom_c:.8e}")
    print("relative compliance err:   ", f"{abs(full_c - rom_c) / max(abs(full_c), 1.0e-15):.3e}")


if __name__ == "__main__":
    main()

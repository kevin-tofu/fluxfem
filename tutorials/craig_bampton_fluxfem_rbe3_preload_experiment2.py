#!/usr/bin/env python
"""FluxFEM version of the experiment-2 explicit RBE3/preload CB-ROM sample.

This mirrors `skfem-Craig-Bampton-ROM/experiment-2` while using FluxFEM for:

* tetrahedral linear-elastic stiffness assembly,
* Craig-Bampton basis construction,
* RBE3-style reference-point fixture MPC construction,
* full and reduced KKT solves.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton_fluxfem_rbe3_preload_experiment2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff

DTYPE = jnp.float64
NP_DTYPE = np.float64


WORKPIECE_X_RANGE = (0.0, 4.0)
WORKPIECE_Y_RANGE = (0.0, 3.0)
WORKPIECE_Z_RANGE = (0.0, 0.25)
WORKPIECE_NX = 32
WORKPIECE_NY = 12
WORKPIECE_NZ = 2
FIXTURE_RANDOM_SEED = 20260614
MIN_FIXTURES_PER_EDGE = 2
MAX_FIXTURES_PER_EDGE = 5
FIXTURES_PER_EDGE = 5


@dataclass(frozen=True)
class OuterFixtureSpec:
    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    nx: int
    ny: int


@dataclass(frozen=True)
class ElasticPart:
    mesh: ff.TetMesh
    space: object
    stiffness: object
    nodal_dofs: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class WorkpieceRom:
    cb: ff.CraigBamptonBasis
    stiffness: jnp.ndarray
    master_dofs: np.ndarray
    internal_dofs: np.ndarray
    fixed_interface_eigenvalues: np.ndarray


def workpiece_notches() -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return (
        ((0.875, 1.375), (0.0, 0.25)),
        ((2.625, 3.125), (2.5, 3.0)),
    )


def workpiece_holes() -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return (
        ((1.25, 1.625), (1.25, 1.75)),
        ((2.375, 2.75), (1.25, 1.75)),
    )


def fixture_specs(prefix: str) -> list[OuterFixtureSpec]:
    def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return max(a[0], b[0]) < min(a[1], b[1])

    def count_grid_points(values: np.ndarray, r: tuple[float, float], tol: float = 1e-10) -> int:
        return int(np.count_nonzero((values >= r[0] - tol) & (values <= r[1] + tol)))

    def valid_patch(x_range: tuple[float, float], y_range: tuple[float, float]) -> bool:
        return count_grid_points(x_grid, x_range) >= 2 and count_grid_points(y_grid, y_range) >= 2

    patch = 0.25
    x_grid = np.linspace(WORKPIECE_X_RANGE[0], WORKPIECE_X_RANGE[1], WORKPIECE_NX + 1)
    y_grid = np.linspace(WORKPIECE_Y_RANGE[0], WORKPIECE_Y_RANGE[1], WORKPIECE_NY + 1)
    bottom_notch_x, _ = workpiece_notches()[0]
    top_notch_x, _ = workpiece_notches()[1]
    by_edge: dict[str, list[OuterFixtureSpec]] = {"bottom": [], "top": [], "left": [], "right": []}

    for i, x0 in enumerate(np.arange(WORKPIECE_X_RANGE[0], WORKPIECE_X_RANGE[1], patch)):
        x_range = (float(x0), float(x0 + patch))
        bottom_y_range = (WORKPIECE_Y_RANGE[0], WORKPIECE_Y_RANGE[0] + patch)
        if not overlaps(x_range, bottom_notch_x) and valid_patch(x_range, bottom_y_range):
            by_edge["bottom"].append(OuterFixtureSpec(f"{prefix}_bottom_{i:02d}", x_range, bottom_y_range, 2, 2))
        top_y_range = (WORKPIECE_Y_RANGE[1] - patch, WORKPIECE_Y_RANGE[1])
        if not overlaps(x_range, top_notch_x) and valid_patch(x_range, top_y_range):
            by_edge["top"].append(OuterFixtureSpec(f"{prefix}_top_{i:02d}", x_range, top_y_range, 2, 2))

    for j, y0 in enumerate(np.arange(WORKPIECE_Y_RANGE[0], WORKPIECE_Y_RANGE[1], patch)[1:-1], start=1):
        y_range = (float(y0), float(y0 + patch))
        left_x_range = (WORKPIECE_X_RANGE[0], WORKPIECE_X_RANGE[0] + patch)
        if valid_patch(left_x_range, y_range):
            by_edge["left"].append(OuterFixtureSpec(f"{prefix}_left_{j:02d}", left_x_range, y_range, 2, 2))
        right_x_range = (WORKPIECE_X_RANGE[1] - patch, WORKPIECE_X_RANGE[1])
        if valid_patch(right_x_range, y_range):
            by_edge["right"].append(OuterFixtureSpec(f"{prefix}_right_{j:02d}", right_x_range, y_range, 2, 2))

    rng = np.random.default_rng(FIXTURE_RANDOM_SEED)
    specs: list[OuterFixtureSpec] = []
    for edge in ("bottom", "top", "left", "right"):
        candidates = by_edge[edge]
        if len(candidates) < MIN_FIXTURES_PER_EDGE:
            raise ValueError(f"{edge}: only {len(candidates)} fixture candidates are available.")
        sample_size = int(rng.integers(MIN_FIXTURES_PER_EDGE, min(MAX_FIXTURES_PER_EDGE, len(candidates)) + 1))
        indices = np.sort(rng.choice(len(candidates), size=sample_size, replace=False))
        specs.extend(candidates[int(index)] for index in indices)
    return specs


def fixture_case_sets(fixture_names: list[str]) -> list[set[str]]:
    return [set(fixture_names)]


def make_notched_workpiece_mesh() -> ff.TetMesh:
    mesh = ff.StructuredTetTensorBox(
        nx=WORKPIECE_NX,
        ny=WORKPIECE_NY,
        nz=WORKPIECE_NZ,
        lx=WORKPIECE_X_RANGE[1] - WORKPIECE_X_RANGE[0],
        ly=WORKPIECE_Y_RANGE[1] - WORKPIECE_Y_RANGE[0],
        lz=WORKPIECE_Z_RANGE[1] - WORKPIECE_Z_RANGE[0],
    ).build()
    points = np.asarray(mesh.coords, dtype=float)
    tetrahedra = np.asarray(mesh.conn, dtype=np.int32)
    centroids = points[tetrahedra].mean(axis=1)
    keep = np.ones(tetrahedra.shape[0], dtype=bool)
    for x_range, y_range in (*workpiece_notches(), *workpiece_holes()):
        in_cutout = (
            (centroids[:, 0] >= x_range[0])
            & (centroids[:, 0] <= x_range[1])
            & (centroids[:, 1] >= y_range[0])
            & (centroids[:, 1] <= y_range[1])
        )
        keep &= ~in_cutout

    tetrahedra = tetrahedra[keep]
    used_nodes = np.unique(tetrahedra)
    old_to_new = -np.ones(points.shape[0], dtype=np.int32)
    old_to_new[used_nodes] = np.arange(used_nodes.size, dtype=np.int32)
    return ff.TetMesh(
        coords=jnp.asarray(points[used_nodes], dtype=DTYPE),
        conn=jnp.asarray(old_to_new[tetrahedra], dtype=jnp.int32),
    )


def assemble_workpiece() -> ElasticPart:
    mesh = make_notched_workpiece_mesh()
    space = ff.make_tet_space(mesh, dim=3, intorder=2)
    stiffness = space.assemble_bilinear_form(ff.linear_elasticity_form, params=ff.isotropic_3d_D(1.0, 0.30))
    n_nodes = int(np.asarray(mesh.coords).shape[0])
    nodal_dofs = np.arange(3 * n_nodes, dtype=np.int32).reshape(n_nodes, 3).T
    return ElasticPart(
        mesh=mesh,
        space=space,
        stiffness=stiffness,
        nodal_dofs=nodal_dofs,
        points=np.asarray(mesh.coords, dtype=float),
    )


def nodes_on(part: ElasticPart, *, x: float | None = None, y: float | None = None, z: float | None = None) -> np.ndarray:
    mask = np.ones(part.points.shape[0], dtype=bool)
    if x is not None:
        mask &= np.isclose(part.points[:, 0], x, atol=1e-10)
    if y is not None:
        mask &= np.isclose(part.points[:, 1], y, atol=1e-10)
    if z is not None:
        mask &= np.isclose(part.points[:, 2], z, atol=1e-10)
    return np.flatnonzero(mask).astype(np.int32)


def nodes_in_box(nodes: np.ndarray, part: ElasticPart, x_range: tuple[float, float], y_range: tuple[float, float]) -> np.ndarray:
    points = part.points[nodes]
    return nodes[
        (points[:, 0] >= x_range[0] - 1e-10)
        & (points[:, 0] <= x_range[1] + 1e-10)
        & (points[:, 1] >= y_range[0] - 1e-10)
        & (points[:, 1] <= y_range[1] + 1e-10)
    ]


def sorted_by_xy(nodes: np.ndarray, part: ElasticPart) -> np.ndarray:
    points = part.points[nodes]
    return nodes[np.lexsort((points[:, 1], points[:, 0]))]


def dofs_for_nodes(part: ElasticPart, nodes: np.ndarray) -> np.ndarray:
    return part.nodal_dofs[:, nodes].T.reshape(-1).astype(np.int32)


def spatial_321_constraint_dofs(part: ElasticPart) -> np.ndarray:
    left_front_bottom = nodes_on(part, x=0.0, y=0.0, z=0.0)[0]
    right_front_bottom = nodes_on(part, x=4.0, y=0.0, z=0.0)[0]
    left_back_bottom = nodes_on(part, x=0.0, y=1.0, z=0.0)[0]
    return np.asarray(
        [
            part.nodal_dofs[0, left_front_bottom],
            part.nodal_dofs[1, left_front_bottom],
            part.nodal_dofs[2, left_front_bottom],
            part.nodal_dofs[1, right_front_bottom],
            part.nodal_dofs[2, right_front_bottom],
            part.nodal_dofs[2, left_back_bottom],
        ],
        dtype=np.int32,
    )


def make_preload_fixture(
    name: str,
    part: ElasticPart,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    fixture_index: int,
    stiffness: float,
    target_displacement: float,
) -> ff.ReferencePointFixture:
    top_nodes = sorted_by_xy(nodes_in_box(nodes_on(part, z=0.25), part, x_range, y_range), part)
    if top_nodes.size < 2:
        raise ValueError(f"{name}: expected at least two workpiece surface nodes.")
    weights = np.ones(top_nodes.size, dtype=NP_DTYPE) / top_nodes.size
    patch_dofs = part.nodal_dofs[:, top_nodes].T.astype(np.int32)
    n_wp = int(part.space.n_dofs)
    ref = jnp.asarray([n_wp + 3 * fixture_index + i for i in range(3)], dtype=jnp.int32)
    return ff.ReferencePointFixture(
        name,
        ff.RBE3Patch(jnp.asarray(patch_dofs, dtype=jnp.int32), jnp.asarray(weights, dtype=DTYPE)),
        reference_dofs=ref,
        direction=jnp.asarray([0.0, 0.0, 1.0], dtype=DTYPE),
        stiffness=stiffness,
        target_displacement=target_displacement,
    )


def assemble_workpiece_external_force(part: ElasticPart) -> jnp.ndarray:
    f = np.zeros((part.space.n_dofs,), dtype=NP_DTYPE)
    surface_nodes = sorted_by_xy(
        nodes_in_box(nodes_on(part, z=0.0), part, x_range=(1.5, 2.5), y_range=(1.0, 2.0)),
        part,
    )
    if surface_nodes.size == 0:
        raise ValueError("external force patch has no z=0 surface nodes.")
    weights = np.ones(surface_nodes.size, dtype=float) / surface_nodes.size
    force = np.asarray([0.0, 0.0, -1.0], dtype=float)
    for node, weight in zip(surface_nodes, weights, strict=True):
        for component in range(3):
            f[int(part.nodal_dofs[component, node])] += weight * force[component]
    return jnp.asarray(f)


def build_cb_rom(part: ElasticPart, fixtures: list[ff.ReferencePointFixture], fixed_dofs: np.ndarray, *, internal_modes: int) -> WorkpieceRom:
    fixture_dofs = [np.asarray(fixture.retained_dofs, dtype=np.int32) for fixture in fixtures]
    external_force_nodes = sorted_by_xy(
        nodes_in_box(nodes_on(part, z=0.0), part, x_range=(1.5, 2.5), y_range=(1.0, 2.0)),
        part,
    )
    master_dofs = np.unique(
        np.concatenate([*fixture_dofs, fixed_dofs, dofs_for_nodes(part, external_force_nodes)])
    ).astype(np.int32)
    cb = ff.make_craig_bampton_basis(
        part.stiffness,
        jnp.eye(int(part.space.n_dofs), dtype=DTYPE),
        retained_dofs=jnp.asarray(master_dofs, dtype=jnp.int32),
        n_modes=internal_modes,
        constraint_solver="spsolve",
        modal_solver="eigsh",
        modal_tol=1e-8,
        modal_maxiter=500,
    )
    return WorkpieceRom(
        cb=cb,
        stiffness=cb.project_matrix(part.stiffness.to_dense()),
        master_dofs=master_dofs,
        internal_dofs=np.asarray(cb.internal_dofs, dtype=np.int32),
        fixed_interface_eigenvalues=np.asarray(cb.eigenvalues),
    )


def reduced_fixtures(fixtures: list[ff.ReferencePointFixture], n_rom: int) -> list[ff.ReferencePointFixture]:
    reduced: list[ff.ReferencePointFixture] = []
    for fixture_index, fixture in enumerate(fixtures):
        reduced.append(
            replace(
                fixture,
                reference_dofs=jnp.asarray([n_rom + 3 * fixture_index + i for i in range(3)], dtype=jnp.int32),
            )
        )
    return reduced


def solve_full_preload(
    part: ElasticPart,
    fixtures: list[ff.ReferencePointFixture],
    active_names: set[str],
    fixed_dofs: np.ndarray,
) -> jnp.ndarray:
    import scipy.sparse as sp

    n_wp = int(part.space.n_dofs)
    n_ref = 3 * len(fixtures)
    total = n_wp + n_ref
    k = sp.block_diag((part.stiffness.to_csr(), sp.csr_matrix((n_ref, n_ref))), format="csr")
    preload_k, preload_f = ff.assemble_reference_fixture_preload(
        fixtures,
        total_dofs=total,
        active_names=active_names,
        sparse=True,
    )
    f = jnp.zeros((total,), dtype=DTYPE).at[:n_wp].set(assemble_workpiece_external_force(part)) + preload_f
    constraints = ff.linear_constraint_system_from_reference_fixtures(
        fixtures,
        n_structural_dofs=n_wp,
        total_dofs=total,
    )
    return constraints.solve(k + preload_k, f, fixed_dofs=jnp.asarray(fixed_dofs, dtype=jnp.int32), solver="spsolve")


def solve_rom_preload(
    part: ElasticPart,
    rom: WorkpieceRom,
    fixtures: list[ff.ReferencePointFixture],
    active_names: set[str],
    fixed_dofs: np.ndarray,
) -> jnp.ndarray:
    import scipy.sparse as sp

    n_wp = int(part.space.n_dofs)
    n_rom = int(rom.stiffness.shape[0])
    n_ref = 3 * len(fixtures)
    total_full = n_wp + n_ref
    total_rom = n_rom + n_ref
    constraints_full = ff.linear_constraint_system_from_reference_fixtures(
        fixtures,
        n_structural_dofs=n_wp,
        total_dofs=total_full,
    )
    constraints_rom = constraints_full.project(rom.cb, n_extra_dofs=n_ref)
    rfixtures = reduced_fixtures(fixtures, n_rom)
    preload_k, preload_f = ff.assemble_reference_fixture_preload(
        rfixtures,
        total_dofs=total_rom,
        active_names=active_names,
        sparse=True,
    )
    k = sp.block_diag((np.asarray(rom.stiffness), np.zeros((n_ref, n_ref), dtype=NP_DTYPE)), format="csr") + preload_k
    f = jnp.zeros((total_rom,), dtype=DTYPE)
    f = f.at[:n_rom].set(rom.cb.project_vector(assemble_workpiece_external_force(part)))
    f = f + preload_f
    master_to_reduced = {int(dof): i for i, dof in enumerate(rom.master_dofs)}
    constrained = np.asarray([master_to_reduced[int(dof)] for dof in fixed_dofs], dtype=np.int32)
    q = constraints_rom.solve(k, f, fixed_dofs=jnp.asarray(constrained, dtype=jnp.int32), solver="spsolve")
    return constraints_rom.expand(q)


def workpiece_displacement(part: ElasticPart, augmented_u: jnp.ndarray) -> jnp.ndarray:
    return augmented_u[: int(part.space.n_dofs)]


def time_solve(fn, *, repeats: int) -> tuple[object, float]:
    result = fn()
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    return result, (time.perf_counter() - start) / repeats


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def active_label(_active: set[str]) -> str:
    return "active_random_fixtures"


def evaluate_fluxfem_rbe3_preload_sample(
    *,
    output_dir: str | Path = "results/fluxfem_rbe3_preload_component_sample",
    timing_repeats: int = 3,
    internal_modes: int = 12,
) -> dict[str, Path | list[dict[str, float | int | str]]]:
    output_dir = Path(output_dir)
    part = assemble_workpiece()
    fixed_dofs = spatial_321_constraint_dofs(part)
    fixtures = [
        make_preload_fixture(
            spec.name,
            part,
            x_range=spec.x_range,
            y_range=spec.y_range,
            fixture_index=i,
            stiffness=25.0,
            target_displacement=0.030,
        )
        for i, spec in enumerate(fixture_specs("preload"))
    ]
    build_start = time.perf_counter()
    rom = build_cb_rom(part, fixtures, fixed_dofs, internal_modes=internal_modes)
    rom_build_seconds = time.perf_counter() - build_start

    rows: list[dict[str, float | int | str]] = []
    for active in fixture_case_sets([fixture.name for fixture in fixtures]):
        full_u, full_seconds = time_solve(
            lambda active=active: solve_full_preload(part, fixtures, active, fixed_dofs),
            repeats=timing_repeats,
        )
        rom_u, rom_seconds = time_solve(
            lambda active=active: solve_rom_preload(part, rom, fixtures, active, fixed_dofs),
            repeats=timing_repeats,
        )
        full_wp_u = workpiece_displacement(part, full_u)
        rom_wp_u = workpiece_displacement(part, rom_u)
        rel_displacement_error = jnp.linalg.norm(full_wp_u - rom_wp_u) / jnp.maximum(jnp.linalg.norm(full_wp_u), 1e-15)
        max_abs_displacement_error = jnp.max(jnp.abs(full_wp_u - rom_wp_u))
        _, augmented_preload_force = ff.assemble_reference_fixture_preload(
            fixtures,
            total_dofs=int(part.space.n_dofs) + 3 * len(fixtures),
            active_names=active,
        )
        full_compliance = float(augmented_preload_force @ full_u)
        rom_compliance = float(augmented_preload_force @ rom_u)
        label = active_label(active)
        rows.append(
            {
                "active_fixtures": label,
                "n_active": len(active),
                "full_compliance": full_compliance,
                "rom_compliance": rom_compliance,
                "abs_compliance_error": abs(full_compliance - rom_compliance),
                "rel_compliance_error": abs(full_compliance - rom_compliance) / max(abs(full_compliance), 1e-15),
                "full_displacement_norm": float(jnp.linalg.norm(full_wp_u)),
                "rom_displacement_norm": float(jnp.linalg.norm(rom_wp_u)),
                "rel_displacement_error": float(rel_displacement_error),
                "max_abs_displacement_error": float(max_abs_displacement_error),
                "full_dofs": int(part.space.n_dofs),
                "rom_dofs": int(rom.stiffness.shape[0]),
                "full_seconds": float(full_seconds),
                "rom_seconds": float(rom_seconds),
                "speedup_full_over_rom": float(full_seconds / rom_seconds),
            }
        )

    csv_path = output_dir / "comparison.csv"
    json_path = output_dir / "comparison.json"
    write_csv(csv_path, rows)
    summary = {
        "sample": "fluxfem_explicit_reference_point_rbe3_preload_component",
        "timing_repeats": timing_repeats,
        "internal_modes": int(internal_modes),
        "workpiece_full_dofs": int(part.space.n_dofs),
        "workpiece_rom_dofs": int(rom.stiffness.shape[0]),
        "workpiece_master_dofs": int(rom.master_dofs.size),
        "workpiece_internal_dofs": int(rom.internal_dofs.size),
        "spatial_321_fixed_dofs": fixed_dofs.tolist(),
        "n_rbe3_reference_points": len(fixtures),
        "n_reduced_fixture_operators": len(fixtures),
        "rom_build_seconds": float(rom_build_seconds),
        "rows": rows,
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/fluxfem_rbe3_preload_component_sample")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--internal-modes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_fluxfem_rbe3_preload_sample(
        output_dir=args.output_dir,
        timing_repeats=args.timing_repeats,
        internal_modes=args.internal_modes,
    )
    with Path(result["json"]).open() as f:
        summary = json.load(f)
    print("workpiece full dofs:", summary["workpiece_full_dofs"])
    print("workpiece ROM dofs:", summary["workpiece_rom_dofs"])
    print("workpiece master dofs:", summary["workpiece_master_dofs"])
    print("workpiece internal dofs:", summary["workpiece_internal_dofs"])
    print("3D 3-2-1 fixed dofs:", summary["spatial_321_fixed_dofs"])
    print("precomputed reduced fixture operators:", summary["n_reduced_fixture_operators"])
    print("ROM build seconds:", f"{summary['rom_build_seconds']:.6e}")
    print()
    print("active preload fixtures | full ||u|| | rom ||u|| | rel disp error | full ms | rom ms | speedup")
    for row in result["rows"]:
        print(
            f"{str(row['active_fixtures']):48s} "
            f"{float(row['full_displacement_norm']): .8e} "
            f"{float(row['rom_displacement_norm']): .8e} "
            f"{float(row['rel_displacement_error']): .3e} "
            f"{1000.0 * float(row['full_seconds']): .3f} "
            f"{1000.0 * float(row['rom_seconds']): .3f} "
            f"{float(row['speedup_full_over_rom']): .2f}x"
        )
    print()
    print("The fixture preload springs were switched on explicit RBE3 reference point DOFs without rebuilding the workpiece CB-ROM.")
    print(f"wrote: {result['csv']}")
    print(f"wrote: {result['json']}")


if __name__ == "__main__":
    main()

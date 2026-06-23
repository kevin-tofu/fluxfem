"""Compare coarse P1 mortar on small fixture/workpiece interface solves.

This tutorial solves two linear KKT interface problems:

* matching fixture/workpiece surface meshes,
* nonmatching fixture/workpiece surface meshes with a supermesh.

The structural model is intentionally low-order along the interface, so the
coarse P1 multiplier should recover nearly the same response as the dual mortar
reference while using fewer multiplier DOFs.  Pass ``--all-variants`` to print
the lower-level comparison rows used for development diagnostics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import warnings

import numpy as np

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Running in float32 mode.*", category=RuntimeWarning)

import fluxfem as ff


def _matching_contact() -> ff.ContactSurfaceSpace:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords.copy(),
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )


def _nonmatching_contact() -> ff.ContactSurfaceSpace:
    fixture_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    fixture_facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    workpiece_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    workpiece_facets = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    return ff.ContactSurfaceSpace.from_facets(
        fixture_coords,
        fixture_facets,
        workpiece_coords,
        workpiece_facets,
        facet_dofs_master=fixture_facets,
        facet_dofs_slave=workpiece_facets,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )


def _low_order_stiffness(n_fixture: int, n_workpiece: int, *, gamma: float) -> np.ndarray:
    fixture_smoother = np.eye(n_fixture) - np.ones((n_fixture, n_fixture), dtype=float) / n_fixture
    workpiece_smoother = np.eye(n_workpiece) - np.ones((n_workpiece, n_workpiece), dtype=float) / n_workpiece
    return np.block(
        [
            [25.0 * np.eye(n_fixture) + gamma * fixture_smoother, np.zeros((n_fixture, n_workpiece), dtype=float)],
            [np.zeros((n_workpiece, n_fixture), dtype=float), np.eye(n_workpiece) + gamma * workpiece_smoother],
        ]
    )


def _multipliers(
    contact: ff.ContactSurfaceSpace,
    *,
    include_supermesh_p0: bool,
    all_variants: bool,
) -> dict[str, ff.MultiplierSpec]:
    fixture_nodes = int(contact.surface_master.n_nodes)
    core = {
        "dual": ff.MultiplierSpec.dual_mortar(),
        "coarse_p1": ff.MultiplierSpec.coarse_p1_mortar(
            basis=ff.coarse_p1_basis_from_node_groups(fixture_nodes, [list(range(fixture_nodes))])
        ),
    }
    if not all_variants:
        return core

    specs = {
        **core,
        "grid_p1": ff.MultiplierSpec.coarse_p1_mortar(
            basis=ff.coarse_p1_basis_from_surface_grid(contact.surface_master, shape=(2, 2), axes=(0, 1))
        ),
        "nodal": ff.MultiplierSpec.nodal_mortar(),
        "coarse_dual": ff.MultiplierSpec.coarse_dual_mortar(rank=1),
        "p0": ff.MultiplierSpec.p0_mortar(contact),
        "coarse_p0": ff.MultiplierSpec.coarse_p0_mortar(
            contact,
            patch_ids=np.zeros(int(contact.surface_master.conn.shape[0]), dtype=int),
        ),
    }
    if include_supermesh_p0:
        specs["p0_active"] = ff.MultiplierSpec.from_contact(contact, family="p0_active", side="master")
        specs["p0_supermesh"] = ff.MultiplierSpec.from_contact(contact, family="p0_supermesh", side="master")
    return specs


def _solve_variant(
    contact: ff.ContactSurfaceSpace,
    stiffness: np.ndarray,
    force: np.ndarray,
    multiplier: ff.MultiplierSpec,
) -> tuple[np.ndarray, int, int]:
    ops = ff.assemble_multiplier(contact, rho=0.0, multiplier=multiplier, backend="numpy")
    B = np.asarray(ops.B, dtype=float)
    system = np.block(
        [
            [stiffness, B.T],
            [B, np.zeros((B.shape[0], B.shape[0]), dtype=float)],
        ]
    )
    rhs = np.concatenate([force, np.zeros(B.shape[0], dtype=float)])
    rank = int(np.linalg.matrix_rank(system))
    if rank == system.shape[0]:
        solution = np.linalg.solve(system, rhs)
    else:
        solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return solution[: stiffness.shape[0]], int(B.shape[0]), rank


def solve_case(
    name: str,
    contact: ff.ContactSurfaceSpace,
    *,
    gamma: float,
    include_supermesh_p0: bool,
    all_variants: bool,
) -> list[dict[str, float | int | str]]:
    n_fixture = int(contact.surface_master.n_nodes)
    n_workpiece = int(contact.surface_slave.n_nodes)
    stiffness = _low_order_stiffness(n_fixture, n_workpiece, gamma=gamma)
    force = np.concatenate([np.zeros(n_fixture, dtype=float), np.ones(n_workpiece, dtype=float)])

    results = []
    solutions = {}
    for label, multiplier in _multipliers(
        contact,
        include_supermesh_p0=include_supermesh_p0,
        all_variants=all_variants,
    ).items():
        u, lambda_dofs, kkt_rank = _solve_variant(contact, stiffness, force, multiplier)
        solutions[label] = u
        results.append(
            {
                "case": name,
                "mortar": label,
                "lambda_dofs": lambda_dofs,
                "kkt_rank": kkt_rank,
                "kkt_size": int(stiffness.shape[0] + lambda_dofs),
                "work": float(force @ u),
            }
        )

    reference = solutions["dual"]
    reference_work = float(force @ reference)
    for row in results:
        u = solutions[str(row["mortar"])]
        row["disp_err"] = float(np.linalg.norm(u - reference, ord=np.inf))
        row["rel_work_err"] = float(abs(float(force @ u) - reference_work) / max(abs(reference_work), 1.0e-15))
    return results


def _print_table(rows: list[dict[str, float | int | str]]) -> None:
    print("case         mortar         lambda  rank/size  disp_err      rel_work_err")
    print("-----------  -------------  ------  ---------  ------------  ------------")
    for row in rows:
        print(
            f"{str(row['case']):11s}  "
            f"{str(row['mortar']):13s}  "
            f"{int(row['lambda_dofs']):6d}  "
            f"{int(row['kkt_rank']):4d}/{int(row['kkt_size']):<4d}  "
            f"{float(row['disp_err']):12.6e}  "
            f"{float(row['rel_work_err']):12.6e}"
        )


def _write_surface_vtu(path: Path, coords: np.ndarray, facets: np.ndarray, values: np.ndarray) -> None:
    try:
        import meshio
    except ImportError:
        print(f"meshio is not installed; skipping {path}")
        return
    data = np.asarray(values, dtype=float).reshape(-1)
    meshio.Mesh(
        points=np.asarray(coords, dtype=float),
        cells=[("triangle", np.asarray(facets, dtype=np.int32))],
        point_data={"u": data},
    ).write(path)


def write_vtu_outputs(output_dir: Path, *, all_variants: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for case, contact, gamma, include_supermesh in (
        ("matching", _matching_contact(), 30000.0, False),
        ("nonmatching", _nonmatching_contact(), 100000.0, True),
    ):
        n_fixture = int(contact.surface_master.n_nodes)
        n_workpiece = int(contact.surface_slave.n_nodes)
        stiffness = _low_order_stiffness(n_fixture, n_workpiece, gamma=gamma)
        force = np.concatenate([np.zeros(n_fixture, dtype=float), np.ones(n_workpiece, dtype=float)])
        for mortar, multiplier in _multipliers(
            contact,
            include_supermesh_p0=include_supermesh,
            all_variants=all_variants,
        ).items():
            u, _, _ = _solve_variant(contact, stiffness, force, multiplier)
            fixture_u = u[:n_fixture]
            workpiece_u = u[n_fixture:]
            _write_surface_vtu(
                output_dir / f"{case}_{mortar}_fixture.vtu",
                np.asarray(contact.surface_master.coords),
                np.asarray(contact.surface_master.conn),
                fixture_u,
            )
            _write_surface_vtu(
                output_dir / f"{case}_{mortar}_workpiece.vtu",
                np.asarray(contact.surface_slave.coords),
                np.asarray(contact.surface_slave.conn),
                workpiece_u,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-variants", action="store_true", help="Show all mortar variants instead of the coarse-P1-focused table.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for surface VTU outputs.")
    args = parser.parse_args()

    rows = []
    rows.extend(
        solve_case(
            "matching",
            _matching_contact(),
            gamma=30000.0,
            include_supermesh_p0=False,
            all_variants=args.all_variants,
        )
    )
    rows.extend(
        solve_case(
            "nonmatching",
            _nonmatching_contact(),
            gamma=100000.0,
            include_supermesh_p0=True,
            all_variants=args.all_variants,
        )
    )
    _print_table(rows)
    if args.output_dir is not None:
        write_vtu_outputs(args.output_dir, all_variants=args.all_variants)
        print(f"VTU output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

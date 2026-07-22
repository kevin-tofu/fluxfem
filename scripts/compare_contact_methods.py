#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxfem-matplotlib")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np

import fluxfem as ff


def build_nonmatching_hex_dataset() -> dict[str, Any]:
    fixture_mesh = ff.StructuredHexBox(
        nx=2,
        ny=2,
        nz=1,
        lx=1.0,
        ly=1.0,
        lz=0.25,
        origin=(0.0, 0.0, 0.0),
        order=1,
    ).build()
    workpiece_mesh = ff.StructuredHexBox(
        nx=3,
        ny=3,
        nz=1,
        lx=1.0,
        ly=1.0,
        lz=0.25,
        origin=(0.0, 0.0, -0.25),
        order=1,
    ).build()
    fixture_space = ff.make_hex_space(fixture_mesh, dim=3)
    workpiece_space = ff.make_hex_space(workpiece_mesh, dim=3)
    fixture_facets = fixture_mesh.facets_on_plane(axis=2, value=0.0, tol=1.0e-8)
    workpiece_facets = workpiece_mesh.facets_on_plane(axis=2, value=0.0, tol=1.0e-8)
    contact = ff.ContactSurfaceSpace.from_sides(
        ff.ContactSide.from_facets(fixture_mesh, fixture_facets, fixture_space),
        ff.ContactSide.from_facets(workpiece_mesh, workpiece_facets, workpiece_space),
        field_master="fixture",
        field_slave="workpiece",
        quad_order=2,
        backend="numpy",
        normal_sign=-1.0,
    )

    n_fixture = int(fixture_space.n_dofs)
    n_workpiece = int(workpiece_space.n_dofs)
    n_fixture_nodes = int(fixture_mesh.coords.shape[0])
    n_workpiece_nodes = int(workpiece_mesh.coords.shape[0])
    fixture_smoother = np.kron(
        np.eye(n_fixture_nodes) - np.ones((n_fixture_nodes, n_fixture_nodes)) / n_fixture_nodes,
        np.eye(3),
    )
    workpiece_smoother = np.kron(
        np.eye(n_workpiece_nodes) - np.ones((n_workpiece_nodes, n_workpiece_nodes)) / n_workpiece_nodes,
        np.eye(3),
    )
    stiffness = np.block(
        [
            [25.0 * np.eye(n_fixture) + 30000.0 * fixture_smoother, np.zeros((n_fixture, n_workpiece))],
            [np.zeros((n_workpiece, n_fixture)), np.eye(n_workpiece) + 30000.0 * workpiece_smoother],
        ]
    )
    load = np.zeros(n_fixture + n_workpiece, dtype=float)
    load[n_fixture + 2 : n_fixture + n_workpiece : 3] = -1.0e-3
    return {
        "name": "nonmatching_hex_fixture_workpiece",
        "contact": contact,
        "stiffness": stiffness,
        "load": load,
        "master_dofs": np.arange(n_fixture, dtype=int),
        "slave_dofs": n_fixture + np.arange(n_workpiece, dtype=int),
    }


def build_methods(contact) -> list[ff.ContactMethodSpec]:
    p0_patch = np.zeros(int(contact.surface_master.n_facets), dtype=int)
    params_penalty = ff.Params(
        alpha=100.0,
        inv_h=1.0,
        lam=0.0,
        mu=1.0,
        use_penalty=1.0,
        use_traction=0.0,
    )
    params_nitsche = ff.Params(
        alpha=100.0,
        inv_h=1.0,
        lam=0.0,
        mu=1.0,
        use_penalty=1.0,
        use_traction=1.0,
    )
    return [
        ff.ContactMethodSpec("mortar_nodal", "mortar", multiplier=ff.MultiplierSpec.nodal_mortar(value_dim=3)),
        ff.ContactMethodSpec("mortar_dual", "mortar", multiplier=ff.MultiplierSpec.dual_mortar(value_dim=3)),
        ff.ContactMethodSpec(
            "mortar_coarse_dual",
            "mortar",
            multiplier=ff.MultiplierSpec.coarse_dual_mortar(rank=3, value_dim=3),
        ),
        ff.ContactMethodSpec(
            "mortar_p0_active",
            "mortar",
            multiplier=ff.MultiplierSpec.from_contact(
                contact,
                family="p0_active",
                side="master",
                value_dim=3,
            ),
        ),
        ff.ContactMethodSpec(
            "mortar_p0_supermesh",
            "mortar",
            multiplier=ff.MultiplierSpec.from_contact(
                contact,
                family="p0_supermesh",
                side="master",
                value_dim=3,
            ),
        ),
        ff.ContactMethodSpec(
            "mortar_coarse_p0",
            "mortar",
            multiplier=ff.MultiplierSpec.coarse_p0_mortar(contact, patch_ids=p0_patch, value_dim=3),
        ),
        ff.ContactMethodSpec(
            "mortar_grid_p1",
            "mortar",
            multiplier=ff.MultiplierSpec.coarse_p1_mortar(
                basis=ff.coarse_p1_basis_from_surface_grid(contact.surface_master, shape=(2, 2), axes=(0, 1)),
                value_dim=3,
            ),
        ),
        ff.ContactMethodSpec(
            "mortar_patch_qr_p0_supermesh",
            "mortar",
            multiplier=ff.MultiplierSpec.patch_qr_mortar(
                contact,
                family="p0_supermesh",
                value_dim=3,
                constraint_scaling="l2",
            ),
        ),
        ff.ContactMethodSpec(
            "mortar_algebraic_qr_p0_supermesh",
            "mortar",
            multiplier=ff.MultiplierSpec.algebraic_qr_mortar(
                contact,
                family="p0_supermesh",
                value_dim=3,
                constraint_scaling="l2",
            ),
        ),
        ff.ContactMethodSpec(
            "pair_penalty_supermesh",
            "penalty",
            formulation="pair_nitsche_penalty",
            params=params_penalty,
        ),
        ff.ContactMethodSpec(
            "pair_nitsche_supermesh",
            "penalty",
            formulation="pair_nitsche_penalty",
            params=params_nitsche,
        ),
    ]


def write_outputs(output_dir: Path, metrics: list[ff.ContactMethodMetric], comparisons: list[ff.PrimalSolutionComparison]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [row.to_dict() for row in metrics]
    (output_dir / "metrics.json").write_text(json.dumps(metric_rows, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "solution_comparisons.json").write_text(
        json.dumps([row.to_dict() for row in comparisons], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    keys = [
        "method",
        "enforcement",
        "formulation",
        "ok",
        "elapsed_seconds",
        "operator_shape",
        "operator_nnz",
        "multiplier_count",
        "reduction_ratio",
        "rank_deficiency",
        "residual_norm",
        "reference_rel_error",
        "error",
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({key: row.get(key) for key in keys})
    plot_metrics(output_dir / "metrics.png", metrics)


def plot_metrics(path: Path, metrics: list[ff.ContactMethodMetric]) -> None:
    ok = [row for row in metrics if row.ok]
    names = [row.method for row in ok]
    fig, axes = plt.subplots(3, 1, figsize=(max(10.0, 0.65 * len(ok)), 9.0), constrained_layout=True)
    axes[0].bar(names, [row.elapsed_seconds for row in ok], color="#4c78a8")
    axes[0].set_ylabel("assemble+solve [s]")
    axes[1].bar(names, [row.multiplier_count or 0 for row in ok], color="#f58518")
    axes[1].set_ylabel("multiplier rows")
    rel_errors = [np.nan if row.reference_rel_error is None else row.reference_rel_error for row in ok]
    axes[2].bar(names, rel_errors, color="#54a24b")
    axes[2].set_ylabel("rel. solution error")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_table(metrics: list[ff.ContactMethodMetric]) -> None:
    headers = ["method", "ok", "rows", "nnz", "rank_def", "residual", "rel_err", "time_s"]
    rows = []
    for row in metrics:
        rows.append(
            [
                row.method,
                str(row.ok),
                "" if row.multiplier_count is None else str(row.multiplier_count),
                "" if row.operator_nnz is None else str(row.operator_nnz),
                "" if row.rank_deficiency is None else str(row.rank_deficiency),
                "" if row.residual_norm is None else f"{row.residual_norm:.3e}",
                "" if row.reference_rel_error is None else f"{row.reference_rel_error:.3e}",
                f"{row.elapsed_seconds:.3f}",
            ]
        )
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for values in rows:
        print("  ".join(values[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FluxFEM contact methods on one dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/fluxfem_contact_method_comparison"))
    parser.add_argument("--reference", default="mortar_algebraic_qr_p0_supermesh")
    args = parser.parse_args()

    dataset = build_nonmatching_hex_dataset()
    metrics, _, comparisons = ff.compare_contact_methods(
        dataset["contact"],
        build_methods(dataset["contact"]),
        stiffness=dataset["stiffness"],
        load=dataset["load"],
        master_dofs=dataset["master_dofs"],
        slave_dofs=dataset["slave_dofs"],
        reference=args.reference,
    )
    write_outputs(args.output_dir, metrics, comparisons)
    print_table(metrics)
    print(f"\noutput_dir: {args.output_dir}")


if __name__ == "__main__":
    main()

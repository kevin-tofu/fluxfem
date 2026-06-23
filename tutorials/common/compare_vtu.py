#!/usr/bin/env python3
"""
Compare displacement fields between two VTU files.

Example:
PYTHONPATH=src python tutorials/common/compare_vtu.py data/neo_hookean_bar_deformed.vtu tutorials/nonlinear/results/neo_hookean_cantilever.vtu --field1 displacement --field2 displacement
  PYTHONPATH=src python tutorials/common/compare_vtu.py a.vtu b.vtu --field1 u --field2 displacement
"""

import argparse
import sys
from typing import Optional

import numpy as np
import meshio

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def pick_vector_field(
    point_data: dict, preferred: Optional[str]
) -> tuple[str, np.ndarray]:
    """Select a 3-component vector field from point_data."""
    if preferred is not None:
        if preferred not in point_data:
            raise KeyError(f"field '{preferred}' not found; available: {list(point_data.keys())}")
        arr = np.asarray(point_data[preferred])
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"field '{preferred}' is not a vector (shape {arr.shape})")
        return preferred, arr

    # heuristic: prefer common displacement names
    preferred_names = ["u", "displacement", "disp", "ux", "x_def"]
    for name in preferred_names:
        if name in point_data:
            arr = np.asarray(point_data[name])
            if arr.ndim == 2 and arr.shape[1] == 3:
                return name, arr

    # fallback: first 3-component array
    for name, arr in point_data.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim == 2 and arr_np.shape[1] == 3:
            return name, arr_np
    raise ValueError("No vector (n x 3) point data field found.")


def summarize_diff(a: np.ndarray, b: np.ndarray) -> dict:
    diff = b - a
    if diff.ndim == 2 and diff.shape[1] == 3:
        diff_norm = np.linalg.norm(diff, axis=1)
        return {
            "max_norm": float(np.max(diff_norm)),
            "rms_norm": float(np.sqrt(np.mean(diff_norm**2))),
            "mean_norm": float(np.mean(diff_norm)),
            "max_component_abs": float(np.max(np.abs(diff))),
            "mean_component_abs": float(np.mean(np.abs(diff))),
        }
    diff_abs = np.abs(diff)
    return {
        "max_abs": float(np.max(diff_abs)),
        "rms": float(np.sqrt(np.mean(diff_abs**2))),
        "mean_abs": float(np.mean(diff_abs)),
    }


def maybe_plot_diff(a: np.ndarray, b: np.ndarray, coords: np.ndarray, out_path: str):
    """
    Save a quick visualization of the pointwise difference.
    - Left: histogram of norms / absolute error.
    - Right: XY scatter colored by error (use first two coordinate components).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plot") from exc

    diff = b - a
    if diff.ndim == 2 and diff.shape[1] == 3:
        err = np.linalg.norm(diff, axis=1)
        label = "||diff||"
    else:
        err = np.abs(diff).reshape(-1)
        label = "|diff|"

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(err, bins=60, color="tab:blue", alpha=0.8)
    ax[0].set_title("Error histogram")
    ax[0].set_xlabel(label)
    ax[0].set_ylabel("count")

    xy = coords[:, :2] if coords.shape[1] >= 2 else np.stack([coords[:, 0], np.zeros_like(coords[:, 0])], axis=1)
    sc = ax[1].scatter(xy[:, 0], xy[:, 1], c=err, s=8, cmap="viridis")
    ax[1].set_title("XY projection colored by error")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    fig.colorbar(sc, ax=ax[1], label=label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Compare displacement fields between two VTU files.")
    p.add_argument("vtu1", help="reference VTU file (e.g., neo_hookean_bar_deformed.vtu)")
    p.add_argument("vtu2", help="VTU file to compare (e.g., result.vtu)")
    p.add_argument(
        "--field1",
        help="point-data field name in vtu1 (default: first 3-component field)"
    )
    p.add_argument(
        "--field2",
        help="point-data field name in vtu2 (default: first 3-component field)"
    )
    p.add_argument(
        "--all-fields",
        action="store_true",
        help="Compare all common point-data fields (matching names and shapes). field1/field2 are ignored.",
    )
    p.add_argument(
        "--match-coords",
        action="store_true",
        help="Allow point-count mismatch by matching nearest coordinates (requires scipy).",
    )
    p.add_argument("--coord-tol", type=float, default=1e-6, help="Tolerance for coordinate matching (L2 distance).")
    p.add_argument(
        "--plot",
        metavar="PNG",
        help="Save a quick visualization (histogram + XY scatter) of the diff for the selected field.",
    )
    args = p.parse_args()

    v1 = meshio.read(args.vtu1)
    v2 = meshio.read(args.vtu2)

    # helper to align arrays, optionally by coordinate matching
    def align_field(arr1, arr2):
        arr1 = np.asarray(arr1, dtype=np.float64)
        arr2 = np.asarray(arr2, dtype=np.float64)
        if arr1.shape != arr2.shape:
            if not args.match_coords:
                return None, f"shape mismatch {arr1.shape} vs {arr2.shape}"
            if cKDTree is None:
                return None, "scipy is required for --match-coords; please install scipy or match shapes manually."
            tree = cKDTree(np.asarray(v2.points, dtype=np.float64))
            dists, idx = tree.query(np.asarray(v1.points, dtype=np.float64))
            if np.max(dists) > args.coord_tol:
                print(f"Warning: max coordinate mismatch {np.max(dists):.3e} exceeds tol {args.coord_tol:.3e}", file=sys.stderr)
            arr2 = arr2[idx]
            info = (dists.min(), dists.mean(), dists.max(), len(arr1))
        else:
            info = None
        return (arr1, arr2), info

    if not args.all_fields:
        name1, u1 = pick_vector_field(v1.point_data, args.field1)
        name2, u2 = pick_vector_field(v2.point_data, args.field2)
        pair, info = align_field(u1, u2)
        if pair is None:
            sys.exit(info)
        if info:
            mn, mean, mx, n = info
            print(f"Matched {n} points via nearest coords.")
            print(f"  min/mean/max match distance: {mn:.3e}/{mean:.3e}/{mx:.3e}")
        stats = summarize_diff(*pair)
        print(f"vtu1: {args.vtu1} field='{name1}' shape={pair[0].shape}")
        print(f"vtu2: {args.vtu2} field='{name2}' shape={pair[1].shape}")
        print("Diff stats (vtu2 - vtu1):")
        for k, v in stats.items():
            print(f"  {k}: {v:.6e}")
        if args.plot:
            maybe_plot_diff(pair[0], pair[1], np.asarray(v1.points), args.plot)
        return

    # all-fields mode: compare common point-data keys with compatible shapes
    common_keys = [k for k in v1.point_data.keys() if k in v2.point_data]
    if not common_keys:
        sys.exit("No common point-data fields to compare.")

    print(f"Comparing common fields: {common_keys}")
    for k in common_keys:
        pair, info = align_field(v1.point_data[k], v2.point_data[k])
        if pair is None:
            print(f"- {k}: skipped ({info})")
            continue
        if info:
            mn, mean, mx, n = info
            print(f"- {k} shape={pair[0].shape} (matched {n} points, min/mean/max dist {mn:.3e}/{mean:.3e}/{mx:.3e})")
        else:
            print(f"- {k} shape={pair[0].shape}")
        stats = summarize_diff(*pair)
        for kk, vv in stats.items():
            print(f"    {kk}: {vv:.6e}")


if __name__ == "__main__":
    main()

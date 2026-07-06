#!/usr/bin/env python
"""Solve a two-body Nitsche contact example and export a combined VTU.

The generated VTU contains both deformable bodies in one file and includes:

- displacement: vector field for ParaView Warp By Vector
- displacement_magnitude: scalar field for coloring both bodies
- body_id: 0 for the upper body, 1 for the lower body
- node_tag: 1 contact, 2 fixed support, 3 loaded surface

Run from the repository root:

    PYTHONPATH=src python tutorials/contact/two_body_contact_displacement_vtu.py

Open the printed VTU path in ParaView, color by ``displacement_magnitude``, and
apply Warp By Vector using ``displacement``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


sys.path.append(os.path.dirname(__file__))
from nitsche_contact_supermesh_api import NitscheContactParams, run_fluxfem_demo  # noqa: E402


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "two_body_contact_displacement_vtu",
    )
    parser.add_argument("--nx-top", type=_positive_int, default=8)
    parser.add_argument("--ny-top", type=_positive_int, default=8)
    parser.add_argument("--nz-top", type=_positive_int, default=4)
    parser.add_argument("--nx-bot", type=_positive_int, default=6)
    parser.add_argument("--ny-bot", type=_positive_int, default=6)
    parser.add_argument("--nz-bot", type=_positive_int, default=3)
    parser.add_argument("--quad-order", type=_positive_int, default=3)
    parser.add_argument("--total-force", type=float, default=100.0)
    parser.add_argument("--cg-tol", type=float, default=1.0e-8)
    parser.add_argument("--cg-maxiter", type=int, default=3000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vtu_path = args.out_dir / "two_body_contact_displacement.vtu"
    summary_path = args.out_dir / "two_body_contact_summary.json"

    params = NitscheContactParams(
        nx_top=args.nx_top,
        ny_top=args.ny_top,
        nz_top=args.nz_top,
        nx_bot=args.nx_bot,
        ny_bot=args.ny_bot,
        nz_bot=args.nz_bot,
        quad_order=args.quad_order,
        total_force=args.total_force,
    )
    result = run_fluxfem_demo(
        params,
        platform="cpu",
        low_mem=False,
        x64=True,
        linear_solver="spsolve",
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        vtu_path=str(vtu_path),
        verbose=args.verbose,
        return_nodes=True,
    )

    max_displacement = None
    mean_displacement = None
    if result.u_nodes is not None:
        import numpy as np

        disp_mag = np.linalg.norm(result.u_nodes, axis=1)
        max_displacement = float(np.max(disp_mag))
        mean_displacement = float(np.mean(disp_mag))

    summary = _json_safe(dict(result.summary))
    summary.update(
        {
            "vtu": str(vtu_path),
            "fields": ["displacement", "displacement_magnitude", "body_id", "node_tag"],
            "max_displacement": max_displacement,
            "mean_displacement": mean_displacement,
            "paraview": "Color by displacement_magnitude and Warp By Vector using displacement.",
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"VTU written: {vtu_path}")
    print(f"Summary written: {summary_path}")
    print("ParaView: color by displacement_magnitude, then Warp By Vector(displacement).")


if __name__ == "__main__":
    main()

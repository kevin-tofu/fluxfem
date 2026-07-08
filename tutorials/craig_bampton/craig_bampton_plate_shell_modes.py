#!/usr/bin/env python
"""Craig-Bampton modal check for Mindlin plate and Reissner-Mindlin shell.

The script clamps the left edge, retains the right edge as physical interface
DOFs, and compares full-order free-vibration frequencies with a CB-ROM.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_plate_shell_modes.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import scipy.linalg as la

import fluxfem as ff


def _frequencies(k, m, n_report: int) -> np.ndarray:
    w2 = la.eigh(np.asarray(k, dtype=float), np.asarray(m, dtype=float), eigvals_only=True)
    w2 = np.asarray(w2, dtype=float)
    w2 = w2[w2 > 1.0e-8]
    return np.sqrt(w2[:n_report])


def _build_plate(args):
    coords, conn = ff.structured_plate_grid(nx=args.nx, ny=args.ny, length_x=args.length, length_y=args.width)
    section = ff.PlateSection(
        E=args.E,
        nu=args.nu,
        thickness=args.thickness,
        rho=args.rho,
        shear_mode=args.shear_mode,
    )
    k_full = ff.assemble_mindlin_plate_stiffness(coords, conn, section, format="csr")
    m_full = ff.assemble_mindlin_plate_mass(coords, conn, section, format="csr")
    fixed = ff.plate_node_dofs(np.flatnonzero(np.isclose(coords[:, 0], coords[:, 0].min())))
    retained_full = ff.plate_node_dofs(np.flatnonzero(np.isclose(coords[:, 0], coords[:, 0].max())))
    return coords, k_full, m_full, fixed, retained_full


def _build_shell(args):
    coords2, conn = ff.structured_plate_grid(nx=args.nx, ny=args.ny, length_x=args.length, length_y=args.width)
    coords = np.column_stack([coords2[:, 0], coords2[:, 1], args.tilt_z * coords2[:, 0]])
    section = ff.ShellSection(
        E=args.E,
        nu=args.nu,
        thickness=args.thickness,
        rho=args.rho,
        shear_mode=args.shear_mode,
    )
    k_full = ff.assemble_shell_stiffness(coords, conn, section, format="csr")
    m_full = ff.assemble_shell_mass(coords, conn, section, format="csr")
    fixed = ff.shell_node_dofs(np.flatnonzero(np.isclose(coords[:, 0], coords[:, 0].min())))
    retained_full = ff.shell_node_dofs(np.flatnonzero(np.isclose(coords[:, 0], coords[:, 0].max())))
    return coords, k_full, m_full, fixed, retained_full


def run_case(name: str, args) -> dict[str, np.ndarray | int]:
    coords, k_full, m_full, fixed, retained_full = _build_plate(args) if name == "plate" else _build_shell(args)
    free = np.asarray(ff.free_dofs(k_full.shape[0], fixed), dtype=int)
    retained = np.flatnonzero(np.isin(free, retained_full)).astype(np.int32)
    k = k_full[free, :][:, free]
    m = m_full[free, :][:, free]
    n_modes = min(args.n_modes, free.size - retained.size)

    cb = ff.make_craig_bampton_basis(
        k,
        m,
        retained_dofs=retained,
        n_modes=n_modes,
        backend=args.cb_backend,
        constraint_solver="spsolve",
        modal_solver="dense",
    )
    k_red = cb.project_matrix(k)
    m_red = cb.project_matrix(m)

    n_report = min(args.n_report, cb.n_reduced, free.size)
    full = _frequencies(k.toarray(), m.toarray(), n_report)
    rom = _frequencies(k_red, m_red, n_report)
    return {
        "nodes": coords.shape[0],
        "free_dofs": free.size,
        "retained_dofs": retained.size,
        "cb_modes": n_modes,
        "reduced_dofs": cb.n_reduced,
        "full": full,
        "rom": rom,
        "rel_err": np.abs(rom - full) / np.maximum(full, 1.0e-30),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CB-ROM modal comparison for plate and shell mass/stiffness matrices.")
    p.add_argument("--case", choices=["plate", "shell", "both"], default="both")
    p.add_argument("--nx", type=int, default=4)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--length", type=float, default=2.0)
    p.add_argument("--width", type=float, default=0.6)
    p.add_argument("--thickness", type=float, default=0.04)
    p.add_argument("--E", type=float, default=70.0e9)
    p.add_argument("--nu", type=float, default=0.33)
    p.add_argument("--rho", type=float, default=2700.0)
    p.add_argument("--shear-mode", choices=["reduced", "full", "mitc4"], default="mitc4")
    p.add_argument("--tilt-z", type=float, default=0.15)
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--n-report", type=int, default=5)
    p.add_argument("--cb-backend", choices=["scipy", "jax"], default="scipy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cases = ["plate", "shell"] if args.case == "both" else [args.case]
    for case in cases:
        result = run_case(case, args)
        print(f"{case} CB modal check")
        print("nodes:                ", result["nodes"])
        print("free DOFs:            ", result["free_dofs"])
        print("retained DOFs:        ", result["retained_dofs"])
        print("fixed-interface modes:", result["cb_modes"])
        print("reduced DOFs:         ", result["reduced_dofs"])
        print("mode  omega_full      omega_rom       rel_error")
        print("----  --------------  --------------  ----------")
        for i, (wf, wr, err) in enumerate(zip(result["full"], result["rom"], result["rel_err"]), start=1):
            print(f"{i:4d}  {wf:14.8e}  {wr:14.8e}  {err:10.3e}")
        print()


if __name__ == "__main__":
    main()

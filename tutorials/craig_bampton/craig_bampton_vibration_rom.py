#!/usr/bin/env python
"""Craig-Bampton vibration ROM with stiffness and mass projection.

This example builds a small scalar bar, fixes the left face, retains the right
face as physical interface coordinates, and compares free-vibration frequencies
from the full system and the Craig-Bampton reduced system.

Run from the repository root:

    PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_vibration_rom.py
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=12, help="Number of hex elements along the bar.")
    parser.add_argument("--n-modes", type=int, default=6, help="Fixed-interface modes kept in the CB basis.")
    parser.add_argument("--n-report", type=int, default=5, help="Number of frequencies to print.")
    return parser.parse_args()


def generalized_frequencies(stiffness: np.ndarray, mass: np.ndarray, *, n_report: int) -> np.ndarray:
    eigvals = la.eigh(stiffness, mass, eigvals_only=True)
    eigvals = np.maximum(np.asarray(eigvals, dtype=float), 0.0)
    return np.sqrt(eigvals)[:n_report]


def main() -> None:
    args = parse_args()

    mesh = ff.StructuredHexBox(nx=args.nx, ny=1, nz=1, lx=1.0, ly=0.2, lz=0.2).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)
    stiffness = space.assemble(ff.diffusion_form, params=1.0)
    mass = space.assemble_mass_matrix()

    coords = np.asarray(mesh.coords, dtype=float)
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    fixed = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_min, atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    right = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_max, atol=1.0e-12),
        components=[0],
        dof_per_node=1,
    )
    free = np.asarray(ff.free_dofs(space.n_dofs, fixed), dtype=int)
    retained = np.flatnonzero(np.isin(free, np.asarray(right, dtype=int))).astype(int)

    k_free = stiffness.to_csr()[free, :][:, free]
    m_free = mass.to_csr()[free, :][:, free]

    cb = ff.make_craig_bampton_basis(
        k_free,
        m_free,
        retained_dofs=retained,
        n_modes=args.n_modes,
        constraint_solver="spsolve",
        modal_solver="eigsh",
    )
    k_red = np.asarray(cb.project_matrix(k_free), dtype=float)
    m_red = np.asarray(cb.project_matrix(m_free), dtype=float)

    n_report = min(args.n_report, cb.n_reduced, free.size)
    full_freq = generalized_frequencies(k_free.toarray(), m_free.toarray(), n_report=n_report)
    rom_freq = generalized_frequencies(k_red, m_red, n_report=n_report)
    rel_err = np.abs(rom_freq - full_freq) / np.maximum(full_freq, 1.0e-30)

    m_sym_err = np.linalg.norm(m_red - m_red.T)
    m_min_eig = float(np.min(la.eigvalsh(m_red)))

    print("Craig-Bampton vibration ROM")
    print("full free DOFs:       ", free.size)
    print("retained DOFs:        ", cb.n_retained)
    print("fixed-interface modes:", cb.n_modes)
    print("reduced DOFs:         ", cb.n_reduced)
    print("M_red symmetry err:   ", f"{m_sym_err:.3e}")
    print("min eig(M_red):       ", f"{m_min_eig:.6e}")
    print()
    print("mode  omega_full      omega_rom       rel_error")
    print("----  --------------  --------------  ----------")
    for i, (wf, wr, err) in enumerate(zip(full_freq, rom_freq, rel_err), start=1):
        print(f"{i:4d}  {wf:14.8e}  {wr:14.8e}  {err:10.3e}")


if __name__ == "__main__":
    main()

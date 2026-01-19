"""Shared helpers for thermoelastic bar tutorials."""
from __future__ import annotations

import os
import numpy as np

import fluxfem as ff


def build_bar_mesh(*, nx: int, ny: int, nz: int, lx: float, ly: float, lz: float):
    return ff.StructuredHexBox(
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
    ).build()


def x_bounds(mesh):
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())
    return xmin, xmax, coords


def boundary_dofs_at_x(mesh, x_value: float, *, tol: float = 1e-8):
    return mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], x_value, atol=tol),
        components="x",
    )


def boundary_bc_at_x(mesh, x_value: float, *, tol: float = 1e-8):
    return ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], x_value, atol=tol),
        components="x",
    )


def default_output_path(filename: str) -> str:
    return os.path.join("result", "tutorials", filename)

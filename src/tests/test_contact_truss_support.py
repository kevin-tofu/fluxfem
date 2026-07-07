from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "contact" / "contact_supported_box_by_truss_springs.py"
    spec = importlib.util.spec_from_file_location("contact_supported_box_by_truss_springs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contact_supported_box_by_truss_springs_solves_with_downward_motion():
    tutorial = _load_tutorial_module()
    model = tutorial.build_contact_supported_box_by_truss_springs(gravity=-0.05)
    system = model["system"]

    diagonal_shift = 1.0e-8
    fixed = model["fixed_dofs"]
    sol = np.asarray(
        system.solve(
            format="csr",
            dirichlet_dofs=fixed,
            dirichlet_vals=np.zeros((fixed.size,), dtype=float),
            diagonal_shift=diagonal_shift,
        ),
        dtype=float,
    )
    n_top = int(model["top_space"].n_dofs)
    u_top = sol[:n_top].reshape(-1, 3)
    u_support = sol[n_top : n_top + int(model["support_space"].n_dofs)]

    assert np.all(np.isfinite(sol))
    assert float(np.mean(u_top[:, 2])) < 0.0
    assert float(np.mean(u_support[model["bottom_z_dofs"]])) < 0.0
    assert model["truss_column_stiffness"] > 0.0
    assert model["spring_per_bottom_dof"] > 0.0

    K_csr, F = system.assemble(format="csr")
    residual = K_csr @ sol - np.asarray(F, dtype=float) + diagonal_shift * sol
    free = np.ones((K_csr.shape[0],), dtype=bool)
    free[fixed] = False
    assert float(np.linalg.norm(residual[free])) < 1.0e-7

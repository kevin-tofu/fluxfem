from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / "solid_shell_translational_tie.py"
    spec = importlib.util.spec_from_file_location("solid_shell_translational_tie", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_solid_shell_translational_tie_matches_interface_displacements():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_shell_tie(nx=2, ny=1, nz=1, pressure_z=-1.0)
    system = model["system"]
    fixed = model["fixed_dofs"]

    u_all = np.asarray(
        system.solve(
            format="csr",
            dirichlet_dofs=fixed,
            dirichlet_vals=np.zeros((fixed.size,), dtype=float),
        ),
        dtype=float,
    )

    solid_tie_u = u_all[model["solid_tie_dofs"]].reshape(-1, 3)
    shell_tie_u = u_all[model["shell_tie_dofs"]].reshape(-1, 3)
    np.testing.assert_allclose(solid_tie_u, shell_tie_u, rtol=1.0e-9, atol=1.0e-9)

    shell_u = u_all[model["shell_offset"] : model["shell_offset"] + model["shell_n_dofs"]].reshape(-1, 6)
    assert float(np.min(shell_u[:, 2])) < 0.0

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False
    assert float(np.linalg.norm(residual[free])) < 1.0e-6

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / "solid_beam_rbe3_coupling.py"
    spec = importlib.util.spec_from_file_location("solid_beam_rbe3_coupling", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_solid_beam_rbe3_coupling_ties_remote_to_beam_root():
    tutorial = _load_tutorial_module()
    model = tutorial.build_solid_beam_coupling(solid_nx=2, solid_ny=1, solid_nz=1, beam_elems=2, tip_load_z=-5.0)
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

    remote_q = u_all[model["remote_dofs"]]
    beam_root_q = u_all[model["beam_root_dofs"]]
    beam_tip_q = u_all[model["beam_tip_dofs"]]

    np.testing.assert_allclose(remote_q, beam_root_q, rtol=1.0e-9, atol=1.0e-9)
    assert float(beam_tip_q[2]) < float(beam_root_q[2])

    K_full, F_full = system.assemble(format="csr")
    residual = K_full @ u_all - np.asarray(F_full, dtype=float)
    free = np.ones((K_full.shape[0],), dtype=bool)
    free[fixed] = False
    assert float(np.linalg.norm(residual[free])) < 1.0e-6

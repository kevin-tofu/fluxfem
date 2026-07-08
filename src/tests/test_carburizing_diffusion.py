from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "diffusion" / "carburizing_diffusion_bar.py"
    spec = importlib.util.spec_from_file_location("carburizing_diffusion_bar", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_carburizing_diffusion_bar_runs_and_matches_erfc_order(tmp_path):
    tutorial = _load_tutorial_module()
    vtu = tmp_path / "carburizing.vtu"
    csv = tmp_path / "carburizing.csv"

    result = tutorial.run_carburizing(
        nx=20,
        ny=1,
        nz=1,
        dt=0.05,
        steps=20,
        output_vtu=str(vtu),
        output_csv=str(csv),
    )

    profile = result["carbon_profile"]
    assert result["space"].n_dofs == 84
    np.testing.assert_allclose(profile[0], result["c_surface"], atol=1.0e-12)
    assert float(profile[-1]) < 0.201
    assert result["case_depth"] is not None
    assert 0.0 < float(result["case_depth"]) < 0.2
    assert float(result["rms_error"]) < 0.02
    assert vtu.exists()
    assert csv.exists()

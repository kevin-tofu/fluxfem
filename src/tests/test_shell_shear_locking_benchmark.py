from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / "flat_shell_shear_locking_benchmark.py"
    spec = importlib.util.spec_from_file_location("flat_shell_shear_locking_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flat_shell_shear_locking_benchmark_orders_modes():
    tutorial = _load_tutorial_module()
    result = tutorial.run_benchmark(nx=12, ny=2, thickness=0.02, tip_load=-100.0)

    full = result["modes"]["full"]
    reduced = result["modes"]["reduced"]
    mitc4 = result["modes"]["mitc4"]

    assert full["tip"] < 0.0
    assert reduced["tip"] < 0.0
    assert mitc4["tip"] < 0.0
    assert full["rel_error"] > 0.5
    assert reduced["rel_error"] < 0.05
    assert mitc4["rel_error"] < 0.05
    assert abs(full["tip"]) < abs(mitc4["tip"]) < abs(reduced["tip"])


def test_flat_shell_shear_locking_benchmark_tilted_matches_flat():
    tutorial = _load_tutorial_module()
    flat = tutorial.run_benchmark(nx=8, ny=2, thickness=0.02, tip_load=-100.0, tilt_z=0.0)
    tilted = tutorial.run_benchmark(nx=8, ny=2, thickness=0.02, tip_load=-100.0, tilt_z=0.25)

    for mode in ("full", "reduced", "mitc4"):
        np.testing.assert_allclose(tilted["modes"][mode]["tip"], flat["modes"][mode]["tip"], rtol=1.0e-10, atol=1.0e-12)

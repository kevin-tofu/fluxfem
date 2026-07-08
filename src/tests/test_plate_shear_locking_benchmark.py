from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / "mindlin_plate_shear_locking_benchmark.py"
    spec = importlib.util.spec_from_file_location("mindlin_plate_shear_locking_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mindlin_plate_shear_locking_benchmark_orders_modes():
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

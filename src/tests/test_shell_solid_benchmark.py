from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / "elasticity" / "shell_solid_cantilever_benchmark.py"
    spec = importlib.util.spec_from_file_location("shell_solid_cantilever_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shell_solid_cantilever_benchmark_matches_beam_order():
    tutorial = _load_tutorial_module()
    result = tutorial.run_benchmark(
        shell_nx=8,
        shell_ny=2,
        solid_nx=12,
        solid_ny=2,
        solid_nz=2,
        length=2.0,
        width=0.3,
        thickness=0.08,
        tip_load_z=-100.0,
    )

    assert result["shell_tip"] < 0.0
    assert result["solid_tip"] < 0.0
    assert result["shell_rel_error"] < 0.05
    assert result["solid_rel_error"] < 0.75
    assert abs(result["solid_tip"]) < abs(result["shell_tip"])

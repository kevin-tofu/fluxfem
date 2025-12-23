from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class NewtonIterRecord:
    iter: int
    res_inf: float
    res_two: float
    rel_res_inf: float
    alpha: float
    step_norm: float
    lin_iters: Optional[int] = None
    lin_converged: Optional[bool] = None
    lin_residual: Optional[float] = None
    nan_detected: bool = False
    assemble_time: Optional[float] = None
    linear_time: Optional[float] = None


@dataclass
class LoadStepResult:
    load_factor: float
    info: Any
    solve_time: float
    u: Any
    iter_history: List[NewtonIterRecord] = field(default_factory=list)
    exception: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

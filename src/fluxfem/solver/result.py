from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class SolverResult:
    """Common solver result for linear/nonlinear solves."""

    converged: bool
    iters: int

    residual_norm: Optional[float] = None
    residual0: Optional[float] = None
    rel_residual: Optional[float] = None

    line_search_steps: int = 0

    # linear-solver stats (for Newton inner solve or standalone linear solve)
    linear_iters: Optional[int] = None
    linear_converged: Optional[bool] = None
    linear_residual: Optional[float] = None
    linear_solve_time: Optional[float] = None
    pc_setup_time: Optional[float] = None
    pmat_build_time: Optional[float] = None
    pmat_rebuilds: Optional[int] = None
    pmat_mode: Optional[str] = None
    linear_fallbacks: Optional[List[str]] = None

    tol: Optional[float] = None
    atol: Optional[float] = None
    stopping_criterion: Optional[float] = None
    step_norm: Optional[float] = None

    stop_reason: Optional[str] = None  # converged|maxiter|linfail|nan|exception|unknown
    nan_detected: bool = False

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        status = "converged" if self.converged else "not converged"
        parts = [f"{status} in {self.iters} iters"]
        if self.residual_norm is not None:
            parts.append(f"||R||={self.residual_norm:.3e}")
        if self.residual0 is not None:
            parts.append(f"||R0||={self.residual0:.3e}")
        if self.rel_residual is not None:
            parts.append(f"rel={self.rel_residual:.3e}")
        if self.tol is not None:
            parts.append(f"tol={self.tol:.1e}")
        if self.atol is not None and self.atol > 0:
            parts.append(f"atol={self.atol:.1e}")
        if self.stopping_criterion is not None:
            parts.append(f"crit={self.stopping_criterion:.3e}")
        if self.step_norm is not None:
            parts.append(f"step2={self.step_norm:.3e}")
        if self.line_search_steps:
            parts.append(f"ls_steps={self.line_search_steps}")
        if self.linear_iters is not None:
            parts.append(f"lin_iters={self.linear_iters}")
        if self.linear_converged is not None:
            parts.append(f"lin_conv={self.linear_converged}")
        if self.linear_residual is not None:
            parts.append(f"lin_res={self.linear_residual:.3e}")
        if self.linear_solve_time is not None:
            parts.append(f"lin_solve_dt={self.linear_solve_time:.3e}s")
        if self.pc_setup_time is not None:
            parts.append(f"pc_setup_dt={self.pc_setup_time:.3e}s")
        if self.pmat_build_time is not None:
            parts.append(f"pmat_dt={self.pmat_build_time:.3e}s")
        if self.pmat_rebuilds is not None:
            parts.append(f"pmat_rebuilds={self.pmat_rebuilds}")
        if self.pmat_mode is not None:
            parts.append(f"pmat_mode={self.pmat_mode}")
        if self.linear_fallbacks:
            parts.append(f"lin_fallbacks={self.linear_fallbacks}")
        if self.stop_reason:
            parts.append(f"reason={self.stop_reason}")
        if self.nan_detected:
            parts.append("nan_detected=True")
        return ", ".join(parts)

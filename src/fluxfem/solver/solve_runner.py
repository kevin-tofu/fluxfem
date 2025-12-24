from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Sequence

import numpy as np
import jax.numpy as jnp

from ..core.assembly import assemble_bilinear_form
from ..core.solver import spdirect_solve_cpu, spdirect_solve_gpu
from .cg import cg_solve, cg_solve_jax
from .sparse import FluxSparseMatrix
from .dirichlet import expand_dirichlet_solution
from .newton import newton_solve
from .result import SolverResult
from .history import NewtonIterRecord, LoadStepResult
from ..tools.timer import SectionTimer, NullTimer


@dataclass
class NonlinearAnalysis:
    """
    Bundle problem data needed for a Newton solve with load scaling.

    Attributes
    ----------
    space : Any
        FE space containing topology/dofs.
    residual_form : Any
        Internal residual form (e.g., neo_hookean_residual_form).
    params : Any
        Parameters forwarded to the residual form.
    base_external_vector : Any | None
        Unscaled external load vector (scaled by load factor in `external_for_load`).
    dirichlet : tuple | None
        (dofs, values) for Dirichlet boundary conditions.
    jacobian_pattern : Any | None
        Optional sparsity pattern to reuse between load steps.
    dtype : Any
        dtype for the solution vector (defaults to float64).
    """

    space: Any
    residual_form: Any
    params: Any
    base_external_vector: Any | None = None
    dirichlet: tuple | None = None
    jacobian_pattern: Any | None = None
    dtype: Any = jnp.float64

    def external_for_load(self, load_factor: float):
        if self.base_external_vector is None:
            return None
        return jnp.asarray(load_factor * self.base_external_vector, dtype=self.dtype)


@dataclass
class NewtonLoopConfig:
    """
    Control parameters for the Newton loop with load stepping.
    """

    tol: float = 1e-8
    atol: float = 0.0
    maxiter: int = 20
    line_search: bool = False
    max_ls: int = 10
    ls_c: float = 1e-4
    linear_solver: str = "spsolve"
    linear_maxiter: int | None = None
    linear_tol: float | None = None
    linear_preconditioner: Any | None = None
    load_sequence: Sequence[float] | None = None
    n_steps: int = 1

    def schedule(self) -> List[float]:
        """
        Return a monotonically increasing list of load factors (0->1].
        """
        if self.load_sequence is not None:
            return list(self.load_sequence)
        n = max(1, self.n_steps)
        return list(np.linspace(0.0, 1.0, n + 1, endpoint=True)[1:])


class NewtonSolveRunner:
    """
    Run one or more Newton solves across load factors.

    This orchestrates load stepping, assembles external load per step,
    and returns the full (Dirichlet-expanded) solution and per-step history.
    """

    def __init__(self, analysis: NonlinearAnalysis, config: NewtonLoopConfig):
        self.analysis = analysis
        self.config = config

    def run(
        self,
        u0=None,
        *,
        load_sequence: Sequence[float] | None = None,
        newton_callback: Callable | None = None,
        step_callback: Callable[[LoadStepResult], None] | None = None,
        timer: "SectionTimer | None" = None,
        report_timing: bool = True
    ):
        """
        Execute Newton solves over the configured load schedule.

        Parameters
        ----------
        u0 : array-like | None
            Initial guess (defaults to zeros).
        load_sequence : sequence | None
            Optional per-call load schedule; if omitted, uses config.schedule().
        newton_callback : callable | None
            Per-iteration callback passed to `newton_solve`.
        step_callback : callable | None
            Optional hook called after each load step with LoadStepResult.
        Returns:
            u: full solution (Dirichlet expanded), dtype per analysis.dtype
            history: list of LoadStepResult per load factor
        """
        # timer = timer or NullTimer()
        timer = timer or SectionTimer(hierarchical=True)
        with timer.section("run_total"):
            dtype = self.analysis.dtype

            with timer.section("preprocess"):

                if u0 is None:
                    u = jnp.zeros(self.analysis.space.n_dofs, dtype=dtype)
                else:
                    u = jnp.asarray(u0, dtype=dtype)

                schedule_raw = list(load_sequence) if load_sequence is not None else self.config.schedule()
                # enforce monotone increasing 0->1 schedule, warn if dropped
                schedule = []
                dropped = []
                prev = 0.0
                for lf in schedule_raw:
                    lf_clamped = float(lf)
                    if not np.isfinite(lf_clamped):
                        dropped.append(("nonfinite", lf_clamped))
                        continue
                    if lf_clamped < 0.0 or lf_clamped > 1.0:
                        dropped.append(("out_of_range", lf_clamped))
                        continue
                    if lf_clamped < prev:
                        dropped.append(("nonmonotone", lf_clamped))
                        continue
                    schedule.append(lf_clamped)
                    prev = lf_clamped
                history: List[LoadStepResult] = []
                for step_i, load_factor in enumerate(schedule, start=1):
                    with timer.section("step"):
                        external = self.analysis.external_for_load(load_factor)
                        iter_log: List[NewtonIterRecord] = []
                        ext_nnz = int(jnp.count_nonzero(external))
                        ext_inf = int(jnp.count_nonzero(external))
                        # assumes u is full dofs (xyz)
                        u_nodes0 = np.asarray(u).reshape(-1, 3)
                        max_u0 = float(np.linalg.norm(u_nodes0, axis=1).max()) if u_nodes0.size else 0.0
                        print(
                            f"[load factor step {step_i}/{len(schedule)}] lf={load_factor:.3f} "
                            f"||F||_inf={ext_inf:.3e} nnz={ext_nnz} max|u|_start={max_u0:.3e}"
                        )

                        def cb(d):
                            # d: {"iter": k, "residual_norm": ..., "rel_residual": ..., "alpha": ..., "step_norm": ...}
                            lin_iters = d.get("linear_iters")
                            lin_res = d.get("linear_residual")
                            lin_conv = d.get("linear_converged")
                            iter_log.append(
                                NewtonIterRecord(
                                    iter=int(d.get("iter", -1)),
                                    res_inf=float(d.get("res_inf", float("nan"))),
                                    res_two=float(d.get("res_two", float("nan"))),
                                    rel_res_inf=float(d.get("rel_residual", float("nan"))),
                                    alpha=float(d.get("alpha", 1.0)),
                                    step_norm=float(d.get("step_norm", float("nan"))),
                                    lin_iters=int(lin_iters) if lin_iters is not None else None,
                                    lin_converged=bool(lin_conv) if lin_conv is not None else None,
                                    lin_residual=float(lin_res) if lin_res is not None else None,
                                    nan_detected=bool(d.get("nan_detected", False)),
                                )
                            )
                            if newton_callback is not None:
                                newton_callback(d)

                    try:
                        u, info = newton_solve(
                            self.analysis.space,
                            self.analysis.residual_form,
                            u,
                            self.analysis.params,
                            tol=self.config.tol,
                            atol=self.config.atol,
                            maxiter=self.config.maxiter,
                            linear_solver=self.config.linear_solver,
                            linear_maxiter=self.config.linear_maxiter,
                            linear_tol=self.config.linear_tol,
                            linear_preconditioner=self.config.linear_preconditioner,
                            dirichlet=self.analysis.dirichlet,
                            line_search=self.config.line_search,
                            max_ls=self.config.max_ls,
                            ls_c=self.config.ls_c,
                            external_vector=external,
                            callback=cb,
                            jacobian_pattern=self.analysis.jacobian_pattern,
                        )
                        exception = None
                    except Exception as e:  # pragma: no cover - defensive
                        info = SolverResult(converged=False, iters=0, stop_reason="exception", nan_detected=False)
                        exception = repr(e)

                # ===== [B] OUTER LOOP PRINT (STEP END) =====
                u_nodes1 = np.asarray(u).reshape(-1, 3)
                max_u1 = float(np.linalg.norm(u_nodes1, axis=1).max()) if u_nodes1.size else 0.0
                step_solve_time = timer._records.get("step>newton_solve", [0.0])[-1]
                print(
                    f"  -> converged={getattr(info,'converged',None)} iters={getattr(info,'iters',None)} "
                    f"time={step_solve_time:.3f}s max|u|_end={max_u1:.3e}"
                    + (f" EXC={exception}" if exception else "")
                )

                meta = {
                    "load_factor": load_factor,
                    "linear_solver": self.config.linear_solver,
                    "line_search": self.config.line_search,
                    "maxiter": self.config.maxiter,
                    "n_dofs": self.analysis.space.n_dofs,
                    "dtype": str(self.analysis.dtype),
                    "u_layout": "full",
                    "schedule": schedule,
                    "schedule_dropped": dropped,
                }

                result = LoadStepResult(
                    load_factor=load_factor,
                    info=info,
                    solve_time=step_solve_time,
                    u=u,
                    iter_history=iter_log,
                    exception=exception,
                    meta=meta,
                )
                history.append(result)
                if step_callback is not None:
                    step_callback(result)

        if report_timing:
            timer.report(sort_by="total")
        return u, history


def _condense_flux_dirichlet(K: FluxSparseMatrix, F, dirichlet):
    dir_dofs, dir_vals = dirichlet
    dir_arr = np.asarray(dir_dofs, dtype=int)
    dir_vals_arr = np.asarray(dir_vals, dtype=float)
    K_csr = K.to_csr()
    mask = np.ones(K_csr.shape[0], dtype=bool)
    mask[dir_arr] = False
    free = np.nonzero(mask)[0]
    F_full = np.asarray(F, dtype=float)
    K_fd = K_csr[free][:, dir_arr] if dir_arr.size > 0 else None
    F_free_base = F_full[free]
    offset = K_fd @ dir_vals_arr if K_fd is not None and dir_arr.size > 0 else None
    K_ff = K_csr[free][:, free]
    return K_ff, F_free_base, offset, free, dir_arr, dir_vals_arr


def solve_nonlinear(
    space,
    residual_form,
    params,
    *,
    dirichlet: tuple | None = None,
    base_external_vector=None,
    dtype=jnp.float64,
    maxiter: int = 20,
    tol: float = 1e-8,
    atol: float = 1e-10,
    linear_solver: str = "spsolve",
    linear_maxiter: int | None = None,
    linear_tol: float | None = None,
    linear_preconditioner=None,
    line_search: bool = False,
    max_ls: int = 10,
    ls_c: float = 1e-4,
    n_steps: int = 1,
    jacobian_pattern=None,
    u0=None,
):
    """
    Convenience wrapper: build NonlinearAnalysis and run NewtonSolveRunner.
    """
    analysis = NonlinearAnalysis(
        space=space,
        residual_form=residual_form,
        params=params,
        base_external_vector=base_external_vector,
        dirichlet=dirichlet,
        dtype=dtype,
        jacobian_pattern=jacobian_pattern,
    )
    cfg = NewtonLoopConfig(
        maxiter=maxiter,
        tol=tol,
        atol=atol,
        linear_solver=linear_solver,
        linear_maxiter=linear_maxiter,
        linear_tol=linear_tol,
        linear_preconditioner=linear_preconditioner,
        line_search=line_search,
        max_ls=max_ls,
        ls_c=ls_c,
        n_steps=n_steps,
    )
    runner = NewtonSolveRunner(analysis, cfg)
    u0_use = jnp.zeros(space.n_dofs, dtype=dtype) if u0 is None else u0
    u, history = runner.run(u0=u0_use)
    return u, history


@dataclass
class LinearAnalysis:
    """
    Bundle linear problem data for a single solve or a load-scaled sequence.

    The matrix is assembled once from ``bilinear_form``; the RHS is scaled
    by the load factor.
    """

    space: Any
    bilinear_form: Any
    params: Any
    base_rhs_vector: Any
    dirichlet: tuple | None = None
    pattern: Any | None = None
    dtype: Any = jnp.float64

    def assemble_matrix(self):
        return self.space.assemble_bilinear_form(
            self.bilinear_form,
            params=self.params,
            pattern=self.pattern,
        )

    def rhs_for_load(self, load_factor: float):
        return jnp.asarray(load_factor * self.base_rhs_vector, dtype=self.dtype)


@dataclass
class LinearSolveConfig:
    """
    Control parameters for the linear solve with optional load scaling.
    """

    method: str = "spsolve"  # "spsolve" | "spdirect_solve_gpu" | "cg" | "cg_custom"
    tol: float = 1e-8
    maxiter: int | None = None
    preconditioner: Any | None = None


@dataclass
class LinearStepResult:
    """
    Result record for one linear solve step.

    Attributes
    ----------
    info : SolverResult
        Solver status and iteration metadata.
    solve_time : float
        Wall time for the solve section.
    u : Any
        Full solution vector (Dirichlet-expanded).
    """
    info: SolverResult
    solve_time: float
    u: Any


class LinearSolveRunner:
    """
    Solve linear systems for one or more load factors using a unified interface.
    """

    def __init__(self, analysis: LinearAnalysis, config: LinearSolveConfig):
        self.analysis = analysis
        self.config = config

    def run(
        self,
        *,
        step_callback: Callable[[LinearStepResult], None] | None = None,
        timer: "SectionTimer | None" = None,
        report_timing: bool = True
    ) -> tuple[np.ndarray | None, list[LinearStepResult]]:

        timer = timer or SectionTimer(hierarchical=True)
        # timer = timer or NullTimer()
        with timer.section("linear_run_total"):
            with timer.section("assemble_matrix"):
                K = self.analysis.assemble_matrix()

            with timer.section("build_rhs"):
                base_rhs = jnp.asarray(
                    self.analysis.base_rhs_vector, dtype=self.analysis.dtype
                )
                if self.analysis.dirichlet is not None:
                    K_ff, F_free_base, offset, free, dir_arr, dir_vals_arr = _condense_flux_dirichlet(
                        K, base_rhs, self.analysis.dirichlet
                    )
                    n_total = K.shape[0] if hasattr(K, "shape") else self.analysis.space.n_dofs
                else:
                    K_ff = K.to_csr()
                    F_free_base = np.asarray(base_rhs, dtype=float)
                    offset = None
                    free = None
                    dir_arr = dir_vals_arr = None
                    n_total = K_ff.shape[0]

                F_free = np.asarray(F_free_base, dtype=float)
                if offset is not None:
                    F_free = F_free - offset

            with timer.section(f"solve>{self.config.method}"):
                try:
                    if self.config.method == "spsolve":
                        u_free = spdirect_solve_cpu(K_ff, F_free)
                        lin_iters = 1
                        lin_conv = True
                        lin_res = None
                        info = SolverResult(
                            converged=True,
                            iters=lin_iters,
                            linear_iters=lin_iters,
                            linear_converged=lin_conv,
                            linear_residual=lin_res,
                            tol=self.config.tol,
                            stop_reason="converged",
                        )
                    elif self.config.method == "spdirect_solve_gpu":
                        u_free = spdirect_solve_gpu(K_ff, F_free)
                        lin_iters = 1
                        lin_conv = True
                        lin_res = None
                        info = SolverResult(
                            converged=True,
                            iters=lin_iters,
                            linear_iters=lin_iters,
                            linear_converged=lin_conv,
                            linear_residual=lin_res,
                            tol=self.config.tol,
                            stop_reason="converged",
                        )
                    elif self.config.method in ("cg", "cg_custom"):
                        coo = K_ff.tocoo()
                        A_cg = FluxSparseMatrix.from_bilinear(
                            (
                                jnp.asarray(coo.row, dtype=jnp.int32),
                                jnp.asarray(coo.col, dtype=jnp.int32),
                                jnp.asarray(coo.data),
                                K_ff.shape[0],
                            )
                        )
                        cg_solver = cg_solve_jax if self.config.method == "cg" else cg_solve
                        u_free, cg_info = cg_solver(
                            A_cg, jnp.asarray(F_free),
                            tol=self.config.tol,
                            maxiter=self.config.maxiter,
                            preconditioner=self.config.preconditioner
                        )
                        lin_iters = cg_info.get("iters")
                        lin_conv = bool(cg_info.get("converged", True))
                        lin_res = cg_info.get("residual_norm", cg_info.get("residual"))
                        info = SolverResult(
                            converged=lin_conv,
                            iters=int(lin_iters) if lin_iters is not None else 0,
                            linear_iters=int(lin_iters) if lin_iters is not None else None,
                            linear_converged=lin_conv,
                            linear_residual=float(lin_res) if lin_res is not None else None,
                            tol=self.config.tol,
                            stop_reason=("converged" if lin_conv else "linfail"),
                            nan_detected=bool(np.isnan(lin_res)) if lin_res is not None else False,
                        )
                    else:
                        raise ValueError(f"Unknown linear solve method: {self.config.method}")

                except Exception as e:  # pragma: no cover - defensive
                    exception = repr(e)
                    info = SolverResult(
                        converged=False,
                        iters=0,
                        linear_iters=None,
                        linear_converged=False,
                        linear_residual=None,
                        tol=self.config.tol,
                        stop_reason=f"exception: {exception}",
                        nan_detected=False,
                    )
                    # keep u as None to signal failure; caller should treat as Optional
                    result = LinearStepResult(info=info, solve_time=0.0, u=None)
                    if step_callback is not None:
                        step_callback(result)
                    # Report outside.
                    if report_timing:
                        timer.report(sort_by="total")
                    return None, [result], timer

            with timer.section("expand_dirichlet"):
                if free is None:
                    u_full = np.asarray(u_free, dtype=float)
                else:
                    u_full = expand_dirichlet_solution(
                        u_free, free, dir_arr, dir_vals_arr, n_total
                    )

            solve_key = f"solve>{self.config.method}"
            solve_time = timer._records.get(solve_key, [0.0])[-1]
            result = LinearStepResult(
                info=info, solve_time=solve_time, u=u_full
            )
            if step_callback is not None:
                step_callback(result)

        if report_timing:
            timer.report(sort_by="total")

        return u_full, [result]

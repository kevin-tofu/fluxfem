from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .contact import assemble_contact_operators
from .mortar_problem import (
    assemble_mortar_contact_problem,
    contact_matrix_nnz,
    dense_contact_operator_matrix,
)
from .mortar_multiplier import ContactMultiplierSpace


@dataclass(frozen=True)
class ContactMethodSpec:
    """One contact formulation to assemble and evaluate on a common dataset."""

    name: str
    enforcement: str
    formulation: str | None = None
    multiplier: ContactMultiplierSpace | None = None
    params: Any | None = None
    rho: float = 0.0
    sparse: bool = False
    backend: str | None = None
    master_dofs: npt.ArrayLike | None = None
    slave_dofs: npt.ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactMethodResult:
    """Raw result for one compared contact method."""

    method: ContactMethodSpec
    ok: bool
    elapsed_seconds: float
    operators: Any | None = None
    solution: np.ndarray | None = None
    residual_norm: float | None = None
    error: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactMethodMetric:
    """Flat metric row suitable for JSON/CSV/table output."""

    method: str
    enforcement: str
    formulation: str | None
    ok: bool
    elapsed_seconds: float
    operator_shape: tuple[int, int] | None = None
    operator_nnz: int | None = None
    operator_norm: float | None = None
    symmetry_error: float | None = None
    multiplier_count: int | None = None
    fine_multiplier_count: int | None = None
    reduction_ratio: float | None = None
    rank_estimate: int | None = None
    rank_deficiency: int | None = None
    residual_norm: float | None = None
    reference_rel_error: float | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "enforcement": self.enforcement,
            "formulation": self.formulation,
            "ok": self.ok,
            "elapsed_seconds": self.elapsed_seconds,
            "operator_shape": self.operator_shape,
            "operator_nnz": self.operator_nnz,
            "operator_norm": self.operator_norm,
            "symmetry_error": self.symmetry_error,
            "multiplier_count": self.multiplier_count,
            "fine_multiplier_count": self.fine_multiplier_count,
            "reduction_ratio": self.reduction_ratio,
            "rank_estimate": self.rank_estimate,
            "rank_deficiency": self.rank_deficiency,
            "residual_norm": self.residual_norm,
            "reference_rel_error": self.reference_rel_error,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PrimalSolutionComparison:
    """One-to-one comparison between two primal solution vectors."""

    method: str
    reference: str
    abs_l2: float
    rel_l2: float
    max_abs: float
    comparable: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "reference": self.reference,
            "abs_l2": self.abs_l2,
            "rel_l2": self.rel_l2,
            "max_abs": self.max_abs,
            "comparable": self.comparable,
            "reason": self.reason,
        }


def compare_primal_solutions(
    solutions: Mapping[str, npt.ArrayLike | None],
    *,
    reference: str,
    atol: float = 1.0e-30,
) -> list[PrimalSolutionComparison]:
    """Compare only solution vectors that share the reference primal DOF layout."""

    ref_value = solutions.get(reference)
    if ref_value is None:
        raise ValueError(f"reference solution {reference!r} is missing.")
    ref = np.asarray(ref_value, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(ref)), float(atol))
    rows: list[PrimalSolutionComparison] = []
    for name, value in solutions.items():
        if value is None:
            rows.append(
                PrimalSolutionComparison(
                    method=str(name),
                    reference=str(reference),
                    abs_l2=float("nan"),
                    rel_l2=float("nan"),
                    max_abs=float("nan"),
                    comparable=False,
                    reason="solution is unavailable",
                )
            )
            continue
        vec = np.asarray(value, dtype=float).reshape(-1)
        if vec.shape != ref.shape:
            rows.append(
                PrimalSolutionComparison(
                    method=str(name),
                    reference=str(reference),
                    abs_l2=float("nan"),
                    rel_l2=float("nan"),
                    max_abs=float("nan"),
                    comparable=False,
                    reason=f"shape mismatch: {tuple(vec.shape)} != {tuple(ref.shape)}",
                )
            )
            continue
        diff = vec - ref
        rows.append(
            PrimalSolutionComparison(
                method=str(name),
                reference=str(reference),
                abs_l2=float(np.linalg.norm(diff)),
                rel_l2=float(np.linalg.norm(diff) / denom),
                max_abs=float(np.max(np.abs(diff))) if diff.size else 0.0,
            )
        )
    return rows


def compare_contact_methods(
    contact: Any,
    methods: Sequence[ContactMethodSpec],
    *,
    stiffness: Any | None = None,
    load: npt.ArrayLike | None = None,
    master_dofs: npt.ArrayLike | None = None,
    slave_dofs: npt.ArrayLike | None = None,
    solve: bool = True,
    reference: str | None = None,
    normal_source: str = "master",
) -> tuple[list[ContactMethodMetric], list[ContactMethodResult], list[PrimalSolutionComparison]]:
    """Assemble contact formulations on one dataset and return comparable metric rows.

    Distribution-level comparisons are intentionally separate: only methods that
    produce primal vectors with the same DOF layout are passed to
    :func:`compare_primal_solutions`.
    """

    results: list[ContactMethodResult] = []
    metrics: list[ContactMethodMetric] = []
    solutions: dict[str, np.ndarray | None] = {}
    for method in methods:
        result = _evaluate_method(
            contact,
            method,
            stiffness=stiffness,
            load=load,
            master_dofs=master_dofs,
            slave_dofs=slave_dofs,
            solve=solve,
            normal_source=normal_source,
        )
        metric = _metric_from_result(result)
        results.append(result)
        metrics.append(metric)
        solutions[method.name] = result.solution

    comparisons: list[PrimalSolutionComparison] = []
    if reference is not None:
        if solutions.get(reference) is not None:
            comparisons = compare_primal_solutions(solutions, reference=reference)
            rel_by_name = {row.method: row.rel_l2 if row.comparable else None for row in comparisons}
            metrics = [
                ContactMethodMetric(
                    method=row.method,
                    enforcement=row.enforcement,
                    formulation=row.formulation,
                    ok=row.ok,
                    elapsed_seconds=row.elapsed_seconds,
                    operator_shape=row.operator_shape,
                    operator_nnz=row.operator_nnz,
                    operator_norm=row.operator_norm,
                    symmetry_error=row.symmetry_error,
                    multiplier_count=row.multiplier_count,
                    fine_multiplier_count=row.fine_multiplier_count,
                    reduction_ratio=row.reduction_ratio,
                    rank_estimate=row.rank_estimate,
                    rank_deficiency=row.rank_deficiency,
                    residual_norm=row.residual_norm,
                    reference_rel_error=rel_by_name.get(row.method),
                    error=row.error,
                    metadata=row.metadata,
                )
                for row in metrics
            ]
    return metrics, results, comparisons


def _evaluate_method(
    contact: Any,
    method: ContactMethodSpec,
    *,
    stiffness: Any | None,
    load: npt.ArrayLike | None,
    master_dofs: npt.ArrayLike | None,
    slave_dofs: npt.ArrayLike | None,
    solve: bool,
    normal_source: str,
) -> ContactMethodResult:
    started = time.perf_counter()
    try:
        ops = assemble_contact_operators(
            contact,
            enforcement=method.enforcement,
            formulation=method.formulation,
            multiplier=method.multiplier,
            rho=method.rho,
            params=method.params,
            backend=method.backend,
            sparse=method.sparse,
            normal_source=normal_source,
        )
        solution = None
        residual_norm = None
        solve_error = None
        if solve and stiffness is not None and load is not None:
            try:
                solution, residual_norm = _solve_contact_contribution(
                    ops,
                    stiffness=stiffness,
                    load=load,
                    master_dofs=method.master_dofs if method.master_dofs is not None else master_dofs,
                    slave_dofs=method.slave_dofs if method.slave_dofs is not None else slave_dofs,
                    metadata=method.metadata,
                )
            except Exception as exc:
                solve_error = f"solve {type(exc).__name__}: {exc}"
        diagnostics = dict(getattr(ops, "diagnostics", {}) or {})
        if solve_error is not None:
            diagnostics["solve_error"] = solve_error
        return ContactMethodResult(
            method=method,
            ok=True,
            elapsed_seconds=time.perf_counter() - started,
            operators=ops,
            solution=solution,
            residual_norm=residual_norm,
            error=solve_error,
            diagnostics=diagnostics,
        )
    except Exception as exc:  # pragma: no cover - exercised by optional methods.
        return ContactMethodResult(
            method=method,
            ok=False,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _solve_contact_contribution(
    ops: Any,
    *,
    stiffness: Any,
    load: npt.ArrayLike,
    master_dofs: npt.ArrayLike | None,
    slave_dofs: npt.ArrayLike | None,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    K = dense_contact_operator_matrix(stiffness, backend="numpy")
    if K is None:
        raise ValueError("stiffness is required for solving.")
    f = np.asarray(load, dtype=float).reshape(-1)
    if getattr(ops, "B", None) is not None:
        if master_dofs is None or slave_dofs is None:
            raise ValueError("master_dofs and slave_dofs are required for mortar solves.")
        problem = assemble_mortar_contact_problem(
            stiffness=stiffness,
            load=f,
            contact_pairs=[
                {
                    "name": str(metadata.get("pair_name", "contact")),
                    "operators": ops,
                    "master_dofs": master_dofs,
                    "slave_dofs": slave_dofs,
                }
            ],
            metadata=metadata,
        )
        solve_result = problem.solve_with_info()
        residual = problem.matrix @ solve_result.solution - problem.rhs
        n_primal = int(K.shape[0])
        return np.asarray(solve_result.solution[:n_primal], dtype=float), float(np.linalg.norm(np.asarray(residual)))
    if getattr(ops, "jacobian", None) is not None:
        Kc = dense_contact_operator_matrix(ops.jacobian, backend="numpy")
        if Kc is None:
            raise ValueError("assembled penalty/Nitsche jacobian is required for solving.")
        if Kc.shape != K.shape:
            raise ValueError(f"contact jacobian shape {Kc.shape} does not match stiffness shape {K.shape}.")
        system = K + Kc
        solution = np.linalg.lstsq(system, f, rcond=None)[0]
        residual = system @ solution - f
        return np.asarray(solution, dtype=float), float(np.linalg.norm(residual))
    raise ValueError("contact operators contain neither B nor jacobian.")


def _metric_from_result(result: ContactMethodResult) -> ContactMethodMetric:
    method = result.method
    if not result.ok or result.operators is None:
        return ContactMethodMetric(
            method=method.name,
            enforcement=method.enforcement,
            formulation=method.formulation,
            ok=False,
            elapsed_seconds=result.elapsed_seconds,
            residual_norm=result.residual_norm,
            error=result.error,
            metadata=method.metadata,
        )
    ops = result.operators
    matrix = getattr(ops, "B", None)
    if matrix is None:
        matrix = getattr(ops, "jacobian", None)
    dense = dense_contact_operator_matrix(matrix, backend="numpy")
    shape = None if dense is None else (int(dense.shape[0]), int(dense.shape[1]))
    symmetry_error = None
    if dense is not None and dense.ndim == 2 and dense.shape[0] == dense.shape[1]:
        denom = max(float(np.linalg.norm(dense)), 1.0e-30)
        symmetry_error = float(np.linalg.norm(dense - dense.T) / denom)
    rank_estimate = None
    rank_deficiency = None
    if dense is not None and dense.ndim == 2 and dense.size:
        rank_estimate = int(np.linalg.matrix_rank(dense))
        rank_deficiency = int(max(0, dense.shape[0] - rank_estimate))
    diagnostics = dict(result.diagnostics)
    fine = diagnostics.get("constraint_rows_before_reduction")
    after = diagnostics.get("constraint_rows_after_reduction")
    reduction_ratio = None
    if fine is not None and int(fine) > 0 and after is not None:
        reduction_ratio = float(after) / float(fine)
    return ContactMethodMetric(
        method=method.name,
        enforcement=getattr(ops, "enforcement", method.enforcement),
        formulation=getattr(ops, "formulation", method.formulation),
        ok=True,
        elapsed_seconds=result.elapsed_seconds,
        operator_shape=shape,
        operator_nnz=contact_matrix_nnz(matrix),
        operator_norm=None if dense is None else float(np.linalg.norm(dense)),
        symmetry_error=symmetry_error,
        multiplier_count=None if getattr(ops, "B", None) is None else int(ops.B.shape[0]),
        fine_multiplier_count=None if fine is None else int(fine),
        reduction_ratio=reduction_ratio,
        rank_estimate=rank_estimate,
        rank_deficiency=rank_deficiency,
        residual_norm=result.residual_norm,
        error=result.error,
        metadata={**method.metadata, **diagnostics},
    )

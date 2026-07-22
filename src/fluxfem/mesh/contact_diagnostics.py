from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .mortar_problem import dense_contact_operator_matrix


@dataclass(frozen=True)
class ContactConstraintDiagnostics:
    """Numerical diagnostics for a mortar constraint matrix."""

    n_rows: int
    n_cols: int
    nnz: int
    zero_row_count: int
    row_norm_min: float
    row_norm_max: float
    row_norm_mean: float
    estimated_rank: int
    rank_deficiency: int
    condition_number: float
    singular_values: np.ndarray
    singular_value_count: int
    rtol: float
    atol: float


@dataclass(frozen=True)
class ContactConstraintQualityIssue:
    """One opt-in quality-policy finding for a mortar constraint matrix."""

    check: str
    severity: str
    message: str
    value: float | int
    threshold: float | int
    hint: str = ""


@dataclass(frozen=True)
class ContactConstraintQualityReport:
    """Pass/warn/fail quality report derived from contact constraint diagnostics."""

    diagnostics: ContactConstraintDiagnostics
    status: str
    issues: tuple[ContactConstraintQualityIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    @property
    def warnings(self) -> tuple[ContactConstraintQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warn")

    @property
    def failures(self) -> tuple[ContactConstraintQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "fail")


def contact_constraint_matrix_diagnostics(
    B: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-14,
    max_singular_values: int | None = 20,
) -> ContactConstraintDiagnostics:
    """Return row-scaling and rank diagnostics for a mortar matrix."""

    B_np = np.asarray(dense_contact_operator_matrix(B, backend="numpy"), dtype=float)
    if B_np.ndim != 2:
        raise ValueError("constraint diagnostics require a rank-2 B matrix.")
    n_rows, n_cols = (int(B_np.shape[0]), int(B_np.shape[1]))
    row_norms = np.linalg.norm(B_np, axis=1) if n_rows else np.zeros((0,), dtype=float)
    zero_rows = row_norms <= float(atol)
    if n_rows and n_cols:
        singular_all = np.linalg.svd(B_np, compute_uv=False)
    else:
        singular_all = np.zeros((0,), dtype=float)
    if singular_all.size:
        threshold = max(float(atol), float(rtol) * float(singular_all[0]))
        estimated_rank = int(np.count_nonzero(singular_all > threshold))
        condition_number = (
            float(singular_all[0] / singular_all[estimated_rank - 1])
            if estimated_rank > 0
            else float("inf")
        )
    else:
        estimated_rank = 0
        condition_number = float("inf")
    if max_singular_values is None:
        singular_values = singular_all
    else:
        singular_values = singular_all[: max(0, int(max_singular_values))]
    rank_deficiency = max(0, min(n_rows, n_cols) - int(estimated_rank))
    nnz = int(np.count_nonzero(np.abs(B_np) > float(atol)))
    return ContactConstraintDiagnostics(
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=nnz,
        zero_row_count=int(np.count_nonzero(zero_rows)),
        row_norm_min=float(np.min(row_norms)) if row_norms.size else 0.0,
        row_norm_max=float(np.max(row_norms)) if row_norms.size else 0.0,
        row_norm_mean=float(np.mean(row_norms)) if row_norms.size else 0.0,
        estimated_rank=estimated_rank,
        rank_deficiency=rank_deficiency,
        condition_number=condition_number,
        singular_values=np.asarray(singular_values, dtype=float),
        singular_value_count=int(singular_all.size),
        rtol=float(rtol),
        atol=float(atol),
    )


def _constraint_quality_severity(value: str, *, name: str) -> str:
    severity = str(value).lower()
    if severity not in {"warn", "fail"}:
        raise ValueError(f"{name} must be 'warn' or 'fail'.")
    return severity


def _constraint_quality_status(issues: Sequence[ContactConstraintQualityIssue]) -> str:
    order = {"pass": 0, "warn": 1, "fail": 2}
    status = "pass"
    for issue in issues:
        if order[issue.severity] > order[status]:
            status = issue.severity
    return status


def _constraint_quality_hint(check: str) -> str:
    if check == "zero_rows":
        return (
            "Inspect active contact facets, empty overlap regions, and supermesh clipping; "
            "zero rows usually mean a constraint row has no contributing overlap."
        )
    if check == "rank_deficiency":
        return (
            "Try a coarser multiplier space such as coarse_p0/coarse_p1, or apply "
            "QR/SVD-style row reduction before solving the KKT system."
        )
    if check == "condition_number":
        return (
            "Inspect row scaling and material/penalty scaling; use the block-scaled "
            "KKT solver diagnostics to confirm the scaled system is balanced."
        )
    if check == "row_norm_min":
        return (
            "Inspect very small overlap patches, nearly degenerate facets, or overly "
            "tight clipping tolerances that can create weak constraint rows."
        )
    return "Inspect the contact geometry, multiplier space, and KKT scaling diagnostics."


def assess_contact_constraint_quality(
    B_or_diagnostics: Any,
    *,
    max_zero_rows: int = 0,
    zero_row_severity: str = "fail",
    max_rank_deficiency: int = 0,
    rank_deficiency_severity: str = "fail",
    max_condition_number: float | None = None,
    condition_number_severity: str = "warn",
    min_row_norm: float | None = None,
    row_norm_severity: str = "warn",
    rtol: float = 1e-10,
    atol: float = 1e-14,
    max_singular_values: int | None = 20,
) -> ContactConstraintQualityReport:
    """
    Evaluate opt-in quality thresholds for a mortar constraint matrix.

    Defaults fail on zero rows and rank deficiency, while condition-number and
    row-norm checks are enabled only when their thresholds are supplied.
    """

    if isinstance(B_or_diagnostics, ContactConstraintDiagnostics):
        diag = B_or_diagnostics
    else:
        diag = contact_constraint_matrix_diagnostics(
            B_or_diagnostics,
            rtol=rtol,
            atol=atol,
            max_singular_values=max_singular_values,
        )

    zero_sev = _constraint_quality_severity(zero_row_severity, name="zero_row_severity")
    rank_sev = _constraint_quality_severity(rank_deficiency_severity, name="rank_deficiency_severity")
    cond_sev = _constraint_quality_severity(condition_number_severity, name="condition_number_severity")
    row_sev = _constraint_quality_severity(row_norm_severity, name="row_norm_severity")
    issues: list[ContactConstraintQualityIssue] = []

    max_zero = int(max_zero_rows)
    if max_zero < 0:
        raise ValueError("max_zero_rows must be non-negative.")
    if int(diag.zero_row_count) > max_zero:
        issues.append(
            ContactConstraintQualityIssue(
                check="zero_rows",
                severity=zero_sev,
                message="constraint matrix contains zero rows",
                value=int(diag.zero_row_count),
                threshold=max_zero,
                hint=_constraint_quality_hint("zero_rows"),
            )
        )

    max_rank = int(max_rank_deficiency)
    if max_rank < 0:
        raise ValueError("max_rank_deficiency must be non-negative.")
    if int(diag.rank_deficiency) > max_rank:
        issues.append(
            ContactConstraintQualityIssue(
                check="rank_deficiency",
                severity=rank_sev,
                message="constraint matrix is rank deficient beyond the configured tolerance",
                value=int(diag.rank_deficiency),
                threshold=max_rank,
                hint=_constraint_quality_hint("rank_deficiency"),
            )
        )

    if max_condition_number is not None:
        cond_threshold = float(max_condition_number)
        if cond_threshold <= 0.0:
            raise ValueError("max_condition_number must be positive when provided.")
        if not np.isfinite(float(diag.condition_number)) or float(diag.condition_number) > cond_threshold:
            issues.append(
                ContactConstraintQualityIssue(
                    check="condition_number",
                    severity=cond_sev,
                    message="constraint matrix condition number exceeds the configured threshold",
                    value=float(diag.condition_number),
                    threshold=cond_threshold,
                    hint=_constraint_quality_hint("condition_number"),
                )
            )

    if min_row_norm is not None:
        row_threshold = float(min_row_norm)
        if row_threshold < 0.0:
            raise ValueError("min_row_norm must be non-negative when provided.")
        if float(diag.row_norm_min) < row_threshold:
            issues.append(
                ContactConstraintQualityIssue(
                    check="row_norm_min",
                    severity=row_sev,
                    message="constraint matrix row norm is below the configured threshold",
                    value=float(diag.row_norm_min),
                    threshold=row_threshold,
                    hint=_constraint_quality_hint("row_norm_min"),
                )
            )

    return ContactConstraintQualityReport(
        diagnostics=diag,
        status=_constraint_quality_status(issues),
        issues=tuple(issues),
    )


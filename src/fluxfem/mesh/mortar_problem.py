from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .contact_diagnostics import ContactConstraintDiagnostics
    from .contact_kkt_solver import ContactKKTSolveConfig, ContactKKTSolveResult


@dataclass(frozen=True)
class MortarContactProblemPair:
    """One mortar contact pair embedded into a global primal system."""

    name: str
    operators: Any
    master_dofs: np.ndarray
    slave_dofs: np.ndarray
    row_offset: int = 0

    @property
    def multiplier_count(self) -> int:
        if self.operators.B is None:
            return 0
        return int(self.operators.B.shape[0])

    @property
    def fine_multiplier_count(self) -> int | None:
        return self.operators.diagnostics.get("constraint_rows_before_reduction")

    @property
    def reduction_ratio(self) -> float | None:
        fine = self.fine_multiplier_count
        if fine is None or int(fine) == 0:
            return None
        return float(self.multiplier_count) / float(fine)

    def constraint_diagnostics(self, **kwargs) -> "ContactConstraintDiagnostics":
        return self.operators.constraint_diagnostics(**kwargs)

    def diagnostics(self, **kwargs) -> dict[str, Any]:
        diag = self.constraint_diagnostics(**kwargs)
        return {
            "name": self.name,
            "row_offset": int(self.row_offset),
            "multiplier_count": self.multiplier_count,
            "fine_multiplier_count": self.fine_multiplier_count,
            "reduction_ratio": self.reduction_ratio,
            "multiplier_family": str(getattr(self.operators.multiplier, "family", "")),
            "constraint_reduction": self.operators.diagnostics.get("constraint_reduction", "none"),
            "constraint_scaling": self.operators.diagnostics.get("constraint_scaling", "none"),
            "coupling_nnz": int(contact_matrix_nnz(self.operators.B)),
            "rank_estimate": int(diag.estimated_rank),
            "rank_deficiency": int(diag.rank_deficiency),
            "row_norm_min": float(diag.row_norm_min),
            "row_norm_max": float(diag.row_norm_max),
            "row_norm_mean": float(diag.row_norm_mean),
            "zero_row_count": int(diag.zero_row_count),
            "condition_number": float(diag.condition_number),
        }


@dataclass(frozen=True)
class MortarContactProblem:
    """Assembled mortar KKT problem view with embedded contact pairs."""

    stiffness: Any
    coupling_matrix: Any
    load: np.ndarray
    contact_pairs: tuple[MortarContactProblemPair, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def multiplier_count(self) -> int:
        return int(self.coupling_matrix.shape[0])

    @property
    def matrix(self):
        return contact_kkt_matrix_from_blocks(self.stiffness, self.coupling_matrix)

    @property
    def rhs(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.load, dtype=float), np.zeros(self.multiplier_count, dtype=float)])

    def constraint_diagnostics(self, *, include_pairs: bool = False, **kwargs) -> dict[str, Any]:
        from .contact_diagnostics import contact_constraint_matrix_diagnostics

        diag = contact_constraint_matrix_diagnostics(self.coupling_matrix, **kwargs)
        result: dict[str, Any] = {
            "n_rows": int(diag.n_rows),
            "n_cols": int(diag.n_cols),
            "nnz": int(diag.nnz),
            "zero_row_count": int(diag.zero_row_count),
            "row_norm_min": float(diag.row_norm_min),
            "row_norm_max": float(diag.row_norm_max),
            "row_norm_mean": float(diag.row_norm_mean),
            "estimated_rank": int(diag.estimated_rank),
            "rank_deficiency": int(diag.rank_deficiency),
            "condition_number": float(diag.condition_number),
            "singular_values": np.asarray(diag.singular_values, dtype=float),
            "singular_value_count": int(diag.singular_value_count),
            "contact_pair_count": len(self.contact_pairs),
            "multiplier_count": self.multiplier_count,
        }
        fine_counts = [pair.fine_multiplier_count for pair in self.contact_pairs if pair.fine_multiplier_count is not None]
        if fine_counts:
            fine_total = int(sum(int(value) for value in fine_counts))
            result["fine_multiplier_count"] = fine_total
            result["reduction_ratio"] = float(self.multiplier_count) / float(fine_total) if fine_total else None
        else:
            result["fine_multiplier_count"] = None
            result["reduction_ratio"] = None
        if include_pairs:
            result["contact_pairs"] = [pair.diagnostics(**kwargs) for pair in self.contact_pairs]
        return result

    def solve_with_info(self, config: "ContactKKTSolveConfig | None" = None) -> "ContactKKTSolveResult":
        from .contact_kkt_solver import ContactKKTSolveConfig, solve_contact_kkt_with_info

        n_primal = int(self.stiffness.shape[0])
        cfg = ContactKKTSolveConfig(backend="numpy", numpy_solver="block_scaled", n_primal=n_primal)
        if config is not None:
            cfg = replace(config, n_primal=n_primal if config.n_primal is None else config.n_primal)
        return solve_contact_kkt_with_info(self.matrix, self.rhs, config=cfg)

    def solve(self, config: "ContactKKTSolveConfig | None" = None) -> tuple[np.ndarray, np.ndarray]:
        solution = np.asarray(self.solve_with_info(config=config).solution)
        split = int(self.stiffness.shape[0])
        return solution[:split], solution[split:]


def dense_contact_operator_matrix(matrix: Any, *, backend: str | None = None):
    if matrix is None:
        return None
    if backend == "jax":
        import jax.numpy as jnp

        if hasattr(matrix, "toarray"):
            return jnp.asarray(matrix.toarray())
        return (
            jnp.asarray(matrix.to_dense())
            if hasattr(matrix, "to_dense")
            else jnp.asarray(matrix)
        )
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return (
        np.asarray(matrix.to_dense(), dtype=float)
        if hasattr(matrix, "to_dense")
        else np.asarray(matrix, dtype=float)
    )


def contact_matrix_nnz(matrix: Any) -> int:
    if matrix is None:
        return 0
    if hasattr(matrix, "nnz"):
        return int(matrix.nnz)
    return int(np.count_nonzero(np.asarray(matrix)))


def contact_to_csr_matrix(matrix: Any):
    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover
        return np.asarray(dense_contact_operator_matrix(matrix, backend="numpy"), dtype=float)
    if sp.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "to_coo"):
        rows, cols, data, shape = matrix.to_coo()
        if not isinstance(shape, tuple):
            shape = (int(shape), int(shape))
        return sp.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def contact_vstack(matrices: Sequence[Any]):
    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover
        return np.vstack([np.asarray(mat, dtype=float) for mat in matrices]) if matrices else np.zeros((0, 0))
    return sp.vstack([contact_to_csr_matrix(mat) for mat in matrices], format="csr")


def contact_kkt_matrix_from_blocks(stiffness: Any, coupling_matrix: Any):
    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover
        K = np.asarray(stiffness, dtype=float)
        B = np.asarray(coupling_matrix, dtype=float)
        return np.block([[K, B.T], [B, np.zeros((B.shape[0], B.shape[0]), dtype=float)]])
    K = contact_to_csr_matrix(stiffness)
    B = contact_to_csr_matrix(coupling_matrix)
    zero = sp.csr_matrix((B.shape[0], B.shape[0]), dtype=K.dtype)
    return sp.bmat([[K, B.T], [B, zero]], format="csr")


def embed_contact_pair_B(
    ops: Any,
    *,
    master_dofs: npt.ArrayLike,
    slave_dofs: npt.ArrayLike,
    n_primal: int,
):
    if ops.B is None:
        raise ValueError("mortar contact pair operators must include B.")
    B_local = contact_to_csr_matrix(ops.B)
    master = np.asarray(master_dofs, dtype=int).reshape(-1)
    slave = np.asarray(slave_dofs, dtype=int).reshape(-1)
    n_master = int(master.shape[0])
    n_slave = int(slave.shape[0])
    if B_local.shape[1] != n_master + n_slave:
        raise ValueError("master_dofs/slave_dofs length must match ops.B columns.")
    if np.any(master < 0) or np.any(slave < 0):
        raise ValueError("contact pair dof indices must be non-negative.")
    if master.size and int(np.max(master)) >= int(n_primal):
        raise ValueError("master_dofs contain indices outside the primal system.")
    if slave.size and int(np.max(slave)) >= int(n_primal):
        raise ValueError("slave_dofs contain indices outside the primal system.")
    global_cols = np.concatenate([master, slave])
    coo = B_local.tocoo()
    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover
        dense = np.zeros((B_local.shape[0], int(n_primal)), dtype=float)
        dense[coo.row, global_cols[coo.col]] += coo.data
        return dense
    return sp.coo_matrix(
        (coo.data, (coo.row, global_cols[coo.col])),
        shape=(B_local.shape[0], int(n_primal)),
    ).tocsr()


def assemble_mortar_contact_problem(
    *,
    stiffness: Any,
    load: npt.ArrayLike,
    contact_pairs: Sequence[Mapping[str, Any] | MortarContactProblemPair],
    metadata: Mapping[str, Any] | None = None,
) -> MortarContactProblem:
    """Assemble a global mortar KKT problem from embedded contact pairs."""

    n_primal = int(stiffness.shape[0])
    load_np = np.asarray(load, dtype=float).reshape(-1)
    if load_np.shape[0] != n_primal:
        raise ValueError("load length must match stiffness.shape[0].")
    embedded_blocks: list[Any] = []
    resolved_pairs: list[MortarContactProblemPair] = []
    row_offset = 0
    for idx, pair in enumerate(contact_pairs):
        if isinstance(pair, MortarContactProblemPair):
            ops = pair.operators
            master = pair.master_dofs
            slave = pair.slave_dofs
            name = pair.name
        else:
            ops = pair.get("operators", pair.get("ops"))
            if ops is None or not hasattr(ops, "B"):
                raise TypeError("contact_pairs entries require mortar operators under 'operators' or 'ops'.")
            master = pair["master_dofs"]
            slave = pair["slave_dofs"]
            name = str(pair.get("name", f"contact-{idx}"))
        block = embed_contact_pair_B(
            ops,
            master_dofs=master,
            slave_dofs=slave,
            n_primal=n_primal,
        )
        embedded_blocks.append(block)
        resolved_pairs.append(
            MortarContactProblemPair(
                name=name,
                operators=ops,
                master_dofs=np.asarray(master, dtype=int).reshape(-1),
                slave_dofs=np.asarray(slave, dtype=int).reshape(-1),
                row_offset=row_offset,
            )
        )
        row_offset += int(block.shape[0])
    if embedded_blocks:
        B_global = contact_vstack(embedded_blocks)
    else:
        try:
            import scipy.sparse as sp
        except Exception:  # pragma: no cover
            B_global = np.zeros((0, n_primal), dtype=float)
        else:
            B_global = sp.csr_matrix((0, n_primal), dtype=float)
    return MortarContactProblem(
        stiffness=contact_to_csr_matrix(stiffness),
        coupling_matrix=B_global,
        load=load_np,
        contact_pairs=tuple(resolved_pairs),
        metadata={} if metadata is None else dict(metadata),
    )

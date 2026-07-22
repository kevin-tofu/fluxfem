from __future__ import annotations

import numpy as np

from .mortar_multiplier import ContactMultiplierSpace


def p0_reduction_matrix_from_facets(facet_conn: np.ndarray, n_nodes: int) -> np.ndarray:
    facets = np.asarray(facet_conn, dtype=int)
    S = np.zeros((int(facets.shape[0]), int(n_nodes)), dtype=float)
    for f, nodes in enumerate(facets):
        S[int(f), np.asarray(nodes, dtype=int)] = 1.0
    return S


def p0_patch_group_matrix(patch_ids: np.ndarray, n_rows: int) -> np.ndarray:
    patches = np.asarray(patch_ids, dtype=int).reshape(-1)
    if int(patches.size) != int(n_rows):
        raise ValueError("coarse_patch_ids must have one entry per fine P0 multiplier row.")
    if np.any(patches < 0):
        raise ValueError("coarse_patch_ids must not contain negative ids.")
    unique = np.unique(patches)
    row_of_patch = {int(patch): i for i, patch in enumerate(unique.tolist())}
    P = np.zeros((int(unique.size), int(n_rows)), dtype=float)
    for fine_row, patch in enumerate(patches.tolist()):
        P[row_of_patch[int(patch)], int(fine_row)] = 1.0
    return P


def apply_integrated_coarse_p0_groups(B_a, B_b, patch_ids: np.ndarray | None, *, backend: str):
    if patch_ids is None:
        return B_a, B_b
    P_np = p0_patch_group_matrix(patch_ids, int(B_a.shape[0]))
    if backend == "jax":
        import jax.numpy as jnp

        P = jnp.asarray(P_np)
    else:
        P = P_np
    return P @ B_a, P @ B_b


def expand_scalar_constraint_dense(B_scalar, *, value_dim: int, backend: str):
    vd = int(value_dim)
    if vd <= 1:
        return B_scalar
    B_np = np.asarray(B_scalar, dtype=float)
    out = np.zeros((vd * B_np.shape[0], vd * B_np.shape[1]), dtype=B_np.dtype)
    for comp in range(vd):
        out[comp::vd, comp::vd] = B_np
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(out)
    return out


def expand_scalar_constraint_coo(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    value_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    vd = int(value_dim)
    if vd <= 1:
        return rows, cols, data, int(n_rows), int(n_cols)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    data = np.asarray(data, dtype=float)
    rows_exp = np.concatenate([vd * rows + comp for comp in range(vd)], axis=0)
    cols_exp = np.concatenate([vd * cols + comp for comp in range(vd)], axis=0)
    data_exp = np.concatenate([data for _ in range(vd)], axis=0)
    return rows_exp, cols_exp, data_exp, vd * int(n_rows), vd * int(n_cols)


def dual_nodal_blocks_from_dense(M_aa, M_ab, *, backend: str):
    """Build master-side dual nodal mortar blocks."""

    if int(M_aa.shape[0]) != int(M_aa.shape[1]):
        raise ValueError("dual_nodal requires a square master-side nodal coupling block.")
    if int(M_ab.shape[0]) != int(M_aa.shape[0]):
        raise ValueError("dual_nodal requires compatible master/slave coupling row counts.")
    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np
    B_a = xp.eye(int(M_aa.shape[0]), dtype=M_aa.dtype)
    B_b = xp.linalg.pinv(M_aa) @ M_ab
    return B_a, B_b


def dense_to_coo_entries(mat: np.ndarray, *, tol: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(mat, dtype=float)
    if tol > 0.0:
        rows, cols = np.nonzero(np.abs(arr) > float(tol))
    else:
        rows, cols = np.nonzero(arr)
    return rows.astype(int), cols.astype(int), arr[rows, cols].astype(float)


def coarse_row_projection_from_rank(B, rank: int, *, backend: str):
    if int(rank) <= 0:
        raise ValueError("coarse_rank must be positive.")
    max_rank = min(int(B.shape[0]), int(B.shape[1]))
    if int(rank) > max_rank:
        raise ValueError("coarse_rank cannot exceed min(B.shape).")
    if backend == "jax":
        import jax.numpy as jnp

        q, _ = jnp.linalg.qr(B, mode="reduced")
        return q[:, : int(rank)].T
    q, _ = np.linalg.qr(np.asarray(B), mode="reduced")
    return q[:, : int(rank)].T


def coarse_row_projection_from_svd(
    B,
    *,
    energy_tol: float,
    rtol: float,
    max_rank: int | None,
    backend: str,
):
    if backend == "jax":
        import jax.numpy as jnp

        u, s, _ = jnp.linalg.svd(B, full_matrices=False)
        s_np = np.asarray(s, dtype=float)
        xp = jnp
    else:
        u_np, s_np, _ = np.linalg.svd(np.asarray(B, dtype=float), full_matrices=False)
        u = u_np
        xp = np
    if s_np.size == 0:
        raise ValueError("Cannot build a coarse mortar projection from an empty B matrix.")
    total = float(np.sum(s_np**2))
    if total <= 0.0:
        rank_energy = 1
    else:
        cumulative = np.cumsum(s_np**2) / total
        rank_energy = int(np.searchsorted(cumulative, float(energy_tol), side="left") + 1)
    threshold = float(rtol) * float(s_np[0]) if s_np.size else 0.0
    rank_numeric = int(np.count_nonzero(s_np > threshold)) if threshold > 0.0 else int(s_np.size)
    rank = max(1, min(rank_energy, rank_numeric if rank_numeric > 0 else 1))
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    rank = min(rank, int(u.shape[1]))
    return xp.asarray(u[:, :rank]).T


def constraint_row_ids_from_pivoted_qr(
    B: np.ndarray,
    *,
    rtol: float,
    max_rank: int | None,
) -> np.ndarray:
    """Return independent row ids selected by pivoted QR of ``B.T``."""

    B_np = np.asarray(B, dtype=float)
    if B_np.ndim != 2:
        raise ValueError("algebraic_qr reduction requires a rank-2 constraint matrix.")
    n_rows, n_cols = B_np.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros((0,), dtype=int)
    try:
        from scipy.linalg import qr
    except Exception as exc:
        raise ImportError("algebraic_qr mortar reduction requires scipy.linalg.qr with pivoting.") from exc

    _q, r, pivots = qr(B_np.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        rank = 0
    else:
        tol = float(rtol)
        threshold = tol * float(diagonal[0]) if tol < 1.0 else tol
        rank = int(np.count_nonzero(diagonal > threshold))
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    rank = max(0, min(rank, int(pivots.shape[0])))
    return np.sort(np.asarray(pivots[:rank], dtype=int))


def expanded_constraint_patch_ids(patch_ids: np.ndarray, n_rows: int, value_dim: int) -> np.ndarray:
    patch_np = np.asarray(patch_ids, dtype=int).reshape(-1)
    if patch_np.size == int(n_rows):
        return patch_np
    vd = int(value_dim)
    if vd > 1 and patch_np.size * vd == int(n_rows):
        return np.repeat(patch_np, vd)
    raise ValueError("patch_qr requires one patch id per constraint row before or after value_dim expansion.")


def constraint_row_ids_from_patch_qr(
    B: np.ndarray,
    patch_ids: np.ndarray,
    *,
    rtol: float,
    max_rank: int | None,
    value_dim: int,
) -> np.ndarray:
    """Return independent row ids selected by pivoted QR inside each row patch."""

    B_np = np.asarray(B, dtype=float)
    if B_np.ndim != 2:
        raise ValueError("patch_qr reduction requires a rank-2 constraint matrix.")
    n_rows, n_cols = B_np.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros((0,), dtype=int)
    patch_np = expanded_constraint_patch_ids(patch_ids, n_rows, value_dim)
    selected: list[int] = []
    remaining = None if max_rank is None else int(max_rank)
    for patch in np.unique(patch_np):
        if remaining is not None and remaining <= 0:
            break
        rows = np.flatnonzero(patch_np == int(patch))
        if rows.size == 0:
            continue
        local_max = None if remaining is None else min(remaining, int(rows.size))
        local_ids = constraint_row_ids_from_pivoted_qr(
            B_np[rows, :],
            rtol=rtol,
            max_rank=local_max,
        )
        chosen = rows[local_ids]
        selected.extend(int(row) for row in chosen.tolist())
        if remaining is not None:
            remaining -= int(chosen.size)
    return np.asarray(sorted(selected), dtype=int)


def apply_coarse_mortar_projection(B_a, B_b, multiplier: ContactMultiplierSpace, *, backend: str):
    projection = multiplier.coarse_projection
    rank = multiplier.coarse_rank
    mode = None if multiplier.coarse_mode is None else str(multiplier.coarse_mode).lower()
    if projection is None and rank is None and mode is None:
        return B_a, B_b
    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np
    B = xp.concatenate([B_a, -B_b], axis=1)
    if mode in {"algebraic_qr", "row_qr", "pivoted_qr"}:
        row_ids = constraint_row_ids_from_pivoted_qr(
            np.asarray(B, dtype=float),
            rtol=1e-10 if multiplier.coarse_rtol is None else float(multiplier.coarse_rtol),
            max_rank=multiplier.coarse_max_rank,
        )
        B_coarse = xp.asarray(np.asarray(B, dtype=float)[row_ids, :])
        n_a = int(B_a.shape[1])
        return B_coarse[:, :n_a], -B_coarse[:, n_a:]
    if mode == "patch_qr":
        if multiplier.coarse_patch_ids is None:
            raise ValueError("patch_qr mortar reduction requires coarse_patch_ids.")
        row_ids = constraint_row_ids_from_patch_qr(
            np.asarray(B, dtype=float),
            np.asarray(multiplier.coarse_patch_ids, dtype=int),
            rtol=1e-10 if multiplier.coarse_rtol is None else float(multiplier.coarse_rtol),
            max_rank=multiplier.coarse_max_rank,
            value_dim=int(multiplier.value_dim),
        )
        B_coarse = xp.asarray(np.asarray(B, dtype=float)[row_ids, :])
        n_a = int(B_a.shape[1])
        return B_coarse[:, :n_a], -B_coarse[:, n_a:]
    if projection is not None:
        P = xp.asarray(projection)
        if int(P.shape[1]) != int(B.shape[0]):
            raise ValueError("coarse_projection must have shape (n_coarse, n_multiplier_rows).")
    elif mode in {"svd", "auto"} and rank is None:
        P = coarse_row_projection_from_svd(
            B,
            energy_tol=0.999 if multiplier.coarse_energy_tol is None else float(multiplier.coarse_energy_tol),
            rtol=1e-10 if multiplier.coarse_rtol is None else float(multiplier.coarse_rtol),
            max_rank=multiplier.coarse_max_rank,
            backend=backend,
        )
    else:
        rank_eff = int(rank) if rank is not None else int(multiplier.coarse_max_rank or min(B.shape))
        P = coarse_row_projection_from_rank(B, rank_eff, backend=backend)
    B_coarse = P @ B
    n_a = int(B_a.shape[1])
    return B_coarse[:, :n_a], -B_coarse[:, n_a:]


def apply_constraint_row_scaling(B_a, B_b, scaling: str, *, backend: str):
    scaling_key = str(scaling).lower()
    if scaling_key == "none":
        return B_a, B_b
    if scaling_key != "l2":
        raise ValueError("constraint scaling must be 'none' or 'l2'.")
    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np
    B = xp.concatenate([B_a, -B_b], axis=1)
    row_norms = xp.linalg.norm(B, axis=1)
    scale = xp.where(row_norms > 0.0, 1.0 / row_norms, 1.0)
    B_scaled = scale[:, None] * B
    n_a = int(B_a.shape[1])
    return B_scaled[:, :n_a], -B_scaled[:, n_a:]


def is_patch_qr_multiplier(multiplier: ContactMultiplierSpace) -> bool:
    return str(multiplier.coarse_mode).lower() == "patch_qr"

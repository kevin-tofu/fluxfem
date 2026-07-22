from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING
import warnings

import numpy as np
import numpy.typing as npt

try:
    from .._runtime_warn import warn_float32_assembly_once
except Exception:  # pragma: no cover
    _WARNED_FLOAT32_CONTACT_ASSEMBLY = False

    def warn_float32_assembly_once(*, context: str = "assembly") -> None:
        global _WARNED_FLOAT32_CONTACT_ASSEMBLY
        if _WARNED_FLOAT32_CONTACT_ASSEMBLY:
            return
        try:
            import jax
        except Exception:
            return
        if bool(jax.config.read("jax_enable_x64")):
            return
        _WARNED_FLOAT32_CONTACT_ASSEMBLY = True
        warnings.warn(
            "Running in float32 mode (x64 disabled). "
            f"{context} can suffer from residual/conditioning degradation; "
            "use x64 for reliable diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
from .contact_interface import (
    assemble_contact_interface_jacobian as _assemble_contact_interface_jacobian,
    assemble_contact_interface_residual as _assemble_contact_interface_residual,
    assemble_contact_coupling_matrices as _assemble_contact_coupling_matrices,
    _facet_shape_values,
    _tri_centroid,
    _tri_area,
)
from .mortar_problem import (
    MortarContactProblemPair,
    MortarContactProblem,
    assemble_mortar_contact_problem,
)
from .mortar_multiplier import (
    ContactMultiplierSpace,
    MultiplierSpec,
    infer_contact_side_facets as _infer_contact_side_facets,
)
from .contact_api import (
    ContactSide,
    ContactSideSpec,
    ContactSpaces,
    ContactPairSpec,
    ContactGroupSpaces,
    ContactGroupSpec,
    OneSidedContactSpaces,
    OneSidedContactSpec,
)
from .contact_diagnostics import (
    ContactConstraintDiagnostics,
    ContactConstraintQualityIssue,
    ContactConstraintQualityReport,
    contact_constraint_matrix_diagnostics,
    assess_contact_constraint_quality,
)
from .contact_forms import (
    ContactBilinear,
    ContactBilinearLike,
    ContactJacobianReturn,
    ContactOperators,
    ContactSolveResult,
    ContactState,
    MixedSurfaceResidualForm,
    MultiplierContactContribution,
    PenaltyContactContribution,
    SurfaceHatFn,
    _compile_contact_bilinear,
    _infer_contact_backend,
    _is_compiled_contact_bilinear,
    compile_tagged_pair_nitsche_penalty_residual,
    make_tagged_pair_nitsche_penalty_bilinear,
)
from .contact_kkt_solver import (
    ContactKKTSolveConfig,
    ContactKKTSolveInfo,
    ContactKKTSolveResult,
    solve_contact_kkt,
    solve_contact_kkt_with_info,
)
from .contact_solvers import (
    AugmentedLagrangianState,
    AugmentedLagrangianResult,
    UnilateralContactActiveSetRecord,
    UnilateralContactActiveSetResult,
    solve_augmented_lagrangian_outer_loop,
    solve_unilateral_contact_active_set_kkt,
)
from .contact_embedding import (
    EmbeddingMap,
    build_nodal_embedding_map,
    build_barycentric_embedding_map,
    build_barycentric_embedding_map_from_meshes,
    assemble_embedding_constraint_matrix,
    assemble_fixed_rigid_hub_constraint_matrix,
    assemble_rigid_hub_constraint_matrix,
    assemble_rbe2_constraint_matrix,
    assemble_rbe3_constraint_matrix,
    build_rbe3_weights,
    build_rbe3_remote_resultant,
)
from .contact_nitsche import (
    assemble_pair_nitsche_supermesh_impl as _assemble_pair_nitsche_supermesh_impl,
    make_pair_nitsche_supermesh_bilinear,
)
from .contact_surface_helpers import (
    OneSidedContact,
    active_contact_facets,
    contact_space_side_n_dofs as _contact_space_side_n_dofs,
    facet_gap_values,
    onesided_gap_diagnostics as _onesided_gap_diagnostics,
    summarize_contact_field_state as _summarize_contact_field_state,
    surface_node_normals as _surface_node_normals,
)
from .contact_surface_space import ContactSurfaceSpace, OneSidedContactSurfaceSpace, OneToManyContactSurfaceSpace
from .mortar_operators import (
    p0_reduction_matrix_from_facets as _p0_reduction_matrix_from_facets,
    p0_patch_group_matrix as _p0_patch_group_matrix,
    apply_integrated_coarse_p0_groups as _apply_integrated_coarse_p0_groups,
    expand_scalar_constraint_dense as _expand_scalar_constraint_dense,
    expand_scalar_constraint_coo as _expand_scalar_constraint_coo,
    dual_nodal_blocks_from_dense as _dual_nodal_blocks_from_dense,
    dense_to_coo_entries as _dense_to_coo_entries,
    apply_coarse_mortar_projection as _apply_coarse_mortar_projection,
    apply_constraint_row_scaling as _apply_constraint_row_scaling,
    is_patch_qr_multiplier as _is_patch_qr_multiplier,
)

if TYPE_CHECKING:
    from .contact_interface import ContactCouplingMatrix
    from ..core.weakform import Params as WeakParams
    from .contact_interface import SurfaceMixedFormContext
    from ..solver import FluxSparseMatrix, FluxSparseOperator


def _warn_contact_legacy_name(old: str, new: str) -> None:
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _contact_sparse_to_coo(jacobian: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if hasattr(jacobian, "to_coo"):
        rows, cols, data, shape_or_n_dofs = jacobian.to_coo()
        if isinstance(shape_or_n_dofs, tuple):
            if int(shape_or_n_dofs[0]) != int(shape_or_n_dofs[1]):
                raise ValueError("Rectangular contact operators are not supported in this path.")
            n_dofs = int(shape_or_n_dofs[0])
        else:
            n_dofs = int(shape_or_n_dofs)
        return (
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
            np.asarray(data, dtype=float),
            n_dofs,
        )
    rows, cols, data, n_dofs = jacobian
    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(data, dtype=float),
        int(n_dofs),
    )


def assemble_contact_interface_residual(*args, **kwargs):
    """Assemble residual on a contact interface supermesh."""
    return _assemble_contact_interface_residual(*args, **kwargs)


def assemble_contact_interface_jacobian(*args, **kwargs):
    """Assemble Jacobian on a contact interface supermesh."""
    return _assemble_contact_interface_jacobian(*args, **kwargs)


def assemble_contact_coupling_matrices(*args, **kwargs):
    """Assemble coupling matrices for contact interface constraints."""
    return _assemble_contact_coupling_matrices(*args, **kwargs)


def _coo_to_dense(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, shape: tuple[int, int], *, backend: str):
    if backend == "jax":
        import jax.numpy as jnp

        out = jnp.zeros(shape, dtype=jnp.asarray(data).dtype if np.asarray(data).size else float)
        if len(rows) == 0:
            return out
        return out.at[np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)].add(np.asarray(data))
    out = np.zeros(shape, dtype=float)
    for r, c, v in zip(np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), np.asarray(data, dtype=float)):
        out[int(r), int(c)] += float(v)
    return out


def coarse_p1_basis_from_node_groups(
    n_fine_nodes: int,
    groups,
    *,
    weights=None,
    normalize: bool = True,
) -> np.ndarray:
    """Build coarse P1 rows from groups of fine master-side nodes.

    Each group defines one coarse multiplier shape function represented in the
    fine nodal basis.  With the default ``normalize=True``, every row sums to
    one, giving a simple partition-style averaging basis.
    """

    n_nodes = int(n_fine_nodes)
    if n_nodes <= 0:
        raise ValueError("n_fine_nodes must be positive.")
    rows = []
    weight_rows = None if weights is None else list(weights)
    group_rows = list(groups)
    if not group_rows:
        raise ValueError("groups must contain at least one node group.")
    if weight_rows is not None and len(weight_rows) != len(group_rows):
        raise ValueError("weights must have the same number of rows as groups.")
    for row_id, group in enumerate(group_rows):
        nodes = np.asarray(group, dtype=int).reshape(-1)
        if nodes.size == 0:
            raise ValueError("each node group must be non-empty.")
        if np.any(nodes < 0) or np.any(nodes >= n_nodes):
            raise ValueError("node group contains an out-of-range node id.")
        if weight_rows is None:
            values = np.ones((int(nodes.size),), dtype=float)
        else:
            values = np.asarray(weight_rows[row_id], dtype=float).reshape(-1)
            if int(values.size) != int(nodes.size):
                raise ValueError("each weight row must match the corresponding node group size.")
        if normalize:
            total = float(np.sum(values))
            if abs(total) <= np.finfo(float).eps:
                raise ValueError("cannot normalize a coarse P1 basis row with zero weight sum.")
            values = values / total
        row = np.zeros((n_nodes,), dtype=float)
        for node, value in zip(nodes.tolist(), values.tolist()):
            row[int(node)] += float(value)
        rows.append(row)
    return np.vstack(rows)


def coarse_p1_basis_from_surface_grid(
    surface,
    *,
    shape: tuple[int, int],
    axes: tuple[int, int] = (0, 1),
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    clamp: bool = True,
) -> np.ndarray:
    """Build coarse P1 rows by bilinear interpolation on a surface coordinate grid.

    The returned matrix has shape ``(shape[0] * shape[1], n_surface_nodes)``.
    It is intended for planar or nearly planar surfaces where two coordinate
    axes provide a reasonable parameterization.
    """

    coords = np.asarray(getattr(surface, "coords", surface), dtype=float)
    if coords.ndim != 2:
        raise ValueError("surface must provide coords with shape (n_nodes, dim).")
    n_nodes = int(coords.shape[0])
    if n_nodes <= 0:
        raise ValueError("surface must contain at least one node.")
    ax0, ax1 = (int(axes[0]), int(axes[1]))
    if ax0 == ax1:
        raise ValueError("axes must contain two distinct coordinate axes.")
    if ax0 < 0 or ax1 < 0 or ax0 >= int(coords.shape[1]) or ax1 >= int(coords.shape[1]):
        raise ValueError("axes are out of range for surface coordinates.")
    nu, nv = int(shape[0]), int(shape[1])
    if nu < 2 or nv < 2:
        raise ValueError("shape must be at least (2, 2) for P1 grid basis.")
    uv = coords[:, [ax0, ax1]]
    if bounds is None:
        umin, vmin = np.min(uv, axis=0)
        umax, vmax = np.max(uv, axis=0)
    else:
        (umin, umax), (vmin, vmax) = bounds
        umin, umax, vmin, vmax = float(umin), float(umax), float(vmin), float(vmax)
    if not (umax > umin and vmax > vmin):
        raise ValueError("surface grid bounds must have positive extent.")

    u = (uv[:, 0] - umin) / (umax - umin) * (nu - 1)
    v = (uv[:, 1] - vmin) / (vmax - vmin) * (nv - 1)
    if clamp:
        u = np.clip(u, 0.0, float(nu - 1))
        v = np.clip(v, 0.0, float(nv - 1))
    elif np.any((u < 0.0) | (u > nu - 1) | (v < 0.0) | (v > nv - 1)):
        raise ValueError("surface node lies outside the requested grid bounds.")

    iu0 = np.floor(u).astype(int)
    iv0 = np.floor(v).astype(int)
    iu0 = np.clip(iu0, 0, nu - 2)
    iv0 = np.clip(iv0, 0, nv - 2)
    du = u - iu0
    dv = v - iv0

    basis = np.zeros((nu * nv, n_nodes), dtype=float)
    for node_id in range(n_nodes):
        i = int(iu0[node_id])
        j = int(iv0[node_id])
        weights = (
            ((1.0 - du[node_id]) * (1.0 - dv[node_id]), i, j),
            (du[node_id] * (1.0 - dv[node_id]), i + 1, j),
            ((1.0 - du[node_id]) * dv[node_id], i, j + 1),
            (du[node_id] * dv[node_id], i + 1, j + 1),
        )
        for value, ii, jj in weights:
            basis[int(jj) * nu + int(ii), node_id] += float(value)
    return basis


def _expand_scalar_constraint_dense(B_scalar, *, value_dim: int, backend: str):
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


def _expand_scalar_constraint_coo(
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


def _resolve_multiplier_spec(
    contact,
    *,
    multiplier: ContactMultiplierSpace | None,
    facet_conn_master: np.ndarray | None,
) -> tuple[str, np.ndarray | None, ContactMultiplierSpace]:
    if multiplier is not None and not isinstance(multiplier, ContactMultiplierSpace):
        raise TypeError("multiplier must be a ContactMultiplierSpace.")
    if multiplier is None:
        multiplier = ContactMultiplierSpace.from_contact(contact, family="dual_nodal", side="master")
    fam = str(multiplier.family).lower()
    if fam in {"p0", "p0_active", "p0_supermesh"} and str(multiplier.side).lower() != "master":
        raise NotImplementedError(
            "p0-like multipliers currently support only side='master' "
            "(current implementation limitation)."
        )
    facet = multiplier.facet_conn
    if facet is None and fam == "p0":
        facet = _infer_contact_side_facets(contact, side=str(multiplier.side))
    if facet is None:
        facet = facet_conn_master
    if fam == "dual_nodal" and str(multiplier.side).lower() != "master":
        raise NotImplementedError(
            "dual_nodal multipliers currently support only side='master' "
            "(requires the master-side nodal mass block)."
        )
    if fam == "coarse_p1" and str(multiplier.side).lower() != "master":
        raise NotImplementedError(
            "coarse_p1 multipliers currently support only side='master' "
            "(coarse basis is defined in the master-side nodal space)."
        )
    if fam not in {"nodal", "dual_nodal", "coarse_p1", "p0", "p0_active", "p0_supermesh"}:
        raise ValueError(
            "multiplier.family must be 'nodal', 'dual_nodal', 'coarse_p1', "
            "'p0', 'p0_active', or 'p0_supermesh'"
        )
    if fam in {"p0", "p0_active"} and facet is None:
        raise ValueError(f"facet_conn_master is required when multiplier.family='{fam}'.")
    facet_arr = None if facet is None else np.asarray(facet, dtype=int)
    resolved_multiplier = ContactMultiplierSpace(
        family=fam,
        side=str(multiplier.side).lower(),
        value_dim=int(multiplier.value_dim),
        facet_conn=facet_arr,
        coarse_rank=None if multiplier.coarse_rank is None else int(multiplier.coarse_rank),
        coarse_projection=(
            None
            if multiplier.coarse_projection is None
            else np.asarray(multiplier.coarse_projection, dtype=float)
        ),
        coarse_mode=multiplier.coarse_mode,
        coarse_energy_tol=multiplier.coarse_energy_tol,
        coarse_rtol=multiplier.coarse_rtol,
        coarse_max_rank=multiplier.coarse_max_rank,
        coarse_patch_ids=(
            None
            if multiplier.coarse_patch_ids is None
            else np.asarray(multiplier.coarse_patch_ids, dtype=int)
        ),
        coarse_basis=(
            None
            if multiplier.coarse_basis is None
            else np.asarray(multiplier.coarse_basis, dtype=float)
        ),
        constraint_scaling=str(getattr(multiplier, "constraint_scaling", "none")).lower(),
    )
    return fam, facet_arr, resolved_multiplier


def _coalesce_int_coo(rows: np.ndarray, cols: np.ndarray, data: np.ndarray):
    from ..solver.sparse import coalesce_coo

    r, c, d = coalesce_coo(rows, cols, data)
    return np.asarray(r, dtype=int), np.asarray(c, dtype=int), np.asarray(d, dtype=float)


def _kkt_coo_from_coupling(
    coupling_aa,
    coupling_ab,
    *,
    rho: float,
    multiplier_space: str,
    facet_conn_master: np.ndarray | None,
    multiplier_value_dim: int = 1,
    coarse_rank: int | None = None,
    coarse_projection: np.ndarray | None = None,
    coarse_mode: str | None = None,
    coarse_energy_tol: float | None = None,
    coarse_rtol: float | None = None,
    coarse_max_rank: int | None = None,
    coarse_patch_ids: np.ndarray | None = None,
    coarse_basis: np.ndarray | None = None,
    constraint_scaling: str = "none",
):
    if multiplier_space == "p0_supermesh":
        raise NotImplementedError(
            "multiplier_space='p0_supermesh' requires direct B/Kuu assembly from contact operators."
        )
    rows_aa, cols_aa, data_aa = _coalesce_int_coo(coupling_aa.rows, coupling_aa.cols, coupling_aa.data)
    rows_ab, cols_ab, data_ab = _coalesce_int_coo(coupling_ab.rows, coupling_ab.cols, coupling_ab.data)
    n_a = int(coupling_aa.shape[0])
    n_b = int(coupling_ab.shape[1])
    n_u = n_a + n_b

    if multiplier_space == "nodal":
        n_l = n_a
        b_rows = np.concatenate([rows_aa, rows_ab])
        b_cols = np.concatenate([cols_aa, n_a + cols_ab])
        b_data = np.concatenate([data_aa, -data_ab])
    elif multiplier_space == "dual_nodal":
        M_aa = _coo_to_dense(rows_aa, cols_aa, data_aa, coupling_aa.shape, backend="numpy")
        M_ab = _coo_to_dense(rows_ab, cols_ab, data_ab, coupling_ab.shape, backend="numpy")
        B_a, B_b = _dual_nodal_blocks_from_dense(M_aa, M_ab, backend="numpy")
        rows_a, cols_a, data_a = _dense_to_coo_entries(B_a)
        rows_b, cols_b, data_b = _dense_to_coo_entries(B_b)
        n_l = int(B_a.shape[0])
        b_rows = np.concatenate([rows_a, rows_b])
        b_cols = np.concatenate([cols_a, n_a + cols_b])
        b_data = np.concatenate([data_a, -data_b])
    elif multiplier_space == "coarse_p1":
        if coarse_basis is None:
            raise ValueError("coarse_basis is required when multiplier_space='coarse_p1'.")
        M_aa = _coo_to_dense(rows_aa, cols_aa, data_aa, coupling_aa.shape, backend="numpy")
        M_ab = _coo_to_dense(rows_ab, cols_ab, data_ab, coupling_ab.shape, backend="numpy")
        C = np.asarray(coarse_basis, dtype=float)
        if C.ndim != 2 or int(C.shape[1]) != int(M_aa.shape[0]):
            raise ValueError("coarse_basis must have shape (n_coarse_nodes, n_master_nodes).")
        B_a = C @ M_aa
        B_b = C @ M_ab
        rows_a, cols_a, data_a = _dense_to_coo_entries(B_a)
        rows_b, cols_b, data_b = _dense_to_coo_entries(B_b)
        n_l = int(B_a.shape[0])
        b_rows = np.concatenate([rows_a, rows_b])
        b_cols = np.concatenate([cols_a, n_a + cols_b])
        b_data = np.concatenate([data_a, -data_b])
    elif multiplier_space == "p0":
        if facet_conn_master is None:
            raise ValueError("facet_conn_master is required when multiplier_space='p0'.")
        facets = np.asarray(facet_conn_master, dtype=int)
        n_l = int(facets.shape[0])
        row_map: dict[int, list[int]] = {i: [] for i in range(n_a)}
        for k, r in enumerate(rows_aa):
            row_map[int(r)].append(int(k))
        row_map_ab: dict[int, list[int]] = {i: [] for i in range(n_a)}
        for k, r in enumerate(rows_ab):
            row_map_ab[int(r)].append(int(k))
        b_rows_l: list[int] = []
        b_cols_l: list[int] = []
        b_data_l: list[float] = []
        for lf, nodes in enumerate(facets):
            acc: dict[int, float] = {}
            for n in np.asarray(nodes, dtype=int):
                for k in row_map.get(int(n), []):
                    c = int(cols_aa[k])
                    acc[c] = acc.get(c, 0.0) + float(data_aa[k])
                for k in row_map_ab.get(int(n), []):
                    c = n_a + int(cols_ab[k])
                    acc[c] = acc.get(c, 0.0) - float(data_ab[k])
            for c, v in acc.items():
                b_rows_l.append(int(lf))
                b_cols_l.append(int(c))
                b_data_l.append(float(v))
        if b_rows_l:
            b_rows = np.asarray(b_rows_l, dtype=int)
            b_cols = np.asarray(b_cols_l, dtype=int)
            b_data = np.asarray(b_data_l, dtype=float)
            b_rows, b_cols, b_data = _coalesce_int_coo(b_rows, b_cols, b_data)
        else:
            b_rows = np.zeros((0,), dtype=int)
            b_cols = np.zeros((0,), dtype=int)
            b_data = np.zeros((0,), dtype=float)
    else:
        raise ValueError("multiplier_space must be 'nodal', 'dual_nodal', 'coarse_p1', or 'p0'")
    if coarse_patch_ids is not None and str(coarse_mode).lower() != "patch_qr":
        if multiplier_space != "p0":
            raise ValueError("coarse_patch_ids are supported only for p0 multiplier_space in sparse KKT assembly.")
        B_dense = np.zeros((n_l, n_u), dtype=float)
        B_dense[b_rows, b_cols] += b_data
        P = _p0_patch_group_matrix(coarse_patch_ids, int(n_l))
        B_dense = P @ B_dense
        b_rows, b_cols, b_data = _dense_to_coo_entries(B_dense)
        n_l = int(B_dense.shape[0])
    b_rows, b_cols, b_data, n_l, n_u = _expand_scalar_constraint_coo(
        b_rows,
        b_cols,
        b_data,
        n_rows=n_l,
        n_cols=n_u,
        value_dim=int(multiplier_value_dim),
    )
    if coarse_rank is not None or coarse_projection is not None or coarse_mode is not None:
        B_dense = np.zeros((n_l, n_u), dtype=float)
        B_dense[b_rows, b_cols] += b_data
        coarse_multiplier = ContactMultiplierSpace(
            family="p0" if str(coarse_mode).lower() == "patch_qr" else "nodal",
            value_dim=int(multiplier_value_dim) if str(coarse_mode).lower() == "patch_qr" else 1,
            coarse_rank=coarse_rank,
            coarse_projection=coarse_projection,
            coarse_mode=coarse_mode,
            coarse_energy_tol=coarse_energy_tol,
            coarse_rtol=coarse_rtol,
            coarse_max_rank=coarse_max_rank,
            coarse_patch_ids=coarse_patch_ids if str(coarse_mode).lower() == "patch_qr" else None,
            coarse_basis=None,
            constraint_scaling="none",
        )
        n_a_expanded = int(n_a) * int(multiplier_value_dim)
        B_a_dense = B_dense[:, :n_a_expanded]
        B_b_dense = -B_dense[:, n_a_expanded:]
        B_a_dense, B_b_dense = _apply_coarse_mortar_projection(
            B_a_dense,
            B_b_dense,
            coarse_multiplier,
            backend="numpy",
        )
        B_dense = np.concatenate([B_a_dense, -B_b_dense], axis=1)
        b_rows, b_cols, b_data = _dense_to_coo_entries(B_dense)
        n_l = int(B_dense.shape[0])
        n_u = int(B_dense.shape[1])
    if str(constraint_scaling).lower() != "none":
        B_dense = np.zeros((n_l, n_u), dtype=float)
        B_dense[b_rows, b_cols] += b_data
        n_a_expanded = int(n_a) * int(multiplier_value_dim)
        B_a_dense = B_dense[:, :n_a_expanded]
        B_b_dense = -B_dense[:, n_a_expanded:]
        B_a_dense, B_b_dense = _apply_constraint_row_scaling(
            B_a_dense,
            B_b_dense,
            str(constraint_scaling).lower(),
            backend="numpy",
        )
        B_dense = np.concatenate([B_a_dense, -B_b_dense], axis=1)
        b_rows, b_cols, b_data = _dense_to_coo_entries(B_dense)
        n_l = int(B_dense.shape[0])
        n_u = int(B_dense.shape[1])

    # Build Kuu = rho * B^T B from row-wise products.
    by_row: dict[int, list[int]] = {}
    for k, r in enumerate(b_rows):
        by_row.setdefault(int(r), []).append(int(k))
    kuu_acc: dict[tuple[int, int], float] = {}
    if float(rho) != 0.0:
        rr = float(rho)
        for ids in by_row.values():
            for i in ids:
                ci = int(b_cols[i])
                vi = float(b_data[i])
                for j in ids:
                    cj = int(b_cols[j])
                    vj = float(b_data[j])
                    key = (ci, cj)
                    kuu_acc[key] = kuu_acc.get(key, 0.0) + rr * vi * vj

    kuu_rows = np.fromiter((k[0] for k in kuu_acc.keys()), dtype=int, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=int)
    kuu_cols = np.fromiter((k[1] for k in kuu_acc.keys()), dtype=int, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=int)
    kuu_data = np.fromiter((v for v in kuu_acc.values()), dtype=float, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=float)

    # KKT COO assembly:
    # [Kuu  B^T]
    # [ B    0 ]
    k_rows = []
    k_cols = []
    k_data = []
    if kuu_rows.size:
        k_rows.append(kuu_rows)
        k_cols.append(kuu_cols)
        k_data.append(kuu_data)
    if b_rows.size:
        # B^T block (top-right)
        k_rows.append(b_cols)
        k_cols.append(n_u + b_rows)
        k_data.append(b_data)
        # B block (bottom-left)
        k_rows.append(n_u + b_rows)
        k_cols.append(b_cols)
        k_data.append(b_data)
    if k_rows:
        rows = np.concatenate(k_rows)
        cols = np.concatenate(k_cols)
        data = np.concatenate(k_data)
        rows, cols, data = _coalesce_int_coo(rows, cols, data)
    else:
        rows = np.zeros((0,), dtype=int)
        cols = np.zeros((0,), dtype=int)
        data = np.zeros((0,), dtype=float)
    n_total = int(n_u + n_l)
    return rows, cols, data, n_total


def _assemble_supermesh_triangle_p0_blocks(
    contact,
    *,
    backend: str,
    value_dim: int,
    coarse_patch_ids: np.ndarray | None = None,
):
    if not all(
        hasattr(contact, name)
        for name in (
            "supermesh_coords",
            "supermesh_conn",
            "source_facets_master",
            "source_facets_slave",
            "surface_master",
            "surface_slave",
            "tol",
        )
    ):
        raise TypeError("contact must expose supermesh geometry for multiplier.family='p0_supermesh'.")

    supermesh_coords = np.asarray(contact.supermesh_coords, dtype=float)
    supermesh_conn = np.asarray(contact.supermesh_conn, dtype=int)
    source_facets_master = np.asarray(contact.source_facets_master, dtype=int)
    source_facets_slave = np.asarray(contact.source_facets_slave, dtype=int)
    facet_conn_master = np.asarray(contact.surface_master.conn, dtype=int)
    facet_conn_slave = np.asarray(contact.surface_slave.conn, dtype=int)
    coords_master = np.asarray(contact.surface_master.coords, dtype=float)
    coords_slave = np.asarray(contact.surface_slave.coords, dtype=float)
    tol = float(contact.tol)

    n_tri = int(supermesh_conn.shape[0])
    n_master_dofs = int(contact.surface_master.n_nodes)
    n_slave_dofs = int(contact.surface_slave.n_nodes)
    B_a = np.zeros((n_tri, n_master_dofs), dtype=float)
    B_b = np.zeros((n_tri, n_slave_dofs), dtype=float)

    for tri_id, (tri, fa, fb) in enumerate(zip(supermesh_conn, source_facets_master, source_facets_slave)):
        a = supermesh_coords[int(tri[0])]
        b = supermesh_coords[int(tri[1])]
        c = supermesh_coords[int(tri[2])]
        centroid = _tri_centroid(a, b, c)
        area = _tri_area(a, b, c)

        facet_master = facet_conn_master[int(fa)]
        facet_slave = facet_conn_slave[int(fb)]
        N_master = _facet_shape_values(centroid, facet_master, coords_master, tol=tol)
        N_slave = _facet_shape_values(centroid, facet_slave, coords_slave, tol=tol)
        B_a[tri_id, facet_master] += area * N_master
        B_b[tri_id, facet_slave] += area * N_slave

    B_a, B_b = _apply_integrated_coarse_p0_groups(B_a, B_b, coarse_patch_ids, backend=backend)
    B_a = _expand_scalar_constraint_dense(B_a, value_dim=int(value_dim), backend=backend)
    B_b = _expand_scalar_constraint_dense(B_b, value_dim=int(value_dim), backend=backend)
    return B_a, B_b


def _assemble_active_master_facet_p0_blocks(
    contact,
    *,
    backend: str,
    value_dim: int,
    coarse_patch_ids: np.ndarray | None = None,
):
    if not all(
        hasattr(contact, name)
        for name in (
            "supermesh_coords",
            "supermesh_conn",
            "source_facets_master",
            "source_facets_slave",
            "surface_master",
            "surface_slave",
            "tol",
        )
    ):
        raise TypeError("contact must expose supermesh geometry for multiplier.family='p0_active'.")

    supermesh_coords = np.asarray(contact.supermesh_coords, dtype=float)
    supermesh_conn = np.asarray(contact.supermesh_conn, dtype=int)
    source_facets_master = np.asarray(contact.source_facets_master, dtype=int)
    source_facets_slave = np.asarray(contact.source_facets_slave, dtype=int)
    active_facets = np.unique(source_facets_master)
    facet_row = {int(f): i for i, f in enumerate(active_facets.tolist())}

    facet_conn_master_all = np.asarray(contact.surface_master.conn, dtype=int)
    facet_conn_slave_all = np.asarray(contact.surface_slave.conn, dtype=int)
    coords_master = np.asarray(contact.surface_master.coords, dtype=float)
    coords_slave = np.asarray(contact.surface_slave.coords, dtype=float)
    tol = float(contact.tol)

    B_a = np.zeros((int(active_facets.shape[0]), int(contact.surface_master.n_nodes)), dtype=float)
    B_b = np.zeros((int(active_facets.shape[0]), int(contact.surface_slave.n_nodes)), dtype=float)

    for tri, fa, fb in zip(supermesh_conn, source_facets_master, source_facets_slave):
        row = facet_row[int(fa)]
        a = supermesh_coords[int(tri[0])]
        b = supermesh_coords[int(tri[1])]
        c = supermesh_coords[int(tri[2])]
        centroid = _tri_centroid(a, b, c)
        area = _tri_area(a, b, c)

        facet_master = facet_conn_master_all[int(fa)]
        facet_slave = facet_conn_slave_all[int(fb)]
        N_master = _facet_shape_values(centroid, facet_master, coords_master, tol=tol)
        N_slave = _facet_shape_values(centroid, facet_slave, coords_slave, tol=tol)
        B_a[row, facet_master] += area * N_master
        B_b[row, facet_slave] += area * N_slave

    B_a, B_b = _apply_integrated_coarse_p0_groups(B_a, B_b, coarse_patch_ids, backend=backend)
    B_a = _expand_scalar_constraint_dense(B_a, value_dim=int(value_dim), backend=backend)
    B_b = _expand_scalar_constraint_dense(B_b, value_dim=int(value_dim), backend=backend)
    return B_a, B_b, facet_conn_master_all[active_facets]


def assemble_contact_constraint_operators(
    contact,
    *,
    law: str | None = None,
    formulation: str | None = None,
    rho: float = 0.0,
    multiplier: ContactMultiplierSpace | None = None,
    backend: str | None = None,
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    warn_float32_assembly_once(context="contact constraint assembly")
    """Assemble constraint-family operators (coupling/B/Kuu, optionally residual/jacobian metadata)."""
    backend = _infer_contact_backend(state, u, params, res_form, weak_form, rho, default="numpy") if backend is None else str(backend).lower()
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    if state is not None and u is not None and state is not u:
        raise ValueError("state and u are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form
    u_eff = state if state is not None else u
    has_eval_inputs = (res_form_eff is not None) or (u_eff is not None) or (params is not None)
    if has_eval_inputs and (res_form_eff is None or u_eff is None or params is None):
        raise ValueError(
            "weak_form/state/params (or res_form/u/params) must be provided together for constraint residual/jacobian evaluation."
        )
    f_arg = None if formulation is None else str(formulation).lower()
    if f_arg is not None and f_arg in {"penalty", "penalty_consistent", "nitsche"}:
        raise ValueError(
            "Constraint operators are multiplier-family only. Use a multiplier/augmented_lagrangian formulation."
        )
    resolved = "mortar"
    law_resolved = str(law) if law is not None else "one_sided_normal_frictionless"
    formulation_resolved = str(formulation) if formulation is not None else "multiplier"
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    if has_eval_inputs and backend != "jax":
        raise NotImplementedError(
            "weak-form contact residual/jacobian evaluation requires backend='jax'. "
            "backend='numpy' remains available for coupling/KKT assembly only."
        )

    if not hasattr(contact, "assemble_contact_coupling_matrices"):
        raise TypeError("contact must provide assemble_contact_coupling_matrices() for constraint operators.")
    coupling_aa, coupling_ab = contact.assemble_contact_coupling_matrices()

    mult_space, facet_conn_master, multiplier_resolved = _resolve_multiplier_spec(
        contact,
        multiplier=multiplier,
        facet_conn_master=None,
    )

    M_aa = _coo_to_dense(coupling_aa.rows, coupling_aa.cols, coupling_aa.data, coupling_aa.shape, backend=backend)
    M_ab = _coo_to_dense(coupling_ab.rows, coupling_ab.cols, coupling_ab.data, coupling_ab.shape, backend=backend)

    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np

    if mult_space == "nodal":
        B_a = M_aa
        B_b = M_ab
    elif mult_space == "dual_nodal":
        B_a, B_b = _dual_nodal_blocks_from_dense(M_aa, M_ab, backend=backend)
    elif mult_space == "coarse_p1":
        C = xp.asarray(multiplier_resolved.coarse_basis)
        if int(C.shape[1]) != int(M_aa.shape[0]):
            raise ValueError("coarse_basis must have shape (n_coarse_nodes, n_master_nodes).")
        B_a = C @ M_aa
        B_b = C @ M_ab
        B_a = _expand_scalar_constraint_dense(
            B_a,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
        B_b = _expand_scalar_constraint_dense(
            B_b,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
    elif mult_space == "p0":
        n_master_nodes = int(coupling_aa.shape[0])
        S_np = _p0_reduction_matrix_from_facets(facet_conn_master, n_master_nodes)
        S = xp.asarray(S_np)
        B_a = S @ M_aa
        B_b = S @ M_ab
        B_a, B_b = _apply_integrated_coarse_p0_groups(
            B_a,
            B_b,
            None if _is_patch_qr_multiplier(multiplier_resolved) else multiplier_resolved.coarse_patch_ids,
            backend=backend,
        )
        B_a = _expand_scalar_constraint_dense(
            B_a,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
        B_b = _expand_scalar_constraint_dense(
            B_b,
            value_dim=int(multiplier_resolved.value_dim),
            backend=backend,
        )
    elif mult_space == "p0_active":
        B_a, B_b, facet_conn_master = _assemble_active_master_facet_p0_blocks(
            contact,
            backend=backend,
            value_dim=int(multiplier_resolved.value_dim),
            coarse_patch_ids=None if _is_patch_qr_multiplier(multiplier_resolved) else multiplier_resolved.coarse_patch_ids,
        )
    elif mult_space == "p0_supermesh":
        B_a, B_b = _assemble_supermesh_triangle_p0_blocks(
            contact,
            backend=backend,
            value_dim=int(multiplier_resolved.value_dim),
            coarse_patch_ids=None if _is_patch_qr_multiplier(multiplier_resolved) else multiplier_resolved.coarse_patch_ids,
        )
        if backend == "jax":
            B_a = xp.asarray(B_a)
            B_b = xp.asarray(B_b)
    else:
        raise ValueError(
            "multiplier.family must be 'nodal', 'dual_nodal', 'coarse_p1', "
            "'p0', 'p0_active', or 'p0_supermesh'."
        )

    n_rows_before_reduction = int(B_a.shape[0])
    B_a, B_b = _apply_coarse_mortar_projection(B_a, B_b, multiplier_resolved, backend=backend)
    n_rows_after_reduction = int(B_a.shape[0])
    B_a, B_b = _apply_constraint_row_scaling(
        B_a,
        B_b,
        multiplier_resolved.constraint_scaling,
        backend=backend,
    )
    B = xp.concatenate([B_a, -B_b], axis=1)
    Kuu = xp.asarray(rho) * (B.T @ B)
    residual = None
    jacobian = None
    if has_eval_inputs:
        if not hasattr(contact, "assemble_residual") or not hasattr(contact, "assemble_jacobian"):
            raise TypeError("contact must provide assemble_residual() and assemble_jacobian() for weak-form evaluation.")
        residual = contact.assemble_residual(res_form_eff, u_eff, params, normal_source=normal_source)
        jacobian = contact.assemble_jacobian(
            res_form_eff,
            u_eff,
            params,
            normal_source=normal_source,
            sparse=sparse,
            backend=backend,
            batch_jac=batch_jac,
        )
    return MultiplierContactContribution(
        enforcement=resolved,
        law=law_resolved,
        formulation=formulation_resolved,
        coupling_aa=coupling_aa,
        coupling_ab=coupling_ab,
        B_a=B_a,
        B_b=B_b,
        B=B,
        Kuu=Kuu,
        residual=residual,
        jacobian=jacobian,
        facet_conn_master=facet_conn_master,
        rho=rho,
        multiplier=multiplier_resolved,
        diagnostics={
            "constraint_scaling": str(multiplier_resolved.constraint_scaling).lower(),
            "constraint_reduction": str(multiplier_resolved.coarse_mode).lower()
            if multiplier_resolved.coarse_mode is not None
            else "none",
            "constraint_rows_before_reduction": n_rows_before_reduction,
            "constraint_rows_after_reduction": n_rows_after_reduction,
        },
    )


def _resolve_contact_operator_enforcement(
    *,
    enforcement: str | None = None,
    method: str | None = None,
    formulation: str | None = None,
    multiplier: ContactMultiplierSpace | None = None,
) -> str:
    if enforcement is not None and method is not None and str(enforcement).lower() != str(method).lower():
        raise ValueError("enforcement and method are aliases; provide only one effective value.")
    value = enforcement if enforcement is not None else method
    if value is None and formulation is not None:
        formulation_key = str(formulation).lower()
        if formulation_key in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
            value = "constraint"
        elif formulation_key in {"penalty", "penalty_consistent", "nitsche"}:
            value = "penalty"
    if value is None:
        value = "constraint" if multiplier is not None else "penalty"
    value_key = str(value).lower()
    if value_key in {"penalty", "nitsche", "penalty_family", "penalty-family"}:
        return "penalty"
    if value_key in {"constraint", "mortar", "multiplier", "constraint_family", "constraint-family", "augmented_lagrangian"}:
        return "constraint"
    raise ValueError("enforcement must resolve to either 'penalty' or 'constraint'.")


def assemble_contact_penalty_operators(
    contact,
    *,
    law: str | None = None,
    formulation: str | None = None,
    backend: str | None = None,
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    warn_float32_assembly_once(context="contact penalty assembly")
    """Assemble penalty-family operators (residual/jacobian)."""
    f_arg = None if formulation is None else str(formulation).lower()
    if f_arg is not None and f_arg in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
        raise ValueError(
            "Penalty operators are penalty-family only. Use penalty/penalty_consistent formulation."
        )
    resolved = "nitsche"
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    if state is not None and u is not None and state is not u:
        raise ValueError("state and u are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form
    u_eff = state if state is not None else u

    law_resolved = str(law) if law is not None else "one_sided_normal_frictionless"
    formulation_resolved = str(formulation) if formulation is not None else "penalty_consistent"
    backend = _infer_contact_backend(contact, res_form_eff, u_eff, params, default="jax") if backend is None else str(backend).lower()
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    if backend != "jax":
        raise NotImplementedError(
            "Penalty-family weak-form Jacobian assembly requires backend='jax'. "
            "backend='numpy' for contact Jacobians has been removed."
        )
    if res_form_eff is None or u_eff is None or params is None:
        raise ValueError("weak_form/state/params (or res_form/u/params) are required for penalty operators.")
    if not hasattr(contact, "assemble_residual") or not hasattr(contact, "assemble_jacobian"):
        raise TypeError("contact must provide assemble_residual() and assemble_jacobian() for penalty operators.")
    residual = contact.assemble_residual(res_form_eff, u_eff, params, normal_source=normal_source)
    jacobian = contact.assemble_jacobian(
        res_form_eff,
        u_eff,
        params,
        normal_source=normal_source,
        sparse=sparse,
        backend=backend,
        batch_jac=batch_jac,
    )
    return PenaltyContactContribution(
        enforcement=resolved,
        law=law_resolved,
        formulation=formulation_resolved,
        residual=residual,
        jacobian=jacobian,
    )


def assemble_contact_kkt(
    coupling_aa,
    coupling_ab,
    *,
    rho: float = 0.0,
    multiplier: ContactMultiplierSpace | None = None,
    facet_conn_master: np.ndarray | None = None,
    backend: str | None = None,
    format: str = "fluxsparse",
    return_blocks: bool = False,
):
    warn_float32_assembly_once(context="contact KKT assembly")
    """
    Assemble contact KKT block from coupling matrices.

    KKT is assembled as:
      B = [B_a, -B_b]
      Kuu = rho * (B^T B)
      KKT = [[Kuu, B^T], [B, 0]]

    multiplier:
    - ``family="nodal"``: lambda lives on interface nodal basis (B_a=M_aa, B_b=M_ab)
    - ``family="coarse_p1"``: lambda lives on user-supplied coarse P1 rows (B_*=C M_*)
    - ``family="dual_nodal"``: master-side dual nodal basis (B_a=I, B_b=pinv(M_aa) M_ab)
    - ``family="p0"``: lambda is facet-wise constant on master side (B_* = S * M_*)
    - ``family="p0_active"``/``family="p0_supermesh"``: use ``assemble_contact_constraint_operators`` and pass ``ops`` to the builder
    """
    backend = _infer_contact_backend(coupling_aa, coupling_ab, rho, multiplier, default="numpy") if backend is None else str(backend).lower()
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    multiplier_eff = ContactMultiplierSpace() if multiplier is None else multiplier
    mult_space, facet_conn_master, _ = _resolve_multiplier_spec(
        None,
        multiplier=multiplier_eff,
        facet_conn_master=facet_conn_master,
    )
    if mult_space in {"p0_active", "p0_supermesh"}:
        raise NotImplementedError(
            "assemble_contact_kkt(..., multiplier.family in {'p0_active', 'p0_supermesh'}) is not supported; "
            "use assemble_contact_constraint_operators(...) and CoupledSystemBuilder.add_contact_mortar(...)."
        )
    if format not in {"dense", "fluxsparse", "bcoo"}:
        raise ValueError("format must be 'dense', 'fluxsparse', or 'bcoo'")
    if return_blocks and format != "dense":
        raise ValueError("return_blocks=True is supported only with format='dense'.")

    if format != "dense":
        import jax

        if isinstance(rho, jax.core.Tracer):
            raise ValueError("format='fluxsparse'/'bcoo' currently requires rho to be a concrete scalar.")
        rows, cols, data, n_total = _kkt_coo_from_coupling(
            coupling_aa,
            coupling_ab,
            rho=float(rho),
            multiplier_space=mult_space,
            facet_conn_master=facet_conn_master,
            multiplier_value_dim=int(getattr(multiplier_eff, "value_dim", 1)),
            coarse_rank=getattr(multiplier_eff, "coarse_rank", None),
            coarse_projection=getattr(multiplier_eff, "coarse_projection", None),
            coarse_mode=getattr(multiplier_eff, "coarse_mode", None),
            coarse_energy_tol=getattr(multiplier_eff, "coarse_energy_tol", None),
            coarse_rtol=getattr(multiplier_eff, "coarse_rtol", None),
            coarse_max_rank=getattr(multiplier_eff, "coarse_max_rank", None),
            coarse_patch_ids=getattr(multiplier_eff, "coarse_patch_ids", None),
            coarse_basis=getattr(multiplier_eff, "coarse_basis", None),
            constraint_scaling=getattr(multiplier_eff, "constraint_scaling", "none"),
        )
        if format == "fluxsparse":
            from ..solver import FluxSparseMatrix

            return FluxSparseMatrix(rows, cols, data, n_dofs=n_total)
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        idx = jnp.stack([jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(cols, dtype=jnp.int32)], axis=-1)
        return jsparse.BCOO((jnp.asarray(data), idx), shape=(n_total, n_total))

    M_aa = _coo_to_dense(coupling_aa.rows, coupling_aa.cols, coupling_aa.data, coupling_aa.shape, backend=backend)
    M_ab = _coo_to_dense(coupling_ab.rows, coupling_ab.cols, coupling_ab.data, coupling_ab.shape, backend=backend)

    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np

    if mult_space == "nodal":
        B_a = M_aa
        B_b = M_ab
    elif mult_space == "dual_nodal":
        B_a, B_b = _dual_nodal_blocks_from_dense(M_aa, M_ab, backend=backend)
    elif mult_space == "coarse_p1":
        C = xp.asarray(multiplier_eff.coarse_basis)
        if int(C.shape[1]) != int(M_aa.shape[0]):
            raise ValueError("coarse_basis must have shape (n_coarse_nodes, n_master_nodes).")
        B_a = C @ M_aa
        B_b = C @ M_ab
    else:
        n_master_nodes = int(coupling_aa.shape[0])
        S_np = _p0_reduction_matrix_from_facets(facet_conn_master, n_master_nodes)
        S = xp.asarray(S_np)
        B_a = S @ M_aa
        B_b = S @ M_ab
        B_a, B_b = _apply_integrated_coarse_p0_groups(
            B_a,
            B_b,
            getattr(multiplier_eff, "coarse_patch_ids", None),
            backend=backend,
        )
    B_a = _expand_scalar_constraint_dense(
        B_a,
        value_dim=int(getattr(multiplier_eff, "value_dim", 1)),
        backend=backend,
    )
    B_b = _expand_scalar_constraint_dense(
        B_b,
        value_dim=int(getattr(multiplier_eff, "value_dim", 1)),
        backend=backend,
    )
    B_a, B_b = _apply_coarse_mortar_projection(B_a, B_b, multiplier_eff, backend=backend)
    B_a, B_b = _apply_constraint_row_scaling(
        B_a,
        B_b,
        getattr(multiplier_eff, "constraint_scaling", "none"),
        backend=backend,
    )

    B = xp.concatenate([B_a, -B_b], axis=1)
    Kuu = xp.asarray(rho) * (B.T @ B)
    n_lambda = int(B.shape[0])
    Zll = xp.zeros((n_lambda, n_lambda), dtype=Kuu.dtype)
    KKT = xp.block([[Kuu, B.T], [B, Zll]])

    if return_blocks:
        return KKT, B_a, B_b
    return KKT


def _params_with_updates(params: "WeakParams", **updates: Any) -> "WeakParams":
    data = dict(getattr(params, "_data", {}))
    if not data:
        data = dict(vars(params))
    data.update(updates)
    from ..core.weakform import Params
    return Params(**data)


def _make_al_u_hat_fn(
    contact: OneSidedContactSurfaceSpace,
    base_u_hat_fn: SurfaceHatFn,
    lambda_n: np.ndarray,
    *,
    alpha: float,
) -> SurfaceHatFn:
    coords = np.asarray(contact.surface_slave.coords, dtype=float)
    node_normals = _surface_node_normals(contact.surface_slave, normal_sign=float(contact.normal_sign))
    if node_normals is None:
        raise ValueError("surface normals are required for augmented-Lagrangian one-sided updates")
    corr_nodes = (np.asarray(lambda_n, dtype=float).reshape(-1, 1) / max(float(alpha), 1e-30)) * node_normals

    def _u_hat_eff(x_q: np.ndarray) -> np.ndarray:
        x_q = np.asarray(x_q, dtype=float)
        base = np.asarray(base_u_hat_fn(x_q), dtype=float)
        diffs = x_q[:, None, :] - coords[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        exact = d2 <= 1e-24
        weights = 1.0 / np.maximum(d2, 1e-24)
        weights /= np.sum(weights, axis=1, keepdims=True)
        corr = weights @ corr_nodes
        if np.any(exact):
            row_ids = np.nonzero(np.any(exact, axis=1))[0]
            for row in row_ids:
                corr[row] = corr_nodes[int(np.argmax(exact[row]))]
        return base - corr

    return _u_hat_eff

def update_contact_state_penalty(
    *,
    state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | npt.ArrayLike | None,
    contact_state: ContactState | None = None,
    geometry: str = "current",
    gap_n: npt.ArrayLike | None = None,
    active_mask: npt.ArrayLike | None = None,
    lambda_n: npt.ArrayLike | None = None,
    penalty_param: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContactState:
    """Update numeric penalty-contact diagnostics in a state-explicit form."""
    base = ContactState(interface_kind="penalty", geometry="reference") if contact_state is None else contact_state
    merged_metadata = dict(base.metadata)
    if metadata is not None:
        merged_metadata.update(dict(metadata))
    gap_np = None if gap_n is None else np.asarray(gap_n)
    active_mask_np = None if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active_mask_np is None and gap_np is not None:
        active_mask_np = np.asarray(gap_np < 0.0, dtype=bool)
    lambda_np = None if lambda_n is None else np.asarray(lambda_n)
    resolved_penalty = penalty_param if penalty_param is not None else base.penalty_param
    active_set = base.active_set
    if active_mask_np is not None:
        active_set = "active" if bool(np.any(active_mask_np)) else "inactive"
    return replace(
        base,
        geometry=str(geometry),
        iteration=int(base.iteration) + 1,
        active_set=active_set,
        field_summary=_summarize_contact_field_state(state),
        gap_n=gap_np,
        active_mask=active_mask_np,
        lambda_n=lambda_np,
        penalty_param=resolved_penalty,
        metadata=merged_metadata,
    )


def solve_contact_penalty_jax(
    contact,
    *,
    weak_form: MixedSurfaceResidualForm | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    state0: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
    params: "WeakParams",
    normal_source: str = "master",
    u_hat_fn: SurfaceHatFn | None = None,
    u_master: npt.ArrayLike | None = None,
    state_field: str | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    diagonal_shift: float = 0.0,
    contact_state: ContactState | None = None,
    geometry: str = "current",
    metadata: Mapping[str, Any] | None = None,
    state_updater: Callable[..., ContactState] | None = None,
) -> ContactSolveResult:
    """Solve a penalty-contact residual with a JAX-friendly dense Newton loop."""
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form

    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from ..solver.newton_jax import newton_solve_jax

    state_vec0, unravel_state = ravel_pytree(state0)

    def _primary_state_entry(u_state):
        if isinstance(u_state, Mapping):
            if state_field is not None:
                if state_field not in u_state:
                    raise KeyError(f"state_field {state_field!r} was not found in state.")
                return state_field, u_state[state_field]
            if len(u_state) != 1:
                raise ValueError("One-sided penalty solve requires a single-state mapping or explicit state_field.")
            key = next(iter(u_state))
            return str(key), u_state[key]
        if isinstance(u_state, Sequence) and not hasattr(u_state, "shape"):
            if len(u_state) != 1:
                raise ValueError("One-sided penalty solve requires a single state vector.")
            return "arg0", u_state[0]
        return "arg0", u_state

    is_onesided = isinstance(contact, OneSidedContactSurfaceSpace)
    if not is_onesided and res_form_eff is None:
        raise ValueError("weak_form or res_form is required.")
    if is_onesided and u_hat_fn is None:
        raise ValueError("u_hat_fn is required when contact is OneSidedContactSurfaceSpace.")

    u_master_arr = None if u_master is None else np.asarray(u_master)

    def _assemble_ops(u_vec):
        u_state = unravel_state(u_vec)
        if is_onesided:
            _field_name, u_local = _primary_state_entry(u_state)
            K, f = contact.assemble_bilinear(u_hat_fn, params, u_master=u_master_arr)
            K_jax = jnp.asarray(K)
            f_jax = jnp.asarray(f)
            u_local_jax = jnp.ravel(jnp.asarray(u_local))
            return PenaltyContactContribution(
                enforcement="nitsche",
                law="one_sided_normal_frictionless",
                formulation="penalty_consistent",
                residual=K_jax @ u_local_jax + f_jax,
                jacobian=K_jax,
            )
        return assemble_contact_penalty_operators(
            contact,
            weak_form=res_form_eff,
            state=u_state,
            params=params,
            backend="jax",
            normal_source=normal_source,
            sparse=False,
        )

    def residual_fn(u_vec, _params):
        _ = _params
        return jnp.ravel(jnp.asarray(_assemble_ops(u_vec).residual))

    def jacobian_fn(u_vec, _params):
        _ = _params
        J = _assemble_ops(u_vec).jacobian
        if hasattr(J, "to_dense"):
            J = J.to_dense()
        return jnp.asarray(J)

    u_sol_vec, info = newton_solve_jax(
        residual_fn,
        jacobian_fn,
        jnp.asarray(state_vec0),
        params,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        diagonal_shift=diagonal_shift,
    )
    state_sol = unravel_state(u_sol_vec)
    updater = update_contact_state_penalty if state_updater is None else state_updater
    gap_n = None
    active_mask = None
    if is_onesided:
        gap_n, active_mask = _onesided_gap_diagnostics(
            contact,
            state_sol,
            u_hat_fn=u_hat_fn,
            state_field=state_field,
        )
    contact_state_sol = updater(
        state=state_sol,
        contact_state=contact_state,
        geometry=geometry,
        gap_n=gap_n,
        active_mask=active_mask,
        penalty_param=float(getattr(params, "alpha", 0.0)) if hasattr(params, "alpha") else None,
        metadata=metadata,
    )
    return ContactSolveResult(
        state=state_sol,
        contact_state=contact_state_sol,
        converged=info.converged,
        iters=info.iters,
        residual_norm=info.residual_norm,
    )

def solve_contact_al_jax(
    contact,
    *,
    state0: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
    params: "WeakParams",
    u_hat_fn: SurfaceHatFn,
    state_field: str | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    outer_maxiter: int = 3,
    gap_tol: float = 1e-6,
    penalty_growth: float = 2.0,
    diagonal_shift: float = 0.0,
    contact_state: ContactState | None = None,
    geometry: str = "current",
    metadata: Mapping[str, Any] | None = None,
) -> ContactSolveResult:
    """Minimal one-sided augmented-Lagrangian outer loop built on penalty Newton solves."""
    if not isinstance(contact, OneSidedContactSurfaceSpace):
        raise TypeError("solve_contact_al_jax currently supports only OneSidedContactSurfaceSpace.")
    alpha = float(getattr(params, "alpha", 0.0))
    if alpha <= 0.0:
        raise ValueError("params.alpha must be positive for solve_contact_al_jax.")
    n_nodes = int(contact.surface_slave.n_nodes)
    lambda_n = (
        np.zeros((n_nodes,), dtype=float)
        if contact_state is None or contact_state.lambda_n is None
        else np.asarray(contact_state.lambda_n, dtype=float).reshape(-1)
    )
    state_curr = state0
    contact_state_curr = contact_state
    inner_result: ContactSolveResult | None = None
    converged = False

    for outer in range(int(outer_maxiter)):
        params_eff = _params_with_updates(params, alpha=alpha)
        u_hat_eff = _make_al_u_hat_fn(contact, u_hat_fn, lambda_n, alpha=alpha)
        inner_result = solve_contact_penalty_jax(
            contact,
            state0=state_curr,
            params=params_eff,
            u_hat_fn=u_hat_eff,
            state_field=state_field,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            diagonal_shift=diagonal_shift,
            contact_state=contact_state_curr,
            geometry=geometry,
            metadata=metadata,
        )
        gap_n = None if inner_result.contact_state.gap_n is None else np.asarray(inner_result.contact_state.gap_n, dtype=float)
        if gap_n is None:
            raise RuntimeError("solve_contact_al_jax requires one-sided gap diagnostics.")
        lambda_n = np.maximum(0.0, lambda_n - alpha * gap_n)
        active_mask = gap_n < 0.0
        contact_state_curr = update_contact_state_penalty(
            state=inner_result.state,
            contact_state=inner_result.contact_state,
            geometry=geometry,
            gap_n=gap_n,
            active_mask=active_mask,
            lambda_n=lambda_n,
            penalty_param=alpha,
            metadata={**dict(metadata or {}), "al_outer_iter": outer + 1},
        )
        state_curr = inner_result.state
        penetration = float(np.max(np.maximum(-gap_n, 0.0))) if gap_n.size else 0.0
        if penetration <= float(gap_tol):
            converged = True
            break
        alpha *= float(penalty_growth)

    if inner_result is None:
        raise RuntimeError("solve_contact_al_jax executed zero outer iterations.")
    return ContactSolveResult(
        state=inner_result.state,
        contact_state=contact_state_curr,
        converged=np.asarray(converged),
        iters=np.asarray(int(contact_state_curr.iteration)),
        residual_norm=inner_result.residual_norm,
    )


def assemble_pair_nitsche_supermesh(
    contact,
    params: "WeakParams",
    *,
    sparse: bool = False,
    normal_source: str = "master",
    use_penalty: float | None = None,
    use_traction: float | None = None,
    backend_fastpath: str = "numpy_local_kernel",
) -> PenaltyContactContribution:
    """Assemble pair-Nitsche contact terms over a prepared contact supermesh."""
    return _assemble_pair_nitsche_supermesh_impl(
        contact,
        params,
        contribution_cls=PenaltyContactContribution,
        sparse=sparse,
        normal_source=normal_source,
        use_penalty=use_penalty,
        use_traction=use_traction,
        backend_fastpath=backend_fastpath,
    )


def assemble_contact_operators(
    contact,
    *,
    enforcement: str | None = None,
    method: str | None = None,
    law: str | None = None,
    formulation: str | None = None,
    rho: float = 0.0,
    multiplier: ContactMultiplierSpace | None = None,
    backend: str | None = None,
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    """Unified public contact assembly entry that routes to penalty or constraint operators."""
    resolved = _resolve_contact_operator_enforcement(
        enforcement=enforcement,
        method=method,
        formulation=formulation,
        multiplier=multiplier,
    )
    if resolved == "penalty":
        formulation_key = None if formulation is None else str(formulation).lower().replace("-", "_")
        has_explicit_weak_form = any(value is not None for value in (weak_form, res_form, state, u))
        if formulation_key in {"pair_nitsche_penalty", "pair_nitsche", "nitsche_supermesh"} and not has_explicit_weak_form:
            if params is None:
                raise ValueError("params is required for formulation='pair_nitsche_penalty'.")
            return assemble_pair_nitsche_supermesh(
                contact,
                params,
                sparse=sparse,
                normal_source=normal_source,
            )
        use_backend = "jax" if backend is None else backend
        return assemble_contact_penalty_operators(
            contact,
            law=law,
            formulation=formulation,
            backend=use_backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )
    use_backend = "numpy" if backend is None else backend
    return assemble_contact_constraint_operators(
        contact,
        law=law,
        formulation=formulation,
        rho=rho,
        multiplier=multiplier,
        backend=use_backend,
        weak_form=weak_form,
        state=state,
        res_form=res_form,
        u=u,
        params=params,
        normal_source=normal_source,
        sparse=sparse,
        batch_jac=batch_jac,
    )


def assemble_multiplier(contact, **kwargs):
    """Public alias for assemble_contact_constraint_operators()."""
    return assemble_contact_constraint_operators(contact, **kwargs)


def assemble_penalty(contact, **kwargs):
    """Public alias for assemble_contact_penalty_operators()."""
    return assemble_contact_penalty_operators(contact, **kwargs)


__all__ = [
    "ContactSideSpec",
    "ContactSide",
    "OneSidedContact",
    "PreparedOneSidedContactInterface",
    "OneSidedContactSurfaceSpace",
    "PreparedContactInterface",
    "ContactSurfaceSpace",
    "PreparedOneToManyContactInterface",
    "OneToManyContactSurfaceSpace",
    "ContactOperators",
    "ContactConstraintDiagnostics",
    "ContactConstraintQualityIssue",
    "ContactConstraintQualityReport",
    "MortarContactProblemPair",
    "MortarContactProblem",
    "MultiplierContactContribution",
    "PenaltyContactContribution",
    "ContactState",
    "ContactKKTSolveInfo",
    "ContactKKTSolveResult",
    "AugmentedLagrangianState",
    "AugmentedLagrangianResult",
    "MultiplierSpec",
    "ContactMultiplierSpace",
    "coarse_p1_basis_from_node_groups",
    "coarse_p1_basis_from_surface_grid",
    "ContactPairSpec",
    "ContactGroupSpec",
    "OneSidedContactSpec",
    "ContactKKTSolveConfig",
    "UnilateralContactActiveSetRecord",
    "UnilateralContactActiveSetResult",
    "EmbeddingMap",
    "build_nodal_embedding_map",
    "build_barycentric_embedding_map",
    "build_barycentric_embedding_map_from_meshes",
    "assemble_embedding_constraint_matrix",
    "assemble_fixed_rigid_hub_constraint_matrix",
    "assemble_rigid_hub_constraint_matrix",
    "assemble_rbe2_constraint_matrix",
    "assemble_rbe3_constraint_matrix",
    "build_rbe3_weights",
    "make_pair_nitsche_supermesh_bilinear",
    "assemble_pair_nitsche_supermesh",
    "assemble_mortar_contact_problem",
    "assemble_contact_constraint_operators",
    "contact_constraint_matrix_diagnostics",
    "assess_contact_constraint_quality",
    "assemble_multiplier",
    "assemble_contact_operators",
    "assemble_contact_penalty_operators",
    "assemble_penalty",
    "assemble_contact_interface_residual",
    "assemble_contact_interface_jacobian",
    "assemble_contact_coupling_matrices",
    "assemble_contact_kkt",
    "solve_contact_kkt",
    "solve_contact_kkt_with_info",
    "solve_unilateral_contact_active_set_kkt",
    "solve_augmented_lagrangian_outer_loop",
    "facet_gap_values",
    "active_contact_facets",
]

# Phase-1 public naming aliases. These remain thin wrappers over the existing
# contact implementation until the state-explicit redesign is introduced.
PreparedContactInterface = ContactSurfaceSpace
PreparedOneToManyContactInterface = OneToManyContactSurfaceSpace
PreparedOneSidedContactInterface = OneSidedContactSurfaceSpace

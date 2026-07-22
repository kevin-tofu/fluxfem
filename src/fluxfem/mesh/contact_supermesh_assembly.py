from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from .contact_geometry import (
    _diag_quad_dump,
    _diag_quad_override,
    _diag_quad_source,
    _tri_area,
    _tri_quadrature,
)
from .contact_pair_basis import _gather_u_local, _global_dof_indices


@dataclass(eq=False)
class _JacobianTriangleGeometryData:
    detJ: float
    quad_pts: np.ndarray
    quad_w: np.ndarray
    quad_source: str
    facet_a: np.ndarray
    facet_b: np.ndarray
    x_q: np.ndarray


@dataclass(frozen=True)
class ContactAssemblyCallbacks:
    mixed_surface_space_aliases: Callable[..., tuple[str | None, str | None, str | None, str | None]]
    build_mixed_surface_context: Callable[..., Any]
    surface_u_elem_with_space_aliases: Callable[..., dict[str, Any]]
    compute_mixed_surface_local_jacobian: Callable[..., np.ndarray]


def _prepare_supermesh_jacobian_triangle_geometry(
    *,
    it: int,
    log_tri: bool,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    fa: int,
    fb: int,
    tol: float,
    skip_small_tri: bool,
    area_scale: float,
    facet_area_a: np.ndarray | None,
    facet_area_b: np.ndarray | None,
    diag_force: bool,
    diag_abs_detj: bool,
    guard: bool,
    skip_nonfinite: bool,
    detj_eps: float,
    quad_order: int,
    diag_qp_mode: str,
    diag_qp_path: str,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    tri_check: Callable[[str], None],
    trace_fn: Callable[[str], None],
    trace_time_fn: Callable[[str, float], None],
) -> _JacobianTriangleGeometryData | None:
    t_geom = time.perf_counter()
    area = _tri_area(a, b, c)
    if area <= tol:
        return None
    if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
        area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
        if area_ref > 0.0 and area < area_scale * area_ref:
            return None
    detJ = 2.0 * area
    if diag_force and diag_abs_detj:
        detJ = abs(detJ)
    if guard:
        if not np.isfinite(detJ):
            if log_tri:
                trace_fn(f"[CONTACT] tri {it} detJ non-finite; skip")
            if skip_nonfinite:
                return None
            raise RuntimeError(f"[CONTACT] tri {it} detJ non-finite")
        if detj_eps > 0.0 and abs(detJ) < detj_eps:
            if log_tri:
                trace_fn(f"[CONTACT] tri {it} detJ too small {detJ:.3e}; skip")
            return None
    if quad_order <= 0:
        quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        quad_w = np.array([0.5], dtype=float)
    else:
        quad_pts, quad_w = _tri_quadrature(quad_order)
    quad_source = "fluxfem"
    quad_override = _diag_quad_override(diag_force, diag_qp_mode, diag_qp_path)
    if quad_override is not None:
        quad_pts, quad_w = quad_override
        quad_source = _diag_quad_source("override")
    _diag_quad_dump(diag_force, diag_qp_mode, diag_qp_path, quad_pts, quad_w)

    facet_a = facets_a[int(fa)]
    facet_b = facets_b[int(fb)]
    x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)
    if guard and not np.isfinite(x_q).all():
        if log_tri:
            trace_fn(f"[CONTACT] tri {it} x_q non-finite; skip")
        if skip_nonfinite:
            return None
        raise RuntimeError(f"[CONTACT] tri {it} x_q non-finite")
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} geom_done", t_geom)
    tri_check("geom_done")
    return _JacobianTriangleGeometryData(
        detJ=float(detJ),
        quad_pts=quad_pts,
        quad_w=quad_w,
        quad_source=quad_source,
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
    )


def _accumulate_projection_jacobian_batch(
    *,
    batch: dict[str, np.ndarray],
    field_a: str,
    field_b: str,
    value_dim_a: int,
    value_dim_b: int,
    u_a: np.ndarray,
    u_b: np.ndarray,
    offset_a: int,
    offset_b: int,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    sparse: bool,
    rows: list[int],
    cols: list[int],
    data: list[float],
    K_dense: np.ndarray | None,
    callbacks: ContactAssemblyCallbacks,
) -> None:
    test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = callbacks.mixed_surface_space_aliases(
        res_form,
        field_a=field_a,
        field_b=field_b,
    )
    Na = batch["Na"]
    Nb = batch["Nb"]
    gradNa = batch["gradNa"]
    gradNb = batch["gradNb"]
    nodes_a = batch["nodes_a"]
    nodes_b = batch["nodes_b"]
    normal_q = batch["normal"]

    ctx = callbacks.build_mixed_surface_context(
        field_a=field_a,
        field_b=field_b,
        test_space_key_a=test_space_key_a,
        test_space_key_b=test_space_key_b,
        unknown_space_key_a=unknown_space_key_a,
        unknown_space_key_b=unknown_space_key_b,
        test_Na=Na,
        test_Nb=Nb,
        trial_Na=Na,
        trial_Nb=Nb,
        test_gradNa=gradNa,
        test_gradNb=gradNb,
        trial_gradNa=gradNa,
        trial_gradNb=gradNb,
        test_value_dim_a=value_dim_a,
        test_value_dim_b=value_dim_b,
        trial_value_dim_a=value_dim_a,
        trial_value_dim_b=value_dim_b,
        x_q=batch["x_q"],
        w=batch["w"],
        detJ=batch["detJ"],
        normal_q=normal_q,
    )

    u_elem = callbacks.surface_u_elem_with_space_aliases(
        field_a=field_a,
        field_b=field_b,
        unknown_space_key_a=unknown_space_key_a,
        unknown_space_key_b=unknown_space_key_b,
        u_local_a=_gather_u_local(u_a, nodes_a, value_dim_a),
        u_local_b=_gather_u_local(u_b, nodes_b, value_dim_b),
    )
    u_local = np.concatenate([u_elem[field_a], u_elem[field_b]], axis=0)
    sizes = (u_elem[field_a].shape[0], u_elem[field_b].shape[0])
    slices = {
        field_a: slice(0, sizes[0]),
        field_b: slice(sizes[0], sizes[0] + sizes[1]),
    }

    J_local_np = callbacks.compute_mixed_surface_local_jacobian(
        u_local=np.asarray(u_local, dtype=float),
        backend=backend,
        fd_eps=fd_eps,
        fd_mode=fd_mode,
        fd_block_size=fd_block_size,
        field_a=field_a,
        field_b=field_b,
        slices=slices,
        res_form=res_form,
        ctx=ctx,
        params=params,
        includes_measure=includes_measure,
    )

    dofs_a = _global_dof_indices(nodes_a, value_dim_a, int(offset_a))
    dofs_b = _global_dof_indices(nodes_b, value_dim_b, int(offset_b))
    dofs = np.concatenate([dofs_a, dofs_b], axis=0)
    for i, gi in enumerate(dofs):
        for j, gj in enumerate(dofs):
            val = float(J_local_np[i, j])
            if sparse:
                rows.append(int(gi))
                cols.append(int(gj))
                data.append(val)
            else:
                assert K_dense is not None
                K_dense[int(gi), int(gj)] += val


def _apply_projection_jacobian_batches(
    *,
    batches: list[dict[str, np.ndarray]],
    field_a: str,
    field_b: str,
    value_dim_a: int,
    value_dim_b: int,
    u_a: np.ndarray,
    u_b: np.ndarray,
    offset_a: int,
    offset_b: int,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    sparse: bool,
    rows: list[int],
    cols: list[int],
    data: list[float],
    K_dense: np.ndarray | None,
    callbacks: ContactAssemblyCallbacks,
) -> None:
    u_a_np = np.asarray(u_a, dtype=float)
    u_b_np = np.asarray(u_b, dtype=float)
    for batch in batches:
        _accumulate_projection_jacobian_batch(
            batch=batch,
            field_a=field_a,
            field_b=field_b,
            value_dim_a=value_dim_a,
            value_dim_b=value_dim_b,
            u_a=u_a_np,
            u_b=u_b_np,
            offset_a=int(offset_a),
            offset_b=int(offset_b),
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            res_form=res_form,
            params=params,
            includes_measure=includes_measure,
            sparse=sparse,
            rows=rows,
            cols=cols,
            data=data,
            K_dense=K_dense,
            callbacks=callbacks,
        )

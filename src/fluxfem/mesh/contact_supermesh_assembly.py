from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .contact_geometry import (
    _diag_contact_projection,
    _diag_quad_dump,
    _diag_quad_override,
    _diag_quad_source,
    _facet_label,
    _local_indices,
    _proj_diag_set_context,
    _tri_area,
    _tri_quadrature,
)
from .contact_nitsche import _fast_pair_nitsche_penalty_local_matrix
from .contact_pair_basis import (
    _SupermeshPairBasisData,
    _gather_u_local,
    _global_dof_indices,
    _merge_trial_pair_basis_data,
    _same_optional_int_array,
)


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


_DEBUG_CONTACT_MAP_ONCE = False
_DEBUG_CONTACT_N_ONCE = False


def _resolve_contact_normal(
    *,
    facet_id_a: int,
    facet_id_b: int,
    normals_a: np.ndarray | None,
    normals_b: np.ndarray | None,
    normal_source: str,
    normal_sign: float,
    tol: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    na = normals_a[int(facet_id_a)] if normals_a is not None else None
    nb = normals_b[int(facet_id_b)] if normals_b is not None else None
    if normal_source == "a":
        normal = na
    elif normal_source == "b":
        normal = nb
    else:
        if na is not None and nb is not None:
            avg = na + nb
            norm = np.linalg.norm(avg)
            normal = avg / norm if norm > tol else na
        else:
            normal = na if na is not None else nb
    if normal is not None:
        normal = normal_sign * normal
    return normal, na, nb


def _accumulate_supermesh_jacobian_triangle_core(
    *,
    it: int,
    log_tri: bool,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    quad_pts: np.ndarray,
    quad_w: np.ndarray,
    quad_source: str,
    detJ: float,
    tol: float,
    pair_basis_builder: Callable[..., _SupermeshPairBasisData],
    trial_pair_basis_builder: Callable[..., _SupermeshPairBasisData],
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    trial_value_dim_a: int,
    trial_value_dim_b: int,
    dof_source: str,
    grad_source: str,
    space_mode_a: str,
    space_mode_b: str,
    trial_space_mode_a: str,
    trial_space_mode_b: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    trial_facet_dofs_a: np.ndarray | None,
    trial_facet_dofs_b: np.ndarray | None,
    proj_diag: bool,
    diag_map: bool,
    diag_n: bool,
    diag_force: bool,
    diag_facet: int,
    diag_max_q: int,
    guard: bool,
    skip_nonfinite: bool,
    normal_source: str,
    normal_sign: float,
    normals_a: np.ndarray | None,
    normals_b: np.ndarray | None,
    field_a: str,
    field_b: str,
    u_a: np.ndarray,
    u_b: np.ndarray,
    res_form: Callable[..., Any],
    params: Any,
    includes_measure: dict[str, bool],
    offset_a: int,
    offset_b: int,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    sparse: bool,
    rows: list[int],
    cols: list[int],
    data: list[float],
    K_dense: np.ndarray | None,
    callbacks: ContactAssemblyCallbacks,
    tri_check: Callable[[str], None],
    trace_fn: Callable[[str], None],
    trace_time_fn: Callable[[str, float], None],
) -> None:
    pair_data = pair_basis_builder(
        fa=int(fa),
        fb=int(fb),
        facet_a=facet_a,
        facet_b=facet_b,
        x_q=x_q,
        coords_a=coords_a,
        coords_b=coords_b,
        facets_a=facets_a,
        facets_b=facets_b,
        value_dim_a=value_dim_a,
        value_dim_b=value_dim_b,
        dof_source=dof_source,
        grad_source=grad_source,
        use_elem_a=use_elem_a,
        use_elem_b=use_elem_b,
        elem_conn_a=elem_conn_a,
        elem_conn_b=elem_conn_b,
        facet_to_elem_a=facet_to_elem_a,
        facet_to_elem_b=facet_to_elem_b,
        facet_dofs_a=facet_dofs_a,
        facet_dofs_b=facet_dofs_b,
        tol=tol,
    )
    if (
        int(trial_value_dim_a) != int(value_dim_a)
        or int(trial_value_dim_b) != int(value_dim_b)
        or str(trial_space_mode_a) != str(space_mode_a)
        or str(trial_space_mode_b) != str(space_mode_b)
        or not _same_optional_int_array(trial_facet_dofs_a, facet_dofs_a)
        or not _same_optional_int_array(trial_facet_dofs_b, facet_dofs_b)
    ):
        trial_pair = trial_pair_basis_builder(
            fa=int(fa),
            fb=int(fb),
            facet_a=facet_a,
            facet_b=facet_b,
            x_q=x_q,
            coords_a=coords_a,
            coords_b=coords_b,
            facets_a=facets_a,
            facets_b=facets_b,
            value_dim_a=trial_value_dim_a,
            value_dim_b=trial_value_dim_b,
            dof_source=dof_source,
            grad_source=grad_source,
            use_elem_a=use_elem_a,
            use_elem_b=use_elem_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            facet_dofs_a=trial_facet_dofs_a,
            facet_dofs_b=trial_facet_dofs_b,
            tol=tol,
        )
        pair_data = _merge_trial_pair_basis_data(pair_data, trial_pair)
    elem_id_a = pair_data.elem_id_a
    elem_id_b = pair_data.elem_id_b
    local_a = pair_data.local_a
    local_b = pair_data.local_b
    if proj_diag:
        _proj_diag_set_context(
            fa=int(fa),
            fb=int(fb),
            face_a=_facet_label(facet_a),
            face_b=_facet_label(facet_b),
            elem_a=elem_id_a,
            elem_b=elem_id_b,
        )

    t_basis = time.perf_counter()
    test_Na = pair_data.test_Na if pair_data.test_Na is not None else pair_data.Na
    test_Nb = pair_data.test_Nb if pair_data.test_Nb is not None else pair_data.Nb
    trial_Na = pair_data.trial_Na if pair_data.trial_Na is not None else pair_data.Na
    trial_Nb = pair_data.trial_Nb if pair_data.trial_Nb is not None else pair_data.Nb
    test_gradNa = pair_data.test_gradNa if pair_data.test_gradNa is not None else pair_data.gradNa
    test_gradNb = pair_data.test_gradNb if pair_data.test_gradNb is not None else pair_data.gradNb
    trial_gradNa = pair_data.trial_gradNa if pair_data.trial_gradNa is not None else pair_data.gradNa
    trial_gradNb = pair_data.trial_gradNb if pair_data.trial_gradNb is not None else pair_data.gradNb
    test_dofs_local_a = pair_data.test_dofs_local_a if pair_data.test_dofs_local_a is not None else pair_data.dofs_local_a
    test_dofs_local_b = pair_data.test_dofs_local_b if pair_data.test_dofs_local_b is not None else pair_data.dofs_local_b
    trial_dofs_local_a = pair_data.trial_dofs_local_a if pair_data.trial_dofs_local_a is not None else pair_data.dofs_local_a
    trial_dofs_local_b = pair_data.trial_dofs_local_b if pair_data.trial_dofs_local_b is not None else pair_data.dofs_local_b
    nodes_a = pair_data.nodes_a
    nodes_b = pair_data.nodes_b
    if guard and (not np.isfinite(test_Na).all() or not np.isfinite(test_Nb).all()):
        if log_tri:
            trace_fn(f"[CONTACT] tri {it} N non-finite; skip")
        if skip_nonfinite:
            return
        raise RuntimeError(f"[CONTACT] tri {it} N non-finite")
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} basis_done", t_basis)
    tri_check("basis_done")

    global _DEBUG_CONTACT_MAP_ONCE
    if diag_map and not _DEBUG_CONTACT_MAP_ONCE:
        if use_elem_a:
            assert facet_to_elem_a is not None
            elem_id_a = int(facet_to_elem_a[int(fa)])
        else:
            elem_id_a = -1
        if use_elem_b:
            assert facet_to_elem_b is not None
            elem_id_b = int(facet_to_elem_b[int(fb)])
        else:
            elem_id_b = -1
        print("[fluxfem][diag][contact-map] first facet")
        print(f"  fa={int(fa)} fb={int(fb)} elem_a={elem_id_a} elem_b={elem_id_b}")
        print(f"  facet_nodes_a={facet_a.tolist()}")
        print(f"  facet_nodes_b={facet_b.tolist()}")
        print(f"  facet_coords_a={coords_a[facet_a].tolist()}")
        print(f"  facet_coords_b={coords_b[facet_b].tolist()}")
        if elem_nodes_a is not None:
            if local_a is None:
                local_a = _local_indices(elem_nodes_a, facet_a)
            match_a = np.all(elem_nodes_a[local_a] == facet_a)
            print(f"  elem_nodes_a={elem_nodes_a.tolist()}")
            print(f"  local_indices_a={local_a.tolist()} match={bool(match_a)}")
        if elem_nodes_b is not None:
            if local_b is None:
                local_b = _local_indices(elem_nodes_b, facet_b)
            match_b = np.all(elem_nodes_b[local_b] == facet_b)
            print(f"  elem_nodes_b={elem_nodes_b.tolist()}")
            print(f"  local_indices_b={local_b.tolist()} match={bool(match_b)}")
        _DEBUG_CONTACT_MAP_ONCE = True

    global _DEBUG_CONTACT_N_ONCE
    if diag_n and not _DEBUG_CONTACT_N_ONCE:
        dofs_a = int(offset_a) + np.asarray(test_dofs_local_a, dtype=int)
        dofs_b = int(offset_b) + np.asarray(test_dofs_local_b, dtype=int)
        samples = min(3, Na.shape[0])
        print("[fluxfem][diag][contact-n] first facet q-points")
        print(f"  nodes_a={nodes_a.tolist()} nodes_b={nodes_b.tolist()}")
        print(f"  dofs_a={dofs_a.tolist()} dofs_b={dofs_b.tolist()}")
        for qi in range(samples):
            print(f"  q{qi} x={x_q[qi].tolist()} Na={test_Na[qi].tolist()} Nb={test_Nb[qi].tolist()}")
        _DEBUG_CONTACT_N_ONCE = True

    normal = None
    na = normals_a[int(fa)] if normals_a is not None else None
    nb = normals_b[int(fb)] if normals_b is not None else None
    if normal_source == "a":
        normal = na
    elif normal_source == "b":
        normal = nb
    else:
        if na is not None and nb is not None:
            avg = na + nb
            norm = np.linalg.norm(avg)
            normal = avg / norm if norm > tol else na
        else:
            normal = na if na is not None else nb
    if normal is not None:
        normal = normal_sign * normal

    normal_q = None if normal is None else np.repeat(normal[None, :], quad_pts.shape[0], axis=0)
    t_jac = time.perf_counter()
    formulation_tag = getattr(res_form, "_ff_contact_formulation", None)
    fastpath_tag = getattr(res_form, "_ff_contact_backend_fastpath", None)
    use_fast_pair_nitsche = (
        backend == "numpy"
        and formulation_tag == "pair_nitsche_penalty"
        and fastpath_tag == "numpy_local_kernel"
    )
    if use_fast_pair_nitsche:
        if normal_q is None:
            raise ValueError("pair_nitsche_penalty fast path requires surface normals.")
        J_local_np = _fast_pair_nitsche_penalty_local_matrix(
            Na=np.asarray(test_Na, dtype=float),
            Nb=np.asarray(test_Nb, dtype=float),
            gradNa=np.asarray(test_gradNa, dtype=float),
            gradNb=np.asarray(test_gradNb, dtype=float),
            normal_q=np.asarray(normal_q, dtype=float),
            w=np.asarray(quad_w, dtype=float),
            detJ=np.array([detJ], dtype=float),
            alpha=float(getattr(params, "alpha")),
            inv_h=float(getattr(params, "inv_h")),
            lam=float(getattr(params, "lam")),
            mu=float(getattr(params, "mu")),
            use_penalty=float(getattr(params, "use_penalty", 1.0)),
            use_traction=float(getattr(params, "use_traction", 1.0)),
            value_dim_a=int(value_dim_a),
            value_dim_b=int(value_dim_b),
        )
    else:
        test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = (
            callbacks.mixed_surface_space_aliases(
                res_form,
                field_a=field_a,
                field_b=field_b,
            )
        )
        ctx = callbacks.build_mixed_surface_context(
            field_a=field_a,
            field_b=field_b,
            test_space_key_a=test_space_key_a,
            test_space_key_b=test_space_key_b,
            unknown_space_key_a=unknown_space_key_a,
            unknown_space_key_b=unknown_space_key_b,
            test_Na=test_Na,
            test_Nb=test_Nb,
            trial_Na=trial_Na,
            trial_Nb=trial_Nb,
            test_gradNa=test_gradNa,
            test_gradNb=test_gradNb,
            trial_gradNa=trial_gradNa,
            trial_gradNb=trial_gradNb,
            test_value_dim_a=value_dim_a,
            test_value_dim_b=value_dim_b,
            trial_value_dim_a=value_dim_a,
            trial_value_dim_b=value_dim_b,
            x_q=x_q,
            w=quad_w,
            detJ=np.array([detJ], dtype=float),
            normal_q=normal_q,
        )
        u_elem = callbacks.surface_u_elem_with_space_aliases(
            field_a=field_a,
            field_b=field_b,
            unknown_space_key_a=unknown_space_key_a,
            unknown_space_key_b=unknown_space_key_b,
            u_local_a=np.asarray(u_a, dtype=float)[np.asarray(trial_dofs_local_a, dtype=int)],
            u_local_b=np.asarray(u_b, dtype=float)[np.asarray(trial_dofs_local_b, dtype=int)],
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
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} jac_done", t_jac)
    tri_check("jac_done")

    row_dofs_a = int(offset_a) + np.asarray(test_dofs_local_a, dtype=int)
    row_dofs_b = int(offset_b) + np.asarray(test_dofs_local_b, dtype=int)
    col_dofs_a = int(offset_a) + np.asarray(trial_dofs_local_a, dtype=int)
    col_dofs_b = int(offset_b) + np.asarray(trial_dofs_local_b, dtype=int)
    if diag_force:
        _diag_contact_projection(
            fa=int(fa),
            fb=int(fb),
            quad_pts=quad_pts,
            quad_w=quad_w,
            x_q=x_q,
            Na=test_Na,
            Nb=test_Nb,
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            dofs_a=row_dofs_a,
            dofs_b=row_dofs_b,
            elem_coords_a=None,
            elem_coords_b=None,
            na=na,
            nb=nb,
            normal=normal,
            normal_source=normal_source,
            normal_sign=normal_sign,
            detJ=detJ,
            diag_facet=diag_facet,
            diag_max_q=diag_max_q,
            quad_source=quad_source,
            tol=tol,
        )
    t_scatter = time.perf_counter()
    row_dofs = np.concatenate([row_dofs_a, row_dofs_b], axis=0)
    col_dofs = np.concatenate([col_dofs_a, col_dofs_b], axis=0)
    if sparse:
        n_row_ldofs = int(row_dofs.shape[0])
        n_col_ldofs = int(col_dofs.shape[0])
        rows.extend(np.repeat(row_dofs, n_col_ldofs).tolist())
        cols.extend(np.tile(col_dofs, n_row_ldofs).tolist())
        data.extend(J_local_np.reshape(-1).tolist())
    else:
        assert K_dense is not None
        if backend == "jax":
            import jax.numpy as jnp

            row_idx = jnp.asarray(row_dofs, dtype=jnp.int32).reshape(-1, 1)
            col_idx = jnp.asarray(col_dofs, dtype=jnp.int32).reshape(1, -1)
            K_dense = K_dense.at[row_idx, col_idx].add(jnp.asarray(J_local_np))
        else:
            K_dense[np.ix_(row_dofs, col_dofs)] += J_local_np
        return K_dense
    if log_tri:
        trace_time_fn(f"[CONTACT] tri {it} scatter_done", t_scatter)
    tri_check("scatter_done")
    return K_dense



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

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from .surface import SurfaceMesh
from .contact_geometry import (
    _SupermeshTriangleQuadratureCache,
    _diag_contact_projection,
    _diag_quad_dump,
    _diag_quad_override,
    _diag_quad_source,
    _facet_area_estimate,
    _facet_label,
    _facet_shape_values,
    _facet_triangles,
    _iter_supermesh_tris,
    _local_indices,
    _point_in_tri,
    _proj_diag_enabled,
    _proj_diag_report,
    _proj_diag_reset,
    _proj_diag_set_context,
    _project_point_to_quad9,
    _project_point_to_tri6,
    _surface_gradN,
    _tet_gradN_at_points,
    _tri3_shape_values_jax,
    _tri_area,
    _tri_centroid,
    _tri_quadrature,
    _volume_shape_values_at_points,
    build_supermesh_triangle_quadrature_cache,
    facet_shape_values,
    facet_triangles,
    hex27_gradN,
    map_surface_facets_to_hex_elements,
    map_surface_facets_to_tet_elements,
    quad9_shape_values,
    quad_shape_and_local,
    tri_area,
    tri_quadrature,
    volume_shape_values_at_points,
)
from .contact_pair_basis import (
    _SupermeshPairBasisData,
    _field_n_dofs,
    _gather_u_local,
    _global_dof_indices,
    _merge_trial_pair_basis_data,
    _same_optional_int_array,
    _select_supermesh_pair_basis_builder,
)
from .contact_supermesh_assembly import (
    ContactAssemblyCallbacks,
    _JacobianTriangleGeometryData,
    _accumulate_supermesh_jacobian_triangle_core,
    _accumulate_supermesh_residual_triangle,
    _apply_projection_jacobian_batches,
    _prepare_supermesh_jacobian_triangle_geometry,
    _projection_surface_batches,
    _resolve_contact_normal,
)
from .contact_nitsche import (
    _fast_pair_nitsche_penalty_local_matrix,
    _get_direct_pair_nitsche_batch_fun,
)
from ..core.forms import FormFieldLike
if TYPE_CHECKING:
    from ..core.forms import FieldPair
    from ..core.weakform import Params as WeakParams


@dataclass(eq=False)
class _SurfaceBasis:
    dofs_per_node: int


@dataclass(eq=False)
class SurfaceMixedFormField:
    """Surface form field for mixed weak-form evaluation."""
    N: np.ndarray
    gradN: np.ndarray | None
    value_dim: int
    basis: _SurfaceBasis


@dataclass(eq=False)
class SurfaceMixedFormContext:
    """Surface mixed context for weak-form evaluation on supermesh."""
    bindings: dict[str, "FieldPair"]
    x_q: np.ndarray
    w: np.ndarray
    detJ: np.ndarray
    normal: np.ndarray | None = None
    spaces: dict[str, "FieldPair"] | None = None


def _make_surface_field_pair(
    *,
    test_N: np.ndarray,
    test_gradN: np.ndarray | None,
    trial_N: np.ndarray,
    trial_gradN: np.ndarray | None,
    test_value_dim: int,
    trial_value_dim: int,
) -> "FieldPair":
    from ..core.forms import FieldPair

    test_field = SurfaceMixedFormField(
        N=test_N,
        gradN=test_gradN,
        value_dim=test_value_dim,
        basis=_SurfaceBasis(dofs_per_node=test_value_dim),
    )
    trial_field = SurfaceMixedFormField(
        N=trial_N,
        gradN=trial_gradN,
        value_dim=trial_value_dim,
        basis=_SurfaceBasis(dofs_per_node=trial_value_dim),
    )
    return FieldPair(
        test=cast("FormFieldLike", test_field),
        trial=cast("FormFieldLike", trial_field),
        unknown=cast("FormFieldLike", trial_field),
    )






_DEBUG_SURFACE_SOURCE_ONCE = False


def _contact_interface_dbg_enabled() -> bool:
    return os.getenv("FLUXFEM_CONTACT_INTERFACE_DEBUG", "0") not in ("0", "", "false", "False")


def _contact_interface_dbg(msg: str) -> None:
    if _contact_interface_dbg_enabled():
        print(msg, flush=True)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in ("0", "", "false", "False")


@dataclass(eq=False)
class ContactCouplingMatrix:
    """COO storage for contact coupling matrices (can be rectangular)."""
    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]


def _is_jax_value(x: Any) -> bool:
    return isinstance(x, jax.core.Tracer) or isinstance(x, jax.Array)


def _uses_jax_geometry(*xs: Any) -> bool:
    for x in xs:
        if isinstance(x, jax.core.Tracer):
            return True
    return False


def _has_jax_leaves(x: Any) -> bool:
    try:
        leaves = jax.tree_util.tree_leaves(x)
    except Exception:
        leaves = [x]
    return any(_is_jax_value(leaf) for leaf in leaves)


def _contact_assembly_callbacks() -> ContactAssemblyCallbacks:
    return ContactAssemblyCallbacks(
        mixed_surface_space_aliases=_mixed_surface_space_aliases,
        build_mixed_surface_context=_build_mixed_surface_context,
        surface_u_elem_with_space_aliases=_surface_u_elem_with_space_aliases,
        compute_mixed_surface_local_jacobian=_compute_mixed_surface_local_jacobian,
        reduce_surface_residual_jax=_reduce_surface_residual_jax,
        reduce_surface_residual_numpy=_reduce_surface_residual_numpy,
    )




def _validate_contact_interface_space_setup(
    *,
    grad_source: str,
    dof_source: str,
    space_mode_a: str,
    space_mode_b: str,
) -> tuple[bool, bool]:
    if grad_source not in {"volume", "surface"}:
        raise ValueError("grad_source must be 'volume' or 'surface'")
    if dof_source not in {"surface", "volume"}:
        raise ValueError("dof_source must be 'surface' or 'volume'")
    if space_mode_a not in {"nodal", "p0"}:
        raise ValueError("space_mode_a must be 'nodal' or 'p0'")
    if space_mode_b not in {"nodal", "p0"}:
        raise ValueError("space_mode_b must be 'nodal' or 'p0'")
    if dof_source == "volume" and grad_source == "surface":
        raise ValueError("dof_source 'volume' requires grad_source 'volume'")
    return space_mode_a == "p0", space_mode_b == "p0"


def _resolve_normal_source_option(
    *,
    normal_source: str,
    normal_from: str | None,
    master_field: str | None,
    field_a: str,
    field_b: str,
    diag_force: bool,
    diag_normal: str,
) -> str:
    resolved = normal_source
    if normal_from is not None:
        if normal_from not in {"master", "slave"}:
            raise ValueError("normal_from must be 'master' or 'slave'")
        master_name = field_a if master_field is None else master_field
        if master_name not in {field_a, field_b}:
            raise ValueError("master_field must match field_a or field_b")
        if normal_from == "master":
            resolved = "a" if master_name == field_a else "b"
        else:
            resolved = "b" if master_name == field_a else "a"
    if diag_force and diag_normal:
        resolved = diag_normal
    if resolved not in {"a", "b", "avg", "master", "slave"}:
        raise ValueError("normal_source must be 'a', 'b', 'avg', 'master', or 'slave'")
    if resolved == "master":
        resolved = "a" if (master_field is None or master_field == field_a) else "b"
    if resolved == "slave":
        resolved = "b" if (master_field is None or master_field == field_a) else "a"
    return resolved



def _build_mixed_surface_context(
    *,
    field_a: str,
    field_b: str,
    test_space_key_a: str | None,
    test_space_key_b: str | None,
    unknown_space_key_a: str | None,
    unknown_space_key_b: str | None,
    test_Na: np.ndarray,
    test_Nb: np.ndarray,
    trial_Na: np.ndarray,
    trial_Nb: np.ndarray,
    test_gradNa: np.ndarray | None,
    test_gradNb: np.ndarray | None,
    trial_gradNa: np.ndarray | None,
    trial_gradNb: np.ndarray | None,
    test_value_dim_a: int,
    test_value_dim_b: int,
    trial_value_dim_a: int,
    trial_value_dim_b: int,
    x_q: np.ndarray,
    w: np.ndarray,
    detJ: np.ndarray,
    normal_q: np.ndarray | None,
) -> SurfaceMixedFormContext:
    fields = {
        field_a: _make_surface_field_pair(
            test_N=test_Na,
            test_gradN=test_gradNa,
            trial_N=trial_Na,
            trial_gradN=trial_gradNa,
            test_value_dim=test_value_dim_a,
            trial_value_dim=trial_value_dim_a,
        ),
        field_b: _make_surface_field_pair(
            test_N=test_Nb,
            test_gradN=test_gradNb,
            trial_N=trial_Nb,
            trial_gradN=trial_gradNb,
            test_value_dim=test_value_dim_b,
            trial_value_dim=trial_value_dim_b,
        ),
    }
    spaces = dict(fields)
    for key in (test_space_key_a, unknown_space_key_a):
        if key is not None:
            spaces[key] = fields[field_a]
    for key in (test_space_key_b, unknown_space_key_b):
        if key is not None:
            spaces[key] = fields[field_b]
    return SurfaceMixedFormContext(
        bindings=fields,
        x_q=x_q,
        w=w,
        detJ=detJ,
        normal=normal_q,
        spaces=spaces,
    )


def _surface_u_elem_with_space_aliases(
    *,
    field_a: str,
    field_b: str,
    unknown_space_key_a: str | None,
    unknown_space_key_b: str | None,
    u_local_a: np.ndarray,
    u_local_b: np.ndarray,
) -> dict[str, np.ndarray]:
    u_elem = {
        field_a: u_local_a,
        field_b: u_local_b,
    }
    if unknown_space_key_a is not None:
        u_elem[unknown_space_key_a] = u_elem[field_a]
    if unknown_space_key_b is not None:
        u_elem[unknown_space_key_b] = u_elem[field_b]
    return u_elem


def _surface_local_u_dict(
    *,
    u_vec: np.ndarray | jnp.ndarray,
    slices: dict[str, slice],
    field_a: str,
    field_b: str,
    ctx: SurfaceMixedFormContext,
) -> dict[str, np.ndarray | jnp.ndarray]:
    u_dict = {name: u_vec[slices[name]] for name in (field_a, field_b)}
    if ctx.spaces is not None:
        for key, pair in ctx.spaces.items():
            if key in u_dict:
                continue
            if pair is ctx.bindings[field_a]:
                u_dict[key] = u_dict[field_a]
            elif pair is ctx.bindings[field_b]:
                u_dict[key] = u_dict[field_b]
    return u_dict


def _mixed_surface_space_aliases(
    res_form: Callable[..., Any],
    *,
    field_a: str,
    field_b: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    test_space_keys = getattr(res_form, "_test_space_by_target", {})
    unknown_space_keys = getattr(res_form, "_unknown_space_by_target", {})
    legacy_space_keys = getattr(res_form, "_space_by_target", {})
    test_space_key_a = test_space_keys.get(field_a, legacy_space_keys.get(field_a))
    test_space_key_b = test_space_keys.get(field_b, legacy_space_keys.get(field_b))
    unknown_space_key_a = unknown_space_keys.get(field_a, legacy_space_keys.get(field_a))
    unknown_space_key_b = unknown_space_keys.get(field_b, legacy_space_keys.get(field_b))
    return test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b


def _reduce_surface_residual_jax(
    fe_field: Any,
    *,
    includes_measure: bool,
    w: np.ndarray,
    detJ: np.ndarray,
) -> jnp.ndarray:
    if includes_measure:
        return jnp.sum(jnp.asarray(fe_field), axis=0)
    wJ = jnp.asarray(w) * jnp.asarray(detJ)
    return jnp.einsum("qi,q->i", jnp.asarray(fe_field), wJ)


def _reduce_surface_residual_numpy(
    fe_field: Any,
    *,
    includes_measure: bool,
    w: np.ndarray,
    detJ: np.ndarray,
) -> np.ndarray:
    if includes_measure:
        return np.sum(np.asarray(fe_field), axis=0)
    wJ = np.asarray(w) * np.asarray(detJ)
    return np.einsum("qi...,q->i...", np.asarray(fe_field), wJ)


def _mixed_surface_local_residual_jax(
    *,
    u_vec: jnp.ndarray,
    slices: dict[str, slice],
    field_a: str,
    field_b: str,
    res_form: Callable[..., Any],
    ctx: SurfaceMixedFormContext,
    params: Any,
    includes_measure: dict[str, bool],
) -> jnp.ndarray:
    u_dict = _surface_local_u_dict(
        u_vec=u_vec,
        slices=slices,
        field_a=field_a,
        field_b=field_b,
        ctx=ctx,
    )
    fe_q = res_form(ctx, u_dict, params)
    res_parts = []
    for name in (field_a, field_b):
        fe_field = fe_q[name]
        fe = _reduce_surface_residual_jax(
            fe_field,
            includes_measure=bool(includes_measure.get(name, False)),
            w=ctx.w,
            detJ=ctx.detJ,
        )
        res_parts.append(fe)
    return jnp.concatenate(res_parts, axis=0)


def _mixed_surface_local_residual_numpy(
    *,
    u_vec: np.ndarray,
    slices: dict[str, slice],
    field_a: str,
    field_b: str,
    res_form: Callable[..., Any],
    ctx: SurfaceMixedFormContext,
    params: Any,
    includes_measure: dict[str, bool],
) -> np.ndarray:
    u_dict = _surface_local_u_dict(
        u_vec=u_vec,
        slices=slices,
        field_a=field_a,
        field_b=field_b,
        ctx=ctx,
    )
    fe_q = res_form(ctx, u_dict, params)
    res_parts = []
    for name in (field_a, field_b):
        fe_field = fe_q[name]
        fe = _reduce_surface_residual_numpy(
            fe_field,
            includes_measure=bool(includes_measure.get(name, False)),
            w=ctx.w,
            detJ=ctx.detJ,
        )
        res_parts.append(np.asarray(fe))
    return np.concatenate(res_parts, axis=0)


def _compute_mixed_surface_local_jacobian(
    *,
    u_local: np.ndarray,
    backend: str,
    fd_eps: float,
    fd_mode: str,
    fd_block_size: int,
    field_a: str,
    field_b: str,
    slices: dict[str, slice],
    res_form: Callable[..., Any],
    ctx: SurfaceMixedFormContext,
    params: Any,
    includes_measure: dict[str, bool],
) -> np.ndarray:
    if backend == "jax":
        def _res_local(u_vec):
            return _mixed_surface_local_residual_jax(
                u_vec=jnp.asarray(u_vec),
                slices=slices,
                field_a=field_a,
                field_b=field_b,
                res_form=res_form,
                ctx=ctx,
                params=params,
                includes_measure=includes_measure,
            )

        J_local = jax.jacrev(_res_local)(jnp.asarray(u_local))
        return np.asarray(J_local)
    if backend != "numpy":
        raise ValueError("backend must be 'jax' or 'numpy'")

    n_u = int(u_local.shape[0])
    J = np.zeros((n_u, n_u), dtype=float)
    for j in range(n_u):
        u_col = np.zeros((n_u,), dtype=float)
        u_col[j] = 1.0
        J[:, j] = _mixed_surface_local_residual_numpy(
            u_vec=u_col,
            slices=slices,
            field_a=field_a,
            field_b=field_b,
            res_form=res_form,
            ctx=ctx,
            params=params,
            includes_measure=includes_measure,
        )
    return J









def assemble_contact_coupling_matrices(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    *,
    tol: float = 1e-8,
    quad_order: int = 0,
) -> tuple[ContactCouplingMatrix, ContactCouplingMatrix]:
    """
    Assemble contact coupling matrices M_aa and M_ab on the supermesh.
    """
    if quad_order <= 0:
        quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        quad_w = np.array([0.5], dtype=float)
    else:
        quad_pts, quad_w = _tri_quadrature(int(quad_order))
    use_jax_geometry = _uses_jax_geometry(supermesh_coords, surface_a.coords, surface_b.coords)
    if use_jax_geometry:
        facets_a = np.asarray(surface_a.conn, dtype=int)
        facets_b = np.asarray(surface_b.conn, dtype=int)
        if facets_a.shape[1] != 3 or facets_b.shape[1] != 3:
            raise NotImplementedError(
                "JAX-traced contact coupling matrices are currently implemented only for tri3 facets."
            )
        coords_a_j = jnp.asarray(surface_a.coords)
        coords_b_j = jnp.asarray(surface_b.coords)
        supermesh_coords_j = jnp.asarray(supermesh_coords)

        rows_aa: list[int] = []
        cols_aa: list[int] = []
        data_aa: list[jnp.ndarray] = []
        rows_ab: list[int] = []
        cols_ab: list[int] = []
        data_ab: list[jnp.ndarray] = []

        for tri, fa, fb in zip(np.asarray(supermesh_conn, dtype=int), source_facets_a, source_facets_b):
            a = supermesh_coords_j[tri[0]]
            b = supermesh_coords_j[tri[1]]
            c = supermesh_coords_j[tri[2]]
            detJ = jnp.linalg.norm(jnp.cross(b - a, c - a))
            facet_a = facets_a[int(fa)]
            facet_b = facets_b[int(fb)]
            for (r, s), w_ref in zip(quad_pts, quad_w):
                x_q = a + float(r) * (b - a) + float(s) * (c - a)
                weight = detJ * float(w_ref)
                Na = _tri3_shape_values_jax(x_q, facet_a, coords_a_j)
                Nb = _tri3_shape_values_jax(x_q, facet_b, coords_b_j)

                for i, node_i in enumerate(facet_a):
                    for j, node_j in enumerate(facet_a):
                        rows_aa.append(int(node_i))
                        cols_aa.append(int(node_j))
                        data_aa.append(weight * Na[i] * Na[j])

                for i, node_i in enumerate(facet_a):
                    for j, node_j in enumerate(facet_b):
                        rows_ab.append(int(node_i))
                        cols_ab.append(int(node_j))
                        data_ab.append(weight * Na[i] * Nb[j])

        n_a = int(coords_a_j.shape[0])
        n_b = int(coords_b_j.shape[0])
        M_aa = ContactCouplingMatrix(
            rows=np.asarray(rows_aa, dtype=int),
            cols=np.asarray(cols_aa, dtype=int),
            data=jnp.stack(data_aa) if data_aa else jnp.zeros((0,), dtype=coords_a_j.dtype),
            shape=(n_a, n_a),
        )
        M_ab = ContactCouplingMatrix(
            rows=np.asarray(rows_ab, dtype=int),
            cols=np.asarray(cols_ab, dtype=int),
            data=jnp.stack(data_ab) if data_ab else jnp.zeros((0,), dtype=coords_b_j.dtype),
            shape=(n_a, n_b),
        )
        return M_aa, M_ab

    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)

    rows_aa: list[int] = []
    cols_aa: list[int] = []
    data_aa: list[float] = []

    rows_ab: list[int] = []
    cols_ab: list[int] = []
    data_ab: list[float] = []

    for (tri, a, b, c), fa, fb in zip(
        _iter_supermesh_tris(supermesh_coords, supermesh_conn),
        source_facets_a,
        source_facets_b,
    ):
        detJ = 2.0 * _tri_area(a, b, c)
        if detJ <= tol:
            continue

        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        for (r, s), w_ref in zip(quad_pts, quad_w):
            x_q = a + float(r) * (b - a) + float(s) * (c - a)
            weight = detJ * float(w_ref)
            Na = _facet_shape_values(x_q, facet_a, coords_a, tol=tol)
            Nb = _facet_shape_values(x_q, facet_b, coords_b, tol=tol)

            for i, node_i in enumerate(facet_a):
                for j, node_j in enumerate(facet_a):
                    rows_aa.append(int(node_i))
                    cols_aa.append(int(node_j))
                    data_aa.append(weight * float(Na[i]) * float(Na[j]))

            for i, node_i in enumerate(facet_a):
                for j, node_j in enumerate(facet_b):
                    rows_ab.append(int(node_i))
                    cols_ab.append(int(node_j))
                    data_ab.append(weight * float(Na[i]) * float(Nb[j]))

    n_a = int(np.asarray(surface_a.coords).shape[0])
    n_b = int(np.asarray(surface_b.coords).shape[0])
    M_aa = ContactCouplingMatrix(
        rows=np.asarray(rows_aa, dtype=int),
        cols=np.asarray(cols_aa, dtype=int),
        data=np.asarray(data_aa, dtype=float),
        shape=(n_a, n_a),
    )
    M_ab = ContactCouplingMatrix(
        rows=np.asarray(rows_ab, dtype=int),
        cols=np.asarray(cols_ab, dtype=int),
        data=np.asarray(data_ab, dtype=float),
        shape=(n_a, n_b),
    )
    return M_aa, M_ab


def assemble_contact_interface_residual(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    res_form,
    u_a: np.ndarray,
    u_b: np.ndarray,
    params,
    *,
    value_dim_a: int = 1,
    value_dim_b: int = 1,
    trial_value_dim_a: int | None = None,
    trial_value_dim_b: int | None = None,
    offset_a: int = 0,
    offset_b: int | None = None,
    field_a: str = "a",
    field_b: str = "b",
    elem_conn_a: np.ndarray | None = None,
    elem_conn_b: np.ndarray | None = None,
    facet_to_elem_a: np.ndarray | None = None,
    facet_to_elem_b: np.ndarray | None = None,
    normal_source: str = "master",
    normal_from: str | None = None,
    master_field: str | None = None,
    normal_sign: float = 1.0,
    grad_source: str = "volume",
    dof_source: str = "surface",
    space_mode_a: str = "nodal",
    space_mode_b: str = "nodal",
    trial_space_mode_a: str | None = None,
    trial_space_mode_b: str | None = None,
    facet_dofs_a: np.ndarray | None = None,
    facet_dofs_b: np.ndarray | None = None,
    trial_facet_dofs_a: np.ndarray | None = None,
    trial_facet_dofs_b: np.ndarray | None = None,
    quad_order: int = 0,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Assemble mixed surface residual over a supermesh (centroid quadrature).

    normal_source can be "master", "slave", "a", "b", or "avg"; use master_field
    to pick which field acts as the master when normal_source is "master"/"slave".
    dof_source="volume" assembles nodal fields into element nodes (requires elem_conn_* mappings).
    space_mode_* supports:
    - "nodal": existing nodal FE space behavior.
    - "p0": facet-wise constant space (one block of value_dim DOFs per facet, or
      custom mapping via facet_dofs_*).
    """
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    n_a = _field_n_dofs(
        n_nodes=int(coords_a.shape[0]),
        n_facets=int(facets_a.shape[0]),
        value_dim=int(value_dim_a),
        space_mode=space_mode_a,
        facet_dofs=facet_dofs_a,
    )
    n_b = _field_n_dofs(
        n_nodes=int(coords_b.shape[0]),
        n_facets=int(facets_b.shape[0]),
        value_dim=int(value_dim_b),
        space_mode=space_mode_b,
        facet_dofs=facet_dofs_b,
    )
    if offset_b is None:
        offset_b = offset_a + n_a
    if trial_value_dim_a is None:
        trial_value_dim_a = value_dim_a
    if trial_value_dim_b is None:
        trial_value_dim_b = value_dim_b
    if trial_space_mode_a is None:
        trial_space_mode_a = space_mode_a
    if trial_space_mode_b is None:
        trial_space_mode_b = space_mode_b
    if trial_facet_dofs_a is None:
        trial_facet_dofs_a = facet_dofs_a
    if trial_facet_dofs_b is None:
        trial_facet_dofs_b = facet_dofs_b
    distinct_trial_layout = (
        int(trial_value_dim_a) != int(value_dim_a)
        or int(trial_value_dim_b) != int(value_dim_b)
        or str(trial_space_mode_a) != str(space_mode_a)
        or str(trial_space_mode_b) != str(space_mode_b)
        or not _same_optional_int_array(trial_facet_dofs_a, facet_dofs_a)
        or not _same_optional_int_array(trial_facet_dofs_b, facet_dofs_b)
    )
    n_total = int(offset_b + n_b)
    use_jax = _has_jax_leaves((u_a, u_b, params))
    if use_jax:
        R = jnp.zeros((n_total,), dtype=jnp.float64)
        u_a_np = jnp.asarray(u_a, dtype=jnp.float64)
        u_b_np = jnp.asarray(u_b, dtype=jnp.float64)
    else:
        R = np.zeros((n_total,), dtype=float)
        u_a_np = np.asarray(u_a, dtype=float)
        u_b_np = np.asarray(u_b, dtype=float)

    trace = os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE", "0") not in ("0", "", "false", "False")

    def _trace_time(msg: str, t0: float) -> None:
        if trace:
            print(f"{msg} dt={time.perf_counter() - t0:.3e}s", flush=True)

    t_norm = time.perf_counter()
    normals_a = None
    normals_b = None
    if hasattr(surface_a, "facet_normals"):
        normals_a = surface_a.facet_normals()
    if hasattr(surface_b, "facet_normals"):
        normals_b = surface_b.facet_normals()
    if trace:
        _trace_time("[CONTACT] normals_done", t_norm)

    area_scale = float(os.getenv("FLUXFEM_SMALL_TRI_EPS_SCALE", "0.0"))
    skip_small_tri = os.getenv("FLUXFEM_SKIP_SMALL_TRI", "0") == "1" and area_scale > 0.0
    facet_area_a = None
    facet_area_b = None
    if area_scale > 0.0:
        t_area = time.perf_counter()
        if hasattr(surface_a, "facet_areas"):
            facet_area_a = np.asarray(surface_a.facet_areas(), dtype=float)
        else:
            facet_area_a = np.array([_facet_area_estimate(fa, coords_a) for fa in facets_a], dtype=float)
        if hasattr(surface_b, "facet_areas"):
            facet_area_b = np.asarray(surface_b.facet_areas(), dtype=float)
        else:
            facet_area_b = np.array([_facet_area_estimate(fb, coords_b) for fb in facets_b], dtype=float)
        if trace:
            _trace_time("[CONTACT] facet_area_done", t_area)

    includes_measure = getattr(res_form, "_includes_measure", {})

    use_elem_a = elem_conn_a is not None and facet_to_elem_a is not None
    use_elem_b = elem_conn_b is not None and facet_to_elem_b is not None
    if use_elem_a:
        assert elem_conn_a is not None
        assert facet_to_elem_a is not None
    if use_elem_b:
        assert elem_conn_b is not None
        assert facet_to_elem_b is not None

    use_p0_a, use_p0_b = _validate_contact_interface_space_setup(
        grad_source=grad_source,
        dof_source=dof_source,
        space_mode_a=space_mode_a,
        space_mode_b=space_mode_b,
    )
    global _DEBUG_SURFACE_SOURCE_ONCE
    if grad_source == "surface" and not _DEBUG_SURFACE_SOURCE_ONCE:
        print("[fluxfem] using surface gradN in contact interface")
        _DEBUG_SURFACE_SOURCE_ONCE = True
    proj_diag = _proj_diag_enabled()
    if proj_diag:
        _proj_diag_reset()
    diag_force = os.getenv("FLUXFEM_PROJ_DIAG_FORCE", "0") == "1"
    diag_qp_mode = os.getenv("FLUXFEM_PROJ_DIAG_QP_MODE", "").strip().lower()
    diag_qp_path = os.getenv("FLUXFEM_PROJ_DIAG_QP_PATH", "").strip()
    diag_normal = os.getenv("FLUXFEM_PROJ_DIAG_NORMAL", "").strip().lower()
    diag_facet = int(os.getenv("FLUXFEM_PROJ_DIAG_FACET", "-1"))
    diag_max_q = int(os.getenv("FLUXFEM_PROJ_DIAG_MAX_Q", "3"))
    diag_abs_detj = os.getenv("FLUXFEM_PROJ_DIAG_ABS_DETJ", "1") == "1"

    normal_source = _resolve_normal_source_option(
        normal_source=normal_source,
        normal_from=normal_from,
        master_field=master_field,
        field_a=field_a,
        field_b=field_b,
        diag_force=diag_force,
        diag_normal=diag_normal,
    )

    contact_interface_mode = os.getenv("FLUXFEM_CONTACT_INTERFACE_MODE", "supermesh").lower()
    callbacks = _contact_assembly_callbacks()
    if contact_interface_mode == "projection" and not (use_p0_a or use_p0_b) and not distinct_trial_layout:
        batches, fallback = _projection_surface_batches(
            source_facets_a,
            source_facets_b,
            surface_a,
            surface_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            quad_order=quad_order,
            grad_source=grad_source,
            dof_source=dof_source,
            normal_source=normal_source,
            normal_sign=normal_sign,
            tol=tol,
        )
        if batches is not None and not fallback:
            test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = (
                _mixed_surface_space_aliases(
                    res_form,
                    field_a=field_a,
                    field_b=field_b,
                )
            )
            for batch in batches:
                Na = batch["Na"]
                Nb = batch["Nb"]
                gradNa = batch["gradNa"]
                gradNb = batch["gradNb"]
                nodes_a = batch["nodes_a"]
                nodes_b = batch["nodes_b"]
                normal_q = batch["normal"]

                ctx = _build_mixed_surface_context(
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
                u_elem = _surface_u_elem_with_space_aliases(
                    field_a=field_a,
                    field_b=field_b,
                    unknown_space_key_a=unknown_space_key_a,
                    unknown_space_key_b=unknown_space_key_b,
                    u_local_a=_gather_u_local(u_a_np, nodes_a, value_dim_a),
                    u_local_b=_gather_u_local(u_b_np, nodes_b, value_dim_b),
                )
                fe_q = res_form(ctx, u_elem, params)
                for name, facet, value_dim, offset in (
                    (field_a, nodes_a, value_dim_a, offset_a),
                    (field_b, nodes_b, value_dim_b, offset_b),
                ):
                    fe_field = fe_q[name]
                    if fe_field.ndim != 2 or fe_field.shape[0] != ctx.x_q.shape[0]:
                        raise ValueError("mixed surface residual must return (n_q, n_ldofs)")
                    if use_jax:
                        fe = _reduce_surface_residual_jax(
                            fe_field,
                            includes_measure=bool(includes_measure.get(name, False)),
                            w=ctx.w,
                            detJ=ctx.detJ,
                        )
                        dofs = _global_dof_indices(facet, value_dim, int(offset))
                        R = R.at[jnp.asarray(dofs, dtype=jnp.int32)].add(jnp.asarray(fe))
                    else:
                        fe = _reduce_surface_residual_jax(
                            fe_field,
                            includes_measure=bool(includes_measure.get(name, False)),
                            w=ctx.w,
                            detJ=ctx.detJ,
                        )
                        dofs = _global_dof_indices(facet, value_dim, int(offset))
                        R[dofs] += np.asarray(fe)
            return R

    pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=use_p0_a,
        use_p0_b=use_p0_b,
    )
    trial_use_p0_a = str(trial_space_mode_a) == "p0"
    trial_use_p0_b = str(trial_space_mode_b) == "p0"
    trial_pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=trial_use_p0_a,
        use_p0_b=trial_use_p0_b,
    )
    for (tri, a, b, c), fa, fb in zip(
        _iter_supermesh_tris(supermesh_coords, supermesh_conn),
        source_facets_a,
        source_facets_b,
    ):
        R = _accumulate_supermesh_residual_triangle(
            R=R,
            a=a,
            b=b,
            c=c,
            fa=int(fa),
            fb=int(fb),
            tol=tol,
            skip_small_tri=skip_small_tri,
            area_scale=area_scale,
            facet_area_a=facet_area_a,
            facet_area_b=facet_area_b,
            diag_force=diag_force,
            diag_abs_detj=diag_abs_detj,
            quad_order=quad_order,
            diag_qp_mode=diag_qp_mode,
            diag_qp_path=diag_qp_path,
            coords_a=coords_a,
            coords_b=coords_b,
            facets_a=facets_a,
            facets_b=facets_b,
            pair_basis_builder=pair_basis_builder,
            trial_pair_basis_builder=trial_pair_basis_builder,
            value_dim_a=value_dim_a,
            value_dim_b=value_dim_b,
            trial_value_dim_a=int(trial_value_dim_a),
            trial_value_dim_b=int(trial_value_dim_b),
            dof_source=dof_source,
            grad_source=grad_source,
            space_mode_a=space_mode_a,
            space_mode_b=space_mode_b,
            trial_space_mode_a=str(trial_space_mode_a),
            trial_space_mode_b=str(trial_space_mode_b),
            use_elem_a=use_elem_a,
            use_elem_b=use_elem_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            facet_dofs_a=facet_dofs_a,
            facet_dofs_b=facet_dofs_b,
            trial_facet_dofs_a=trial_facet_dofs_a,
            trial_facet_dofs_b=trial_facet_dofs_b,
            proj_diag=proj_diag,
            normal_source=normal_source,
            normal_sign=normal_sign,
            normals_a=normals_a,
            normals_b=normals_b,
            field_a=field_a,
            field_b=field_b,
            u_a=u_a_np,
            u_b=u_b_np,
            res_form=res_form,
            params=params,
            includes_measure=includes_measure,
            offset_a=int(offset_a),
            offset_b=int(offset_b),
            diag_facet=diag_facet,
            diag_max_q=diag_max_q,
            callbacks=callbacks,
        )
    if proj_diag:
        _proj_diag_report()
    return R


def assemble_contact_interface_jacobian(
    supermesh_coords: np.ndarray,
    supermesh_conn: np.ndarray,
    source_facets_a: Iterable[int],
    source_facets_b: Iterable[int],
    surface_a: SurfaceMesh,
    surface_b: SurfaceMesh,
    res_form,
    u_a: np.ndarray,
    u_b: np.ndarray,
    params,
    *,
    value_dim_a: int = 1,
    value_dim_b: int = 1,
    trial_value_dim_a: int | None = None,
    trial_value_dim_b: int | None = None,
    offset_a: int = 0,
    offset_b: int | None = None,
    field_a: str = "a",
    field_b: str = "b",
    elem_conn_a: np.ndarray | None = None,
    elem_conn_b: np.ndarray | None = None,
    facet_to_elem_a: np.ndarray | None = None,
    facet_to_elem_b: np.ndarray | None = None,
    normal_source: str = "master",
    normal_from: str | None = None,
    master_field: str | None = None,
    normal_sign: float = 1.0,
    grad_source: str = "volume",
    dof_source: str = "surface",
    space_mode_a: str = "nodal",
    space_mode_b: str = "nodal",
    trial_space_mode_a: str | None = None,
    trial_space_mode_b: str | None = None,
    facet_dofs_a: np.ndarray | None = None,
    facet_dofs_b: np.ndarray | None = None,
    trial_facet_dofs_a: np.ndarray | None = None,
    trial_facet_dofs_b: np.ndarray | None = None,
    quad_order: int = 0,
    tol: float = 1e-8,
    sparse: bool = False,
    backend: str | None = None,
    batch_jac: bool | None = None,
    fd_eps: float = 1e-6,
    fd_mode: str = "central",
    fd_block_size: int = 1,
    supermesh_quad_cache: _SupermeshTriangleQuadratureCache | None = None,
):
    """
    Assemble mixed surface Jacobian over a supermesh (centroid quadrature).

    normal_source can be "master", "slave", "a", "b", or "avg"; use master_field
    to pick which field acts as the master when normal_source is "master"/"slave".
    dof_source="volume" assembles nodal fields into element nodes (requires elem_conn_* mappings).
    space_mode_* supports:
    - "nodal": existing nodal FE space behavior.
    - "p0": facet-wise constant space (one block of value_dim DOFs per facet, or
      custom mapping via facet_dofs_*).
    """
    backend = "jax" if backend is None else str(backend).lower()
    source_facets_a = list(source_facets_a)
    source_facets_b = list(source_facets_b)
    from ..core.forms import FieldPair
    _contact_interface_dbg(
        f"[contact-interface] enter assemble_contact_interface_jacobian quad_order={quad_order} backend={backend}"
    )
    trace = os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE", "0") not in ("0", "", "false", "False")
    trace_max = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE_MAX", "5"))
    trace_every = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE_EVERY", "50"))
    trace_fd_max = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_TRACE_FD_MAX", "5"))
    def _trace(msg: str) -> None:
        if trace:
            print(msg, flush=True)
    def _trace_time(msg: str, t0: float) -> None:
        if trace:
            print(f"{msg}: {time.perf_counter() - t0:.6f}s", flush=True)
    t_prep = time.perf_counter()
    coords_a = np.asarray(surface_a.coords, dtype=float)
    coords_b = np.asarray(surface_b.coords, dtype=float)
    facets_a = np.asarray(surface_a.conn, dtype=int)
    facets_b = np.asarray(surface_b.conn, dtype=int)
    n_a = _field_n_dofs(
        n_nodes=int(coords_a.shape[0]),
        n_facets=int(facets_a.shape[0]),
        value_dim=int(value_dim_a),
        space_mode=space_mode_a,
        facet_dofs=facet_dofs_a,
    )
    n_b = _field_n_dofs(
        n_nodes=int(coords_b.shape[0]),
        n_facets=int(facets_b.shape[0]),
        value_dim=int(value_dim_b),
        space_mode=space_mode_b,
        facet_dofs=facet_dofs_b,
    )
    if offset_b is None:
        offset_b = offset_a + n_a
    if trial_value_dim_a is None:
        trial_value_dim_a = value_dim_a
    if trial_value_dim_b is None:
        trial_value_dim_b = value_dim_b
    if trial_space_mode_a is None:
        trial_space_mode_a = space_mode_a
    if trial_space_mode_b is None:
        trial_space_mode_b = space_mode_b
    if trial_facet_dofs_a is None:
        trial_facet_dofs_a = facet_dofs_a
    if trial_facet_dofs_b is None:
        trial_facet_dofs_b = facet_dofs_b
    distinct_trial_layout = (
        int(trial_value_dim_a) != int(value_dim_a)
        or int(trial_value_dim_b) != int(value_dim_b)
        or str(trial_space_mode_a) != str(space_mode_a)
        or str(trial_space_mode_b) != str(space_mode_b)
        or not _same_optional_int_array(trial_facet_dofs_a, facet_dofs_a)
        or not _same_optional_int_array(trial_facet_dofs_b, facet_dofs_b)
    )
    n_total = int(offset_b + n_b)
    u_a_np = np.asarray(u_a, dtype=float)
    u_b_np = np.asarray(u_b, dtype=float)
    if trace:
        _trace("[CONTACT] assemble_contact_interface_jacobian ENTER")
        _trace(f"[CONTACT] shapes: coords_a={coords_a.shape} coords_b={coords_b.shape} supermesh={supermesh_conn.shape}")
        _trace(f"[CONTACT] dtypes: coords_a={coords_a.dtype} coords_b={coords_b.dtype} supermesh={supermesh_conn.dtype}")
        _trace(f"[CONTACT] finite: coords_a={np.isfinite(coords_a).all()} coords_b={np.isfinite(coords_b).all()}")
        _trace_time("[CONTACT] prep_done", t_prep)

    guard = os.getenv("FLUXFEM_CONTACT_GUARD", "0") == "1"
    detj_eps = float(os.getenv("FLUXFEM_CONTACT_DETJ_EPS", "0.0"))
    tri_timeout = float(os.getenv("FLUXFEM_CONTACT_TRI_TIMEOUT_S", "0.0"))
    skip_nonfinite = os.getenv("FLUXFEM_CONTACT_SKIP_NONFINITE", "1") == "1"
    if guard:
        if not (np.isfinite(coords_a).all() and np.isfinite(coords_b).all()):
            raise RuntimeError("[CONTACT] non-finite coords in contact surfaces")
        if not np.isfinite(supermesh_coords).all():
            raise RuntimeError("[CONTACT] non-finite supermesh coords")
        if supermesh_conn.size:
            min_idx = int(supermesh_conn.min())
            max_idx = int(supermesh_conn.max())
            if min_idx < 0 or max_idx >= supermesh_coords.shape[0]:
                raise RuntimeError(
                    f"[CONTACT] supermesh_conn index out of range: min={min_idx} max={max_idx} n={supermesh_coords.shape[0]}"
                )
        if len(supermesh_conn) != len(source_facets_a) or len(supermesh_conn) != len(source_facets_b):
            raise RuntimeError(
                "[CONTACT] supermesh_conn and source_facets lengths mismatch "
                f"conn={len(supermesh_conn)} fa={len(source_facets_a)} fb={len(source_facets_b)}"
            )

    normals_a = None
    normals_b = None
    if hasattr(surface_a, "facet_normals"):
        normals_a = surface_a.facet_normals()
    if hasattr(surface_b, "facet_normals"):
        normals_b = surface_b.facet_normals()

    area_scale = float(os.getenv("FLUXFEM_SMALL_TRI_EPS_SCALE", "0.0"))
    skip_small_tri = os.getenv("FLUXFEM_SKIP_SMALL_TRI", "0") == "1" and area_scale > 0.0
    facet_area_a = None
    facet_area_b = None
    if area_scale > 0.0:
        if hasattr(surface_a, "facet_areas"):
            facet_area_a = np.asarray(surface_a.facet_areas(), dtype=float)
        else:
            facet_area_a = np.array([_facet_area_estimate(fa, coords_a) for fa in facets_a], dtype=float)
        if hasattr(surface_b, "facet_areas"):
            facet_area_b = np.asarray(surface_b.facet_areas(), dtype=float)
        else:
            facet_area_b = np.array([_facet_area_estimate(fb, coords_b) for fb in facets_b], dtype=float)

    includes_measure = getattr(res_form, "_includes_measure", {})

    rows: list[int] = []
    cols: list[int] = []
    data: list[Any] = []
    if sparse:
        K_dense = None
    elif backend == "jax":
        K_dense = jnp.zeros((n_total, n_total), dtype=jnp.float64)
    else:
        K_dense = np.zeros((n_total, n_total), dtype=float)
    callbacks = _contact_assembly_callbacks()

    use_elem_a = elem_conn_a is not None and facet_to_elem_a is not None
    use_elem_b = elem_conn_b is not None and facet_to_elem_b is not None

    use_p0_a, use_p0_b = _validate_contact_interface_space_setup(
        grad_source=grad_source,
        dof_source=dof_source,
        space_mode_a=space_mode_a,
        space_mode_b=space_mode_b,
    )
    global _DEBUG_SURFACE_SOURCE_ONCE
    if grad_source == "surface" and not _DEBUG_SURFACE_SOURCE_ONCE:
        print("[fluxfem] using surface gradN in contact interface")
        _DEBUG_SURFACE_SOURCE_ONCE = True
    diag_map = os.getenv("FLUXFEM_DIAG_CONTACT_MAP", "0") == "1"
    diag_n = os.getenv("FLUXFEM_DIAG_CONTACT_N", "0") == "1"
    proj_diag = _proj_diag_enabled()
    if proj_diag:
        _proj_diag_reset()
    diag_force = os.getenv("FLUXFEM_PROJ_DIAG_FORCE", "0") == "1"
    diag_qp_mode = os.getenv("FLUXFEM_PROJ_DIAG_QP_MODE", "").strip().lower()
    diag_qp_path = os.getenv("FLUXFEM_PROJ_DIAG_QP_PATH", "").strip()
    diag_normal = os.getenv("FLUXFEM_PROJ_DIAG_NORMAL", "").strip().lower()
    diag_facet = int(os.getenv("FLUXFEM_PROJ_DIAG_FACET", "-1"))
    diag_max_q = int(os.getenv("FLUXFEM_PROJ_DIAG_MAX_Q", "3"))
    diag_abs_detj = os.getenv("FLUXFEM_PROJ_DIAG_ABS_DETJ", "1") == "1"
    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")
    if batch_jac is None:
        batch_jac = _env_flag("FLUXFEM_CONTACT_INTERFACE_BATCH_JAC", True)

    normal_source = _resolve_normal_source_option(
        normal_source=normal_source,
        normal_from=normal_from,
        master_field=master_field,
        field_a=field_a,
        field_b=field_b,
        diag_force=diag_force,
        diag_normal=diag_normal,
    )

    contact_interface_mode = os.getenv("FLUXFEM_CONTACT_INTERFACE_MODE", "supermesh").lower()
    _contact_interface_dbg(f"[contact-interface] mode={contact_interface_mode}")
    if contact_interface_mode == "projection" and not (use_p0_a or use_p0_b) and not distinct_trial_layout:
        batches, fallback = _projection_surface_batches(
            source_facets_a,
            source_facets_b,
            surface_a,
            surface_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            quad_order=quad_order,
            grad_source=grad_source,
            dof_source=dof_source,
            normal_source=normal_source,
            normal_sign=normal_sign,
            tol=tol,
        )
        if batches is not None and not fallback:
            _apply_projection_jacobian_batches(
                batches=batches,
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
            if sparse:
                from ..solver import FluxSparseMatrix

                return FluxSparseMatrix(
                    np.asarray(rows, dtype=int),
                    np.asarray(cols, dtype=int),
                    jnp.asarray(data, dtype=jnp.float64) if backend == "jax" else np.asarray(data, dtype=float),
                    n_dofs=n_total,
                )
            assert K_dense is not None
            return K_dense

    batch_enabled = (
        batch_jac
        and backend == "jax"
        and dof_source == "volume"
        and grad_source == "volume"
        and use_elem_a
        and use_elem_b
        and not use_p0_a
        and not use_p0_b
        and not distinct_trial_layout
        and not proj_diag
        and not diag_force
    )
    if trace and batch_jac and not batch_enabled:
        reasons = []
        if backend != "jax":
            reasons.append(f"backend={backend}")
        if dof_source != "volume":
            reasons.append(f"dof_source={dof_source}")
        if grad_source != "volume":
            reasons.append(f"grad_source={grad_source}")
        if not use_elem_a:
            reasons.append("missing_elem_a")
        if not use_elem_b:
            reasons.append("missing_elem_b")
        if use_p0_a:
            reasons.append("space_mode_a=p0")
        if use_p0_b:
            reasons.append("space_mode_b=p0")
        if proj_diag:
            reasons.append("projection_diag")
        if diag_force:
            reasons.append("diag_force")
        _trace(f"[CONTACT] batch_jac_disabled {' '.join(reasons)}")

    if batch_enabled:
        if trace:
            _trace("[CONTACT] batch_jac_enter")
        batch_items = []
        dofs_batch = []
        u_local_batch = []
        batch_rows: list[np.ndarray] = []
        batch_cols: list[np.ndarray] = []
        batch_data: list[Any] = []
        batch_size = int(os.getenv("FLUXFEM_CONTACT_INTERFACE_BATCH_SIZE", "128"))
        if batch_size <= 0:
            batch_size = 0
        n_q = None
        n_nodes_a = None
        n_nodes_b = None
        n_a_local_const = None
        n_b_local_const = None
        batch_failed = False
        jit_batch = _env_flag("FLUXFEM_CONTACT_INTERFACE_BATCH_JIT", False)
        direct_jit_batch = _env_flag("FLUXFEM_CONTACT_INTERFACE_DIRECT_BATCH_JIT", True)
        formulation_tag = getattr(res_form, "_ff_contact_formulation", None)
        fastpath_tag = getattr(res_form, "_ff_contact_backend_fastpath", None)
        has_pair_nitsche_params = all(hasattr(params, name) for name in ("alpha", "inv_h", "lam", "mu", "use_penalty", "use_traction"))
        use_direct_pair_nitsche_batch = (
            formulation_tag == "pair_nitsche_penalty"
            and fastpath_tag == "numpy_local_kernel"
            and has_pair_nitsche_params
            and int(value_dim_a) == 3
            and int(value_dim_b) == 3
        )

        def _make_jac_fun(n_a_local: int, n_b_local: int):
            test_space_key_a, test_space_key_b, unknown_space_key_a, unknown_space_key_b = (
                _mixed_surface_space_aliases(
                    res_form,
                    field_a=field_a,
                    field_b=field_b,
                )
            )

            def _res_local_batch(u_vec, Na, Nb, gradNa, gradNb, x_q, w, detJ, normal):
                fields = {
                    field_a: _make_surface_field_pair(
                        test_N=Na,
                        test_gradN=gradNa,
                        trial_N=Na,
                        trial_gradN=gradNa,
                        test_value_dim=value_dim_a,
                        trial_value_dim=value_dim_a,
                    ),
                    field_b: _make_surface_field_pair(
                        test_N=Nb,
                        test_gradN=gradNb,
                        trial_N=Nb,
                        trial_gradN=gradNb,
                        test_value_dim=value_dim_b,
                        trial_value_dim=value_dim_b,
                    ),
                }
                spaces = dict(fields)
                for key in (test_space_key_a, unknown_space_key_a):
                    if key is not None:
                        spaces[key] = fields[field_a]
                for key in (test_space_key_b, unknown_space_key_b):
                    if key is not None:
                        spaces[key] = fields[field_b]
                normal_q = jnp.repeat(normal[None, :], x_q.shape[0], axis=0)
                ctx = SurfaceMixedFormContext(
                    bindings=fields,
                    x_q=x_q,
                    w=w,
                    detJ=detJ,
                    normal=normal_q,
                    spaces=spaces,
                )
                u_dict = {
                    field_a: u_vec[:n_a_local],
                    field_b: u_vec[n_a_local:],
                }
                if unknown_space_key_a is not None:
                    u_dict[unknown_space_key_a] = u_dict[field_a]
                if unknown_space_key_b is not None:
                    u_dict[unknown_space_key_b] = u_dict[field_b]
                fe_q = res_form(ctx, u_dict, params)
                res_parts = []
                for name in (field_a, field_b):
                    fe_field = fe_q[name]
                    if includes_measure.get(name, False):
                        fe = jnp.sum(jnp.asarray(fe_field), axis=0)
                    else:
                        wJ = jnp.asarray(ctx.w) * jnp.asarray(ctx.detJ)
                        fe = jnp.einsum("qi,q->i", jnp.asarray(fe_field), wJ)
                    res_parts.append(fe)
                return jnp.concatenate(res_parts, axis=0)

            if trace:
                _trace(f"[CONTACT] batch_jac_build n_a={n_a_local} n_b={n_b_local} jit={jit_batch}")
            jac_fun = jax.vmap(jax.jacrev(_res_local_batch))
            return jax.jit(jac_fun) if jit_batch else jac_fun

        jac_fun_cache: dict[tuple[int, int], Callable[..., jnp.ndarray]] = {}
        direct_batch_fun = None

        def _emit_batch(
            Na_b,
            Nb_b,
            gradNa_b,
            gradNb_b,
            x_q_b,
            w_b,
            detJ_b,
            normal_b,
            u_local_b,
            dofs_batch_np,
            n_a_local,
            n_b_local,
            batch_n,
        ) -> None:
            nonlocal K_dense
            if trace:
                _trace(f"[CONTACT] batch_emit start n={int(Na_b.shape[0])}")
            if jit_batch and batch_size and batch_n < batch_size:
                pad = int(batch_size - batch_n)
                if trace:
                    _trace(f"[CONTACT] batch_pad n={batch_n} target={batch_size}")

                def _pad_batch(x, pad_value: float = 0.0):
                    pad_width = [(0, pad)] + [(0, 0)] * (x.ndim - 1)
                    return jnp.pad(jnp.asarray(x), pad_width, mode="constant", constant_values=pad_value)

                Na_b = _pad_batch(Na_b)
                Nb_b = _pad_batch(Nb_b)
                gradNa_b = _pad_batch(gradNa_b)
                gradNb_b = _pad_batch(gradNb_b)
                x_q_b = _pad_batch(x_q_b)
                w_b = _pad_batch(w_b)
                detJ_b = _pad_batch(detJ_b)
                normal_b = _pad_batch(normal_b)
                u_local_b = _pad_batch(u_local_b)
            t_batch = time.perf_counter()
            if use_direct_pair_nitsche_batch:
                nonlocal direct_batch_fun
                if direct_batch_fun is None:
                    if trace:
                        _trace(f"[CONTACT] batch_direct_build jit={direct_jit_batch}")
                    direct_batch_fun = _get_direct_pair_nitsche_batch_fun(jit=direct_jit_batch)
                J_b = direct_batch_fun(
                    Na_b,
                    Nb_b,
                    gradNa_b,
                    gradNb_b,
                    w_b,
                    detJ_b,
                    normal_b,
                    float(getattr(params, "alpha")),
                    float(getattr(params, "inv_h")),
                    float(getattr(params, "lam")),
                    float(getattr(params, "mu")),
                    float(getattr(params, "use_penalty", 1.0)),
                    float(getattr(params, "use_traction", 1.0)),
                )
            else:
                key = (n_a_local, n_b_local)
                jac_fun = jac_fun_cache.get(key)
                if jac_fun is None:
                    jac_fun = _make_jac_fun(n_a_local, n_b_local)
                    jac_fun_cache[key] = jac_fun
                J_b = jac_fun(u_local_b, Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b)
            if trace:
                _trace_time("[CONTACT] batch_emit jac_done", t_batch)
            n_ldofs = dofs_batch_np.shape[1]
            rows = np.repeat(dofs_batch_np, n_ldofs, axis=1).reshape(-1)
            cols = np.tile(dofs_batch_np, (1, n_ldofs)).reshape(-1)
            if sparse:
                data = jnp.asarray(J_b)[:batch_n].reshape(-1)
                batch_rows.append(rows)
                batch_cols.append(cols)
                batch_data.append(data)
            else:
                assert K_dense is not None
                # rows/cols contain repeated global DOF pairs across triangles in the batch.
                # Advanced indexing with += does not accumulate repeated indices reliably.
                data = jnp.asarray(J_b)[:batch_n].reshape(-1)
                K_dense = K_dense.at[(jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(cols, dtype=jnp.int32))].add(data)
        for (tri, a, b, c), fa, fb in zip(
            _iter_supermesh_tris(supermesh_coords, supermesh_conn),
            source_facets_a,
            source_facets_b,
        ):
            area = _tri_area(a, b, c)
            if area <= tol:
                continue
            if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
                area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
                if area_ref > 0.0 and area < area_scale * area_ref:
                    continue
            detJ = 2.0 * area
            if diag_force and diag_abs_detj:
                detJ = abs(detJ)
            if quad_order <= 0:
                quad_pts = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
                quad_w = np.array([0.5], dtype=float)
            else:
                quad_pts, quad_w = _tri_quadrature(quad_order)

            facet_a = facets_a[int(fa)]
            facet_b = facets_b[int(fb)]
            x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)

            assert facet_to_elem_a is not None
            assert elem_conn_a is not None
            elem_id_a = int(facet_to_elem_a[int(fa)])
            elem_nodes_a = np.asarray(elem_conn_a[elem_id_a], dtype=int)
            elem_coords_a = coords_a[elem_nodes_a]
            assert facet_to_elem_b is not None
            assert elem_conn_b is not None
            elem_id_b = int(facet_to_elem_b[int(fb)])
            elem_nodes_b = np.asarray(elem_conn_b[elem_id_b], dtype=int)
            elem_coords_b = coords_b[elem_nodes_b]

            Na = _volume_shape_values_at_points(x_q, elem_coords_a, tol=tol)
            Nb = _volume_shape_values_at_points(x_q, elem_coords_b, tol=tol)
            gradNa = _tet_gradN_at_points(x_q, elem_coords_a, tol=tol)
            gradNb = _tet_gradN_at_points(x_q, elem_coords_b, tol=tol)

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
            if normal is None:
                if trace:
                    _trace(f"[CONTACT] batch_jac_abort normal_none tri={int(len(dofs_batch))}")
                batch_failed = True
                break

            u_elem = {
                field_a: _gather_u_local(u_a, elem_nodes_a, value_dim_a),
                field_b: _gather_u_local(u_b, elem_nodes_b, value_dim_b),
            }
            u_local = np.concatenate([u_elem[field_a], u_elem[field_b]], axis=0)

            dofs_a = _global_dof_indices(elem_nodes_a, value_dim_a, int(offset_a))
            dofs_b = _global_dof_indices(elem_nodes_b, value_dim_b, int(offset_b))
            dofs = np.concatenate([dofs_a, dofs_b], axis=0)

            batch_items.append((Na, Nb, gradNa, gradNb, x_q, quad_w, detJ, normal))
            dofs_batch.append(dofs)
            u_local_batch.append(u_local)

            if n_q is None:
                n_q = Na.shape[0]
                n_nodes_a = Na.shape[1]
                n_nodes_b = Nb.shape[1]
                n_a_local_const = dofs_a.shape[0]
                n_b_local_const = dofs_b.shape[0]
            else:
                shape_mismatch = (
                    Na.shape[0] != n_q
                    or Nb.shape[0] != n_q
                    or Na.shape[1] != n_nodes_a
                    or Nb.shape[1] != n_nodes_b
                    or dofs_a.shape[0] != n_a_local_const
                    or dofs_b.shape[0] != n_b_local_const
                )
                if shape_mismatch:
                    if trace:
                        _trace(
                            "[CONTACT] batch_jac_shape_mismatch "
                            f"nq={Na.shape[0]}/{n_q} "
                            f"na={Na.shape[1]}/{n_nodes_a} "
                            f"nb={Nb.shape[1]}/{n_nodes_b} "
                            f"da={dofs_a.shape[0]}/{n_a_local_const} "
                            f"db={dofs_b.shape[0]}/{n_b_local_const}"
                        )
                    if batch_items:
                        Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
                        Na_b = jnp.asarray(np.stack(Na_b, axis=0))
                        Nb_b = jnp.asarray(np.stack(Nb_b, axis=0))
                        gradNa_b = jnp.asarray(np.stack(gradNa_b, axis=0))
                        gradNb_b = jnp.asarray(np.stack(gradNb_b, axis=0))
                        x_q_b = jnp.asarray(np.stack(x_q_b, axis=0))
                        w_b = jnp.asarray(np.stack(w_b, axis=0))
                        detJ_b = jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1)
                        normal_b = jnp.asarray(np.stack(normal_b, axis=0))
                        u_local_b = jnp.asarray(np.stack(u_local_batch, axis=0))
                        dofs_batch_np = np.asarray(dofs_batch, dtype=int)
                        assert n_a_local_const is not None
                        assert n_b_local_const is not None
                        _emit_batch(
                            Na_b,
                            Nb_b,
                            gradNa_b,
                            gradNb_b,
                            x_q_b,
                            w_b,
                            detJ_b,
                            normal_b,
                            u_local_b,
                            dofs_batch_np,
                            int(n_a_local_const),
                            int(n_b_local_const),
                            int(Na_b.shape[0]),
                        )
                    batch_items = [(Na, Nb, gradNa, gradNb, x_q, quad_w, detJ, normal)]
                    dofs_batch = [dofs]
                    u_local_batch = [u_local]
                    n_q = Na.shape[0]
                    n_nodes_a = Na.shape[1]
                    n_nodes_b = Nb.shape[1]
                    n_a_local_const = dofs_a.shape[0]
                    n_b_local_const = dofs_b.shape[0]

            if batch_size and len(batch_items) >= batch_size:
                Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
                Na_b = jnp.asarray(np.stack(Na_b, axis=0))
                Nb_b = jnp.asarray(np.stack(Nb_b, axis=0))
                gradNa_b = jnp.asarray(np.stack(gradNa_b, axis=0))
                gradNb_b = jnp.asarray(np.stack(gradNb_b, axis=0))
                x_q_b = jnp.asarray(np.stack(x_q_b, axis=0))
                w_b = jnp.asarray(np.stack(w_b, axis=0))
                detJ_b = jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1)
                normal_b = jnp.asarray(np.stack(normal_b, axis=0))
                u_local_b = jnp.asarray(np.stack(u_local_batch, axis=0))
                dofs_batch_np = np.asarray(dofs_batch, dtype=int)
                assert n_a_local_const is not None
                assert n_b_local_const is not None
                _emit_batch(
                    Na_b,
                    Nb_b,
                    gradNa_b,
                    gradNb_b,
                    x_q_b,
                    w_b,
                    detJ_b,
                    normal_b,
                    u_local_b,
                    dofs_batch_np,
                    int(n_a_local_const),
                    int(n_b_local_const),
                    int(Na_b.shape[0]),
                )
                batch_items = []
                dofs_batch = []
                u_local_batch = []

        if not batch_failed and batch_items:
            Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
            Na_b = jnp.asarray(np.stack(Na_b, axis=0))
            Nb_b = jnp.asarray(np.stack(Nb_b, axis=0))
            gradNa_b = jnp.asarray(np.stack(gradNa_b, axis=0))
            gradNb_b = jnp.asarray(np.stack(gradNb_b, axis=0))
            x_q_b = jnp.asarray(np.stack(x_q_b, axis=0))
            w_b = jnp.asarray(np.stack(w_b, axis=0))
            detJ_b = jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1)
            normal_b = jnp.asarray(np.stack(normal_b, axis=0))
            u_local_b = jnp.asarray(np.stack(u_local_batch, axis=0))
            dofs_batch_np = np.asarray(dofs_batch, dtype=int)
            assert n_a_local_const is not None
            assert n_b_local_const is not None
            _emit_batch(
                Na_b,
                Nb_b,
                gradNa_b,
                gradNb_b,
                x_q_b,
                w_b,
                detJ_b,
                normal_b,
                u_local_b,
                dofs_batch_np,
                int(n_a_local_const),
                int(n_b_local_const),
                int(Na_b.shape[0]),
            )

        if not batch_failed and (batch_rows or (not sparse and K_dense is not None)):
            if sparse:
                if batch_rows:
                    rows_np = np.concatenate(batch_rows)
                    cols_np = np.concatenate(batch_cols)
                    data_np = jnp.concatenate([jnp.asarray(x) for x in batch_data]) if backend == "jax" else np.concatenate(batch_data)
                else:
                    rows_np = np.zeros((0,), dtype=int)
                    cols_np = np.zeros((0,), dtype=int)
                    data_np = jnp.zeros((0,), dtype=jnp.float64) if backend == "jax" else np.zeros((0,), dtype=float)
                from ..solver import FluxSparseMatrix

                return FluxSparseMatrix(rows_np, cols_np, data_np, n_dofs=n_total)
            assert K_dense is not None
            return K_dense

        if trace:
            _trace("[CONTACT] batch_jac_fallback")

    pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=use_p0_a,
        use_p0_b=use_p0_b,
    )
    trial_use_p0_a = str(trial_space_mode_a) == "p0"
    trial_use_p0_b = str(trial_space_mode_b) == "p0"
    trial_pair_basis_builder = _select_supermesh_pair_basis_builder(
        use_p0_a=trial_use_p0_a,
        use_p0_b=trial_use_p0_b,
    )
    if trace:
        _trace("[CONTACT] supermesh_loop_enter")
    _contact_interface_dbg("[contact-interface] step: supermesh loop START")
    t_loop = time.perf_counter()
    use_cached_geom = supermesh_quad_cache is not None and not diag_force
    if use_cached_geom:
        assert supermesh_quad_cache is not None
        quad_pts_cached = np.asarray(supermesh_quad_cache.quad_pts, dtype=float)
        quad_w_cached = np.asarray(supermesh_quad_cache.quad_w, dtype=float)
        detJ_cached = np.asarray(supermesh_quad_cache.detJ, dtype=float)
        x_q_cached = np.asarray(supermesh_quad_cache.x_q, dtype=float)
        tri_iter = enumerate(zip(source_facets_a, source_facets_b))
    else:
        tri_iter = enumerate(
            zip(
                _iter_supermesh_tris(supermesh_coords, supermesh_conn),
                source_facets_a,
                source_facets_b,
            )
        )
    for it, tri_item in tri_iter:
        log_tri = trace and (it < trace_max or it % trace_every == 0)
        t_tri0 = time.perf_counter()
        def _tri_check(stage: str) -> None:
            if tri_timeout > 0.0 and (time.perf_counter() - t_tri0) > tri_timeout:
                raise RuntimeError(f"[CONTACT] tri {it} timeout at {stage}")
        if use_cached_geom:
            fa, fb = tri_item
            if log_tri:
                _trace(f"[CONTACT] tri {it} start fa={int(fa)} fb={int(fb)} [cached-geom]")
            detJ = float(detJ_cached[it])
            if detJ <= 2.0 * tol:
                continue
            if skip_small_tri and facet_area_a is not None and facet_area_b is not None:
                area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
                if area_ref > 0.0 and 0.5 * detJ < area_scale * area_ref:
                    continue
            if guard:
                if not np.isfinite(detJ):
                    if skip_nonfinite:
                        continue
                    raise RuntimeError(f"[CONTACT] tri {it} detJ non-finite")
                if detj_eps > 0.0 and abs(detJ) < detj_eps:
                    if log_tri:
                        _trace(f"[CONTACT] tri {it} detJ too small {detJ:.3e}; skip")
                    continue
                if not np.isfinite(x_q_cached[it]).all():
                    if skip_nonfinite:
                        continue
                    raise RuntimeError(f"[CONTACT] tri {it} x_q non-finite")
            geom = _JacobianTriangleGeometryData(
                detJ=abs(detJ) if diag_abs_detj else detJ,
                quad_pts=quad_pts_cached,
                quad_w=quad_w_cached,
                quad_source="fluxfem-cache",
                facet_a=facets_a[int(fa)],
                facet_b=facets_b[int(fb)],
                x_q=x_q_cached[it],
            )
        else:
            (tri, a, b, c), fa, fb = tri_item
            if log_tri:
                _trace(f"[CONTACT] tri {it} start fa={int(fa)} fb={int(fb)}")
            geom = _prepare_supermesh_jacobian_triangle_geometry(
                it=it,
                log_tri=log_tri,
                a=a,
                b=b,
                c=c,
                fa=int(fa),
                fb=int(fb),
                tol=tol,
                skip_small_tri=skip_small_tri,
                area_scale=area_scale,
                facet_area_a=facet_area_a,
                facet_area_b=facet_area_b,
                diag_force=diag_force,
                diag_abs_detj=diag_abs_detj,
                guard=guard,
                skip_nonfinite=skip_nonfinite,
                detj_eps=detj_eps,
                quad_order=quad_order,
                diag_qp_mode=diag_qp_mode,
                diag_qp_path=diag_qp_path,
                facets_a=facets_a,
                facets_b=facets_b,
                tri_check=_tri_check,
                trace_fn=_trace,
                trace_time_fn=_trace_time,
            )
        if geom is None:
            continue

        K_dense = _accumulate_supermesh_jacobian_triangle_core(
            it=it,
            log_tri=log_tri,
            fa=int(fa),
            fb=int(fb),
            facet_a=geom.facet_a,
            facet_b=geom.facet_b,
            x_q=geom.x_q,
            quad_pts=geom.quad_pts,
            quad_w=geom.quad_w,
            quad_source=geom.quad_source,
            detJ=geom.detJ,
            tol=tol,
            pair_basis_builder=pair_basis_builder,
            trial_pair_basis_builder=trial_pair_basis_builder,
            coords_a=coords_a,
            coords_b=coords_b,
            facets_a=facets_a,
            facets_b=facets_b,
            value_dim_a=value_dim_a,
            value_dim_b=value_dim_b,
            trial_value_dim_a=int(trial_value_dim_a),
            trial_value_dim_b=int(trial_value_dim_b),
            dof_source=dof_source,
            grad_source=grad_source,
            space_mode_a=space_mode_a,
            space_mode_b=space_mode_b,
            trial_space_mode_a=str(trial_space_mode_a),
            trial_space_mode_b=str(trial_space_mode_b),
            use_elem_a=use_elem_a,
            use_elem_b=use_elem_b,
            elem_conn_a=elem_conn_a,
            elem_conn_b=elem_conn_b,
            facet_to_elem_a=facet_to_elem_a,
            facet_to_elem_b=facet_to_elem_b,
            facet_dofs_a=facet_dofs_a,
            facet_dofs_b=facet_dofs_b,
            trial_facet_dofs_a=trial_facet_dofs_a,
            trial_facet_dofs_b=trial_facet_dofs_b,
            proj_diag=proj_diag,
            diag_map=diag_map,
            diag_n=diag_n,
            diag_force=diag_force,
            diag_facet=diag_facet,
            diag_max_q=diag_max_q,
            guard=guard,
            skip_nonfinite=skip_nonfinite,
            normal_source=normal_source,
            normal_sign=normal_sign,
            normals_a=normals_a,
            normals_b=normals_b,
            field_a=field_a,
            field_b=field_b,
            u_a=u_a_np,
            u_b=u_b_np,
            res_form=res_form,
            params=params,
            includes_measure=includes_measure,
            offset_a=int(offset_a),
            offset_b=int(offset_b),
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            sparse=sparse,
            rows=rows,
            cols=cols,
            data=data,
            K_dense=K_dense,
            callbacks=callbacks,
            tri_check=_tri_check,
            trace_fn=_trace,
            trace_time_fn=_trace_time,
        )


    if proj_diag:
        _proj_diag_report()
    if sparse:
        from ..solver import FluxSparseMatrix

        return FluxSparseMatrix(
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
            jnp.asarray(data, dtype=jnp.float64) if backend == "jax" else np.asarray(data, dtype=float),
            n_dofs=n_total,
        )
    assert K_dense is not None
    return K_dense


def assemble_onesided_bilinear(
    surface_slave: SurfaceMesh,
    u_hat_fn,
    params: "WeakParams",
    *,
    surface_master: SurfaceMesh | None = None,
    u_master: np.ndarray | None = None,
    value_dim: int = 3,
    elem_conn: np.ndarray | None = None,
    facet_to_elem: np.ndarray | None = None,
    elem_conn_master: np.ndarray | None = None,
    facet_to_elem_master: np.ndarray | None = None,
    grad_source: str = "volume",
    dof_source: str = "volume",
    quad_order: int = 2,
    normal_sign: float = 1.0,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble one-sided (slave-only) Nitsche matrices without supermesh.

    The master side is treated as prescribed displacement u_hat(x). Provide
    either u_hat_fn(x_q) or u_master with master element mappings to evaluate
    u_hat at slave quadrature points.

    Note: this implementation currently assumes volume-trace bases for both
    gradients and DOFs. Surface-only bases are not supported here yet.
    """
    from ..core.forms import FieldPair
    coords_s = np.asarray(surface_slave.coords, dtype=float)
    facets_s = np.asarray(surface_slave.conn, dtype=int)
    coords_m = np.asarray(surface_master.coords, dtype=float) if surface_master is not None else coords_s
    facets_m = np.asarray(surface_master.conn, dtype=int) if surface_master is not None else facets_s
    n_s = int(coords_s.shape[0] * value_dim)
    K: np.ndarray = np.zeros((n_s, n_s), dtype=float)
    f: np.ndarray = np.zeros((n_s,), dtype=float)

    normals_s = surface_slave.facet_normals() if hasattr(surface_slave, "facet_normals") else None
    use_elem = elem_conn is not None and facet_to_elem is not None
    use_master = u_master is not None

    if use_master:
        if surface_master is None:
            raise ValueError("surface_master is required when u_master is provided")
        if elem_conn_master is None or facet_to_elem_master is None:
            raise ValueError("elem_conn_master and facet_to_elem_master are required when u_master is provided")
    else:
        if u_hat_fn is None:
            raise ValueError("u_hat_fn or u_master must be provided")
        if surface_master is None:
            surface_master = surface_slave

    if grad_source != "volume" or dof_source != "volume":
        raise ValueError("one-sided Nitsche currently supports only volume/volume")

    from ..core.weakform import (
        Params,
        compile_mixed_surface_residual_numpy,
        param_ref,
        test_ref,
        unknown_ref,
    )
    import fluxfem.helpers_wf as h_wf

    u = unknown_ref("u")
    v = test_ref("u")
    p = param_ref()
    n = h_wf.normal()
    t_u = h_wf.traction(u, n, p)
    t_v = h_wf.traction(v, n, p)
    sym_term = h_wf.einsum("qia,qi->qa", t_v, u.val)
    sym_term_hat = h_wf.einsum("qia,qi->qa", t_v, p.u_hat)

    disable_consistency = os.getenv("FF_ONESIDED_DISABLE_CONSISTENCY", "").strip().lower() in {"1", "true", "yes", "on"}
    disable_symmetry = os.getenv("FF_ONESIDED_DISABLE_SYMMETRY", "").strip().lower() in {"1", "true", "yes", "on"}
    disable_penalty = os.getenv("FF_ONESIDED_DISABLE_PENALTY", "").strip().lower() in {"1", "true", "yes", "on"}

    expr = 0.0
    if not disable_consistency:
        expr = expr - h_wf.dot(v, t_u)
    if not disable_symmetry:
        expr = expr - sym_term + sym_term_hat
    if not disable_penalty:
        expr = expr + (p.alpha * p.inv_h) * h_wf.dot(v, u.val) - (p.alpha * p.inv_h) * h_wf.dot(v, p.u_hat)
    expr = expr * h_wf.ds()
    res_form = compile_mixed_surface_residual_numpy({"u": expr})
    includes_measure = res_form._includes_measure

    quad_pts, quad_w = _tri_quadrature(quad_order) if quad_order > 0 else (np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5]))

    for f_id, facet in enumerate(facets_s):
        triangles = _facet_triangles(coords_s, facet)
        if not triangles:
            continue
        area_f = _facet_area_estimate(facet, coords_s)
        if area_f <= tol:
            continue
        inv_h = 1.0 / max(np.sqrt(area_f), tol)

        elem_nodes = None
        elem_coords = None
        local = None
        if use_elem:
            assert facet_to_elem is not None
            assert elem_conn is not None
            elem_id = int(facet_to_elem[int(f_id)])
            if elem_id < 0:
                raise ValueError("facet_to_elem has invalid mapping")
            elem_nodes = np.asarray(elem_conn[elem_id], dtype=int)
            elem_coords = coords_s[elem_nodes]

        for a, b, c in triangles:
            area = _tri_area(a, b, c)
            if area <= tol:
                continue
            detJ = 2.0 * area
            x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)
            if use_master:
                if dof_source == "surface":
                    assert u_master is not None
                    facet_m = facets_m[int(f_id)]
                    u_master_local = _gather_u_local(u_master, facet_m, value_dim).reshape(-1, value_dim)
                    N_master = np.array(
                        [_facet_shape_values(pt, facet_m, coords_m, tol=tol) for pt in x_q],
                        dtype=float,
                    )
                    u_hat = N_master @ u_master_local
                else:
                    assert u_master is not None
                    assert facet_to_elem_master is not None
                    assert elem_conn_master is not None
                    elem_id_m = int(facet_to_elem_master[int(f_id)])
                    if elem_id_m < 0:
                        raise ValueError("facet_to_elem_master has invalid mapping")
                    elem_nodes_m = np.asarray(elem_conn_master[elem_id_m], dtype=int)
                    elem_coords_m = coords_m[elem_nodes_m]
                    u_master_local = _gather_u_local(u_master, elem_nodes_m, value_dim).reshape(-1, value_dim)
                    N_master = _volume_shape_values_at_points(x_q, elem_coords_m, tol=tol)
                    u_hat = N_master @ u_master_local
            else:
                u_hat = np.asarray(u_hat_fn(x_q), dtype=float)
                if u_hat.shape[0] != x_q.shape[0]:
                    raise ValueError("u_hat_fn must return shape (n_q, value_dim)")

            gradN = None
            nodes = facet
            N = None

            if grad_source == "surface":
                gradN = np.array(
                    [_surface_gradN(pt, facet, coords_s, tol=tol) for pt in x_q],
                    dtype=float,
                )
            if use_elem and grad_source == "volume":
                assert elem_nodes is not None
                assert elem_coords is not None
                local = _local_indices(elem_nodes, facet)
                gradN = _tet_gradN_at_points(x_q, elem_coords, local=local, tol=tol)

            if dof_source == "volume":
                if not use_elem or elem_nodes is None or elem_coords is None:
                    raise ValueError("dof_source 'volume' requires elem_conn and facet_to_elem")
                nodes = elem_nodes
                N = _volume_shape_values_at_points(x_q, elem_coords, tol=tol)
                if grad_source == "volume":
                    gradN = _tet_gradN_at_points(x_q, elem_coords, tol=tol)
            else:
                N = np.array([_facet_shape_values(pt, facet, coords_s, tol=tol) for pt in x_q], dtype=float)

            field = SurfaceMixedFormField(
                N=N,
                gradN=gradN,
                value_dim=value_dim,
                basis=_SurfaceBasis(dofs_per_node=value_dim),
            )
            fields = {"u": FieldPair(test=cast("FormFieldLike", field), trial=cast("FormFieldLike", field))}
            normal = normals_s[int(f_id)] if normals_s is not None else None
            if normal is not None:
                normal = normal_sign * normal
            normal_q = None if normal is None else np.repeat(normal[None, :], quad_pts.shape[0], axis=0)
            ctx = SurfaceMixedFormContext(
                bindings=fields,
                x_q=x_q,
                w=quad_w,
                detJ=np.array([detJ], dtype=float),
                normal=normal_q,
                spaces=fields,
            )
            params_local = Params(
                lam=params.lam,
                mu=params.mu,
                alpha=params.alpha,
                inv_h=inv_h,
                u_hat=u_hat,
            )
            u_zero: np.ndarray = np.zeros((len(nodes) * value_dim,), dtype=float)
            u_dict = {"u": u_zero}
            sizes = (u_zero.shape[0],)
            slices = {"u": slice(0, sizes[0])}

            def _res_local_np_single(u_vec: np.ndarray) -> np.ndarray:
                u_local = {"u": u_vec[slices["u"]]}
                fe_q = res_form(ctx, u_local, params_local)["u"]
                if includes_measure.get("u", False):
                    return np.sum(np.asarray(fe_q), axis=0)
                wJ = np.asarray(ctx.w) * np.asarray(ctx.detJ)
                return np.einsum("qi,q->i", np.asarray(fe_q), wJ)

            def _res_local_np(u_vec: np.ndarray) -> np.ndarray:
                if u_vec.ndim == 1:
                    return _res_local_np_single(u_vec)
                out = np.empty((u_vec.shape[0], u_vec.shape[1]), dtype=float)
                for col in range(u_vec.shape[1]):
                    out[:, col] = _res_local_np_single(u_vec[:, col])
                return out

            f_local = _res_local_np(u_zero)
            n_ldofs = int(u_zero.shape[0])
            k_local: np.ndarray = np.zeros((n_ldofs, n_ldofs), dtype=float)
            block = max(1, int(os.getenv("FLUXFEM_ONESIDE_BLOCK_SIZE", "16")))
            for start in range(0, n_ldofs, block):
                idxs: np.ndarray = np.arange(start, min(n_ldofs, start + block), dtype=int)
                u_block: np.ndarray = np.zeros((n_ldofs, idxs.size), dtype=float)
                u_block[idxs, np.arange(idxs.size, dtype=int)] = 1.0
                r_block = _res_local_np(u_block)
                k_local[:, idxs] = r_block - f_local[:, None]

            dofs = _global_dof_indices(nodes, value_dim, 0)
            f[dofs] += f_local
            K[np.ix_(dofs, dofs)] += k_local

    return K, f


def assemble_contact_onesided_floor(
    surface_slave: SurfaceMesh,
    u: np.ndarray,
    *,
    n: np.ndarray | None = None,
    c: float,
    k: float,
    beta: float,
    value_dim: int = 3,
    elem_conn: np.ndarray | None = None,
    facet_to_elem: np.ndarray | None = None,
    quad_order: int = 2,
    normal_sign: float = 1.0,
    tol: float = 1e-8,
    return_metrics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Assemble one-sided contact penalty against a rigid plane g = n·x - c.

    Uses softplus for a smooth contact pressure:
        p(g) = k * softplus(-g; beta)
    with softplus(z; beta) = (1 / beta) * log(1 + exp(beta z)).

    Note: the resulting stiffness matrix can be nonsymmetric; avoid CG.
    """
    if elem_conn is None or facet_to_elem is None:
        raise ValueError("elem_conn and facet_to_elem are required")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    import jax
    import jax.numpy as jnp

    coords_s = np.asarray(surface_slave.coords, dtype=float)
    facets_s = np.asarray(surface_slave.conn, dtype=int)
    n_s = int(coords_s.shape[0] * value_dim)
    K: np.ndarray = np.zeros((n_s, n_s), dtype=float)
    f: np.ndarray = np.zeros((n_s,), dtype=float)

    normals_s = surface_slave.facet_normals() if hasattr(surface_slave, "facet_normals") else None
    if n is not None:
        n = np.asarray(n, dtype=float).reshape(-1)
        if n.shape[0] != 3:
            raise ValueError("n must be a 3-vector")
        n_norm = np.linalg.norm(n)
        if n_norm <= tol:
            raise ValueError("n must be non-zero")
        n = (n / n_norm) * float(normal_sign)
    elif normals_s is None:
        raise ValueError("surface normals are required when n is not provided")

    penetration = 0.0
    min_g = float("inf")
    quad_pts, quad_w = _tri_quadrature(quad_order) if quad_order > 0 else (np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5]))

    for f_id, facet in enumerate(facets_s):
        triangles = _facet_triangles(coords_s, facet)
        if not triangles:
            continue
        area_f = _facet_area_estimate(facet, coords_s)
        if area_f <= tol:
            continue

        elem_id = int(facet_to_elem[int(f_id)])
        if elem_id < 0:
            raise ValueError("facet_to_elem has invalid mapping")
        elem_nodes = np.asarray(elem_conn[elem_id], dtype=int)
        elem_coords = coords_s[elem_nodes]
        u_local = _gather_u_local(u, elem_nodes, value_dim).reshape(-1, value_dim)

        if n is not None:
            normal = n
        else:
            assert normals_s is not None
            normal = normal_sign * normals_s[int(f_id)]

        for a, b, c_tri in triangles:
            area = _tri_area(a, b, c_tri)
            if area <= tol:
                continue
            detJ = 2.0 * area
            x_q_ref = np.array([a + r * (b - a) + s * (c_tri - a) for r, s in quad_pts], dtype=float)
            N = _volume_shape_values_at_points(x_q_ref, elem_coords, tol=tol)

            normal_q = np.repeat(normal[None, :], quad_pts.shape[0], axis=0)

            u_q_np = N @ u_local
            x_q_cur = x_q_ref + u_q_np
            g_np = np.sum(normal_q * x_q_cur, axis=1) - float(c)
            min_g = min(min_g, float(np.min(g_np)))
            z_np = -float(beta) * g_np
            z_clip = np.minimum(z_np, 30.0)
            softplus_np = np.where(z_np > 30.0, z_np, np.log1p(np.exp(z_clip))) / float(beta)
            penetration += float(np.sum(softplus_np * quad_w) * detJ)

            def _res_local(u_vec):
                u_loc = u_vec.reshape(-1, value_dim)
                u_q = jnp.einsum("qi,ia->qa", jnp.asarray(N), u_loc)
                x_q_j = jnp.asarray(x_q_ref)
                n_q = jnp.asarray(normal_q)
                x_q_cur_j = x_q_j + u_q
                g = jnp.einsum("qa,qa->q", n_q, x_q_cur_j) - float(c)
                p = float(k) * jax.nn.softplus(-float(beta) * g) / float(beta)
                t = p[:, None] * n_q
                wJ = jnp.asarray(quad_w) * float(detJ)
                nodal = jnp.einsum("qi,qa,q->ia", jnp.asarray(N), t, wJ)
                return nodal.reshape(-1)

            u_vec0 = np.asarray(u_local.reshape(-1), dtype=float)
            f_local = np.asarray(_res_local(jnp.asarray(u_vec0)))
            k_local = np.asarray(jax.jacrev(_res_local)(jnp.asarray(u_vec0)))

            dofs = _global_dof_indices(elem_nodes, value_dim, 0)
            for i, gi in enumerate(dofs):
                f[int(gi)] += float(f_local[i])
                for j, gj in enumerate(dofs):
                    K[int(gi), int(gj)] += float(k_local[i, j])

    if return_metrics:
        if min_g == float("inf"):
            min_g = 0.0
        metrics = {
            "penetration": float(penetration),
            "min_g": float(min_g),
        }
        return K, f, metrics
    return K, f

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contact_geometry import (
    _facet_shape_values,
    _local_indices,
    _surface_gradN,
    _tet_gradN_at_points,
    _volume_shape_values_at_points,
)


@dataclass(eq=False)
class _SupermeshPairBasisData:
    elem_id_a: int
    elem_id_b: int
    Na: np.ndarray
    Nb: np.ndarray
    gradNa: np.ndarray
    gradNb: np.ndarray
    dofs_local_a: np.ndarray
    dofs_local_b: np.ndarray
    nodes_a: np.ndarray
    nodes_b: np.ndarray
    local_a: np.ndarray | None
    local_b: np.ndarray | None
    test_Na: np.ndarray | None = None
    test_Nb: np.ndarray | None = None
    trial_Na: np.ndarray | None = None
    trial_Nb: np.ndarray | None = None
    test_gradNa: np.ndarray | None = None
    test_gradNb: np.ndarray | None = None
    trial_gradNa: np.ndarray | None = None
    trial_gradNb: np.ndarray | None = None
    test_dofs_local_a: np.ndarray | None = None
    test_dofs_local_b: np.ndarray | None = None
    trial_dofs_local_a: np.ndarray | None = None
    trial_dofs_local_b: np.ndarray | None = None
    test_nodes_a: np.ndarray | None = None
    test_nodes_b: np.ndarray | None = None
    trial_nodes_a: np.ndarray | None = None
    trial_nodes_b: np.ndarray | None = None


def _merge_trial_pair_basis_data(
    base: _SupermeshPairBasisData,
    trial: _SupermeshPairBasisData,
) -> _SupermeshPairBasisData:
    base.trial_Na = trial.Na
    base.trial_Nb = trial.Nb
    base.trial_gradNa = trial.gradNa
    base.trial_gradNb = trial.gradNb
    base.trial_dofs_local_a = trial.dofs_local_a
    base.trial_dofs_local_b = trial.dofs_local_b
    base.trial_nodes_a = trial.nodes_a
    base.trial_nodes_b = trial.nodes_b
    return base


def _same_optional_int_array(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int))


def _gather_u_local(u_field: np.ndarray, nodes: np.ndarray, value_dim: int) -> np.ndarray:
    if value_dim == 1:
        return u_field[nodes]
    idx = np.repeat(nodes * value_dim, value_dim) + np.tile(np.arange(value_dim), len(nodes))
    return u_field[idx]


def _global_dof_indices(nodes: np.ndarray, value_dim: int, offset: int) -> np.ndarray:
    if value_dim == 1:
        return offset + nodes
    idx = np.repeat(nodes * value_dim, value_dim) + np.tile(np.arange(value_dim), len(nodes))
    return offset + idx


def _facet_local_dofs(
    facet_id: int,
    *,
    n_facets: int,
    value_dim: int,
    facet_dofs: np.ndarray | None,
) -> np.ndarray:
    """
    Return local (field-relative) DOF indices for a facet-wise space.

    - When ``facet_dofs`` is provided, it must be shape ``(n_facets, n_ldofs_facet)``.
    - Otherwise, a default contiguous layout is used:
      ``facet_id * value_dim + [0..value_dim-1]``.
    """
    if facet_dofs is not None:
        arr = np.asarray(facet_dofs, dtype=int)
        if arr.ndim != 2:
            raise ValueError("facet_dofs must be a 2D array (n_facets, n_ldofs_facet).")
        if arr.shape[0] != int(n_facets):
            raise ValueError(
                f"facet_dofs first dimension mismatch: got {arr.shape[0]}, expected {int(n_facets)}."
            )
        return arr[int(facet_id)]
    start = int(facet_id) * int(value_dim)
    return start + np.arange(int(value_dim), dtype=int)


def _field_n_dofs(
    *,
    n_nodes: int,
    n_facets: int,
    value_dim: int,
    space_mode: str,
    facet_dofs: np.ndarray | None,
) -> int:
    if space_mode == "p0":
        if facet_dofs is not None:
            arr = np.asarray(facet_dofs, dtype=int)
            if arr.size == 0:
                return 0
            if np.any(arr < 0):
                raise ValueError("facet_dofs must be non-negative.")
            return int(arr.max()) + 1
        return int(n_facets) * int(value_dim)
    return int(n_nodes) * int(value_dim)


def _build_side_field_data_p0(
    *,
    n_q: int,
    facet_id: int,
    n_facets: int,
    value_dim: int,
    facet_nodes: np.ndarray,
    facet_dofs: np.ndarray | None,
    local: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    dofs_local = _facet_local_dofs(
        int(facet_id),
        n_facets=n_facets,
        value_dim=value_dim,
        facet_dofs=facet_dofs,
    )
    n_ldofs = int(dofs_local.shape[0])
    N = np.ones((n_q, n_ldofs), dtype=float)
    gradN = np.zeros((n_q, n_ldofs, 3), dtype=float)
    return N, gradN, dofs_local, facet_nodes, local


def _build_side_field_data_nodal(
    *,
    x_q: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    value_dim: int,
    dof_source: str,
    use_elem: bool,
    elem_nodes: np.ndarray | None,
    elem_coords: np.ndarray | None,
    tol: float,
    volume_dof_error: str,
    grad_source: str,
    gradN: np.ndarray | None,
    local: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    if dof_source == "volume":
        if not use_elem or elem_nodes is None or elem_coords is None:
            raise ValueError(volume_dof_error)
        N = _volume_shape_values_at_points(x_q, elem_coords, tol=tol)
        if grad_source == "volume":
            gradN = _tet_gradN_at_points(x_q, elem_coords, tol=tol)
        dofs_local = _global_dof_indices(elem_nodes, value_dim, 0)
        return N, gradN, dofs_local, elem_nodes, local

    N = np.array([_facet_shape_values(pt, facet_nodes, coords, tol=tol) for pt in x_q], dtype=float)
    dofs_local = _global_dof_indices(facet_nodes, value_dim, 0)
    return N, gradN, dofs_local, facet_nodes, local


def _build_side_grad_data(
    *,
    x_q: np.ndarray,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    grad_source: str,
    use_elem: bool,
    elem_nodes: np.ndarray | None,
    elem_coords: np.ndarray | None,
    tol: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    gradN = None
    local = None
    if grad_source == "surface":
        gradN = np.array(
            [_surface_gradN(pt, facet_nodes, coords, tol=tol) for pt in x_q],
            dtype=float,
        )
    if use_elem and grad_source == "volume":
        assert elem_nodes is not None
        assert elem_coords is not None
        local = _local_indices(elem_nodes, facet_nodes)
        gradN = _tet_gradN_at_points(x_q, elem_coords, local=local, tol=tol)
    return gradN, local


def _prepare_supermesh_side_field_data(
    *,
    x_q: np.ndarray,
    facet_id: int,
    facet_nodes: np.ndarray,
    coords: np.ndarray,
    value_dim: int,
    n_facets: int,
    dof_source: str,
    grad_source: str,
    space_mode: str,
    use_elem: bool,
    elem_nodes: np.ndarray | None,
    elem_coords: np.ndarray | None,
    facet_dofs: np.ndarray | None,
    tol: float,
    volume_dof_error: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    gradN, local = _build_side_grad_data(
        x_q=x_q,
        facet_nodes=facet_nodes,
        coords=coords,
        grad_source=grad_source,
        use_elem=use_elem,
        elem_nodes=elem_nodes,
        elem_coords=elem_coords,
        tol=tol,
    )
    if space_mode == "p0":
        return _build_side_field_data_p0(
            n_q=int(x_q.shape[0]),
            facet_id=int(facet_id),
            n_facets=n_facets,
            value_dim=value_dim,
            facet_nodes=facet_nodes,
            facet_dofs=facet_dofs,
            local=local,
        )
    return _build_side_field_data_nodal(
        x_q=x_q,
        facet_nodes=facet_nodes,
        coords=coords,
        value_dim=value_dim,
        dof_source=dof_source,
        use_elem=use_elem,
        elem_nodes=elem_nodes,
        elem_coords=elem_coords,
        tol=tol,
        volume_dof_error=volume_dof_error,
        grad_source=grad_source,
        gradN=gradN,
        local=local,
    )


def _resolve_facet_element_context(
    *,
    use_elem: bool,
    facet_id: int,
    facet_to_elem: np.ndarray | None,
    elem_conn: np.ndarray | None,
    coords: np.ndarray,
    invalid_map_error: str,
    unsupported_error: str,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    if not use_elem:
        return -1, None, None
    assert facet_to_elem is not None
    assert elem_conn is not None
    elem_id = int(facet_to_elem[int(facet_id)])
    if elem_id < 0:
        raise ValueError(invalid_map_error)
    elem_nodes = np.asarray(elem_conn[elem_id], dtype=int)
    elem_coords = coords[elem_nodes]
    if elem_coords.shape[0] not in {4, 8, 10, 20, 27}:
        raise NotImplementedError(unsupported_error)
    return elem_id, elem_nodes, elem_coords

def _prepare_supermesh_pair_basis_data(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    space_mode_a: str,
    space_mode_b: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    elem_id_a, elem_nodes_a, elem_coords_a = _resolve_facet_element_context(
        use_elem=use_elem_a,
        facet_id=int(fa),
        facet_to_elem=facet_to_elem_a,
        elem_conn=elem_conn_a,
        coords=coords_a,
        invalid_map_error="facet_to_elem_a has invalid mapping",
        unsupported_error="surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only",
    )
    elem_id_b, elem_nodes_b, elem_coords_b = _resolve_facet_element_context(
        use_elem=use_elem_b,
        facet_id=int(fb),
        facet_to_elem=facet_to_elem_b,
        elem_conn=elem_conn_b,
        coords=coords_b,
        invalid_map_error="facet_to_elem_b has invalid mapping",
        unsupported_error="surface sym_grad is implemented for tet4/tet10/hex8/hex20/hex27 only",
    )
    Na, gradNa, dofs_local_a, nodes_a, local_a = _prepare_supermesh_side_field_data(
        x_q=x_q,
        facet_id=int(fa),
        facet_nodes=facet_a,
        coords=coords_a,
        value_dim=value_dim_a,
        n_facets=facets_a.shape[0],
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode=space_mode_a,
        use_elem=use_elem_a,
        elem_nodes=elem_nodes_a,
        elem_coords=elem_coords_a,
        facet_dofs=facet_dofs_a,
        tol=tol,
        volume_dof_error="dof_source 'volume' requires elem_conn_a and facet_to_elem_a",
    )
    Nb, gradNb, dofs_local_b, nodes_b, local_b = _prepare_supermesh_side_field_data(
        x_q=x_q,
        facet_id=int(fb),
        facet_nodes=facet_b,
        coords=coords_b,
        value_dim=value_dim_b,
        n_facets=facets_b.shape[0],
        dof_source=dof_source,
        grad_source=grad_source,
        space_mode=space_mode_b,
        use_elem=use_elem_b,
        elem_nodes=elem_nodes_b,
        elem_coords=elem_coords_b,
        facet_dofs=facet_dofs_b,
        tol=tol,
        volume_dof_error="dof_source 'volume' requires elem_conn_b and facet_to_elem_b",
    )
    return _SupermeshPairBasisData(
        elem_id_a=elem_id_a,
        elem_id_b=elem_id_b,
        Na=Na,
        Nb=Nb,
        gradNa=gradNa,
        gradNb=gradNb,
        dofs_local_a=dofs_local_a,
        dofs_local_b=dofs_local_b,
        nodes_a=nodes_a,
        nodes_b=nodes_b,
        local_a=local_a,
        local_b=local_b,
        test_Na=Na,
        test_Nb=Nb,
        trial_Na=Na,
        trial_Nb=Nb,
        test_gradNa=gradNa,
        test_gradNb=gradNb,
        trial_gradNa=gradNa,
        trial_gradNb=gradNb,
        test_dofs_local_a=dofs_local_a,
        test_dofs_local_b=dofs_local_b,
        trial_dofs_local_a=dofs_local_a,
        trial_dofs_local_b=dofs_local_b,
        test_nodes_a=nodes_a,
        test_nodes_b=nodes_b,
        trial_nodes_a=nodes_a,
        trial_nodes_b=nodes_b,
    )


def _prepare_supermesh_pair_basis_data_nodal(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
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
        space_mode_a="nodal",
        space_mode_b="nodal",
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


def _prepare_supermesh_pair_basis_data_p0_a_nodal_b(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
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
        space_mode_a="p0",
        space_mode_b="nodal",
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


def _prepare_supermesh_pair_basis_data_nodal_a_p0_b(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
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
        space_mode_a="nodal",
        space_mode_b="p0",
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


def _prepare_supermesh_pair_basis_data_p0_p0(
    *,
    fa: int,
    fb: int,
    facet_a: np.ndarray,
    facet_b: np.ndarray,
    x_q: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    facets_a: np.ndarray,
    facets_b: np.ndarray,
    value_dim_a: int,
    value_dim_b: int,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    elem_conn_a: np.ndarray | None,
    elem_conn_b: np.ndarray | None,
    facet_to_elem_a: np.ndarray | None,
    facet_to_elem_b: np.ndarray | None,
    facet_dofs_a: np.ndarray | None,
    facet_dofs_b: np.ndarray | None,
    tol: float,
) -> _SupermeshPairBasisData:
    return _prepare_supermesh_pair_basis_data(
        fa=fa,
        fb=fb,
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
        space_mode_a="p0",
        space_mode_b="p0",
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


def _select_supermesh_pair_basis_builder(
    *,
    use_p0_a: bool,
    use_p0_b: bool,
) -> Callable[..., _SupermeshPairBasisData]:
    if not use_p0_a and not use_p0_b:
        return _prepare_supermesh_pair_basis_data_nodal
    if use_p0_a and use_p0_b:
        return _prepare_supermesh_pair_basis_data_p0_p0
    if use_p0_a:
        return _prepare_supermesh_pair_basis_data_p0_a_nodal_b
    return _prepare_supermesh_pair_basis_data_nodal_a_p0_b

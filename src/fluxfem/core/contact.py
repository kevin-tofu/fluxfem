from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


# -----------------------------------------------------------------------------
# DOF helpers


def vector_dofs_from_nodes(nodes: Array, dim: int) -> Array:
    """Expand node ids to vector-valued DOFs with node-major ordering."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    nodes_np = np.asarray(nodes, dtype=np.int32).reshape(-1)
    if nodes_np.size and nodes_np.min() < 0:
        raise ValueError("nodes must not contain negative ids.")
    nodes_j = jnp.asarray(nodes_np, dtype=jnp.int32)
    offsets = jnp.arange(dim, dtype=jnp.int32)
    return (nodes_j[:, None] * dim + offsets[None, :]).reshape(-1)


def retained_dofs_from_surface(surface, dim: int) -> Array:
    """Return sorted vector DOFs touched by a SurfaceMesh-like object."""
    nodes = np.unique(np.asarray(surface.conn, dtype=np.int32).reshape(-1))
    return vector_dofs_from_nodes(nodes, dim)


# -----------------------------------------------------------------------------
# Geometry helpers


def orthonormal_tangent_basis(normals: Array) -> Array:
    """
    Build an orthonormal tangent basis for each contact normal.

    `normals` must have shape `(n_contact, dim)`. The returned array has shape
    `(n_contact, dim - 1, dim)`, with each tangent vector orthogonal to the
    corresponding normalized normal. For `dim == 1`, the tangent dimension is
    zero and an empty basis is returned.
    """
    normals = jnp.asarray(normals)
    if normals.ndim != 2:
        raise ValueError("normals must have shape (n_contact, dim).")
    normals = normals.astype(jnp.result_type(normals, jnp.float32))
    n_contact = int(normals.shape[0])
    dim = int(normals.shape[1])
    if dim <= 0:
        raise ValueError("normal dimension must be positive.")
    dtype = normals.dtype
    if dim == 1:
        return jnp.zeros((n_contact, 0, 1), dtype=dtype)

    eye = jnp.eye(dim, dtype=dtype)

    def _basis_one(normal: Array) -> Array:
        norm = jnp.linalg.norm(normal)
        n = normal / jnp.maximum(norm, jnp.finfo(dtype).eps)
        align = jnp.abs(eye @ n)
        first_axis = eye[jnp.argmin(align)]
        t0 = first_axis - jnp.dot(first_axis, n) * n
        t0 = t0 / jnp.maximum(jnp.linalg.norm(t0), jnp.finfo(dtype).eps)
        tangents = [t0]
        for _ in range(1, dim - 1):
            scores = jnp.sum(jnp.abs(eye @ jnp.stack([n] + tangents, axis=0).T), axis=1)
            candidate = eye[jnp.argmin(scores)]
            for vector in [n] + tangents:
                candidate = candidate - jnp.dot(candidate, vector) * vector
            candidate = candidate / jnp.maximum(jnp.linalg.norm(candidate), jnp.finfo(dtype).eps)
            tangents.append(candidate)
        return jnp.stack(tangents, axis=0)

    return jax.vmap(_basis_one)(normals)


# -----------------------------------------------------------------------------
# Kinematics builders


def plane_contact_kinematics_from_surface(
    surface,
    *,
    dim: int,
    n_total_nodes: int | None = None,
    normal: Array,
    plane_offset: float,
) -> "ContactKinematics":
    """
    Build nodal plane-contact kinematics from a SurfaceMesh-like object.

    The reference gap is `dot(x0, normal) - plane_offset`, and the current gap is
    `gap0 + dot(u_node, normal)`.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    nodes = np.unique(np.asarray(surface.conn, dtype=np.int32).reshape(-1))
    if nodes.size == 0:
        raise ValueError("surface contains no contact nodes.")
    coords = np.asarray(surface.coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < dim:
        raise ValueError("surface.coords must have at least dim coordinate columns.")
    n_nodes = int(coords.shape[0]) if n_total_nodes is None else int(n_total_nodes)
    n_dofs = n_nodes * dim

    normal_arr = jnp.asarray(normal)
    if normal_arr.shape != (dim,):
        raise ValueError("normal must have shape (dim,).")
    normals = jnp.broadcast_to(normal_arr[None, :], (nodes.size, dim))
    dofs = vector_dofs_from_nodes(nodes, dim).reshape(nodes.size, dim)
    gaps0 = jnp.asarray(coords[nodes, :dim]) @ normal_arr - float(plane_offset)
    return ContactKinematics(dofs=dofs, normals=normals, gaps0=gaps0, n_dofs=n_dofs)


def paired_contact_kinematics_from_surfaces(
    slave_surface,
    master_surface,
    *,
    dim: int,
    normal: Array,
    n_total_nodes: int | None = None,
) -> "PairedContactKinematics":
    """
    Build fixed-normal node-pair contact kinematics from two surfaces.

    Slave nodes are paired to nearest master-surface nodes in the reference
    geometry. The reference gap is `dot(x_slave - x_master, normal)`.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    slave_nodes = np.unique(np.asarray(slave_surface.conn, dtype=np.int32).reshape(-1))
    master_nodes = np.unique(np.asarray(master_surface.conn, dtype=np.int32).reshape(-1))
    if slave_nodes.size == 0 or master_nodes.size == 0:
        raise ValueError("slave and master surfaces must contain contact nodes.")

    slave_coords = np.asarray(slave_surface.coords, dtype=float)
    master_coords = np.asarray(master_surface.coords, dtype=float)
    if slave_coords.ndim != 2 or slave_coords.shape[1] < dim:
        raise ValueError("slave_surface.coords must have at least dim coordinate columns.")
    if master_coords.ndim != 2 or master_coords.shape[1] < dim:
        raise ValueError("master_surface.coords must have at least dim coordinate columns.")
    if n_total_nodes is None:
        n_nodes = max(int(slave_coords.shape[0]), int(master_coords.shape[0]))
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim

    x_slave = slave_coords[slave_nodes, :dim]
    x_master = master_coords[master_nodes, :dim]
    dist2 = np.sum((x_slave[:, None, :] - x_master[None, :, :]) ** 2, axis=2)
    paired_master_nodes = master_nodes[np.argmin(dist2, axis=1)]

    normal_arr = jnp.asarray(normal)
    if normal_arr.shape != (dim,):
        raise ValueError("normal must have shape (dim,).")
    normals = jnp.broadcast_to(normal_arr[None, :], (slave_nodes.size, dim))
    slave_dofs = vector_dofs_from_nodes(slave_nodes, dim).reshape(slave_nodes.size, dim)
    master_dofs = vector_dofs_from_nodes(paired_master_nodes, dim).reshape(slave_nodes.size, dim)
    gaps0 = jnp.asarray(slave_coords[slave_nodes, :dim] - master_coords[paired_master_nodes, :dim]) @ normal_arr
    return PairedContactKinematics(
        slave_dofs=slave_dofs,
        master_dofs=master_dofs,
        normals=normals,
        gaps0=gaps0,
        n_dofs=n_dofs,
    )


# -----------------------------------------------------------------------------
# Projection and quadrature helpers


def _normalize_nonnegative_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(weights.shape, 1.0 / float(weights.size), dtype=float)
    return weights / total


def _line_shape_weights(point: np.ndarray, pts: np.ndarray) -> np.ndarray:
    edge = pts[1] - pts[0]
    denom = float(np.dot(edge, edge))
    if denom <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    t = float(np.dot(point - pts[0], edge) / denom)
    return _normalize_nonnegative_weights(np.array([1.0 - t, t], dtype=float))


def _triangle_shape_weights(point: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a = np.column_stack([pts[0] - pts[2], pts[1] - pts[2]])
    b = point - pts[2]
    try:
        coeff, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(3, 1.0 / 3.0, dtype=float)
    return _normalize_nonnegative_weights(
        np.array([coeff[0], coeff[1], 1.0 - coeff[0] - coeff[1]], dtype=float)
    )


def _quad_shape_functions(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )
    dN_dxi = 0.25 * np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    )
    dN_deta = 0.25 * np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    )
    return N, dN_dxi, dN_deta


def _quad_shape_weights(point: np.ndarray, pts: np.ndarray) -> np.ndarray:
    xi = 0.0
    eta = 0.0
    for _ in range(8):
        N, dN_dxi, dN_deta = _quad_shape_functions(xi, eta)
        x = N @ pts
        residual = x - point
        jac = np.column_stack([dN_dxi @ pts, dN_deta @ pts])
        try:
            delta, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        except np.linalg.LinAlgError:
            return np.full(4, 0.25, dtype=float)
        xi = float(np.clip(xi + delta[0], -1.0, 1.0))
        eta = float(np.clip(eta + delta[1], -1.0, 1.0))
        if float(np.linalg.norm(delta)) < 1e-12:
            break
    N, _, _ = _quad_shape_functions(xi, eta)
    return _normalize_nonnegative_weights(N)


def _facet_shape_weights(point: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 2:
        return _line_shape_weights(point, pts)
    if pts.shape[0] == 3:
        return _triangle_shape_weights(point, pts)
    if pts.shape[0] == 4:
        return _quad_shape_weights(point, pts)
    return np.full(pts.shape[0], 1.0 / float(pts.shape[0]), dtype=float)


def _facet_projected_point_and_weights(point: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = _facet_shape_weights(point, pts)
    return weights @ pts, weights


def _safe_unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.asarray(fallback, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _facet_normal(pts: np.ndarray, dim: int) -> np.ndarray:
    if dim == 1:
        return np.array([1.0], dtype=float)
    if dim == 2:
        if pts.shape[0] < 2:
            raise ValueError("2D automatic normals require at least a line facet.")
        edge = pts[1] - pts[0]
        return _safe_unit_vector(np.array([-edge[1], edge[0]], dtype=float), np.array([0.0, 1.0]))
    if dim == 3:
        if pts.shape[0] < 3:
            raise ValueError("3D automatic normals require triangle or quad facets.")
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        return _safe_unit_vector(normal, np.array([0.0, 0.0, 1.0]))
    raise ValueError("automatic node-surface normals support dim <= 3.")


def _surface_displacement_nodes(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    dim: int,
    n_total_nodes: int | None,
    displacement: Array | None,
) -> tuple[np.ndarray, int]:
    if n_total_nodes is None:
        n_nodes = max(int(coords_a.shape[0]), int(coords_b.shape[0]))
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * int(dim)
    if displacement is None:
        return np.zeros((n_nodes, int(dim)), dtype=float), n_nodes
    disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
    if disp_arr.size != n_dofs:
        raise ValueError("displacement length must match n_total_nodes * dim.")
    return disp_arr.reshape(n_nodes, int(dim)), n_nodes


def _aabb_intersections(box_min: np.ndarray, box_max: np.ndarray, master_min: np.ndarray, master_max: np.ndarray) -> np.ndarray:
    return np.all(master_max >= box_min[None, :], axis=1) & np.all(master_min <= box_max[None, :], axis=1)


@dataclass(frozen=True)
class ContactAABBIndex:
    """Uniform-grid spatial index over master facet AABBs."""

    facet_min: Array
    facet_max: Array
    cell_size: Array
    origin: Array
    cells: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    dim: int

    def __post_init__(self):
        facet_min = np.asarray(self.facet_min, dtype=float)
        facet_max = np.asarray(self.facet_max, dtype=float)
        cell_size = np.asarray(self.cell_size, dtype=float).reshape(-1)
        origin = np.asarray(self.origin, dtype=float).reshape(-1)
        dim = int(self.dim)
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if facet_min.ndim != 2 or facet_min.shape[1] != dim:
            raise ValueError("facet_min must have shape (n_facets, dim).")
        if facet_max.shape != facet_min.shape:
            raise ValueError("facet_max must have the same shape as facet_min.")
        if cell_size.shape != (dim,) or np.any(cell_size <= 0.0) or not np.all(np.isfinite(cell_size)):
            raise ValueError("cell_size must have positive finite entries.")
        if origin.shape != (dim,):
            raise ValueError("origin must have shape (dim,).")
        normalized_cells = []
        for cell_key, facet_ids in self.cells:
            key = tuple(int(v) for v in cell_key)
            ids = tuple(int(v) for v in facet_ids)
            normalized_cells.append((key, ids))
        object.__setattr__(self, "facet_min", jnp.asarray(facet_min))
        object.__setattr__(self, "facet_max", jnp.asarray(facet_max))
        object.__setattr__(self, "cell_size", jnp.asarray(cell_size))
        object.__setattr__(self, "origin", jnp.asarray(origin))
        object.__setattr__(self, "cells", tuple(normalized_cells))
        object.__setattr__(self, "dim", dim)

    @property
    def n_facets(self) -> int:
        return int(np.asarray(self.facet_min).shape[0])

    def query_box(self, box_min: Array, box_max: Array) -> np.ndarray:
        box_min_np = np.asarray(box_min, dtype=float).reshape(self.dim)
        box_max_np = np.asarray(box_max, dtype=float).reshape(self.dim)
        origin = np.asarray(self.origin, dtype=float)
        cell_size = np.asarray(self.cell_size, dtype=float)
        lo = np.floor((box_min_np - origin) / cell_size).astype(np.int32)
        hi = np.floor((box_max_np - origin) / cell_size).astype(np.int32)
        cell_map = dict(self.cells)
        candidates: set[int] = set()
        for key in np.ndindex(*((hi - lo + 1).astype(int))):
            cell_key = tuple((lo + np.asarray(key, dtype=np.int32)).tolist())
            candidates.update(cell_map.get(cell_key, ()))
        if not candidates:
            return np.empty((0,), dtype=np.int32)
        candidate_ids = np.asarray(sorted(candidates), dtype=np.int32)
        facet_min = np.asarray(self.facet_min, dtype=float)[candidate_ids]
        facet_max = np.asarray(self.facet_max, dtype=float)[candidate_ids]
        intersects = _aabb_intersections(box_min_np, box_max_np, facet_min, facet_max)
        return candidate_ids[intersects]


def contact_aabb_index_from_surface(
    master_surface,
    *,
    dim: int,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
    cell_size: float | Array | None = None,
) -> ContactAABBIndex:
    """Build a uniform-grid AABB index over master-surface facets."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    master_facets = np.asarray(master_surface.conn, dtype=np.int32)
    if master_facets.ndim != 2 or master_facets.size == 0:
        raise ValueError("master surface must contain 2D facet connectivity.")
    master_coords = np.asarray(master_surface.coords, dtype=float)
    if master_coords.ndim != 2 or master_coords.shape[1] < dim:
        raise ValueError("master_surface.coords must have at least dim coordinate columns.")
    disp_nodes, _ = _surface_displacement_nodes(
        master_coords,
        master_coords,
        dim=dim,
        n_total_nodes=n_total_nodes,
        displacement=displacement,
    )
    master_points = master_coords[master_facets, :dim] + disp_nodes[master_facets]
    facet_min = np.min(master_points, axis=1)
    facet_max = np.max(master_points, axis=1)
    extents = np.maximum(facet_max - facet_min, 0.0)
    if cell_size is None:
        mean_extent = np.mean(extents, axis=0)
        domain_extent = np.maximum(np.max(facet_max, axis=0) - np.min(facet_min, axis=0), 1.0)
        cell_size_np = np.where(mean_extent > 0.0, mean_extent, domain_extent)
    else:
        cell_size_np = np.asarray(cell_size, dtype=float)
        if cell_size_np.ndim == 0:
            cell_size_np = np.full((dim,), float(cell_size_np), dtype=float)
        cell_size_np = cell_size_np.reshape(-1)
    if cell_size_np.shape != (dim,) or np.any(cell_size_np <= 0.0) or not np.all(np.isfinite(cell_size_np)):
        raise ValueError("cell_size must be positive and finite.")
    origin = np.min(facet_min, axis=0)
    lo = np.floor((facet_min - origin[None, :]) / cell_size_np[None, :]).astype(np.int32)
    hi = np.floor((facet_max - origin[None, :]) / cell_size_np[None, :]).astype(np.int32)
    cell_map: dict[tuple[int, ...], list[int]] = {}
    for facet_id, (lo_i, hi_i) in enumerate(zip(lo, hi)):
        for key in np.ndindex(*((hi_i - lo_i + 1).astype(int))):
            cell_key = tuple((lo_i + np.asarray(key, dtype=np.int32)).tolist())
            cell_map.setdefault(cell_key, []).append(int(facet_id))
    cells = tuple((key, tuple(ids)) for key, ids in sorted(cell_map.items()))
    return ContactAABBIndex(
        facet_min=jnp.asarray(facet_min),
        facet_max=jnp.asarray(facet_max),
        cell_size=jnp.asarray(cell_size_np),
        origin=jnp.asarray(origin),
        cells=cells,
        dim=dim,
    )


def contact_candidate_set_from_bounding_boxes(
    slave_surface,
    master_surface,
    *,
    dim: int,
    search_radius: float,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
) -> "ContactCandidateSet":
    """
    Build a broad-phase master-facet candidate set from surface AABBs.

    The slave surface is reduced to one bounding box over its contact nodes.
    Master facets whose bounding boxes intersect that slave box expanded by
    `search_radius` are retained. This is a simple broad-phase helper: it
    prunes obviously distant facets but does not produce per-contact candidates.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    radius = float(search_radius)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError("search_radius must be non-negative and finite.")

    slave_nodes = np.unique(np.asarray(slave_surface.conn, dtype=np.int32).reshape(-1))
    master_facets = np.asarray(master_surface.conn, dtype=np.int32)
    if master_facets.ndim != 2:
        raise ValueError("master surface connectivity must be a 2D array.")
    if slave_nodes.size == 0 or master_facets.size == 0:
        raise ValueError("slave and master surfaces must contain contact nodes/facets.")

    slave_coords = np.asarray(slave_surface.coords, dtype=float)
    master_coords = np.asarray(master_surface.coords, dtype=float)
    if slave_coords.ndim != 2 or slave_coords.shape[1] < dim:
        raise ValueError("slave_surface.coords must have at least dim coordinate columns.")
    if master_coords.ndim != 2 or master_coords.shape[1] < dim:
        raise ValueError("master_surface.coords must have at least dim coordinate columns.")
    if n_total_nodes is None:
        n_nodes = max(int(slave_coords.shape[0]), int(master_coords.shape[0]))
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim

    if displacement is None:
        disp_nodes = np.zeros((n_nodes, dim), dtype=float)
    else:
        disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
        if disp_arr.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
        disp_nodes = disp_arr.reshape(n_nodes, dim)

    slave_points = slave_coords[slave_nodes, :dim] + disp_nodes[slave_nodes]
    master_points = master_coords[master_facets, :dim] + disp_nodes[master_facets]
    slave_min = np.min(slave_points, axis=0) - radius
    slave_max = np.max(slave_points, axis=0) + radius
    master_min = np.min(master_points, axis=1)
    master_max = np.max(master_points, axis=1)
    intersects = np.all(master_max >= slave_min[None, :], axis=1) & np.all(master_min <= slave_max[None, :], axis=1)
    candidate_ids = np.nonzero(intersects)[0].astype(np.int32)
    if candidate_ids.size == 0:
        raise ValueError("no contact candidate facets found within search_radius.")
    return ContactCandidateSet(candidate_ids)


def node_surface_candidate_set_from_bounding_boxes(
    slave_surface,
    master_surface,
    *,
    dim: int,
    search_radius: float,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
) -> "ContactCandidateSet":
    """
    Build per-slave-node master-facet candidates from point-expanded AABBs.

    Each slave node gets its own candidate segment containing master facets
    whose AABBs intersect the slave point expanded by `search_radius`.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    radius = float(search_radius)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError("search_radius must be non-negative and finite.")

    slave_nodes = np.unique(np.asarray(slave_surface.conn, dtype=np.int32).reshape(-1))
    master_facets = np.asarray(master_surface.conn, dtype=np.int32)
    if master_facets.ndim != 2:
        raise ValueError("master surface connectivity must be a 2D array.")
    if slave_nodes.size == 0 or master_facets.size == 0:
        raise ValueError("slave and master surfaces must contain contact nodes/facets.")

    slave_coords = np.asarray(slave_surface.coords, dtype=float)
    master_coords = np.asarray(master_surface.coords, dtype=float)
    if slave_coords.ndim != 2 or slave_coords.shape[1] < dim:
        raise ValueError("slave_surface.coords must have at least dim coordinate columns.")
    if master_coords.ndim != 2 or master_coords.shape[1] < dim:
        raise ValueError("master_surface.coords must have at least dim coordinate columns.")
    if n_total_nodes is None:
        n_nodes = max(int(slave_coords.shape[0]), int(master_coords.shape[0]))
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim

    if displacement is None:
        disp_nodes = np.zeros((n_nodes, dim), dtype=float)
    else:
        disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
        if disp_arr.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
        disp_nodes = disp_arr.reshape(n_nodes, dim)

    slave_points = slave_coords[slave_nodes, :dim] + disp_nodes[slave_nodes]
    master_points = master_coords[master_facets, :dim] + disp_nodes[master_facets]
    master_min = np.min(master_points, axis=1)
    master_max = np.max(master_points, axis=1)
    rows = []
    for point in slave_points:
        point_min = point - radius
        point_max = point + radius
        intersects = np.all(master_max >= point_min[None, :], axis=1) & np.all(
            master_min <= point_max[None, :], axis=1
        )
        candidate_ids = np.nonzero(intersects)[0].astype(np.int32)
        if candidate_ids.size == 0:
            raise ValueError("no contact candidate facets found for a slave node within search_radius.")
        rows.append(candidate_ids)
    return contact_candidate_set_from_per_contact(rows)


def node_surface_candidate_set_from_aabb_index(
    slave_surface,
    index: ContactAABBIndex,
    *,
    dim: int,
    search_radius: float,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
) -> "ContactCandidateSet":
    """Build per-slave-node candidates by querying a master-facet AABB index."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if int(index.dim) != dim:
        raise ValueError("index dim must match dim.")
    radius = float(search_radius)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError("search_radius must be non-negative and finite.")

    slave_nodes = np.unique(np.asarray(slave_surface.conn, dtype=np.int32).reshape(-1))
    if slave_nodes.size == 0:
        raise ValueError("slave surface must contain contact nodes.")
    slave_coords = np.asarray(slave_surface.coords, dtype=float)
    if slave_coords.ndim != 2 or slave_coords.shape[1] < dim:
        raise ValueError("slave_surface.coords must have at least dim coordinate columns.")
    if n_total_nodes is None:
        n_nodes = int(slave_coords.shape[0])
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim
    if displacement is None:
        disp_nodes = np.zeros((n_nodes, dim), dtype=float)
    else:
        disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
        if disp_arr.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
        disp_nodes = disp_arr.reshape(n_nodes, dim)

    slave_points = slave_coords[slave_nodes, :dim] + disp_nodes[slave_nodes]
    rows = []
    for point in slave_points:
        candidate_ids = index.query_box(point - radius, point + radius)
        if candidate_ids.size == 0:
            raise ValueError("no contact candidate facets found for a slave node within search_radius.")
        rows.append(candidate_ids)
    return contact_candidate_set_from_per_contact(rows)


def surface_quadrature_candidate_set_from_aabb_index(
    slave_surface,
    index: ContactAABBIndex,
    *,
    dim: int,
    search_radius: float,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
    quadrature_rule: str = "centroid",
) -> "ContactCandidateSet":
    """Build per-quadrature-point candidates by querying a master-facet AABB index."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if int(index.dim) != dim:
        raise ValueError("index dim must match dim.")
    radius = float(search_radius)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError("search_radius must be non-negative and finite.")

    slave_facets = np.asarray(slave_surface.conn, dtype=np.int32)
    if slave_facets.ndim != 2 or slave_facets.size == 0:
        raise ValueError("slave surface must contain 2D facet connectivity.")
    slave_coords = np.asarray(slave_surface.coords, dtype=float)
    if slave_coords.ndim != 2 or slave_coords.shape[1] < dim:
        raise ValueError("slave_surface.coords must have at least dim coordinate columns.")
    if n_total_nodes is None:
        n_nodes = int(slave_coords.shape[0])
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim
    if displacement is None:
        disp_nodes = np.zeros((n_nodes, dim), dtype=float)
    else:
        disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
        if disp_arr.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
        disp_nodes = disp_arr.reshape(n_nodes, dim)

    local_weights, _ = _facet_quadrature_rule(slave_facets.shape[1], quadrature_rule)
    slave_facet_search = slave_coords[slave_facets, :dim] + disp_nodes[slave_facets]
    rows = []
    for facet_search in slave_facet_search:
        for weights_s in local_weights:
            point = weights_s @ facet_search
            candidate_ids = index.query_box(point - radius, point + radius)
            if candidate_ids.size == 0:
                raise ValueError("no contact candidate facets found for a quadrature point within search_radius.")
            rows.append(candidate_ids)
    return contact_candidate_set_from_per_contact(rows)


def _node_displacement_drift(displacement: Array, reference_displacement: Array, *, dim: int) -> Array:
    displacement_arr = jnp.asarray(displacement).reshape(-1)
    reference_arr = jnp.asarray(reference_displacement).reshape(-1)
    if displacement_arr.shape != reference_arr.shape:
        raise ValueError("displacement and reference_displacement must have the same shape.")
    if displacement_arr.size % int(dim) != 0:
        raise ValueError("displacement length must be divisible by dim.")
    delta = displacement_arr.reshape(-1, int(dim)) - reference_arr.reshape(-1, int(dim))
    if delta.shape[0] == 0:
        return jnp.asarray(0.0, dtype=delta.dtype)
    return jnp.max(jnp.linalg.norm(delta, axis=1))


@dataclass(frozen=True)
class ContactNeighborList:
    """
    Candidate set plus refresh metadata for broad-phase contact search.

    Build candidates with `search_radius + skin`, then reuse them until nodal
    motion from `reference_displacement` exceeds `skin / 2`.
    """

    candidate_set: "ContactCandidateSet"
    reference_displacement: Array
    dim: int
    search_radius: float
    skin: float

    def __post_init__(self):
        dim = int(self.dim)
        if dim <= 0:
            raise ValueError("dim must be positive.")
        search_radius = float(self.search_radius)
        skin = float(self.skin)
        if search_radius < 0.0 or not np.isfinite(search_radius):
            raise ValueError("search_radius must be non-negative and finite.")
        if skin < 0.0 or not np.isfinite(skin):
            raise ValueError("skin must be non-negative and finite.")
        reference = jnp.asarray(self.reference_displacement).reshape(-1)
        if reference.size % dim != 0:
            raise ValueError("reference_displacement length must be divisible by dim.")
        object.__setattr__(self, "reference_displacement", reference)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "search_radius", search_radius)
        object.__setattr__(self, "skin", skin)

    def needs_refresh(self, displacement: Array) -> Array:
        drift = _node_displacement_drift(displacement, self.reference_displacement, dim=self.dim)
        return drift > (0.5 * float(self.skin))

    def max_drift(self, displacement: Array) -> Array:
        return _node_displacement_drift(displacement, self.reference_displacement, dim=self.dim)


def node_surface_neighbor_list_from_bounding_boxes(
    slave_surface,
    master_surface,
    *,
    dim: int,
    search_radius: float,
    skin: float,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
) -> ContactNeighborList:
    """Build a node-surface broad-phase neighbor list with a Verlet-style skin."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    search_radius = float(search_radius)
    skin = float(skin)
    if search_radius < 0.0 or not np.isfinite(search_radius):
        raise ValueError("search_radius must be non-negative and finite.")
    if skin < 0.0 or not np.isfinite(skin):
        raise ValueError("skin must be non-negative and finite.")
    if n_total_nodes is None:
        n_nodes = max(int(np.asarray(slave_surface.coords).shape[0]), int(np.asarray(master_surface.coords).shape[0]))
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim
    if displacement is None:
        reference = jnp.zeros((n_dofs,))
    else:
        reference = jnp.asarray(displacement).reshape(-1)
        if reference.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")

    candidates = node_surface_candidate_set_from_bounding_boxes(
        slave_surface,
        master_surface,
        dim=dim,
        search_radius=search_radius + skin,
        n_total_nodes=n_nodes,
        displacement=reference,
    )
    return ContactNeighborList(
        candidate_set=candidates,
        reference_displacement=reference,
        dim=dim,
        search_radius=search_radius,
        skin=skin,
    )


def node_surface_neighbor_list_from_aabb_index(
    slave_surface,
    index: ContactAABBIndex,
    *,
    dim: int,
    search_radius: float,
    skin: float,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
) -> ContactNeighborList:
    """Build a node-surface neighbor list by querying an existing AABB index."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    search_radius = float(search_radius)
    skin = float(skin)
    if search_radius < 0.0 or not np.isfinite(search_radius):
        raise ValueError("search_radius must be non-negative and finite.")
    if skin < 0.0 or not np.isfinite(skin):
        raise ValueError("skin must be non-negative and finite.")
    if n_total_nodes is None:
        n_nodes = int(np.asarray(slave_surface.coords).shape[0])
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim
    if displacement is None:
        reference = jnp.zeros((n_dofs,))
    else:
        reference = jnp.asarray(displacement).reshape(-1)
        if reference.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
    candidates = node_surface_candidate_set_from_aabb_index(
        slave_surface,
        index,
        dim=dim,
        search_radius=search_radius + skin,
        n_total_nodes=n_nodes,
        displacement=reference,
    )
    return ContactNeighborList(
        candidate_set=candidates,
        reference_displacement=reference,
        dim=dim,
        search_radius=search_radius,
        skin=skin,
    )


def surface_quadrature_neighbor_list_from_aabb_index(
    slave_surface,
    index: ContactAABBIndex,
    *,
    dim: int,
    search_radius: float,
    skin: float,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
    quadrature_rule: str = "centroid",
) -> ContactNeighborList:
    """Build a surface-quadrature neighbor list by querying an existing AABB index."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    search_radius = float(search_radius)
    skin = float(skin)
    if search_radius < 0.0 or not np.isfinite(search_radius):
        raise ValueError("search_radius must be non-negative and finite.")
    if skin < 0.0 or not np.isfinite(skin):
        raise ValueError("skin must be non-negative and finite.")
    if n_total_nodes is None:
        n_nodes = int(np.asarray(slave_surface.coords).shape[0])
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim
    if displacement is None:
        reference = jnp.zeros((n_dofs,))
    else:
        reference = jnp.asarray(displacement).reshape(-1)
        if reference.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
    candidates = surface_quadrature_candidate_set_from_aabb_index(
        slave_surface,
        index,
        dim=dim,
        search_radius=search_radius + skin,
        n_total_nodes=n_nodes,
        displacement=reference,
        quadrature_rule=quadrature_rule,
    )
    return ContactNeighborList(
        candidate_set=candidates,
        reference_displacement=reference,
        dim=dim,
        search_radius=search_radius,
        skin=skin,
    )


def _validate_contact_search_manager_scalars(manager) -> None:
    dim = int(manager.dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    n_total_nodes = int(manager.n_total_nodes)
    if n_total_nodes <= 0:
        raise ValueError("n_total_nodes must be positive.")
    search_radius = float(manager.search_radius)
    skin = float(manager.skin)
    if search_radius < 0.0 or not np.isfinite(search_radius):
        raise ValueError("search_radius must be non-negative and finite.")
    if skin < 0.0 or not np.isfinite(skin):
        raise ValueError("skin must be non-negative and finite.")
    object.__setattr__(manager, "dim", dim)
    object.__setattr__(manager, "n_total_nodes", n_total_nodes)
    object.__setattr__(manager, "search_radius", search_radius)
    object.__setattr__(manager, "skin", skin)
    object.__setattr__(manager, "penalty", float(manager.penalty))
    object.__setattr__(manager, "smoothing", float(manager.smoothing))


def _contact_search_manager_master_index(manager, displacement: Array) -> ContactAABBIndex:
    return contact_aabb_index_from_surface(
        manager.master_surface,
        dim=manager.dim,
        n_total_nodes=manager.n_total_nodes,
        displacement=displacement,
        cell_size=manager.cell_size,
    )


def _prepared_contact_search_manager(manager, displacement: Array):
    use_manager = manager
    if use_manager.neighbor_list is None or bool(use_manager.neighbor_list.needs_refresh(displacement)):
        use_manager = use_manager._refreshed(displacement)
    candidate_facet_ids = None if use_manager.search_cache is not None else use_manager.neighbor_list.candidate_set
    return use_manager, candidate_facet_ids


@dataclass(frozen=True)
class NodeSurfaceContactSearchManager:
    """
    Search-state manager for node-surface penalty contact.

    `build_contact(u)` returns a contact object and the next manager state. The
    manager reuses an exact search cache when present, refreshes its neighbor
    list when nodal drift exceeds the skin criterion, and rebuilds the AABB index
    only on refresh.
    """

    slave_surface: object
    master_surface: object
    dim: int
    n_total_nodes: int
    search_radius: float
    skin: float
    penalty: float
    smoothing: float = 0.0
    normal: Array | None = None
    cell_size: float | Array | None = None
    index: ContactAABBIndex | None = None
    neighbor_list: ContactNeighborList | None = None
    search_cache: ContactSearchCache | None = None

    def __post_init__(self):
        _validate_contact_search_manager_scalars(self)

    def with_search_cache(self, search_cache: ContactSearchCache | None) -> "NodeSurfaceContactSearchManager":
        return replace(self, search_cache=search_cache)

    def _refreshed(self, displacement: Array) -> "NodeSurfaceContactSearchManager":
        index = _contact_search_manager_master_index(self, displacement)
        neighbor_list = node_surface_neighbor_list_from_aabb_index(
            self.slave_surface,
            index,
            dim=self.dim,
            search_radius=self.search_radius,
            skin=self.skin,
            n_total_nodes=self.n_total_nodes,
            displacement=displacement,
        )
        return replace(self, index=index, neighbor_list=neighbor_list, search_cache=None)

    def build_contact(self, displacement: Array) -> tuple["NodeSurfacePenaltyContact", "NodeSurfaceContactSearchManager"]:
        use_manager, candidate_facet_ids = _prepared_contact_search_manager(self, displacement)
        kin = node_surface_contact_kinematics_from_surfaces(
            use_manager.slave_surface,
            use_manager.master_surface,
            dim=use_manager.dim,
            normal=use_manager.normal,
            n_total_nodes=use_manager.n_total_nodes,
            displacement=displacement,
            candidate_facet_ids=candidate_facet_ids,
            search_cache=use_manager.search_cache,
        )
        contact = NodeSurfacePenaltyContact(kin, penalty=use_manager.penalty, smoothing=use_manager.smoothing)
        return contact, use_manager.with_search_cache(kin.search_cache())


def make_node_surface_contact_search_manager(
    slave_surface,
    master_surface,
    *,
    dim: int,
    n_total_nodes: int,
    search_radius: float,
    skin: float,
    penalty: float,
    smoothing: float = 0.0,
    normal: Array | None = None,
    cell_size: float | Array | None = None,
) -> NodeSurfaceContactSearchManager:
    """Create a node-surface contact search manager."""
    return NodeSurfaceContactSearchManager(
        slave_surface=slave_surface,
        master_surface=master_surface,
        dim=dim,
        n_total_nodes=n_total_nodes,
        search_radius=search_radius,
        skin=skin,
        penalty=penalty,
        smoothing=smoothing,
        normal=normal,
        cell_size=cell_size,
    )


@dataclass(frozen=True)
class SurfaceQuadratureContactSearchManager:
    """Search-state manager for surface-quadrature penalty contact."""

    slave_surface: object
    master_surface: object
    dim: int
    n_total_nodes: int
    search_radius: float
    skin: float
    penalty: float
    smoothing: float = 0.0
    normal: Array | None = None
    quadrature_rule: str = "centroid"
    cell_size: float | Array | None = None
    index: ContactAABBIndex | None = None
    neighbor_list: ContactNeighborList | None = None
    search_cache: ContactSearchCache | None = None

    def __post_init__(self):
        _validate_contact_search_manager_scalars(self)

    def with_search_cache(self, search_cache: ContactSearchCache | None) -> "SurfaceQuadratureContactSearchManager":
        return replace(self, search_cache=search_cache)

    def _refreshed(self, displacement: Array) -> "SurfaceQuadratureContactSearchManager":
        index = _contact_search_manager_master_index(self, displacement)
        neighbor_list = surface_quadrature_neighbor_list_from_aabb_index(
            self.slave_surface,
            index,
            dim=self.dim,
            search_radius=self.search_radius,
            skin=self.skin,
            n_total_nodes=self.n_total_nodes,
            displacement=displacement,
            quadrature_rule=self.quadrature_rule,
        )
        return replace(self, index=index, neighbor_list=neighbor_list, search_cache=None)

    def build_contact(
        self,
        displacement: Array,
    ) -> tuple["SurfaceQuadraturePenaltyContact", "SurfaceQuadratureContactSearchManager"]:
        use_manager, candidate_facet_ids = _prepared_contact_search_manager(self, displacement)
        kin = surface_quadrature_contact_kinematics_from_surfaces(
            use_manager.slave_surface,
            use_manager.master_surface,
            dim=use_manager.dim,
            normal=use_manager.normal,
            n_total_nodes=use_manager.n_total_nodes,
            displacement=displacement,
            quadrature_rule=use_manager.quadrature_rule,
            candidate_facet_ids=candidate_facet_ids,
            search_cache=use_manager.search_cache,
        )
        contact = SurfaceQuadraturePenaltyContact(
            kin,
            penalty=use_manager.penalty,
            smoothing=use_manager.smoothing,
        )
        return contact, use_manager.with_search_cache(kin.search_cache())


def make_surface_quadrature_contact_search_manager(
    slave_surface,
    master_surface,
    *,
    dim: int,
    n_total_nodes: int,
    search_radius: float,
    skin: float,
    penalty: float,
    smoothing: float = 0.0,
    normal: Array | None = None,
    quadrature_rule: str = "centroid",
    cell_size: float | Array | None = None,
) -> SurfaceQuadratureContactSearchManager:
    """Create a surface-quadrature contact search manager."""
    return SurfaceQuadratureContactSearchManager(
        slave_surface=slave_surface,
        master_surface=master_surface,
        dim=dim,
        n_total_nodes=n_total_nodes,
        search_radius=search_radius,
        skin=skin,
        penalty=penalty,
        smoothing=smoothing,
        normal=normal,
        quadrature_rule=quadrature_rule,
        cell_size=cell_size,
    )


def node_surface_contact_kinematics_from_surfaces(
    slave_surface,
    master_surface,
    *,
    dim: int,
    normal: Array | None = None,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
    candidate_facet_ids: "ContactCandidateSet | Array | None" = None,
    search_cache: "ContactSearchCache | None" = None,
) -> "NodeSurfaceContactKinematics":
    """
    Build frozen node-to-surface contact kinematics.

    Each slave node is paired to the master facet with the closest projected
    point in reference geometry, or in the deformed geometry when `displacement`
    is provided.
    Master displacement is interpolated with facet shape weights: linear edge
    weights, triangle barycentric weights, or quad bilinear weights. This is a
    compact closest-point prototype, not a full mortar contact.

    When `normal` is omitted, normals are computed from the selected master
    facets and frozen into the returned kinematics. With `displacement`, those
    normals are computed from the deformed master facets.

    Passing `search_cache` freezes the master facet chosen for each slave node.
    The returned interpolation weights, reference gaps, and automatic normals
    are still rebuilt from the provided geometry/displacement.

    Passing `candidate_facet_ids` limits closest-facet search to a subset of
    master facet rows. This is useful as a hand-built broad-phase pruning hook.
    If both `candidate_facet_ids` and `search_cache` are provided, the exact
    cached pairing is used.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    slave_nodes = np.unique(np.asarray(slave_surface.conn, dtype=np.int32).reshape(-1))
    master_facets = np.asarray(master_surface.conn, dtype=np.int32)
    if slave_nodes.size == 0 or master_facets.size == 0:
        raise ValueError("slave and master surfaces must contain contact nodes/facets.")

    slave_coords = np.asarray(slave_surface.coords, dtype=float)
    master_coords = np.asarray(master_surface.coords, dtype=float)
    if slave_coords.ndim != 2 or slave_coords.shape[1] < dim:
        raise ValueError("slave_surface.coords must have at least dim coordinate columns.")
    if master_coords.ndim != 2 or master_coords.shape[1] < dim:
        raise ValueError("master_surface.coords must have at least dim coordinate columns.")
    if n_total_nodes is None:
        n_nodes = max(int(slave_coords.shape[0]), int(master_coords.shape[0]))
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim

    if displacement is None:
        disp_nodes = np.zeros((n_nodes, dim), dtype=float)
    else:
        disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
        if disp_arr.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
        disp_nodes = disp_arr.reshape(n_nodes, dim)

    x_slave = slave_coords[slave_nodes, :dim]
    facet_coords = master_coords[master_facets, :dim]
    x_slave_search = x_slave + disp_nodes[slave_nodes]
    facet_coords_search = facet_coords + disp_nodes[master_facets]
    n_contact = slave_nodes.size
    if search_cache is not None:
        facet_ids = _validate_contact_search_cache(
            search_cache,
            n_contact=n_contact,
            n_master_facets=master_facets.shape[0],
        )
        projected_weights = []
        for point, facet_id in zip(x_slave_search, facet_ids):
            _, weights_i = _facet_projected_point_and_weights(point, facet_coords_search[facet_id])
            projected_weights.append(weights_i)
        selected_weights = np.stack(projected_weights, axis=0)
    else:
        candidate_ids_by_contact = _validate_contact_candidate_set(
            candidate_facet_ids,
            n_master_facets=master_facets.shape[0],
            n_contact=n_contact,
        )
        facet_ids_rows = []
        selected_weights_rows = []
        for point, candidate_ids in zip(x_slave_search, candidate_ids_by_contact):
            facet_points = []
            facet_weights = []
            for pts in facet_coords_search[candidate_ids]:
                projected, weights_i = _facet_projected_point_and_weights(point, pts)
                facet_points.append(projected)
                facet_weights.append(weights_i)
            points_i = np.stack(facet_points, axis=0)
            weights_i = np.stack(facet_weights, axis=0)
            dist2 = np.sum((point[None, :] - points_i) ** 2, axis=1)
            local_facet_id = int(np.argmin(dist2))
            facet_ids_rows.append(int(candidate_ids[local_facet_id]))
            selected_weights_rows.append(weights_i[local_facet_id])
        facet_ids = np.asarray(facet_ids_rows, dtype=np.int32)
        selected_weights = np.stack(selected_weights_rows, axis=0)
    paired_facets = master_facets[facet_ids]
    paired_facet_coords = facet_coords[facet_ids]
    paired_facet_coords_search = facet_coords_search[facet_ids]

    n_master_nodes = paired_facets.shape[1]
    if normal is None:
        normal_np = np.stack([_facet_normal(pts, dim) for pts in paired_facet_coords_search], axis=0)
    else:
        normal_arr = jnp.asarray(normal)
        if normal_arr.shape != (dim,):
            raise ValueError("normal must have shape (dim,).")
        normal_np = np.broadcast_to(np.asarray(normal_arr), (n_contact, dim))
    normals = jnp.asarray(normal_np)
    slave_dofs = vector_dofs_from_nodes(slave_nodes, dim).reshape(n_contact, dim)
    master_dofs = (
        jnp.asarray(paired_facets, dtype=jnp.int32)[:, :, None] * dim
        + jnp.arange(dim, dtype=jnp.int32)[None, None, :]
    )
    weights = jnp.asarray(selected_weights)
    master_ref_points = np.einsum("im,imd->id", np.asarray(weights), paired_facet_coords)
    gaps0 = jnp.einsum("id,id->i", jnp.asarray(x_slave - master_ref_points), normals)
    return NodeSurfaceContactKinematics(
        slave_dofs=slave_dofs,
        master_dofs=master_dofs,
        master_weights=weights,
        normals=normals,
        gaps0=gaps0,
        n_dofs=n_dofs,
        master_facet_ids=jnp.asarray(facet_ids, dtype=jnp.int32),
        slave_nodes=jnp.asarray(slave_nodes, dtype=jnp.int32),
    )


def _facet_quadrature_rule(n_nodes: int, rule: str) -> tuple[np.ndarray, np.ndarray]:
    if rule == "centroid":
        return (
            np.full((1, n_nodes), 1.0 / float(n_nodes), dtype=float),
            np.ones((1,), dtype=float),
        )
    if rule == "vertices":
        return (
            np.eye(n_nodes, dtype=float),
            np.full((n_nodes,), 1.0 / float(n_nodes), dtype=float),
        )
    raise ValueError("quadrature_rule must be 'centroid' or 'vertices'.")


def surface_quadrature_contact_kinematics_from_surfaces(
    slave_surface,
    master_surface,
    *,
    dim: int,
    normal: Array | None = None,
    n_total_nodes: int | None = None,
    displacement: Array | None = None,
    quadrature_rule: str = "centroid",
    candidate_facet_ids: "ContactCandidateSet | Array | None" = None,
    search_cache: "ContactSearchCache | None" = None,
) -> "SurfaceQuadratureContactKinematics":
    """
    Build frozen slave-surface quadrature to master-surface contact kinematics.

    This prototype evaluates contact at either slave-facet centroids or slave
    facet vertices, projects each quadrature point to the closest master facet,
    and freezes interpolation weights/normals for AD residual evaluation. It is
    an integration-point contact prototype, not a mortar formulation.

    Passing `search_cache` freezes the master facet chosen for each quadrature
    point while still rebuilding interpolation weights and automatic normals.
    Passing `candidate_facet_ids` limits closest-facet search to a subset of
    master facet rows when no exact `search_cache` is provided.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    slave_facets = np.asarray(slave_surface.conn, dtype=np.int32)
    master_facets = np.asarray(master_surface.conn, dtype=np.int32)
    if slave_facets.ndim != 2 or master_facets.ndim != 2:
        raise ValueError("surface connectivity must be a 2D array.")
    if slave_facets.size == 0 or master_facets.size == 0:
        raise ValueError("slave and master surfaces must contain facets.")

    slave_coords = np.asarray(slave_surface.coords, dtype=float)
    master_coords = np.asarray(master_surface.coords, dtype=float)
    if slave_coords.ndim != 2 or slave_coords.shape[1] < dim:
        raise ValueError("slave_surface.coords must have at least dim coordinate columns.")
    if master_coords.ndim != 2 or master_coords.shape[1] < dim:
        raise ValueError("master_surface.coords must have at least dim coordinate columns.")
    if n_total_nodes is None:
        n_nodes = max(int(slave_coords.shape[0]), int(master_coords.shape[0]))
    else:
        n_nodes = int(n_total_nodes)
    n_dofs = n_nodes * dim

    if displacement is None:
        disp_nodes = np.zeros((n_nodes, dim), dtype=float)
    else:
        disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
        if disp_arr.size != n_dofs:
            raise ValueError("displacement length must match n_total_nodes * dim.")
        disp_nodes = disp_arr.reshape(n_nodes, dim)

    slave_facet_coords = slave_coords[slave_facets, :dim]
    master_facet_coords = master_coords[master_facets, :dim]
    slave_facet_search = slave_facet_coords + disp_nodes[slave_facets]
    master_facet_search = master_facet_coords + disp_nodes[master_facets]
    local_weights, local_q_weights = _facet_quadrature_rule(slave_facets.shape[1], quadrature_rule)

    normal_arr = None if normal is None else jnp.asarray(normal)
    if normal_arr is not None and normal_arr.shape != (dim,):
        raise ValueError("normal must have shape (dim,).")

    slave_dofs_rows = []
    slave_weight_rows = []
    master_dofs_rows = []
    master_weight_rows = []
    normal_rows = []
    gaps0_rows = []
    quadrature_weight_rows = []
    master_facet_id_rows = []
    slave_facet_id_rows = []

    n_contact = slave_facets.shape[0] * local_weights.shape[0]
    cached_facet_ids = None
    if search_cache is not None:
        cached_facet_ids = _validate_contact_search_cache(
            search_cache,
            n_contact=n_contact,
            n_master_facets=master_facets.shape[0],
        )
    candidate_ids = None
    if cached_facet_ids is None:
        candidate_ids_by_contact = _validate_contact_candidate_set(
            candidate_facet_ids,
            n_master_facets=master_facets.shape[0],
            n_contact=n_contact,
        )
    contact_id = 0

    for slave_facet_id, (slave_facet, facet_ref, facet_search) in enumerate(
        zip(slave_facets, slave_facet_coords, slave_facet_search)
    ):
        slave_dofs_facet = (
            jnp.asarray(slave_facet, dtype=jnp.int32)[:, None] * dim
            + jnp.arange(dim, dtype=jnp.int32)[None, :]
        )
        for q_weight, weights_s in zip(local_q_weights, local_weights):
            point_search = weights_s @ facet_search
            if cached_facet_ids is None:
                candidate_ids = candidate_ids_by_contact[contact_id]
                projected_points = []
                projected_weights = []
                for pts in master_facet_search[candidate_ids]:
                    projected, weights_m = _facet_projected_point_and_weights(point_search, pts)
                    projected_points.append(projected)
                    projected_weights.append(weights_m)
                projected_points = np.stack(projected_points, axis=0)
                projected_weights = np.stack(projected_weights, axis=0)
                local_facet_id = int(np.argmin(np.sum((point_search[None, :] - projected_points) ** 2, axis=1)))
                facet_id = int(candidate_ids[local_facet_id])
                weights_m = projected_weights[local_facet_id]
            else:
                facet_id = int(cached_facet_ids[contact_id])
                _, weights_m = _facet_projected_point_and_weights(point_search, master_facet_search[facet_id])
            master_facet = master_facets[facet_id]
            master_ref = master_facet_coords[facet_id]
            master_search = master_facet_search[facet_id]
            slave_ref_point = weights_s @ facet_ref
            master_ref_point = weights_m @ master_ref
            if normal_arr is None:
                normal_np = _facet_normal(master_search, dim)
            else:
                normal_np = np.asarray(normal_arr)

            slave_dofs_rows.append(np.asarray(slave_dofs_facet))
            slave_weight_rows.append(weights_s)
            master_dofs_rows.append(
                np.asarray(master_facet, dtype=np.int32)[:, None] * dim + np.arange(dim, dtype=np.int32)[None, :]
            )
            master_weight_rows.append(weights_m)
            normal_rows.append(normal_np)
            gaps0_rows.append(float(np.dot(slave_ref_point - master_ref_point, normal_np)))
            quadrature_weight_rows.append(float(q_weight))
            master_facet_id_rows.append(facet_id)
            slave_facet_id_rows.append(slave_facet_id)
            contact_id += 1

    return SurfaceQuadratureContactKinematics(
        slave_dofs=jnp.asarray(np.stack(slave_dofs_rows, axis=0), dtype=jnp.int32),
        slave_weights=jnp.asarray(np.stack(slave_weight_rows, axis=0)),
        master_dofs=jnp.asarray(np.stack(master_dofs_rows, axis=0), dtype=jnp.int32),
        master_weights=jnp.asarray(np.stack(master_weight_rows, axis=0)),
        normals=jnp.asarray(np.stack(normal_rows, axis=0)),
        gaps0=jnp.asarray(np.asarray(gaps0_rows)),
        quadrature_weights=jnp.asarray(np.asarray(quadrature_weight_rows)),
        n_dofs=n_dofs,
        master_facet_ids=jnp.asarray(np.asarray(master_facet_id_rows), dtype=jnp.int32),
        slave_facet_ids=jnp.asarray(np.asarray(slave_facet_id_rows), dtype=jnp.int32),
    )


# -----------------------------------------------------------------------------
# Residual composition and active-update comparison helpers


def compose_residuals(*residual_fns: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Return a residual function that sums multiple residual contributions."""
    if not residual_fns:
        raise ValueError("at least one residual function is required.")

    def _residual(u: Array) -> Array:
        total = residual_fns[0](u)
        for fn in residual_fns[1:]:
            total = total + fn(u)
        return total

    return _residual


def _contact_array_changed(a, b, *, tol: float = 0.0) -> Array:
    if a is None and b is None:
        return jnp.asarray(False)
    if a is None or b is None:
        return jnp.asarray(True)
    arr_a = jnp.asarray(a)
    arr_b = jnp.asarray(b)
    if arr_a.shape != arr_b.shape:
        return jnp.asarray(True)
    if arr_a.dtype == bool or arr_b.dtype == bool:
        return jnp.any(arr_a.astype(bool) != arr_b.astype(bool))
    if tol > 0.0:
        return jnp.any(jnp.abs(arr_a - arr_b) > float(tol))
    return jnp.any(arr_a != arr_b)


def _contact_kinematics_changed(old_kinematics, new_kinematics, *, tol: float) -> Array:
    changed = jnp.asarray(False)
    for name in (
        "dofs",
        "slave_dofs",
        "slave_weights",
        "master_dofs",
        "master_weights",
        "normals",
        "gaps0",
        "quadrature_weights",
        "master_facet_ids",
        "slave_nodes",
        "slave_facet_ids",
    ):
        changed = jnp.logical_or(
            changed,
            _contact_array_changed(
                getattr(old_kinematics, name, None),
                getattr(new_kinematics, name, None),
                tol=tol,
            ),
        )
    return changed


# -----------------------------------------------------------------------------
# Kinematics dataclasses


@dataclass(frozen=True)
class ContactCandidateSet:
    """Master facet rows allowed during closest-facet contact search."""

    master_facet_ids: Array
    contact_offsets: Array | None = None

    def __post_init__(self):
        facet_ids = np.asarray(self.master_facet_ids, dtype=np.int32).reshape(-1)
        if facet_ids.size == 0:
            raise ValueError("master_facet_ids must contain at least one facet id.")
        if facet_ids.min() < 0:
            raise ValueError("master_facet_ids must not contain negative ids.")
        if self.contact_offsets is None:
            facet_ids = np.unique(facet_ids)
            object.__setattr__(self, "master_facet_ids", jnp.asarray(facet_ids, dtype=jnp.int32))
            return

        offsets = np.asarray(self.contact_offsets, dtype=np.int32).reshape(-1)
        if offsets.size < 2:
            raise ValueError("contact_offsets must have at least two entries.")
        if offsets[0] != 0 or offsets[-1] != facet_ids.size:
            raise ValueError("contact_offsets must start at 0 and end at len(master_facet_ids).")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("each per-contact candidate segment must be non-empty.")
        object.__setattr__(self, "master_facet_ids", jnp.asarray(facet_ids, dtype=jnp.int32))
        object.__setattr__(self, "contact_offsets", jnp.asarray(offsets, dtype=jnp.int32))


def contact_candidate_set_from_per_contact(per_contact_facet_ids) -> ContactCandidateSet:
    """Build a variable-length per-contact candidate set."""
    rows = [np.asarray(row, dtype=np.int32).reshape(-1) for row in per_contact_facet_ids]
    if not rows:
        raise ValueError("per_contact_facet_ids must contain at least one contact row.")
    offsets = [0]
    flat_rows = []
    for row in rows:
        if row.size == 0:
            raise ValueError("each per-contact candidate row must be non-empty.")
        flat_rows.append(row)
        offsets.append(offsets[-1] + int(row.size))
    return ContactCandidateSet(
        master_facet_ids=np.concatenate(flat_rows, axis=0),
        contact_offsets=np.asarray(offsets, dtype=np.int32),
    )


@dataclass(frozen=True)
class ContactSearchCache:
    """
    Frozen master-facet pairing for contact kinematics rebuilds.

    The cache stores master facet row ids, not global node ids. Passing this to
    a kinematics builder avoids a global closest-facet search while still
    recomputing interpolation weights, gaps, and automatic normals from the
    current displaced geometry.
    """

    master_facet_ids: Array
    slave_nodes: Array | None = None
    slave_facet_ids: Array | None = None

    def __post_init__(self):
        master_ids = np.asarray(self.master_facet_ids, dtype=np.int32).reshape(-1)
        if master_ids.size and master_ids.min() < 0:
            raise ValueError("master_facet_ids must not contain negative ids.")
        object.__setattr__(self, "master_facet_ids", jnp.asarray(master_ids, dtype=jnp.int32))
        if self.slave_nodes is not None:
            slave_nodes = np.asarray(self.slave_nodes, dtype=np.int32).reshape(-1)
            if slave_nodes.size and slave_nodes.min() < 0:
                raise ValueError("slave_nodes must not contain negative ids.")
            object.__setattr__(self, "slave_nodes", jnp.asarray(slave_nodes, dtype=jnp.int32))
        if self.slave_facet_ids is not None:
            slave_facet_ids = np.asarray(self.slave_facet_ids, dtype=np.int32).reshape(-1)
            if slave_facet_ids.size and slave_facet_ids.min() < 0:
                raise ValueError("slave_facet_ids must not contain negative ids.")
            object.__setattr__(self, "slave_facet_ids", jnp.asarray(slave_facet_ids, dtype=jnp.int32))


def _validate_contact_search_cache(
    search_cache: ContactSearchCache,
    *,
    n_contact: int,
    n_master_facets: int,
) -> np.ndarray:
    facet_ids = np.asarray(search_cache.master_facet_ids, dtype=np.int32).reshape(-1)
    if facet_ids.shape != (int(n_contact),):
        raise ValueError("search_cache.master_facet_ids must have shape (n_contact,).")
    if facet_ids.size and facet_ids.max() >= int(n_master_facets):
        raise ValueError("search_cache.master_facet_ids contains an invalid master facet id.")
    return facet_ids


def _validate_contact_candidate_set(
    candidate_facet_ids: ContactCandidateSet | Array | None,
    *,
    n_master_facets: int,
    n_contact: int,
) -> tuple[np.ndarray, ...]:
    if candidate_facet_ids is None:
        global_ids = np.arange(int(n_master_facets), dtype=np.int32)
        return tuple(global_ids for _ in range(int(n_contact)))
    if isinstance(candidate_facet_ids, ContactCandidateSet):
        facet_ids = np.asarray(candidate_facet_ids.master_facet_ids, dtype=np.int32).reshape(-1)
        offsets = None
        if candidate_facet_ids.contact_offsets is not None:
            offsets = np.asarray(candidate_facet_ids.contact_offsets, dtype=np.int32).reshape(-1)
    else:
        facet_ids = np.asarray(candidate_facet_ids, dtype=np.int32).reshape(-1)
        offsets = None
    if facet_ids.size == 0:
        raise ValueError("candidate_facet_ids must contain at least one facet id.")
    if facet_ids.min() < 0:
        raise ValueError("candidate_facet_ids must not contain negative ids.")
    if facet_ids.max() >= int(n_master_facets):
        raise ValueError("candidate_facet_ids contains an invalid master facet id.")
    if offsets is None:
        global_ids = np.unique(facet_ids).astype(np.int32, copy=False)
        return tuple(global_ids for _ in range(int(n_contact)))
    if offsets.shape != (int(n_contact) + 1,):
        raise ValueError("candidate_facet_ids contact_offsets must have shape (n_contact + 1,).")
    rows = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        rows.append(np.unique(facet_ids[int(start) : int(stop)]).astype(np.int32, copy=False))
    return tuple(rows)


def contact_search_cache_from_kinematics(kinematics) -> ContactSearchCache:
    """Extract a reusable master-facet pairing cache from contact kinematics."""
    master_facet_ids = getattr(kinematics, "master_facet_ids", None)
    if master_facet_ids is None:
        raise ValueError("kinematics does not carry master_facet_ids.")
    return ContactSearchCache(
        master_facet_ids=master_facet_ids,
        slave_nodes=getattr(kinematics, "slave_nodes", None),
        slave_facet_ids=getattr(kinematics, "slave_facet_ids", None),
    )


@dataclass(frozen=True)
class ContactKinematics:
    """Contact point DOF map, normals, and reference gaps."""

    dofs: Array
    normals: Array
    gaps0: Array
    n_dofs: int

    def __post_init__(self):
        dofs_np = np.asarray(self.dofs, dtype=np.int32)
        if dofs_np.ndim != 2:
            raise ValueError("dofs must have shape (n_contact, dim).")
        if dofs_np.size and (dofs_np.min() < 0 or dofs_np.max() >= int(self.n_dofs)):
            raise ValueError("dofs contains an index outside the full DOF range.")
        normals = jnp.asarray(self.normals)
        gaps0 = jnp.asarray(self.gaps0)
        if normals.shape != dofs_np.shape:
            raise ValueError("normals must have the same shape as dofs.")
        if gaps0.shape != (dofs_np.shape[0],):
            raise ValueError("gaps0 must have shape (n_contact,).")

        object.__setattr__(self, "dofs", jnp.asarray(dofs_np, dtype=jnp.int32))
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "gaps0", gaps0)
        object.__setattr__(self, "n_dofs", int(self.n_dofs))

    def displacements(self, u: Array) -> Array:
        return jnp.asarray(u)[self.dofs]

    def gaps(self, u: Array) -> Array:
        return self.gaps0 + jnp.einsum("id,id->i", self.displacements(u), self.normals)


@dataclass(frozen=True)
class PairedContactKinematics:
    """Slave/master node-pair contact kinematics with fixed normals."""

    slave_dofs: Array
    master_dofs: Array
    normals: Array
    gaps0: Array
    n_dofs: int

    def __post_init__(self):
        slave_np = np.asarray(self.slave_dofs, dtype=np.int32)
        master_np = np.asarray(self.master_dofs, dtype=np.int32)
        if slave_np.ndim != 2:
            raise ValueError("slave_dofs must have shape (n_contact, dim).")
        if master_np.shape != slave_np.shape:
            raise ValueError("master_dofs must have the same shape as slave_dofs.")
        if slave_np.size and (slave_np.min() < 0 or slave_np.max() >= int(self.n_dofs)):
            raise ValueError("slave_dofs contains an index outside the full DOF range.")
        if master_np.size and (master_np.min() < 0 or master_np.max() >= int(self.n_dofs)):
            raise ValueError("master_dofs contains an index outside the full DOF range.")
        normals = jnp.asarray(self.normals)
        gaps0 = jnp.asarray(self.gaps0)
        if normals.shape != slave_np.shape:
            raise ValueError("normals must have the same shape as slave_dofs.")
        if gaps0.shape != (slave_np.shape[0],):
            raise ValueError("gaps0 must have shape (n_contact,).")

        object.__setattr__(self, "slave_dofs", jnp.asarray(slave_np, dtype=jnp.int32))
        object.__setattr__(self, "master_dofs", jnp.asarray(master_np, dtype=jnp.int32))
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "gaps0", gaps0)
        object.__setattr__(self, "n_dofs", int(self.n_dofs))

    def slave_displacements(self, u: Array) -> Array:
        return jnp.asarray(u)[self.slave_dofs]

    def master_displacements(self, u: Array) -> Array:
        return jnp.asarray(u)[self.master_dofs]

    def relative_displacements(self, u: Array) -> Array:
        return self.slave_displacements(u) - self.master_displacements(u)

    def gaps(self, u: Array) -> Array:
        return self.gaps0 + jnp.einsum("id,id->i", self.relative_displacements(u), self.normals)


@dataclass(frozen=True)
class NodeSurfaceContactKinematics:
    """Slave-node to master-facet contact kinematics with fixed normals."""

    slave_dofs: Array
    master_dofs: Array
    master_weights: Array
    normals: Array
    gaps0: Array
    n_dofs: int
    master_facet_ids: Array | None = None
    slave_nodes: Array | None = None

    def __post_init__(self):
        slave_np = np.asarray(self.slave_dofs, dtype=np.int32)
        master_np = np.asarray(self.master_dofs, dtype=np.int32)
        weights = jnp.asarray(self.master_weights)
        if slave_np.ndim != 2:
            raise ValueError("slave_dofs must have shape (n_contact, dim).")
        if master_np.ndim != 3 or master_np.shape[0] != slave_np.shape[0] or master_np.shape[2] != slave_np.shape[1]:
            raise ValueError("master_dofs must have shape (n_contact, n_master_nodes, dim).")
        if weights.shape != master_np.shape[:2]:
            raise ValueError("master_weights must have shape (n_contact, n_master_nodes).")
        if slave_np.size and (slave_np.min() < 0 or slave_np.max() >= int(self.n_dofs)):
            raise ValueError("slave_dofs contains an index outside the full DOF range.")
        if master_np.size and (master_np.min() < 0 or master_np.max() >= int(self.n_dofs)):
            raise ValueError("master_dofs contains an index outside the full DOF range.")
        normals = jnp.asarray(self.normals)
        gaps0 = jnp.asarray(self.gaps0)
        if normals.shape != slave_np.shape:
            raise ValueError("normals must have the same shape as slave_dofs.")
        if gaps0.shape != (slave_np.shape[0],):
            raise ValueError("gaps0 must have shape (n_contact,).")
        master_facet_ids = None
        if self.master_facet_ids is not None:
            master_facet_ids = np.asarray(self.master_facet_ids, dtype=np.int32).reshape(-1)
            if master_facet_ids.shape != (slave_np.shape[0],):
                raise ValueError("master_facet_ids must have shape (n_contact,).")
            if master_facet_ids.size and master_facet_ids.min() < 0:
                raise ValueError("master_facet_ids must not contain negative ids.")
        slave_nodes = None
        if self.slave_nodes is not None:
            slave_nodes = np.asarray(self.slave_nodes, dtype=np.int32).reshape(-1)
            if slave_nodes.shape != (slave_np.shape[0],):
                raise ValueError("slave_nodes must have shape (n_contact,).")
            if slave_nodes.size and slave_nodes.min() < 0:
                raise ValueError("slave_nodes must not contain negative ids.")

        object.__setattr__(self, "slave_dofs", jnp.asarray(slave_np, dtype=jnp.int32))
        object.__setattr__(self, "master_dofs", jnp.asarray(master_np, dtype=jnp.int32))
        object.__setattr__(self, "master_weights", weights)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "gaps0", gaps0)
        object.__setattr__(self, "n_dofs", int(self.n_dofs))
        if master_facet_ids is not None:
            object.__setattr__(self, "master_facet_ids", jnp.asarray(master_facet_ids, dtype=jnp.int32))
        if slave_nodes is not None:
            object.__setattr__(self, "slave_nodes", jnp.asarray(slave_nodes, dtype=jnp.int32))

    def search_cache(self) -> ContactSearchCache:
        return contact_search_cache_from_kinematics(self)

    def slave_displacements(self, u: Array) -> Array:
        return jnp.asarray(u)[self.slave_dofs]

    def master_displacements(self, u: Array) -> Array:
        u_master = jnp.asarray(u)[self.master_dofs]
        return jnp.einsum("im,imd->id", self.master_weights, u_master)

    def relative_displacements(self, u: Array) -> Array:
        return self.slave_displacements(u) - self.master_displacements(u)

    def gaps(self, u: Array) -> Array:
        return self.gaps0 + jnp.einsum("id,id->i", self.relative_displacements(u), self.normals)


@dataclass(frozen=True)
class SurfaceQuadratureContactKinematics:
    """Slave-surface quadrature contact kinematics with master-facet interpolation."""

    slave_dofs: Array
    slave_weights: Array
    master_dofs: Array
    master_weights: Array
    normals: Array
    gaps0: Array
    quadrature_weights: Array
    n_dofs: int
    master_facet_ids: Array | None = None
    slave_facet_ids: Array | None = None

    def __post_init__(self):
        slave_np = np.asarray(self.slave_dofs, dtype=np.int32)
        master_np = np.asarray(self.master_dofs, dtype=np.int32)
        slave_weights = jnp.asarray(self.slave_weights)
        master_weights = jnp.asarray(self.master_weights)
        if slave_np.ndim != 3:
            raise ValueError("slave_dofs must have shape (n_contact, n_slave_nodes, dim).")
        if master_np.ndim != 3 or master_np.shape[0] != slave_np.shape[0] or master_np.shape[2] != slave_np.shape[2]:
            raise ValueError("master_dofs must have shape (n_contact, n_master_nodes, dim).")
        if slave_weights.shape != slave_np.shape[:2]:
            raise ValueError("slave_weights must have shape (n_contact, n_slave_nodes).")
        if master_weights.shape != master_np.shape[:2]:
            raise ValueError("master_weights must have shape (n_contact, n_master_nodes).")
        if slave_np.size and (slave_np.min() < 0 or slave_np.max() >= int(self.n_dofs)):
            raise ValueError("slave_dofs contains an index outside the full DOF range.")
        if master_np.size and (master_np.min() < 0 or master_np.max() >= int(self.n_dofs)):
            raise ValueError("master_dofs contains an index outside the full DOF range.")
        normals = jnp.asarray(self.normals)
        gaps0 = jnp.asarray(self.gaps0)
        quadrature_weights = jnp.asarray(self.quadrature_weights)
        if normals.shape != (slave_np.shape[0], slave_np.shape[2]):
            raise ValueError("normals must have shape (n_contact, dim).")
        if gaps0.shape != (slave_np.shape[0],):
            raise ValueError("gaps0 must have shape (n_contact,).")
        if quadrature_weights.shape != (slave_np.shape[0],):
            raise ValueError("quadrature_weights must have shape (n_contact,).")
        master_facet_ids = None
        if self.master_facet_ids is not None:
            master_facet_ids = np.asarray(self.master_facet_ids, dtype=np.int32).reshape(-1)
            if master_facet_ids.shape != (slave_np.shape[0],):
                raise ValueError("master_facet_ids must have shape (n_contact,).")
            if master_facet_ids.size and master_facet_ids.min() < 0:
                raise ValueError("master_facet_ids must not contain negative ids.")
        slave_facet_ids = None
        if self.slave_facet_ids is not None:
            slave_facet_ids = np.asarray(self.slave_facet_ids, dtype=np.int32).reshape(-1)
            if slave_facet_ids.shape != (slave_np.shape[0],):
                raise ValueError("slave_facet_ids must have shape (n_contact,).")
            if slave_facet_ids.size and slave_facet_ids.min() < 0:
                raise ValueError("slave_facet_ids must not contain negative ids.")

        object.__setattr__(self, "slave_dofs", jnp.asarray(slave_np, dtype=jnp.int32))
        object.__setattr__(self, "slave_weights", slave_weights)
        object.__setattr__(self, "master_dofs", jnp.asarray(master_np, dtype=jnp.int32))
        object.__setattr__(self, "master_weights", master_weights)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "gaps0", gaps0)
        object.__setattr__(self, "quadrature_weights", quadrature_weights)
        object.__setattr__(self, "n_dofs", int(self.n_dofs))
        if master_facet_ids is not None:
            object.__setattr__(self, "master_facet_ids", jnp.asarray(master_facet_ids, dtype=jnp.int32))
        if slave_facet_ids is not None:
            object.__setattr__(self, "slave_facet_ids", jnp.asarray(slave_facet_ids, dtype=jnp.int32))

    def search_cache(self) -> ContactSearchCache:
        return contact_search_cache_from_kinematics(self)

    def slave_displacements(self, u: Array) -> Array:
        u_slave = jnp.asarray(u)[self.slave_dofs]
        return jnp.einsum("is,isd->id", self.slave_weights, u_slave)

    def master_displacements(self, u: Array) -> Array:
        u_master = jnp.asarray(u)[self.master_dofs]
        return jnp.einsum("im,imd->id", self.master_weights, u_master)

    def relative_displacements(self, u: Array) -> Array:
        return self.slave_displacements(u) - self.master_displacements(u)

    def gaps(self, u: Array) -> Array:
        return self.gaps0 + jnp.einsum("id,id->i", self.relative_displacements(u), self.normals)


# -----------------------------------------------------------------------------
# Penalty contact laws


@dataclass(frozen=True)
class PlanePenaltyContact:
    """Frictionless unilateral penalty contact against fixed planes."""

    kinematics: ContactKinematics
    penalty: float
    smoothing: float = 0.0

    def __post_init__(self):
        if float(self.penalty) < 0.0:
            raise ValueError("penalty must be non-negative.")
        if float(self.smoothing) < 0.0:
            raise ValueError("smoothing must be non-negative.")
        object.__setattr__(self, "penalty", float(self.penalty))
        object.__setattr__(self, "smoothing", float(self.smoothing))

    def gaps(self, u: Array) -> Array:
        return self.kinematics.gaps(u)

    def penetration(self, u: Array) -> Array:
        gap = self.gaps(u)
        if self.smoothing > 0.0:
            return self.smoothing * jax.nn.softplus(-gap / self.smoothing)
        return jnp.maximum(-gap, 0.0)

    def active_mask(self, u: Array) -> Array:
        return self.gaps(u) < 0.0

    def pressure(self, u: Array) -> Array:
        return self.penalty * self.penetration(u)

    def penetration_energy(self, u: Array) -> Array:
        penetration = self.penetration(u)
        return 0.5 * self.penalty * jnp.sum(penetration**2)

    def force_norm(self, u: Array) -> Array:
        return jnp.linalg.norm(self.residual(u), ord=2)

    def active_count(self, u: Array) -> Array:
        return jnp.count_nonzero(self.active_mask(u))

    def residual(self, u: Array) -> Array:
        contact_force = -self.pressure(u)[:, None] * self.kinematics.normals
        residual = jnp.zeros(
            (self.kinematics.n_dofs,),
            dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
        )
        return residual.at[self.kinematics.dofs].add(contact_force)

    def state_from_displacement(self, u: Array) -> "ActiveContactState":
        return ActiveContactState.from_contact(self, u)

    def residual_with_state(self, state: "ActiveContactState") -> Callable[[Array], Array]:
        """
        Build a residual with the active set frozen from a previous iterate.

        This is useful for active-set Newton experiments. The residual remains
        differentiable in `u`, while the active mask is treated as outer-loop
        state and is not differentiated.
        """
        state = state.validate(self.kinematics.dofs.shape[0])

        def _residual(u: Array) -> Array:
            penetration = jnp.maximum(-self.gaps(u), 0.0)
            pressure = self.penalty * penetration * state.active.astype(penetration.dtype)
            contact_force = -pressure[:, None] * self.kinematics.normals
            residual = jnp.zeros(
                (self.kinematics.n_dofs,),
                dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
            )
            return residual.at[self.kinematics.dofs].add(contact_force)

        return _residual


@dataclass(frozen=True)
class PairedPenaltyContact:
    """Frictionless unilateral penalty contact between paired slave/master DOFs."""

    kinematics: PairedContactKinematics
    penalty: float
    smoothing: float = 0.0

    def __post_init__(self):
        if float(self.penalty) < 0.0:
            raise ValueError("penalty must be non-negative.")
        if float(self.smoothing) < 0.0:
            raise ValueError("smoothing must be non-negative.")
        object.__setattr__(self, "penalty", float(self.penalty))
        object.__setattr__(self, "smoothing", float(self.smoothing))

    def gaps(self, u: Array) -> Array:
        return self.kinematics.gaps(u)

    def penetration(self, u: Array) -> Array:
        gap = self.gaps(u)
        if self.smoothing > 0.0:
            return self.smoothing * jax.nn.softplus(-gap / self.smoothing)
        return jnp.maximum(-gap, 0.0)

    def active_mask(self, u: Array) -> Array:
        return self.gaps(u) < 0.0

    def pressure(self, u: Array) -> Array:
        return self.penalty * self.penetration(u)

    def penetration_energy(self, u: Array) -> Array:
        penetration = self.penetration(u)
        return 0.5 * self.penalty * jnp.sum(penetration**2)

    def force_norm(self, u: Array) -> Array:
        return jnp.linalg.norm(self.residual(u), ord=2)

    def active_count(self, u: Array) -> Array:
        return jnp.count_nonzero(self.active_mask(u))

    def residual(self, u: Array) -> Array:
        slave_force = -self.pressure(u)[:, None] * self.kinematics.normals
        residual = jnp.zeros(
            (self.kinematics.n_dofs,),
            dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
        )
        residual = residual.at[self.kinematics.slave_dofs].add(slave_force)
        residual = residual.at[self.kinematics.master_dofs].add(-slave_force)
        return residual

    def state_from_displacement(self, u: Array) -> "ActiveContactState":
        return ActiveContactState.from_contact(self, u)

    def residual_with_state(self, state: "ActiveContactState") -> Callable[[Array], Array]:
        """Build a paired-contact residual with the active set frozen."""
        state = state.validate(self.kinematics.slave_dofs.shape[0])

        def _residual(u: Array) -> Array:
            penetration = jnp.maximum(-self.gaps(u), 0.0)
            pressure = self.penalty * penetration * state.active.astype(penetration.dtype)
            slave_force = -pressure[:, None] * self.kinematics.normals
            residual = jnp.zeros(
                (self.kinematics.n_dofs,),
                dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
            )
            residual = residual.at[self.kinematics.slave_dofs].add(slave_force)
            residual = residual.at[self.kinematics.master_dofs].add(-slave_force)
            return residual

        return _residual


@dataclass(frozen=True)
class NodeSurfacePenaltyContact:
    """Frictionless unilateral penalty contact from slave nodes to master facets."""

    kinematics: NodeSurfaceContactKinematics
    penalty: float
    smoothing: float = 0.0

    def __post_init__(self):
        if float(self.penalty) < 0.0:
            raise ValueError("penalty must be non-negative.")
        if float(self.smoothing) < 0.0:
            raise ValueError("smoothing must be non-negative.")
        object.__setattr__(self, "penalty", float(self.penalty))
        object.__setattr__(self, "smoothing", float(self.smoothing))

    def gaps(self, u: Array) -> Array:
        return self.kinematics.gaps(u)

    def penetration(self, u: Array) -> Array:
        gap = self.gaps(u)
        if self.smoothing > 0.0:
            return self.smoothing * jax.nn.softplus(-gap / self.smoothing)
        return jnp.maximum(-gap, 0.0)

    def active_mask(self, u: Array) -> Array:
        return self.gaps(u) < 0.0

    def pressure(self, u: Array) -> Array:
        return self.penalty * self.penetration(u)

    def penetration_energy(self, u: Array) -> Array:
        penetration = self.penetration(u)
        return 0.5 * self.penalty * jnp.sum(penetration**2)

    def force_norm(self, u: Array) -> Array:
        return jnp.linalg.norm(self.residual(u), ord=2)

    def active_count(self, u: Array) -> Array:
        return jnp.count_nonzero(self.active_mask(u))

    def residual(self, u: Array) -> Array:
        slave_force = -self.pressure(u)[:, None] * self.kinematics.normals
        master_force = (
            self.kinematics.master_weights[:, :, None] * (-slave_force)[:, None, :]
        )
        residual = jnp.zeros(
            (self.kinematics.n_dofs,),
            dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
        )
        residual = residual.at[self.kinematics.slave_dofs].add(slave_force)
        residual = residual.at[self.kinematics.master_dofs].add(master_force)
        return residual

    def state_from_displacement(self, u: Array) -> "ActiveContactState":
        return ActiveContactState.from_contact(self, u)

    def residual_with_state(self, state: "ActiveContactState") -> Callable[[Array], Array]:
        """Build a node-surface residual with active set and weights frozen."""
        state = state.validate(self.kinematics.slave_dofs.shape[0])

        def _residual(u: Array) -> Array:
            penetration = jnp.maximum(-self.gaps(u), 0.0)
            pressure = self.penalty * penetration * state.active.astype(penetration.dtype)
            slave_force = -pressure[:, None] * self.kinematics.normals
            master_force = (
                self.kinematics.master_weights[:, :, None] * (-slave_force)[:, None, :]
            )
            residual = jnp.zeros(
                (self.kinematics.n_dofs,),
                dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
            )
            residual = residual.at[self.kinematics.slave_dofs].add(slave_force)
            residual = residual.at[self.kinematics.master_dofs].add(master_force)
            return residual

        return _residual


@dataclass(frozen=True)
class SurfaceQuadraturePenaltyContact:
    """Frictionless unilateral penalty contact over slave-surface quadrature points."""

    kinematics: SurfaceQuadratureContactKinematics
    penalty: float
    smoothing: float = 0.0

    def __post_init__(self):
        if float(self.penalty) < 0.0:
            raise ValueError("penalty must be non-negative.")
        if float(self.smoothing) < 0.0:
            raise ValueError("smoothing must be non-negative.")
        object.__setattr__(self, "penalty", float(self.penalty))
        object.__setattr__(self, "smoothing", float(self.smoothing))

    def gaps(self, u: Array) -> Array:
        return self.kinematics.gaps(u)

    def penetration(self, u: Array) -> Array:
        gap = self.gaps(u)
        if self.smoothing > 0.0:
            return self.smoothing * jax.nn.softplus(-gap / self.smoothing)
        return jnp.maximum(-gap, 0.0)

    def active_mask(self, u: Array) -> Array:
        return self.gaps(u) < 0.0

    def pressure(self, u: Array) -> Array:
        return self.penalty * self.penetration(u)

    def penetration_energy(self, u: Array) -> Array:
        penetration = self.penetration(u)
        return 0.5 * self.penalty * jnp.sum(self.kinematics.quadrature_weights * penetration**2)

    def force_norm(self, u: Array) -> Array:
        return jnp.linalg.norm(self.residual(u), ord=2)

    def active_count(self, u: Array) -> Array:
        return jnp.count_nonzero(self.active_mask(u))

    def residual(self, u: Array) -> Array:
        contact_force = (
            -self.kinematics.quadrature_weights[:, None]
            * self.pressure(u)[:, None]
            * self.kinematics.normals
        )
        master_force = self.kinematics.master_weights[:, :, None] * (-contact_force)[:, None, :]
        slave_force = self.kinematics.slave_weights[:, :, None] * contact_force[:, None, :]
        residual = jnp.zeros(
            (self.kinematics.n_dofs,),
            dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
        )
        residual = residual.at[self.kinematics.slave_dofs].add(slave_force)
        residual = residual.at[self.kinematics.master_dofs].add(master_force)
        return residual

    def state_from_displacement(self, u: Array) -> "ActiveContactState":
        return ActiveContactState.from_contact(self, u)

    def residual_with_state(self, state: "ActiveContactState") -> Callable[[Array], Array]:
        """Build a quadrature contact residual with active set and weights frozen."""
        state = state.validate(self.kinematics.slave_dofs.shape[0])

        def _residual(u: Array) -> Array:
            penetration = jnp.maximum(-self.gaps(u), 0.0)
            pressure = self.penalty * penetration * state.active.astype(penetration.dtype)
            contact_force = (
                -self.kinematics.quadrature_weights[:, None]
                * pressure[:, None]
                * self.kinematics.normals
            )
            master_force = self.kinematics.master_weights[:, :, None] * (-contact_force)[:, None, :]
            slave_force = self.kinematics.slave_weights[:, :, None] * contact_force[:, None, :]
            residual = jnp.zeros(
                (self.kinematics.n_dofs,),
                dtype=jnp.result_type(u, self.kinematics.normals, self.kinematics.gaps0),
            )
            residual = residual.at[self.kinematics.slave_dofs].add(slave_force)
            residual = residual.at[self.kinematics.master_dofs].add(master_force)
            return residual

        return _residual


# -----------------------------------------------------------------------------
# Active contact state and snapshots


@dataclass(frozen=True)
class ActiveContactState:
    """Outer-loop state for active contact experiments."""

    active: Array
    gaps: Array | None = None
    pressure: Array | None = None

    def validate(self, n_contact: int) -> "ActiveContactState":
        active = jnp.asarray(self.active, dtype=bool).reshape(-1)
        if active.shape != (int(n_contact),):
            raise ValueError("active must have shape (n_contact,).")
        gaps = None if self.gaps is None else jnp.asarray(self.gaps).reshape(-1)
        pressure = None if self.pressure is None else jnp.asarray(self.pressure).reshape(-1)
        if gaps is not None and gaps.shape != active.shape:
            raise ValueError("gaps must have shape (n_contact,).")
        if pressure is not None and pressure.shape != active.shape:
            raise ValueError("pressure must have shape (n_contact,).")
        return ActiveContactState(active=active, gaps=gaps, pressure=pressure)

    @classmethod
    def from_contact(cls, contact, u: Array) -> "ActiveContactState":
        gaps = contact.gaps(u)
        return cls(
            active=gaps < 0.0,
            gaps=gaps,
            pressure=contact.pressure(u),
        )

    def changed(self, other: "ActiveContactState") -> Array:
        return jnp.any(jnp.asarray(self.active, dtype=bool) != jnp.asarray(other.active, dtype=bool))


@dataclass(frozen=True)
class ContactUpdateSnapshot:
    """Frozen contact object and active state for outer contact updates."""

    contact: object
    active_state: ActiveContactState
    history: object | None = None
    kinematics_tol: float = 1e-8

    @classmethod
    def from_contact(
        cls,
        contact,
        u: Array,
        *,
        history: object | None = None,
        kinematics_tol: float = 1e-8,
    ) -> "ContactUpdateSnapshot":
        return cls(
            contact=contact,
            active_state=contact.state_from_displacement(u),
            history=history,
            kinematics_tol=kinematics_tol,
        )

    def residual(self) -> Callable[[Array], Array]:
        return self.contact.residual_with_state(self.active_state)

    def with_history(self, history: object | None) -> "ContactUpdateSnapshot":
        return ContactUpdateSnapshot(
            contact=self.contact,
            active_state=self.active_state,
            history=history,
            kinematics_tol=self.kinematics_tol,
        )

    def changed(self, other: "ContactUpdateSnapshot") -> Array:
        active_changed = self.active_state.changed(other.active_state)
        kinematics_changed = _contact_kinematics_changed(
            other.contact.kinematics,
            self.contact.kinematics,
            tol=max(float(self.kinematics_tol), float(other.kinematics_tol)),
        )
        contact_type_changed = type(self.contact) is not type(other.contact)
        return jnp.logical_or(
            jnp.logical_or(active_changed, kinematics_changed),
            jnp.asarray(contact_type_changed),
        )


# -----------------------------------------------------------------------------
# Tangential friction history and residual helpers


@dataclass(frozen=True)
class TangentialPenaltyHistory:
    """History and diagnostics for regularized tangential penalty friction."""

    tangential_slip: Array
    stick: Array
    friction_force: Array

    def __post_init__(self):
        slip = jnp.asarray(self.tangential_slip)
        stick = jnp.asarray(self.stick, dtype=bool).reshape(-1)
        force = jnp.asarray(self.friction_force)
        if slip.ndim != 2:
            raise ValueError("tangential_slip must have shape (n_contact, n_tangent).")
        if stick.shape != (slip.shape[0],):
            raise ValueError("stick must have shape (n_contact,).")
        if force.ndim != 2 or force.shape[0] != slip.shape[0]:
            raise ValueError("friction_force must have shape (n_contact, dim).")
        object.__setattr__(self, "tangential_slip", slip)
        object.__setattr__(self, "stick", stick)
        object.__setattr__(self, "friction_force", force)

    def slip_norm(self) -> Array:
        return jnp.linalg.norm(self.tangential_slip, ord=2)

    def stick_count(self) -> Array:
        return jnp.count_nonzero(self.stick)


def _contact_relative_displacements(contact, u: Array) -> Array:
    kinematics = contact.kinematics
    if hasattr(kinematics, "relative_displacements"):
        return kinematics.relative_displacements(u)
    if hasattr(kinematics, "displacements"):
        return kinematics.displacements(u)
    raise TypeError("contact kinematics must provide displacements or relative_displacements.")


def update_tangential_penalty_history(
    contact,
    u: Array,
    u_prev: Array,
    previous_history: TangentialPenaltyHistory | None,
    *,
    mu: float,
    tangential_penalty: float,
) -> TangentialPenaltyHistory:
    """
    Update regularized tangential penalty friction history.

    This helper computes contact-point tangential slip increments, forms a trial
    penalty friction force, and clips that force by `mu * pressure`. It returns
    history/diagnostics only; friction residual assembly is intentionally left
    to a later contact law.
    """
    mu = float(mu)
    tangential_penalty = float(tangential_penalty)
    if mu < 0.0:
        raise ValueError("mu must be non-negative.")
    if tangential_penalty <= 0.0:
        raise ValueError("tangential_penalty must be positive.")

    normals = jnp.asarray(contact.kinematics.normals)
    tangents = orthonormal_tangent_basis(normals)
    rel_u = _contact_relative_displacements(contact, u)
    rel_prev = _contact_relative_displacements(contact, u_prev)
    delta_slip = jnp.einsum("itd,id->it", tangents, rel_u - rel_prev)
    if previous_history is None:
        previous_slip = jnp.zeros_like(delta_slip)
    else:
        previous_slip = jnp.asarray(previous_history.tangential_slip)
        if previous_slip.shape != delta_slip.shape:
            raise ValueError("previous_history.tangential_slip has an incompatible shape.")

    trial_slip = previous_slip + delta_slip
    trial_force_components = -tangential_penalty * trial_slip
    trial_force_norm = jnp.linalg.norm(trial_force_components, axis=1)
    pressure_limit = float(mu) * contact.pressure(u)
    active = contact.active_mask(u)
    eps = jnp.finfo(trial_force_components.dtype).eps
    scale = jnp.minimum(1.0, pressure_limit / jnp.maximum(trial_force_norm, eps))
    scale = jnp.where(active, scale, 0.0)
    force_components = trial_force_components * scale[:, None]
    friction_force = jnp.einsum("it,itd->id", force_components, tangents)
    tangential_slip = jnp.where(
        active[:, None],
        -force_components / tangential_penalty,
        jnp.zeros_like(force_components),
    )
    stick = active & (trial_force_norm <= pressure_limit + 10.0 * eps)
    return TangentialPenaltyHistory(
        tangential_slip=tangential_slip,
        stick=stick,
        friction_force=friction_force,
    )


def slip_norm(history: TangentialPenaltyHistory) -> Array:
    """Return the total norm of stored tangential slip."""
    return history.slip_norm()


def stick_count(history: TangentialPenaltyHistory) -> Array:
    """Return the number of sticking contact points."""
    return history.stick_count()


def friction_residual_from_history(contact, history: TangentialPenaltyHistory) -> Array:
    """Scatter stored tangential friction forces into a full-space residual."""
    force = jnp.asarray(history.friction_force)
    kinematics = contact.kinematics
    if force.shape != jnp.asarray(kinematics.normals).shape:
        raise ValueError("history.friction_force must match contact normal shape.")

    residual = jnp.zeros(
        (int(kinematics.n_dofs),),
        dtype=jnp.result_type(force, kinematics.normals, kinematics.gaps0),
    )
    if hasattr(kinematics, "dofs"):
        return residual.at[kinematics.dofs].add(force)
    if hasattr(kinematics, "slave_weights") and hasattr(kinematics, "master_weights"):
        if hasattr(kinematics, "quadrature_weights"):
            force = kinematics.quadrature_weights[:, None] * force
        slave_force = kinematics.slave_weights[:, :, None] * force[:, None, :]
        master_force = kinematics.master_weights[:, :, None] * (-force)[:, None, :]
        residual = residual.at[kinematics.slave_dofs].add(slave_force)
        residual = residual.at[kinematics.master_dofs].add(master_force)
        return residual
    if hasattr(kinematics, "slave_dofs") and hasattr(kinematics, "master_weights"):
        master_force = kinematics.master_weights[:, :, None] * (-force)[:, None, :]
        residual = residual.at[kinematics.slave_dofs].add(force)
        residual = residual.at[kinematics.master_dofs].add(master_force)
        return residual
    if hasattr(kinematics, "slave_dofs") and hasattr(kinematics, "master_dofs"):
        residual = residual.at[kinematics.slave_dofs].add(force)
        residual = residual.at[kinematics.master_dofs].add(-force)
        return residual
    raise TypeError("unsupported contact kinematics for friction residual scatter.")


def make_friction_residual(contact, history: TangentialPenaltyHistory) -> Callable[[Array], Array]:
    """
    Build a frozen tangential-friction residual function.

    The returned residual ignores its displacement argument because the friction
    force is stored in `history`. Update history explicitly between solves or
    active-contact outer iterations.
    """

    def _residual(u: Array) -> Array:
        del u
        return friction_residual_from_history(contact, history)

    return _residual


@dataclass(frozen=True)
class FrictionalContactUpdateSnapshot:
    """Contact update snapshot with frozen tangential friction history."""

    base: ContactUpdateSnapshot

    @property
    def contact(self):
        return self.base.contact

    @property
    def active_state(self) -> ActiveContactState:
        return self.base.active_state

    @property
    def history(self) -> TangentialPenaltyHistory | None:
        return self.base.history

    def residual(self) -> Callable[[Array], Array]:
        normal_residual = self.base.residual()
        if self.history is None:
            return normal_residual
        friction_residual = make_friction_residual(self.contact, self.history)
        return compose_residuals(normal_residual, friction_residual)

    def changed(self, other: "FrictionalContactUpdateSnapshot") -> Array:
        return self.base.changed(other.base)


@dataclass(frozen=True)
class TangentialPenaltyFrictionManager:
    """Explicit manager for frozen tangential penalty friction history."""

    mu: float
    tangential_penalty: float
    previous_displacement: Array
    history: TangentialPenaltyHistory | None = None
    kinematics_tol: float = 1e-8

    def __post_init__(self):
        mu = float(self.mu)
        tangential_penalty = float(self.tangential_penalty)
        if mu < 0.0:
            raise ValueError("mu must be non-negative.")
        if tangential_penalty <= 0.0:
            raise ValueError("tangential_penalty must be positive.")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "tangential_penalty", tangential_penalty)
        object.__setattr__(self, "previous_displacement", jnp.asarray(self.previous_displacement).reshape(-1))
        object.__setattr__(self, "kinematics_tol", float(self.kinematics_tol))

    def snapshot(self, contact, u: Array) -> FrictionalContactUpdateSnapshot:
        return FrictionalContactUpdateSnapshot(
            ContactUpdateSnapshot.from_contact(
                contact,
                u,
                history=self.history,
                kinematics_tol=self.kinematics_tol,
            )
        )

    def advance(self, contact, u: Array) -> "TangentialPenaltyFrictionManager":
        u = jnp.asarray(u).reshape(-1)
        history = update_tangential_penalty_history(
            contact,
            u,
            self.previous_displacement,
            self.history,
            mu=self.mu,
            tangential_penalty=self.tangential_penalty,
        )
        return TangentialPenaltyFrictionManager(
            mu=self.mu,
            tangential_penalty=self.tangential_penalty,
            previous_displacement=u,
            history=history,
            kinematics_tol=self.kinematics_tol,
        )

    def snapshot_and_advance(self, contact, u: Array) -> tuple[FrictionalContactUpdateSnapshot, "TangentialPenaltyFrictionManager"]:
        next_manager = self.advance(contact, u)
        return next_manager.snapshot(contact, u), next_manager


def update_active_contact_state(contact, u: Array) -> ActiveContactState:
    """Convenience helper for explicit active-set updates."""
    return ActiveContactState.from_contact(contact, u)


# -----------------------------------------------------------------------------
# Compatibility factories


def make_unilateral_plane_contact_residual(
    n_dofs: int,
    contact_dofs: Array,
    normals: Array,
    gaps0: Array,
    penalty: float,
    *,
    smoothing: float = 0.0,
) -> Callable[[Array], Array]:
    """
    Build a full-space frictionless penalty contact residual against rigid planes.

    `contact_dofs` has shape (n_contact, dim) and maps each contact point to its
    displacement DOFs. The gap is `gap0 + dot(u_contact, normal)`. Negative gaps
    produce residual force `-penalty * penetration * normal`.

    Set `smoothing > 0` to replace max(-gap, 0) with a softplus approximation.
    """
    contact = PlanePenaltyContact(
        ContactKinematics(
            dofs=contact_dofs,
            normals=normals,
            gaps0=gaps0,
            n_dofs=n_dofs,
        ),
        penalty=penalty,
        smoothing=smoothing,
    )
    return contact.residual

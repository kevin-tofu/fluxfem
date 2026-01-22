from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp

from ..mesh.surface import SurfaceMesh
from ..core.forms import vector_load_form


def _quad_area(p0, p1, p2, p3):
    """Compute quad area by splitting into two triangles."""
    a1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p3 - p0))
    a2 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    return a1 + a2


def _polygon_area(pts: np.ndarray) -> float:
    """Polygon area in 3D by fan triangulation (works for tri/quad faces)."""
    if pts.shape[0] < 3:
        return 0.0
    area = 0.0
    p0 = pts[0]
    for i in range(1, pts.shape[0] - 1):
        v1 = pts[i] - p0
        v2 = pts[i + 1] - p0
        area += 0.5 * np.linalg.norm(np.cross(v1, v2))
    return float(area)


@dataclass(eq=False)
class SurfaceFormField:
    N: np.ndarray
    value_dim: int


@dataclass(eq=False)
class SurfaceFormContext:
    v: SurfaceFormField
    x_q: np.ndarray
    w: np.ndarray
    detJ: np.ndarray
    facet_id: int
    normal: np.ndarray | None = None


def facet_area(coords: np.ndarray, nodes: np.ndarray) -> float:
    """Compute facet area for a 4-node quad (uses numpy)."""
    pts = np.asarray(coords)[np.asarray(nodes, dtype=int)]
    if pts.shape[0] != 4:
        raise ValueError("facet_area expects 4-node quadrilateral facet")
    return float(_quad_area(pts[0], pts[1], pts[2], pts[3]))


def add_neumann_load(
    F: npt.ArrayLike,
    facet_nodes: Sequence[int],
    traction: npt.ArrayLike,
    *,
    dim: int = 1,
    coords: Optional[npt.ArrayLike] = None,
    area: float | None = None,
) -> np.ndarray:
    """
    Simple helper to add Neumann traction to the load vector.
    - facet_nodes: face node IDs (length m)
    - traction: scalar or length-dim array (outward traction)
    - dim: dofs per node
    - coords/area: provide area if known; if coords given (4-node quad) area is computed
    """
    facet_nodes = np.asarray(facet_nodes, dtype=int)
    m = len(facet_nodes)
    if m == 0:
        return np.asarray(F)

    if area is None:
        if coords is not None:
            area = facet_area(coords, facet_nodes)
        else:
            area = 1.0  # fallback: treat as unit area

    traction = np.asarray(traction, dtype=float)
    if traction.ndim == 0 and dim > 1:
        traction = np.full(dim, float(traction))

    F_new = np.asarray(F, dtype=float).copy()
    if dim == 1:
        load = float(traction) * area / m
        np.add.at(F_new, facet_nodes, load)
    else:
        for n in facet_nodes:
            for d in range(dim):
                dof = dim * n + d
                F_new[dof] += traction[d] * area / m
    return F_new


def vector_surface_load_form(ctx: SurfaceFormContext, load: npt.ArrayLike) -> np.ndarray:
    """
    Linear form for vector surface load: v · t on a facet.
    load: (dim,) or (n_q, dim)
    returns: (n_q, n_nodes*dim)
    """
    load_arr = np.asarray(load, dtype=float)
    return np.asarray(vector_load_form(ctx.v, load_arr))


vector_surface_load_form._ff_kind = "linear"
vector_surface_load_form._ff_domain = "surface"


def make_vector_surface_load_form(load_fn):
    """
    Build a vector surface load form from a callable f(x_q) -> (n_q, dim).
    """
    def _form(ctx: SurfaceFormContext, _params):
        load_q = load_fn(ctx.x_q)
        return vector_surface_load_form(ctx, load_q)

    _form._ff_kind = "linear"
    _form._ff_domain = "surface"
    return _form


def traction_vector(traction, traction_dir: str) -> np.ndarray:
    """
    Resolve traction magnitude and direction string into a vector.
    """
    dir_map = {
        "x": (1.0, 0.0, 0.0),
        "xpos": (1.0, 0.0, 0.0),
        "xneg": (-1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "ypos": (0.0, 1.0, 0.0),
        "yneg": (0.0, -1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "zpos": (0.0, 0.0, 1.0),
        "zneg": (0.0, 0.0, -1.0),
    }
    key = traction_dir.strip().lower()
    if key not in dir_map:
        raise ValueError("TRACTION_DIR must be one of x/xpos/xneg/y/ypos/yneg/z/zpos/zneg")
    return float(traction) * np.asarray(dir_map[key], dtype=float)


def _surface_quadrature(node_coords: np.ndarray):
    m = node_coords.shape[0]
    if m == 4:
        gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        gauss_wts = np.array([1.0, 1.0])
        N_list = []
        x_list = []
        w_list = []
        detJ_list = []
        for xi, wx in zip(gauss_pts, gauss_wts):
            for eta, wy in zip(gauss_pts, gauss_wts):
                N = 0.25 * np.array(
                    [
                        (1 - xi) * (1 - eta),
                        (1 + xi) * (1 - eta),
                        (1 + xi) * (1 + eta),
                        (1 - xi) * (1 + eta),
                    ]
                )
                dN_dxi = 0.25 * np.array(
                    [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]
                )
                dN_deta = 0.25 * np.array(
                    [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
                )
                dx_dxi = np.sum(node_coords * dN_dxi[:, None], axis=0)
                dx_deta = np.sum(node_coords * dN_deta[:, None], axis=0)
                J_s = np.linalg.norm(np.cross(dx_dxi, dx_deta))
                w = wx * wy
                x_q = np.sum(node_coords * N[:, None], axis=0)
                N_list.append(N)
                x_list.append(x_q)
                w_list.append(w)
                detJ_list.append(J_s)
        return (
            np.asarray(N_list, dtype=float),
            np.asarray(x_list, dtype=float),
            np.asarray(w_list, dtype=float),
            np.asarray(detJ_list, dtype=float),
        )
    if m == 3:
        area = _polygon_area(node_coords)
        N = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        x_q = np.array([node_coords.mean(axis=0)], dtype=float)
        w = np.array([1.0], dtype=float)
        detJ = np.array([area], dtype=float)
        return N, x_q, w, detJ
    area = _polygon_area(node_coords)
    N = np.full((1, m), 1.0 / float(m), dtype=float)
    x_q = np.array([node_coords.mean(axis=0)], dtype=float)
    w = np.array([1.0], dtype=float)
    detJ = np.array([area], dtype=float)
    return N, x_q, w, detJ


def assemble_surface_linear_form(
    surface: SurfaceMesh,
    form,
    params,
    *,
    dim: int,
    n_total_nodes: int | None = None,
    F0: npt.ArrayLike | None = None,
) -> np.ndarray:
    """
    Assemble a linear form over surface facets using a weak-form callback.
    """
    facets = np.asarray(surface.conn, dtype=int)
    coords = np.asarray(surface.coords)
    n_nodes = surface.n_nodes if n_total_nodes is None else int(n_total_nodes)
    n_dofs = n_nodes * dim
    F = np.zeros(n_dofs, dtype=float) if F0 is None else np.asarray(F0, dtype=float).copy()
    if F.shape[0] != n_dofs:
        raise ValueError(f"F length {F.shape[0]} does not match expected {n_dofs}")

    normals = facet_normals(surface, outward_from=np.mean(coords, axis=0), normalize=True)
    for facet_id, facet in enumerate(facets):
        node_coords = coords[facet]
        N, x_q, w, detJ = _surface_quadrature(node_coords)
        ctx = SurfaceFormContext(
            v=SurfaceFormField(N=N, value_dim=dim),
            x_q=x_q,
            w=w,
            detJ=detJ,
            facet_id=facet_id,
            normal=normals[facet_id],
        )
        fe_q = form(ctx, params)
        if fe_q.ndim != 2 or fe_q.shape[0] != N.shape[0]:
            raise ValueError("surface form must return array shape (n_q, n_ldofs)")
        if getattr(form, "_includes_measure", False):
            fe = np.einsum("qi->i", fe_q)
        else:
            wJ = w * detJ
            fe = np.einsum("qi,q->i", fe_q, wJ)
        for a, node in enumerate(facet):
            for d in range(dim):
                local = dim * a + d
                dof = dim * int(node) + d
                F[dof] += fe[local]
    return F


def add_robin(
    F: npt.ArrayLike,
    K: npt.ArrayLike,
    facet_nodes: Sequence[int],
    alpha: float,
    g: npt.ArrayLike,
    *,
    dim: int = 1,
    coords: Optional[npt.ArrayLike] = None,
    area: float | None = None,
):
    """
    Add simple Robin term ∫ alpha (u - g) v dΓ with diagonal approximation.
    - facet_nodes: face node IDs
    - alpha: scalar coefficient
    - g: target value (scalar or length dim)
    - dim: dofs per node
    - coords/area: provide area if known; if coords given (4-node quad) area is computed
    Returns: (F_new, K_new)
    """
    facet_nodes = np.asarray(facet_nodes, dtype=int)
    m = len(facet_nodes)
    if m == 0:
        return np.asarray(F), np.asarray(K)

    if area is None:
        if coords is not None:
            area = facet_area(coords, facet_nodes)
        else:
            area = 1.0

    alpha = float(alpha)
    g = np.asarray(g, dtype=float)
    if g.ndim == 0 and dim > 1:
        g = np.full(dim, float(g))

    F_new = np.asarray(F, dtype=float).copy()
    K_new = np.asarray(K, dtype=float).copy()

    weight = alpha * area / m

    if dim == 1:
        for n in facet_nodes:
            K_new[n, n] += weight
            F_new[n] += weight * float(g)
    else:
        for n in facet_nodes:
            for d in range(dim):
                dof = dim * n + d
                K_new[dof, dof] += weight
                F_new[dof] += weight * g[d]

    return F_new, K_new


def assemble_surface_load(
    surface: SurfaceMesh,
    load: npt.ArrayLike,
    *,
    dim: int,
    n_total_nodes: int | None = None,
    F0: npt.ArrayLike | None = None,
) -> np.ndarray:
    """
    Assemble a global RHS vector from constant surface load (vector per facet node).

    - surface: SurfaceMesh whose node numbering matches the volume mesh
    - load: array-like with shape (dim,) or (n_facets, dim)
    - dim: dofs per node (e.g., 3 for displacement)
    - n_total_nodes: optional total nodes for sizing; defaults to surface mesh node count
    - F0: optional initial RHS to add into (copied)
    """
    facets = np.asarray(surface.conn, dtype=int)
    n_facets = facets.shape[0]
    load_arr = np.asarray(load, dtype=float)
    if load_arr.ndim == 1:
        load_arr = np.tile(load_arr[None, :], (n_facets, 1))
    elif load_arr.shape[0] != n_facets:
        raise ValueError("load must be shape (dim,) or (n_facets, dim)")

    n_nodes = surface.n_nodes if n_total_nodes is None else int(n_total_nodes)
    n_dofs = n_nodes * dim
    F = np.zeros(n_dofs, dtype=float) if F0 is None else np.asarray(F0, dtype=float).copy()
    if F.shape[0] != n_dofs:
        raise ValueError(f"F length {F.shape[0]} does not match expected {n_dofs}")

    coords = np.asarray(surface.coords)
    # Use a simple 2x2 Gauss integration for quad facets (planar assumption).
    gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gauss_wts = np.array([1.0, 1.0])

    for facet, t in zip(facets, load_arr):
        node_coords = coords[facet]
        m = len(facet)
        if m == 4:
            for xi, wx in zip(gauss_pts, gauss_wts):
                for eta, wy in zip(gauss_pts, gauss_wts):
                    # Bilinear shape functions on reference square [-1,1]^2
                    N = 0.25 * np.array(
                        [
                            (1 - xi) * (1 - eta),
                            (1 + xi) * (1 - eta),
                            (1 + xi) * (1 + eta),
                            (1 - xi) * (1 + eta),
                        ]
                    )
                    dN_dxi = 0.25 * np.array(
                        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]
                    )
                    dN_deta = 0.25 * np.array(
                        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
                    )
                    # Surface Jacobian |dX/dxi x dX/deta|
                    dx_dxi = np.sum(node_coords * dN_dxi[:, None], axis=0)
                    dx_deta = np.sum(node_coords * dN_deta[:, None], axis=0)
                    J_s = np.linalg.norm(np.cross(dx_dxi, dx_deta))
                    w = wx * wy
                    for a, Na in enumerate(N):
                        for d in range(dim):
                            dof = dim * facet[a] + d
                            F[dof] += Na * t[d] * J_s * w
        elif m == 3:
            # Linear triangle: one-point (centroid) quadrature is exact for constant traction.
            area = _polygon_area(node_coords)
            N = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
            for a, Na in enumerate(N):
                for d in range(dim):
                    dof = dim * facet[a] + d
                    F[dof] += Na * t[d] * area
        else:
            # fallback: uniform distribution using facet area
            area = _polygon_area(node_coords)
            for n in facet:
                for d in range(dim):
                    dof = dim * n + d
                    F[dof] += area * t[d] / float(m)
    return F


def assemble_surface_traction(
    surface: SurfaceMesh,
    traction: float | Sequence[float],
    *,
    dim: int = 3,
    n_total_nodes: int | None = None,
    F0: npt.ArrayLike | None = None,
    outward_from: npt.ArrayLike | None = None,
) -> np.ndarray:
    """
    Assemble surface load from scalar traction acting along facet normals (mechanics).
    - traction: scalar or (n_facets,) array, positive = along oriented normal, negative = opposite.
    - dim: must be 3 (vector mechanics).
    """
    if dim != 3:
        raise ValueError("assemble_surface_traction expects dim=3 (vector field). Use assemble_surface_load for other cases.")
    normals = facet_normals(surface, outward_from=outward_from, normalize=True)
    traction_arr = np.asarray(traction, dtype=float)
    if traction_arr.ndim > 1:
        raise ValueError("traction must be scalar or shape (n_facets,), not a vector per component")
    if traction_arr.ndim == 0:
        traction_arr = np.full((normals.shape[0],), float(traction_arr))
    if traction_arr.shape[0] != normals.shape[0]:
        raise ValueError("traction must be scalar or match number of facets")
    load = traction_arr[:, None] * normals  # along normal
    return assemble_surface_load(surface, load, dim=dim, n_total_nodes=n_total_nodes, F0=F0)


def facet_normals(surface: SurfaceMesh, *, outward_from: npt.ArrayLike | None = None, normalize: bool = True) -> np.ndarray:
    """
    Compute per-facet normals for a SurfaceMesh.

    - outward_from: optional point (3,) assumed to lie inside the volume; normals are flipped to point away from this point.
    - normalize: return unit normals if True.

    Note: orientation of input facets may be inconsistent; outward_from can be used to enforce a consistent outward direction.
    """
    coords = np.asarray(surface.coords)
    facets = np.asarray(surface.conn, dtype=int)
    normals = np.zeros((facets.shape[0], 3), dtype=float)

    for i, facet in enumerate(facets):
        if len(facet) < 3:
            continue
        n = None
        p0 = coords[facet[0]]
        for j in range(1, len(facet) - 1):
            p1 = coords[facet[j]]
            p2 = coords[facet[j + 1]]
            n_candidate = np.cross(p1 - p0, p2 - p0)
            if np.linalg.norm(n_candidate) > 0.0:
                n = n_candidate
                break
        if n is None:
            continue
        if normalize:
            norm = np.linalg.norm(n)
            n = n / norm if norm != 0.0 else n
        if outward_from is not None:
            c = coords[facet].mean(axis=0)
            v = c - np.asarray(outward_from, dtype=float)
            if np.dot(n, v) < 0:
                n = -n
        normals[i] = n
    return normals


def assemble_surface_flux(
    surface: SurfaceMesh,
    flux: npt.ArrayLike,
    *,
    n_total_nodes: int | None = None,
    F0: npt.ArrayLike | None = None,
    outward_from: npt.ArrayLike | None = None,
) -> np.ndarray:
    """
    Scalar flux along facet normal (dim=1). Positive flux acts along oriented normal.
    """
    normals = facet_normals(surface, outward_from=outward_from, normalize=True)
    flux_arr = np.asarray(flux, dtype=float)
    if flux_arr.ndim == 0:
        flux_arr = np.full((normals.shape[0],), float(flux_arr))
    if flux_arr.shape[0] != normals.shape[0]:
        raise ValueError("flux must be scalar or match number of facets")
    load = flux_arr[:, None] * normals
    return assemble_surface_load(surface, load, dim=1, n_total_nodes=n_total_nodes, F0=F0)

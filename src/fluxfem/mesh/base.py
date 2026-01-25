from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence
import numpy as np
import jax
import jax.numpy as jnp

from .dtypes import INDEX_DTYPE, NP_INDEX_DTYPE


@dataclass
class BaseMeshClosure:
    """
    Base mesh container with coordinates, connectivity, and optional tags.

    Concrete mesh types should implement face_node_patterns() for boundary queries.
    """
    coords: jnp.ndarray
    conn: jnp.ndarray
    cell_tags: Optional[jnp.ndarray] = None
    node_tags: Optional[jnp.ndarray] = None

    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def n_elems(self) -> int:
        return self.conn.shape[0]

    def element_coords(self) -> jnp.ndarray:
        return self.coords[self.conn]

    # ------------------------------------------------------------------
    # Face patterns must be provided by concrete mesh types.
    def face_node_patterns(self):
        """
        Return a list of tuples, each tuple giving local node indices of a face.
        Override in concrete mesh classes (HexMesh, TetMesh, etc).
        """
        raise NotImplementedError("face_node_patterns must be implemented by mesh subtype")

    # Convenience helpers for boundary tagging / DOF lookup
    def node_indices_where(self, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Return node indices whose coordinates satisfy the predicate.
        predicate: callable that takes coords (np.ndarray of shape (n_nodes, dim)) and returns boolean mask.
        """
        coords_np = np.asarray(self.coords)
        mask = predicate(coords_np)
        return np.nonzero(mask)[0]

    def node_indices_where_point(self, predicate: Callable[[np.ndarray], bool]) -> np.ndarray:
        """
        Return node indices for which predicate(coord) is True.
        predicate: callable accepting a single point (dim,) -> bool
        """
        coords_np = np.asarray(self.coords)
        mask = [bool(predicate(pt)) for pt in coords_np]
        return np.nonzero(mask)[0]

    def axis_extrema_nodes(self, axis: int = 0, side: str = "min", tol: float = 1e-8) -> np.ndarray:
        """
        Nodes lying on min or max of a given axis.
        side: "min" or "max"
        """
        coords_np = np.asarray(self.coords)
        vals = coords_np[:, axis]
        target = vals.min() if side == "min" else vals.max()
        mask = np.isclose(vals, target, atol=tol)
        return np.nonzero(mask)[0]

    def boundary_nodes_bbox(self, tol: float = 1e-8) -> np.ndarray:
        """
        Nodes on the axis-aligned bounding box (min/max in each coordinate).
        Useful for box-shaped meshes like StructuredHexBox.
        """
        coords_np = np.asarray(self.coords)
        mins = coords_np.min(axis=0)
        maxs = coords_np.max(axis=0)
        mask = np.zeros(coords_np.shape[0], dtype=bool)
        for axis in range(coords_np.shape[1]):
            mask |= np.isclose(coords_np[:, axis], mins[axis], atol=tol)
            mask |= np.isclose(coords_np[:, axis], maxs[axis], atol=tol)
        return np.nonzero(mask)[0]

    def node_dofs(self, nodes: Iterable[int], components: Sequence[int] | str = "xyz", dof_per_node: Optional[int] = None) -> np.ndarray:
        """
        Build flattened DOF indices for given node ids.

        components:
            - sequence of component indices (e.g., [0,1,2])
            - or string like "x", "xy", "xyz" (case-insensitive; maps x/y/z -> 0/1/2)
        dof_per_node: optional; inferred from max component index + 1 if not provided.
        """
        nodes_arr = np.asarray(list(nodes), dtype=int)
        if isinstance(components, str):
            comp_map = {"x": 0, "y": 1, "z": 2}
            comps = np.asarray([comp_map[c.lower()] for c in components], dtype=int)
        else:
            comps = np.asarray(list(components), dtype=int)
        inferred = int(comps.max()) + 1 if comps.size else 1
        dofpn = inferred if dof_per_node is None else int(dof_per_node)
        if dofpn <= comps.max():
            raise ValueError(f"dof_per_node={dofpn} is inconsistent with requested component {comps.max()}")
        dofs = [dofpn * int(n) + int(c) for n in nodes_arr for c in comps]
        return np.asarray(dofs, dtype=int)

    def dofs_where(self, predicate: Callable[[np.ndarray], np.ndarray], components: Sequence[int] | str = "xyz", dof_per_node: Optional[int] = None) -> np.ndarray:
        """
        DOF indices for nodes selected by a predicate over all coords.
        predicate takes coords (np.ndarray, shape (n_nodes, dim)) and returns boolean mask.
        """
        nodes = self.node_indices_where(predicate)
        return self.node_dofs(nodes, components=components, dof_per_node=dof_per_node)

    def dofs_where_point(self, predicate: Callable[[np.ndarray], bool], components: Sequence[int] | str = "xyz", dof_per_node: Optional[int] = None) -> np.ndarray:
        """
        DOF indices for nodes selected by a per-point predicate.
        predicate takes a single coord (dim,) -> bool.
        """
        nodes = self.node_indices_where_point(predicate)
        return self.node_dofs(nodes, components=components, dof_per_node=dof_per_node)

    def boundary_dofs_where(self, predicate: Callable[[np.ndarray], np.ndarray], components: Sequence[int] | str = "xyz", dof_per_node: Optional[int] = None) -> np.ndarray:
        """
        Return DOF indices for boundary nodes whose coordinates satisfy predicate.
        predicate takes coords (np.ndarray, shape (n_nodes, dim)) and returns boolean mask.
        """
        coords_np = np.asarray(self.coords)
        mask = np.asarray(predicate(coords_np), dtype=bool)
        bmask = self.boundary_node_mask()
        nodes = np.nonzero(mask & bmask)[0]
        return self.node_dofs(nodes, components=components, dof_per_node=dof_per_node)

    def boundary_dofs_bbox(
        self,
        *,
        components: Sequence[int] | str = "xyz",
        dof_per_node: Optional[int] = None,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        DOF indices on the axis-aligned bounding box (min/max in each coordinate).
        """
        nodes = self.boundary_nodes_bbox(tol=tol)
        return self.node_dofs(nodes, components=components, dof_per_node=dof_per_node)

    def boundary_node_indices(self) -> np.ndarray:
        """
        Return node indices on the boundary based on element face adjacency.
        """
        cached = getattr(self, "_boundary_nodes_cache", None)
        if cached is not None:
            return cached
        conn = np.asarray(self.conn)
        patterns = self.face_node_patterns()
        face_counts: dict[tuple[int, ...], int] = {}
        for elem_conn in conn:
            for pattern in patterns:
                nodes = tuple(sorted(int(elem_conn[i]) for i in pattern))
                face_counts[nodes] = face_counts.get(nodes, 0) + 1
        bnodes: set[int] = set()
        for nodes, count in face_counts.items():
            if count == 1:
                bnodes.update(nodes)
        out = np.asarray(sorted(bnodes), dtype=int)
        setattr(self, "_boundary_nodes_cache", out)
        return out

    def boundary_node_mask(self) -> np.ndarray:
        """
        Return boolean mask for boundary nodes (shape: n_nodes).
        """
        mask: np.ndarray = np.zeros(self.n_nodes, dtype=bool)
        nodes = self.boundary_node_indices()
        mask[nodes] = True
        return mask

    def make_node_tags(self, predicate: Callable[[np.ndarray], np.ndarray], tag: int, base: Optional[np.ndarray] = None) -> jnp.ndarray:
        """
        Build a node_tags array by applying predicate to coords and setting tag where True.
        Returns a jnp.ndarray (int64). Does not mutate the mesh.
        """
        base_tags = np.zeros(self.n_nodes, dtype=NP_INDEX_DTYPE) if base is None else np.asarray(base, dtype=NP_INDEX_DTYPE).copy()
        mask = predicate(np.asarray(self.coords))
        base_tags[mask] = int(tag)
        return jnp.asarray(base_tags, dtype=INDEX_DTYPE)

    def with_node_tags(self, node_tags: np.ndarray | jnp.ndarray):
        """
        Return a new mesh instance with provided node_tags.
        """
        return self.__class__(coords=self.coords, conn=self.conn, cell_tags=self.cell_tags, node_tags=jnp.asarray(node_tags))

    def boundary_facets_where(self, predicate: Callable[[np.ndarray], bool], tag: int | None = None):
        """
        Collect boundary facets whose node coordinates satisfy predicate.

        predicate receives a (n_face_nodes, dim) NumPy array and returns True/False.
        Returns facets (and optional tags if tag is provided).
        """
        coords = np.asarray(self.coords)
        conn = np.asarray(self.conn)
        patterns = self.face_node_patterns()

        facet_map: dict[tuple[int, ...], tuple[list[int], Optional[int]]] = {}

        for elem_conn in conn:
            elem_nodes = coords[elem_conn]
            for pattern in patterns:
                nodes = [int(elem_conn[i]) for i in pattern]
                face_coords = elem_nodes[list(pattern)]
                if not predicate(face_coords):
                    continue
                key = tuple(sorted(nodes))
                if key not in facet_map:
                    facet_map[key] = (nodes, tag)

        if not facet_map:
            if tag is None:
                return jnp.empty((0, len(patterns[0]) if patterns else 0), dtype=INDEX_DTYPE)
            return jnp.empty((0, len(patterns[0]) if patterns else 0), dtype=INDEX_DTYPE), jnp.empty((0,), dtype=INDEX_DTYPE)

        facets = []
        tags = []
        for nodes, t in facet_map.values():
            facets.append(nodes)
            if tag is not None:
                tags.append(t if t is not None else 0)

        facets_arr = jnp.array(facets, dtype=INDEX_DTYPE)
        if tag is None:
            return facets_arr
        return facets_arr, jnp.array(tags, dtype=INDEX_DTYPE)

    def boundary_facets_plane(
        self,
        axis: int = 2,
        value: float = 0.0,
        *,
        tol: float = 1e-8,
        tag: int | None = None,
    ):
        """
        Boundary facets on the plane x[axis] == value (within tol).
        """
        def pred(face: np.ndarray) -> bool:
            return bool(np.allclose(face[:, axis], value, atol=tol))
        return self.boundary_facets_where(pred, tag=tag)

    def facets_on_plane(
        self,
        axis: int = 2,
        value: float = 0.0,
        *,
        tol: float = 1e-8,
        tag: int | None = None,
    ):
        """Alias for boundary_facets_plane (skfem-like naming)."""
        cache = getattr(self, "_boundary_facets_cache", None)
        key = ("plane", int(axis), float(value), float(tol), int(tag) if tag is not None else None)
        if cache is not None and key in cache:
            return cache[key]
        coords = np.asarray(self.coords)
        conn = np.asarray(self.conn)
        patterns = self.face_node_patterns()
        if patterns:
            facets_list = []
            for pattern in patterns:
                face_nodes = conn[:, pattern]
                face_coords = coords[face_nodes]
                mask = np.all(np.isclose(face_coords[..., axis], value, atol=tol), axis=1)
                if np.any(mask):
                    facets_list.append(face_nodes[mask])
            if facets_list:
                facets = np.concatenate(facets_list, axis=0)
                keys = np.sort(facets, axis=1)
                _, idx = np.unique(keys, axis=0, return_index=True)
                facets = facets[np.sort(idx)]
            else:
                facets = np.empty((0, len(patterns[0])), dtype=int)
            facets = jnp.asarray(facets, dtype=INDEX_DTYPE)
            if tag is not None:
                tags = jnp.full((facets.shape[0],), int(tag), dtype=INDEX_DTYPE)
                facets = (facets, tags)
        else:
            facets = self.boundary_facets_plane(axis=axis, value=value, tol=tol, tag=tag)
        if cache is None:
            cache = {}
            setattr(self, "_boundary_facets_cache", cache)
        cache[key] = facets
        return facets

    def boundary_facets_plane_box(
        self,
        axis: int,
        value: float,
        *,
        ranges: Sequence[tuple[float, float] | None] | None = None,
        mode: str = "centroid",
        tol: float = 1e-8,
        tag: int | None = None,
    ):
        """
        Boundary facets on a plane with additional box constraints.

        ranges: sequence of (min, max) or None per axis. The plane axis can be None.
        mode: "centroid" checks the face centroid, "all" requires all vertices inside.
        """
        dim = int(np.asarray(self.coords).shape[1])
        if ranges is None:
            ranges = [None] * dim
        if len(ranges) != dim:
            raise ValueError("ranges must have length equal to mesh dimension.")
        if mode not in ("centroid", "all"):
            raise ValueError("mode must be 'centroid' or 'all'.")

        def pred(face: np.ndarray) -> bool:
            if not np.allclose(face[:, axis], value, atol=tol):
                return False
            pts = face.mean(axis=0)[None, :] if mode == "centroid" else face
            for ax, bounds in enumerate(ranges):
                if bounds is None or ax == axis:
                    continue
                lo, hi = bounds
                if np.any(pts[:, ax] < lo - tol) or np.any(pts[:, ax] > hi + tol):
                    return False
            return True

        return self.boundary_facets_where(pred, tag=tag)

    def facets_on_plane_box(
        self,
        axis: int,
        value: float,
        *,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        ranges: Sequence[tuple[float, float] | None] | None = None,
        mode: str = "centroid",
        tol: float = 1e-8,
        tag: int | None = None,
    ):
        """
        Alias for boundary_facets_plane_box with axis-aligned range helpers.
        Provide x/y/z or a full ranges sequence.
        """
        dim = int(np.asarray(self.coords).shape[1])
        if ranges is None:
            ranges = [None] * dim
            if dim > 0:
                ranges[0] = x
            if dim > 1:
                ranges[1] = y
            if dim > 2:
                ranges[2] = z
        cache = getattr(self, "_boundary_facets_cache", None)
        ranges_key = tuple(ranges)
        key = (
            "plane_box",
            int(axis),
            float(value),
            ranges_key,
            str(mode),
            float(tol),
            int(tag) if tag is not None else None,
        )
        if cache is not None and key in cache:
            return cache[key]
        coords = np.asarray(self.coords)
        conn = np.asarray(self.conn)
        patterns = self.face_node_patterns()
        if patterns:
            facets_list = []
            for pattern in patterns:
                face_nodes = conn[:, pattern]
                face_coords = coords[face_nodes]
                mask = np.all(np.isclose(face_coords[..., axis], value, atol=tol), axis=1)
                if np.any(mask):
                    if mode == "centroid":
                        pts = face_coords[mask].mean(axis=1)
                        mask_local = np.ones(pts.shape[0], dtype=bool)
                        for ax, bounds in enumerate(ranges):
                            if bounds is None or ax == axis:
                                continue
                            lo, hi = bounds
                            mask_local &= (pts[:, ax] >= lo - tol) & (pts[:, ax] <= hi + tol)
                        face_nodes = face_nodes[mask][mask_local]
                    else:
                        face_coords = face_coords[mask]
                        mask_local = np.ones(face_coords.shape[0], dtype=bool)
                        for ax, bounds in enumerate(ranges):
                            if bounds is None or ax == axis:
                                continue
                            lo, hi = bounds
                            in_range = (face_coords[..., ax] >= lo - tol) & (face_coords[..., ax] <= hi + tol)
                            mask_local &= np.all(in_range, axis=1)
                        face_nodes = face_nodes[mask][mask_local]
                    if face_nodes.size:
                        facets_list.append(face_nodes)
            if facets_list:
                facets = np.concatenate(facets_list, axis=0)
                keys = np.sort(facets, axis=1)
                _, idx = np.unique(keys, axis=0, return_index=True)
                facets = facets[np.sort(idx)]
            else:
                facets = np.empty((0, len(patterns[0])), dtype=int)
            facets = jnp.asarray(facets, dtype=INDEX_DTYPE)
            if tag is not None:
                tags = jnp.full((facets.shape[0],), int(tag), dtype=INDEX_DTYPE)
                facets = (facets, tags)
        else:
            facets = self.boundary_facets_plane_box(
                axis=axis,
                value=value,
                ranges=ranges,
                mode=mode,
                tol=tol,
                tag=tag,
            )
        if cache is None:
            cache = {}
            setattr(self, "_boundary_facets_cache", cache)
        cache[key] = facets
        return facets

    def boundary_dofs_plane(
        self,
        axis: int = 2,
        value: float = 0.0,
        *,
        components: Sequence[int] | str = "xyz",
        dof_per_node: Optional[int] = None,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        DOF indices for boundary nodes on the plane x[axis] == value (within tol).
        """
        def pred(coords: np.ndarray) -> np.ndarray:
            return np.isclose(coords[:, axis], value, atol=tol)
        return self.boundary_dofs_where(pred, components=components, dof_per_node=dof_per_node)

    def elements_touching_nodes(self, nodes: Iterable[int]) -> np.ndarray:
        """
        Return element indices that touch any node in the provided set.
        """
        nodes_arr = np.asarray(list(nodes), dtype=int)
        if nodes_arr.size == 0:
            return np.asarray([], dtype=int)
        mark: np.ndarray = np.zeros(self.n_nodes, dtype=bool)
        mark[nodes_arr] = True
        conn = np.asarray(self.conn)
        return np.nonzero(np.any(mark[conn], axis=1))[0]

    def nodes_from_facets(self, facets: np.ndarray | jnp.ndarray) -> np.ndarray:
        """
        Return unique node indices contained in the given facets array.
        """
        facets_arr = np.asarray(facets, dtype=int)
        if facets_arr.size == 0:
            return np.asarray([], dtype=int)
        return np.unique(facets_arr.reshape(-1))

    def elements_from_nodes(self, nodes: Iterable[int]) -> np.ndarray:
        """
        Alias for elements_touching_nodes (skfem-like naming).
        """
        return self.elements_touching_nodes(nodes)

    def elements_from_facets(self, facets: np.ndarray | jnp.ndarray, *, mode: str = "touching") -> np.ndarray:
        """
        Return element indices associated with facets.

        mode:
            - "touching": any element sharing at least one facet node.
            - "adjacent": elements that own the facet (exact face match).
        """
        if mode not in ("touching", "adjacent"):
            raise ValueError("mode must be 'touching' or 'adjacent'.")
        facets_arr = np.asarray(facets, dtype=int)
        if facets_arr.size == 0:
            return np.asarray([], dtype=int)
        if mode == "touching":
            nodes = self.nodes_from_facets(facets_arr)
            return self.elements_touching_nodes(nodes)

        patterns = self.face_node_patterns()
        facet_keys = {tuple(sorted(face)) for face in facets_arr}
        conn = np.asarray(self.conn)
        elems = []
        for e_idx, elem_conn in enumerate(conn):
            for pattern in patterns:
                face_nodes = tuple(sorted(int(elem_conn[i]) for i in pattern))
                if face_nodes in facet_keys:
                    elems.append(e_idx)
                    break
        return np.asarray(elems, dtype=int)

    def elements_touching_facets(self, facets: np.ndarray | jnp.ndarray) -> np.ndarray:
        """
        Return element indices that touch any node in the provided facets.
        """
        facets_arr = np.asarray(facets, dtype=int)
        if facets_arr.size == 0:
            return np.asarray([], dtype=int)
        nodes = np.unique(facets_arr.reshape(-1))
        return self.elements_touching_nodes(nodes)

    def surface_from_facets(self, facets, *, facet_tags=None):
        """
        Build a SurfaceMesh from facet connectivity.
        """
        from .surface import SurfaceMesh
        return SurfaceMesh.from_facets(self.coords, facets, facet_tags=facet_tags, node_tags=self.node_tags)

    def surface_with_elem_conn_from_facets(self, facets, *, mode: str = "touching"):
        """
        Build a SurfaceMesh and matching elem_conn for the given facets.
        """
        from .surface import surface_with_elem_conn
        return surface_with_elem_conn(self, facets, mode=mode)

    def surface_with_facet_map_from_facets(self, facets):
        """
        Build a SurfaceMesh and facet-to-element map for the given facets.
        """
        surface = self.surface_from_facets(facets)
        conn = np.asarray(self.conn, dtype=int)
        from .mortar import map_surface_facets_to_tet_elements, map_surface_facets_to_hex_elements
        if conn.shape[1] in {4, 10}:
            facet_map = map_surface_facets_to_tet_elements(surface, conn)
        elif conn.shape[1] in {8, 20, 27}:
            facet_map = map_surface_facets_to_hex_elements(surface, conn)
        else:
            raise NotImplementedError("elem_conn must be tet/hex (4/10/8/20/27)")
        return SurfaceWithFacetMap(surface=surface, facet_map=facet_map)


@jax.tree_util.register_pytree_node_class
class BaseMeshPytree(BaseMeshClosure):
    """BaseMesh variant that registers as a JAX pytree."""
    def tree_flatten(self):
        children = (self.coords, self.conn, self.cell_tags, self.node_tags)
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        coords, conn, cell_tags, node_tags = children
        return cls(coords, conn, cell_tags, node_tags)


BaseMesh = BaseMeshClosure
@dataclass(frozen=True)
class SurfaceWithFacetMap:
    surface: object
    facet_map: np.ndarray

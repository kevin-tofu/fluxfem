from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence
import numpy as np
import jax
import jax.numpy as jnp


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
        bnodes = set()
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
        mask = np.zeros(self.n_nodes, dtype=bool)
        nodes = self.boundary_node_indices()
        mask[nodes] = True
        return mask

    def make_node_tags(self, predicate: Callable[[np.ndarray], np.ndarray], tag: int, base: Optional[np.ndarray] = None) -> jnp.ndarray:
        """
        Build a node_tags array by applying predicate to coords and setting tag where True.
        Returns a jnp.ndarray (int32). Does not mutate the mesh.
        """
        base_tags = np.zeros(self.n_nodes, dtype=np.int32) if base is None else np.asarray(base, dtype=np.int32).copy()
        mask = predicate(np.asarray(self.coords))
        base_tags[mask] = int(tag)
        return jnp.asarray(base_tags, dtype=jnp.int32)

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
                return jnp.empty((0, len(patterns[0]) if patterns else 0), dtype=jnp.int32)
            return jnp.empty((0, len(patterns[0]) if patterns else 0), dtype=jnp.int32), jnp.empty((0,), dtype=jnp.int32)

        facets = []
        tags = []
        for nodes, t in facet_map.values():
            facets.append(nodes)
            if tag is not None:
                tags.append(t if t is not None else 0)

        facets_arr = jnp.array(facets, dtype=jnp.int32)
        if tag is None:
            return facets_arr
        return facets_arr, jnp.array(tags, dtype=jnp.int32)


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

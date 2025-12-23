from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np

DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

from .base import BaseMesh, BaseMeshPytree
from .hex import HexMesh, HexMeshPytree


def _polygon_area(pts: np.ndarray) -> float:
    """
    Polygon area in 3D by fan triangulation (works for tri/quad faces).
    Assumes points are planar and ordered.
    """
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
class SurfaceMesh(BaseMesh):
    """
    Simple boundary mesh made of facets (tri/quad) that live in the volume mesh nodes.
    Uses BaseMesh.conn to store facets.
    """

    facet_tags: Optional[jnp.ndarray] = None

    def __post_init__(self):
        # Keep facet_tags mirrored in cell_tags for BaseMesh compat.
        if self.cell_tags is None and self.facet_tags is not None:
            self.cell_tags = self.facet_tags
        if self.facet_tags is None and self.cell_tags is not None:
            self.facet_tags = self.cell_tags

    @classmethod
    def from_facets(
        cls,
        coords: jnp.ndarray,
        facets: jnp.ndarray,
        facet_tags: Optional[jnp.ndarray] = None,
        node_tags: Optional[jnp.ndarray] = None,
    ) -> "SurfaceMesh":
        coords_j = jnp.asarray(coords, dtype=DTYPE)
        facets_j = jnp.asarray(facets, dtype=jnp.int32)
        tags_j = None if facet_tags is None else jnp.asarray(facet_tags, dtype=jnp.int32)
        node_tags_j = None if node_tags is None else jnp.asarray(node_tags)
        return cls(coords=coords_j, conn=facets_j, cell_tags=tags_j, node_tags=node_tags_j, facet_tags=tags_j)

    @classmethod
    def from_hex_mesh(
        cls,
        mesh: HexMesh,
        facets: jnp.ndarray,
        facet_tags: Optional[jnp.ndarray] = None,
    ) -> "SurfaceMesh":
        """
        Build a surface mesh that reuses the volume mesh coordinates.
        Facets must reference the volume mesh node numbering.
        """
        return cls.from_facets(mesh.coords, facets, facet_tags=facet_tags, node_tags=mesh.node_tags)

    @property
    def n_facets(self) -> int:
        return self.n_elems

    def facet_areas(self) -> np.ndarray:
        """Return per-facet area (uses NumPy for simplicity)."""
        coords = np.asarray(self.coords)
        facets = np.asarray(self.conn, dtype=int)
        areas = np.zeros(facets.shape[0], dtype=float)
        for i, nodes in enumerate(facets):
            pts = coords[nodes]
            areas[i] = _polygon_area(pts)
        return areas

    def select_by_tag(self, tag: int) -> "SurfaceMesh":
        """Return a new SurfaceMesh containing only facets with given tag."""
        if self.facet_tags is None:
            raise ValueError("facet_tags not set on this SurfaceMesh")
        mask = np.asarray(self.facet_tags) == tag
        return SurfaceMesh.from_facets(
            self.coords,
            self.conn[mask],
            facet_tags=self.facet_tags[mask],
            node_tags=self.node_tags,
        )

    def facet_normals(self, *, outward_from=None, normalize: bool = True) -> np.ndarray:
        from ..solver.bc import facet_normals
        return facet_normals(self, outward_from=outward_from, normalize=normalize)

    def assemble_load(self, load, *, dim: int, n_total_nodes: int | None = None, F0=None):
        from ..solver.bc import assemble_surface_load
        return assemble_surface_load(self, load, dim=dim, n_total_nodes=n_total_nodes, F0=F0)

    def assemble_linear_form(self, form, params, *, dim: int, n_total_nodes: int | None = None, F0=None):
        from ..solver.bc import assemble_surface_linear_form
        return assemble_surface_linear_form(self, form, params, dim=dim, n_total_nodes=n_total_nodes, F0=F0)

    def assemble_linear_form_on_space(self, space, form, params, *, F0=None):
        """
        Assemble surface linear form using global size inferred from a volume space.
        """
        dim = int(getattr(space, "value_dim", 1))
        n_total_nodes = int(getattr(space, "mesh", self).n_nodes)
        return self.assemble_linear_form(form, params, dim=dim, n_total_nodes=n_total_nodes, F0=F0)

    def assemble_traction(
        self,
        traction,
        *,
        dim: int = 3,
        n_total_nodes: int | None = None,
        F0=None,
        outward_from=None,
    ):
        from ..solver.bc import assemble_surface_traction
        return assemble_surface_traction(
            self,
            traction,
            dim=dim,
            n_total_nodes=n_total_nodes,
            F0=F0,
            outward_from=outward_from,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class SurfaceMeshPytree(BaseMeshPytree):
    """
    Simple boundary mesh made of facets (tri/quad) that live in the volume mesh nodes.
    Uses BaseMesh.conn to store facets.
    """

    facet_tags: Optional[jnp.ndarray] = None

    def __post_init__(self):
        if self.cell_tags is None and self.facet_tags is not None:
            self.cell_tags = self.facet_tags
        if self.facet_tags is None and self.cell_tags is not None:
            self.facet_tags = self.cell_tags

    @classmethod
    def from_facets(
        cls,
        coords: jnp.ndarray,
        facets: jnp.ndarray,
        facet_tags: Optional[jnp.ndarray] = None,
        node_tags: Optional[jnp.ndarray] = None,
    ) -> "SurfaceMeshPytree":
        coords_j = jnp.asarray(coords, dtype=DTYPE)
        facets_j = jnp.asarray(facets, dtype=jnp.int32)
        tags_j = None if facet_tags is None else jnp.asarray(facet_tags, dtype=jnp.int32)
        node_tags_j = None if node_tags is None else jnp.asarray(node_tags)
        return cls(coords=coords_j, conn=facets_j, cell_tags=tags_j, node_tags=node_tags_j, facet_tags=tags_j)

    @classmethod
    def from_hex_mesh(
        cls,
        mesh: HexMesh | HexMeshPytree,
        facets: jnp.ndarray,
        facet_tags: Optional[jnp.ndarray] = None,
    ) -> "SurfaceMeshPytree":
        return cls.from_facets(mesh.coords, facets, facet_tags=facet_tags, node_tags=mesh.node_tags)

    @property
    def n_facets(self) -> int:
        return self.n_elems

    def facet_areas(self) -> np.ndarray:
        coords = np.asarray(self.coords)
        facets = np.asarray(self.conn, dtype=int)
        areas = np.zeros(facets.shape[0], dtype=float)
        for i, nodes in enumerate(facets):
            pts = coords[nodes]
            areas[i] = _polygon_area(pts)
        return areas

    def select_by_tag(self, tag: int) -> "SurfaceMeshPytree":
        if self.facet_tags is None:
            raise ValueError("facet_tags not set on this SurfaceMesh")
        mask = np.asarray(self.facet_tags) == tag
        return SurfaceMeshPytree.from_facets(
            self.coords,
            self.conn[mask],
            facet_tags=self.facet_tags[mask],
            node_tags=self.node_tags,
        )

    def facet_normals(self, *, outward_from=None, normalize: bool = True) -> np.ndarray:
        from ..solver.bc import facet_normals
        return facet_normals(self, outward_from=outward_from, normalize=normalize)

    def assemble_load(self, load, *, dim: int, n_total_nodes: int | None = None, F0=None):
        from ..solver.bc import assemble_surface_load
        return assemble_surface_load(self, load, dim=dim, n_total_nodes=n_total_nodes, F0=F0)

    def assemble_linear_form(self, form, params, *, dim: int, n_total_nodes: int | None = None, F0=None):
        from ..solver.bc import assemble_surface_linear_form
        return assemble_surface_linear_form(self, form, params, dim=dim, n_total_nodes=n_total_nodes, F0=F0)

    def assemble_linear_form_on_space(self, space, form, params, *, F0=None):
        """
        Assemble surface linear form using global size inferred from a volume space.
        """
        dim = int(getattr(space, "value_dim", 1))
        n_total_nodes = int(getattr(space, "mesh", self).n_nodes)
        return self.assemble_linear_form(form, params, dim=dim, n_total_nodes=n_total_nodes, F0=F0)

    def assemble_traction(
        self,
        traction,
        *,
        dim: int = 3,
        n_total_nodes: int | None = None,
        F0=None,
        outward_from=None,
    ):
        from ..solver.bc import assemble_surface_traction
        return assemble_surface_traction(
            self,
            traction,
            dim=dim,
            n_total_nodes=n_total_nodes,
            F0=F0,
            outward_from=outward_from,
        )

    def assemble_flux(
        self,
        flux,
        *,
        n_total_nodes: int | None = None,
        F0=None,
        outward_from=None,
    ):
        from ..solver.bc import assemble_surface_flux
        return assemble_surface_flux(
            self,
            flux,
            n_total_nodes=n_total_nodes,
            F0=F0,
            outward_from=outward_from,
        )

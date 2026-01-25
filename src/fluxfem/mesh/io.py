from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

from .dtypes import NP_INDEX_DTYPE

DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

try:
    import meshio
except Exception as e:  # pragma: no cover
    meshio = None
    meshio_import_error: Optional[Exception] = e
else:
    meshio_import_error = None

from .hex import HexMesh
from .tet import TetMesh
from .surface import SurfaceMesh


def load_gmsh_mesh(path: str):
    """
    Load a Gmsh .msh (v2/v4) file containing hex or tet elements (and optional boundary facets).

    Returns:
      mesh: HexMesh or TetMesh
      facets: (n_facets, 3 or 4) or None
      facet_tags: (n_facets,) or None  (gmsh physical tags if present)
    """
    if meshio is None:
        raise ImportError(f"meshio is required to load gmsh meshes: {meshio_import_error}")

    msh = meshio.read(path)
    coords = np.asarray(msh.points[:, :3], dtype=DTYPE)

    mesh: HexMesh | TetMesh | None = None
    if "hexahedron" in msh.cells_dict:
        conn = np.asarray(msh.cells_dict["hexahedron"], dtype=NP_INDEX_DTYPE)
        mesh = HexMesh(jnp.asarray(coords), jnp.asarray(conn))
    elif "tetra" in msh.cells_dict:
        conn = np.asarray(msh.cells_dict["tetra"], dtype=NP_INDEX_DTYPE)
        mesh = TetMesh(jnp.asarray(coords), jnp.asarray(conn))
    else:
        raise ValueError("gmsh mesh does not contain hexahedron or tetra cells")

    facets = None
    facet_tags = None
    # surface facets (quad/tri)
    cell_dict = msh.cells_dict
    for fkey in ("quad", "triangle", "tri"):
        if fkey in cell_dict:
            facets = np.asarray(cell_dict[fkey], dtype=NP_INDEX_DTYPE)
            break
    tags_raw = None
    if facets is not None:
        if "gmsh:physical" in msh.cell_data_dict:
            tags_raw = msh.cell_data_dict["gmsh:physical"].get(fkey, None)
        else:
            if "gmsh:physical" in msh.cell_data:
                for block, data in zip(msh.cells, msh.cell_data["gmsh:physical"]):
                    if block.type == fkey:
                        tags_raw = data
                        break
        if tags_raw is not None:
            facet_tags = np.asarray(tags_raw, dtype=NP_INDEX_DTYPE)

    return mesh, facets, facet_tags


def load_gmsh_hex_mesh(path: str):
    """Load a Gmsh mesh and return a HexMesh with optional facets/tags."""
    mesh, facets, tags = load_gmsh_mesh(path)
    if not isinstance(mesh, HexMesh):
        raise ValueError("gmsh mesh is not hexahedral")
    return mesh, facets, tags


def load_gmsh_tet_mesh(path: str):
    """Load a Gmsh mesh and return a TetMesh with optional facets/tags."""
    mesh, facets, tags = load_gmsh_mesh(path)
    if not isinstance(mesh, TetMesh):
        raise ValueError("gmsh mesh is not tetrahedral")
    return mesh, facets, tags


def make_surface_from_facets(coords: np.ndarray, facets: np.ndarray, tags=None) -> SurfaceMesh:
    """Helper to build SurfaceMesh from raw coords/facets (optional tags)."""
    return SurfaceMesh.from_facets(coords, facets, facet_tags=tags)

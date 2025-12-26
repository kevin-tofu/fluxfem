

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Callable
import jax
import jax.numpy as jnp
from ..core.dtypes import DEFAULT_DTYPE
import numpy as np

from .base import BaseMesh, BaseMeshPytree


@dataclass
class HexMesh(BaseMesh):
    """
    Structured / unstructured hex mesh (8-node linear hex elements).
    
    coords: (n_nodes, 3) float32
    conn:   (n_elems, 8) int32  # node indices of each element
    """
    coords: jnp.ndarray  # shape (n_nodes, 3)
    conn: jnp.ndarray    # shape (n_elems, 8), int32

    def face_node_patterns(self):
        return [
            (0, 1, 2, 3),  # -z
            (4, 5, 6, 7),  # +z
            (0, 1, 5, 4),  # -y
            (3, 2, 6, 7),  # +y
            (0, 4, 7, 3),  # -x
            (1, 2, 6, 5),  # +x
        ]


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class HexMeshPytree(BaseMeshPytree):
    """Hex mesh registered as a JAX pytree."""
    def face_node_patterns(self):
        return [
            (0, 1, 2, 3),  # -z
            (4, 5, 6, 7),  # +z
            (0, 1, 5, 4),  # -y
            (3, 2, 6, 7),  # +y
            (0, 4, 7, 3),  # -x
            (1, 2, 6, 5),  # +x
        ]


def tag_axis_minmax_facets(
    mesh: HexMesh,
    axis: int = 0,
    dirichlet_tag: int = 1,
    neumann_tag: int = 2,
    tol: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Tag boundary facets on min/max of the given axis.

    Returns:
        facets: (n_facets, 4) int32, quad node ids
        facet_tags: (n_facets,) int32, dirichlet_tag on min side, neumann_tag on max side
    """
    coords = np.asarray(mesh.coords)
    conn = np.asarray(mesh.conn)

    axis_vals = coords[:, axis]
    v_min = float(axis_vals.min())
    v_max = float(axis_vals.max())

    face_patterns: List[Tuple[int, int, int, int]] = [
        (0, 1, 2, 3),  # -z
        (4, 5, 6, 7),  # +z
        (0, 1, 5, 4),  # -y
        (3, 2, 6, 7),  # +y
        (0, 4, 7, 3),  # -x
        (1, 2, 6, 5),  # +x
    ]

    facet_map: Dict[Tuple[int, ...], Tuple[List[int], int]] = {}

    for elem_conn in conn:
        elem_nodes = coords[elem_conn]
        for pattern in face_patterns:
            nodes = [int(elem_conn[i]) for i in pattern]
            vals = elem_nodes[list(pattern), axis]
            tag = None
            if np.allclose(vals, v_min, atol=tol):
                tag = dirichlet_tag
            elif np.allclose(vals, v_max, atol=tol):
                tag = neumann_tag

            if tag is None:
                continue

            key = tuple(sorted(nodes))
            if key not in facet_map:
                facet_map[key] = (nodes, tag)

    if not facet_map:
        return jnp.empty((0, 4), dtype=jnp.int32), jnp.empty((0,), dtype=jnp.int32)

    facets = []
    tags = []
    for nodes, tag in facet_map.values():
        facets.append(nodes)
        tags.append(tag)

    return jnp.array(facets, dtype=jnp.int32), jnp.array(tags, dtype=jnp.int32)


@dataclass
class StructuredHexBox:
    """
    Uniform hex mesh generator on a box, returned as an unstructured HexMesh.
    """
    nx: int
    ny: int
    nz: int
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    order: int = 1  # 1: 8-node Hex, 2: 20-node serendipity Hex, 3: 27-node triquadratic Hex

    def build(self) -> HexMesh:
        """
        Build a regular grid of nx×ny×nz elements over [origin, origin + (lx, ly, lz)].
        order=1 → 8-node Hex, order=2 → 20-node serendipity Hex。
        """
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("nx, ny, nz must be positive")
        if self.order not in (1, 2, 3):
            raise ValueError("order must be 1, 2, or 3")

        ox, oy, oz = self.origin
        xs = jnp.linspace(ox, ox + self.lx, self.nx + 1, dtype=DEFAULT_DTYPE)
        ys = jnp.linspace(oy, oy + self.ly, self.ny + 1, dtype=DEFAULT_DTYPE)
        zs = jnp.linspace(oz, oz + self.lz, self.nz + 1, dtype=DEFAULT_DTYPE)

        if self.order == 1:
            return self._build_hex8(xs, ys, zs)
        if self.order == 2:
            return self._build_hex20(xs, ys, zs)
        return self._build_hex27(xs, ys, zs)

    def _build_hex8(self, xs, ys, zs) -> HexMesh:
        coords_list = []
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    coords_list.append([xs[i], ys[j], zs[k]])
        coords = jnp.array(coords_list, dtype=DEFAULT_DTYPE)

        def node_id(i: int, j: int, k: int) -> int:
            return k * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + i

        conn_list = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n000 = node_id(i,     j,     k)
                    n100 = node_id(i + 1, j,     k)
                    n110 = node_id(i + 1, j + 1, k)
                    n010 = node_id(i,     j + 1, k)
                    n001 = node_id(i,     j,     k + 1)
                    n101 = node_id(i + 1, j,     k + 1)
                    n111 = node_id(i + 1, j + 1, k + 1)
                    n011 = node_id(i,     j + 1, k + 1)
                    conn_list.append([n000, n100, n110, n010, n001, n101, n111, n011])

        conn = jnp.array(conn_list, dtype=jnp.int32)
        return HexMesh(coords=coords, conn=conn)

    def _build_hex20(self, xs, ys, zs) -> HexMesh:
        """
        Build 20-node serendipity Hex mesh (corner + edge midpoints).
        """
        coords_list: List[List[float]] = []
        node_map: Dict[Tuple[float, float, float], int] = {}

        def add_node(pt: np.ndarray) -> int:
            key = tuple(np.round(pt.astype(np.float64), 8))
            if key not in node_map:
                node_map[key] = len(coords_list)
                coords_list.append([float(pt[0]), float(pt[1]), float(pt[2])])
            return node_map[key]

        # pre-create corner nodes
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    add_node(np.array([xs[i], ys[j], zs[k]], dtype=np.float64))

        conn_list = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    p000 = np.array([xs[i],     ys[j],     zs[k]],     dtype=np.float64)
                    p100 = np.array([xs[i + 1], ys[j],     zs[k]],     dtype=np.float64)
                    p110 = np.array([xs[i + 1], ys[j + 1], zs[k]],     dtype=np.float64)
                    p010 = np.array([xs[i],     ys[j + 1], zs[k]],     dtype=np.float64)
                    p001 = np.array([xs[i],     ys[j],     zs[k + 1]], dtype=np.float64)
                    p101 = np.array([xs[i + 1], ys[j],     zs[k + 1]], dtype=np.float64)
                    p111 = np.array([xs[i + 1], ys[j + 1], zs[k + 1]], dtype=np.float64)
                    p011 = np.array([xs[i],     ys[j + 1], zs[k + 1]], dtype=np.float64)

                    c000 = add_node(p000)
                    c100 = add_node(p100)
                    c110 = add_node(p110)
                    c010 = add_node(p010)
                    c001 = add_node(p001)
                    c101 = add_node(p101)
                    c111 = add_node(p111)
                    c011 = add_node(p011)

                    # edge midpoints
                    e01 = add_node(0.5 * (p000 + p100))
                    e12 = add_node(0.5 * (p100 + p110))
                    e23 = add_node(0.5 * (p110 + p010))
                    e30 = add_node(0.5 * (p010 + p000))

                    e45 = add_node(0.5 * (p001 + p101))
                    e56 = add_node(0.5 * (p101 + p111))
                    e67 = add_node(0.5 * (p111 + p011))
                    e74 = add_node(0.5 * (p011 + p001))

                    e04 = add_node(0.5 * (p000 + p001))
                    e15 = add_node(0.5 * (p100 + p101))
                    e26 = add_node(0.5 * (p110 + p111))
                    e37 = add_node(0.5 * (p010 + p011))

                    conn_list.append(
                        [
                            c000, c100, c110, c010,  # corners
                            c001, c101, c111, c011,
                            e01, e12, e23, e30,     # edge mids
                            e45, e56, e67, e74,
                            e04, e15, e26, e37,
                        ]
                    )

        coords = jnp.array(coords_list, dtype=DEFAULT_DTYPE)
        conn = jnp.array(conn_list, dtype=jnp.int32)
        return HexMesh(coords=coords, conn=conn)

    def _build_hex27(self, xs, ys, zs) -> HexMesh:
        """
        Build 27-node triquadratic Hex mesh (tensor-product nodes: corners, edge mids, face centers, body center).
        """
        coords_list: List[List[float]] = []
        node_map: Dict[Tuple[float, float, float], int] = {}

        def add_node(pt: np.ndarray) -> int:
            key = tuple(np.round(pt.astype(np.float64), 8))
            if key not in node_map:
                node_map[key] = len(coords_list)
                coords_list.append([float(pt[0]), float(pt[1]), float(pt[2])])
            return node_map[key]

        def mid(a, b):
            return 0.5 * (a + b)

        # pre-create grid nodes at original vertices and midpoints on axes
        xs_mid = [mid(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
        ys_mid = [mid(ys[j], ys[j + 1]) for j in range(len(ys) - 1)]
        zs_mid = [mid(zs[k], zs[k + 1]) for k in range(len(zs) - 1)]

        # create all possible nodes (vertices + edge mids + face centers + cell centers)
        for k_idx, zk in enumerate(zs):
            for j_idx, yj in enumerate(ys):
                for i_idx, xi in enumerate(xs):
                    add_node(np.array([xi, yj, zk], dtype=np.float64))
        for k_idx, zk in enumerate(zs):
            for j_idx, yj in enumerate(ys):
                for i_mid in xs_mid:
                    add_node(np.array([i_mid, yj, zk], dtype=np.float64))
        for k_idx, zk in enumerate(zs):
            for i_idx, xi in enumerate(xs):
                for j_mid in ys_mid:
                    add_node(np.array([xi, j_mid, zk], dtype=np.float64))
        for j_idx, yj in enumerate(ys):
            for i_idx, xi in enumerate(xs):
                for k_mid in zs_mid:
                    add_node(np.array([xi, yj, k_mid], dtype=np.float64))
        # face centers
        for k_idx, zk in enumerate(zs):
            for j_mid in ys_mid:
                for i_mid in xs_mid:
                    add_node(np.array([i_mid, j_mid, zk], dtype=np.float64))
        for k_mid in zs_mid:
            for j_idx, yj in enumerate(ys):
                for i_mid in xs_mid:
                    add_node(np.array([i_mid, yj, k_mid], dtype=np.float64))
        for k_mid in zs_mid:
            for j_mid in ys_mid:
                for i_idx, xi in enumerate(xs):
                    add_node(np.array([xi, j_mid, k_mid], dtype=np.float64))
        # cell centers (unique per cell)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    cx = mid(xs[i], xs[i + 1])
                    cy = mid(ys[j], ys[j + 1])
                    cz = mid(zs[k], zs[k + 1])
                    add_node(np.array([cx, cy, cz], dtype=np.float64))

        conn_list = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    x_vals = [xs[i], mid(xs[i], xs[i + 1]), xs[i + 1]]
                    y_vals = [ys[j], mid(ys[j], ys[j + 1]), ys[j + 1]]
                    z_vals = [zs[k], mid(zs[k], zs[k + 1]), zs[k + 1]]

                    nodes = []
                    for kk in range(3):
                        for jj in range(3):
                            for ii in range(3):
                                nodes.append(add_node(np.array([x_vals[ii], y_vals[jj], z_vals[kk]], dtype=np.float64)))

                    # order in lexicographic k,j,i -> length 27
                    conn_list.append(nodes)

        coords = jnp.array(coords_list, dtype=DEFAULT_DTYPE)
        conn = jnp.array(conn_list, dtype=jnp.int32)
        return HexMesh(coords=coords, conn=conn)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np

DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


from .base import BaseMesh, BaseMeshPytree


@dataclass
class TetMesh(BaseMesh):
    """Unstructured tetra mesh."""
    def face_node_patterns(self):
        # 4-node faces of a tet
        return [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ]


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class TetMeshPytree(BaseMeshPytree):
    """Unstructured tetra mesh (pytree)."""
    def face_node_patterns(self):
        return [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ]


@dataclass
class StructuredTetBox:
    """Regular grid subdivided into 5 tets per cube (simplex) in a structured layout."""

    nx: int
    ny: int
    nz: int
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    order: int = 1  # 1: 4-node, 2: 10-node (edge mids)

    def _fix_orientation(self, coords: np.ndarray, conn: np.ndarray) -> np.ndarray:
        conn = conn.copy()
        for idx, tet in enumerate(conn):
            p = coords[tet]
            p0, p1, p2, p3 = p[0], p[1], p[2], p[3]  # first 4 are corners
            J = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=1)
            if np.linalg.det(J) < 0:
                tet[[1, 2]] = tet[[2, 1]]  # swap corner1/corner2
                conn[idx] = tet
        return conn

    def _midpoint(self, a, b):
        return 0.5 * (a + b)

    def _build_tet10(self, xs, ys, zs) -> TetMesh:
        coords_list: list[list[float]] = []
        node_map: dict[tuple[float, float, float], int] = {}

        def add_node(pt: np.ndarray) -> int:
            key = tuple(np.round(pt.astype(np.float64), 10))
            if key not in node_map:
                node_map[key] = len(coords_list)
                coords_list.append([float(pt[0]), float(pt[1]), float(pt[2])])
            return node_map[key]

        # corners
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    add_node(np.array([xs[i], ys[j], zs[k]], dtype=np.float64))

        conn_list = []
        mid = self._midpoint
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    p000 = np.array([xs[i], ys[j], zs[k]], dtype=np.float64)
                    p100 = np.array([xs[i + 1], ys[j], zs[k]], dtype=np.float64)
                    p010 = np.array([xs[i], ys[j + 1], zs[k]], dtype=np.float64)
                    p110 = np.array([xs[i + 1], ys[j + 1], zs[k]], dtype=np.float64)
                    p001 = np.array([xs[i], ys[j], zs[k + 1]], dtype=np.float64)
                    p101 = np.array([xs[i + 1], ys[j], zs[k + 1]], dtype=np.float64)
                    p011 = np.array([xs[i], ys[j + 1], zs[k + 1]], dtype=np.float64)
                    p111 = np.array([xs[i + 1], ys[j + 1], zs[k + 1]], dtype=np.float64)

                    corners = [
                        (p000, p100, p010, p001),
                        (p100, p110, p010, p111),
                        (p100, p010, p001, p111),
                        (p100, p001, p101, p111),
                        (p010, p001, p011, p111),
                    ]

                    for p0, p1, p2, p3 in corners:
                        n0 = add_node(p0)
                        n1 = add_node(p1)
                        n2 = add_node(p2)
                        n3 = add_node(p3)
                        # edge midpoints
                        n01 = add_node(mid(p0, p1))
                        n02 = add_node(mid(p0, p2))
                        n03 = add_node(mid(p0, p3))
                        n12 = add_node(mid(p1, p2))
                        n13 = add_node(mid(p1, p3))
                        n23 = add_node(mid(p2, p3))
                        conn_list.append([n0, n1, n2, n3, n01, n02, n03, n12, n13, n23])

        coords = np.asarray(coords_list, dtype=DTYPE)
        conn = np.asarray(conn_list, dtype=np.int32)
        conn = self._fix_orientation(coords, conn)
        return TetMesh(coords=jnp.array(coords), conn=jnp.array(conn))

    def build(self) -> TetMesh:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("nx, ny, nz must be positive")
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")

        ox, oy, oz = self.origin
        xs = np.linspace(ox, ox + self.lx, self.nx + 1, dtype=np.float64)
        ys = np.linspace(oy, oy + self.ly, self.ny + 1, dtype=np.float64)
        zs = np.linspace(oz, oz + self.lz, self.nz + 1, dtype=np.float64)

        if self.order == 1:
            return self._build_linear(xs, ys, zs)
        return self._build_tet10(xs, ys, zs)

    # keep linear build for order=1
    def _build_linear(self, xs, ys, zs) -> TetMesh:
        ox, oy, oz = 0.0, 0.0, 0.0  # unused but keep signature consistent
        coords_list = []
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    coords_list.append([xs[i], ys[j], zs[k]])
        coords = np.asarray(coords_list, dtype=DTYPE)
        def node_id(i: int, j: int, k: int) -> int:
            return k * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + i
        conn_list = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    v000 = node_id(i, j, k)
                    v100 = node_id(i + 1, j, k)
                    v010 = node_id(i, j + 1, k)
                    v110 = node_id(i + 1, j + 1, k)
                    v001 = node_id(i, j, k + 1)
                    v101 = node_id(i + 1, j, k + 1)
                    v011 = node_id(i, j + 1, k + 1)
                    v111 = node_id(i + 1, j + 1, k + 1)
                    conn_list.extend(
                        [
                            [v000, v100, v010, v001],
                            [v100, v110, v010, v111],
                            [v100, v010, v001, v111],
                            [v100, v001, v101, v111],
                            [v010, v001, v011, v111],
                        ]
                    )
        conn = np.asarray(conn_list, dtype=np.int32)
        conn = self._fix_orientation(coords, conn)
        return TetMesh(coords=jnp.array(coords), conn=jnp.array(conn))


@dataclass
class StructuredTetTensorBox:
    """
    Regular grid subdivided into 6 tets per cube (matches skfem MeshTet.init_tensor).
    """

    nx: int
    ny: int
    nz: int
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    order: int = 1  # only linear supported

    def _fix_orientation(self, coords: np.ndarray, conn: np.ndarray) -> np.ndarray:
        conn = conn.copy()
        for idx, tet in enumerate(conn):
            p = coords[tet]
            p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
            J = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=1)
            if np.linalg.det(J) < 0:
                tet[[1, 2]] = tet[[2, 1]]
                conn[idx] = tet
        return conn

    def build(self) -> TetMesh:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("nx, ny, nz must be positive")
        if self.order != 1:
            raise ValueError("StructuredTetTensorBox only supports order=1")

        ox, oy, oz = self.origin
        xs = np.linspace(ox, ox + self.lx, self.nx + 1, dtype=np.float64)
        ys = np.linspace(oy, oy + self.ly, self.ny + 1, dtype=np.float64)
        zs = np.linspace(oz, oz + self.lz, self.nz + 1, dtype=np.float64)
        return self._build_linear(xs, ys, zs)

    def _build_linear(self, xs, ys, zs) -> TetMesh:
        # Mirror scikit-fem MeshTet.init_tensor.
        npx = len(xs)
        npy = len(ys)
        npz = len(zs)
        X, Y, Z = np.meshgrid(np.sort(xs), np.sort(ys), np.sort(zs))
        p = np.vstack((X.flatten("F"), Y.flatten("F"), Z.flatten("F")))
        ix = np.arange(npx * npy * npz)
        ne = (npx - 1) * (npy - 1) * (npz - 1)
        t = np.zeros((8, ne), dtype=np.int64)
        ix = ix.reshape(npy, npx, npz, order="F").copy()
        t[0] = ix[0:(npy - 1), 0:(npx - 1), 0:(npz - 1)].reshape(ne, 1, order="F").copy().flatten()
        t[1] = ix[1:npy, 0:(npx - 1), 0:(npz - 1)].reshape(ne, 1, order="F").copy().flatten()
        t[2] = ix[0:(npy - 1), 1:npx, 0:(npz - 1)].reshape(ne, 1, order="F").copy().flatten()
        t[3] = ix[0:(npy - 1), 0:(npx - 1), 1:npz].reshape(ne, 1, order="F").copy().flatten()
        t[4] = ix[1:npy, 1:npx, 0:(npz - 1)].reshape(ne, 1, order="F").copy().flatten()
        t[5] = ix[1:npy, 0:(npx - 1), 1:npz].reshape(ne, 1, order="F").copy().flatten()
        t[6] = ix[0:(npy - 1), 1:npx, 1:npz].reshape(ne, 1, order="F").copy().flatten()
        t[7] = ix[1:npy, 1:npx, 1:npz].reshape(ne, 1, order="F").copy().flatten()

        T = np.zeros((4, 6 * ne), dtype=np.int64)
        T[:, :ne] = t[[0, 1, 5, 7]]
        T[:, ne:(2 * ne)] = t[[0, 1, 4, 7]]
        T[:, (2 * ne):(3 * ne)] = t[[0, 2, 4, 7]]
        T[:, (3 * ne):(4 * ne)] = t[[0, 3, 5, 7]]
        T[:, (4 * ne):(5 * ne)] = t[[0, 2, 6, 7]]
        T[:, (5 * ne):] = t[[0, 3, 6, 7]]

        coords = p.T.astype(DTYPE, copy=False)
        conn = T.T.astype(np.int32, copy=False)
        conn = self._fix_orientation(coords, conn)
        return TetMesh(coords=jnp.array(coords), conn=jnp.array(conn))

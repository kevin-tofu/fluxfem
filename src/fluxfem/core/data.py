from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(eq=False)
class MeshData:
    """Lightweight mesh data container for JAX-friendly serialization."""
    coords: jnp.ndarray
    conn: jnp.ndarray
    cell_tags: jnp.ndarray | None = None
    node_tags: jnp.ndarray | None = None

    @classmethod
    def from_mesh(cls, mesh: Any) -> "MeshData":
        return cls(
            coords=jnp.asarray(mesh.coords),
            conn=jnp.asarray(mesh.conn),
            cell_tags=None if mesh.cell_tags is None else jnp.asarray(mesh.cell_tags),
            node_tags=None if mesh.node_tags is None else jnp.asarray(mesh.node_tags),
        )


@dataclass(eq=False)
class BasisData:
    """Quadrature and basis metadata for reproducible assembly."""
    quad_points: jnp.ndarray
    quad_weights: jnp.ndarray
    dofs_per_node: int
    kind: str

    @classmethod
    def from_basis(cls, basis: Any) -> "BasisData":
        return cls(
            quad_points=jnp.asarray(basis.quad_points),
            quad_weights=jnp.asarray(basis.quad_weights),
            dofs_per_node=int(basis.dofs_per_node),
            kind=type(basis).__name__,
        )


@dataclass(eq=False)
class SpaceData:
    """Snapshot of space-related data used in assembly."""
    mesh: MeshData
    basis: BasisData
    elem_dofs: jnp.ndarray
    value_dim: int
    n_dofs: int
    n_ldofs: int

    @classmethod
    def from_space(cls, space: Any) -> "SpaceData":
        return cls(
            mesh=MeshData.from_mesh(space.mesh),
            basis=BasisData.from_basis(space.basis),
            elem_dofs=jnp.asarray(space.elem_dofs),
            value_dim=int(space.value_dim),
            n_dofs=int(space.n_dofs),
            n_ldofs=int(space.n_ldofs),
        )

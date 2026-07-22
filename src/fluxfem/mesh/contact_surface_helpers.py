from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .contact_api import ContactSide
from .contact_interface import (
    assemble_contact_onesided_floor,
    map_surface_facets_to_hex_elements,
    map_surface_facets_to_tet_elements,
)
from .surface import SurfaceMesh


def summarize_contact_field_state(state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None) -> dict[str, Any]:
    def _shape_summary(value: Any) -> tuple[int, ...]:
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = np.asarray(value).shape
        return tuple(int(x) for x in shape)

    if state is None:
        return {}
    if isinstance(state, Mapping):
        summary: dict[str, Any] = {}
        for key, value in state.items():
            summary[str(key)] = _shape_summary(value)
        return summary
    if isinstance(state, Sequence) and not hasattr(state, "shape"):
        summary = {}
        for i, value in enumerate(state):
            summary[f"arg{i}"] = _shape_summary(value)
        return summary
    return {"arg0": _shape_summary(state)}


def facet_map_for_elem_conn(surface: SurfaceMesh, elem_conn: np.ndarray | None) -> np.ndarray:
    if elem_conn is None:
        raise ValueError("elem_conn is required to build facet_to_elem mapping.")
    if elem_conn.shape[1] in {4, 10}:
        return map_surface_facets_to_tet_elements(surface, elem_conn)
    if elem_conn.shape[1] in {8, 20, 27}:
        return map_surface_facets_to_hex_elements(surface, elem_conn)
    raise NotImplementedError("elem_conn must be tet4/tet10/hex8/hex20/hex27")


def surface_node_normals(surface: SurfaceMesh, *, normal_sign: float = 1.0, tol: float = 1e-12) -> np.ndarray | None:
    if not hasattr(surface, "facet_normals"):
        return None
    facet_normals = np.asarray(surface.facet_normals(), dtype=float)
    facets = np.asarray(surface.conn, dtype=int)
    n_nodes = int(np.asarray(surface.coords).shape[0])
    if facet_normals.ndim != 2 or facet_normals.shape[0] != facets.shape[0]:
        return None
    node_normals = np.zeros((n_nodes, facet_normals.shape[1]), dtype=float)
    counts = np.zeros((n_nodes,), dtype=float)
    for f_id, facet in enumerate(facets):
        normal = float(normal_sign) * facet_normals[int(f_id)]
        for node in np.asarray(facet, dtype=int):
            node_normals[int(node)] += normal
            counts[int(node)] += 1.0
    valid = counts > 0.0
    if not np.any(valid):
        return None
    node_normals[valid] /= counts[valid, None]
    norms = np.linalg.norm(node_normals, axis=1)
    good = norms > float(tol)
    node_normals[good] /= norms[good, None]
    return node_normals


def onesided_gap_diagnostics(
    contact: Any,
    state_sol: Mapping[str, Any] | Sequence[Any] | Any,
    *,
    u_hat_fn: Any | None,
    state_field: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if u_hat_fn is None:
        return None, None
    if isinstance(state_sol, Mapping):
        if state_field is not None:
            u_state = state_sol.get(state_field)
        elif len(state_sol) == 1:
            u_state = next(iter(state_sol.values()))
        else:
            return None, None
    elif isinstance(state_sol, Sequence) and not hasattr(state_sol, "shape"):
        if len(state_sol) != 1:
            return None, None
        u_state = state_sol[0]
    else:
        u_state = state_sol
    if u_state is None:
        return None, None

    coords = np.asarray(contact.surface_slave.coords, dtype=float)
    value_dim = int(contact.value_dim)
    u_nodes = np.asarray(u_state, dtype=float).reshape(-1, value_dim)
    u_hat = np.asarray(u_hat_fn(coords), dtype=float)
    if u_hat.shape != u_nodes.shape:
        return None, None
    node_normals = surface_node_normals(contact.surface_slave, normal_sign=float(contact.normal_sign))
    if node_normals is None or node_normals.shape != u_nodes.shape:
        return None, None
    gap_n = np.einsum("ni,ni->n", u_nodes - u_hat, node_normals)
    active_mask = gap_n < 0.0
    return gap_n, active_mask


def facet_gap_values(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
) -> tuple[np.ndarray, float]:
    """
    Compute per-facet gap values for a one-sided contact plane.

    Returns (g_f, min_g_all) where g_f is reduced per facet and min_g_all is
    the global minimum node gap.
    """
    coords_np = np.asarray(coords, dtype=float)
    if value_dim is None:
        value_dim = int(coords_np.shape[1])
    u_nodes = np.asarray(u, dtype=float).reshape(-1, value_dim)
    x_cur = coords_np + u_nodes
    g_all = np.dot(x_cur, np.asarray(n, dtype=float)) - float(c)
    min_g_all = float(np.min(g_all)) if g_all.size else 0.0
    if facets is None or len(facets) == 0:
        return np.zeros((0,), dtype=float), min_g_all
    if reduce == "min":
        g_f = np.array([np.min(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    elif reduce == "mean":
        g_f = np.array([np.mean(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    else:
        raise ValueError("reduce must be 'min' or 'mean'")
    return g_f, min_g_all


def active_contact_facets(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
    threshold: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return active facet indices and global minimum gap for one-sided contact."""
    g_f, min_g_all = facet_gap_values(
        coords,
        facets,
        u,
        n,
        c,
        value_dim=value_dim,
        reduce=reduce,
    )
    active_ids = np.nonzero(g_f < threshold)[0]
    return active_ids, min_g_all


@dataclass(frozen=True)
class OneSidedContact:
    side: ContactSide
    n: np.ndarray | None
    c: float
    k: float
    beta: float
    quad_order: int = 2
    normal_sign: float = 1.0
    tol: float = 1e-8
    facet_map: np.ndarray | None = None

    @classmethod
    def from_side(
        cls,
        side: ContactSide,
        *,
        n: np.ndarray | None,
        c: float,
        k: float,
        beta: float,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
        facet_map: np.ndarray | None = None,
    ) -> "OneSidedContact":
        if facet_map is None:
            facet_map = facet_map_for_elem_conn(side.surface, side.elem_conn)
        return cls(
            side=side,
            n=n,
            c=float(c),
            k=float(k),
            beta=float(beta),
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
            facet_map=facet_map,
        )

    def assemble(self, u, *, return_metrics: bool = False):
        return assemble_contact_onesided_floor(
            self.side.surface,
            np.asarray(u, dtype=float),
            n=None if self.n is None else np.asarray(self.n, dtype=float),
            c=self.c,
            k=self.k,
            beta=self.beta,
            value_dim=self.side.value_dim,
            elem_conn=np.asarray(self.side.elem_conn) if self.side.elem_conn is not None else None,
            facet_to_elem=self.facet_map,
            quad_order=self.quad_order,
            normal_sign=self.normal_sign,
            tol=self.tol,
            return_metrics=return_metrics,
        )


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


def contact_space_side_n_dofs(space: Any, *, side: str, role: str = "test") -> int:
    if role not in {"test", "trial"}:
        raise ValueError("role must be 'test' or 'trial'")
    if side == "master":
        if role == "trial":
            value_dim, space_mode, facet_dofs = space._trial_layout(side="master")
        else:
            value_dim, space_mode, facet_dofs = int(space.value_dim_master), space.space_mode_master, space.facet_dofs_master
        return _field_n_dofs(
            n_nodes=int(np.asarray(space.surface_master.coords).shape[0]),
            n_facets=int(np.asarray(space.surface_master.conn).shape[0]),
            value_dim=int(value_dim),
            space_mode=space_mode,
            facet_dofs=facet_dofs,
        )
    if side == "slave":
        if role == "trial":
            value_dim, space_mode, facet_dofs = space._trial_layout(side="slave")
        else:
            value_dim, space_mode, facet_dofs = int(space.value_dim_slave), space.space_mode_slave, space.facet_dofs_slave
        return _field_n_dofs(
            n_nodes=int(np.asarray(space.surface_slave.coords).shape[0]),
            n_facets=int(np.asarray(space.surface_slave.conn).shape[0]),
            value_dim=int(value_dim),
            space_mode=space_mode,
            facet_dofs=facet_dofs,
        )
    raise ValueError("side must be 'master' or 'slave'")

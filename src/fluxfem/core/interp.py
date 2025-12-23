from __future__ import annotations

import numpy as np

from .basis import HexTriLinearBasis
from .space import FESpace


def eval_shape_functions_hex8(xi_eta_zeta: np.ndarray) -> np.ndarray:
    """
    Evaluate trilinear Hex8 shape functions at given local coords (xi, eta, zeta) in [-1,1]^3.
    Returns N with shape (n_q, 8).
    """
    pts = np.atleast_2d(np.asarray(xi_eta_zeta, dtype=float))
    xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
    N = np.stack(
        [
            0.125 * (1 - xi) * (1 - eta) * (1 - zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 - zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 - zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 - zeta),
            0.125 * (1 - xi) * (1 - eta) * (1 + zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 + zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 + zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 + zeta),
        ],
        axis=1,
    )
    return N


def interpolate_field_at_element_points(space: FESpace, u: np.ndarray, xi_eta_zeta: np.ndarray) -> np.ndarray:
    """
    Interpolate vector field u (3 dof/node ordering) at given local points for all elements.
    - xi_eta_zeta: (m,3) local coords in [-1,1]^3
    Returns: (n_elem, m, 3)
    """
    if not isinstance(space.basis, HexTriLinearBasis):
        raise NotImplementedError("interpolate_field_at_element_points currently supports Hex8 (trilinear) only.")
    N = eval_shape_functions_hex8(xi_eta_zeta)  # (m,8)
    u_arr = np.asarray(u)
    n_nodes = space.mesh.coords.shape[0]
    if u_arr.shape[0] != 3 * n_nodes:
        raise ValueError(f"Expected 3 dof/node; got {u_arr.shape[0]} for {n_nodes} nodes")
    u_nodes = u_arr.reshape(n_nodes, 3)
    conn = np.asarray(space.elem_dofs) // 3  # node indices
    elem_u = u_nodes[conn]  # (n_elem,8,3)
    vals = np.einsum("pq,eqr->epr", N, elem_u)  # (n_elem, m, 3)
    return vals


__all__ = [
    "eval_shape_functions_hex8",
    "interpolate_field_at_element_points",
]

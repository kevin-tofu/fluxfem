from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from .dtypes import default_dtype


def build_B_matrices(dN_dx: jnp.ndarray) -> jnp.ndarray:
    """
    Build B matrices for all quadrature points.

    dN_dx: (n_q, 8, 3)
    Returns:
      B: (n_q, 6, 24)  # 6 strain components, 3 dofs/node * 8 nodes
    """
    n_q = dN_dx.shape[0]
    n_nodes = 8
    dofs_per_node = 3
    n_dofs = n_nodes * dofs_per_node

    def B_single(dN):
        # dN: (8,3) for one quad point
        B = jnp.zeros((6, n_dofs), dtype=dN.dtype)

        # loop over nodes (fixed small, ok as Python loop)
        def body_fun(i, B):
            dNdx = dN[i, 0]
            dNdy = dN[i, 1]
            dNdz = dN[i, 2]

            col = 3 * i
            # ε_xx, ε_yy, ε_zz
            B = B.at[0, col + 0].set(dNdx)
            B = B.at[1, col + 1].set(dNdy)
            B = B.at[2, col + 2].set(dNdz)

            # γ_xy
            B = B.at[3, col + 0].set(dNdy)
            B = B.at[3, col + 1].set(dNdx)

            # γ_yz
            B = B.at[4, col + 1].set(dNdz)
            B = B.at[4, col + 2].set(dNdy)

            # γ_zx
            B = B.at[5, col + 0].set(dNdz)
            B = B.at[5, col + 2].set(dNdx)

            return B

        B = jax.lax.fori_loop(0, n_nodes, body_fun, B)
        return B

    B = jax.vmap(B_single)(dN_dx)  # (n_q, 6, 24)
    return B


def build_B_matrices_finite(dN_dX: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
    """
    Build finite-strain B matrices (Voigt) at each quadrature point.

    Args:
        dN_dX: (n_q, n_nodes, 3) gradients of shape functions in reference config.
        F: (n_q, 3, 3) deformation gradient at each quadrature point.
    Returns:
        B: (n_q, 6, n_dofs) where n_dofs = 3 * n_nodes
           Voigt order: [xx, yy, zz, xy, yz, zx]
    """
    n_q, n_nodes, _ = dN_dX.shape
    dofs_per_node = 3
    n_dofs = n_nodes * dofs_per_node

    def B_single(dN, Fq):
        B = jnp.zeros((6, n_dofs), dtype=dN.dtype)

        def body_fun(i, B):
            dNa = dN[i, :]  # (3,)
            col = dofs_per_node * i

            ex = jnp.array([1.0, 0.0, 0.0], dtype=dN.dtype)
            ey = jnp.array([0.0, 1.0, 0.0], dtype=dN.dtype)
            ez = jnp.array([0.0, 0.0, 1.0], dtype=dN.dtype)
            grads = jnp.stack(
                (
                    jnp.outer(ex, dNa),  # (3,3)
                    jnp.outer(ey, dNa),
                    jnp.outer(ez, dNa),
                ),
                axis=0,
            )  # (3,3,3)

            def fill_dof(k, B):
                grad_delta = grads[k]
                # Total Lagrange variation: dE = 0.5 * (∇δu · F + (∇δu · F)^T)
                dE = 0.5 * (grad_delta @ Fq + (grad_delta @ Fq).T)
                B = B.at[0, col + k].set(dE[0, 0])  # xx
                B = B.at[1, col + k].set(dE[1, 1])  # yy
                B = B.at[2, col + k].set(dE[2, 2])  # zz
                B = B.at[3, col + k].set(dE[0, 1])  # xy
                B = B.at[4, col + k].set(dE[1, 2])  # yz
                B = B.at[5, col + k].set(dE[2, 0])  # zx
                return B

            B = jax.lax.fori_loop(0, dofs_per_node, fill_dof, B)
            return B

        B = jax.lax.fori_loop(0, n_nodes, body_fun, B)
        return B

    B = jax.vmap(B_single)(dN_dX, F)  # (n_q, 6, n_dofs)
    return B


class SmallStrainBMixin:
    dofs_per_node: int = 3

    def B_small_strain(self, dN_dx: jnp.ndarray) -> jnp.ndarray:
        """
        dN_dx: (n_q, n_nodes, 3)
        returns: (n_q, 6, 3*n_nodes)  Voigt: [xx,yy,zz,xy,yz,zx]
        """
        n_q, n_nodes, _ = dN_dx.shape
        n_dofs = self.dofs_per_node * n_nodes

        def B_single(dN):
            B = jnp.zeros((6, n_dofs), dtype=dN.dtype)

            def body_fun(i, B):
                dNdx, dNdy, dNdz = dN[i, 0], dN[i, 1], dN[i, 2]
                col = self.dofs_per_node * i

                # xx, yy, zz
                B = B.at[0, col + 0].set(dNdx)
                B = B.at[1, col + 1].set(dNdy)
                B = B.at[2, col + 2].set(dNdz)
                # xy
                B = B.at[3, col + 0].set(dNdy)
                B = B.at[3, col + 1].set(dNdx)
                # yz
                B = B.at[4, col + 1].set(dNdz)
                B = B.at[4, col + 2].set(dNdy)
                # zx
                B = B.at[5, col + 0].set(dNdz)
                B = B.at[5, col + 2].set(dNdx)
                return B

            return jax.lax.fori_loop(0, n_nodes, body_fun, B)

        return jax.vmap(B_single)(dN_dx)


class TotalLagrangeBMixin:
    dofs_per_node: int = 3

    def B_total_lagrange(self, dN_dX: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
        """
        dN_dX: (n_q, n_nodes, 3), F: (n_q, 3, 3)
        returns: (n_q, 6, 3*n_nodes)  Voigt: [xx,yy,zz,xy,yz,zx]
        """

        n_q, n_nodes, _ = dN_dX.shape
        n_dofs = self.dofs_per_node * n_nodes

        ex = jnp.array([1.0, 0.0, 0.0], dtype=dN_dX.dtype)
        ey = jnp.array([0.0, 1.0, 0.0], dtype=dN_dX.dtype)
        ez = jnp.array([0.0, 0.0, 1.0], dtype=dN_dX.dtype)

        def B_single(dN, Fq):
            B = jnp.zeros((6, n_dofs), dtype=dN.dtype)

            def node_fun(i, B):
                dNa = dN[i, :]
                col = self.dofs_per_node * i

                grads = jnp.stack(
                    (jnp.outer(ex, dNa), jnp.outer(ey, dNa), jnp.outer(ez, dNa)),
                    axis=0,
                )  # (3,3,3)

                def dof_fun(k, B):
                    grad_delta = grads[k]
                    dE = 0.5 * (grad_delta @ Fq + (grad_delta @ Fq).T)
                    B = B.at[0, col + k].set(dE[0, 0])
                    B = B.at[1, col + k].set(dE[1, 1])
                    B = B.at[2, col + k].set(dE[2, 2])
                    B = B.at[3, col + k].set(dE[0, 1])
                    B = B.at[4, col + k].set(dE[1, 2])
                    B = B.at[5, col + k].set(dE[2, 0])
                    return B

                return jax.lax.fori_loop(0, self.dofs_per_node, dof_fun, B)

            return jax.lax.fori_loop(0, n_nodes, node_fun, B)

        return jax.vmap(B_single)(dN_dX, F)


class Basis3D(Protocol):
    def B_small_strain(self, dN_dx: jnp.ndarray) -> jnp.ndarray: ...
    def B_total_lagrange(
        self, dN_dX: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray: ...

    quad_points: jnp.ndarray   # (n_q, 3)
    quad_weights: jnp.ndarray  # (n_q,)
    dofs_per_node: int         # usually 3 for vector mechanics

    @property
    def n_q(self) -> int: ...

    @property
    def n_nodes(self) -> int: ...

    def shape_functions(self) -> jnp.ndarray: ...

    def shape_grads_ref(self) -> jnp.ndarray: ...

    def spatial_grads_and_detJ(
        self, elem_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...


def _quadratic_1d(x: jnp.ndarray):
    """1D serendipity shape funcs (vertex, mid-edge) for [-1,1]."""
    N1 = -0.5 * x * (1.0 - x)   # at -1
    N2 = 1.0 - x * x            # at 0
    N3 = 0.5 * x * (1.0 + x)    # at 1
    dN1 = 0.5 * (2.0 * x - 1.0)
    dN2 = -2.0 * x
    dN3 = 0.5 * (2.0 * x + 1.0)
    return (N1, N2, N3), (dN1, dN2, dN3)


def _quad1d_full(x: jnp.ndarray):
    """1D quadratic (Lagrange) shape funcs at nodes (-1, 0, 1)."""
    N0 = 0.5 * x * (x - 1.0)
    N1 = 1.0 - x * x
    N2 = 0.5 * x * (x + 1.0)
    dN0 = x - 0.5
    dN1 = -2.0 * x
    dN2 = x + 0.5
    return (N0, N1, N2), (dN0, dN1, dN2)


@dataclass(eq=False)
class TetLinearBasis(SmallStrainBMixin, TotalLagrangeBMixin):
    """4-node linear tetra basis with simple quadrature."""

    quad_points: jnp.ndarray # (n_q, 3)
    quad_weights: jnp.ndarray # (n_q,)
    dofs_per_node: int = 3

    @property
    def n_nodes(self) -> int:
        return 4

    def tree_flatten(self):
        children = (self.quad_points, self.quad_weights)
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        qp, qw = children
        return cls(qp, qw)

    @property
    def n_q(self) -> int:
        return int(self.quad_points.shape[0])

    @property
    def ref_node_coords(self) -> jnp.ndarray:
        dtype = default_dtype()
        return jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
        )

    def shape_functions(self) -> jnp.ndarray:
        qp = self.quad_points  # (n_q, 3)
        xi = qp[:, 0]
        eta = qp[:, 1]
        zeta = qp[:, 2]
        N1 = 1.0 - xi - eta - zeta
        N2 = xi
        N3 = eta
        N4 = zeta
        return jnp.stack([N1, N2, N3, N4], axis=1)  # (n_q,4)

    def shape_grads_ref(self) -> jnp.ndarray:
        # constant gradients in reference tetra
        dtype = default_dtype()
        dN = jnp.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
        )
        dN = jnp.tile(dN[None, :, :], (self.n_q, 1, 1))  # (n_q,4,3)
        return dN

    def spatial_grads_and_detJ(
        self, elem_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dN_dxi = self.shape_grads_ref()[0]  # (4,3) constant
        J = jnp.einsum("ia,ik->ak", elem_coords, dN_dxi)  # (3,3)
        J_inv = jnp.linalg.inv(J)
        detJ = jnp.linalg.det(J)
        dN_dx = jnp.einsum("ik,ka->ia", dN_dxi, J_inv)  # (4,3)
        dN_dx = jnp.tile(dN_dx[None, :, :], (self.n_q, 1, 1))
        detJ = jnp.full((self.n_q,), detJ, dtype=elem_coords.dtype)
        return dN_dx, detJ


@dataclass(eq=False)
class TetQuadraticBasis10(SmallStrainBMixin, TotalLagrangeBMixin):
    """10-node quadratic tetra basis (corner + edge mids)."""

    quad_points: jnp.ndarray
    quad_weights: jnp.ndarray
    dofs_per_node: int = 3
    @property
    def n_nodes(self) -> int:
        return 10

    def tree_flatten(self):
        return (self.quad_points, self.quad_weights), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        qp, qw = children
        return cls(qp, qw)

    @property
    def n_q(self) -> int:
        return int(self.quad_points.shape[0])

    def shape_functions(self) -> jnp.ndarray:
        qp = self.quad_points
        L1 = 1.0 - qp[:, 0] - qp[:, 1] - qp[:, 2]
        L2 = qp[:, 0]
        L3 = qp[:, 1]
        L4 = qp[:, 2]
        N1 = L1 * (2 * L1 - 1)
        N2 = L2 * (2 * L2 - 1)
        N3 = L3 * (2 * L3 - 1)
        N4 = L4 * (2 * L4 - 1)
        N5 = 4 * L1 * L2
        N6 = 4 * L2 * L3
        N7 = 4 * L1 * L3
        N8 = 4 * L1 * L4
        N9 = 4 * L2 * L4
        N10 = 4 * L3 * L4
        return jnp.stack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10], axis=1)

    def shape_grads_ref(self) -> jnp.ndarray:
        qp = self.quad_points
        L1 = 1.0 - qp[:, 0] - qp[:, 1] - qp[:, 2]
        L2 = qp[:, 0]
        L3 = qp[:, 1]
        L4 = qp[:, 2]

        dL1 = jnp.array([-1.0, -1.0, -1.0])
        dL2 = jnp.array([1.0, 0.0, 0.0])
        dL3 = jnp.array([0.0, 1.0, 0.0])
        dL4 = jnp.array([0.0, 0.0, 1.0])

        grads = []
        for a, dLa in zip([L1, L2, L3, L4], [dL1, dL2, dL3, dL4]):
            grads.append((4 * a - 1)[..., None] * dLa[None, :])

        dN1 = grads[0]
        dN2 = grads[1]
        dN3 = grads[2]
        dN4 = grads[3]
        dN5 = 4 * (L2[..., None] * dL1[None, :] + L1[..., None] * dL2[None, :])
        dN6 = 4 * (L3[..., None] * dL2[None, :] + L2[..., None] * dL3[None, :])
        dN7 = 4 * (L3[..., None] * dL1[None, :] + L1[..., None] * dL3[None, :])
        dN8 = 4 * (L4[..., None] * dL1[None, :] + L1[..., None] * dL4[None, :])
        dN9 = 4 * (L4[..., None] * dL2[None, :] + L2[..., None] * dL4[None, :])
        dN10 = 4 * (L4[..., None] * dL3[None, :] + L3[..., None] * dL4[None, :])

        dN = jnp.stack([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10], axis=1)
        return dN  # (n_q, 10, 3)

    def spatial_grads_and_detJ(
        self, elem_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dN_dxi = self.shape_grads_ref()  # (n_q,10,3)
        J = jnp.einsum("ia,qik->qak", elem_coords, dN_dxi)
        J_inv = jnp.linalg.inv(J)
        detJ = jnp.linalg.det(J)
        dN_dx = jnp.einsum("qik,qka->qia", dN_dxi, J_inv)
        return dN_dx, detJ



@dataclass(eq=False)
class HexTriLinearBasis(SmallStrainBMixin, TotalLagrangeBMixin):
    """
    Trilinear 8-node hex element basis with given quadrature rule.

    quad_points:  (n_q, 3)  # (xi, eta, zeta) in [-1, 1]^3
    quad_weights: (n_q,)    # weights
    """
    quad_points: jnp.ndarray  # (n_q, 3)
    quad_weights: jnp.ndarray # (n_q,)
    dofs_per_node: int = 3

    @property
    def n_nodes(self) -> int:
        return 8

    def tree_flatten(self):
        children = (self.quad_points, self.quad_weights)
        aux_data: dict[str, object] = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        qp, qw = children
        return cls(qp, qw)

    @property
    def n_q(self) -> int:
        return int(self.quad_points.shape[0])

    @property
    def ref_node_signs(self) -> jnp.ndarray:
        """
        Node signs (8,3) for (-1,1)^3 reference hex.
        Node ordering: 
        0: (-1,-1,-1)
        1: ( 1,-1,-1)
        2: ( 1, 1,-1)
        3: (-1, 1,-1)
        4: (-1,-1, 1)
        5: ( 1,-1, 1)
        6: ( 1, 1, 1)
        7: (-1, 1, 1)
        """
        dtype = default_dtype()
        return jnp.array(
            [
                [-1.0, -1.0, -1.0],
                [ 1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0],
                [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0,  1.0],
                [-1.0,  1.0,  1.0],
            ],
            dtype=dtype,
        )

    # ---------- reference shape functions & gradients ----------
    def shape_functions(self) -> jnp.ndarray:
        """
        Evaluate shape functions at all quadrature points.
        Returns: (n_q, 8)
        """
        qp = self.quad_points  # (n_q, 3)
        s = self.ref_node_signs  # (8, 3)

        # broadcast: (n_q, 1, 3) * (1, 8, 3) -> (n_q, 8, 3)
        # but we only need linear forms of xi,eta,ζ
        xi   = qp[:, 0:1]  # (n_q, 1)
        eta  = qp[:, 1:2]
        zeta = qp[:, 2:3]

        s_xi   = s[:, 0]  # (8,)
        s_eta  = s[:, 1]
        s_zeta = s[:, 2]

        # (n_q, 8) via broadcasting
        f_xi   = 1.0 + xi   * s_xi
        f_eta  = 1.0 + eta  * s_eta
        f_zeta = 1.0 + zeta * s_zeta

        N = 0.125 * f_xi * f_eta * f_zeta  # (n_q, 8)
        return N

    def shape_grads_ref(self) -> jnp.ndarray:
        """
        Gradients in reference coords (ξ,η,ζ) at all quad points.
        Returns: (n_q, 8, 3)  [dN/dξ, dN/dη, dN/dζ]
        """
        qp = self.quad_points  # (n_q, 3)
        s = self.ref_node_signs  # (8, 3)
        xi   = qp[:, 0:1]
        eta  = qp[:, 1:2]
        zeta = qp[:, 2:3]

        s_xi   = s[:, 0]  # (8,)
        s_eta  = s[:, 1]
        s_zeta = s[:, 2]

        # helper linear terms
        f_xi   = 1.0 + xi   * s_xi
        f_eta  = 1.0 + eta  * s_eta
        f_zeta = 1.0 + zeta * s_zeta

        # dN/dξ
        dN_dxi = 0.125 * s_xi * f_eta * f_zeta  # (n_q, 8)
        # dN/dη
        dN_deta = 0.125 * s_eta * f_xi * f_zeta
        # dN/dζ
        dN_dzeta = 0.125 * s_zeta * f_xi * f_eta

        # stack into (n_q, 8, 3)
        dN = jnp.stack([dN_dxi, dN_deta, dN_dzeta], axis=-1)
        return dN  # (n_q, 8, 3)

    # ---------- mapping to physical element ----------
    def spatial_grads_and_detJ(
        self, elem_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute spatial gradients and detJ for one element.

        Parameters
        ----------
        elem_coords : jnp.ndarray
            Element coordinates of shape (8, 3).

        Returns
        -------
        dN_dx : jnp.ndarray
            Spatial gradients of shape (n_q, 8, 3).
        detJ : jnp.ndarray
            Determinant of the Jacobian of shape (n_q,).
        """
        dN_dxi = self.shape_grads_ref()  # (n_q, 8, 3)

        # Jacobian: J(q)[α,k] = sum_i X_i[α] * ∂N_i/∂ξ_k
        # elem_coords: (8,3)   -> i,α
        # dN_dxi:      (n_q,8,3)-> q,i,k
        J = jnp.einsum("ia,qik->qak", elem_coords, dN_dxi)  # (n_q, 3, 3)

        J_inv = jnp.linalg.inv(J)           # (n_q, 3, 3)
        detJ = jnp.linalg.det(J)           # (n_q,)

        # ∇_x N = ∇_ξ N · J^{-1}
        dN_dx = jnp.einsum("qik,qka->qia", dN_dxi, J_inv)  # (n_q, 8, 3)
        return dN_dx, detJ


@dataclass(eq=False)
class HexSerendipityBasis20(SmallStrainBMixin, TotalLagrangeBMixin):
    """
    20-node serendipity hex element basis (corner + edge midpoints).
    """
    quad_points: jnp.ndarray  # (n_q, 3)
    quad_weights: jnp.ndarray # (n_q,)
    dofs_per_node: int = 3

    @property
    def n_nodes(self) -> int:
        return 20

    def tree_flatten(self):
        children = (self.quad_points, self.quad_weights)
        aux_data: dict[str, object] = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        qp, qw = children
        return cls(qp, qw)

    @property
    def n_q(self) -> int:
        return int(self.quad_points.shape[0])

    @property
    def ref_node_coords(self) -> jnp.ndarray:
        dtype = default_dtype()
        corners = jnp.array(
            [
                [-1.0, -1.0, -1.0],
                [ 1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0],
                [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0,  1.0],
                [-1.0,  1.0,  1.0],
            ],
            dtype=dtype,
        )
        edges = jnp.array(
            [
                [ 0.0, -1.0, -1.0],  # 0-1
                [ 1.0,  0.0, -1.0],  # 1-2
                [ 0.0,  1.0, -1.0],  # 2-3
                [-1.0,  0.0, -1.0],  # 3-0
                [ 0.0, -1.0,  1.0],  # 4-5
                [ 1.0,  0.0,  1.0],  # 5-6
                [ 0.0,  1.0,  1.0],  # 6-7
                [-1.0,  0.0,  1.0],  # 7-4
                [-1.0, -1.0,  0.0],  # 0-4
                [ 1.0, -1.0,  0.0],  # 1-5
                [ 1.0,  1.0,  0.0],  # 2-6
                [-1.0,  1.0,  0.0],  # 3-7
            ],
            dtype=dtype,
        )
        return jnp.concatenate([corners, edges], axis=0)  # (20,3)

    def shape_functions(self) -> jnp.ndarray:
        qp = self.quad_points  # (n_q, 3)
        xi = qp[:, 0:1]
        eta = qp[:, 1:2]
        zeta = qp[:, 2:3]

        # corners
        dtype = default_dtype()
        s = jnp.array(
            [
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
            ],
            dtype=dtype,
        )
        sx = s[:, 0]
        sy = s[:, 1]
        sz = s[:, 2]
        term = xi * sx + eta * sy + zeta * sz - 2.0  # (n_q,8)
        N_corner = 0.125 * (1 + sx * xi) * (1 + sy * eta) * (1 + sz * zeta) * term  # (n_q,8)

        # edges: order matches hex20 connectivity (e01, e12, e23, e30, e45, e56, e67, e74, e04, e15, e26, e37)
        def edge_x(sy, sz):
            return 0.25 * (1 - xi * xi) * (1 + sy * eta) * (1 + sz * zeta)

        def edge_y(sx, sz):
            return 0.25 * (1 - eta * eta) * (1 + sx * xi) * (1 + sz * zeta)

        def edge_z(sx, sy):
            return 0.25 * (1 - zeta * zeta) * (1 + sx * xi) * (1 + sy * eta)

        N_edges = [
            edge_x(-1, -1),
            edge_y(1, -1),
            edge_x(1, -1),
            edge_y(-1, -1),
            edge_x(-1, 1),
            edge_y(1, 1),
            edge_x(1, 1),
            edge_y(-1, 1),
            edge_z(-1, -1),
            edge_z(1, -1),
            edge_z(1, 1),
            edge_z(-1, 1),
        ]
        N_edges = jnp.concatenate(N_edges, axis=1)  # (n_q, 12)
        return jnp.concatenate([N_corner, N_edges], axis=1)  # (n_q,20)

    def shape_grads_ref(self) -> jnp.ndarray:
        qp = self.quad_points
        xi = qp[:, 0:1]
        eta = qp[:, 1:2]
        zeta = qp[:, 2:3]

        dtype = default_dtype()
        s = jnp.array(
            [
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
            ],
            dtype=dtype,
        )
        sx = s[:, 0]
        sy = s[:, 1]
        sz = s[:, 2]
        term = xi * sx + eta * sy + zeta * sz - 2.0

        dN_dxi_corner = (sx / 8.0) * (1 + sy * eta) * (1 + sz * zeta) * (term + (1 + sx * xi))
        dN_deta_corner = (sy / 8.0) * (1 + sx * xi) * (1 + sz * zeta) * (term + (1 + sy * eta))
        dN_dzeta_corner = (sz / 8.0) * (1 + sx * xi) * (1 + sy * eta) * (term + (1 + sz * zeta))

        d_corner = jnp.stack(
            [
                dN_dxi_corner,
                dN_deta_corner,
                dN_dzeta_corner,
            ],
            axis=2,
        )  # (n_q,8,3)

        # edges derivatives
        def d_edge_x(sy_, sz_):
            dxi = -0.5 * xi * (1 + sy_ * eta) * (1 + sz_ * zeta)
            deta = 0.25 * (1 - xi * xi) * sy_ * (1 + sz_ * zeta)
            dzeta = 0.25 * (1 - xi * xi) * (1 + sy_ * eta) * sz_
            return jnp.stack([dxi, deta, dzeta], axis=2)

        def d_edge_y(sx_, sz_):
            dxi = 0.25 * (1 - eta * eta) * sx_ * (1 + sz_ * zeta)
            deta = -0.5 * eta * (1 + sx_ * xi) * (1 + sz_ * zeta)
            dzeta = 0.25 * (1 - eta * eta) * (1 + sx_ * xi) * sz_
            return jnp.stack([dxi, deta, dzeta], axis=2)

        def d_edge_z(sx_, sy_):
            dxi = 0.25 * (1 - zeta * zeta) * sx_ * (1 + sy_ * eta)
            deta = 0.25 * (1 - zeta * zeta) * (1 + sx_ * xi) * sy_
            dzeta = -0.5 * zeta * (1 + sx_ * xi) * (1 + sy_ * eta)
            return jnp.stack([dxi, deta, dzeta], axis=2)

        d_list = [
            d_edge_x(-1, -1),
            d_edge_y(1, -1),
            d_edge_x(1, -1),
            d_edge_y(-1, -1),
            d_edge_x(-1, 1),
            d_edge_y(1, 1),
            d_edge_x(1, 1),
            d_edge_y(-1, 1),
            d_edge_z(-1, -1),
            d_edge_z(1, -1),
            d_edge_z(1, 1),
            d_edge_z(-1, 1),
        ]

        d_edges = jnp.concatenate(d_list, axis=1)  # (n_q,12,3)
        return jnp.concatenate([d_corner, d_edges], axis=1)  # (n_q,20,3)

    def spatial_grads_and_detJ(
        self, elem_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dN_dxi = self.shape_grads_ref()  # (n_q, 20, 3)
        J = jnp.einsum("ia,qik->qak", elem_coords, dN_dxi)  # (n_q, 3, 3)
        J_inv = jnp.linalg.inv(J)
        detJ = jnp.linalg.det(J)
        dN_dx = jnp.einsum("qik,qka->qia", dN_dxi, J_inv)
        return dN_dx, detJ


@dataclass(eq=False)
class HexTriQuadraticBasis27(SmallStrainBMixin, TotalLagrangeBMixin):
    """27-node triquadratic hex (tensor-product)."""

    quad_points: jnp.ndarray
    quad_weights: jnp.ndarray
    dofs_per_node: int = 3
    @property
    def n_nodes(self) -> int:
        return 27

    def tree_flatten(self):
        return (self.quad_points, self.quad_weights), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        qp, qw = children
        return cls(qp, qw)

    @property
    def n_q(self) -> int:
        return int(self.quad_points.shape[0])

    def shape_functions(self) -> jnp.ndarray:
        qp = self.quad_points
        xi = qp[:, 0]
        eta = qp[:, 1]
        zeta = qp[:, 2]

        (Nx0, Nx1, Nx2), _ = _quad1d_full(xi)
        (Ny0, Ny1, Ny2), _ = _quad1d_full(eta)
        (Nz0, Nz1, Nz2), _ = _quad1d_full(zeta)

        N = []
        for k, Nk in enumerate([Nz0, Nz1, Nz2]):
            for j, Nj in enumerate([Ny0, Ny1, Ny2]):
                for i, Ni in enumerate([Nx0, Nx1, Nx2]):
                    N.append(Ni * Nj * Nk)
        return jnp.stack(N, axis=1)  # (n_q, 27)

    def shape_grads_ref(self) -> jnp.ndarray:
        qp = self.quad_points
        xi = qp[:, 0]
        eta = qp[:, 1]
        zeta = qp[:, 2]

        (Nx0, Nx1, Nx2), (dNx0, dNx1, dNx2) = _quad1d_full(xi)
        (Ny0, Ny1, Ny2), (dNy0, dNy1, dNy2) = _quad1d_full(eta)
        (Nz0, Nz1, Nz2), (dNz0, dNz1, dNz2) = _quad1d_full(zeta)

        grads = []
        for k in range(3):
            Nz = [Nz0, Nz1, Nz2][k]
            dNz = [dNz0, dNz1, dNz2][k]
            for j in range(3):
                Ny = [Ny0, Ny1, Ny2][j]
                dNy = [dNy0, dNy1, dNy2][j]
                for i in range(3):
                    Nx = [Nx0, Nx1, Nx2][i]
                    dNx = [dNx0, dNx1, dNx2][i]
                    dxi = dNx * Ny * Nz
                    deta = Nx * dNy * Nz
                    dzeta = Nx * Ny * dNz
                    grads.append(jnp.stack([dxi, deta, dzeta], axis=1))
        return jnp.stack(grads, axis=1)  # (n_q, 27, 3)

    def spatial_grads_and_detJ(
        self, elem_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dN_dxi = self.shape_grads_ref()  # (n_q, 27, 3)
        J = jnp.einsum("ia,qik->qak", elem_coords, dN_dxi)
        J_inv = jnp.linalg.inv(J)
        detJ = jnp.linalg.det(J)
        dN_dx = jnp.einsum("qik,qka->qia", dN_dxi, J_inv)
        return dN_dx, detJ


@jax.tree_util.register_pytree_node_class
class TetLinearBasisPytree(TetLinearBasis):
    pass


@jax.tree_util.register_pytree_node_class
class TetQuadraticBasis10Pytree(TetQuadraticBasis10):
    pass


@jax.tree_util.register_pytree_node_class
class HexTriLinearBasisPytree(HexTriLinearBasis):
    pass


@jax.tree_util.register_pytree_node_class
class HexSerendipityBasis20Pytree(HexSerendipityBasis20):
    pass


@jax.tree_util.register_pytree_node_class
class HexTriQuadraticBasis27Pytree(HexTriQuadraticBasis27):
    pass


def _gauss_legendre_1d(order: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """1D Gauss-Legendre points and weights of given order."""
    if order <= 0:
        raise ValueError("quadrature order must be positive")
    pts, wts = np.polynomial.legendre.leggauss(order)
    dtype = default_dtype()
    return jnp.array(pts, dtype=dtype), jnp.array(wts, dtype=dtype)


def _gl_points_for_degree(degree: int) -> int:
    """
    Map polynomial exactness degree to Gauss-Legendre point count in 1D.
    n points integrate degree (2n-1) exactly.
    """
    if degree <= 0:
        return 1
    return int(np.ceil((degree + 1) / 2))


def _tet_quadrature(degree: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Degree-based quadrature rules for reference tetra (volume = 1/6).
    degree<=1: 1-point; degree<=2: 4-point; degree>=3: 5-point (Stroud T3-5).
    """
    if degree <= 1:
        dtype = default_dtype()
        qp = jnp.array([[0.25, 0.25, 0.25]], dtype=dtype)
        qw = jnp.array([1.0 / 6.0], dtype=dtype)
        return qp, qw
    if degree <= 2:
        dtype = default_dtype()
        qp = jnp.array(
            [
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660],
            ],
            dtype=dtype,
        )
        qw = jnp.full((4,), (1.0 / 24.0), dtype=dtype)
        return qp, qw
    # degree 3 rule: centroid + 4 symmetric points
    dtype = default_dtype()
    qp = jnp.array(
        [
            [0.25, 0.25, 0.25],
            [0.50, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 0.50, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 0.50],
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        ],
        dtype=dtype,
    )
    qw = jnp.array(
        [-2.0 / 15.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0],
        dtype=dtype,
    )
    return qp, qw


def make_tet_basis(intorder: int = 2) -> TetLinearBasis:
    """Create a linear tet basis with degree-based quadrature."""
    qp, qw = _tet_quadrature(intorder)
    return TetLinearBasis(qp, qw)


def make_tet_basis_pytree(intorder: int = 2) -> TetLinearBasisPytree:
    """Create a pytree linear tet basis with degree-based quadrature."""
    qp, qw = _tet_quadrature(intorder)
    return TetLinearBasisPytree(qp, qw)


def make_hex_basis(intorder: int = 2) -> HexTriLinearBasis:
    """
    Trilinear hex basis with tensor-product Gauss-Legendre quadrature.
    intorder = polynomial exactness degree (scikit-fem style).
    degree=1 → 1×1×1, degree=2/3 → 2×2×2, degree=4/5 → 3×3×3, etc.
    """
    n_1d = _gl_points_for_degree(intorder)
    pt_1d, wt_1d = _gauss_legendre_1d(n_1d)
    xi, eta, zeta = jnp.meshgrid(pt_1d, pt_1d, pt_1d, indexing="ij")
    qp = jnp.stack([xi, eta, zeta], axis=-1).reshape(-1, 3)  # (intorder^3, 3)

    w = jnp.meshgrid(wt_1d, wt_1d, wt_1d, indexing="ij")
    qw = jnp.stack(w, axis=-1).prod(axis=-1).reshape(-1)  # (intorder^3,)
    return HexTriLinearBasis(qp, qw)


def make_hex_basis_pytree(intorder: int = 2) -> HexTriLinearBasisPytree:
    """Create a pytree trilinear hex basis with tensor-product quadrature."""
    n_1d = _gl_points_for_degree(intorder)
    pt_1d, wt_1d = _gauss_legendre_1d(n_1d)
    xi, eta, zeta = jnp.meshgrid(pt_1d, pt_1d, pt_1d, indexing="ij")
    qp = jnp.stack([xi, eta, zeta], axis=-1).reshape(-1, 3)

    w = jnp.meshgrid(wt_1d, wt_1d, wt_1d, indexing="ij")
    qw = jnp.stack(w, axis=-1).prod(axis=-1).reshape(-1)
    return HexTriLinearBasisPytree(qp, qw)


def make_hex20_basis(intorder: int = 2) -> HexSerendipityBasis20:
    """Create a serendipity hex basis with tensor-product quadrature."""
    n_1d = _gl_points_for_degree(intorder)
    pt_1d, wt_1d = _gauss_legendre_1d(n_1d)
    xi, eta, zeta = jnp.meshgrid(pt_1d, pt_1d, pt_1d, indexing="ij")
    qp = jnp.stack([xi, eta, zeta], axis=-1).reshape(-1, 3)

    w = jnp.meshgrid(wt_1d, wt_1d, wt_1d, indexing="ij")
    qw = jnp.stack(w, axis=-1).prod(axis=-1).reshape(-1)
    return HexSerendipityBasis20(qp, qw)


def make_hex20_basis_pytree(intorder: int = 2) -> HexSerendipityBasis20Pytree:
    """Create a pytree serendipity hex basis with tensor-product quadrature."""
    n_1d = _gl_points_for_degree(intorder)
    pt_1d, wt_1d = _gauss_legendre_1d(n_1d)
    xi, eta, zeta = jnp.meshgrid(pt_1d, pt_1d, pt_1d, indexing="ij")
    qp = jnp.stack([xi, eta, zeta], axis=-1).reshape(-1, 3)

    w = jnp.meshgrid(wt_1d, wt_1d, wt_1d, indexing="ij")
    qw = jnp.stack(w, axis=-1).prod(axis=-1).reshape(-1)
    return HexSerendipityBasis20Pytree(qp, qw)


def make_hex27_basis(intorder: int = 3) -> HexTriQuadraticBasis27:
    """Create a triquadratic hex basis with tensor-product quadrature."""
    n_1d = _gl_points_for_degree(intorder)
    pt_1d, wt_1d = _gauss_legendre_1d(n_1d)
    xi, eta, zeta = jnp.meshgrid(pt_1d, pt_1d, pt_1d, indexing="ij")
    qp = jnp.stack([xi, eta, zeta], axis=-1).reshape(-1, 3)
    w = jnp.meshgrid(wt_1d, wt_1d, wt_1d, indexing="ij")
    qw = jnp.stack(w, axis=-1).prod(axis=-1).reshape(-1)
    return HexTriQuadraticBasis27(qp, qw)


def make_hex27_basis_pytree(intorder: int = 3) -> HexTriQuadraticBasis27Pytree:
    """Create a pytree triquadratic hex basis with tensor-product quadrature."""
    n_1d = _gl_points_for_degree(intorder)
    pt_1d, wt_1d = _gauss_legendre_1d(n_1d)
    xi, eta, zeta = jnp.meshgrid(pt_1d, pt_1d, pt_1d, indexing="ij")
    qp = jnp.stack([xi, eta, zeta], axis=-1).reshape(-1, 3)
    w = jnp.meshgrid(wt_1d, wt_1d, wt_1d, indexing="ij")
    qw = jnp.stack(w, axis=-1).prod(axis=-1).reshape(-1)
    return HexTriQuadraticBasis27Pytree(qp, qw)


def make_tet10_basis(intorder: int = 2) -> TetQuadraticBasis10:
    """Create a quadratic tet basis with degree-based quadrature."""
    qp, qw = _tet_quadrature(intorder)
    return TetQuadraticBasis10(qp, qw)


def make_tet10_basis_pytree(intorder: int = 2) -> TetQuadraticBasis10Pytree:
    """Create a pytree quadratic tet basis with degree-based quadrature."""
    qp, qw = _tet_quadrature(intorder)
    return TetQuadraticBasis10Pytree(qp, qw)

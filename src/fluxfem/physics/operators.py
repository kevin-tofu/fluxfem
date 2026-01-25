

# fluxfem/mechanics/operators.py
from __future__ import annotations

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp

from ..core.context_types import FormFieldLike

ArrayLike: TypeAlias = jnp.ndarray


def dot(a: FormFieldLike | ArrayLike, b: ArrayLike) -> ArrayLike:
    """
    Batched matrix product on the last two axes.

    If the first argument is a FormField, dispatch to vector_load_form to build
    the linear form contribution for a vector load.
    """
    if hasattr(a, "N") and getattr(a, "value_dim", None) is not None:
        from ..core.forms import vector_load_form
        return vector_load_form(a, b)
    return jnp.matmul(a, b)


def ddot(a: ArrayLike, b: ArrayLike, c: ArrayLike | None = None) -> ArrayLike:
    """
    Double contraction on the last two axes.

    - ddot(a, b): sum_ij a_ij * b_ij
    - ddot(a, b, c): a^T b c (Voigt-style linear elasticity blocks)
    """
    if c is None:
        return jnp.einsum("...ij,...ij->...", a, b)
    a_t = jnp.swapaxes(a, -1, -2)
    return jnp.einsum("...ik,kl,...lm->...im", a_t, b, c)


def transpose_last2(a: ArrayLike) -> ArrayLike:
    """Swap the last two axes (batched transpose)."""
    return jnp.swapaxes(a, -1, -2)


def sym_grad(field: FormFieldLike) -> jnp.ndarray:
    """
    Symmetric gradient operator for vector mechanics (small strain).

    Parameters
    ----------
    field : FormField-like
        Must provide:
          - field.gradN : (n_q, n_nodes, 3)
          - field.basis.dofs_per_node (usually 3)

    Returns
    -------
    B : jnp.ndarray
        (n_q, 6, dofs_per_node*n_nodes) Voigt order [xx, yy, zz, xy, yz, zx]
        Such that eps_voigt(q,:) = B(q,:,:) @ u_elem
    """
    gradN = field.gradN                      # (n_q, n_nodes, 3)
    dofs = getattr(field.basis, "dofs_per_node", 3)
    n_q, n_nodes, _ = gradN.shape
    n_dofs = dofs * n_nodes

    def B_single(dN):
        B = jnp.zeros((6, n_dofs), dtype=dN.dtype)

        def node_fun(a, B):
            dNdx, dNdy, dNdz = dN[a, 0], dN[a, 1], dN[a, 2]
            col = dofs * a

            # eps_xx, eps_yy, eps_zz
            B = B.at[0, col + 0].set(dNdx)  # dux/dx
            B = B.at[1, col + 1].set(dNdy)  # duy/dy
            B = B.at[2, col + 2].set(dNdz)  # duz/dz

            # eps_xy = 1/2(dux/dy + duy/dx)
            B = B.at[3, col + 0].set(dNdy)
            B = B.at[3, col + 1].set(dNdx)

            # eps_yz = 1/2(duy/dz + duz/dy)
            B = B.at[4, col + 1].set(dNdz)
            B = B.at[4, col + 2].set(dNdy)

            # eps_zx = 1/2(duz/dx + dux/dz)
            B = B.at[5, col + 0].set(dNdz)
            B = B.at[5, col + 2].set(dNdx)
            return B

        return jax.lax.fori_loop(0, n_nodes, node_fun, B)

    return jax.vmap(B_single)(gradN)


def sym_grad_u(field: FormFieldLike, u_elem: jnp.ndarray) -> jnp.ndarray:
    """
    Apply sym_grad(field) to a local displacement vector.

    Parameters
    ----------
    field : FormField-like
        Vector field basis data.
    u_elem : jnp.ndarray
        Element displacement vector (dofs_per_node*n_nodes,).

    Returns
    -------
    jnp.ndarray
        Symmetric strain in Voigt form with shape (n_q, 6).
    """
    B = sym_grad(field)
    return jnp.einsum("qik,k->qi", B, u_elem)

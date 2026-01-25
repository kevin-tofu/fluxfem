from typing import Mapping, TYPE_CHECKING, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from ...core.forms import FormContext
from ...core.space import FESpace
from ...mesh import BaseMesh
from ...core.basis import build_B_matrices_finite
from ..postprocess import make_point_data_displacement, write_point_data_vtu

if TYPE_CHECKING:
    from jax import Array as JaxArray

    ArrayLike: TypeAlias = np.ndarray | JaxArray
else:
    ArrayLike: TypeAlias = np.ndarray
ParamsLike: TypeAlias = Mapping[str, float] | tuple[float, float]


def right_cauchy_green(F: jnp.ndarray) -> jnp.ndarray:
    """C = F^T F (right Cauchy-Green)."""
    return jnp.einsum("...ik,...jk->...ij", F, F)


def green_lagrange_strain(F: jnp.ndarray) -> jnp.ndarray:
    """E = 0.5 (C - I)."""
    I = jnp.eye(F.shape[-1], dtype=F.dtype)
    C = right_cauchy_green(F)
    return 0.5 * (C - I)


def deformation_gradient(ctx: FormContext, u_elem: jnp.ndarray) -> jnp.ndarray:
    """
    Compute deformation gradient F = I + grad_u for a 3D vector displacement.

    ctx: FormContext with test/trial grads (reference configuration)
    u_elem: (n_ldofs,) element displacement in dof ordering [u0,v0,w0, u1,v1,w1, ...]
    returns: (n_q, 3, 3) F per quadrature point
    """
    u_nodes = u_elem.reshape(-1, 3)  # (n_nodes, 3)
    grad_u = jnp.einsum("qaj,ai->qij", ctx.trial.gradN, u_nodes)  # ∂u_i/∂X_j
    I = jnp.eye(3, dtype=u_elem.dtype)
    return I[None, ...] + grad_u


def pk2_neo_hookean(F: jnp.ndarray, mu: float, lam: float) -> jnp.ndarray:
    """
    Compressible Neo-Hookean PK2 stress:
      S = mu * (I - C^{-1}) + lam * ln J * C^{-1}
      C = F^T F, J = sqrt(det C)
    """
    C = right_cauchy_green(F)
    C_inv = jnp.linalg.inv(C)
    J = jnp.sqrt(jnp.linalg.det(C))
    I = jnp.eye(3, dtype=F.dtype)
    return mu * (I - C_inv) + lam * jnp.log(J)[..., None, None] * C_inv


def neo_hookean_residual_form(
    ctx: FormContext, u_elem: jnp.ndarray, params: ParamsLike
) -> jnp.ndarray:
    """
    Compressible Neo-Hookean residual (Total Lagrangian, PK2).
    params: dict-like with keys \"mu\", \"lam\" or tuple (mu, lam)
    returns: (n_q, n_ldofs)
    """
    if isinstance(params, dict):
        mu = params["mu"]
        lam = params["lam"]
    else:
        mu, lam = params

    F = deformation_gradient(ctx, u_elem)          # (n_q, 3, 3)
    S = pk2_neo_hookean(F, mu, lam)                # (n_q, 3, 3)

    S_voigt = jnp.stack(
        [
            S[..., 0, 0],
            S[..., 1, 1],
            S[..., 2, 2],
            S[..., 0, 1],
            S[..., 1, 2],
            S[..., 2, 0],
        ],
        axis=-1,
    )  # (n_q, 6)

    B = build_B_matrices_finite(ctx.trial.gradN, F)           # (n_q, 6, n_ldofs)
    BT = jnp.swapaxes(B, 1, 2)                               # (n_q, n_ldofs, 6)
    return jnp.einsum("qik,qk->qi", BT, S_voigt)   # (n_q, n_ldofs)


neo_hookean_residual_form._ff_kind = "residual"
neo_hookean_residual_form._ff_domain = "volume"

__all__ = [
    "right_cauchy_green",
    "green_lagrange_strain",
    "deformation_gradient",
    "pk2_neo_hookean",
    "neo_hookean_residual_form",
    "make_elastic_point_data",
    "write_elastic_vtu",
]


def make_elastic_point_data(
    mesh: BaseMesh,
    space: FESpace,
    u: ArrayLike,
    *,
    compute_j: bool = True,
    deformed_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Alias to postprocess.make_point_data_displacement for backward compatibility."""
    return make_point_data_displacement(mesh, space, u, compute_j=compute_j, deformed_scale=deformed_scale)


def write_elastic_vtu(
    mesh: BaseMesh,
    space: FESpace,
    u: ArrayLike,
    filepath: str,
    *,
    compute_j: bool = True,
    deformed_scale: float = 1.0,
) -> None:
    """Alias to postprocess.write_point_data_vtu for backward compatibility."""
    return write_point_data_vtu(mesh, space, u, filepath, compute_j=compute_j, deformed_scale=deformed_scale)

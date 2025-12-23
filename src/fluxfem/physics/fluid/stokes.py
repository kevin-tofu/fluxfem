from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from ...core.forms import FormContext
from ...core.space import FESpace, make_hex_space, make_hex20_space, make_tet_space, make_tet10_space
from ...solver import FluxSparseMatrix


@dataclass
class StokesSpaces:
    """Velocity/pressure spaces for mixed Stokes problems."""
    velocity: FESpace
    pressure: FESpace

    @property
    def n_dofs(self) -> int:
        return int(self.velocity.n_dofs + self.pressure.n_dofs)

    @property
    def offsets(self) -> Tuple[int, int]:
        vel_off = 0
        pres_off = int(self.velocity.n_dofs)
        return vel_off, pres_off


def make_stokes_spaces(mesh, *, vel_dim: int = 3, vel_intorder: int = 2, pres_intorder: int = 1) -> StokesSpaces:
    """
    Build Taylor–Hood-style spaces on a single mesh.
    - Hex mesh: velocity = Hex20 (quadratic), pressure = Hex8 (linear)
    - Tet mesh: velocity = Tet10 (quadratic), pressure = Tet4 (linear)
    """
    if mesh.__class__.__name__.startswith("Hex"):
        vel_space = make_hex20_space(mesh, dim=vel_dim, intorder=vel_intorder)
        pres_space = make_hex_space(mesh, dim=1, intorder=pres_intorder)
    else:
        vel_space = make_tet10_space(mesh, dim=vel_dim, intorder=max(2, vel_intorder))
        pres_space = make_tet_space(mesh, dim=1, intorder=pres_intorder)
    return StokesSpaces(velocity=vel_space, pressure=pres_space)


def _viscosity_form(ctx: FormContext, mu: float) -> jnp.ndarray:
    """μ * grad(v) : grad(u) for vector velocity."""
    grad_v = ctx.test.gradN
    grad_u = ctx.trial.gradN
    G = jnp.einsum("qia,qja->qij", grad_v, grad_u)
    return mu * G


def assemble_viscosity_matrix(space: FESpace, mu: float) -> FluxSparseMatrix:
    """Velocity-velocity block A."""
    return space.assemble_bilinear_form(_viscosity_form, params=mu)


def assemble_divergence_block(spaces: StokesSpaces) -> Tuple[FluxSparseMatrix, FluxSparseMatrix]:
    """
    Assemble B (pressure test, velocity trial) and its transpose BT.
    B: rows = pressure dofs, cols = velocity dofs.
    """
    vel, pres = spaces.velocity, spaces.pressure
    vel_ctxs = vel.build_form_contexts()
    pres_ctxs = pres.build_form_contexts()

    wJ = vel_ctxs.w * vel_ctxs.test.detJ  # (n_elem, n_q)

    def per_element(p_ctx: FormContext, v_ctx: FormContext, w_q: jnp.ndarray):
        Np = p_ctx.test.N                         # (n_q, n_pnodes)
        gradNv = v_ctx.trial.gradN                # (n_q, n_vnodes, 3)
        # div(velocity basis) per q,a,node,comp
        div_block = jnp.einsum("qa,qbj->qabj", Np, gradNv)   # (n_q, n_p, n_vnodes, 3)
        be = jnp.einsum("qabj,q->abj", div_block, w_q)       # integrate over q → (n_p, n_vnodes, 3)
        return be.reshape(Np.shape[1], -1)                   # (n_p, n_vldofs)

    B_e_all = jax.vmap(per_element)(pres_ctxs, vel_ctxs, wJ)    # (n_elem, n_p, n_vldofs)

    rows = jnp.repeat(pres.elem_dofs, vel.n_ldofs, axis=1).reshape(-1) + vel.n_dofs
    cols = jnp.tile(vel.elem_dofs, (1, pres.n_ldofs)).reshape(-1)
    data = B_e_all.reshape(-1)

    n_total = vel.n_dofs + pres.n_dofs
    B = FluxSparseMatrix(rows, cols, data, n_dofs=n_total)
    BT = FluxSparseMatrix(cols, rows, data, n_dofs=n_total)
    return B, BT


def assemble_stokes_system(spaces: StokesSpaces, mu: float):
    """
    Assemble block matrices (A, B, BT) for steady Stokes without stabilization.
    Returns tuple (A, B, BT).
    """
    A = assemble_viscosity_matrix(spaces.velocity, mu)
    B, BT = assemble_divergence_block(spaces)
    return A, B, BT


__all__ = [
    "StokesSpaces",
    "make_stokes_spaces",
    "assemble_viscosity_matrix",
    "assemble_divergence_block",
    "assemble_stokes_system",
]

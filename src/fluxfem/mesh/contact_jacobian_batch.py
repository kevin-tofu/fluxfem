from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ContactBatchJacobianGate:
    enabled: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContactBatchJacobianStack:
    Na: Any
    Nb: Any
    gradNa: Any
    gradNb: Any
    x_q: Any
    w: Any
    detJ: Any
    normal: Any
    u_local: Any
    dofs: np.ndarray
    n_a_local: int
    n_b_local: int
    batch_n: int


def contact_batch_jacobian_gate(
    *,
    requested: bool,
    backend: str,
    dof_source: str,
    grad_source: str,
    use_elem_a: bool,
    use_elem_b: bool,
    use_p0_a: bool,
    use_p0_b: bool,
    distinct_trial_layout: bool,
    proj_diag: bool,
    diag_force: bool,
) -> ContactBatchJacobianGate:
    reasons: list[str] = []
    if not requested:
        reasons.append("not_requested")
    if backend != "jax":
        reasons.append(f"backend={backend}")
    if dof_source != "volume":
        reasons.append(f"dof_source={dof_source}")
    if grad_source != "volume":
        reasons.append(f"grad_source={grad_source}")
    if not use_elem_a:
        reasons.append("missing_elem_a")
    if not use_elem_b:
        reasons.append("missing_elem_b")
    if use_p0_a:
        reasons.append("space_mode_a=p0")
    if use_p0_b:
        reasons.append("space_mode_b=p0")
    if distinct_trial_layout:
        reasons.append("distinct_trial_layout")
    if proj_diag:
        reasons.append("projection_diag")
    if diag_force:
        reasons.append("diag_force")
    return ContactBatchJacobianGate(enabled=not reasons, reasons=tuple(reasons))


def stack_contact_batch_items(
    batch_items: Sequence[tuple[Any, Any, Any, Any, Any, Any, Any, Any]],
    dofs_batch: Sequence[np.ndarray],
    u_local_batch: Sequence[np.ndarray],
    *,
    n_a_local: int,
    n_b_local: int,
) -> ContactBatchJacobianStack:
    Na_b, Nb_b, gradNa_b, gradNb_b, x_q_b, w_b, detJ_b, normal_b = zip(*batch_items)
    return ContactBatchJacobianStack(
        Na=jnp.asarray(np.stack(Na_b, axis=0)),
        Nb=jnp.asarray(np.stack(Nb_b, axis=0)),
        gradNa=jnp.asarray(np.stack(gradNa_b, axis=0)),
        gradNb=jnp.asarray(np.stack(gradNb_b, axis=0)),
        x_q=jnp.asarray(np.stack(x_q_b, axis=0)),
        w=jnp.asarray(np.stack(w_b, axis=0)),
        detJ=jnp.asarray(np.array(detJ_b, dtype=float)).reshape(-1, 1),
        normal=jnp.asarray(np.stack(normal_b, axis=0)),
        u_local=jnp.asarray(np.stack(u_local_batch, axis=0)),
        dofs=np.asarray(dofs_batch, dtype=int),
        n_a_local=int(n_a_local),
        n_b_local=int(n_b_local),
        batch_n=int(len(batch_items)),
    )


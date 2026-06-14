from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class NewtonJaxResult:
    converged: Any
    iters: Any
    residual_norm: Any
    residual0: Any
    stopping_criterion: Any


def _as_dense_jax(mat: Any) -> jnp.ndarray:
    if hasattr(mat, "to_dense"):
        mat = mat.to_dense()
    return jnp.asarray(mat)


def newton_solve_jax(
    residual_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
    jacobian_fn: Callable[[jnp.ndarray, Any], Any],
    u0: jnp.ndarray,
    params: Any,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    diagonal_shift: float = 0.0,
) -> tuple[jnp.ndarray, NewtonJaxResult]:
    """Autodiff-friendly dense Newton solve with fixed iteration schedule."""
    u0 = jnp.asarray(u0)
    r0 = jnp.asarray(residual_fn(u0, params))
    res0_inf = jnp.linalg.norm(r0, ord=jnp.inf)
    crit = jnp.maximum(jnp.asarray(atol, dtype=r0.dtype), jnp.asarray(tol, dtype=r0.dtype) * res0_inf)
    shift = jnp.asarray(diagonal_shift, dtype=r0.dtype)

    def body(carry, _):
        u, r, converged, iters = carry
        res_inf = jnp.linalg.norm(r, ord=jnp.inf)
        done = converged | (res_inf <= crit)

        def _step(_):
            J = _as_dense_jax(jacobian_fn(u, params))
            if diagonal_shift != 0.0:
                J = J + shift * jnp.eye(J.shape[0], dtype=J.dtype)
            du = jnp.linalg.solve(J, -r)
            u_next = u + du
            r_next = jnp.asarray(residual_fn(u_next, params))
            return u_next, r_next, jnp.asarray(False), iters + 1

        return jax.lax.cond(done, lambda _: (u, r, jnp.asarray(True), iters), _step, operand=None), None

    init = (u0, r0, jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32))
    (u_fin, r_fin, converged_fin, iters_fin), _ = jax.lax.scan(body, init, xs=None, length=int(maxiter))
    res_fin_inf = jnp.linalg.norm(r_fin, ord=jnp.inf)
    converged_fin = converged_fin | (res_fin_inf <= crit)
    return u_fin, NewtonJaxResult(
        converged=converged_fin,
        iters=iters_fin,
        residual_norm=res_fin_inf,
        residual0=res0_inf,
        stopping_criterion=crit,
    )

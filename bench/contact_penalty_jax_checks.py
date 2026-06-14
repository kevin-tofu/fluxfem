from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

import fluxfem as ff


class _PenaltyStub:
    def assemble_residual(self, _res_form, u, params, *, normal_source="master"):
        _ = normal_source
        return params.k * jnp.asarray(u["a"]) - jnp.asarray([params.f])

    def assemble_jacobian(self, _res_form, u, params, *, normal_source="master", sparse=False, backend="jax", batch_jac=None):
        _ = (u, normal_source, sparse, backend, batch_jac)
        return jnp.asarray([[params.k]])


def _dummy_res_form(ctx, u, p):
    _ = (ctx, u, p)
    return {"a": jnp.asarray([0.0])}


def check_update_contact_state_penalty() -> None:
    state = ff.update_contact_state_penalty(
        state={"a": np.zeros((2,), dtype=float)},
        gap_n=np.array([0.2, -0.1], dtype=float),
        lambda_n=np.array([0.0, 3.0], dtype=float),
        penalty_param=10.0,
        metadata={"source": "bench"},
    )

    assert state.iteration == 1
    assert state.active_set == "active"
    assert np.allclose(np.asarray(state.gap_n), np.array([0.2, -0.1]))
    assert np.array_equal(np.asarray(state.active_mask), np.array([False, True]))
    assert np.allclose(np.asarray(state.lambda_n), np.array([0.0, 3.0]))
    assert state.penalty_param == 10.0
    assert state.field_summary == {"a": (2,)}
    assert state.metadata["source"] == "bench"


def check_solve_contact_penalty_jax_stub() -> None:
    result = ff.solve_contact_penalty_jax(
        _PenaltyStub(),
        weak_form=_dummy_res_form,
        state0={"a": jnp.asarray([0.0])},
        params=ff.Params(k=4.0, f=2.0, alpha=7.0),
        maxiter=4,
    )

    assert bool(result.converged)
    assert int(result.iters) == 1
    assert np.allclose(np.asarray(result.state["a"]), np.array([0.5]), atol=1e-12)
    assert result.contact_state.penalty_param == 7.0
    assert result.contact_state.field_summary == {"a": (1,)}


def check_solve_contact_penalty_jax_autodiff() -> None:
    def objective(k):
        result = ff.solve_contact_penalty_jax(
            _PenaltyStub(),
            weak_form=_dummy_res_form,
            state0={"a": jnp.asarray([0.0])},
            params=ff.Params(k=k, f=2.0, alpha=5.0),
            maxiter=4,
        )
        return result.state["a"][0]

    value = float(objective(4.0))
    grad = float(jax.grad(objective)(4.0))

    assert np.isclose(value, 0.5, atol=1e-12)
    assert np.isclose(grad, -0.125, atol=1e-10)


def _build_onesided_contact():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2]], dtype=int)
    surf = ff.SurfaceMesh.from_facets(coords, facets)
    side = ff.ContactSide.from_surfaces(surf, elem_conn=conn, value_dim=3)
    return ff.OneSidedContactSurfaceSpace.from_side(side, quad_order=2)


def _u_hat_fn(x_q):
    x_q = np.asarray(x_q, dtype=float)
    return np.stack(
        [
            0.1 + 0.05 * x_q[:, 0],
            -0.02 + 0.03 * x_q[:, 1],
            0.04 + 0.01 * x_q[:, 2],
        ],
        axis=1,
    )


def check_solve_contact_penalty_jax_real_onesided() -> None:
    contact = _build_onesided_contact()
    params = ff.Params(lam=1.5, mu=0.7, alpha=10.0)
    state0 = {"a": jnp.zeros(int(contact.surface_slave.n_nodes * contact.value_dim), dtype=jnp.float32)}
    result = ff.solve_contact_penalty_jax(
        contact,
        state0=state0,
        params=params,
        u_hat_fn=_u_hat_fn,
        maxiter=4,
        atol=1e-6,
        diagonal_shift=1e-8,
    )

    k_mat, f_vec = contact.assemble_bilinear(_u_hat_fn, params)
    residual = np.asarray(k_mat) @ np.asarray(result.state["a"]) + np.asarray(f_vec)

    assert bool(result.converged)
    assert residual.shape == (int(contact.surface_slave.n_nodes * contact.value_dim),)
    assert np.linalg.norm(residual, ord=np.inf) < 1e-5
    assert result.contact_state.gap_n is not None
    assert result.contact_state.active_mask is not None
    assert np.asarray(result.contact_state.gap_n).shape == (int(contact.surface_slave.n_nodes),)
    assert np.asarray(result.contact_state.active_mask).shape == (int(contact.surface_slave.n_nodes),)
    assert np.linalg.norm(np.asarray(result.contact_state.gap_n), ord=np.inf) < 1e-5


def check_solve_contact_al_jax_real_onesided() -> None:
    contact = _build_onesided_contact()
    params = ff.Params(lam=1.5, mu=0.7, alpha=2.0)
    state0 = {"a": jnp.zeros(int(contact.surface_slave.n_nodes * contact.value_dim), dtype=jnp.float32)}
    result = ff.solve_contact_al_jax(
        contact,
        state0=state0,
        params=params,
        u_hat_fn=_u_hat_fn,
        maxiter=4,
        outer_maxiter=3,
        atol=1e-6,
        gap_tol=1e-5,
        diagonal_shift=1e-8,
    )

    assert result.contact_state.lambda_n is not None
    lam = np.asarray(result.contact_state.lambda_n)
    assert lam.shape == (int(contact.surface_slave.n_nodes),)
    assert np.all(lam >= -1e-12)
    assert result.contact_state.penalty_param is not None
    assert float(result.contact_state.penalty_param) >= 2.0


CHECKS = {
    "state": check_update_contact_state_penalty,
    "stub": check_solve_contact_penalty_jax_stub,
    "autodiff": check_solve_contact_penalty_jax_autodiff,
    "onesided-penalty": check_solve_contact_penalty_jax_real_onesided,
    "onesided-al": check_solve_contact_al_jax_real_onesided,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual checks for contact JAX Newton / AL paths.")
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=["all", *CHECKS.keys()],
        default=["all"],
        help="Subset of checks to run.",
    )
    args = parser.parse_args()

    selected = list(CHECKS) if "all" in args.checks else args.checks
    for name in selected:
        CHECKS[name]()
        print(f"[ok] {name}")


if __name__ == "__main__":
    main()

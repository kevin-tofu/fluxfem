#!/usr/bin/env python3
"""
Inverse heat source ID: Gaussian volumetric source in 2D-ish 3D diffusion.

- Scalar diffusion (Poisson) on a thin 3D box.
- Dirichlet T=0 on x=xmin.
- Recover Gaussian source params from boundary observations (x=xmax).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import fluxfem as ff

jax.config.update("jax_enable_x64", True)


@dataclass
class Config:
    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    intorder: int
    kappa: float
    true_cx: float
    true_cy: float
    true_log_sigma: float
    true_logA: float
    init_cx: float
    init_cy: float
    init_log_sigma: float
    init_logA: float
    noise_std: float
    obs_fraction: float
    obs_boundary: str
    obs_region: str
    obs_interior_fraction: float
    steps: int
    lr: float
    reg_sigma: float
    reg_A: float
    stage_steps: str
    cases: int
    seed: int
    out_json: str
    out_vtu_true: str
    out_vtu_est: str


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Gaussian heat source identification (FluxFEM).")
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=20)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--ly", type=float, default=1.0)
    p.add_argument("--lz", type=float, default=0.05)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--true-cx", type=float, default=0.7)
    p.add_argument("--true-cy", type=float, default=0.5)
    p.add_argument("--true-log-sigma", type=float, default=-2.3)
    p.add_argument("--true-logA", type=float, default=0.0)
    p.add_argument("--init-cx", type=float, default=0.3)
    p.add_argument("--init-cy", type=float, default=0.3)
    p.add_argument("--init-log-sigma", type=float, default=-1.5)
    p.add_argument("--init-logA", type=float, default=-0.2)
    p.add_argument("--noise-std", type=float, default=1e-4)
    p.add_argument("--obs-fraction", type=float, default=0.5, help="Fraction of boundary nodes observed.")
    p.add_argument("--obs-internal-fraction", type=float, default=0.1, help="Fraction of interior nodes (mixed).")
    p.add_argument("--obs-boundary", type=str, default="all", choices=["xmax", "all"])
    p.add_argument("--obs-region", type=str, default="mixed", choices=["boundary", "interior", "mixed"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--reg-sigma", type=float, default=1e-4)
    p.add_argument("--reg-A", type=float, default=1e-4)
    p.add_argument(
        "--stage-steps",
        type=str,
        default="0,0,0",
        help="CSV: (fix_sigma_amp, fix_sigma_only, free_all)",
    )
    p.add_argument("--cases", type=int, default=2, choices=[1, 2], help="Number of BC cases.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=str, default="result/bench/gaussian_heat_source_id.json")
    p.add_argument("--out-vtu-true", type=str, default="result/bench/gaussian_heat_source_true.vtu")
    p.add_argument("--out-vtu-est", type=str, default="result/bench/gaussian_heat_source_est.vtu")
    args = p.parse_args()
    return Config(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        intorder=args.intorder,
        kappa=args.kappa,
        true_cx=args.true_cx,
        true_cy=args.true_cy,
        true_log_sigma=args.true_log_sigma,
        true_logA=args.true_logA,
        init_cx=args.init_cx,
        init_cy=args.init_cy,
        init_log_sigma=args.init_log_sigma,
        init_logA=args.init_logA,
        noise_std=args.noise_std,
        obs_fraction=args.obs_fraction,
        obs_boundary=args.obs_boundary,
        obs_region=args.obs_region,
        obs_interior_fraction=args.obs_internal_fraction,
        steps=args.steps,
        lr=args.lr,
        reg_sigma=args.reg_sigma,
        reg_A=args.reg_A,
        stage_steps=args.stage_steps,
        cases=args.cases,
        seed=args.seed,
        out_json=args.out_json,
        out_vtu_true=args.out_vtu_true,
        out_vtu_est=args.out_vtu_est,
    )


def _unpack_theta(
    theta: jnp.ndarray, mins: jnp.ndarray, maxs: jnp.ndarray, *, fix_sigma=None, fix_amp=None
):
    cx = mins[0] + jax.nn.sigmoid(theta[0]) * (maxs[0] - mins[0])
    cy = mins[1] + jax.nn.sigmoid(theta[1]) * (maxs[1] - mins[1])
    sigma = jnp.exp(theta[2]) if fix_sigma is None else jnp.asarray(fix_sigma, dtype=theta.dtype)
    amp = jnp.exp(theta[3]) if fix_amp is None else jnp.asarray(fix_amp, dtype=theta.dtype)
    return jnp.array([cx, cy]), sigma, amp


def _pack_theta(cxy: np.ndarray, log_sigma: float, logA: float, mins: np.ndarray, maxs: np.ndarray) -> jnp.ndarray:
    span = np.maximum(maxs - mins, 1e-12)
    scaled = (cxy - mins) / span
    scaled = np.clip(scaled, 1e-6, 1.0 - 1e-6)
    raw = jnp.log(scaled / (1.0 - scaled))
    return jnp.array([raw[0], raw[1], log_sigma, logA], dtype=jnp.float64)


def _gaussian_source(
    x_q: jnp.ndarray, theta: jnp.ndarray, mins: jnp.ndarray, maxs: jnp.ndarray, *, fix_sigma=None, fix_amp=None
):
    c, sigma, amp = _unpack_theta(theta, mins, maxs, fix_sigma=fix_sigma, fix_amp=fix_amp)
    r2 = jnp.sum((x_q[:, :2] - c[None, :]) ** 2, axis=1)
    return amp * jnp.exp(-0.5 * r2 / (sigma * sigma))


def _gaussian_body_force_form(mins: jnp.ndarray, maxs: jnp.ndarray, *, fix_sigma=None, fix_amp=None):
    def _form(ctx, params):
        q = _gaussian_source(ctx.x_q, params, mins, maxs, fix_sigma=fix_sigma, fix_amp=fix_amp)
        return q[..., None] * ctx.test.N

    _form._ff_kind = "linear"  # type: ignore[attr-defined]
    _form._ff_domain = "volume"  # type: ignore[attr-defined]
    return _form


def main() -> None:
    cfg = _parse_args()
    rng = np.random.default_rng(cfg.seed)

    mesh = ff.StructuredHexBox(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, lx=cfg.lx, ly=cfg.ly, lz=cfg.lz
    ).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=cfg.intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    coords = np.asarray(mesh.coords)
    mins = jnp.asarray(coords.min(axis=0)[:2])
    maxs = jnp.asarray(coords.max(axis=0)[:2])
    xmin = float(coords[:, 0].min())
    xmax = float(coords[:, 0].max())
    ymin = float(coords[:, 1].min())

    bc_x = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components=[0],
        dof_per_node=1,
    )
    bc_y = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 1], ymin, atol=1e-8),
        components=[0],
        dof_per_node=1,
    )
    cases = [("x", bc_x)]
    if cfg.cases == 2:
        cases.append(("y", bc_y))

    # Preassemble stiffness (kappa constant)
    K0 = jnp.asarray(
        ff.assemble_bilinear_form(
            ff.BilinearSpaces(test=V, trial=U),
            ff.diffusion_form,
            cfg.kappa,
        ).to_dense()
    )

    # Observation dofs
    if cfg.obs_boundary == "xmax":
        boundary_dofs = ff.DirichletBC.from_boundary_dofs(
            mesh,
            lambda pts: np.isclose(pts[:, 0], xmax, atol=1e-8),
            components=[0],
            dof_per_node=1,
        ).dofs
    else:
        boundary_nodes = mesh.boundary_node_indices()
        boundary_dofs = np.asarray(boundary_nodes, dtype=int)
    interior_nodes = np.setdiff1d(np.arange(mesh.n_nodes), mesh.boundary_node_indices())
    interior_dofs = interior_nodes.astype(int)

    obs_idx_cases = []
    for _name, bc in cases:
        boundary_free = np.setdiff1d(boundary_dofs, bc.dofs)
        obs_idx = []
        if cfg.obs_region in ("boundary", "mixed"):
            obs_count = max(2, int(np.ceil(cfg.obs_fraction * boundary_free.size)))
            obs_count = min(obs_count, max(2, boundary_free.size))
            obs_idx.append(rng.choice(boundary_free, size=obs_count, replace=False))
        if cfg.obs_region in ("interior", "mixed"):
            if interior_dofs.size:
                int_count = max(2, int(np.ceil(cfg.obs_interior_fraction * interior_dofs.size)))
                int_count = min(int_count, max(2, interior_dofs.size))
                obs_idx.append(rng.choice(interior_dofs, size=int_count, replace=False))
        if obs_idx:
            obs_idx = np.unique(np.concatenate(obs_idx, axis=0))
        else:
            obs_idx = np.array([], dtype=int)
        obs_idx_cases.append(jnp.asarray(obs_idx))

    form_full = _gaussian_body_force_form(mins, maxs)

    fix_sigma = float(np.exp(cfg.init_log_sigma))
    fix_amp = float(np.exp(cfg.init_logA))
    form_fix_both = _gaussian_body_force_form(mins, maxs, fix_sigma=fix_sigma, fix_amp=fix_amp)
    form_fix_sigma = _gaussian_body_force_form(mins, maxs, fix_sigma=fix_sigma, fix_amp=None)

    def solve_u(theta, case_idx: int, *, fixed=None):
        _, bc = cases[case_idx]
        free_dofs = bc.free_dofs(space.n_dofs)
        free_dofs_j = jnp.asarray(free_dofs)
        if fixed == "both":
            form = form_fix_both
        elif fixed == "sigma":
            form = form_fix_sigma
        else:
            form = form_full
        F = ff.assemble_linear_form(
            ff.LinearSpaces(test=V),
            form,
            theta,
        )
        F = jnp.asarray(F)
        K_ff = K0[free_dofs_j][:, free_dofs_j]
        F_ff = F[free_dofs_j]
        u_free = jnp.linalg.solve(K_ff, F_ff)
        u = jnp.zeros(space.n_dofs, dtype=K0.dtype)
        return u.at[free_dofs_j].set(u_free)

    # Synthetic observations (per case)
    theta_true = _pack_theta(
        np.array([cfg.true_cx, cfg.true_cy], dtype=float),
        cfg.true_log_sigma,
        cfg.true_logA,
        np.asarray(mins),
        np.asarray(maxs),
    )
    u_true_cases = []
    u_obs_cases = []
    for case_idx in range(len(cases)):
        u_true = solve_u(theta_true, case_idx)
        u_obs = u_true + jnp.asarray(
            rng.normal(scale=cfg.noise_std, size=space.n_dofs),
            dtype=jnp.float64,
        )
        u_true_cases.append(u_true)
        u_obs_cases.append(u_obs)

    def loss(theta, *, fixed=None):
        diff_all = []
        for case_idx, obs_idx_j in enumerate(obs_idx_cases):
            if obs_idx_j.size == 0:
                continue
            u = solve_u(theta, case_idx, fixed=fixed)
            diff_all.append(u[obs_idx_j] - u_obs_cases[case_idx][obs_idx_j])
        if not diff_all:
            return jnp.array(0.0, dtype=jnp.float64)
        diff = jnp.concatenate(diff_all, axis=0)
        _, sigma, amp = _unpack_theta(
            theta,
            mins,
            maxs,
            fix_sigma=fix_sigma if fixed in ("both", "sigma") else None,
            fix_amp=fix_amp if fixed == "both" else None,
        )
        reg = cfg.reg_sigma * (jnp.log(sigma) ** 2) + cfg.reg_A * (jnp.log(amp) ** 2)
        return 0.5 * jnp.mean(diff * diff) + reg

    grad_fn = jax.grad(loss)
    grad_fn_fix_both = jax.grad(lambda th: loss(th, fixed="both"))
    grad_fn_fix_sigma = jax.grad(lambda th: loss(th, fixed="sigma"))

    theta = _pack_theta(
        np.array([cfg.init_cx, cfg.init_cy], dtype=float),
        cfg.init_log_sigma,
        cfg.init_logA,
        np.asarray(mins),
        np.asarray(maxs),
    )

    history = []
    stage_steps = [int(x) for x in cfg.stage_steps.split(",")]
    if len(stage_steps) != 3:
        stage_steps = [0, 0, 0]
    total_steps = sum(stage_steps) if sum(stage_steps) > 0 else cfg.steps

    def _log(step, loss_val):
        c, sigma, amp = _unpack_theta(theta, mins, maxs)
        payload = {
            "step": step,
            "loss": loss_val,
            "cx": float(c[0]),
            "cy": float(c[1]),
            "sigma": float(sigma),
            "A": float(amp),
        }
        history.append(payload)
        print(
            f"step={step:03d} loss={loss_val:.3e} "
            f"cx={payload['cx']:.4f} cy={payload['cy']:.4f} "
            f"sigma={payload['sigma']:.4f} A={payload['A']:.4f}"
        )

    step = 0
    # Sanity checks: loss at true params and gradient magnitude at init
    loss_true = float(loss(theta_true))
    g_init = grad_fn(theta)
    print(
        f"[check] loss(theta_true)={loss_true:.3e} "
        f"|g_init|={float(jnp.linalg.norm(g_init)):.3e} "
        f"g_cx={float(g_init[0]):.3e} g_cy={float(g_init[1]):.3e}"
    )
    for _ in range(stage_steps[0]):
        g = grad_fn_fix_both(theta)
        theta = theta - cfg.lr * g
        if step % 10 == 0:
            _log(step, float(loss(theta, fixed="both")))
        step += 1
    for _ in range(stage_steps[1]):
        g = grad_fn_fix_sigma(theta)
        theta = theta - cfg.lr * g
        if step % 10 == 0:
            _log(step, float(loss(theta, fixed="sigma")))
        step += 1
    for _ in range(stage_steps[2]):
        g = grad_fn(theta)
        theta = theta - cfg.lr * g
        if step % 10 == 0:
            _log(step, float(loss(theta)))
        step += 1

    if sum(stage_steps) == 0:
        for _ in range(cfg.steps):
            g = grad_fn(theta)
            theta = theta - cfg.lr * g
            if step % 10 == 0:
                _log(step, float(loss(theta)))
            step += 1

    # Outputs
    out_json = Path(cfg.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "config": cfg.__dict__,
                "theta_true": [float(x) for x in theta_true],
                "theta_est": [float(x) for x in theta],
                "history": history,
                "obs_counts": [int(x.size) for x in obs_idx_cases],
                "cases": [name for name, _bc in cases],
            },
            indent=2,
        )
    )

    u_est = solve_u(theta, 0)
    out_true = Path(cfg.out_vtu_true)
    out_est = Path(cfg.out_vtu_est)
    out_true.parent.mkdir(parents=True, exist_ok=True)
    out_est.parent.mkdir(parents=True, exist_ok=True)
    ff.write_vtu(mesh, str(out_true), point_data={"temperature": np.asarray(u_true_cases[0])})
    ff.write_vtu(mesh, str(out_est), point_data={"temperature": np.asarray(u_est)})

    print(f"VTU true: {out_true}")
    print(f"VTU est:  {out_est}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()

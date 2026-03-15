#!/usr/bin/env python3
"""
Unknown traction identification benchmark (cantilever, linear elasticity).

- 2D-ish setup: thin 3D cantilever (nz=1) for fastest plotting.
- Load parameterization: RBF basis on the x = xmax surface.
- Synthetic observations: displacement at selected sensor nodes.
- Solve ridge regression for RBF coefficients (linear inverse).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxfem as ff

jax.config.update("jax_enable_x64", True)


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@dataclass
class BenchConfig:
    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    intorder: int
    E: float
    nu: float
    rbf_ny: int
    rbf_nz: int
    rbf_sigma: float
    load_basis: str
    bspline_n: int
    bspline_degree: int
    traction_dir: str
    noise_std: float
    reg: float
    sparse_k: int
    sparse_near_tip: bool
    obs_tip_only: bool
    seed: int
    obs_components: str
    out_json: str
    out_plot: str


def _parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(description="Unknown traction identification benchmark.")
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=4)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--ly", type=float, default=0.2)
    p.add_argument("--lz", type=float, default=0.05)
    p.add_argument("--intorder", type=int, default=2)
    p.add_argument("--E", type=float, default=210_000.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--rbf-ny", type=int, default=6)
    p.add_argument("--rbf-nz", type=int, default=2)
    p.add_argument("--rbf-sigma", type=float, default=0.06)
    p.add_argument("--load-basis", type=str, default="rbf", choices=["rbf", "bspline"])
    p.add_argument("--bspline-n", type=int, default=4, help="Number of B-spline control points (1D in y).")
    p.add_argument("--bspline-degree", type=int, default=2, help="B-spline degree (1D in y).")
    p.add_argument("--traction-dir", type=str, default="-y", choices=["x", "y", "z", "-x", "-y", "-z"])
    p.add_argument("--noise-std", type=float, default=0.01, help="Relative noise std (scaled by std(u_obs)).")
    p.add_argument("--reg", type=float, default=1e-6, help="Ridge regularization.")
    p.add_argument("--sparse-k", type=int, default=2, help="Number of nonzero true coefficients.")
    p.add_argument(
        "--sparse-near-tip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pick active coefficients near the loaded tip region.",
    )
    p.add_argument(
        "--obs-tip-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use observation points only on the loaded tip face.",
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument(
        "--obs-components",
        type=str,
        default="y",
        choices=["x", "y", "z", "xy", "xz", "yz", "xyz"],
    )
    p.add_argument("--out-json", type=str, default="result/unknown_load_id.json")
    p.add_argument("--out-plot", type=str, default="result/unknown_load_id.png")
    args = p.parse_args()
    return BenchConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        intorder=args.intorder,
        E=args.E,
        nu=args.nu,
        rbf_ny=args.rbf_ny,
        rbf_nz=args.rbf_nz,
        rbf_sigma=args.rbf_sigma,
        load_basis=args.load_basis,
        bspline_n=args.bspline_n,
        bspline_degree=args.bspline_degree,
        traction_dir=args.traction_dir,
        noise_std=args.noise_std,
        reg=args.reg,
        sparse_k=args.sparse_k,
        sparse_near_tip=args.sparse_near_tip,
        obs_tip_only=args.obs_tip_only,
        seed=args.seed,
        obs_components=args.obs_components,
        out_json=args.out_json,
        out_plot=args.out_plot,
    )


def _dir_to_vec(dir_key: str) -> np.ndarray:
    mapping = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    return mapping[dir_key].astype(float)


def _rbf_load_fn(center: np.ndarray, sigma: float, direction: np.ndarray):
    inv_two_sigma2 = 0.5 / (sigma * sigma)
    center_j = jnp.asarray(center, dtype=jnp.float64)
    direction_j = jnp.asarray(direction, dtype=jnp.float64)

    def _load_fn(x_q: np.ndarray) -> np.ndarray:
        x_q = jnp.asarray(x_q, dtype=center_j.dtype)
        r2 = jnp.sum((x_q - center_j) ** 2, axis=1)
        phi = jnp.exp(-r2 * inv_two_sigma2)
        return phi[:, None] * direction_j[None, :]

    return _load_fn


def _bspline_knots(n_ctrl: int, degree: int, y_min: float, y_max: float) -> np.ndarray:
    if n_ctrl <= degree:
        raise ValueError("bspline_n must be > degree")
    n_knots = n_ctrl + degree + 1
    interior = np.linspace(y_min, y_max, n_knots - 2 * (degree + 1) + 2)
    knots = np.concatenate(
        [
            np.full(degree + 1, y_min),
            interior[1:-1] if interior.size > 2 else np.array([], dtype=float),
            np.full(degree + 1, y_max),
        ]
    )
    return knots


def _bspline_basis(y: np.ndarray, knots: np.ndarray, degree: int, i: int) -> np.ndarray:
    if degree == 0:
        left = knots[i]
        right = knots[i + 1]
        return np.where((y >= left) & (y < right), 1.0, 0.0)
    denom1 = knots[i + degree] - knots[i]
    denom2 = knots[i + degree + 1] - knots[i + 1]
    term1 = 0.0 if denom1 == 0.0 else (y - knots[i]) / denom1 * _bspline_basis(y, knots, degree - 1, i)
    term2 = (
        0.0
        if denom2 == 0.0
        else (knots[i + degree + 1] - y) / denom2 * _bspline_basis(y, knots, degree - 1, i + 1)
    )
    return term1 + term2


def _bspline_basis_matrix(y: np.ndarray, n_ctrl: int, degree: int, y_min: float, y_max: float) -> np.ndarray:
    knots = _bspline_knots(n_ctrl, degree, y_min, y_max)
    Phi = np.stack([_bspline_basis(y, knots, degree, i) for i in range(n_ctrl)], axis=1)
    # Fix right boundary inclusion
    mask = np.isclose(y, y_max)
    Phi[mask, :] = 0.0
    Phi[mask, -1] = 1.0
    return Phi


def _bspline_load_fn(i: int, n_ctrl: int, degree: int, y_min: float, y_max: float, direction: np.ndarray):
    knots = _bspline_knots(n_ctrl, degree, y_min, y_max)
    direction_j = jnp.asarray(direction, dtype=jnp.float64)

    def _load_fn(x_q: np.ndarray) -> np.ndarray:
        y = x_q[:, 1]
        phi = _bspline_basis(y, knots, degree, i)
        phi = jnp.asarray(phi, dtype=direction_j.dtype)
        return phi[:, None] * direction_j[None, :]

    return _load_fn


def _select_sensor_nodes(
    coords: np.ndarray, lx: float, ly: float, lz: float, *, tip_only: bool
) -> np.ndarray:
    xmax = float(coords[:, 0].max())
    if tip_only:
        ys = np.array([0.05 * ly, 0.175 * ly, 0.3 * ly, 0.425 * ly, 0.55 * ly, 0.675 * ly, 0.8 * ly, 0.925 * ly, 0.975 * ly])
        zs = np.array([0.5 * lz])
        targets = np.array([[xmax, y, z] for y in ys for z in zs], dtype=float)
    else:
        # Prefer tip/mid/root + near-top/bottom to see dominant modes.
        xs = np.array([0.05 * lx, 0.5 * lx, 0.95 * lx])
        ys = np.array([0.1 * ly, 0.5 * ly, 0.9 * ly])
        zs = np.array([0.5 * lz])
        targets = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)

        # Add a few points near the loaded face to increase sensitivity to traction.
        tip_targets = np.array(
            [
                [xmax, 0.2 * ly, 0.5 * lz],
                [xmax, 0.5 * ly, 0.5 * lz],
                [xmax, 0.8 * ly, 0.5 * lz],
            ],
            dtype=float,
        )
        targets = np.concatenate([targets, tip_targets], axis=0)

    nodes = []
    for tgt in targets:
        d2 = np.sum((coords - tgt) ** 2, axis=1)
        nodes.append(int(np.argmin(d2)))
    return np.unique(np.array(nodes, dtype=int))


def _obs_dofs(nodes: np.ndarray, components: str, dim: int = 3) -> np.ndarray:
    comp_map = {"x": 0, "y": 1, "z": 2}
    comps = [comp_map[c] for c in components]
    dofs = []
    for n in nodes:
        for c in comps:
            dofs.append(dim * int(n) + c)
    return np.array(dofs, dtype=int)


def _make_basis_centers(coords: np.ndarray, lx: float, rbf_ny: int, rbf_nz: int) -> np.ndarray:
    xmax = float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    ys = np.linspace(y_min, y_max, rbf_ny)
    zs = np.linspace(z_min, z_max, rbf_nz)
    centers = np.array([[xmax, y, z] for y in ys for z in zs], dtype=float)
    return centers


def main() -> None:
    cfg = _parse_args()

    # Mesh and space
    mesh = ff.StructuredHexBox(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, lx=cfg.lx, ly=cfg.ly, lz=cfg.lz
    ).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=cfg.intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)
    coords = np.asarray(mesh.coords)

    # Dirichlet clamp on x = xmin
    xmin = float(coords[:, 0].min())
    bc = ff.DirichletBC.from_boundary_dofs(
        mesh, lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8), components="xyz"
    )
    free_dofs = bc.free_dofs(space.n_dofs)

    # Load surface (x = xmax)
    xmax = float(coords[:, 0].max())
    facets = np.asarray(mesh.boundary_facets_plane(axis=0, value=xmax, tol=1e-8))
    surface = ff.SurfaceMesh.from_facets(coords, facets)

    # Assemble stiffness (dense for simplicity)
    D = ff.isotropic_3d_D(cfg.E, cfg.nu)
    K = np.asarray(
        ff.assemble_bilinear_form(
            ff.BilinearSpaces(test=V, trial=U),
            ff.linear_elasticity_form,
            D,
        ).to_dense()
    )
    K_ff = K[np.ix_(free_dofs, free_dofs)]

    # Basis loads
    direction = _dir_to_vec(cfg.traction_dir)
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    centers = None
    if cfg.load_basis == "rbf":
        centers = _make_basis_centers(coords, cfg.lx, cfg.rbf_ny, cfg.rbf_nz)
        n_basis = centers.shape[0]
        load_fns = [_rbf_load_fn(c, cfg.rbf_sigma, direction) for c in centers]
        basis_meta = {"type": "rbf", "centers": centers}
    else:
        n_basis = cfg.bspline_n
        load_fns = [
            _bspline_load_fn(i, cfg.bspline_n, cfg.bspline_degree, y_min, y_max, direction)
            for i in range(cfg.bspline_n)
        ]
        basis_meta = {"type": "bspline", "n": cfg.bspline_n, "degree": cfg.bspline_degree}

    F_basis = []
    u_basis = []
    for load_fn in load_fns:
        form = ff.make_vector_surface_load_form(load_fn)
        F_k = surface.assemble_linear_form_on_space(space, form, params=None)
        F_basis.append(F_k)
        F_ff = F_k[free_dofs]
        u_ff = np.linalg.solve(K_ff, F_ff)
        u_k = np.zeros(space.n_dofs, dtype=float)
        u_k[free_dofs] = u_ff
        u_basis.append(u_k)

    F_basis = np.stack(F_basis, axis=1)  # (n_dofs, n_basis)
    u_basis = np.stack(u_basis, axis=1)  # (n_dofs, n_basis)

    # Observations
    sensor_nodes = _select_sensor_nodes(
        coords, cfg.lx, cfg.ly, cfg.lz, tip_only=cfg.obs_tip_only
    )
    obs_dofs = _obs_dofs(sensor_nodes, cfg.obs_components)
    U_obs = u_basis[obs_dofs, :]

    rng = np.random.default_rng(cfg.seed)
    a_true = np.zeros(n_basis, dtype=float)
    k = int(np.clip(cfg.sparse_k, 1, n_basis))
    if cfg.sparse_near_tip and cfg.load_basis == "rbf":
        tip_targets = np.array(
            [
                [xmax, 0.2 * cfg.ly, 0.5 * cfg.lz],
                [xmax, 0.5 * cfg.ly, 0.5 * cfg.lz],
                [xmax, 0.8 * cfg.ly, 0.5 * cfg.lz],
            ],
            dtype=float,
        )
        d2 = np.min(
            np.sum((centers[:, None, :] - tip_targets[None, :, :]) ** 2, axis=2), axis=1
        )
        order = np.argsort(d2)
        pool = order[: min(n_basis, max(k, 6))]
        active = rng.choice(pool, size=k, replace=False)
    elif cfg.sparse_near_tip and cfg.load_basis == "bspline":
        ctrl_y = np.linspace(y_min, y_max, n_basis)
        tip_y = np.array([0.2 * cfg.ly, 0.5 * cfg.ly, 0.8 * cfg.ly], dtype=float)
        d2 = np.min((ctrl_y[:, None] - tip_y[None, :]) ** 2, axis=1)
        order = np.argsort(d2)
        pool = order[: min(n_basis, max(k, 3))]
        active = rng.choice(pool, size=k, replace=False)
    else:
        active = rng.choice(n_basis, size=k, replace=False)
    a_true[active] = rng.normal(scale=1.0, size=k)
    a_true *= 1.0 / max(1.0, np.linalg.norm(a_true))
    u_obs = U_obs @ a_true
    noise_scale = cfg.noise_std * max(1e-12, float(np.std(u_obs)))
    u_obs_noisy = u_obs + rng.normal(scale=noise_scale, size=u_obs.shape)

    # Ridge solve
    A = U_obs.T @ U_obs + cfg.reg * np.eye(n_basis)
    b = U_obs.T @ u_obs_noisy
    a_hat = np.linalg.solve(A, b)

    # Reconstructions and metrics
    u_hat = u_basis @ a_hat
    u_true = u_basis @ a_true
    rel_a = float(np.linalg.norm(a_hat - a_true) / (np.linalg.norm(a_true) + 1e-12))
    rel_u = float(np.linalg.norm(u_hat - u_true) / (np.linalg.norm(u_true) + 1e-12))
    rel_obs = float(np.linalg.norm(U_obs @ a_hat - u_obs) / (np.linalg.norm(u_obs) + 1e-12))

    # Plot traction field on load surface
    out_plot = Path(cfg.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Build 2D grid over (y, z) for plotting
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    if cfg.load_basis == "rbf":
        grid_y = np.linspace(y_min, y_max, cfg.rbf_ny)
        grid_z = np.linspace(z_min, z_max, cfg.rbf_nz)
    else:
        grid_y = np.linspace(y_min, y_max, cfg.bspline_n)
        grid_z = np.linspace(z_min, z_max, max(1, cfg.rbf_nz))
    grid = np.array([[xmax, y, z] for y in grid_y for z in grid_z], dtype=float)

    def _rbf_eval_grid(coeffs: np.ndarray) -> np.ndarray:
        vals = []
        for c in centers:
            r2 = np.sum((grid - c) ** 2, axis=1)
            vals.append(np.exp(-r2 * 0.5 / (cfg.rbf_sigma ** 2)))
        Phi = np.stack(vals, axis=1)
        return (Phi @ coeffs).reshape(len(grid_y), len(grid_z))

    def _bspline_eval_grid(coeffs: np.ndarray) -> np.ndarray:
        Phi = _bspline_basis_matrix(grid[:, 1], cfg.bspline_n, cfg.bspline_degree, y_min, y_max)
        return (Phi @ coeffs).reshape(len(grid_y), len(grid_z))

    if cfg.load_basis == "rbf":
        field_true = _rbf_eval_grid(a_true)
        field_hat = _rbf_eval_grid(a_hat)
    else:
        field_true = _bspline_eval_grid(a_true)
        field_hat = _bspline_eval_grid(a_hat)

    im0 = axes[0].imshow(
        field_true,
        origin="lower",
        extent=[z_min, z_max, y_min, y_max],
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_title("True traction (scalar)")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        field_hat,
        origin="lower",
        extent=[z_min, z_max, y_min, y_max],
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_title("Recovered traction (scalar)")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    out_json = Path(cfg.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg.__dict__,
        "basis": basis_meta,
        "n_basis": int(n_basis),
        "n_obs": int(obs_dofs.shape[0]),
        "rel_coeff_error": rel_a,
        "rel_u_error": rel_u,
        "rel_obs_error": rel_obs,
        "noise_std_abs": noise_scale,
        "seed": cfg.seed,
    }
    out_json.write_text(json.dumps(_jsonable(payload), indent=2))

    print(f"[unknown load id] n_basis={n_basis} n_obs={obs_dofs.shape[0]}")
    print(f"  rel coeff error: {rel_a:.3e}")
    print(f"  rel disp error:  {rel_u:.3e}")
    print(f"  rel obs error:   {rel_obs:.3e}")
    print(f"  plot: {out_plot}")
    print(f"  json: {out_json}")


if __name__ == "__main__":
    main()

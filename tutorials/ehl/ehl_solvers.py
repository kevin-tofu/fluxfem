from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import fluxfem as ff

from ehl_mesh import PlateModel, ReynoldsMesh, integrate_scalar_on_tri_nodes


@dataclass
class CoupledState:
    h0: float
    p_re: np.ndarray
    p_nodes_plate: np.ndarray
    u: np.ndarray
    load_n: float
    picard_iters: int
    picard_rel: float


def solve_reynolds_isoviscous(
    re_mesh: ReynoldsMesh,
    h_nodes_mm: np.ndarray,
    eta_mpa_s: float,
    u_mm_s: float,
    solver_backend: str = "scipy",
    petsc_ksp_type: str = "cg",
    petsc_pc_type: str = "ilu",
    petsc_rtol: float = 1e-8,
    petsc_atol: float = 0.0,
    petsc_max_it: int = 2000,
) -> np.ndarray:
    n = re_mesh.xy.shape[0]
    tri = re_mesh.tri
    xy = re_mesh.xy

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n, dtype=np.float64)

    h_nodes_mm = np.asarray(h_nodes_mm, dtype=np.float64)
    h_nodes_mm = np.nan_to_num(h_nodes_mm, nan=1e-9, posinf=1e3, neginf=1e-9)
    h_nodes_mm = np.maximum(h_nodes_mm, 1e-9)

    for e in tri:
        ids = np.asarray(e, dtype=np.int64)
        x1, y1 = xy[ids[0]]
        x2, y2 = xy[ids[1]]
        x3, y3 = xy[ids[2]]

        det2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(det2)
        if area <= 0.0:
            continue

        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64) / (2.0 * area)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64) / (2.0 * area)
        grads = np.stack([b, c], axis=1)

        h_e = np.nan_to_num(h_nodes_mm[ids], nan=1e-9, posinf=1e3, neginf=1e-9)
        h_e = np.maximum(h_e, 1e-9)
        k_e = float(np.mean(h_e**3) / (12.0 * max(eta_mpa_s, 1e-18)))
        ke = k_e * area * (grads @ grads.T)

        dhdx_e = float(np.dot(h_e, grads[:, 0]))
        if not np.isfinite(dhdx_e):
            dhdx_e = 0.0
        fe = (0.5 * u_mm_s * dhdx_e) * (area / 3.0) * np.ones(3, dtype=np.float64)

        for i_local, i_global in enumerate(ids):
            rhs[i_global] += fe[i_local]
            for j_local, j_global in enumerate(ids):
                rows.append(int(i_global))
                cols.append(int(j_global))
                data.append(float(ke[i_local, j_local]))

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    dir_nodes = np.unique(re_mesh.boundary_nodes)
    free_mask = np.ones(n, dtype=bool)
    free_mask[dir_nodes] = False
    free = np.where(free_mask)[0]

    p = np.zeros(n, dtype=np.float64)
    if free.size > 0:
        A_ff = A[free][:, free].tocsr()
        rhs_f = rhs[free].copy()

        nnz_row = np.asarray(A_ff.getnnz(axis=1)).reshape(-1)
        zero_rows = np.where(nnz_row == 0)[0]
        if zero_rows.size > 0:
            A_ff = A_ff.tolil()
            for i in zero_rows:
                A_ff[i, i] = 1.0
                rhs_f[i] = 0.0
            A_ff = A_ff.tocsr()

        if solver_backend == "petsc":
            try:
                p_f = ff.petsc_solve(
                    A_ff,
                    rhs_f,
                    ksp_type=petsc_ksp_type,
                    pc_type=petsc_pc_type,
                    rtol=float(petsc_rtol),
                    atol=float(petsc_atol),
                    max_it=int(petsc_max_it),
                )
            except Exception as exc:
                raise RuntimeError(f"PETSc Reynolds solve failed: {exc}") from exc
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                try:
                    p_f = spla.spsolve(A_ff, rhs_f)
                except spla.MatrixRankWarning:
                    p_f = spla.lsmr(A_ff, rhs_f, atol=1e-12, btol=1e-12, maxiter=5000)[0]
        p_f = np.nan_to_num(np.asarray(p_f, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        p[free] = np.asarray(p_f, dtype=np.float64)

    return np.maximum(p, 0.0)


def pressure_surface_form(ctx: ff.SurfaceFormContext, params: dict[str, np.ndarray]) -> np.ndarray:
    p_facet = params["p_facet"]
    p_q = jnp.full((ctx.v.N.shape[0],), p_facet[ctx.facet_id], dtype=jnp.asarray(p_facet).dtype)

    traction = jnp.zeros((ctx.v.N.shape[0], 3), dtype=p_q.dtype)
    traction = traction.at[:, 2].set(-p_q)

    elem = ctx.v.N[:, :, None] * traction[:, None, :]
    return elem.reshape(elem.shape[0], -1)


def build_elastic_solver(
    plate: PlateModel,
    intorder: int,
    e_mpa: float,
    nu: float,
    linear_solver: str,
    linear_tol: float,
    linear_maxiter: int,
):
    mesh = ff.TetMesh(coords=jnp.asarray(plate.coords), conn=jnp.asarray(plate.conn))
    space = ff.make_tet_space(mesh, dim=3, intorder=intorder)
    U = ff.NamedSpace("U", space)
    V = ff.NamedSpace("V", space)

    D = ff.isotropic_3d_D(e_mpa, nu)
    K = ff.assemble_bilinear_form(
        ff.BilinearSpaces(test=V, trial=U),
        ff.linear_elasticity_form,
        D,
    )

    bottom_nodes = np.unique(np.asarray(plate.bottom_facets).reshape(-1))
    dir_dofs = mesh.node_dofs(bottom_nodes, components="xyz", dof_per_node=3)

    top_surface = ff.make_surface_from_facets(plate.coords, plate.top_facets)
    top_facets = np.asarray(top_surface.conn, dtype=np.int64)

    solver = ff.LinearSolver(method=linear_solver, tol=linear_tol, maxiter=linear_maxiter)

    def solve_u_from_pressure_nodes(p_nodes_plate: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        p_facet = np.mean(np.asarray(p_nodes_plate, dtype=np.float64)[top_facets], axis=1)
        F = top_surface.assemble_linear_form_on_space(
            space,
            pressure_surface_form,
            params={"p_facet": jnp.asarray(p_facet, dtype=jnp.float64)},
        )
        F = jnp.asarray(F, dtype=jnp.float64)
        u, info = solver.solve(
            K,
            F,
            dirichlet=ff.DirichletBC(dir_dofs, None),
            dirichlet_mode="condense",
        )
        return np.asarray(u, dtype=np.float64), dict(info)

    return mesh, space, solve_u_from_pressure_nodes


def run_picard_for_h0(
    h0_mm: float,
    re_mesh: ReynoldsMesh,
    plate: PlateModel,
    solve_u,
    r_mm: float,
    eta_mpa_s: float,
    u_mm_s: float,
    max_iters: int,
    rel_tol: float,
    pressure_relax: float,
    min_film_mm: float,
    reynolds_solver: str,
    reynolds_ksp_type: str,
    reynolds_pc_type: str,
    reynolds_rtol: float,
    reynolds_atol: float,
    reynolds_max_it: int,
) -> CoupledState:
    n_plate_nodes = plate.coords.shape[0]
    n_re = re_mesh.xy.shape[0]

    x = re_mesh.xy[:, 0]
    y = re_mesh.xy[:, 1]
    g_mm = (x * x + y * y) / (2.0 * r_mm)

    p_prev = np.zeros(n_re, dtype=np.float64)
    u = np.zeros(3 * n_plate_nodes, dtype=np.float64)

    rel = np.inf
    for it in range(1, max_iters + 1):
        u_nodes = u.reshape(-1, 3)
        un_re = u_nodes[re_mesh.plate_nodes, 2]
        if not np.all(np.isfinite(un_re)):
            raise RuntimeError(f"non-finite displacement detected before Reynolds at Picard iter {it}")

        h_nodes = h0_mm + g_mm + un_re
        h_nodes = np.maximum(h_nodes, min_film_mm)

        p_new = solve_reynolds_isoviscous(
            re_mesh,
            h_nodes,
            eta_mpa_s,
            u_mm_s,
            solver_backend=reynolds_solver,
            petsc_ksp_type=reynolds_ksp_type,
            petsc_pc_type=reynolds_pc_type,
            petsc_rtol=reynolds_rtol,
            petsc_atol=reynolds_atol,
            petsc_max_it=reynolds_max_it,
        )
        if not np.all(np.isfinite(p_new)):
            raise RuntimeError(f"non-finite pressure detected at Picard iter {it}")
        p_re = pressure_relax * p_new + (1.0 - pressure_relax) * p_prev

        p_nodes_plate = np.zeros(n_plate_nodes, dtype=np.float64)
        p_nodes_plate[re_mesh.plate_nodes] = p_re

        u_new, lin_info = solve_u(p_nodes_plate)
        if not np.all(np.isfinite(u_new)):
            raise RuntimeError(
                f"non-finite displacement after linear solve at Picard iter {it}; "
                f"lin_info={lin_info}"
            )

        den = np.linalg.norm(p_prev)
        rel = float(np.linalg.norm(p_re - p_prev) / (den + 1e-12))

        load_n = integrate_scalar_on_tri_nodes(p_re, re_mesh.xy, re_mesh.tri)
        uz_top = u_new.reshape(-1, 3)[re_mesh.plate_nodes, 2]
        h_new_nodes = np.maximum(h0_mm + g_mm + uz_top, min_film_mm)
        lin_it = lin_info.get("iters", None)
        lin_conv = lin_info.get("converged", None)
        print(
            f"  [picard {it:02d}] h0={h0_mm:.6e} mm, load={load_n:.6e} N, "
            f"p_max={p_re.max():.6e} MPa, rel_p={rel:.3e}, "
            f"h_min/max={h_new_nodes.min():.3e}/{h_new_nodes.max():.3e} mm, "
            f"uz_min/max={uz_top.min():.3e}/{uz_top.max():.3e} mm, "
            f"lin(it={lin_it}, conv={lin_conv})"
        )

        p_prev = p_re
        u = u_new

        if rel < rel_tol and it >= 2:
            break

    p_nodes_plate = np.zeros(n_plate_nodes, dtype=np.float64)
    p_nodes_plate[re_mesh.plate_nodes] = p_prev
    load_n = integrate_scalar_on_tri_nodes(p_prev, re_mesh.xy, re_mesh.tri)

    return CoupledState(
        h0=h0_mm,
        p_re=p_prev,
        p_nodes_plate=p_nodes_plate,
        u=u,
        load_n=load_n,
        picard_iters=it,
        picard_rel=rel,
    )


def solve_with_bisection(
    re_mesh: ReynoldsMesh,
    plate: PlateModel,
    solve_u,
    args: argparse.Namespace,
    eta_mpa_s: float,
    u_mm_s: float,
) -> CoupledState:
    target = float(args.target_load_n)

    def eval_h0(h0_mm: float) -> CoupledState:
        print(f"[outer] evaluate h0={h0_mm:.6e} mm")
        return run_picard_for_h0(
            h0_mm=h0_mm,
            re_mesh=re_mesh,
            plate=plate,
            solve_u=solve_u,
            r_mm=float(args.R_mm),
            eta_mpa_s=eta_mpa_s,
            u_mm_s=u_mm_s,
            max_iters=int(args.max_picard_iters),
            rel_tol=float(args.picard_rel_tol),
            pressure_relax=float(args.pressure_relax),
            min_film_mm=float(args.min_film_mm),
            reynolds_solver=str(args.reynolds_solver),
            reynolds_ksp_type=str(args.reynolds_ksp_type),
            reynolds_pc_type=str(args.reynolds_pc_type),
            reynolds_rtol=float(args.reynolds_rtol),
            reynolds_atol=float(args.reynolds_atol),
            reynolds_max_it=int(args.reynolds_max_it),
        )

    h_low = float(args.h0_min_mm)
    h_high = float(args.h0_max_mm)
    s_low = eval_h0(h_low)
    s_high = eval_h0(h_high)

    f_low = s_low.load_n - target
    f_high = s_high.load_n - target

    expands = 0
    while f_low * f_high > 0.0 and expands < int(args.max_bracket_expands):
        expands += 1
        if f_low <= 0.0 and f_high <= 0.0:
            h_low *= 0.5
            s_low = eval_h0(h_low)
            f_low = s_low.load_n - target
        else:
            h_high *= 2.0
            s_high = eval_h0(h_high)
            f_high = s_high.load_n - target

    if f_low * f_high > 0.0:
        raise RuntimeError(
            "Failed to bracket target load. Adjust h0 range or target load. "
            f"f(h_low)={f_low:.3e}, f(h_high)={f_high:.3e}"
        )

    best = s_low if abs(f_low) < abs(f_high) else s_high
    load_tol = float(args.load_rel_tol) * max(abs(target), 1.0)

    for it in range(1, int(args.max_bisect_iters) + 1):
        h_mid = 0.5 * (h_low + h_high)
        s_mid = eval_h0(h_mid)
        f_mid = s_mid.load_n - target
        best = s_mid if abs(f_mid) < abs(best.load_n - target) else best

        print(
            f"[bisect {it:02d}] h_mid={h_mid:.6e} mm, "
            f"F={f_mid:.6e} N (target={target:.6e} N)"
        )

        if abs(f_mid) <= load_tol:
            return s_mid

        if f_mid > 0.0:
            h_low = h_mid
        else:
            h_high = h_mid

    return best


def save_result_plots(
    state: CoupledState,
    re_mesh: ReynoldsMesh,
    plate: PlateModel,
    plot_prefix: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    out_paths: list[Path] = []
    plot_prefix.parent.mkdir(parents=True, exist_ok=True)

    tri_re = mtri.Triangulation(re_mesh.xy[:, 0], re_mesh.xy[:, 1], triangles=re_mesh.tri)
    fig1, ax1 = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    pc = ax1.tripcolor(tri_re, state.p_re, shading="gouraud", cmap="viridis")
    ax1.set_title("Reynolds Pressure p [MPa]")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_aspect("equal", adjustable="box")
    cbar = fig1.colorbar(pc, ax=ax1)
    cbar.set_label("p [MPa]")
    p_path = plot_prefix.with_name(plot_prefix.name + "_pressure.png")
    fig1.savefig(p_path, dpi=180)
    plt.close(fig1)
    out_paths.append(p_path)

    u_nodes = state.u.reshape(-1, 3)
    uz = u_nodes[:, 2]
    tri_top = np.asarray(plate.top_facets, dtype=np.int64)
    tri_top_plot = mtri.Triangulation(
        plate.coords[:, 0],
        plate.coords[:, 1],
        triangles=tri_top,
    )
    fig2, ax2 = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    uc = ax2.tripcolor(tri_top_plot, uz, shading="gouraud", cmap="coolwarm")
    ax2.set_title("TOP Displacement $u_z$ [mm]")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_aspect("equal", adjustable="box")
    cbar2 = fig2.colorbar(uc, ax=ax2)
    cbar2.set_label("$u_z$ [mm]")
    u_path = plot_prefix.with_name(plot_prefix.name + "_uz_top.png")
    fig2.savefig(u_path, dpi=180)
    plt.close(fig2)
    out_paths.append(u_path)

    return out_paths

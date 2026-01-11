from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class NitscheContactParams:
    nx_top: int = 19
    ny_top: int = 19
    nz_top: int = 9
    nx_bot: int = 19
    ny_bot: int = 19
    nz_bot: int = 4
    quad_order: int = 5
    E: float = 210e9
    nu: float = 0.3
    total_force: float = 100.0
    alpha_w: float = 10000.0
    alpha_scale: float = 20.0


@dataclass(frozen=True)
class NitscheContactResult:
    summary: dict[str, Any]
    u: np.ndarray
    points: np.ndarray | None = None
    u_nodes: np.ndarray | None = None


def _alpha_from_params(params: NitscheContactParams) -> float:
    lam = params.E * params.nu / ((1.0 + params.nu) * (1.0 - 2.0 * params.nu))
    mu = params.E / (2.0 * (1.0 + params.nu))
    return params.alpha_scale * (params.alpha_w * mu + lam)


def _lame_parameters(params: NitscheContactParams) -> tuple[float, float]:
    lam = params.E * params.nu / ((1.0 + params.nu) * (1.0 - 2.0 * params.nu))
    mu = params.E / (2.0 * (1.0 + params.nu))
    return float(lam), float(mu)


def _isotropic_3d_D(lam: float, mu: float, dtype) -> np.ndarray:
    return np.array(
        [
            [lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=dtype,
    )


def _export_skfem_combined_vtu(mesh_top, mesh_bot, u: np.ndarray, file_path: str) -> None:
    import meshio
    import skfem

    if isinstance(mesh_top, skfem.MeshTet):
        cell_type = "tetra"
    elif isinstance(mesh_top, skfem.MeshHex):
        cell_type = "hexahedron"
    else:
        raise ValueError("Unsupported mesh type")

    n_top_nodes = mesh_top.p.shape[1]
    n_bot_nodes = mesh_bot.p.shape[1]
    points = np.vstack([mesh_top.p.T, mesh_bot.p.T])
    cells = [
        (
            cell_type,
            np.vstack([mesh_top.t.T, mesh_bot.t.T + n_top_nodes]),
        )
    ]

    n_nodes_total = n_top_nodes + n_bot_nodes
    u_top_bot = u[: n_nodes_total * 3].reshape(n_nodes_total, 3)

    node_tag = np.zeros(n_top_nodes + n_bot_nodes, dtype=int)
    top_nodes_contact = np.unique(mesh_top.facets[:, mesh_top.boundaries["contact"]])
    bot_nodes_contact = np.unique(mesh_bot.facets[:, mesh_bot.boundaries["contact"]]) + n_top_nodes
    nodes_contact = np.hstack([top_nodes_contact, bot_nodes_contact])
    bot_nodes_dirichlet = np.unique(mesh_bot.facets[:, mesh_bot.boundaries["dirichlet"]]) + n_top_nodes
    nodes_dirichlet = bot_nodes_dirichlet
    top_nodes_force = np.unique(mesh_top.facets[:, mesh_top.boundaries["force"]])
    nodes_force = top_nodes_force
    node_tag[nodes_contact] = 1
    node_tag[nodes_dirichlet] = 2
    node_tag[nodes_force] = 3

    meshio.Mesh(
        points=points,
        cells=cells,
        point_data={
            "u": u_top_bot,
            "node_tag": node_tag,
        },
    ).write(file_path)


def _export_fluxfem_combined_vtu(
    mesh_top,
    mesh_bot,
    u: np.ndarray,
    file_path: str,
    *,
    contact_facets_top,
    contact_facets_bot,
    force_facets_top,
    dirichlet_facets_bot,
) -> None:
    import meshio

    n_top_nodes = mesh_top.coords.shape[0]
    n_bot_nodes = mesh_bot.coords.shape[0]

    points = np.vstack([np.asarray(mesh_top.coords), np.asarray(mesh_bot.coords)])
    cells = [
        (
            "tetra",
            np.vstack(
                [
                    np.asarray(mesh_top.conn, dtype=int),
                    np.asarray(mesh_bot.conn, dtype=int) + n_top_nodes,
                ]
            ),
        )
    ]

    n_nodes_total = n_top_nodes + n_bot_nodes
    u_nodes = u[: n_nodes_total * 3].reshape(n_nodes_total, 3)
    node_tag = np.zeros(n_nodes_total, dtype=int)

    top_contact = np.unique(np.asarray(contact_facets_top, dtype=int))
    bot_contact = np.unique(np.asarray(contact_facets_bot, dtype=int)) + n_top_nodes
    node_tag[np.hstack([top_contact, bot_contact])] = 1

    bot_dirichlet = np.unique(np.asarray(dirichlet_facets_bot, dtype=int)) + n_top_nodes
    node_tag[bot_dirichlet] = 2

    top_force = np.unique(np.asarray(force_facets_top, dtype=int))
    node_tag[top_force] = 3

    meshio.Mesh(
        points=points,
        cells=cells,
        point_data={
            "u": u_nodes,
            "node_tag": node_tag,
        },
    ).write(file_path)


def run_skfem_demo(
    params: NitscheContactParams,
    *,
    log_path: str | None = None,
    npz_path: str | None = None,
    vtu_path: str | None = None,
    plot_path: str | None = None,
    verbose: bool = True,
    return_nodes: bool = False,
) -> NitscheContactResult:
    import os
    import scipy
    import skfem
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.sparse import bmat
    from scipy.sparse.linalg import minres
    from skfem import InteriorBasis, FacetBasis, ElementTetP1, ElementVectorH1
    from skfem.helpers import dot, ddot, sym_grad, mul
    from skfem.models.elasticity import lame_parameters, linear_stress
    from skfem.supermeshing import intersect, elementwise_quadrature

    if plot_path:
        matplotlib.use("Agg")

    def is_contact_surface(x):
        return np.isclose(x[2], 0.0)

    def is_force_surface(x):
        return np.isclose(x[2], 1.0)

    def is_dirichlet_surface(x):
        return np.isclose(x[2], -0.5)

    mesh_top = skfem.MeshTet.init_tensor(
        np.linspace(0, 2, params.nx_top + 1),
        np.linspace(0, 2, params.ny_top + 1),
        np.linspace(0, 1, params.nz_top + 1),
    )
    mesh_bot = skfem.MeshTet.init_tensor(
        np.linspace(0.5, 1.5, params.nx_bot + 1),
        np.linspace(0.5, 1.5, params.ny_bot + 1),
        np.linspace(-0.5, 0.0, params.nz_bot + 1),
    )

    mesh_top = mesh_top.with_boundaries(
        {
            "contact": is_contact_surface,
            "force": is_force_surface,
        }
    )
    mesh_bot = mesh_bot.with_boundaries(
        {
            "contact": is_contact_surface,
            "dirichlet": is_dirichlet_surface,
        }
    )

    elem = ElementTetP1()
    u_elem = ElementVectorH1(elem)
    basis_top = InteriorBasis(mesh_top, u_elem)
    basis_bot = InteriorBasis(mesh_bot, u_elem)

    mu = params.E / (2.0 * (1.0 + params.nu))
    lam = params.E * params.nu / ((1.0 + params.nu) * (1.0 - 2.0 * params.nu))
    C = linear_stress(*lame_parameters(params.E, params.nu))

    @skfem.BilinearForm
    def a_elastic(u, v, w):
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        I = np.eye(3).reshape(3, 3, 1)
        return lam * ddot(eps_u, I) * ddot(eps_v, I) + 2 * mu * ddot(eps_u, eps_v)

    K1 = a_elastic.assemble(basis_top)
    K2 = a_elastic.assemble(basis_bot)

    basis_top = skfem.Basis(mesh_top, ElementVectorH1(ElementTetP1()))
    basis_bot = skfem.Basis(mesh_bot, ElementVectorH1(ElementTetP1()))

    elem_s = ElementTetP1()
    elem_v = ElementVectorH1(elem_s)
    m1t, orig1 = mesh_top.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m2t, orig2 = mesh_bot.trace("contact", mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
    m12, t1, t2 = intersect(m1t, m2t)
    quad_order = int(params.quad_order)
    try:
        quad1 = elementwise_quadrature(m1t, m12, t1, intorder=quad_order)
        quad2 = elementwise_quadrature(m2t, m12, t2, intorder=quad_order)
    except TypeError:
        quad1 = elementwise_quadrature(m1t, m12, t1)
        quad2 = elementwise_quadrature(m2t, m12, t2)

    fb_u_top = FacetBasis(mesh_top, elem_v, facets=orig1[t1], quadrature=quad1)
    fb_u_bot = FacetBasis(mesh_bot, elem_v, facets=orig2[t2], quadrature=quad2)
    fbasis = fb_u_top * fb_u_bot

    alpha = _alpha_from_params(params)

    @skfem.BilinearForm
    def bilin_mortar(u1, u2, v1, v2, w):
        ju = u1 - u2
        jv = v1 - v2
        t_u = 0.5 * (mul(C(sym_grad(u1)), w.n) + mul(C(sym_grad(u2)), w.n))
        t_v = 0.5 * (mul(C(sym_grad(v1)), w.n) + mul(C(sym_grad(v2)), w.n))
        return (
            (alpha / w.h) * dot(ju, jv)
            - dot(t_u, jv)
            - dot(t_v, ju)
        )

    B = skfem.asm(bilin_mortar, fbasis, h=fb_u_top.mesh_parameters())
    K = bmat([[K1, None], [None, K2]], "csr") + B

    F_facet_ids = mesh_top.facets_satisfying(is_force_surface)
    fbasis_force = skfem.FacetBasis(mesh_top, elem_v, facets=F_facet_ids)

    @skfem.Functional
    def surface_measure(w):
        return 1.0

    A_proj_z = surface_measure.assemble(fbasis_force)
    pressure = params.total_force / A_proj_z

    @skfem.LinearForm
    def l_comp(v, w):
        return pressure * v[2]

    top_F = l_comp.assemble(fbasis_force)
    bot_F = np.zeros(mesh_bot.p.shape[1] * mesh_bot.dim())
    F = np.hstack([top_F, bot_F])

    D_facets = mesh_bot.facets_satisfying(is_dirichlet_surface)
    D_dofs = basis_bot.get_dofs(facets=D_facets).all() + K1.shape[0]

    Kc, Fc, uc, I = skfem.condense(K, F, D=D_dofs)
    uc, _info = minres(Kc, Fc, rtol=1e-10, maxiter=20000)
    uc = scipy.sparse.linalg.spsolve(Kc, Fc)

    u = np.zeros(K.shape[0])
    u[I] = uc

    F_from_U = Kc.dot(u[I])
    F_pred_full = K @ u
    residual_full = F_pred_full - F

    res_norm = np.linalg.norm(Fc - Kc @ uc)
    rhs_norm = np.linalg.norm(Fc)
    rel_res = res_norm / rhs_norm

    if verbose:
        print("K1:", K1.shape)
        print("K2:", K2.shape)
        print("B1:", B.shape)
        print("len(t1):", len(t1), "len(t2):", len(t2))
        print(u)
        print(np.sum(np.abs(F_from_U - Fc)))
        print("K.shape, u.shape, F_pred_full.shape", K.shape, u.shape, F_pred_full.shape)
        print("F_pred_full.shape, F.shape", F_pred_full.shape, F.shape)
        print("len(F_pred_full) =", len(F_pred_full))
        print("len(F) =", len(F))
        print("‖residual‖₂ =", np.linalg.norm(residual_full))
        print("relative residual =", rel_res)

    summary = {
        "K1_shape": K1.shape,
        "K2_shape": K2.shape,
        "B_shape": B.shape,
        "len_t1": len(t1),
        "len_t2": len(t2),
        "sum_abs_Fdiff": float(np.sum(np.abs(F_from_U - Fc))),
        "residual_norm_2": float(np.linalg.norm(residual_full)),
        "relative_residual": float(rel_res),
    }

    if log_path:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"{k}={v}" for k, v in summary.items()) + "\n")

    points = None
    u_nodes = None
    if npz_path or return_nodes:
        n_top_nodes = mesh_top.p.shape[1]
        n_bot_nodes = mesh_bot.p.shape[1]
        points = np.vstack([mesh_top.p.T, mesh_bot.p.T])
        n_nodes_total = n_top_nodes + n_bot_nodes
        u_nodes = u[: n_nodes_total * 3].reshape(n_nodes_total, 3)
    if npz_path:
        np.savez(npz_path, u=u, points=points, u_nodes=u_nodes)

    if vtu_path:
        _export_skfem_combined_vtu(mesh_top, mesh_bot, u, vtu_path)

    if plot_path:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(F_from_U)
        ax2.plot(Fc)
        fig.savefig(plot_path)

    return NitscheContactResult(summary=summary, u=u, points=points, u_nodes=u_nodes)


def run_fluxfem_demo(
    params: NitscheContactParams,
    *,
    contact_backend: str = "jax",
    platform: str = "auto",
    x64: bool = True,
    xla_cpu_parallelism: int | None = None,
    omp_threads: int | None = None,
    low_mem: bool = False,
    n_chunks: int | None = None,
    timing: bool = False,
    cg_tol: float = 1e-8,
    cg_maxiter: int | None = None,
    cg_precond: str = "none",
    symmetry_check: bool = False,
    bench_contact: bool = False,
    debug_contact: bool = False,
    batch_jac: bool = True,
    fd_eps: float = 1e-6,
    fd_mode: str = "central",
    fd_block_size: int = 1,
    log_path: str | None = None,
    vtu_path: str | None = None,
    plot_path: str | None = None,
    npz_path: str | None = None,
    verbose: bool = True,
    return_nodes: bool = False,
) -> NitscheContactResult:
    import os
    import time
    import jax
    import jax.numpy as jnp
    import fluxfem as ff
    import fluxfem.helpers_wf as h_wf
    from fluxfem.core.weakform import einsum as wf_einsum

    def _append_xla_flag(flag: str) -> None:
        existing = os.environ.get("XLA_FLAGS", "")
        if flag in existing.split():
            return
        if existing:
            os.environ["XLA_FLAGS"] = f"{existing} {flag}"
        else:
            os.environ["XLA_FLAGS"] = flag

    if low_mem:
        if platform == "auto":
            platform = "cpu"
        x64 = False
        if omp_threads is None:
            omp_threads = 1

    if platform != "auto":
        os.environ.setdefault("JAX_PLATFORMS", platform)
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
    if xla_cpu_parallelism is not None:
        _append_xla_flag(f"--xla_cpu_compilation_parallelism={xla_cpu_parallelism}")
    if omp_threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))

    jax.config.update("jax_enable_x64", x64)
    timing_state = {"start": time.perf_counter(), "last": time.perf_counter()}

    def _mark(label: str) -> None:
        if not timing:
            return
        now = time.perf_counter()
        elapsed = now - timing_state["last"]
        total = now - timing_state["start"]
        print(f"[timing] {label}: +{elapsed:.3f}s (total {total:.3f}s)", flush=True)
        timing_state["last"] = now

    def _mesh_spacing(box):
        return box.lx / box.nx, box.ly / box.ny, box.lz / box.nz

    if timing:
        _mark("start")

    lam, mu = _lame_parameters(params)
    D = _isotropic_3d_D(lam, mu, dtype=jnp.asarray(0.0).dtype)
    _mark("material params")

    box_top = ff.StructuredTetTensorBox(
        nx=params.nx_top,
        ny=params.ny_top,
        nz=params.nz_top,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        origin=(0.0, 0.0, 0.0),
    )
    box_bot = ff.StructuredTetTensorBox(
        nx=params.nx_bot,
        ny=params.ny_bot,
        nz=params.nz_bot,
        lx=1.0,
        ly=1.0,
        lz=0.5,
        origin=(0.5, 0.5, -0.5),
    )
    mesh_top = box_top.build()
    mesh_bot = box_bot.build()
    _mark("mesh build")

    space_top = ff.make_tet_space(mesh_top, dim=3)
    space_bot = ff.make_tet_space(mesh_bot, dim=3)
    _mark("space build")

    K1 = space_top.assemble_bilinear_form(ff.linear_elasticity_form, params=D, n_chunks=n_chunks)
    _mark("K1 assemble")
    K2 = space_bot.assemble_bilinear_form(ff.linear_elasticity_form, params=D, n_chunks=n_chunks)
    _mark("K2 assemble")

    contact_facets_top = mesh_top.facets_on_plane(axis=2, value=0.0)
    contact_facets_bot = mesh_bot.facets_on_plane(axis=2, value=0.0)
    x0, y0, _ = box_bot.origin
    x1 = x0 + box_bot.lx
    y1 = y0 + box_bot.ly
    dx_top, dy_top, _ = _mesh_spacing(box_top)
    pad = 2.0 * min(dx_top, dy_top)
    contact_facets_top = mesh_top.facets_on_plane_box(
        axis=2,
        value=0.0,
        x=(x0 - pad, x1 + pad),
        y=(y0 - pad, y1 + pad),
        mode="centroid",
    )
    force_facets_top = mesh_top.facets_on_plane(axis=2, value=1.0)
    dirichlet_facets_bot = mesh_bot.facets_on_plane(axis=2, value=-0.5)
    _mark("boundary facets")

    side_top = ff.ContactSide.from_facets(mesh_top, contact_facets_top, space_top)
    side_bot = ff.ContactSide.from_facets(mesh_bot, contact_facets_bot, space_bot)
    contact = ff.ContactSurfaceSpace.from_sides(
        side_top,
        side_bot,
        quad_order=int(params.quad_order),
        backend=contact_backend,
        batch_jac=batch_jac,
        fd_eps=fd_eps,
        fd_mode=fd_mode,
        fd_block_size=fd_block_size,
    )
    _mark("contact setup")
    if verbose and contact_backend == "numpy":
        print("[contact] numpy backend uses FD for Jacobian (non-differentiable)", flush=True)

    dx_top, dy_top, dz_top = _mesh_spacing(box_top)
    dx_bot, dy_bot, dz_bot = _mesh_spacing(box_bot)
    h = min(dx_top, dy_top, dz_top, dx_bot, dy_bot, dz_bot)

    alpha = _alpha_from_params(params)

    def bilin(v1, v2, u1, u2, p):
        n = h_wf.normal()
        ju = u1.val - u2.val
        t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
        t_v1 = h_wf.traction(v1, n, p)
        t_v2 = h_wf.traction(v2, n, p)
        penalty = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        traction = -h_wf.dot(v1, t_u) + h_wf.dot(v2, t_u)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
        return (penalty + traction) * h_wf.ds()

    if contact_backend == "numpy":
        u_top0 = np.zeros(space_top.n_dofs)
        u_bot0 = np.zeros(space_bot.n_dofs)
    else:
        u_top0 = jnp.zeros(space_top.n_dofs)
        u_bot0 = jnp.zeros(space_bot.n_dofs)
    params_contact = ff.Params(alpha=float(alpha), inv_h=float(1.0 / h), lam=float(lam), mu=float(mu))
    if debug_contact:
        print("n_contact_facets_top =", int(contact_facets_top.shape[0]))
        print("n_contact_facets_bot =", int(contact_facets_bot.shape[0]))
        print("quad_order =", int(contact.quad_order))
        print("n_supermesh_tris =", int(contact.supermesh_conn.shape[0]))
    if timing:
        print("[timing] K_contact assemble: start", flush=True)
        t_contact = time.perf_counter()
    contact_coo = contact.assemble_bilinear(
        bilin,
        (u_top0, u_bot0),
        params_contact,
        sparse=True,
    )
    if timing:
        print(f"[timing] K_contact assemble: bilinear {time.perf_counter() - t_contact:.3f}s", flush=True)
        t_sparse = time.perf_counter()
    K_contact = ff.FluxSparseMatrix.from_bilinear(contact_coo)
    if timing:
        print(f"[timing] K_contact assemble: to_sparse {time.perf_counter() - t_sparse:.3f}s", flush=True)
    if debug_contact:
        print("K_contact nnz =", int(K_contact.data.shape[0]))
    _mark("K_contact assemble")
    if bench_contact:
        K_contact_2 = ff.FluxSparseMatrix.from_bilinear(
            contact.assemble_bilinear(
                bilin,
                (u_top0, u_bot0),
                params_contact,
                sparse=True,
            )
        )
        if timing:
            try:
                K_contact_2.data.block_until_ready()
            except Exception:
                pass
        _mark("K_contact assemble (2nd)")

    K_block = ff.block_diag_flux(K1, K2)
    K = ff.concat_flux(K_block, K_contact, n_dofs=K_block.n_dofs)

    force_surface = ff.SurfaceMesh.from_facets(mesh_top.coords, force_facets_top)
    area = float(np.sum(force_surface.facet_areas()))
    pressure = params.total_force / area
    top_F = force_surface.assemble_load(
        load=np.array([0.0, 0.0, pressure]),
        dim=3,
        n_total_nodes=mesh_top.n_nodes,
    )
    bot_F = np.zeros(space_bot.n_dofs, dtype=float)
    F = np.hstack([top_F, bot_F])

    dir_dofs_bot = mesh_bot.boundary_dofs_plane(axis=2, value=-0.5, dof_per_node=3)
    dir_dofs = dir_dofs_bot + space_top.n_dofs
    dir_vals = np.zeros(dir_dofs.shape[0], dtype=float)

    n_total = int(K.n_dofs)
    dir_dofs = np.asarray(dir_dofs, dtype=int)
    free_mask = np.ones(n_total, dtype=bool)
    free_mask[dir_dofs] = False
    free = np.nonzero(free_mask)[0]
    free_j = jnp.asarray(free)
    dir_vals_j = jnp.asarray(dir_vals)

    F_j = jnp.asarray(F)
    if np.all(dir_vals == 0.0):
        F_free = F_j[free_j]
    else:
        u_dir = jnp.zeros(n_total, dtype=F_j.dtype).at[dir_dofs].set(dir_vals_j)
        F_free = (F_j - K.matvec(u_dir))[free_j]
    _mark("apply dirichlet")

    g2l = -np.ones(n_total, dtype=np.int32)
    g2l[free] = np.arange(free.size, dtype=np.int32)
    rows = np.asarray(K.pattern.rows)
    cols = np.asarray(K.pattern.cols)
    data = np.asarray(K.data)
    r2 = g2l[rows]
    c2 = g2l[cols]
    mask = (r2 >= 0) & (c2 >= 0)
    K_free = ff.FluxSparseMatrix(
        jnp.asarray(r2[mask]),
        jnp.asarray(c2[mask]),
        jnp.asarray(data[mask]),
        int(free.size),
    )
    K_free = K_free.coalesce()
    _mark("build K_free")

    mv_free = jax.jit(K_free.matvec)
    if timing:
        mv_free(F_free).block_until_ready()
        _mark("mv_free warmup")
    if symmetry_check:
        key_x = jax.random.PRNGKey(0)
        key_y = jax.random.PRNGKey(1)
        x = jax.random.normal(key_x, (K_free.n_dofs,), dtype=F_free.dtype)
        y = jax.random.normal(key_y, (K_free.n_dofs,), dtype=F_free.dtype)
        a = jnp.vdot(x, mv_free(y))
        b = jnp.vdot(y, mv_free(x))
        denom = jnp.abs(a) + 1e-30
        print("symmetry ratio =", float(jnp.abs(a - b) / denom))
    u_free, info = ff.cg_solve(
        K_free,
        F_free,
        tol=cg_tol,
        maxiter=cg_maxiter,
        preconditioner=None if cg_precond == "none" else cg_precond,
    )
    _mark("cg solve")

    u = jnp.zeros(n_total, dtype=u_free.dtype).at[free_j].set(u_free)
    if not np.all(dir_vals == 0.0):
        u = u.at[dir_dofs].set(dir_vals_j)
    F_from_U = mv_free(u_free)
    F_pred_full = K.matvec(u)
    residual_full = F_pred_full - F_j

    res_norm = np.linalg.norm(np.asarray(F_free - F_from_U))
    rhs_norm = np.linalg.norm(np.asarray(F_free))
    rel_res = res_norm / rhs_norm

    if verbose:
        print("K1:", (int(K1.n_dofs), int(K1.n_dofs)))
        print("K2:", (int(K2.n_dofs), int(K2.n_dofs)))
        print("B1:", (int(K_contact.n_dofs), int(K_contact.n_dofs)))
        print("len(t1):", int(contact_facets_top.shape[0]), "len(t2):", int(contact_facets_bot.shape[0]))
        print("sum|F_from_U - F_free|:", float(np.sum(np.abs(np.asarray(F_from_U) - np.asarray(F_free)))))
        print("cg iters:", int(info.get("iters", -1)))
        print("cg residual_norm:", float(info.get("residual_norm", float("nan"))))
        print("K.shape, u.shape, F_pred_full.shape", (n_total, n_total), u.shape, F_pred_full.shape)
        print("F_pred_full.shape, F.shape", F_pred_full.shape, F_j.shape)
        print("len(F_pred_full) =", int(F_pred_full.shape[0]))
        print("len(F) =", len(F))
        print("‖residual‖₂ =", float(np.linalg.norm(np.asarray(residual_full))))
        print("relative residual =", float(rel_res))

    summary = {
        "K1_shape": (int(K1.n_dofs), int(K1.n_dofs)),
        "K2_shape": (int(K2.n_dofs), int(K2.n_dofs)),
        "K_contact_shape": (int(K_contact.n_dofs), int(K_contact.n_dofs)),
        "len_contact_top": int(contact_facets_top.shape[0]),
        "len_contact_bot": int(contact_facets_bot.shape[0]),
        "sum_abs_Fdiff": float(np.sum(np.abs(np.asarray(F_from_U) - np.asarray(F_free)))),
        "residual_norm_2": float(np.linalg.norm(np.asarray(residual_full))),
        "relative_residual": float(rel_res),
    }

    if log_path:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"{k}={v}" for k, v in summary.items()) + "\n")

    points = None
    u_nodes = None
    u_np = np.asarray(u)
    if vtu_path:
        _export_fluxfem_combined_vtu(
            mesh_top,
            mesh_bot,
            u_np,
            vtu_path,
            contact_facets_top=contact_facets_top,
            contact_facets_bot=contact_facets_bot,
            force_facets_top=force_facets_top,
            dirichlet_facets_bot=dirichlet_facets_bot,
        )

    if plot_path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(np.asarray(F_from_U))
        ax2.plot(np.asarray(F_free))
        fig.savefig(plot_path)

    if npz_path or return_nodes:
        n_nodes_total = mesh_top.n_nodes + mesh_bot.n_nodes
        points = np.vstack([np.asarray(mesh_top.coords), np.asarray(mesh_bot.coords)])
        u_nodes = np.asarray(u_np[: n_nodes_total * 3]).reshape(n_nodes_total, 3)
    if npz_path:
        np.savez(npz_path, u=u_np, points=points, u_nodes=u_nodes)

    return NitscheContactResult(summary=summary, u=u_np, points=points, u_nodes=u_nodes)


def run_fluxfem_oneside_demo(
    params: NitscheContactParams,
    *,
    platform: str = "auto",
    x64: bool = True,
    xla_cpu_parallelism: int | None = None,
    omp_threads: int | None = None,
    low_mem: bool = False,
    n_chunks: int | None = None,
    timing: bool = False,
    cg_tol: float = 1e-8,
    cg_maxiter: int | None = None,
    cg_precond: str = "none",
    symmetry_check: bool = False,
    bench_contact: bool = False,
    debug_contact: bool = False,
    log_path: str | None = None,
    vtu_path: str | None = None,
    plot_path: str | None = None,
    npz_path: str | None = None,
    verbose: bool = True,
    return_nodes: bool = False,
) -> NitscheContactResult:
    import os
    import time
    import jax
    import jax.numpy as jnp
    import fluxfem as ff

    def _append_xla_flag(flag: str) -> None:
        existing = os.environ.get("XLA_FLAGS", "")
        if flag in existing.split():
            return
        if existing:
            os.environ["XLA_FLAGS"] = f"{existing} {flag}"
        else:
            os.environ["XLA_FLAGS"] = flag

    if low_mem:
        if platform == "auto":
            platform = "cpu"
        x64 = False
        if omp_threads is None:
            omp_threads = 1

    if platform != "auto":
        os.environ.setdefault("JAX_PLATFORMS", platform)
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
    if xla_cpu_parallelism is not None:
        _append_xla_flag(f"--xla_cpu_compilation_parallelism={xla_cpu_parallelism}")
    if omp_threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))

    jax.config.update("jax_enable_x64", x64)
    timing_state = {"start": time.perf_counter(), "last": time.perf_counter()}

    def _mark(label: str) -> None:
        if not timing:
            return
        now = time.perf_counter()
        elapsed = now - timing_state["last"]
        total = now - timing_state["start"]
        print(f"[timing] {label}: +{elapsed:.3f}s (total {total:.3f}s)", flush=True)
        timing_state["last"] = now

    if timing:
        _mark("start")

    lam, mu = _lame_parameters(params)
    D = _isotropic_3d_D(lam, mu, dtype=jnp.asarray(0.0).dtype)
    _mark("material params")

    box_top = ff.StructuredTetTensorBox(
        nx=params.nx_top,
        ny=params.ny_top,
        nz=params.nz_top,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        origin=(0.0, 0.0, 0.0),
    )
    box_bot = ff.StructuredTetTensorBox(
        nx=params.nx_bot,
        ny=params.ny_bot,
        nz=params.nz_bot,
        lx=1.0,
        ly=1.0,
        lz=0.5,
        origin=(0.5, 0.5, -0.5),
    )
    mesh_top = box_top.build()
    mesh_bot = box_bot.build()
    _mark("mesh build")

    space_top = ff.make_tet_space(mesh_top, dim=3)
    space_bot = ff.make_tet_space(mesh_bot, dim=3)
    _mark("space build")

    K1 = space_top.assemble_bilinear_form(ff.linear_elasticity_form, params=D, n_chunks=n_chunks)
    _mark("K1 assemble")
    K2 = space_bot.assemble_bilinear_form(ff.linear_elasticity_form, params=D, n_chunks=n_chunks)
    _mark("K2 assemble")

    contact_facets_top = mesh_top.facets_on_plane(axis=2, value=0.0)
    contact_facets_bot = mesh_bot.facets_on_plane(axis=2, value=0.0)
    x0, y0, _ = box_bot.origin
    x1 = x0 + box_bot.lx
    y1 = y0 + box_bot.ly
    dx_top = box_top.lx / box_top.nx
    dy_top = box_top.ly / box_top.ny
    pad = 2.0 * min(dx_top, dy_top)
    contact_facets_top = mesh_top.facets_on_plane_box(
        axis=2,
        value=0.0,
        x=(x0 - pad, x1 + pad),
        y=(y0 - pad, y1 + pad),
        mode="centroid",
    )
    force_facets_top = mesh_top.facets_on_plane(axis=2, value=1.0)
    dirichlet_facets_bot = mesh_bot.facets_on_plane(axis=2, value=-0.5)
    _mark("boundary facets")

    side_top = ff.ContactSide.from_facets(mesh_top, contact_facets_top, space_top)
    contact_space = ff.OneSidedContactSurfaceSpace.from_side(
        side_top,
        quad_order=int(params.quad_order),
    )
    _mark("contact setup")

    dx_top, dy_top, dz_top = box_top.lx / box_top.nx, box_top.ly / box_top.ny, box_top.lz / box_top.nz
    dx_bot, dy_bot, dz_bot = box_bot.lx / box_bot.nx, box_bot.ly / box_bot.ny, box_bot.lz / box_bot.nz
    h = min(dx_top, dy_top, dz_top, dx_bot, dy_bot, dz_bot)

    alpha = _alpha_from_params(params)
    params_contact = ff.Params(alpha=float(alpha), lam=float(lam), mu=float(mu))

    def u_hat_fn(x_q: np.ndarray) -> np.ndarray:
        return np.zeros((x_q.shape[0], 3), dtype=float)

    if debug_contact:
        print("n_contact_facets_top =", int(contact_facets_top.shape[0]))
        print("quad_order =", int(params.quad_order))

    K_contact_dense, f_contact = contact_space.assemble_bilinear(
        u_hat_fn,
        params_contact,
    )
    if verbose:
        print("[contact] onesided Dirichlet uses numpy-only assembly", flush=True)
    n_contact_dofs = int(K_contact_dense.shape[0])
    K1 = K1.add_dense(K_contact_dense)
    _mark("K_contact assemble")
    if bench_contact:
        _K_contact_2, _f_contact_2 = contact_space.assemble_bilinear(
            u_hat_fn,
            params_contact,
        )
        _mark("K_contact assemble (2nd)")

    K = ff.block_diag_flux(K1, K2)

    force_surface = ff.SurfaceMesh.from_facets(mesh_top.coords, force_facets_top)
    area = float(np.sum(force_surface.facet_areas()))
    pressure = params.total_force / area
    top_F = force_surface.assemble_load(
        load=np.array([0.0, 0.0, pressure]),
        dim=3,
        n_total_nodes=mesh_top.n_nodes,
    )
    top_F = np.asarray(top_F) + np.asarray(f_contact)
    bot_F = np.zeros(space_bot.n_dofs, dtype=float)
    F = np.hstack([top_F, bot_F])

    dir_dofs_bot = mesh_bot.boundary_dofs_plane(axis=2, value=-0.5, dof_per_node=3)
    dir_dofs = dir_dofs_bot + space_top.n_dofs
    dir_vals = np.zeros(dir_dofs.shape[0], dtype=float)

    n_total = int(K.n_dofs)
    dir_dofs = np.asarray(dir_dofs, dtype=int)
    free_mask = np.ones(n_total, dtype=bool)
    free_mask[dir_dofs] = False
    free = np.nonzero(free_mask)[0]
    free_j = jnp.asarray(free)
    dir_vals_j = jnp.asarray(dir_vals)

    F_j = jnp.asarray(F)
    if np.all(dir_vals == 0.0):
        F_free = F_j[free_j]
    else:
        u_dir = jnp.zeros(n_total, dtype=F_j.dtype).at[dir_dofs].set(dir_vals_j)
        F_free = (F_j - K.matvec(u_dir))[free_j]
    _mark("apply dirichlet")

    g2l = -np.ones(n_total, dtype=np.int32)
    g2l[free] = np.arange(free.size, dtype=np.int32)
    rows = np.asarray(K.pattern.rows)
    cols = np.asarray(K.pattern.cols)
    data = np.asarray(K.data)
    r2 = g2l[rows]
    c2 = g2l[cols]
    mask = (r2 >= 0) & (c2 >= 0)
    K_free = ff.FluxSparseMatrix(
        jnp.asarray(r2[mask]),
        jnp.asarray(c2[mask]),
        jnp.asarray(data[mask]),
        int(free.size),
    )
    K_free = K_free.coalesce()
    _mark("build K_free")

    mv_free = jax.jit(K_free.matvec)
    if timing:
        mv_free(F_free).block_until_ready()
        _mark("mv_free warmup")
    if symmetry_check:
        key_x = jax.random.PRNGKey(0)
        key_y = jax.random.PRNGKey(1)
        x = jax.random.normal(key_x, (K_free.n_dofs,), dtype=F_free.dtype)
        y = jax.random.normal(key_y, (K_free.n_dofs,), dtype=F_free.dtype)
        a = jnp.vdot(x, mv_free(y))
        b = jnp.vdot(y, mv_free(x))
        denom = jnp.abs(a) + 1e-30
        print("symmetry ratio =", float(jnp.abs(a - b) / denom))
    u_free, info = ff.cg_solve(
        K_free,
        F_free,
        tol=cg_tol,
        maxiter=cg_maxiter,
        preconditioner=None if cg_precond == "none" else cg_precond,
    )
    _mark("cg solve")

    u = jnp.zeros(n_total, dtype=u_free.dtype).at[free_j].set(u_free)
    if not np.all(dir_vals == 0.0):
        u = u.at[dir_dofs].set(dir_vals_j)
    F_from_U = mv_free(u_free)
    F_pred_full = K.matvec(u)
    residual_full = F_pred_full - F_j

    res_norm = np.linalg.norm(np.asarray(F_free - F_from_U))
    rhs_norm = np.linalg.norm(np.asarray(F_free))
    rel_res = res_norm / rhs_norm

    if verbose:
        print("K1:", (int(K1.n_dofs), int(K1.n_dofs)))
        print("K2:", (int(K2.n_dofs), int(K2.n_dofs)))
        print("B1:", (n_contact_dofs, n_contact_dofs))
        print("len(t1):", int(contact_facets_top.shape[0]), "len(t2):", int(contact_facets_bot.shape[0]))
        print("sum|F_from_U - F_free|:", float(np.sum(np.abs(np.asarray(F_from_U) - np.asarray(F_free)))))
        print("cg iters:", int(info.get("iters", -1)))
        print("cg residual_norm:", float(info.get("residual_norm", float("nan"))))
        print("K.shape, u.shape, F_pred_full.shape", (n_total, n_total), u.shape, F_pred_full.shape)
        print("F_pred_full.shape, F.shape", F_pred_full.shape, F_j.shape)
        print("len(F_pred_full) =", int(F_pred_full.shape[0]))
        print("len(F) =", len(F))
        print("‖residual‖₂ =", float(np.linalg.norm(np.asarray(residual_full))))
        print("relative residual =", float(rel_res))

    summary = {
        "K1_shape": (int(K1.n_dofs), int(K1.n_dofs)),
        "K2_shape": (int(K2.n_dofs), int(K2.n_dofs)),
        "K_contact_shape": (n_contact_dofs, n_contact_dofs),
        "len_contact_top": int(contact_facets_top.shape[0]),
        "len_contact_bot": int(contact_facets_bot.shape[0]),
        "sum_abs_Fdiff": float(np.sum(np.abs(np.asarray(F_from_U) - np.asarray(F_free)))),
        "residual_norm_2": float(np.linalg.norm(np.asarray(residual_full))),
        "relative_residual": float(rel_res),
    }

    if log_path:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"{k}={v}" for k, v in summary.items()) + "\n")

    points = None
    u_nodes = None
    u_np = np.asarray(u)

    if vtu_path:
        _export_fluxfem_combined_vtu(
            mesh_top,
            mesh_bot,
            u_np,
            vtu_path,
            contact_facets_top=contact_facets_top,
            contact_facets_bot=contact_facets_bot,
            force_facets_top=force_facets_top,
            dirichlet_facets_bot=dirichlet_facets_bot,
        )

    if plot_path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(np.asarray(F_from_U))
        ax2.plot(np.asarray(F_free))
        fig.savefig(plot_path)

    if npz_path or return_nodes:
        n_nodes_total = mesh_top.n_nodes + mesh_bot.n_nodes
        points = np.vstack([np.asarray(mesh_top.coords), np.asarray(mesh_bot.coords)])
        u_nodes = np.asarray(u_np[: n_nodes_total * 3]).reshape(n_nodes_total, 3)
    if npz_path:
        np.savez(npz_path, u=u_np, points=points, u_nodes=u_nodes)

    return NitscheContactResult(summary=summary, u=u_np, points=points, u_nodes=u_nodes)


def compare_displacement_fields(
    ref: NitscheContactResult,
    cmp: NitscheContactResult,
    *,
    tol: float = 1e-10,
) -> dict[str, float]:
    if ref.points is None or ref.u_nodes is None or cmp.points is None or cmp.u_nodes is None:
        raise ValueError("both results must include points and u_nodes for comparison")

    points_ref = np.asarray(ref.points, dtype=float)
    points_cmp = np.asarray(cmp.points, dtype=float)
    u_ref = np.asarray(ref.u_nodes, dtype=float)
    u_cmp = np.asarray(cmp.u_nodes, dtype=float)

    if points_ref.shape[0] != u_ref.shape[0] or points_cmp.shape[0] != u_cmp.shape[0]:
        raise ValueError("points and u_nodes lengths must match")

    idx = np.empty(points_ref.shape[0], dtype=int)
    dist2 = np.empty(points_ref.shape[0], dtype=float)
    for i, p in enumerate(points_ref):
        d2 = np.sum((points_cmp - p) ** 2, axis=1)
        j = int(np.argmin(d2))
        idx[i] = j
        dist2[i] = d2[j]

    max_dist = float(np.max(np.sqrt(dist2))) if dist2.size else 0.0
    if max_dist > tol:
        print(f"warning: max coord mismatch = {max_dist:.3e} (tol {tol:.3e})")

    u_cmp_reordered = u_cmp[idx]
    diff = u_ref - u_cmp_reordered
    diff_norm = np.linalg.norm(diff, axis=1)
    max_u = float(np.max(diff_norm)) if diff_norm.size else 0.0
    rms_u = float(np.sqrt(np.mean(diff_norm**2))) if diff_norm.size else 0.0
    return {
        "n_nodes": int(points_ref.shape[0]),
        "max_coord_mismatch": max_dist,
        "max_u_diff": max_u,
        "rms_u_diff": rms_u,
    }

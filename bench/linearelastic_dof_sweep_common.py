from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from fluxfem import (
    FluxSparseMatrix,
    LinearSolver,
    StructuredHexBox,
    build_cg_operator,
    condense_dirichlet_system,
)
from fluxfem.tools.timer import SectionTimer
from fluxfem.physics.operators import sym_grad



def make_structured_mesh(n: int, ny_mult: float, nz_mult: float, lx: float, ly: float, lz: float):
    ny = max(1, int(round(n * ny_mult)))
    nz = max(1, int(round(n * nz_mult)))
    mesh = StructuredHexBox(nx=n, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
    return mesh, ny, nz


def compute_dirichlet_dofs(mesh) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(mesh.coords)
    xmin = float(coords[:, 0].min())
    dir_dofs = mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
        components="xyz",
        dof_per_node=3,
    )
    dir_vals = np.zeros(len(dir_dofs), dtype=float)
    return dir_dofs, dir_vals


def condense_flux_dirichlet(K, F, dir_dofs, dir_vals):
    system = condense_dirichlet_system(K, F, dir_dofs, dir_vals)
    return system.K, system.F, system.free_dofs


def _residual_error(K_ff, F_free, u) -> float:
    rhs = np.asarray(F_free, dtype=float)
    res = K_ff @ np.asarray(u, dtype=float) - rhs
    denom = np.linalg.norm(rhs)
    return float(np.linalg.norm(res) / (denom if denom > 0 else 1.0))


def time_flux_cg_samples(
    K_ff,
    F_free,
    repeats: int,
    tol: float,
    maxiter: int,
    cg_impl: str,
    cg_matvec: str,
    cg_precon: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      times: (repeats,) seconds
      iters: (repeats,) iteration counts (ints as float array)
    Note:
      First call includes any JAX first-call overhead (compile) for cg_solve path.
    """
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")

    coo = K_ff.tocoo()
    coo_tuple = (
        jnp.asarray(coo.row, dtype=jnp.int32),
        jnp.asarray(coo.col, dtype=jnp.int32),
        jnp.asarray(coo.data),
        K_ff.shape[0],
    )
    b = jnp.asarray(F_free)

    times = []
    iters = []
    residual_errors = []

    timer = SectionTimer()

    if cg_precon == "block_jacobi":
        precon = "block_jacobi"
    else:
        precon = "jacobi"

    solver = "cg" if cg_impl == "custom" else "cg_jax"
    cg_op = build_cg_operator(
        coo_tuple,
        matvec=cg_matvec,
        preconditioner=precon,
        solver=solver,
        dof_per_node=3,
    )

    for _ in range(repeats):
        with timer.section("solve_cg"):
            u, info = cg_op.solve(b, tol=tol, maxiter=maxiter)
            jax.block_until_ready(u)
        times.append(timer.last("solve_cg"))
        iters.append(int(info.get("iters", 0)))
        residual_errors.append(_residual_error(K_ff, F_free, u))

    return (
        np.asarray(times, dtype=float),
        np.asarray(iters, dtype=float),
        np.asarray(residual_errors, dtype=float),
    )


def time_spsolve_samples(
    K_full,
    F_full,
    dirichlet,
    K_ff,
    F_free,
    free,
    repeats: int,
    backend: str,
    spsolve_impl: str,
) -> tuple[np.ndarray, np.ndarray]:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    times = []
    residual_errors = []
    if backend == "gpu":
        solver = LinearSolver(method="spdirect_solve_gpu")
    elif spsolve_impl == "jax":
        solver = LinearSolver(method="spsolve_jax")
    else:
        solver = LinearSolver(method="spsolve")
    timer = SectionTimer()
    for _ in range(repeats):
        try:
            with timer.section("solve_spsolve"):
                u = solver.solve(K_full, F_full, dirichlet=dirichlet, n_total=K_full.n_dofs)[0]
        except Exception as exc:
            if backend == "gpu":
                print(f"[bench] GPU spsolve failed ({exc}); recording NaN and skipping remaining repeats.")
                times.extend([float('nan')] * (repeats - len(times)))
                residual_errors.extend([float('nan')] * (repeats - len(residual_errors)))
                break
            raise
        times.append(timer.last("solve_spsolve"))
        residual_errors.append(_residual_error(K_ff, F_free, u[free]))
    return np.asarray(times, dtype=float), np.asarray(residual_errors, dtype=float)


def time_petsc_samples(
    K_ff,
    F_free,
    repeats: int,
    ksp_type: str,
    pc_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    times = []
    residual_errors = []
    timer = SectionTimer()
    from fluxfem.solver.petsc import petsc_solve

    for _ in range(repeats):
        with timer.section("solve_petsc"):
            u = petsc_solve(K_ff, F_free, ksp_type=ksp_type, pc_type=pc_type)
        times.append(timer.last("solve_petsc"))
        residual_errors.append(_residual_error(K_ff, F_free, u))
    return np.asarray(times, dtype=float), np.asarray(residual_errors, dtype=float)


def time_petsc_shell_samples(
    K_ff,
    F_free,
    repeats: int,
    ksp_type: str,
    pc_type: str,
    preconditioner: str | None,
    rtol: float,
    atol: float,
    max_it: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    times = []
    residual_errors = []
    iters = []
    timer = SectionTimer()
    from fluxfem.solver.petsc import petsc_shell_solve

    for _ in range(repeats):
        with timer.section("solve_petsc_shell"):
            u, info = petsc_shell_solve(
                K_ff,
                F_free,
                ksp_type=ksp_type,
                pc_type=pc_type,
                preconditioner=preconditioner,
                rtol=rtol,
                atol=atol,
                max_it=max_it,
                return_info=True,
            )
        times.append(timer.last("solve_petsc_shell"))
        iters.append(info.get("iters", np.nan))
        residual_errors.append(_residual_error(K_ff, F_free, u))
    return (
        np.asarray(times, dtype=float),
        np.asarray(residual_errors, dtype=float),
        np.asarray(iters, dtype=float),
    )


def summarize(samples: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(samples)),
        "mean": float(np.mean(samples)),
        "max": float(np.max(samples)),
        "median": float(np.median(samples)),
    }


def prepare_kernel_breakdown(elem_data, kernel, D, pattern, *, breakdown, kernel_breakdown):
    kernel_stage = None
    backend_stage = None
    sym_stage = None
    bdb_stage = None
    if breakdown:
        kernel_stage = jax.jit(lambda: jax.vmap(kernel)(elem_data))
        backend_stage = jax.jit(lambda Ke: FluxSparseMatrix(pattern, Ke.reshape(-1)))
    if kernel_breakdown:
        def _sym(ctx):
            Bu = sym_grad(ctx.trial)
            Bv = Bu if ctx.test is ctx.trial else sym_grad(ctx.test)
            return Bu, Bv

        sym_stage = jax.jit(lambda: jax.vmap(_sym)(elem_data))
        bdb_stage = jax.jit(
            lambda Bu, Bv: jnp.einsum(
                "eqik,kl,eqlm->eqim", jnp.swapaxes(Bv, 2, 3), D, Bu
            )
        )
    return kernel_stage, backend_stage, sym_stage, bdb_stage

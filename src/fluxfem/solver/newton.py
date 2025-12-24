from __future__ import annotations
import time

import numpy as np
import jax
import jax.numpy as jnp

from ..core.assembly import (
    assemble_residual_scatter,
    assemble_jacobian_scatter,
    make_element_residual_kernel,
    make_element_jacobian_kernel,
    make_sparsity_pattern,
)
from ..core.solver import spdirect_solve_cpu, spdirect_solve_gpu
from .cg import cg_solve, cg_solve_jax
from .result import SolverResult
from .sparse import SparsityPattern, FluxSparseMatrix
from .dirichlet import _normalize_dirichlet


def newton_solve(
    space,
    res_form,
    u0,
    params,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 20,
    linear_solver: str = "spsolve",  # "spsolve", "spdirect_solve_gpu", "cg" (jax), "cg_jax", or "cg_custom"
    linear_maxiter: int | None = None,
    linear_tol: float | None = None,
    linear_preconditioner=None,
    dirichlet=None,
    callback=None,
    line_search: bool = False,
    max_ls: int = 10,
    ls_c: float = 1e-4,
    external_vector=None,
    jacobian_pattern=None,
):
    """
    Gridap-style Newton–Raphson solver on free DOFs only.

    - Unknown vector = free DOFs (Dirichlet eliminated).
    - Residual/Jacobian are assembled on full DOFs; we slice to free DOFs.
    - Convergence: ||R_free||_inf < max(atol, tol * ||R_free0||_inf).
    - external_vector: optional global RHS (internal - external).
    - CG path accepts an operator with matvec that acts on free DOFs via a wrapper.
    - linear_preconditioner: forwarded to cg_solve/cg_solve_jax (None | "jacobi" | "block_jacobi" | callable).
    - linear_tol: CG tolerance (defaults to 0.1 * tol if not provided).
    - jacobian_pattern: optional SparsityPattern to reuse sparsity across load steps.
    """

    if dirichlet is not None:
        dir_dofs, dir_vals = dirichlet
        dir_dofs, dir_vals = _normalize_dirichlet(dir_dofs, dir_vals)
        if dir_vals.ndim == 0:
            dir_vals = np.full(dir_dofs.shape[0], float(dir_vals))
        all_dofs = np.arange(space.n_dofs, dtype=int)
        mask = np.ones(space.n_dofs, dtype=bool)
        mask[dir_dofs] = False
        free_dofs = all_dofs[mask]
    else:
        dir_dofs = dir_vals = None
        free_dofs = np.arange(space.n_dofs, dtype=int)

    free_dofs_j = jnp.asarray(free_dofs, dtype=jnp.int32)
    # For block-Jacobi (3x3 per node) we keep node ids of free dofs.
    node_ids = free_dofs // 3
    node_ids_unique, node_ids_inv = np.unique(node_ids, return_inverse=True)
    n_block = len(node_ids_unique)
    dir_dofs_j = jnp.asarray(dir_dofs, dtype=jnp.int32) if dir_dofs is not None else None
    dir_vals_j = jnp.asarray(dir_vals, dtype=jnp.asarray(u0).dtype) if dir_vals is not None else None

    # Unknown is free DOFs only
    u = jnp.asarray(u0)[free_dofs]

    # Sparsity pattern does not depend on u; cache once
    J_pattern = jacobian_pattern if jacobian_pattern is not None else make_sparsity_pattern(
        space, with_idx=True
    )

    # Build free-DOF subpattern once to avoid scatter/gather in every matvec.
    free_map = -np.ones(space.n_dofs, dtype=np.int32)
    free_map[free_dofs] = np.arange(len(free_dofs), dtype=np.int32)
    pat_rows = np.asarray(J_pattern.rows)
    pat_cols = np.asarray(J_pattern.cols)
    mask_free = (free_map[pat_rows] >= 0) & (free_map[pat_cols] >= 0)
    free_data_idx = jnp.asarray(np.nonzero(mask_free)[0], dtype=jnp.int32)
    rows_f = free_map[pat_rows[mask_free]]
    cols_f = free_map[pat_cols[mask_free]]
    diag_idx_f = np.nonzero(rows_f == cols_f)[0].astype(np.int32)
    J_free_pattern = SparsityPattern(
        rows=jnp.asarray(rows_f, dtype=jnp.int32),
        cols=jnp.asarray(cols_f, dtype=jnp.int32),
        n_dofs=int(len(free_dofs)),
        idx=None,
        diag_idx=jnp.asarray(diag_idx_f, dtype=jnp.int32),
    )

    def restrict_free_matrix(J: FluxSparseMatrix) -> FluxSparseMatrix:
        data_f = jnp.asarray(J.data)[free_data_idx]
        return FluxSparseMatrix(J_free_pattern, data_f)

    def build_block_jacobi(J_free: FluxSparseMatrix):
        """
        Build 3x3 block-Jacobi inverse per free node.
        Assumes DOF ordering per node is [ux, uy, uz].
        """
        if len(free_dofs) % 3 != 0:
            raise ValueError("block_jacobi assumes 3 DOFs per node.")
        rows = np.asarray(J_free.rows)
        cols = np.asarray(J_free.cols)
        data = np.asarray(J_free.data)
        block_rows = node_ids_inv[rows]
        block_cols = node_ids_inv[cols]
        local_r = rows % 3
        local_c = cols % 3
        mask_blk = block_rows == block_cols
        blk_rows = block_rows[mask_blk]
        blk_lr = local_r[mask_blk]
        blk_lc = local_c[mask_blk]
        blk_data = data[mask_blk]
        inv_blocks = np.zeros((n_block, 3, 3), dtype=blk_data.dtype)
        inv_blocks[blk_rows, blk_lr, blk_lc] += blk_data
        inv_blocks = jnp.asarray(inv_blocks)
        # Add tiny damping to avoid singular blocks
        inv_blocks = inv_blocks + 1e-12 * jnp.eye(3)[None, :, :]
        inv_blocks = jnp.linalg.inv(inv_blocks)

        def precon(r):
            r_blocks = r.reshape((n_block, 3))
            z_blocks = jnp.einsum("bij,bj->bi", inv_blocks, r_blocks)
            return z_blocks.reshape((-1,))

        return precon

    def expand_full(u_free: jnp.ndarray) -> jnp.ndarray:
        if dir_dofs is None:
            return u_free
        u_full = jnp.zeros((space.n_dofs,), dtype=u_free.dtype)
        u_full = u_full.at[free_dofs_j].set(u_free)
        u_full = u_full.at[dir_dofs_j].set(dir_vals_j)
        return u_full

    def eval_residual(u_free_vec):
        """Residual on free DOFs only."""
        u_full = expand_full(u_free_vec)
        R_full = assemble_R(u_full)
        if external_vector is not None:
            R_full = R_full - external_vector
        R_free = R_full[free_dofs_j]
        res_inf = float(jnp.linalg.norm(R_free, ord=jnp.inf))
        res_two = float(jnp.linalg.norm(R_free, ord=2))
        return R_free, res_inf, res_two, u_full

    # Pre-jitted element kernels to avoid recompiling inside Newton
    res_kernel = make_element_residual_kernel(res_form, params)
    jac_kernel = make_element_jacobian_kernel(res_form, params)

    def assemble_R(u_full_vec):
        return assemble_residual_scatter(space, res_form, u_full_vec, params, kernel=res_kernel)

    eff_linear_tol = linear_tol if linear_tol is not None else max(0.1 * tol, 1e-12)

    def assemble_J(u_full_vec):
        return assemble_jacobian_scatter(
            space,
            res_form,
            u_full_vec,
            params,
            kernel=jac_kernel,
            sparse=True,
            return_flux_matrix=True,
            pattern=J_pattern,
        )

    # Initial residual/Jacobian
    R_full_init = assemble_R(expand_full(u))
    if external_vector is not None:
        R_full_init = R_full_init - external_vector
    finite_init = jnp.all(jnp.isfinite(R_full_init))
    if not bool(jax.block_until_ready(finite_init)):
        n_bad = int(jnp.size(R_full_init) - jnp.count_nonzero(jnp.isfinite(R_full_init)))
        rows_dbg, data_dbg, n_dofs_dbg = assemble_residual_scatter(
            space, res_form, expand_full(u), params, sparse=True
        )
        rows_np = np.asarray(rows_dbg)
        data_np = np.asarray(data_dbg)
        bad_data = np.count_nonzero(~np.isfinite(data_np))
        row_min = int(rows_np.min()) if rows_np.size else -1
        row_max = int(rows_np.max()) if rows_np.size else -1
        raise RuntimeError(f"[newton] init residual has non-finite entries: {n_bad}")
    R_free = R_full_init[free_dofs_j]
    res0_inf = float(jnp.linalg.norm(R_free, ord=jnp.inf))
    res0_two = float(jnp.linalg.norm(R_free, ord=2))
    u_full = expand_full(u)
    if res0_inf == 0.0:
        return expand_full(u), SolverResult(
            converged=True,
            iters=0,
            residual_norm=0.0,
            residual0=0.0,
            rel_residual=0.0,
            tol=tol,
            atol=atol,
            stopping_criterion=max(atol, tol * 0.0),
        )

    if callback is not None:
        callback({"iter": 0, "res_inf": res0_inf, "res_two": res0_two, "rel_residual": 1.0, "alpha": 1.0, "step_norm": np.nan})

    J = assemble_J(u_full)
    finite_j = jnp.all(jnp.isfinite(J.data))
    if not bool(jax.block_until_ready(finite_j)):
        n_bad = int(jnp.size(J.data) - jnp.count_nonzero(jnp.isfinite(J.data)))
        raise RuntimeError(f"[newton] init Jacobian has non-finite entries: {n_bad}")
    J_free = restrict_free_matrix(J)
    for k in range(maxiter):
        # --- Newton residual (iteration start) ---
        t_iter0 = time.perf_counter()

        # Always log this to show progress.
        res_prev_inf = jnp.linalg.norm(R_free, ord=jnp.inf)
        res_prev_two = jnp.linalg.norm(R_free, ord=2)
        # JAX is async; synchronize to ensure logs are emitted.
        res_prev_inf_f = float(jax.block_until_ready(res_prev_inf))
        res_prev_two_f = float(jax.block_until_ready(res_prev_two))
        if not (np.isfinite(res_prev_inf_f) and np.isfinite(res_prev_two_f)):
            raise RuntimeError("[newton] residual became non-finite; aborting.")

        crit = max(atol, tol * res0_inf)
        print(
            f"[newton] k={k:02d} START  |R|inf={res_prev_inf_f:.3e} |R|2={res_prev_two_f:.3e}  crit={crit:.3e}",
            flush=True,
        )

        # --- Linear solve (J_free * du = -R_free) ---
        rhs = jnp.asarray(-R_free, dtype=u.dtype)

        # Separate preconditioner build time from linear solve time.
        t_pre0 = time.perf_counter()
        cg_precon = linear_preconditioner
        linear_converged = True
        linear_residual = None
        lin_iters = None

        if linear_solver in ("cg", "cg_jax", "cg_custom"):
            # Preconditioner build
            if linear_preconditioner == "jacobi":
                print(f"[newton] k={k:02d}  PRECOND jacobi: diag...", flush=True)
                diag = jnp.asarray(J_free.diag(), dtype=rhs.dtype)
                diag = jax.block_until_ready(diag)  # If it's heavy, it blocks here.
                inv_diag = jnp.where(diag != 0.0, 1.0 / diag, 0.0)

                def cg_preconditioner_fn(r):
                    return inv_diag * r

                cg_precon = cg_preconditioner_fn

            elif linear_preconditioner == "block_jacobi":
                print(f"[newton] k={k:02d}  PRECOND block_jacobi: build...", flush=True)
                cg_precon = build_block_jacobi(J_free)
                # Sync point if build is heavy (apply once).
                _ = jax.block_until_ready(cg_precon(rhs))

            pre_dt = time.perf_counter() - t_pre0

            # Linear solve
            cg_solver = cg_solve_jax if linear_solver in ("cg", "cg_jax") else cg_solve
            print(f"[linear] k={k:02d} {linear_solver}: solve...", flush=True)
            t_lin0 = time.perf_counter()
            du_free, lin_info = cg_solver(
                J_free,
                rhs,
                tol=eff_linear_tol,
                maxiter=linear_maxiter,
                preconditioner=cg_precon,
            )
            du_free = jax.block_until_ready(du_free)
            lin_dt = time.perf_counter() - t_lin0

            linear_residual = lin_info.get("residual_norm")
            linear_converged = bool(lin_info.get("converged", True))
            lin_iters = lin_info.get("iters", None)

        elif linear_solver in ("spsolve", "spdirect_solve_gpu"):
            pre_dt = 0.0
            print(f"[linear] k={k:02d} {linear_solver}: csr/slice...", flush=True)
            t_lin0 = time.perf_counter()
            J_csr = J.to_csr()
            J_ff = J_csr[np.ix_(free_dofs, free_dofs)]
            print(f"[linear] k={k:02d} {linear_solver}: solve...", flush=True)
            if linear_solver == "spdirect_solve_gpu":
                du_free = spdirect_solve_gpu(J_ff, rhs)
            else:
                du_free = spdirect_solve_cpu(J_ff, rhs)
            du_free = jax.block_until_ready(du_free)
            lin_dt = time.perf_counter() - t_lin0
            lin_info = {"iters": 1, "converged": True}
            linear_converged = True
            lin_iters = 1

        else:
            raise ValueError(f"Unknown linear solver: {linear_solver}")

        lr = float(linear_residual) if linear_residual is not None else float("nan")
        print(
            f"[linear] k={k:02d} done iters={lin_iters} conv={linear_converged} lin_res={lr:.3e} "
            f"pre_dt={pre_dt:.3f}s lin_dt={lin_dt:.3f}s",
            flush=True,
        )

        # --- Trial update & residual evaluation ---
        # Start with alpha=1 and eval_residual (if heavy, assemble_R is heavy/compiled).
        alpha = 1.0
        u_trial_free = u + alpha * du_free

        print(f"[newton] k={k:02d}  EVAL alpha={alpha:.3e} ...", flush=True)
        t_eval0 = time.perf_counter()
        R_free_trial, res_trial_inf, res_trial_two, u_full_trial = eval_residual(u_trial_free)
        # eval_residual casts to float so it usually syncs, but keep this for safety.
        _ = jax.block_until_ready(R_free_trial)
        eval_dt = time.perf_counter() - t_eval0

        # --- Backtracking line search ---
        if line_search:
            accepted = False
            ls_used = 0
            for ls_iter in range(max_ls):
                ls_used = ls_iter + 1
                # Armijo on 2-norm
                if res_trial_two <= (1.0 - ls_c * alpha) * res_prev_two_f:
                    accepted = True
                    break
                alpha *= 0.5
                u_trial_free = u + alpha * du_free
                t_eval0 = time.perf_counter()
                R_free_trial, res_trial_inf, res_trial_two, u_full_trial = eval_residual(u_trial_free)
                _ = jax.block_until_ready(R_free_trial)
                eval_dt += time.perf_counter() - t_eval0  # Accumulate eval time.

            print(
                f"[newton] k={k:02d}  LS done  alpha={alpha:.3e} accepted={accepted} steps={ls_used}  |R|inf={res_trial_inf:.3e} |R|2={res_trial_two:.3e} eval_dt={eval_dt:.3f}s",
                flush=True,
            )
        else:
            print(
                f"[newton] k={k:02d}  STEP alpha={alpha:.3e}  |R|inf={res_trial_inf:.3e} |R|2={res_trial_two:.3e} eval_dt={eval_dt:.3f}s",
                flush=True,
            )

        # --- Commit update ---
        u = u_trial_free
        R_free = R_free_trial
        u_full = u_full_trial

        # Step norm (minimize host transfer: compute in jnp then sync to float).
        step_norm = float(jax.block_until_ready(jnp.linalg.norm(alpha * du_free, ord=2)))

        # callback
        if callback is not None:
            callback(
                {
                    "iter": k + 1,
                    "res_inf": res_trial_inf,
                    "res_two": res_trial_two,
                    "rel_residual": res_trial_inf / res0_inf,
                    "alpha": alpha,
                    "step_norm": step_norm,
                    "linear_iters": lin_info.get("iters"),
                    "linear_converged": linear_converged,
                    "linear_residual": lr,
                    "nan_detected": bool(np.isnan(res_trial_inf)),
                }
            )

        # --- Convergence check ---
        if res_trial_inf < crit and linear_converged and not np.isnan(res_trial_inf):
            it_dt = time.perf_counter() - t_iter0
            print(f"[newton] k={k:02d}  CONVERGED  dt={it_dt:.3f}s", flush=True)
            return u_full, SolverResult(
                converged=True,
                iters=k + 1,
                residual_norm=res_trial_inf,
                residual0=res0_inf,
                rel_residual=res_trial_inf / res0_inf,
                line_search_steps=(0 if not line_search else ls_used),
                linear_iters=lin_info.get("iters"),
                linear_converged=linear_converged,
                linear_residual=lr,
                tol=tol,
                atol=atol,
                stopping_criterion=crit,
                step_norm=step_norm,
                stop_reason="converged",
                nan_detected=bool(np.isnan(res_trial_inf)),
            )

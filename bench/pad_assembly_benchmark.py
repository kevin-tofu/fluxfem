#!/usr/bin/env python3
"""
Benchmark pad-enabled assembly paths to check compile churn and speed.

Usage (recommended):
  JAX_LOG_COMPILES=1 PYTHONPATH=src python bench/pad_assembly_benchmark.py --mode compare
"""

from __future__ import annotations

import argparse
import os
import time


def _parse_sizes(arg: str) -> list[int]:
    out: list[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="4,5,6", help="comma list of nx values")
    parser.add_argument("--intorder", type=int, default=2)
    parser.add_argument("--mode", choices=["compare", "pad", "nopad", "sweep"], default="compare")
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--jit", dest="jit", action="store_true")
    parser.add_argument("--no-jit", dest="jit", action="store_false")
    parser.set_defaults(jit=True)
    parser.add_argument("--log-compiles", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument("--sweep-repeats", type=int, default=2)
    parser.add_argument("--sweep-zigzag", action="store_true")
    parser.add_argument("--sweep-verbose", action="store_true")
    parser.add_argument("--pad-trace", action="store_true")
    return parser.parse_args()


def _block_until_ready(x):
    if hasattr(x, "block_until_ready"):
        return x.block_until_ready()
    if hasattr(x, "data"):
        data = getattr(x, "data")
        if hasattr(data, "block_until_ready"):
            data.block_until_ready()
    if isinstance(x, tuple):
        for item in x:
            _block_until_ready(item)
    return x


def _time_call(fn, *, warmup: int, repeat: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    out = fn()
    _block_until_ready(out)
    t1 = time.perf_counter()
    first = t1 - t0

    for _ in range(max(warmup - 1, 0)):
        out = fn()
        _block_until_ready(out)

    times = []
    for _ in range(repeat):
        t_start = time.perf_counter()
        out = fn()
        _block_until_ready(out)
        times.append(time.perf_counter() - t_start)
    avg = sum(times) / max(len(times), 1)
    return first, avg


def _bench_case(label, fn, *, warmup: int, repeat: int):
    first, avg = _time_call(fn, warmup=warmup, repeat=repeat)
    print(f"{label}: first={first:.4f}s avg={avg:.4f}s", flush=True)


def main() -> int:
    args = _parse_args()
    if args.cpu_only:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("JAX_PJRT_CLIENTS", "cpu")
    if args.log_compiles:
        os.environ.setdefault("JAX_LOG_COMPILES", "1")

    import jax
    import jax.numpy as jnp

    import fluxfem as ff
    import fluxfem.helpers_wf as h_wf
    from fluxfem.core.mixed_space import MixedFESpace

    jax.config.update("jax_enable_x64", True)

    def _sweep_times_args(fn, seq, make_args, *, verbose: bool):
        times = []
        for n_active in seq:
            args = make_args(n_active)
            t0 = time.perf_counter()
            out = fn(*args)
            _block_until_ready(out)
            dt = time.perf_counter() - t0
            times.append(dt)
            if verbose:
                print(f"  n_active={n_active}: {dt:.4f}s", flush=True)
        avg = sum(times) / max(len(times), 1)
        return avg, max(times) if times else 0.0

    def _sweep_times_fn(make_fn, seq, *, verbose: bool):
        times = []
        for n_active in seq:
            fn = make_fn(n_active)
            t0 = time.perf_counter()
            out = fn()
            _block_until_ready(out)
            dt = time.perf_counter() - t0
            times.append(dt)
            if verbose:
                print(f"  n_active={n_active}: {dt:.4f}s", flush=True)
        avg = sum(times) / max(len(times), 1)
        return avg, max(times) if times else 0.0

    def _run_sweep_case(label, seq, *, pad_fn, make_pad_args, make_nopad_fn, verbose: bool):
        print(label, flush=True)
        avg, peak = _sweep_times_args(pad_fn, seq, make_pad_args, verbose=verbose)
        print(f"  pad avg={avg:.4f}s peak={peak:.4f}s", flush=True)
        avg, peak = _sweep_times_fn(make_nopad_fn, seq, verbose=verbose)
        print(f"  nopad avg={avg:.4f}s peak={peak:.4f}s", flush=True)

    def linear_residual(ctx, u_elem, kappa):
        grad_u = jnp.einsum("qai,a->qi", ctx.trial.gradN, u_elem)
        r_int = kappa * jnp.einsum("qai,qi->qa", ctx.test.gradN, grad_u)
        return r_int

    def linear_kernel(ctx):
        integrand = ff.scalar_body_force_form(ctx, 1.0)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    def mixed_residuals():
        def res_u(v, u, p):
            p_ref = ff.unknown_ref("p")
            return (v * (u.val + p.alpha * p_ref.val)) * h_wf.dOmega()

        def res_p(q, p_field, p):
            u_ref = ff.unknown_ref("u")
            return (q * (p_field.val + p.beta * u_ref.val)) * h_wf.dOmega()

        return {"u": res_u, "p": res_p}

    sizes = _parse_sizes(args.sizes)
    modes = ["pad", "nopad"] if args.mode == "compare" else [args.mode]

    if args.mode == "sweep":
        nx_max = max(sizes)
        mesh = ff.StructuredHexBox(nx=nx_max, ny=nx_max, nz=nx_max, lx=1.0, ly=1.0, lz=1.0).build()
        space = ff.make_hex_space(mesh, dim=1, intorder=args.intorder)
        u0 = jnp.zeros(space.n_dofs, dtype=jnp.float64)
        kappa = 1.0
        batch = space.make_batched_assembler()

        res_ker = ff.make_element_residual_kernel(linear_residual, kappa)
        jac_ker = ff.make_element_jacobian_kernel(linear_residual, kappa)

        n_elems_list = [int(nx**3) for nx in sizes]
        seq = list(n_elems_list)
        if args.sweep_zigzag and len(n_elems_list) > 1:
            seq = n_elems_list + list(reversed(n_elems_list[1:-1]))
        seq = seq * max(args.sweep_repeats, 1)

        def _jit(fn):
            return jax.jit(fn) if args.jit else fn

        bilin_pad_fn = _jit(
            lambda mask: batch.assemble_bilinear_with_kernel(bilin_ker, mask=mask)
        )
        linear_pad_fn = _jit(
            lambda mask: batch.assemble_linear_with_kernel(linear_ker, mask=mask)
        )
        mass_pad_fn = _jit(
            lambda mask: batch.assemble_mass_matrix(mask=mask)
        )
        res_pad_fn = _jit(
            lambda mask, u=u0: batch.assemble_residual_with_kernel(res_ker, u, mask=mask)
        )
        jac_pad_fn = _jit(
            lambda mask, u=u0: batch.assemble_jacobian_with_kernel(
                jac_ker, u, mask=mask, sparse=False
            )
        )

        def _make_pad_mask(n_active: int):
            return (batch.make_mask(n_active),)

        def _make_nopad_bilinear(n_active: int):
            assembler = batch.slice(n_active)
            return _jit(lambda: assembler.assemble_bilinear_with_kernel(bilin_ker))

        def _make_nopad_linear(n_active: int):
            assembler = batch.slice(n_active)
            return _jit(lambda: assembler.assemble_linear_with_kernel(linear_ker))

        def _make_nopad_mass(n_active: int):
            assembler = batch.slice(n_active)
            return _jit(lambda: assembler.assemble_mass_matrix())

        def _make_nopad_residual(n_active: int):
            assembler = batch.slice(n_active)
            return _jit(lambda u=u0: assembler.assemble_residual_with_kernel(res_ker, u))

        def _make_nopad_jacobian(n_active: int):
            assembler = batch.slice(n_active)
            return _jit(
                lambda u=u0: assembler.assemble_jacobian_with_kernel(jac_ker, u, sparse=False)
            )

        if not args.skip_mixed:
            print("[sweep] mixed cases are skipped (no batched assembler yet).", flush=True)

        print("[sweep] sizes:", sizes, flush=True)
        print("[sweep] sequence:", seq, flush=True)
        for label, pad_fn, make_pad, make_nopad in [
            ("bilinear", bilin_pad_fn, _make_pad_mask, _make_nopad_bilinear),
            ("linear", linear_pad_fn, _make_pad_mask, _make_nopad_linear),
            ("mass", mass_pad_fn, _make_pad_mask, _make_nopad_mass),
            ("residual", res_pad_fn, _make_pad_mask, _make_nopad_residual),
            ("jacobian", jac_pad_fn, _make_pad_mask, _make_nopad_jacobian),
        ]:
            _run_sweep_case(
                label,
                seq,
                pad_fn=pad_fn,
                make_pad_args=make_pad,
                make_nopad_fn=make_nopad,
                verbose=args.sweep_verbose,
            )
        return 0

    for nx in sizes:
        mesh = ff.StructuredHexBox(nx=nx, ny=nx, nz=nx, lx=1.0, ly=1.0, lz=1.0).build()
        space = ff.make_hex_space(mesh, dim=1, intorder=args.intorder)
        u0 = jnp.zeros(space.n_dofs, dtype=jnp.float64)
        kappa = 1.0
        bilin_ker = ff.make_element_bilinear_kernel(ff.diffusion_form, kappa, jit=args.jit)
        linear_ker = jax.jit(linear_kernel) if args.jit else linear_kernel
        res_ker = ff.make_element_residual_kernel(linear_residual, kappa)
        jac_ker = ff.make_element_jacobian_kernel(linear_residual, kappa)

        if args.skip_mixed:
            mixed = None
            mixed_u = None
            mixed_params = None
        else:
            mixed = MixedFESpace({"u": space, "p": space})
            mixed_u = jnp.zeros(mixed.n_dofs, dtype=jnp.float64)
            mixed_params = ff.Params(alpha=1.2, beta=-0.4)

        n_elems = int(space.elem_dofs.shape[0])
        print(f"\n[nx={nx}] n_elems={n_elems}", flush=True)

        for mode in modes:
            n_chunks = args.n_chunks if mode == "pad" else None
            mode_label = f"mode={mode} n_chunks={n_chunks}"
            print(f"--- {mode_label}", flush=True)

            def _jit(fn):
                return jax.jit(fn) if args.jit else fn

            if args.pad_trace and n_chunks is not None:
                stats = ff.chunk_pad_stats(int(space.elem_dofs.shape[0]), int(n_chunks))
                print(
                    f"[pad] n_chunks={n_chunks} chunk_size={stats['chunk_size']} "
                    f"pad={stats['pad']} pad_ratio={stats['pad_ratio']:.4f}",
                    flush=True,
                )

            bilinear_fn = _jit(
                lambda: space.assemble(
                    ff.diffusion_form,
                    kappa,
                    n_chunks=n_chunks,
                    pad_trace=args.pad_trace,
                )
            )
            linear_fn = _jit(
                lambda: space.assemble(
                    ff.scalar_body_force_form,
                    1.0,
                    n_chunks=n_chunks,
                    pad_trace=args.pad_trace,
                )
            )
            mass_fn = _jit(lambda: space.assemble_mass_matrix(n_chunks=n_chunks, pad_trace=args.pad_trace))
            residual_fn = _jit(
                lambda u=u0: space.assemble_residual(
                    linear_residual,
                    u,
                    kappa,
                    kernel=res_ker,
                    n_chunks=n_chunks,
                    pad_trace=args.pad_trace,
                )
            )
            jac_fn = _jit(
                lambda u=u0: space.assemble_jacobian(
                    linear_residual,
                    u,
                    kappa,
                    kernel=jac_ker,
                    n_chunks=n_chunks,
                    sparse=False,
                    pad_trace=args.pad_trace,
                )
            )

            _bench_case("bilinear", bilinear_fn, warmup=args.warmup, repeat=args.repeat)
            _bench_case("linear", linear_fn, warmup=args.warmup, repeat=args.repeat)
            _bench_case("mass", mass_fn, warmup=args.warmup, repeat=args.repeat)
            _bench_case("residual", residual_fn, warmup=args.warmup, repeat=args.repeat)
            _bench_case("jacobian", jac_fn, warmup=args.warmup, repeat=args.repeat)

            if mixed is not None:
                mixed_problem = ff.MixedProblem(
                    mixed,
                    ff.make_mixed_residuals(mixed_residuals()),
                    params=mixed_params,
                    n_chunks=n_chunks,
                    pad_trace=args.pad_trace,
                )
                mixed_res_fn = _jit(lambda u=mixed_u: mixed_problem.assemble_residual(u))
                mixed_jac_fn = _jit(lambda u=mixed_u: mixed_problem.assemble_jacobian(u, sparse=False))
                _bench_case("mixed_residual", mixed_res_fn, warmup=args.warmup, repeat=args.repeat)
                _bench_case("mixed_jacobian", mixed_jac_fn, warmup=args.warmup, repeat=args.repeat)

    print("\nTips:", flush=True)
    print("- Use JAX_LOG_COMPILES=1 to see recompiles.", flush=True)
    print("- Try larger sizes or different --n-chunks to stress tail batches.", flush=True)
    print("- Set JAX_PLATFORM_NAME=cpu to suppress CUDA plugin warnings on CPU-only setups.", flush=True)
    print("- Use --mode sweep to stress recompile behavior with varying element counts.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

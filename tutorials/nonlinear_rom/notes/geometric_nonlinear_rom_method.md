# Geometric Nonlinear Full FEM vs ROM Method

This note documents the calculation used by
`tutorials/nonlinear_rom/compare_geometric_nonlinear_full_vs_rom.py`.

## Problem

The tutorial solves a small 3D cantilever-like solid with a geometric nonlinear
Neo-Hookean residual. The left face is fixed by Dirichlet constraints and a
small vertical load is applied to the upper free corner node.

The full-order unknown is the displacement vector

```text
u in R^n.
```

The full nonlinear equilibrium equation is

```text
R(u) = f_int(u) - f_ext = 0.
```

The full solve uses the existing FluxFEM nonlinear load-step/Newton path. At
each Newton iteration, the tangent is assembled and the linearized correction is
solved on the free DOFs.

## Direct Galerkin ROM

The ROM uses a dense basis

```text
Phi in R^(n x r),  q in R^r,  u = Phi q.
```

The reduced nonlinear residual is formed by direct Galerkin projection:

```text
R_r(q) = Phi^T R(Phi q).
```

The reduced tangent used by Newton is

```text
K_r(q) = Phi^T K(Phi q) Phi,
```

where `K(u) = dR/du`. The full residual and tangent are still evaluated through
the normal nonlinear FE assembly path. This means the tutorial is a correctness
and API demonstration, not a hyper-reduced large-scale nonlinear ROM benchmark.

## Compared Bases

`identity-full`
: Uses the full coordinate basis. This is a regression check: it should
  reproduce the full FEM solution up to numerical tolerance.

`free-dofs`
: Uses only unconstrained coordinates. This checks that fixed coordinates are
  handled correctly while keeping the full free subspace.

`linearized-modes`
: Assembles the initial linear elastic stiffness and mass matrices, removes
  fixed DOFs, and solves

```text
K_ff phi_i = lambda_i M_ff phi_i.
```

The lowest modes are lifted back to the full coordinate vector and used as a
low-dimensional nonlinear ROM basis.

`cantilever-bending-y`
: Uses one assumed bending shape in the loading direction. It is intentionally
  too small and is included to show visible projection error.

## Error Metrics

For each ROM solution `u_rom`, the tutorial compares against the full solution
`u_full`.

```text
e = u_full - u_rom
```

The JSON output stores:

```text
error_mean_abs     = mean(abs(e))
error_inf          = max(abs(e))
error_l2           = ||e||_2
relative_error_inf = max(abs(e)) / max(abs(u_full))
```

`summary_convergence.png` shows these quantities together with nonlinear solve
quality:

- relative displacement error,
- absolute displacement error, mean and max,
- final Newton residual infinity norm,
- Newton iteration count.

`summary_build_time.png` and `summary_solve_time.png` are kept separate so that
setup cost and nonlinear solve cost are not visually mixed with accuracy.

## Interpretation

A small reduced residual only means that the projected equation is solved in the
chosen subspace. It does not prove that the subspace is accurate. The
displacement error plots are therefore the primary check for whether the ROM
matches the full FEM solution.

For this tutorial, `identity-full` and `free-dofs` should match the full-order
solution. `linearized-modes` is expected to be a reasonable low-dimensional
approximation for the small deformation case. `cantilever-bending-y` is expected
to converge in its one-dimensional subspace but remain inaccurate in full-space
displacement error.

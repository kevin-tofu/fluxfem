Weak form Assembly
===================

FluxFEM provides a compact expression API for writing FEM weak forms as expression trees.
The goal is to let you write forms close to mathematics while keeping the assembly
kernels JAX-friendly. Expression trees are compiled into element kernels before
assembly.

This page explains the **recommended way to build weak forms**, with examples that
match the reference test suite.

.. contents::
   :local:
   :depth: 2


Core idea
---------

A weak form is written as a Python function returning an expression.
FluxFEM then compiles it into an element kernel and assembles it over a space:

.. code-block:: python

   import fluxfem as ff
   import fluxfem.helpers_wf as wf

   form = ff.BilinearForm.volume(lambda u, v, p: (u * v) * wf.dOmega())
   K = space.assemble_bilinear_form(form.get_compiled(), params=0.0)

**Important:** volume forms must include `dOmega()` and surface forms must include `ds()`.
The compiler checks this and raises an error otherwise.


Forms and signatures
--------------------

BilinearForm (volume)
^^^^^^^^^^^^^^^^^^^^^

Signature: `(u, v, params) -> Expr`

- `u` : trial field (shape-function data for the unknown)
- `v` : test field
- `params` : coefficient(s), can be a scalar/array or `ff.Params`

.. code-block:: python

   form = ff.BilinearForm.volume(lambda u, v, kappa: kappa * (v.grad @ u.grad) * wf.dOmega())


LinearForm (volume)
^^^^^^^^^^^^^^^^^^^

Signature: `(v, params) -> Expr`

.. code-block:: python

   form = ff.LinearForm.volume(lambda v, f: (v * f) * wf.dOmega())


LinearForm (surface)
^^^^^^^^^^^^^^^^^^^^

Signature: `(v, params) -> Expr` but evaluated on a `SurfaceFormContext`.

.. code-block:: python

   surface_form = ff.LinearForm.surface(lambda v, t: wf.dot(v, t) * wf.ds())


Parameters: scalar vs ff.Params
-------------------------------

If you pass a raw scalar/array, it is received as `params` directly:

.. code-block:: python

   form = ff.BilinearForm.volume(lambda u, v, kappa: kappa * (v.grad @ u.grad) * wf.dOmega())

If you need multiple coefficients, use `ff.Params` for attribute access:

.. code-block:: python

   p = ff.Params(rho=3.0, kappa=2.0)

   form = ff.BilinearForm.volume(
       lambda u, v, p: (p.rho * (u * v) + p.kappa * (v.grad @ u.grad)) * wf.dOmega()
   )


Volume measure: dOmega()
------------------------

`wf.dOmega()` represents the volume quadrature measure `w * detJ`.
All **volume** integrands must multiply by it.

.. code-block:: python

   a = (v.grad @ u.grad) * wf.dOmega()


Surface measure: ds()
---------------------

`wf.ds()` represents the surface quadrature measure `w * detJ` on facets.
All **surface** integrands must multiply by it.

.. code-block:: python

   g = wf.dot(v, traction) * wf.ds()


Mass matrix: `u * v` (scalar only)
------------------------------------

For scalar spaces (`dim=1`), `u * v` is defined and matches the assembled mass matrix:

.. code-block:: python

   form = ff.BilinearForm.volume(lambda u, v, _p: (u * v) * wf.dOmega())
   K_expr = space.assemble_bilinear_form(form.get_compiled(), params=0.0).to_dense()
   K_mass = space.assemble_mass_matrix().to_dense()

   assert np.allclose(np.asarray(K_expr), np.asarray(K_mass))

For vector-valued spaces (`dim>1`), `u * v` is **rejected** to avoid ambiguity:

.. code-block:: python

   space = ff.make_hex_space(mesh, dim=3, intorder=2)
   form = ff.BilinearForm.volume(lambda u, v, _p: (u * v) * wf.dOmega())

   with pytest.raises(ValueError, match="scalar fields"):
       space.assemble_bilinear_form(form.get_compiled(), params=0.0)

Use `dot(...)` or tensor contractions (`inner(...)` / `einsum(...)`) instead.


Diffusion / Poisson: `v.grad @ u.grad`
----------------------------------------

A diffusion form can be written in two equivalent styles:

**(1) Explicit style**

.. code-block:: python

   form = ff.BilinearForm.volume(
       lambda u, v, kappa: kappa * wf.dot(wf.grad(v), wf.grad(u)) * wf.dOmega()
   )

**(2) Operator style (recommended)**

.. code-block:: python

   form = ff.BilinearForm.volume(
       lambda u, v, p: p.kappa * (v.grad @ u.grad) * wf.dOmega()
   )

Both match `ff.diffusion_form` in the tests.

What does `v.grad @ u.grad` mean?

- `u.grad` / `v.grad` represent basis gradients at quadrature points.
- The `@` operator performs a **quadrature-wise contraction over the spatial dimension**,
  producing local stiffness contributions (conceptually `Σ_k ∂N_v/∂x_k * ∂N_u/∂x_k`).


Vector loads and dot(field, load)
---------------------------------

In mechanics operators, `dot` has a special dispatch:

- If the first argument is a FormField (has `N` and `value_dim`), `dot(field, load_vec)`
  builds the **vector linear form** contribution via `vector_load_form`.
- Otherwise, it falls back to a batched matrix product.

This is why, on surfaces, traction can be expressed naturally as a dot-like operation.


Linear elasticity: sym_grad and ddot (Voigt B-matrix)
-----------------------------------------------------

For small-strain 3D linear elasticity, use:

- `wf.sym_grad(u)` : symmetric gradient **Voigt B-matrix**
- `D` : constitutive matrix in Voigt notation (6x6)
- `wf.ddot(sym_grad(v), wf.matmul_std(D, wf.sym_grad(u)))` : contraction to form local stiffness blocks

Definition (as implemented):

- `sym_grad(field)` returns `B` with shape `(n_q, 6, n_dofs)`
  in Voigt order `[xx, yy, zz, xy, yz, zx]` such that

  `eps_voigt(q,:) = B(q,:,:) @ u_elem`

Example that matches `ff.linear_elasticity_form`:

.. code-block:: python

   D = ff.isotropic_3d_D(210_000.0, 0.3)

   form = ff.BilinearForm.volume(
       lambda u, v, D: wf.ddot(wf.sym_grad(v), wf.matmul_std(D, wf.sym_grad(u))) * wf.dOmega()
   )

   K_expr = space.assemble_bilinear_form(form.get_compiled(), params=D).to_dense()
   K_ref  = space.assemble_bilinear_form(ff.linear_elasticity_form, params=D).to_dense()
   assert np.allclose(np.asarray(K_expr), np.asarray(K_ref))

About `ddot`:

- `ddot(a, b)`:
  - If `a` and `b` are 2D arrays, it computes elementwise double contraction
    `Σ_ij a_ij b_ij` (last two axes), returning a scalar per quadrature point.
  - If `a` and `b` are 3D arrays shaped like `(q, i, k)` and `(q, i, m)`,
    it contracts over the middle axis and returns a local block `(q, k, m)`
    (used by `sym_grad` in elasticity).
- `ddot(a, b, c)` computes `aᵀ b c` (used for Voigt-style elasticity blocks).


Surface traction: `dot(v, t)` (and `v | t` shorthand)
----------------------------------------------------------

For surface linear forms, the expression API supports a convenient shorthand:

- `v | t` (with `t` a vector expression) is treated as `dot(v, t)`.

In the tests, this matches the tensor-based reference traction form:

.. code-block:: python

   surface_form = ff.LinearForm.surface(lambda v, t: wf.dot(v, t) * wf.ds())

   def traction_form(ctx: ff.SurfaceFormContext, t):
       return ff.dot(ctx.v, t)

Both assemblies match.

Normal traction (pressure)
--------------------------

If the traction is a **scalar pressure** acting along the outward normal, use
`wf.normal()` inside the weak form:

.. code-block:: python

   surface_form = ff.LinearForm.surface(
       lambda v, p: (v | (p * wf.normal())) * wf.ds()
   )


Recipes (from tests)
--------------------

Mass (scalar)
^^^^^^^^^^^^^

.. code-block:: python

   ff.BilinearForm.volume(lambda u, v, p: (u * v) * wf.dOmega())

Diffusion
^^^^^^^^^

.. code-block:: python

   ff.BilinearForm.volume(lambda u, v, p: p.kappa * (v.grad @ u.grad) * wf.dOmega())

Linear elasticity
^^^^^^^^^^^^^^^^^

.. code-block:: python

   ff.BilinearForm.volume(
       lambda u, v, D: wf.ddot(wf.sym_grad(v), wf.matmul_std(D, wf.sym_grad(u))) * wf.dOmega()
   )

Surface traction
^^^^^^^^^^^^^^^^

.. code-block:: python

   ff.LinearForm.surface(lambda v, t: wf.dot(v, t) * wf.ds())



About the `|` operator
^^^^^^^^^^^^^^^^^^^^^^^^

The `|` operator is intentionally limited:

- `FieldRef | FieldRef` is **not allowed**. Use `outer(test, trial)` for basis kernels.
- `FieldRef | Expr` is treated as `dot(v, expr)` (e.g., traction loads).
- `Expr | Expr` is a tensor inner product over the last axis (`inner`), e.g. `u.val | v.val`.

For surface traction, prefer `dot(v, t) * ds()` for clarity.

About the `@` operator
^^^^^^^^^^^^^^^^^^^^^^^^

The `@` operator is **FEM-specific** and only defined for batched 3D arrays shaped like
`(q, *, dim)`:

`A @ B` with `A.ndim==B.ndim==3` and matching last axis performs a contraction over `dim`
and returns `(q, *, *)` (conceptually `einsum("qia,qja->qij")`). In other words, for each
quadrature point `q`, the last axis is contracted to produce a `(i, j)` block.

This is used for expressions like `v.grad @ u.grad`.
If the operands are not in this 3D batched form, it raises a `TypeError`.

Use `wf.matmul_std(A, B)` for **standard** matrix multiplication
(no special 3D contraction).

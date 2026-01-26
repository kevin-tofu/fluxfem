Assembly
=========

FluxFEM provides two complementary assembly styles:

- **Tensor-based assembly**: write per-quadrature array integrands directly (scikit-fem style).
- **Weak-form-based assembly**: write expressions close to the mathematical weak form
  and let FluxFEM compile them into element kernels.


Both styles target the same assembly routines, so you can mix them in one project.

When to use which?

- Tensor-based: explicit data flow and shapes; good when you want full control,
  custom kernels, or to follow scikit-fem-like patterns.
- Weak-form-based: concise and expressive; good for rapid prototyping and for
  matching equations in papers.

Measure handling (weak form vs tensor vs kernel)
------------------------------------------------

All three assembly styles use the same ``space.assemble(...)`` entry point, but
they differ in **who owns the quadrature measure** (``w * detJ``). This is the
most common source of confusion, so the rules are summarized here:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Style
     - What your form returns
     - Measure rule
   * - Weak-form DSL
     - Expression for the integrand
     - **Must** multiply by ``dOmega()``/``ds()``
   * - Tensor-based form
     - Per-quadrature integrand arrays
     - **Must NOT** include ``dOmega()``/``ds()``
   * - Element kernel (JIT)
     - Integrated element vector/matrix
     - **Must** already include the measure

Quick examples:

.. code-block:: python

   import fluxfem.helpers_wf as wf

   # weak form: include dOmega()
   form = ff.BilinearForm.volume(lambda u, v, p: (v.grad @ u.grad) * p.kappa * wf.dOmega())

.. code-block:: python

   # tensor form: integrand only (no dOmega)
   def diffusion_form(ctx, kappa):
       return kappa * jnp.einsum("qia,qja->qij", ctx.test.gradN, ctx.trial.gradN)

.. code-block:: python

   # element kernel: already integrated over quadrature
   def linear_kernel(ctx):
       integrand = ff.scalar_body_force_form(ctx, 2.0)
       wJ = ctx.w * ctx.test.detJ
       return (integrand * wJ[:, None]).sum(axis=0)


.. toctree::
   :maxdepth: 1
   :caption: Assembly

   form_context
   assembly_tensor
   assembly_weakform
   

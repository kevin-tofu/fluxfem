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

Single-Space vs Role-Explicit Assembly
--------------------------------------

FluxFEM supports two complementary calling styles for the same core assembly
machinery:

- ``space.assemble_*``: the shortest path when test/trial/unknown all live on
  the same space.
- ``ff.assemble_*(*Spaces(...), ...)``: the explicit path when you want to bind
  named roles such as ``test`` and ``trial`` directly in the public API.

Deprecated compatibility paths are documented in
``migration_role_spaces.rst``. The preferred assembly surface shown here uses
the ``*Spaces`` family directly.

For standard Galerkin problems, the single-space form remains the simplest:

.. code-block:: python

   import fluxfem as ff

   K = space.assemble_bilinear_form(ff.diffusion_form, params=1.0)
   F = space.assemble_linear_form(ff.make_scalar_body_force_form(lambda x: 1.0), params=None)

The paired helper ``space.assemble_bilinear_linear_pair(...)`` remains
supported for the common same-space case, but it is only convenience sugar over
the separate bilinear and linear assembly calls. There is no separate
role-explicit ``PairSpaces`` abstraction; the underlying public building blocks
are still ``BilinearSpaces`` and ``LinearSpaces``.

For role-explicit assembly, use the ``*Spaces`` family:

.. code-block:: python

   import fluxfem as ff
   import fluxfem.helpers_wf as wf

   U = ff.NamedSpace("U", trial_space)
   V = ff.NamedSpace("V", test_space)

   bilinear = ff.BilinearForm.volume(
       lambda u, v, p: p.kappa * wf.dot(wf.grad(v), wf.grad(u)) * wf.dOmega()
   )
   linear = ff.LinearForm.volume(
       lambda v, p: wf.dot(v, p.force) * wf.dOmega()
   )

   A = ff.assemble_bilinear_form(
       ff.BilinearSpaces(test=V, trial=U),
       bilinear,
       ff.Params(kappa=1.0),
   )
   b = ff.assemble_linear_form(
       ff.LinearSpaces(test=V),
       linear,
       ff.Params(force=1.0),
   )

The same pattern extends to nonlinear volume assembly:

.. code-block:: python

   residual = ff.ResidualForm.volume(
       lambda v, u, p: p.kappa * wf.gaction(v, wf.grad(u)) * wf.dOmega()
   )

   R = ff.assemble_residual(
       ff.ResidualSpaces(test=V, unknown=U),
       residual,
       u_vec,
       ff.Params(kappa=1.0),
   )
   J = ff.assemble_jacobian(
       ff.JacobianSpaces(test=V, trial=U),
       residual,
       u_vec,
       ff.Params(kappa=1.0),
   )

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
   

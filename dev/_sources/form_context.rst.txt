FormContext and Expr Resolution
================================

FluxFEM assembly operates on **element-local data** stored in a ``FormContext``
(or ``SurfaceFormContext``). The context provides per-element shape functions,
gradients, quadrature weights, and geometry. Both assembly styles use the same
context, but they differ in how you write the integrand.

What lives in a FormContext
---------------------------

Typical fields used by assembly:

- ``ctx.test.N`` / ``ctx.trial.N``: shape-function values ``(n_q, n_nodes)``
- ``ctx.test.gradN`` / ``ctx.trial.gradN``: gradients ``(n_q, n_nodes, dim)``
- ``ctx.x_q``: quadrature points in physical coordinates
- ``ctx.w``: quadrature weights
- ``ctx.detJ``: Jacobian determinant (surface contexts may expose ``ctx.normal``)

Tensor vs weak-form resolution
------------------------------

**Tensor assembly** reads these arrays directly:

.. code-block:: python

   def diffusion_form(ctx: ff.FormContext, kappa):
       return kappa * ff.jnp.einsum("qia,qja->qij", ctx.test.gradN, ctx.trial.gradN)

**Weak-form assembly** builds an ``Expr`` tree, then resolves it against the
context and params during compilation/evaluation:

.. code-block:: python

   form = ff.BilinearForm.volume(
       lambda u, v, p: p.kappa * (v.grad @ u.grad) * wf.dOmega()
   )

   compiled = form.get_compiled()
   K = space.assemble_bilinear_form(compiled, params=ff.Params(kappa=1.0))

In the weak-form path, symbolic refs like ``u`` and ``v`` are resolved to the
corresponding context fields (``ctx.trial`` / ``ctx.test``), and measure terms
like ``dOmega()``/``ds()`` inject quadrature weights.

Why use Expr (benefits and drawbacks)
-------------------------------------

Expr-based weak forms are a **high-level way to build element kernels**.
They can make nonlinear models readable while keeping the assembly interface the same.

Benefits
^^^^^^^^

- **Concise weak forms**: you write expressions close to mathematics (e.g., ``v.grad @ u.grad``).
- **Single source of truth**: the same Expr tree can be reused for residuals, Jacobians,
  and consistency checks (measure validation, shape constraints).
- **Less boilerplate**: FormContext access is automatic through symbolic refs
  (``test_ref()``, ``trial_ref()``, ``param_ref()``).
- **Fewer ad-hoc kernels**: nonlinear models like Neo-Hookean can be expressed
  without hand-written B-matrix assembly.

Drawbacks
^^^^^^^^^

- **Python overhead**: expression evaluation still walks a Python-level tree
  (mitigated by compilation and caching).
- **Less explicit shapes**: tensor-based forms expose shapes directly, which can
  be easier to debug for complex models.
- **Operator coverage**: very specialized models may still need custom tensor
  kernels or new Expr operators.

Example: Neo-Hookean in Expr form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following weak-form residual matches the tensor-based reference used in the tests:

.. code-block:: python

   def neo_hookean_wf(v, u, p):
       F = wf.I(3) + u.grad
       C = wf.ddot(F, F)
       C_inv = wf.inv(C)
       logJ = wf.log(wf.det(F))
       S = p.mu * (wf.I(3) - C_inv) + p.lam * logJ * C_inv
       P = wf.matmul_std(F, S)
       return wf.gaction(v, P) * wf.dOmega()


Comparison with scikit-fem
--------------------------

scikit-fem is closer to **tensor-based assembly**: you work directly with
``ctx`` arrays and return per-quadrature integrands. FluxFEM's Expr-based
weak forms sit one level higher.

Advantages vs scikit-fem style
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Weak-form readability**: expressions look closer to the mathematical form.
- **Uniform compile pipeline**: forms are validated and compiled once.
- **Less boilerplate**: common patterns (measures, refs) are encoded in the DSL.

Tradeoffs vs scikit-fem style
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **More abstraction**: tensor shapes are less explicit during debugging.
- **Operator limits**: very specialized kernels may still be clearer in tensor form.

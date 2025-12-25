Assembly
=========

FluxFEM provides two complementary assembly styles:

- **Weak-form-based assembly**: write expressions close to the mathematical weak form
  and let FluxFEM compile them into element kernels.
- **Tensor-based assembly**: write per-quadrature array integrands directly (scikit-fem style).

Both styles target the same assembly routines, so you can mix them in one project.

When to use which?

- Weak-form-based: concise and expressive; good for rapid prototyping and for
  matching equations in papers.
- Tensor-based: explicit data flow and shapes; good when you want full control,
  custom kernels, or to follow scikit-fem-like patterns.

.. toctree::
   :maxdepth: 1
   :caption: Assembly

   assembly_weakform
   assembly_tensor

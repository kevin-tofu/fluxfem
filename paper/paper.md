---
title: "FluxFEM: A weak-form-centric differentiable finite element framework in JAX"
tags:
  - finite element method
  - automatic differentiation
  - JAX
  - computational mechanics
  - weak form
authors:
  - name: Kohei Watanabe
    orcid: 0009-0008-2278-6418
    affiliation: 1
affiliations:
 - name: JTEKT Corporation, Kariya, Japan
   index: 1
date: 28 Dec 2025
bibliography: paper.bib
---

## Summary

FluxFEM is a Python-based finite element method (FEM) framework built on top of JAX. The framework is designed so that the use of JAX does not preclude users from writing weak forms directly and explicitly. Maintaining a close correspondence between weak forms in code and their mathematical formulation is important for readability and productivity, and FluxFEM seeks to support this style while making use of automatic differentiation and just-in-time (JIT) compilation where applicable. In FluxFEM, the weak form of a partial differential equation (PDE) is treated as a **first-class residual operator**, with fields and coefficients exposed explicitly. These components are represented using PyTree-compatible data structures, allowing the weak form to integrate naturally with JAX transformations such as `jit`, `vmap`, and `grad`. This design supports not only linear problems but also **nonlinear problems solved via Newton-type methods** within the same unified framework. By preserving the expressiveness of weak forms while clarifying the input–output relationships between weak forms, residuals, and objective functions, FluxFEM enables PDE solvers to be embedded as differentiable computational components in larger optimization workflows.

---

## Statement of Need

Optimization and inverse problems in computational mechanics require PDE solvers to be treated not as black boxes, but as **computational components with clearly defined inputs and outputs**. However, in many FEM implementations, weak forms, geometry, and boundary conditions are implicitly embedded inside solver logic, which can complicate integration with automatic differentiation and end-to-end optimization workflows.

**scikit-fem** [@skfem] provides a stable and widely used Python environment for writing weak forms concisely and explicitly, and it has proven effective for forward simulations and many research applications. Its design, however, does not primarily target automatic differentiation or workflows in which weak forms or geometry are treated as optimization variables, and additional effort is required to integrate such use cases.

**JAX-FEM** [@xue2023jax] achieves acceleration and automatic differentiation using JAX, but to ensure JIT stability, it adopts a design in which mesh and function space objects are kept outside JAX’s tracing boundary. While this approach is effective in practice, applications in which geometry or weak forms themselves are treated as optimization variables require additional considerations or specialized design choices.

FluxFEM is designed to bridge this gap by providing an API in which **weak forms are explicitly defined as residual operators**. A PyTree-centric design allows fields and coefficients to be handled uniformly, minimizing special-case logic on the FEM side. As a result, the definition of weak forms connects naturally to automatic differentiation and
optimization logic.

---

## Purpose and Prior Art

Widely used finite element frameworks such as FEniCS [@fenics] and Firedrake [@firedrake] provide high-level domain-specific languages for defining weak forms and generating efficient forward solvers. These frameworks are highly mature and robust, and are primarily designed for large-scale forward simulations in scientific and engineering applications. In contrast, FluxFEM represents weak forms as differentiable residual operators with explicit inputs and outputs. This formulation enables weak forms to be treated as computational components that integrate naturally with automatic differentiation and optimization workflows. The design of FluxFEM is also informed by weak-form-centric FEM frameworks. In particular, scikit-fem enables concise and explicit expression of weak forms directly in Python, while Gridap [@Verdugo2022] emphasizes treating variational formulations as first-class objects rather than focusing on low-level matrix assembly. FluxFEM builds on these ideas by formulating weak forms as differentiable residual mappings compatible with JAX’s program transformation mechanisms. One of the fundamental strengths of the finite element method is its ability to express problems in terms of weak forms that closely reflect their mathematical formulation. FluxFEM adopts the explicit stance that using JAX should not require abandoning this weak-form expressiveness. By representing weak forms as residual mappings with explicit inputs and outputs, residual evaluation, differentiation, and composition can be combined directly with standard JAX transformations. To support this design, FluxFEM adopts a PyTree-based data model that integrates fields and coefficients in a form natural to JAX, while keeping residual evaluation and differentiation logic simple and composable. This approach is intended to support future extensions, such as geometry-dependent coefficients or advanced formulations, without requiring substantial changes to the solver structure. At present, FluxFEM includes several example cases—for example, linear elasticity, nonlinear elasticity (Neo-Hookean), diffusion/Poisson, and inverse problems—demonstrating the practical feasibility of combining weak-form expressiveness with differentiable programming.


### Example: Diffusion bilinear form (direct kernel vs Expr-based form)

This example illustrates two complementary ways to express the same weak form in FluxFEM, highlighting the distinction between low-level element kernels and higher-level Expr-based weak-form representations. FluxFEM supports a "direct kernel" style, in which the element contribution is written explicitly using quantities provided by the form context, such as basis gradients and quadrature data. This provides maximum control over element-level computations and can be useful for performance tuning or implementing specialized operators. For a diffusion-type bilinear form, this can be written as:


```python
import fluxfem as ff
import jax.numpy as jnp

@ff.kernel(kind="bilinear", domain="volume")
def diffusion_form(ctx: ff.FormContext, kappa):
    return kappa * jnp.einsum("qia,qja->qij", ctx.test.gradN, ctx.trial.gradN)
```

Alternatively, the same weak form can be expressed using the Expr-based assembly interface, which represents the weak form symbolically and resolves it against the evaluation context (geometry, basis functions, and quadrature data) at compile time.

```python
import fluxfem.helpers_wf as h_wf

diffusion_form_wf = ff.BilinearForm.volume(
    lambda u, v, p: p.kappa * (v.grad @ u.grad) * h_wf.dOmega()
).get_compiled()

params = ff.Params(kappa=1.0)
# K = space.assemble(diffusion_form, params=params)
K = space.assemble(diffusion_form_wf, params=params)
```

Both snippets assemble the same diffusion stiffness operator, but the Expr-based formulation separates the symbolic structure of the weak form from runtime inputs, performs basic structural checks during compilation (for example, requiring a single `dOmega()`), and reduces the need to work with `jnp` directly in weak-form definitions. This separation also enables composition, reuse, and differentiation of residual operators using standard JAX transformations.

---

## Acknowledgements

I acknowledge the use of open-source tools and libraries that made this research possible.

---

## References

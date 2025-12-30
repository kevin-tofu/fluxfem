[![PyPI version](https://img.shields.io/pypi/v/fluxfem.svg?cacheSeconds=60)](https://pypi.org/project/fluxfem/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/fluxfem.svg)](https://pypi.org/project/fluxfem/)
![CI](https://github.com/kevin-tofu/fluxfem/actions/workflows/python-tests.yml/badge.svg)
![CI](https://github.com/kevin-tofu/fluxfem/actions/workflows/sphinx.yml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18055465.svg)](https://doi.org/10.5281/zenodo.18055465)


# FluxFEM
A weak-form-centric differentiable finite element framework in JAX,
where variational forms are treated as first-class, differentiable programs.

## Examples and Features
<table>
  <tr>
    <td align="center"><b>Example 1: Diffusion</b></td>
    <td align="center"><b>Example 2: Neo Neohookean Hyper Elasticity</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://media.githubusercontent.com/media/kevin-tofu/fluxfem/main/assets/diffusion_mms_timeseries.gif" alt="Diffusion-mms" width="400">
    </td>
    <td align="center">
      <img src="https://media.githubusercontent.com/media/kevin-tofu/fluxfem/main/assets/Neo-Hookean-deformedx20000.png" alt="Neo-Hookean" width="400">
    </td>
  </tr>
</table>


## Features
- Built on JAX, enabling automatic differentiation with grad, jit, vmap, and related transformations.
- Weak-form–centric API that keeps formulations close to code; weak forms are represented as expression trees and compiled into element kernels, enabling automatic differentiation of residuals, tangents, and objectives.
- Two assembly approaches: tensor-based (scikit-fem–style) assembly and weak-form-based assembly.
- Handles both linear and nonlinear analyses with AD in JAX.

## Usage 

This library provides two assembly approaches.

- A tensor-based assembly, where trial and test functions are represented explicitly as element-level tensors and assembled accordingly (in the style of scikit-fem).
- A weak-form-based assembly, where the variational form is written symbolically and compiled before assembly.

The two approaches are functionally equivalent and share the same element-level execution model,
but they differ in how you author the weak form. The example below mirrors the paper's diffusion
case and makes the distinction explicit with `jnp`.


## Assembly Flow
All expressions are first compiled into an element-level evaluation plan,
which operates on quadrature-point–major tensors.
This plan is then executed independently for each element during assembly.

As a result, both assembly approaches:
- use the same quadrature-major (q, a, i) data layout,
- perform element-local tensor contractions,
- and are fully compatible with JAX transformations such as `jit`, `vmap`, and automatic differentiation.

### tensor-based vs weak-form-based (diffusion example)

#### tensor-based assembly
The tensor-based assembly provides an explicit, low-level formulation with element kernels written using jax.numpy.(`jnp`).
```Python
import fluxfem as ff
import jax.numpy as jnp

def diffusion_form(ctx: ff.FormContext, kappa):
    # ctx.test.gradN / ctx.trial.gradN: (n_qp, n_nodes, dim)
    # output tensor: (n_qp, n_nodes, n_nodes)
    return kappa * jnp.einsum("qia,qja->qij", ctx.test.gradN, ctx.trial.gradN)

space = ff.make_hex_space(mesh, dim=3, intorder=2)
params = ff.Params(kappa=1.0)
K_ts = space.assemble_bilinear_form(diffusion_form, params=params.kappa)
```

#### weak-form-based assembly
In the weak-form-based assembly, the variational formulation itself is the primary object.
The expression below defines a symbolic computation graph, which is later compiled and executed at the element level.

```Python
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

space = ff.make_hex_space(mesh, dim=3, intorder=2)
params = ff.Params(kappa=1.0)

# u, v are symbolic trial/test fields (weak-form DSL objects).
# u.grad / v.grad are symbolic nodes (expression tree), not numeric arrays.
# dOmega() is the integral measure; the whole expression is compiled before assembly.
form_wf = ff.BilinearForm.volume(
    lambda u, v, p: p.kappa * (v.grad @ u.grad) * h_wf.dOmega()
).get_compiled()

K_wf = space.assemble_bilinear_form(form_wf, params=params)
```

### Linear Elasticity assembly (weak-form based assembly)

```Python
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

space = ff.make_hex_space(mesh, dim=3, intorder=2)
D = ff.isotropic_3d_D(1.0, 0.3)

form_wf = ff.BilinearForm.volume(
    lambda u, v, D: h_wf.ddot(v.sym_grad, D @ u.sym_grad) * h_wf.dOmega()
).get_compiled()

K = space.assemble_bilinear_form(form_wf, params=D)
```

### Neo-Hookean residual assembly (weak-form DSL)
Below is a Neo-Hookean hyperelasticity example written in weak form.
The residual is expressed symbolically and compiled into element-level kernels executed per element.
No manual derivation of tangent operators is required; consistent tangents (Jacobians) for Newton-type solvers are obtained automatically via JAX AD.

```Python
def neo_hookean_residual_wf(v, u, params):
    mu = params["mu"]
    lam = params["lam"]
    F = h_wf.I(3) + h_wf.grad(u)  # deformation gradient
    C = h_wf.matmul(h_wf.transpose(F), F)
    C_inv = h_wf.inv(C)
    J = h_wf.det(F)

    S = mu * (h_wf.I(3) - C_inv) + lam * h_wf.log(J) * C_inv
    dE = 0.5 * (h_wf.matmul(h_wf.grad(v), F) + h_wf.transpose(h_wf.matmul(h_wf.grad(v), F)))
    return h_wf.ddot(S, dE) * h_wf.dOmega()
```


### autodiff + jit compile

You can differentiate through the solve and JIT compile the hot path.
The inverse diffusion tutorial shows this pattern:

```Python
def loss_theta(theta):
    kappa = jnp.exp(theta)
    u = solve_u_jit(kappa, traction_true)
    diff = u[obs_idx_j] - u_obs[obs_idx_j]
    return 0.5 * jnp.mean(diff * diff)

solve_u_jit = jax.jit(solve_u)
loss_theta_jit = jax.jit(loss_theta)
grad_fn = jax.jit(jax.grad(loss_theta))
```

## Documentation

Full documentation, tutorials, and API reference are hosted at [this site](https://fluxfem.readthedocs.io/en/latest/).

## SetUp

You can install **FluxFEM** either via **pip** or **Poetry**.

#### Supported Python Versions

FluxFEM supports **Python 3.11–3.13**:


**Choose one of the following methods:**

### Using pip
```bash
pip install fluxfem
```

### Using poetry
```bash
poetry add fluxfem
```

## Acknowledgements
I acknowledge the open-source software, libraries, and communities that made this work possible.

[![PyPI version](https://img.shields.io/pypi/v/fluxfem.svg?cacheSeconds=60)](https://pypi.org/project/fluxfem/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/fluxfem.svg)](https://pypi.org/project/fluxfem/)
![CI](https://github.com/kevin-tofu/fluxfem/actions/workflows/python-tests.yml/badge.svg)
![CI](https://github.com/kevin-tofu/fluxfem/actions/workflows/sphinx.yml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18055465.svg)](https://doi.org/10.5281/zenodo.18055465)


# FluxFEM
 A weak-form-centric differentiable finite element framework in JAX

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
- Weak-form–centric API that keeps formulations close to code; weak forms are represented as expression trees and compiled to element kernels.
- Two assembly approaches: weak-form-based assembly and a tensor-based (scikit-fem–style) assembly.
- Handles both linear and nonlinear analyses with AD in JAX.

## Usage 

This library provides two assembly approaches.

- A weak-form-based assembly, where the variational form is written and assembled directly.  
- A tensor-based assembly, where trial and test functions are represented explicitly as tensors and assembled accordingly (in the style of scikit-fem).  
The first approach offers simplicity and convenience, as mathematical expressions can be written almost directly in code.
However, for more complex operations, the second approach can be easier to implement in practice.
This is because the weak-form-based assembly is ultimately transformed into the tensor-based representation internally during computation.

## Weak Form Compile Flow
Weak-form expressions are compiled into an evaluation plan and then executed per element.

### weak-form-based assembly
```Python
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

space = ff.make_hex_space(mesh, dim=3, intorder=2)
D = ff.isotropic_3d_D(1.0, 0.3)
bilinear_form = ff.BilinearForm.volume(
    lambda u, v, D: h_wf.ddot(v.sym_grad, h_wf.matmul_std(D, u.sym_grad)) * h_wf.dOmega()
)
K_wf = space.assemble_bilinear_form(
    bilinear_form.get_compiled(),
    params=D,
)
```

### tensor-based assembly (scikit-fem-style)

```Python
import fluxfem as ff
import numpy as np
import fluxfem.helpers_ts as h_ts

def linear_elasticity_form(ctx: ff.FormContext, D: np.ndarray) -> ff.jnp.ndarray:
        Bu = h_ts.sym_grad(ctx.trial)
        Bv = h_ts.sym_grad(ctx.test)
        return h_ts.ddot(Bv, D, Bu)


space = ff.make_hex_space(mesh, dim=3, intorder=2)
D = ff.isotropic_3d_D(1.0, 0.3)
K = space.assemble_bilinear_form(linear_elasticity_form, params=D)
```

## Documentation


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
 I acknoldege everythings that made this work possible.

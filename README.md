[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/fluxfem.svg)](https://pypi.org/project/fluxfem/)
![CI](https://github.com/kevin-tofu/fluxfem/actions/workflows/python-tests.yml/badge.svg)
![CI](https://github.com/kevin-tofu/fluxfem/actions/workflows/sphinx.yml/badge.svg)

# FluxFEM
 A weak-form-centric differentiable finite element framework in JAX

## Examples and Features
### Example 1 : Diffusion
<p align="center">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/fluxfem/main/assets/diffusion_mms_timeseries.gif" alt="Optimization Process Pull-Down-0" width="400" style="margin-right: 20px;">
</p>

## Features
- Built on JAX, enabling automatic differentiation and high-performance execution via grad, jit, vmap, and related transformations.

- A FEM framework with a weak-form–centric API, emphasizing a smooth transition from theoretical formulations to practical code implementations.

- Supports two assembly approaches: weak-form-based assembly and a tensor-based (scikit-fem–style) assembly.

- enables to handle both Linear / Non-Linear Analysis with AD with JAX

## Usage 

This library provides two assembly approaches.

- A weak-form-based assembly, where the variational form is written and assembled directly.  
- A tensor-based assembly, where trial and test functions are represented explicitly as tensors and assembled accordingly (in the style of scikit-fem).  
The first approach offers simplicity and convenience, as mathematical expressions can be written almost directly in code.
However, for more complex operations, the second approach can be easier to implement in practice.
This is because the weak-form-based assembly is ultimately transformed into the tensor-based representation internally during computation.

### weak-form-based assembly
```Python
import fluxfem as ff

space = ff.make_hex_space(mesh, dim=3, intorder=2)
D = ff.isotropic_3d_D(1.0, 0.3)
bilinear_form = ff.BilinearForm.volume(
    lambda u, v, D: h_wf.ddot(v.sym_grad, D @ u.sym_grad) * h_wf.dOmega()
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

def linear_elasticity_form(ctx: ff.FormContext, D: np.ndarray) -> ff.jnp.ndarray:
        Bu = h_num.sym_grad(ctx.trial)
        Bv = h_num.sym_grad(ctx.test)
        return h_num.ddot(Bv, D, Bu)


space = ff.make_hex_space(mesh, dim=3, intorder=2)
D = ff.isotropic_3d_D(1.0, 0.3)
K = space.assemble_bilinear_form(linear_elasticity_form, params=D)
```

## Documentation


## SetUp

You can install **Scikit-Topt** either via **pip** or **Poetry**.

#### Supported Python Versions

Scikit-Topt supports **Python 3.10–3.13**:


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
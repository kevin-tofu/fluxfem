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

```Python
bilinear_form = ff.BilinearForm.volume(
    lambda u, v, D: h_wf.ddot(v.sym_grad, D @ u.sym_grad) * h_wf.dOmega()
)
K_wf = space.assemble_bilinear_form(
    bilinear_form.compile(),
    params=D,
)
```
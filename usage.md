# Usage (New assemble API)

This note shows the updated `assemble` workflow where the form kind is inferred
from `BilinearForm`/`LinearForm` or compiled kernels carrying metadata.

## 1) Weak-form classes

```python
import fluxfem as ff
import fluxfem.helpers_wf as wf

form = ff.BilinearForm.volume(lambda u, v, p: (v.grad @ u.grad) * p.kappa * wf.dOmega())

# `kind` is inferred from BilinearForm
K = space.assemble(form, params=ff.Params(kappa=1.0))
```

```python
import fluxfem.helpers_wf as wf

form = ff.LinearForm.volume(lambda v, p: (v * p.f) * wf.dOmega())

# `kind` is inferred from LinearForm
F = space.assemble(form, params=ff.Params(f=2.0))
```

## 2) Compiled weak forms

```python
import fluxfem.helpers_wf as wf

form = ff.BilinearForm.volume(lambda u, v, p: (v.grad @ u.grad) * p.kappa * wf.dOmega())
compiled = form.get_compiled()

# `kind` is inferred from compiled metadata
K = space.assemble(compiled, params=ff.Params(kappa=1.0))
```

```python
import fluxfem.helpers_wf as wf

form = ff.LinearForm.surface(lambda v, p: (v | p.t) * wf.ds())
compiled = form.get_compiled()

# Surface compiled forms are rejected by Space.assemble (volume only).
# Use a surface-specific assembly API instead:
# F = surface.assemble_linear_form_on_space(space, compiled, params=ff.Params(t=traction))
```

## 3) Raw kernels (tagged by metadata)

Built-in kernels in `fluxfem.physics` are tagged with `_ff_kind`/`_ff_domain`
so `kind` can be inferred.

```python
import fluxfem as ff

# diffusion_form is tagged as bilinear/volume
K = space.assemble(ff.diffusion_form, params=1.0)

# vector_body_force_form is tagged as linear/volume
F = space.assemble(ff.vector_body_force_form, params=load_vec)
```

If you define your own kernel, add tags once and you can omit `kind` as well.

```python
import jax.numpy as jnp

@ff.kernel(kind="bilinear", domain="volume")
def diffusion_form(ctx: ff.FormContext, kappa):
    return kappa * jnp.einsum(
        "qia,qja->qij",
        ctx.test.gradN,
        ctx.trial.gradN,
    )

K = space.assemble(diffusion_form, params=1.0)
```

## 4) Explicit kind (still supported)

You can still force the kind explicitly. If it conflicts with the metadata,
`assemble` raises a ValueError.

```python
K = space.assemble(ff.diffusion_form, params=1.0, kind="bilinear")
```

## 5) Custom kernels without metadata

If you pass a plain callable without `_ff_kind`, you must set `kind` explicitly,
or add metadata yourself.

```python
# Option A: pass kind explicitly
K = space.assemble(my_kernel, params=1.0, kind="bilinear")

# Option B: tag the kernel
my_kernel = ff.kernel(kind="bilinear", domain="volume")(my_kernel)
K = space.assemble(my_kernel, params=1.0)
```

Note: passing an untagged kernel with `kind=` emits a one-time warning to
encourage tagging. Suppress it with:

```python
import warnings

warnings.filterwarnings(
    "ignore",
    message="Raw kernel has no _ff_kind metadata",
    category=UserWarning,
)
```

## 6) `*Spaces` family

For new code, prefer explicit role specs over ad-hoc dictionaries.

### Linear form

```python
V = ff.NamedSpace("V", space)

form = ff.LinearForm.volume(lambda v, p: (v * p.f) * wf.dOmega())
F = ff.assemble_linear_form(
    ff.LinearSpaces(test=V),
    form.get_compiled(),
    params=ff.Params(f=2.0),
)
```

### Bilinear form

```python
U = ff.NamedSpace("U", trial_space)
V = ff.NamedSpace("V", test_space)

form = ff.BilinearForm.volume(
    lambda u, v, p: p.kappa * wf.dot(wf.grad(v), wf.grad(u)) * wf.dOmega()
)
A = ff.assemble_bilinear_form(
    ff.BilinearSpaces(test=V, trial=U),
    form.get_compiled(),
    params=ff.Params(kappa=1.0),
)
```

### Nonlinear residual / Jacobian

```python
U = ff.NamedSpace("U", space)
V = ff.NamedSpace("V", space)

residual = ff.ResidualForm.volume(lambda v, u, p: (v * (u.val**2)) * wf.dOmega())
R = ff.assemble_residual(
    ff.ResidualSpaces(test=V, unknown=U),
    residual.get_compiled(),
    u_vec,
    params=None,
)
J = ff.assemble_jacobian(
    ff.JacobianSpaces(test=V, trial=U),
    residual.get_compiled(),
    u_vec,
    params=None,
)
```

### Mixed / contact naming layers

```python
mixed = ff.MixedSpaces({
    "disp": ff.NamedSpace("V", V_space),
    "press": ff.NamedSpace("Q", Q_space),
}).to_fe_space()

contact = ff.ContactSpaces(master=master_side, slave=slave_side).to_contact_surface_space()
group = ff.ContactGroupSpaces(master=master_side, slaves=[slave_1, slave_2]).to_contact_surface_space()
```

The older single-space APIs remain supported. Use the `*Spaces` family when the
roles of test/trial/unknown/master/slave should be explicit in the code.

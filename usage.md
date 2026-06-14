# Usage

This note shows the current assembly flow. Prefer object-centered entrypoints:
`space.assemble(...)`, `*Spaces(...).assemble(...)`, `mixed.assemble_*`, and
`contact.assemble_*`. Top-level `ff.assemble_*` helpers remain available as
compatibility entrypoints.

## 1) Weak-form classes

```python
import fluxfem as ff
import fluxfem.helpers_wf as wf

form = ff.BilinearForm.volume(lambda u, v, p: (v.grad @ u.grad) * p.kappa * wf.dOmega())
K = space.assemble(form, params=ff.Params(kappa=1.0))
```

```python
import fluxfem.helpers_wf as wf

form = ff.LinearForm.volume(lambda v, p: (v * p.f) * wf.dOmega())
F = space.assemble(form, params=ff.Params(f=2.0))
```

## 2) Compiled weak forms

Compiled forms are optional. Use them when you want to cache and reuse the
lowered form explicitly.

```python
import fluxfem.helpers_wf as wf

form = ff.BilinearForm.volume(lambda u, v, p: (v.grad @ u.grad) * p.kappa * wf.dOmega())
compiled = form.get_compiled()
K = space.assemble(compiled, params=ff.Params(kappa=1.0))
```

```python
import fluxfem.helpers_wf as wf

form = ff.LinearForm.surface(lambda v, p: (v | p.t) * wf.ds())
compiled = form.get_compiled()

# Surface forms go through the surface/domain-aware linear-form API:
V = ff.NamedSpace("V", space)
F = ff.LinearSpaces(test=V).assemble(
    compiled,
    params=ff.Params(t=traction),
    domain=surface,
)
```

## 3) Raw kernels (tagged by metadata)

Built-in kernels in `fluxfem.physics` are tagged with `_ff_kind`/`_ff_domain`
so `kind` can be inferred. This is where `space.assemble(...)` remains the natural path.

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
F = ff.LinearSpaces(test=V).assemble(
    form,
    params=ff.Params(f=2.0),
)
```

```python
V = ff.NamedSpace("V", space)

traction = ff.LinearForm.surface(lambda v, p: wf.dot(v, p.pressure * wf.normal()) * wf.ds())
F = ff.LinearSpaces(test=V).assemble(
    traction,
    params=ff.Params(pressure=1.0),
    domain=surface,
)
```

### Bilinear form

```python
U = ff.NamedSpace("U", trial_space)
V = ff.NamedSpace("V", test_space)

form = ff.BilinearForm.volume(
    lambda u, v, p: p.kappa * wf.dot(wf.grad(v), wf.grad(u)) * wf.dOmega()
)
A = ff.BilinearSpaces(test=V, trial=U).assemble(
    form,
    params=ff.Params(kappa=1.0),
)
```

### Nonlinear residual / Jacobian

```python
U = ff.NamedSpace("U", space)
V = ff.NamedSpace("V", space)

residual = ff.ResidualForm.volume(lambda v, u, p: (v * (u.val**2)) * wf.dOmega())
R = ff.ResidualSpaces(test=V, unknown=U).assemble(
    residual,
    u_vec,
    params=None,
)
J = ff.JacobianSpaces(test=V, trial=U).assemble(
    residual,
    u_vec,
    params=None,
)
```

### Mixed residuals

```python
residuals = ff.make_mixed_residuals(
    u=res_u,
    p=res_p,
)

mixed_form = ff.ResidualForm.mixed(residuals)
R = mixed.assemble_residual(mixed_form, u_vec, params)
J = mixed.assemble_jacobian(mixed_form, u_vec, params)
```

### Contact bilinear forms

```python
contact_form = ff.BilinearForm.contact(a_contact)
B = contact.assemble_bilinear_form(contact_form, params)
```

If you want explicit reuse:

```python
compiled = ff.BilinearForm.contact(a_contact).get_compiled()
B = contact.assemble_bilinear_form(compiled, params)
```

### Mixed problems with the same flow

```python
residuals = ff.make_mixed_residuals(u=res_u, p=res_p)
mixed_form = ff.ResidualForm.mixed(residuals)

R = mixed.assemble_residual(mixed_form, u_vec, params)
J = mixed.assemble_jacobian(mixed_form, u_vec, params)
```

### Mixed / contact naming layers

```python
u_field = ff.NamedSpace("u", V_space)
p_field = ff.NamedSpace("p", Q_space)
mixed = ff.MixedSpace(u_field, p_field)

contact = ff.ContactSpaces(master=master_side, slave=slave_side).to_contact_surface_space()
group = ff.ContactGroupSpaces(master=master_side, slaves=[slave_1, slave_2]).to_contact_surface_space()
```

For an end-to-end multi-body example, see
[`tutorials/contact_supported_box_by_pillars.py`](/home/kohei/project/physics/fem/tutorials/contact_supported_box_by_pillars.py),
which couples one top body to multiple supports (merged as a disconnected
support mesh) through contact and solves the lifted structural system. For the
minimal one-master/many-slaves API shape, see
[`src/tests/test_contact_one_to_many.py`](/home/kohei/project/physics/fem/src/tests/test_contact_one_to_many.py).

The older single-space APIs remain supported. Use the `*Spaces` family when the
roles of test/trial/unknown/master/slave should be explicit in the code.

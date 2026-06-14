Kernel-based Assembly (JIT-Friendly)
====================================

This tutorial shows how to assemble using explicit, JIT-compiled element kernels.
The idea is to separate:

- **element kernels** (what gets JIT-compiled), and
- **assembly** (scatter into global vectors/matrices).

This makes the JIT boundary visible and reusable.

Core idea
---------

``make_element_*_kernel`` returns a JIT-compiled function that operates per element.
You can pass that kernel to ``space.assemble_*``.

The kernel **must return the integrated element contribution** (not the integrand).
By contrast, a **form** (``form(ctx, params)``) returns the per-quadrature
integrand, and the assembler applies ``wJ`` and sums over quadrature points.
If you pass an untagged raw kernel to ``space.assemble`` with ``kind=``,
FluxFEM emits a one-time warning; prefer tagging with ``@ff.kernel``.

.. code-block:: python

   import fluxfem as ff
   import jax
   import jax.numpy as jnp

   mesh = ff.StructuredHexBox(nx=2, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
   space = ff.make_hex_space(mesh, dim=1, intorder=2)

   # bilinear: kernel(ctx) -> (n_ldofs, n_ldofs)
   ker_K = ff.make_element_bilinear_kernel(ff.diffusion_form, 1.0, jit=True)
   K = space.assemble(ff.diffusion_form, 1.0, kernel=ker_K)

   # linear: kernel(ctx) -> (n_ldofs,)
   def linear_kernel(ctx):
       integrand = ff.scalar_body_force_form(ctx, 2.0)
       wJ = ctx.w * ctx.test.detJ
       return (integrand * wJ[:, None]).sum(axis=0)

   ker_F = jax.jit(linear_kernel)
   F = space.assemble(ff.scalar_body_force_form, 2.0, kernel=ker_F)

You can also use the unified entry point for all kernel kinds:

.. code-block:: python

   # bilinear: kernel(ctx) -> (n_ldofs, n_ldofs)
   ker_K = ff.make_element_kernel(ff.diffusion_form, 1.0, kind="bilinear")
   K = space.assemble(ff.diffusion_form, 1.0, kernel=ker_K)

   # linear: kernel(ctx) -> (n_ldofs,)
   ker_F = ff.make_element_kernel(ff.scalar_body_force_form, 2.0, kind="linear")
   F = space.assemble(ff.scalar_body_force_form, 2.0, kernel=ker_F)

   # residual: kernel(ctx, u_elem) -> (n_ldofs,)
   ker_R = ff.make_element_kernel(res_form, params, kind="residual")
   R = space.assemble_residual(res_form, u, params, kernel=ker_R)

   # jacobian: kernel(u_elem, ctx) -> (n_ldofs, n_ldofs)
   ker_J = ff.make_element_kernel(res_form, params, kind="jacobian")
   J = space.assemble_jacobian(res_form, u, params, kernel=ker_J)
   J_dense = J.to_dense()


Using compiled DSL forms
------------------------

The kernel helpers also work with weak-form DSL objects as long as you pass
the compiled form (``get_compiled()``).

.. code-block:: python

   import fluxfem as ff
   import fluxfem.helpers_wf as h_wf

   form_wf = ff.BilinearForm.volume(
       lambda u, v, p: p.kappa * (v.grad @ u.grad) * h_wf.dOmega()
   ).get_compiled()

   params = ff.Params(kappa=1.0)
   ker = ff.make_element_bilinear_kernel(form_wf, params, jit=True)
   K = space.assemble(form_wf, params, kernel=ker)


Residual and Jacobian
---------------------

Residual kernels take ``(ctx, u_elem)``, while Jacobian kernels take
``(u_elem, ctx)`` to match JAX's ``jacrev`` convention.

.. code-block:: python

   def simple_residual(ctx, u_elem, _params):
       # Return integrand only; assembly applies wJ and sums.
       return jnp.broadcast_to(u_elem, (ctx.w.shape[0], u_elem.shape[0]))

   u = jnp.zeros(space.n_dofs)
   ker_R = ff.make_element_residual_kernel(simple_residual, params=None)
   ker_J = ff.make_element_jacobian_kernel(simple_residual, params=None)

   R = space.assemble_residual(simple_residual, u, params=None, kernel=ker_R)
   J = space.assemble_jacobian(simple_residual, u, params=None, kernel=ker_J)
   J_dense = J.to_dense()


Notes
-----

- The kernel controls JIT compilation. If shapes are stable, the compiled code
  is reused across calls.
- You can combine this with chunked assembly (``policy=ff.AssemblyPolicy.chunked(...)``)
  to stabilize compilation when the last chunk is smaller.

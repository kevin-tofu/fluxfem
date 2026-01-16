Tensor Assembly
================

FluxFEM also supports tensor-based assembly, where you write element integrands
directly in terms of arrays (scikit-fem style). This pairs naturally with JAX
and makes the data flow explicit.

This page mirrors the weak-form chapter, but with tensor-style forms.

.. contents::
   :local:
   :depth: 2


Core idea
---------

A tensor-based form is a Python function that returns a per-quadrature integrand.
The assembly routines handle the quadrature weights and Jacobian determinants.

.. code-block:: python

   import fluxfem as ff
   import fluxfem.helpers_ts as h_ts

   def diffusion_form(ctx: ff.FormContext, kappa: float) -> ff.jnp.ndarray:
       grad_v = ctx.test.gradN  # (n_q, n_nodes, dim)
       grad_u = ctx.trial.gradN
       return kappa * ff.jnp.einsum("qia,qja->qij", grad_v, grad_u)

   K = space.assemble_bilinear_form(diffusion_form, params=1.0)


Forms and signatures
--------------------

Bilinear form (volume)
^^^^^^^^^^^^^^^^^^^^^^

Signature: ``(ctx, params) -> ndarray``

- ``ctx`` : ``FormContext`` with ``test`` and ``trial`` fields
- ``params`` : scalar/array or a custom object
- Return shape: ``(n_q, n_ldofs, n_ldofs)``

.. code-block:: python

   def mass_form(ctx: ff.FormContext, _p) -> ff.jnp.ndarray:
       N = ctx.test.N  # (n_q, n_nodes)
       return ff.jnp.einsum("qa,qb->qab", N, N)


Linear form (volume)
^^^^^^^^^^^^^^^^^^^^

Signature: ``(ctx, params) -> ndarray``

- Return shape: ``(n_q, n_ldofs)``

.. code-block:: python

   def body_force_form(ctx: ff.FormContext, f: float) -> ff.jnp.ndarray:
       return ctx.test.N * f


Linear form (surface)
^^^^^^^^^^^^^^^^^^^^^

Signature: ``(ctx, params) -> ndarray`` with ``SurfaceFormContext``

- Return shape: ``(n_q, n_ldofs)``

.. code-block:: python

   import numpy as np
   import fluxfem.helpers_ts as h_ts

   def traction_form(ctx: ff.SurfaceFormContext, t: np.ndarray) -> np.ndarray:
       return h_ts.dot(ctx.v, t)


Quadrature handling
-------------------

Tensor-based forms should return the **integrand only**. Assembly multiplies by
``w * detJ`` and sums over quadrature points. Do not include ``dOmega()`` or
``ds()`` in tensor-based forms.


Common building blocks
----------------------

FormContext fields
^^^^^^^^^^^^^^^^^^

- ``ctx.test.N`` / ``ctx.trial.N``: shape-function values ``(n_q, n_nodes)``
- ``ctx.test.gradN`` / ``ctx.trial.gradN``: spatial gradients ``(n_q, n_nodes, dim)``
- ``ctx.x_q``: quadrature points in physical coordinates
- ``ctx.w``: quadrature weights (used by assembly)


Helpers (helpers_ts)
^^^^^^^^^^^^^^^^^^^^

``helpers_ts`` exposes tensor operators used in the physics modules:

- ``h_ts.sym_grad(field)``: Voigt B-matrix for linear elasticity
- ``h_ts.ddot(a, b, c)``: contractions for elasticity blocks
- ``h_ts.dot(field, load)``: vector load form for surface/volume loads


Recipes (from tests)
--------------------

Mass (scalar)
^^^^^^^^^^^^^

.. code-block:: python

   def mass_form(ctx: ff.FormContext, _p) -> ff.jnp.ndarray:
       N = ctx.test.N
       return ff.jnp.einsum("qa,qb->qab", N, N)


Diffusion
^^^^^^^^^

.. code-block:: python

   def diffusion_form(ctx: ff.FormContext, kappa: float) -> ff.jnp.ndarray:
       grad_v = ctx.test.gradN
       grad_u = ctx.trial.gradN
       return kappa * ff.jnp.einsum("qia,qja->qij", grad_v, grad_u)


Linear elasticity
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import fluxfem.helpers_ts as h_ts

   def linear_elasticity_form(ctx: ff.FormContext, D: np.ndarray) -> ff.jnp.ndarray:
       Bu = h_ts.sym_grad(ctx.trial)
       Bv = h_ts.sym_grad(ctx.test)
       return h_ts.ddot(Bv, D, Bu)


Surface traction
^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import fluxfem.helpers_ts as h_ts

   def traction_form(ctx: ff.SurfaceFormContext, traction_vec: np.ndarray) -> np.ndarray:
       return h_ts.dot(ctx.v, traction_vec)

   F_tensor = surface.assemble_linear_form_on_space(
       space, traction_form, params=traction_vec
   )


See also
--------

For explicit JIT boundary control with element kernels, see
:doc:`tutorial_kernel_assembly`.

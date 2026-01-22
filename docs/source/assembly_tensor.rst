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
   import jax.numpy as jnp

   @ff.kernel(kind="bilinear", domain="volume")
   def diffusion_form(ctx: ff.FormContext, kappa: float) -> jnp.ndarray:
       grad_v = ctx.test.gradN  # (n_q, n_nodes, dim)
       grad_u = ctx.trial.gradN
       return kappa * jnp.einsum("qia,qja->qij", grad_v, grad_u)

   K = space.assemble(diffusion_form, params=1.0)


Kernel metadata (ff.kernel)
---------------------------

Tensor-based kernels do not include measure terms, so FluxFEM cannot infer
whether they are volume or surface forms. Use ``@ff.kernel`` to tag them:

- ``kind``: ``"bilinear"``, ``"linear"``, ``"residual"``, ``"jacobian"`` (future-ready)
- ``domain``: ``"volume"`` or ``"surface"``

If you pass an untagged kernel and also provide ``kind=`` explicitly,
FluxFEM will emit a one-time warning to encourage tagging. You can silence it with:

.. code-block:: python

   import warnings

   warnings.filterwarnings(
       "ignore",
       message="Raw kernel has no _ff_kind metadata",
       category=UserWarning,
   )


Forms and signatures
--------------------

Bilinear form (volume)
^^^^^^^^^^^^^^^^^^^^^^

Signature: ``(ctx, params) -> ndarray``

- ``ctx`` : ``FormContext`` with ``test`` and ``trial`` fields
- ``params`` : scalar/array or a custom object
- Return shape: ``(n_q, n_ldofs, n_ldofs)``

.. code-block:: python

   import jax.numpy as jnp

   def mass_form(ctx: ff.FormContext, _p) -> jnp.ndarray:
       N = ctx.test.N  # (n_q, n_nodes)
       return jnp.einsum("qa,qb->qab", N, N)


Linear form (volume)
^^^^^^^^^^^^^^^^^^^^

Signature: ``(ctx, params) -> ndarray``

- Return shape: ``(n_q, n_ldofs)``

.. code-block:: python

   import jax.numpy as jnp

   def body_force_form(ctx: ff.FormContext, f: float) -> jnp.ndarray:
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

   import jax.numpy as jnp

   def mass_form(ctx: ff.FormContext, _p) -> jnp.ndarray:
       N = ctx.test.N
       return jnp.einsum("qa,qb->qab", N, N)


Diffusion
^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp

   def diffusion_form(ctx: ff.FormContext, kappa: float) -> jnp.ndarray:
       grad_v = ctx.test.gradN
       grad_u = ctx.trial.gradN
       return kappa * jnp.einsum("qia,qja->qij", grad_v, grad_u)


Linear elasticity
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import fluxfem.helpers_ts as h_ts
   import jax.numpy as jnp

   def linear_elasticity_form(ctx: ff.FormContext, D: np.ndarray) -> jnp.ndarray:
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

Autodiff Tutorial: Diffusion Sensitivity
========================================

This tutorial explains ``tutorials/autodiff_diffusion_sensitivity.py`` and shows how to
compute gradients of a scalar loss with respect to a diffusion coefficient and a
boundary traction using JAX.

Run the example
^^^^^^^^^^^^^^^

.. code-block:: bash

   python tutorials/autodiff_diffusion_sensitivity.py

Problem setup
^^^^^^^^^^^^^

We solve a scalar diffusion problem on a structured hex mesh (``dim=1``):

- Unknown field: ``u``
- Diffusion coefficient: ``kappa``
- Dirichlet boundary: ``u = 0`` on ``x = xmin``
- Neumann boundary: constant traction on ``x = xmax``

Weak form
^^^^^^^^^

Find ``u`` such that for all ``v``:

.. math::

   \int_{\Omega} \kappa \, \nabla v \cdot \nabla u \, d\Omega
   = \int_{\Gamma_t} v \, t \, ds

Loss and sensitivities
^^^^^^^^^^^^^^^^^^^^^^

The tutorial defines a simple quadratic loss and differentiates it:

.. math::

   \mathcal{L}(u) = \tfrac{1}{2} \|u\|^2, \quad
   \frac{d\mathcal{L}}{d\kappa},\; \frac{d\mathcal{L}}{dt}

Implementation flow
^^^^^^^^^^^^^^^^^^^

1) Precompute reference operators
"""""""""""""""""""""""""

Because the stiffness operator is linear in ``kappa`` and the surface load is
linear in the traction, the script precomputes reference matrices and scales them
inside the JAX-traced function:

.. code-block:: python

   K0 = jnp.asarray(space.assemble_bilinear_form(ff.diffusion_form, params=1.0).to_dense())
   F_base = surface.assemble_linear_form_on_space(
       space, surface_form.get_compiled(), params=1.0
   )
   F_base = jnp.asarray(F_base)

2) Solve and differentiate
"""""""""""""""""""""""

.. code-block:: python

   def loss_fn(kappa, traction):
       K = kappa * K0
       F = traction * F_base
       u = solve_linear_system(K, F)
       return 0.5 * jnp.dot(u, u)

   grad_kappa, grad_trac = jax.grad(loss_fn, argnums=(0, 1))(kappa0, traction0)

This produces sensitivities of the loss with respect to the diffusion coefficient
and the boundary traction.

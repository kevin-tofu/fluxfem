Autodiff Tutorial: Inverse Diffusion 
=====================================

This tutorial explains ``tutorials/inverse_diffusion_kappa.py`` and shows how to
recover a diffusion coefficient from synthetic observations using JAX autodiff.

Run the example
^^^^^^^^^^^^^^^

.. code-block:: bash

   python tutorials/inverse_diffusion_kappa.py

Problem setup
^^^^^^^^^^^^^

We solve a scalar diffusion problem on a structured hex mesh (``dim=1``):

- Unknown field: ``u``
- Diffusion coefficient: ``kappa`` (unknown)
- Dirichlet boundary: ``u = 0`` on ``x = xmin``
- Neumann boundary: constant traction on ``x = xmax``

Inverse problem
^^^^^^^^^^^^^^^

We first generate synthetic observations by solving with a known coefficient
``kappa_true`` and adding small noise to the resulting field. We then fit ``kappa``
by minimizing a mean-squared error on observed dofs on the ``x = xmax`` boundary
(the traction boundary, not the Dirichlet boundary).

Forward model (weak form)
^^^^^^^^^^^^^^^^^^^^^^^^^

Find ``u`` such that for all ``v``:

.. math::

   \int_{\Omega} \kappa \, \nabla v \cdot \nabla u \, d\Omega
   = \int_{\Gamma_t} v \, t \, ds

Loss and gradient
^^^^^^^^^^^^^^^^^

We optimize a scalar parameter ``kappa`` using a log-parameterization
``kappa = exp(theta)`` so the coefficient stays positive during gradient descent:

.. math::

   \mathcal{L}(\kappa) = \tfrac{1}{2N} \sum_i (u_i(\kappa) - u_i^{obs})^2

Implementation flow
^^^^^^^^^^^^^^^^^^^

1) Precompute reference operators
"""""""""""""""""""""""""""""""""

Because the stiffness operator is linear in ``kappa`` and the surface load is
linear in the traction, the script precomputes reference matrices and scales them
inside the JAX-traced function:

.. code-block:: python

   K0 = jnp.asarray(
       space.assemble(ff.diffusion_form, params=1.0).to_dense(),
       dtype=jnp.float64,
   )
   F_base = surface.assemble_linear_form_on_space(
       space, surface_form.get_compiled(), params=1.0
   )
   F_base = jnp.asarray(F_base, dtype=jnp.float64)

2) Generate synthetic observations
""""""""""""""""""""""""""""""""""

.. code-block:: python

   kappa_true = jnp.array(2.5, dtype=jnp.float64)
   traction_true = jnp.array(1.0, dtype=jnp.float64)
   u_synth = solve_u(kappa_true, traction_true)
   u_obs = u_synth + noise

   boundary_dofs = mesh.boundary_dofs_where(
       lambda pts: np.isclose(pts[:, 0], xmax, atol=1e-8),
       components=[0],
       dof_per_node=1,
   )
   boundary_free_dofs = np.setdiff1d(boundary_dofs, dir_dofs)
   obs_count = rng.integers(obs_min, obs_max + 1)
   obs_idx = rng.choice(boundary_free_dofs, size=obs_count, replace=False)

3) Optimize kappa with autodiff
"""""""""""""""""""""""""""""""

.. code-block:: python

   def loss_theta(theta):
       kappa = jnp.exp(theta)
       u = solve_u(kappa, traction_true)
       diff = u[obs_idx] - u_obs[obs_idx]
       return 0.5 * jnp.mean(diff * diff)

   grad_fn = jax.grad(loss_theta)
   theta = jnp.log(jnp.array(1.0, dtype=jnp.float64))
   for _ in range(steps):
       theta = theta - lr * grad_fn(theta)

This yields an estimate ``kappa_est = exp(theta)`` that can be compared to
``kappa_true`` (used only to generate synthetic observations). With real data,
replace ``u_obs`` with measurements and remove the synthetic generation block.

JIT compilation
^^^^^^^^^^^^^^^

You can speed up the forward solve and loss/gradient evaluation with JAX JIT.
In ``tutorials/inverse_diffusion_kappa.py`` the functions are JIT-compiled and
then reused in the optimization loop:

.. code-block:: python

   solve_u_jit = jax.jit(solve_u)

   def loss_theta(theta):
       kappa = jnp.exp(theta)
       u = solve_u_jit(kappa, traction_true)
       diff = u[obs_idx_j] - u_obs[obs_idx_j]
       return 0.5 * jnp.mean(diff * diff)

   loss_theta_jit = jax.jit(loss_theta)
   grad_fn = jax.jit(jax.grad(loss_theta))

If you want to separate compilation from execution, you can call
``loss_theta_jit.compile`` or ``grad_fn.compile`` once before the loop to pay
the compile cost up front. The compiled functions assume fixed shapes (mesh,
observation indices, and dof layout), so reusing the same inputs across steps
avoids recompilation.

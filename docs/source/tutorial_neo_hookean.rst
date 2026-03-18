Hyper-Elasticity: Hyperelastic Cantilever
=========================================

This tutorial summarizes ``tutorials/neo_hookean_cantilever.py`` and
explains the nonlinear formulation, loads, and solver flow used for a 3D
Neo-Hookean cantilever.

Run the example
^^^^^^^^^^^^^^^

.. code-block:: bash

   python tutorials/neo_hookean_cantilever.py

Problem statement
^^^^^^^^^^^^^^^^^

We solve a finite-strain neo-Hookean problem on a 3D cantilever:

- Unknown displacement field: ``u``
- Test function: ``v``
- Material: compressible Neo-Hookean, with Lamé parameters ``(lam, mu)``
- Boundary conditions:
  - Dirichlet (clamped) on ``x = xmin``
  - Uniform traction on ``x = xmax`` along ``+y``

Hyperelastic formulation
^^^^^^^^^^^^^^^^^^^^^^^^

Let the deformation gradient be:

.. math::

   F = I + \nabla u, \quad J = \det F

For a compressible Neo-Hookean material, a standard strain-energy density is:

.. math::

   W(F) = \frac{\mu}{2}(I_1 - 3) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2

where :math:`I_1 = tr(F^T F)`. The first Piola-Kirchhoff stress is
:math:`P = \partial W / \partial F`. The weak form is: find ``u`` such that for all
``v``:

.. math::

   R(u; v) = \int_{\Omega} P(F(u)) : \nabla v \, d\Omega
   - \int_{\Omega} v \cdot b \, d\Omega
   - \int_{\Gamma_t} v \cdot t \, ds = 0

In this example, the body force ``b`` is zero and the traction
``t = (0, traction, 0)`` is applied on ``x = xmax``.

Implementation flow (FluxFEM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1) Build mesh and space
"""""""""""""""""""""""

.. code-block:: python

   mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
   space = ff.make_hex_space(mesh, dim=3, intorder=intorder)

2) Assemble external forces
"""""""""""""""""""""""""""

Body force:

.. code-block:: python

   f_body = jnp.array(body_force, dtype=dtype)
   F_ext = space.assemble(
       ff.vector_body_force_form, params=f_body, sparse=False
   )

Surface traction on ``x = xmax``:

.. code-block:: python

   def traction_form(ctx: ff.SurfaceFormContext, traction_vec: np.ndarray) -> np.ndarray:
       return h_ts.dot(ctx.v, traction_vec)

   traction_vec = np.array([0.0, traction, 0.0], dtype=float)
   F_ext = surf.assemble_linear_form_on_space(space, traction_form, params=traction_vec, F0=F_ext)

3) Apply Dirichlet clamp
""""""""""""""""""""""""

.. code-block:: python

   dir_dofs = ff.DirichletBC.from_boundary_dofs(
       mesh,
       lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
       components="xyz",
   ).dofs

4) Nonlinear analysis and Newton solve
""""""""""""""""""""""""""""""""""""""

FluxFEM provides a nonlinear analysis wrapper and a Newton loop. The residual can be
expressed as a weak-form function (conceptually), and then assembled per element.
The script uses the built-in ``ff.neo_hookean_residual_form`` internally,
but the weak-form mapping looks like this (PK2 form):

.. code-block:: python

   import fluxfem as ff
   import fluxfem.helpers_wf as h_wf

   def neo_hookean_residual_wf(v, u, params):
       mu = params["mu"]
       lam = params["lam"]
       F = h_wf.I(3) + h_wf.grad(u)  # deformation gradient
       C = h_wf.matmul(h_wf.transpose(F), F)
       C_inv = h_wf.inv(C)
       J = h_wf.det(F)

       S = mu * (h_wf.I(3) - C_inv) + lam * h_wf.log(J) * C_inv
       dE = 0.5 * (h_wf.matmul(h_wf.grad(v), F) + h_wf.transpose(h_wf.matmul(h_wf.grad(v), F)))
       return h_wf.ddot(S, dE) * h_wf.dOmega()

   residual_form = ff.ResidualForm.volume(neo_hookean_residual_wf)
   R = space.assemble_residual(residual_form, u, params=params)

The role-explicit equivalent keeps the same weak form but binds test/unknown
roles through ``NamedSpace``:

.. code-block:: python

   U = ff.NamedSpace("U", space)
   V = ff.NamedSpace("V", space)
   residual = ff.ResidualForm.volume(neo_hookean_residual_wf)

   R = ff.assemble_residual(
       ff.ResidualSpaces(test=V, unknown=U),
       residual,
       u,
       params,
   )
   J = ff.assemble_jacobian(
       ff.JacobianSpaces(test=V, trial=U),
       residual,
       u,
       params,
   )

In FluxFEM, ``ff.neo_hookean_residual_form`` uses the PK2 stress ``S`` internally,
and the weak-form expression above is consistent with that choice.

.. code-block:: python

   analysis = ff.NonlinearAnalysis(
       space=space,
       residual_form=ff.neo_hookean_residual_form,
       params=params,
       base_external_vector=F_ext,
       dirichlet=(dir_dofs, None),
       jacobian_pattern=J_pattern,
       dtype=dtype,
   )

In weak-form terms, the element residual corresponds to:

.. math::

   \begin{aligned}
   r_e(u; v) &= \int_{\Omega_e} S(u) : \delta E(u; v) \, d\Omega - f_{ext,e} \\
            &= \int_{\Omega_e} P(u) : \nabla v \, d\Omega - f_{ext,e}
   \end{aligned}

FluxFEM then assembles the element contributions over the mesh using the current
state ``u`` and subtracts the external force vector ``F_ext`` (from body forces and
surface traction) when forming the global residual.

   runner = ff.NewtonSolveRunner(analysis, newton_cfg)
   u, history = runner.run(u0=jnp.zeros(space.n_dofs, dtype=dtype))

The script reports the maximum displacement magnitude and optionally writes
``result.vtu`` for visualization.

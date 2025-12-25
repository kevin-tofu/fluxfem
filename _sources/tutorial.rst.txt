Tutorial
========

This section contains tutorials for fluxfem.

Linear Elasticity: Weak Form to Implementation
----------------------------------------------

This section documents how the weak form used in
``tutorials/linearelastic_tensile_bar.py`` maps to the fluxfem implementation.

Problem statement
^^^^^^^^^^^^^^^^^

We solve a small-strain linear elasticity problem on a 3D bar:

- Unknown displacement field: ``u``
- Test function: ``v``
- Material: isotropic, given by ``D(E, nu)``
- Boundary conditions:
  - Dirichlet (clamped) on ``x = 0``
  - Traction on ``x = L``

Weak form
^^^^^^^^^

Find ``u`` such that for all ``v``:

.. math::

   \int_{\Omega} \varepsilon(v) : D : \varepsilon(u)\, d\Omega
   = \int_{\Gamma_t} v \cdot t \, ds

where ``\varepsilon(u) = sym(grad(u))``.

Implementation mapping (fluxfem)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

weak-form-based assembly:

.. code-block:: python

   import fluxfem.helpers_wf as h_wf

   bilinear_form = ff.BilinearForm.volume(
       lambda u, v, D: h_wf.ddot(v.sym_grad, D @ u.sym_grad) * h_wf.dOmega()
   )
   K_wf = space.assemble_bilinear_form(bilinear_form.bilinear_form(), params=D)


tensor-based assembly (scikit-fem-style):

.. code-block:: python

   import fluxfem.helpers_ts as h_ts

   def linear_elasticity_form(ctx: ff.FormContext, D: np.ndarray) -> ff.jnp.ndarray:
       Bu = h_ts.sym_grad(ctx.u)
       Bv = h_ts.sym_grad(ctx.v)
       return h_ts.ddot(Bv, D, Bu)

   K = space.assemble_bilinear_form(linear_elasticity_form, params=D)


Surface traction (weak-form-based assembly):

.. code-block:: python

    import fluxfem.helpers_wf as h_wf

    surface_form = ff.LinearForm.surface(
        lambda v, p: (v | p) * h_wf.ds()
    )
    F_wf = surface.assemble_linear_form_on_space(
        space, surface_form.linear_form(), params=traction_vec
    )


Surface traction (tensor-based assembly):

.. code-block:: python

    import fluxfem.helpers_ts as h_ts

    def surface_traction_form(
       ctx: ff.SurfaceFormContext, traction_vec: np.ndarray
    ) -> np.ndarray:
       return h_ts.dot(ctx.v, traction_vec)

    F_tensor = surface.assemble_linear_form_on_space(
       space, surface_traction_form, params=traction_vec
    )


Dirichlet clamp:

.. code-block:: python

   dir_dofs = mesh.boundary_dofs_where(
       lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
       components="xyz",
   )

   u, _ = ff.LinearSolver(method="spsolve").solve(
       K, F, dirichlet=(dir_dofs, None), dirichlet_mode="condense"
   )

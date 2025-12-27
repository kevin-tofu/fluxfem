Linear Elasticity: Tensile Bar (Simplified)
===========================================

This tutorial walks through the minimal tensile-bar example in
``tutorials/linearelastic_tensile_bar_simplified.py`` and explains the weak form,
assembly, and boundary conditions with the key equations.

Run the example
^^^^^^^^^^^^^^^

.. code-block:: bash

   python tutorials/linearelastic_tensile_bar_simplified.py

Problem statement
^^^^^^^^^^^^^^^^^

We solve a small-strain linear elasticity problem on a 3D bar:

- Unknown displacement field: ``u``
- Test function: ``v``
- Material: isotropic, given by ``D(E, nu)``
- Boundary conditions:
  - Dirichlet (clamped) on ``x = xmin``
  - Uniform traction on ``x = xmax`` along ``+x``

Weak form
^^^^^^^^^

Find ``u`` such that for all ``v``:

.. math::

   \int_{\Omega} \varepsilon(v) : D : \varepsilon(u)\, d\Omega
   = \int_{\Gamma_t} v \cdot t \, ds

where :math:`\varepsilon(u) = \tfrac{1}{2}(\nabla u + \nabla u^T)`.

In this example, ``t = (traction, 0, 0)`` is applied on ``x = xmax``.

Implementation flow (FluxFEM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1) Build mesh and space
"""""""""""""""""""""""

The script builds a structured hexahedral mesh and creates a vector-valued FEM space:

.. code-block:: python

   mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
   space = ff.make_hex_space(mesh, dim=3, intorder=intorder)

2) Assemble the bilinear form (weak form)
"""""""""""""""""""""""""""""""""""""""""

.. math::

   a(u, v) = \int_{\Omega} \varepsilon(v) : D : \varepsilon(u)\, d\Omega

.. code-block:: python

   import fluxfem.helpers_wf as h_wf

   bilinear_form = ff.BilinearForm.volume(
       lambda u, v, D_mat: h_wf.ddot(v.sym_grad, h_wf.matmul_std(D_mat, u.sym_grad))
       * h_wf.dOmega()
   )
   K = space.assemble_bilinear_form(bilinear_form.get_compiled(), params=D)

3) Assemble the surface traction
""""""""""""""""""""""""""""""""

.. math::

   \ell(v) = \int_{\Gamma_t} v \cdot t \, ds

.. code-block:: python

   surface_form = ff.LinearForm.surface(lambda v, p: (v | p) * h_wf.ds())
   traction_vec = np.array([traction, 0.0, 0.0], dtype=float)
   F = surface.assemble_linear_form_on_space(
       space, surface_form.get_compiled(), params=traction_vec
   )

4) Apply Dirichlet clamp
""""""""""""""""""""""""

The clamp fixes all components on the ``x = xmin`` face:

.. code-block:: python

   dir_dofs = mesh.boundary_dofs_where(
       lambda pts: np.isclose(pts[:, 0], xmin, atol=1e-8),
       components="xyz",
   )

5) Solve the linear system
""""""""""""""""""""""""""

.. code-block:: python

   solver = ff.LinearSolver(method="spsolve")
   u, _ = solver.solve(
       K, F, dirichlet=(dir_dofs, None), dirichlet_mode="condense"
   )

The script also prints the maximum axial displacement at ``x = xmax`` and compares
it with the 1D bar theory ``u_x(L) = traction * L / E``.

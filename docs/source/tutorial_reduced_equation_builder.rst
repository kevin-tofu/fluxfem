Reduced Equation Builder
========================

This tutorial summarizes ``tutorials/reduced_equation_builder_nonlinear.py``.
It shows the residual-first layer for future differentiable reduced
formulations.

Run
---

.. code-block:: bash

   PYTHONPATH=src python tutorials/reduced_equation_builder_nonlinear.py

What It Does
------------

The script creates two Craig-Bampton-reduced fields and composes:

- one nonlinear field-local residual per subsystem
- one user-defined constraint residual across the retained interface coordinates
- a global reduced residual vector
- a global reduced Jacobian from ``jax.jacrev``
- optional static Newton and implicit Newmark solves on the same residual API

This complements ``ReducedCoupledSystemBuilder``.  Use the coupled-system
builder for assembled linear ``K/M/f`` workflows; use ``ReducedEquationBuilder``
when the governing equation is naturally expressed as residual functions.

Key API Shape
-------------

.. code-block:: python

   builder = ff.ReducedEquationBuilder()
   builder.register_field("part_a", basis=cb_a)
   builder.register_field("part_b", basis=cb_b)

   builder.add_field_residual("part_a", residual_a)
   builder.add_field_residual("part_b", residual_b)
   builder.add_constraint(("part_a", "part_b"), coupling_residual)

   class GapConstraint:
       fields = ("part_a", "part_b")

       def residual(self, qa, qb):
           ...

   builder.add_constraint(GapConstraint())

   problem = builder.build()
   q, info = ff.solve_reduced_equation(problem, q0)
   r = problem.residual(q)
   j = problem.jacobian(q)

For transient reduced dynamics, use the same residual as the internal force:

.. code-block:: python

   next_state, info = ff.reduced_equation_newmark_step(
       problem, mass, damping, external_force, state, config
   )

Residual functions own the physics.  The builder only handles reduced-field
slicing and scattering.

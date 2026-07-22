Craig-Bampton Two-Subsystem ROM
===============================

This tutorial summarizes ``tutorials/craig_bampton/craig_bampton_two_bar_subsystems.py``.
It shows the intended workflow for multiple larger subsystems: each subsystem
owns its own sparse ``K`` and ``M``, each gets its own Craig-Bampton basis, and
the reduced subsystems are coupled through retained interface DOFs.

Run
---

.. code-block:: bash

   PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_two_bar_subsystems.py

What It Does
------------

The script builds two scalar hex bars as separate FE subsystems:

- ``part_a`` has its support and right interface face retained.
- ``part_b`` has its left interface face and loaded end retained.
- both parts are reduced independently with ``reduce_field(...)``.
- ``part_a:interface`` and ``part_b:interface`` are tied in the reduced system.

The reduced solve is compared with a full-order block KKT solve using the same
interface tie constraints.  The output reports full and reduced DOF counts,
the reduced mass matrix shapes for each subsystem, the interface constraint
residual, and the relative error against the full solution.

Key API Shape
-------------

.. code-block:: python

   builder = ff.ReducedCoupledSystemBuilder.from_structural(
       "part_a", K_a, f_a, mass=M_a
   )
   builder.register_structural("part_b", K_b, f_b, mass=M_b)

   builder.retain_dof_group("part_a", "interface", dofs_a)
   builder.retain_dof_group("part_b", "interface", dofs_b)

   builder.reduce_field("part_a", retained_groups=["support", "interface"], n_modes=6)
   builder.reduce_field("part_b", retained_groups=["interface", "load"], n_modes=6)
   builder.tie_retained_groups("part_a:interface", "part_b:interface")

   system = builder.build()

This is the same pattern to use when different components represent different
physical phenomena or different assembled FE models.

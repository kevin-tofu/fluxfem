Supported Box Contact Tutorial
==============================

This tutorial demonstrates a large box supported by multiple smaller boxes.

- Top box: gravity load.
- Support boxes: bottom faces are fixed with Dirichlet BC.
- Contact coupling: penalty-style interface residual/Jacobian assembly.

Script
------

- ``tutorials/contact_supported_box_by_pillars.py``

Run
---

.. code-block:: bash

   PYTHONPATH=src python tutorials/contact_supported_box_by_pillars.py

What it showcases
-----------------

- Building contact between a large top box and a merged support mesh
  (multiple disconnected small boxes).
- Creating the interface through ``ContactSide`` + ``ContactSpaces`` so the
  master/slave roles are explicit in the public API.
- Assembling penalty-family operators via ``assemble_contact_operators(..., enforcement="penalty")``.
- Lifting interface Jacobian/residual into structural DOFs via ``CoupledSystem.add_contact_nitsche(...)``.
- Building gravity/body-force vectors and support-bottom Dirichlet DOFs.

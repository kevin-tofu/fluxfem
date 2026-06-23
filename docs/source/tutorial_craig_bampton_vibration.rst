Craig-Bampton Vibration ROM
===========================

This tutorial summarizes ``tutorials/craig_bampton/craig_bampton_vibration_rom.py``.
It shows that Craig-Bampton reduction in FluxFEM is not stiffness-only: the
same basis projects the mass matrix for modal and transient dynamics.

Run
---

.. code-block:: bash

   PYTHONPATH=src python tutorials/craig_bampton/craig_bampton_vibration_rom.py

What It Does
------------

The script builds a small scalar hex bar, fixes the left face, and retains the
right face as physical interface coordinates.  It assembles sparse full-order
stiffness and mass matrices, builds a Craig-Bampton basis from ``K`` and ``M``,
then forms

.. code-block:: python

   Kr = cb.project_matrix(K)
   Mr = cb.project_matrix(M)

Finally, it solves the generalized eigenproblems
``K phi = omega^2 M phi`` and ``Kr q = omega^2 Mr q`` and prints the frequency
error against the full-order model.

Notes
-----

- ``make_craig_bampton_basis(K, M, ...)`` uses the internal mass block in the
  fixed-interface modal solve.
- ``cb.project_matrix(M)`` gives the reduced mass matrix used for vibration or
  Newmark-style dynamics.
- Sparse inputs can be passed directly; ``project_matrix`` avoids unnecessary
  densification during the full-matrix multiply.

Mortar Fixture-Workpiece Comparison
===================================

This tutorial summarizes ``tutorials/contact/mortar_fixture_workpiece_comparison.py``.
It is a compact numerical check for coarse P1 mortar on small
fixture/workpiece interface solves.

Run
---

.. code-block:: bash

   PYTHONPATH=src python tutorials/contact/mortar_fixture_workpiece_comparison.py

The script runs two cases:

- matching fixture/workpiece surface meshes
- nonmatching fixture/workpiece surface meshes with a supermesh

The dual mortar result is used as the reference.  The main quantity to inspect
is the coarse P1 row, which should stay close to the dual reference while using
fewer multiplier DOFs for this low-order interface response.

Pass ``--all-variants`` when you want to inspect the lower-level diagnostic
rows used for development comparisons.

Output
------

The table reports:

- multiplier DOFs
- KKT rank and size
- displacement infinity-norm error against dual mortar
- relative work-like scalar error against dual mortar

Optional VTU output:

.. code-block:: bash

   PYTHONPATH=src python tutorials/contact/mortar_fixture_workpiece_comparison.py \
       --output-dir result/tutorials/mortar_fixture_workpiece

The VTU path writes fixture and workpiece surface files for the displayed rows
when ``meshio`` is installed.  Combine it with ``--all-variants`` to export the
diagnostic variants too.

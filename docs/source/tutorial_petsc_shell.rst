PETSc Shell Solvers
===================

This tutorial summarizes the PETSc shell-matrix demos:

- ``tutorials/petsc_shell_poisson_demo.py``
- ``tutorials/petsc_shell_poisson_pmat_demo.py``

They show how to wrap FluxFEM operators in PETSc shells for flexible
preconditioning and matrix-free solves.

Run the examples
^^^^^^^^^^^^^^^^

.. code-block:: bash

   python tutorials/petsc_shell_poisson_demo.py
   python tutorials/petsc_shell_poisson_pmat_demo.py

Prerequisites
^^^^^^^^^^^^^

You need PETSc + petsc4py:

.. code-block:: bash

   poetry add fluxfem --extras "petsc"

What the demos do
^^^^^^^^^^^^^^^^

- Build a Poisson problem on a structured mesh.
- Assemble the residual and wrap it in a PETSc shell matrix.
- Solve with PETSc, optionally providing a matrix for preconditioning.

Notes
^^^^^

- If PETSc is not installed, the scripts will error at import time.
- These demos are a good starting point for matrix-free workflows.

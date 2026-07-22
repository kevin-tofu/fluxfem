PETSc Shell Solvers
===================

This tutorial summarizes the PETSc shell-matrix demos:

- ``tutorials/petsc/petsc_shell_poisson_demo.py``
- ``tutorials/petsc/petsc_shell_poisson_pmat_demo.py``

They show how to wrap FluxFEM operators in PETSc shells for flexible
preconditioning and matrix-free solves.

Prerequisites
^^^^^^^^^^^^^

The demos require PETSc and ``petsc4py`` (the ``petsc`` extra). For source
checkouts with a custom PETSc build, install the regular Poetry environment and
then build ``petsc4py`` against that PETSc:

.. code-block:: bash

   export PETSC_DIR=/path/to/petsc-3.24.4
   export PETSC_ARCH=arch-linux-c-opt
   scripts/install_petsc4py

Ensure ``PETSC_DIR`` (and optionally ``PETSC_ARCH``) point to a valid PETSc build
so ``petsc4py`` can import successfully. Once PETSc is configured, run:

Run the examples
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python tutorials/petsc/petsc_shell_poisson_demo.py
   python tutorials/petsc/petsc_shell_poisson_pmat_demo.py

What the demos do
^^^^^^^^^^^^^^^^^^^

- Build a structured 2D Poisson residual and wrap its action in a PETSc shell
  matrix via ``ff.petsc_shell_solve``.
- Let PETSc control the Krylov method (``ksp_type``) and preconditioning via the
  options database.
- Show both a pure shell solve with a diagonal preconditioner and a hybrid mode
  where ``pmat`` supplies an explicit matrix for PETSc preconditioners such as
  ILU or Jacobi.

Details per script
^^^^^^^^^^^^^^^^^^

- ``tutorials/petsc/petsc_shell_poisson_demo.py`` keeps everything matrix-free and
  sets ``preconditioner="diag0"`` so PETSc only sees the shell operator.
- ``tutorials/petsc/petsc_shell_poisson_pmat_demo.py`` still uses the shell for the
  mat-vec but passes ``pmat=A`` plus ``pc_type="jacobi"`` so PETSc can build an
  explicit preconditioner from the assembled stiffness while the shell retains
  flexibility for the operator action.

The ``pmat`` argument is handy whenever a preconditioner needs access to the
numerical matrix or sparsity pattern even though the mat-vec is computed on
the fly. These demos thus document both a fully matrix-free workflow and a
hybrid matrix-preconditioner setup that keeps the best of both worlds.

Notes
^^^^^

- If PETSc is not installed, the scripts will raise during import.
- The demos are a good starting point for matrix-free or hybrid preconditioning
  workflows inside FluxFEM.

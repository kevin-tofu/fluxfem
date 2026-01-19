Installation
============

FluxFEM supports Python 3.11-3.13 (inclusive) and depends on JAX >= 0.8.2.
The base install includes CPU JAX. Use the CUDA extra if you need GPU support.

JAX (CPU)
---------

.. code-block:: bash

   pip install -U "jax[cpu]"

JAX (GPU)
---------

Install the CUDA-enabled JAX build that matches your CUDA toolkit. See the
official JAX install guide for the latest wheels:

https://jax.readthedocs.io/en/latest/installation.html

Example (CUDA 12):

.. code-block:: bash

   pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Quick install (pip)
-------------------

.. code-block:: bash

   pip install fluxfem
   pip install "fluxfem[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Poetry
------

.. code-block:: bash

   poetry add fluxfem
   poetry add fluxfem[cuda12]

If you need a specific CUDA wheel, you can install JAX inside the Poetry env:

.. code-block:: bash

   poetry run pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

PETSc (optional)
----------------

PETSc-based solvers are available via `petsc4py` (pinned to 3.23.6 to match
PETSC_DIR in this repo):

.. code-block:: bash

   poetry add fluxfem --extras "petsc"

If you build or link against a custom PETSc installation, set `PETSC_DIR`
(and `PETSC_ARCH` if applicable) so `petsc4py` can locate the libraries:

.. code-block:: bash

   export PETSC_DIR=/path/to/petsc
   export PETSC_ARCH=arch-linux-c-opt  # optional

Note: newer `petsc4py` expects PETSc builds that include the `PetscRegressor`
API. If your PETSc build does not have it, `petsc4py` will fail to compile.

GPU note: this repo currently tests CUDA via the `cuda12` extra only. Other
CUDA versions are not covered by CI and may require manual JAX installation.

From source
----------------------

.. code-block:: bash

   git clone https://github.com/kevin-tofu/fluxfem.git
   cd fluxfem
   pip install -e .

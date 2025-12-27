Core
====

Spaces
------

.. autoclass:: fluxfem.core.FESpaceBase
.. autoclass:: fluxfem.core.FESpace
.. autoclass:: fluxfem.core.FESpacePytree

.. autofunction:: fluxfem.core.make_space
.. autofunction:: fluxfem.core.make_space_pytree
.. autofunction:: fluxfem.core.make_hex_space
.. autofunction:: fluxfem.core.make_hex_space_pytree
.. autofunction:: fluxfem.core.make_tet_space
.. autofunction:: fluxfem.core.make_tet_space_pytree

Forms
-----

.. autoclass:: fluxfem.core.FormContext
.. autoclass:: fluxfem.core.MixedFormContext
.. autoclass:: fluxfem.core.VolumeContext
.. autoclass:: fluxfem.core.SurfaceContext

.. autoclass:: fluxfem.core.LinearForm
.. autoclass:: fluxfem.core.BilinearForm
.. autoclass:: fluxfem.core.ResidualForm
.. autoclass:: fluxfem.core.MixedWeakForm

Assembly
--------

.. autofunction:: fluxfem.core.make_sparsity_pattern

Role-Explicit Specs
===================

These helper specs make test/trial/unknown or master/slave roles explicit in
the public API while reusing the same underlying assembly and space objects.

For volume assembly, prefer the ``*Spaces`` family over transitional helpers.
Deprecated compatibility paths are summarized in
``migration_role_spaces.rst``; this page focuses on the preferred public API.

Volume
------

.. autoclass:: fluxfem.NamedSpace

.. autoclass:: fluxfem.LinearSpaces

.. autoclass:: fluxfem.BilinearSpaces

.. autoclass:: fluxfem.ResidualSpaces

.. autoclass:: fluxfem.JacobianSpaces

.. autofunction:: fluxfem.assemble_linear_form

.. autofunction:: fluxfem.assemble_bilinear_form

.. autofunction:: fluxfem.assemble_residual

.. autofunction:: fluxfem.assemble_jacobian

Mixed
-----

.. autoclass:: fluxfem.MixedSpaces

Contact
-------

.. autoclass:: fluxfem.ContactSpaces

.. autoclass:: fluxfem.ContactGroupSpaces

.. autoclass:: fluxfem.OneSidedContactSpaces

Sparse operators
----------------

.. autoclass:: fluxfem.FluxSparseOperator

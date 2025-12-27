Concept
=======

FluxFEM is a weak-form-centric, differentiable finite element framework built on JAX.
The project name is FluxFEM, and the Python package name is ``fluxfem``. It provides
a compact, expressive way to define variational forms and assemble finite element
systems while remaining compatible with automatic differentiation.

At a high level, FluxFEM focuses on:

- writing weak forms close to their mathematical definitions
- compiling weak-form expressions into element kernels
- supporting both weak-form-based and tensor-based (scikit-fem-style) assembly
- enabling linear and nonlinear analyses with JAX transformations such as ``grad``
  and ``jit``

The library provides modular tools for building finite element spaces, assembling
systems, and solving PDEs while remaining flexible enough for rapid experimentation.

Contributing
============

FluxFEM is an open-source project, and we warmly welcome community contributions.

- Open an issue to discuss ideas, report bugs, or suggest improvements.
- Submit a pull request for fixes, enhancements, or new examples.

Please feel free to join the discussion on GitHub Issues:
https://github.com/kevin-tofu/fluxfem/issues

We appreciate all forms of contributions, from small documentation improvements to
larger feature proposals.

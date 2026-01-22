Chunked Assembly
================

Large meshes can cause JAX to recompile a new kernel every time the global
assembly trim of data changes (e.g., the last chunk is smaller than the
others). Chunked assembly puts a ceiling on the per-call compilation shape by
splitting the mesh into ``n_chunks`` pieces and processing each chunk with a
fixed-size, padded batch.

.. contents::
   :local:
   :depth: 2

Why chunked assembly?
---------------------

- **Stable JIT shapes:** Each chunk is padded to a constant ``chunk_size`` so JAX
  sees the same array dimensions during tracing, rather than a barely smaller
  last element batch that would trigger recompilation.
- **Smaller working sets:** Chunking keeps the per-call intermediate footprint
  bounded, which can reduce memory pressure when assembling very large meshes
  (or running batched ``jit``+``vmap`` loops).
- **Optional tracing hints:** `pad_trace=True` emits statistics about the chosen
  chunk layout, which helps tune ``n_chunks`` for your problem.

How to use ``n_chunks``
-----------------------

The ``n_chunks`` argument is supported by the functional APIs
``fluxfem.core.assemble_bilinear_form``, ``assemble_linear_form``,
``assemble_residual``, ``assemble_jacobian``, and the ``space.assemble_*``/
``space.assemble`` helpers. Pass a positive integer to request chunking:

.. code-block:: python

   K = space.assemble(
       ff.diffusion_form,
       params=1.0,
       n_chunks=16,
       pad_trace=True,
   )

FluxFEM automatically chooses ``chunk_size = ceil(n_elems / n_chunks)``
and rounds up the padded element data to the nearest multiple of that size.
During chunked assembly the padded elements are evaluated but never used in the
final sparse matrix because the final data vector is sliced back to
``n_elems * n_ldofs^2`` (or ``n_elems * n_ldofs`` for RHS assembly).

Best practices
--------------

- **Pick powers of two when possible;** they keep ``chunk_size`` regular and mesh
  partitions predictable.
- **Keep ``n_chunks`` ≤ ``n_elems``;** the helper raises if you pass a larger
  value.
- **Combine with ``pad_trace=True`` when tuning.** The trace output reports how
  many elements were padded so you can balance between padding overhead and
  JIT stability.
- **Use the same ``n_chunks`` for residual/Jacobian pairs** so the sparsity
  pattern remains stable for nonlinear solves.

Chunked assembly is a lightweight way to improve JAX stability without rewriting
your weak form or introducing new scalar-friendly kernels. When ``n_chunks`` is
``None`` (the default) the entire mesh is assembled at once, so you only need
to set the option once you see JIT recompile churn or trace warnings for
variable-sized inputs.

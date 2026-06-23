from __future__ import annotations

import jax.numpy as jnp


class DenseBasis:
    """Small dense projection basis used by tutorial ROM examples."""

    def __init__(self, basis):
        self.basis = jnp.asarray(basis, dtype=jnp.float64)

    @property
    def n_full(self):
        return int(self.basis.shape[0])

    @property
    def n_reduced(self):
        return int(self.basis.shape[1])

    def expand(self, q):
        return self.basis @ q

    def project_vector(self, vector):
        return self.basis.T @ jnp.asarray(vector)

    def project_matrix(self, matrix):
        dense = matrix.to_dense() if hasattr(matrix, "to_dense") else matrix
        return self.basis.T @ jnp.asarray(dense) @ self.basis


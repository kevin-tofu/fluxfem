import jax
import jax.numpy as jnp
import numpy as np


def default_dtype() -> jnp.dtype:
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


DEFAULT_DTYPE = default_dtype()
INDEX_DTYPE = jnp.int64 if jax.config.read("jax_enable_x64") else jnp.int32
NP_INDEX_DTYPE = np.int64 if jax.config.read("jax_enable_x64") else np.int32

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
INDEX_DTYPE = jnp.int64
NP_INDEX_DTYPE = np.int64

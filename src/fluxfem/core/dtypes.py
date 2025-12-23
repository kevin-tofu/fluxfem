import jax
import jax.numpy as jnp

DEFAULT_DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

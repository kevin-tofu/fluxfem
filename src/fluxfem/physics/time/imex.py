import numpy as np
import jax.numpy as jnp

from ...core.assembly import assemble_mass_matrix
from ...core.forms import FormContext
from ...core.space import FESpace
from ...physics.diffusion import diffusion_form
from ...core import spdirect_solve_cpu

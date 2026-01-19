import os
import sys
from datetime import datetime

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "src")))
sys.path.insert(0, os.path.abspath('../../src'))

project = "FluxFEM"
author = "Kohei Watanabe"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.8"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_sitemap",
    "myst_parser",
]

napoleon_use_ivar = True

autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "jaxlib.gpu_sparse",
    "jax_cuda12_plugin",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = ["extra"]
html_meta = {
    "description": "FluxFEM is a finite element toolkit built on JAX.",
    "keywords": "FEM, JAX, scientific computing",
    "author": "Kohei Watanabe",
    "robots": "index, follow",
}

html_baseurl = "https://kevin-tofu.github.io/fluxfem/"

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    # RTD
    html_baseurl = "https://fluxfem.readthedocs.io/en/latest/"
else:
    # GitHub Pages
    html_baseurl = "https://kevin-tofu.github.io/fluxfem/"

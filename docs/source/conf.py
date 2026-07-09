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
release = "0.3.10"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]

if os.environ.get("FLUXFEM_DOCS_DISABLE_SITEMAP") != "1":
    extensions.append("sphinx_sitemap")

napoleon_use_ivar = True

autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "jaxlib.gpu_sparse",
    "jax_cuda12_plugin",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_align": "content",
    "navigation_with_keys": True,
    "show_prev_next": True,
    "search_bar_text": "Search the docs...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/kevin-tofu/fluxfem",
            "icon": "fab fa-github",
        },
    ],
    "collapse_navigation": True,
}
base_dir = os.path.dirname(__file__)
static_dir = os.path.join(base_dir, "_static")
extra_dir = os.path.join(base_dir, "extra")
html_static_path = ["_static"] if os.path.isdir(static_dir) else []
html_extra_path = ["extra"] if os.path.isdir(extra_dir) else []
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

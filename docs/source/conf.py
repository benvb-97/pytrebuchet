# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir))  # Adjust the path to include the project root

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pytrebuchet"
copyright = "2026, Ben Van Bavel"
author = "Ben Van Bavel"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google and Numpy style docstrings
    "nbsphinx",  # For Jupyter notebook integration
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "private-members": True,
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

# -- nbsphinx configuration --------------------------------------------------
# Execute notebooks before conversion (can be overridden by environment variable)
nbsphinx_execute = "auto"  # 'auto', 'always', or 'never'

# Allow errors in notebook execution (useful during development)
nbsphinx_allow_errors = False

# Timeout for each notebook cell (in seconds)
nbsphinx_timeout = 180

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

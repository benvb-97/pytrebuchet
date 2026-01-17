# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
from pathlib import Path
from inspect import getsourcefile

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir))  # Adjust the path to include the project root
docs_dir = Path(__file__).parent.resolve()

# Get path to directory containing this file, conf.py.

def ensure_pandoc_installed(_) -> None:
    """Ensure that pandoc is installed, downloading it if necessary."""
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = docs_dir / "bin"
    # Add dir containing pandoc binary to the PATH environment variable
    if str(pandoc_dir) not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + str(pandoc_dir)
    pypandoc.ensure_pandoc_installed(
        targetfolder=str(pandoc_dir),
        delete_installer=True,
    )


def setup(app) -> None:
    """Connect the ensure_pandoc_installed function to the Sphinx build process."""
    app.connect("builder-inited", ensure_pandoc_installed)

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

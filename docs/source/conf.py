# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("./"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SPINE"
copyright = "2025, DeepLearningPhysics Collaboration"
author = "DeepLearningPhysics Collaboration"

# Get version from spine package
try:
    from spine.version import __version__

    release = __version__
    version = __version__
except ImportError:
    release = "0.1.0"
    version = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": True,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock imports for optional dependencies that may not be available during doc build
autodoc_mock_imports = [
    "larcv",
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_cluster",
    "torch_sparse",
    "MinkowskiEngine",
    "MinkowskiFunctional",
    "MinkowskiNonlinearity",
    "networkx",
    "matplotlib",
    "plotly",
    "seaborn",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "show_toc_level": 2,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Napoleon custom sections
napoleon_custom_sections = [
    "Shapes",
    ("Configuration", "params_style"),
    ("Output", "params_style"),
]

autosectionlabel_prefix_document = True

master_doc = "index"

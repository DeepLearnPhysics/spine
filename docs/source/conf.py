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
copyright = "2024, DeepLearningPhysics Collaboration"
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
    "sphinx.ext.autosummary",
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
    "undoc-members": False,
    "private-members": False,
    "exclude-members": "__weakref__, __dataclass_fields__, __dataclass_params__, __dataclass_transform__, __post_init__, __match_args__, __init__",
}

# Autosummary settings for automatic API generation
autosummary_generate = True
autosummary_imported_members = False

# Show all inherited members in docs
autodoc_inherit_docstrings = True

# Show only class docstring, not __init__ (since dataclass attributes are already documented)
autoclass_content = "class"

# Mock imports for optional dependencies that may not be available during doc build
# These packages need to be installed separately by users for full functionality
autodoc_mock_imports = [
    "larcv",
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_cluster",
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
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "logo_only": True,
    "version_selector": True,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/img/spine-logo-dark.png"

# The name of an image file (within the static path) to use as favicon of the docs
html_favicon = "_static/img/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS files
html_css_files = [
    "css/custom.css",
]

# Custom JavaScript files
html_js_files = [
    "js/version.js",
]

# Napoleon custom sections
napoleon_custom_sections = [
    "Shapes",
    ("Configuration", "params_style"),
    ("Output", "params_style"),
]

# Numpydoc settings
numpydoc_xref_param_type = False

autosectionlabel_prefix_document = True

master_doc = "index"

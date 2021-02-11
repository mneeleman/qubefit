# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sphinx_rtd_theme
import os
import sys
import glob
import subprocess
sys.path.insert(0, os.path.abspath('.'))
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("qubefit").version
except DistributionNotFound:
    __version__ = "unknown version"

# -- Project information -----------------------------------------------------

project = 'qubefit'
copyright = '2020-2021, Marcel Neeleman'
author = 'Marcel Neeleman'

# The full version, including alpha/beta/rc tags
release = __version__
version = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.imgmath',
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ".rst"
master_doc = "index"

# Convert the tutorials
for fn in glob.glob("_static/*.ipynb"):
    name = os.path.splitext(os.path.split(fn)[1])[0]
    outfn = os.path.join("Tutorials", name + ".rst")
    print("Building {0}...".format(name))
    subprocess.check_call(
        "jupyter nbconvert --to rst "
        + fn
        + " --output-dir tutorials",
        shell=True,
    )

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = 'sphinx_rtd_theme'

# html_sidebars = {
#    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html']
# }
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_favicon = "QubeFitLogo.png"
html_logo = "./Fig/QubeFitLogoText.png"
# html_theme_options = {"logo_only": True}

imgmath_image_format = 'svg'

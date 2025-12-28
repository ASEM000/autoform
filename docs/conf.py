# autoform documentation

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "autoform"
copyright = "2026, Mahmoud Asem"
author = "Mahmoud Asem"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
master_doc = "index"

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Napoleon settings
napoleon_google_docstring = True

# nbsphinx settings
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# Custom CSS
html_css_files = ["custom.css"]

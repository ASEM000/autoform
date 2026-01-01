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
    "sphinx.ext.intersphinx",  # link to external docs
    "sphinx_autodoc_typehints",  # better type annotations
    "sphinx_copybutton",  # copy button on code blocks
    "sphinx_design",  # grids, cards, tabs, dropdowns
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "examples/*.py",  # use .ipynb only
    "conf.py",  # config, not docs
]
master_doc = "index"
suppress_warnings = ["epub.duplicated_toc_entry"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_toc_level": 3,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "repository_url": "https://github.com/ASEM000/autoform",
    "use_repository_button": True,
    "use_issues_button": True,
    "collapse_navigation": False,
}

# Napoleon settings
napoleon_google_docstring = True

# autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
typehints_defaults = "braces"
simplify_optional_unions = True
always_use_bars_union = True

# Simplify complex type displays
autodoc_type_aliases = {
    "P": "P",
    "R": "R",
    "ParamSpec": "...",
    "Tree": "Tree",
    "Collected": "dict",
}

# intersphinx links
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# nbsphinx settings
html_css_files = ["_static/custom.css"]
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_codecell_lexer = "python3"

# copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

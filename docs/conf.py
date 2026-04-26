# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# autoform documentation

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

project = "autoform"
copyright = "2026, Mahmoud Asem"
author = "Mahmoud Asem"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",  # link to external docs
    "sphinx_copybutton",  # copy button on code blocks
    "sphinx_design",  # grids, cards, tabs, dropdowns
    "myst_parser",
    "simple_mermaid",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "conf.py",  # config, not docs
]
master_doc = "index"
suppress_warnings = ["epub.duplicated_toc_entry"]
myst_fence_as_directive = ["mermaid"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "autoform"
html_short_title = "autoform"
html_theme_options = {
    "source_repository": "https://github.com/ASEM000/autoform/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        "color-brand-primary": "#0051ff",
        "color-brand-content": "#0051ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7aa2ff",
        "color-brand-content": "#7aa2ff",
    },
}
pygments_style = "default"
pygments_dark_style = "monokai"

# Napoleon settings
napoleon_google_docstring = True

# autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
typehints_defaults = "braces"
simplify_optional_unions = False
always_use_bars_union = True

# intersphinx links
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# nbsphinx settings
html_css_files = ["custom.css"]
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_codecell_lexer = "python3"

# copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

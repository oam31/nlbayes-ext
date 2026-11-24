# Configuration de la documentation Sphinx de nlbayes-ext.
#
# Sources : les documents techniques en Markdown (MyST) dans technique/,
# le CHANGELOG de la racine (inclus par changelog.md) et la référence de
# l'API, générée par sphinx-autoapi dans api/ (non versionnée).
# Construction locale : `make html` dans ce répertoire → _build/html.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from nlbayes_ext import __version__ as _version  # noqa: E402

project = "nlbayes-ext"
copyright = "2026, Olivier Amram"
author = "Olivier Amram"
release = _version
version = _version

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "REFERENCE_TECHNIQUE.md"]

language = "fr"

# -- MyST ----------------------------------------------------------------------
# Ancres des titres identiques à celles de GitHub : les liens internes des
# documents fonctionnent dans les deux rendus.
myst_heading_anchors = 5
myst_enable_extensions = ["dollarmath", "amsmath", "deflist"]

# -- Sortie HTML ---------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- API (sphinx-autoapi) ------------------------------------------------------
autoapi_dirs = ["../src"]
autoapi_type = "python"
autoapi_root = "api"
autoapi_keep_files = False
autodoc_typehints = "signature"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

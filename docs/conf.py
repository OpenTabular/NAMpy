# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime
from pathlib import Path

# Keep Matplotlib cache writes out of user home directories in local and CI builds.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nampy-matplotlib")

# Add the project root independently of the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the package to get version
import nampy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nampy"
copyright = (
    f"{datetime.now().year}, "
    "Ananyapam De, Anton Thielmann, and OpenTabular Contributors"
)
author = "Ananyapam De, Anton Thielmann"
release = nampy.__version__
version = nampy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

extensions = [
    "public_reference",
    "feature_reference",
    "site_integrity",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",  # For Markdown support
    "tutorial_downloads",
    "nbsphinx",  # Render checked-in notebook sources (via nbconvert/Pandoc)
    "sphinx_design",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_kernel_name = "python3"
nbsphinx_prolog = r"""
:download:`Download notebook <{{ env.docname.split('/')[-1] ~ '.ipynb' }}>`
"""
highlight_language = "python3"  # Use python3 for syntax highlighting
pygments_style = "default"
pygments_style_dark = "monokai"

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    # Exclude sklearn internal methods that cause duplicate target errors
    "exclude-members": "__weakref__,__init_subclass__,set_decision_function_request,set_fit_request,set_predict_proba_request,set_predict_request,set_score_request,set_transform_request,get_metadata_routing",
    "inherited-members": False,  # Don't document inherited sklearn methods
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = False  # Don't inherit sklearn docstrings

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "README.md",
    "Thumbs.db",
    ".DS_Store",
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Ignore certain docstring issues from sklearn
nitpicky = False
nitpick_ignore = [
    ("py:class", "sklearn.utils._metadata_requests._MetadataRequester"),
    ("py:class", "sklearn.utils._metadata_requests.RequestMethod"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "collapse_navigation": False,
    "logo_only": False,
    "navigation_depth": 4,
    "titles_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "sticky_navigation": False,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ["_static"]
html_css_files = ["sidebar.css"]
html_js_files = [
    "page-navigation.js",
    ("contributors.js", {"defer": "defer"}),
    ("site-navigation.js", {"defer": "defer"}),
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo-sidebar-refined.svg"
html_title = "NAMpy"
html_last_updated_fmt = "%Y-%m-%d"

# The name of an image file (within the static path) to use as favicon
html_favicon = "_static/logo-sidebar-refined.svg"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = "NAMpydoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "NAMpy.tex",
        "NAMpy Documentation",
        author,
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "nampy", "NAMpy Documentation", [author], 1)]

# Include rendered notebook pages in EPUB while omitting their raw source files.
epub_exclude_files = [
    path.relative_to(Path(__file__).parent).as_posix()
    for folder in ("notebooks", "development/notebooks")
    for path in (Path(__file__).parent / folder).glob("*.ipynb")
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "NAMpy",
        "NAMpy Documentation",
        author,
        "NAMpy",
        "Interpretable (Additive) Tabular Deep Learning",
        "Miscellaneous",
    ),
]


# Every HTML sidebar contains the global navigation tree. Re-render all pages
# after an environment update so incremental builds cannot retain old menus.
def _refresh_html_navigation(app, env):
    if app.builder.format == "html":
        # Viewcode otherwise retains source pages whose Python source is unchanged,
        # even when their shared navigation has changed.
        for page in (Path(app.outdir) / "_modules").rglob("*.html"):
            page.unlink()
        (Path(app.outdir) / "py-modindex.html").unlink(missing_ok=True)
        return sorted(env.found_docs)
    return []


def setup(app):
    app.connect("env-updated", _refresh_html_navigation)

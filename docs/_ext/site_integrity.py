"""Stable legacy URLs and visible documentation provenance."""

import posixpath
import subprocess
from pathlib import Path

REDIRECTS = {
    "quickstart": "notebooks/01_getting_started",
    "overview": "index#additive-models",
    "installation": "faq#installation-and-setup",
    "examples/index": "howto/index",
    "user_guide": "howto/index",
    "notebook_tutorials": "tutorials/index",
    "tutorials/monotone_effect": "tutorials/first_gam#constrain-a-smooth-with-prior-knowledge",
    "tutorials/interaction_surface": "tutorials/first_gam#add-a-joint-spatial-effect",
    "tutorials/periodic_effect": "tutorials/first_gam#interpret-the-effects",
    "tutorials/nam_nbm": "tutorials/gam_nam#compare-independent-networks-with-shared-bases",
}


def pages(app):
    for old, target in REDIRECTS.items():
        path, _, fragment = target.partition("#")
        relative = posixpath.relpath(path, posixpath.dirname(old) or ".") + ".html"
        if fragment:
            relative += "#" + fragment
        yield (
            old,
            {
                "redirect_url": relative,
                "redirect_title": path.rsplit("/", 1)[-1].replace("_", " ").title(),
            },
            "redirect.html",
        )


def provenance(app):
    from nampy.__version__ import __version__

    root = Path(app.srcdir).parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
        ).strip()
        if subprocess.call(["git", "diff", "--quiet"], cwd=root) != 0:
            commit += "-dirty"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    app.config.html_context.update(nampy_version=__version__, nampy_commit=commit)


def setup(app):
    app.connect("html-collect-pages", pages)
    app.connect("html-page-context", navigation_context)
    app.connect("builder-inited", provenance)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def navigation_context(app, pagename, templatename, context, doctree):
    """Keep generated symbols on reference pages, outside the global sidebar."""
    from docutils import nodes
    from sphinx.environment.adapters.toctree import TocTree

    def navigation(**kwargs):
        collapse = kwargs.pop("collapse", False)
        tree = TocTree(app.env).get_toctree_for(
            pagename, app.builder, collapse, **kwargs
        )
        if tree is None:
            return ""
        for reference in list(tree.findall(nodes.reference)):
            uri = reference.get("refuri", "")
            if "generated/" not in uri and not reference.astext().startswith("nampy."):
                continue
            parent = reference.parent
            while parent is not None and not isinstance(parent, nodes.list_item):
                parent = parent.parent
            if parent is not None and parent.parent is not None:
                parent.parent.remove(parent)
        return app.builder.render_partial(tree)["fragment"]

    context["toctree"] = navigation

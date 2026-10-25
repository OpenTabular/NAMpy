"""Stable legacy URLs and visible documentation provenance."""

import posixpath
import subprocess
from pathlib import Path

REDIRECTS = {
    "examples/basic_regression": "examples/fit_models#predict-a-continuous-response",
    "examples/basic_classification": "examples/fit_models#predict-class-probabilities",
    "examples/distributional_regression": "examples/fit_models#predict-a-response-distribution",
    "howto/monotonicity": "user_guide/shape_constrained_gams",
    "tutorials/modeling_workflows": "index#quick-start-fitting-a-model",
    "howto/interaction_selection": "tutorials/interaction_selection",
    "tutorials/changing_variability": "notebooks/04_distributional_models",
    "in_depth/index": "tutorials/index",
    "concepts/index": "howto/index",
    "concepts/additive_predictors": "user_guide/interpretability#additive-predictor-interpretation",
    "concepts/smoothness": "user_guide/statistical_gams#smoothness-and-basis-dimension",
    "concepts/distributional_regression": "user_guide/gamlss#distributional-model-concepts",
    "quickstart": "notebooks/01_getting_started",
    "overview": "index#additive-models",
    "installation": "faq#installation-and-setup",
    "examples/index": "howto/index",
    "user_guide": "howto/index",
    "notebook_tutorials": "tutorials/index",
    "tutorials/monotone_effect": "user_guide/shape_constrained_gams#compare-an-unconstrained-and-monotone-fit",
    "tutorials/interaction_surface": "user_guide/formula_visual_reference#add-a-joint-spatial-effect",
    "tutorials/periodic_effect": "tutorials/first_gam#interpret-the-effects",
    "tutorials/nam_nbm": "notebooks/03_neural_models#Fit-NAM-and-NBM",
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
    app.connect("doctree-resolved", notebook_image_descriptions)
    app.connect("build-finished", version_local_images)
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


def notebook_image_descriptions(app, doctree, docname):
    """Use authored descriptions instead of nbsphinx-generated filenames."""
    import json

    from docutils import nodes

    path = Path(app.srcdir) / (docname + ".ipynb")
    if not path.is_file():
        return
    notebook = json.loads(path.read_text())
    descriptions = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if any(
                mime in output.get("data", {})
                for mime in ["image/png", "image/svg+xml", "image/jpeg"]
            ):
                descriptions.append(cell.get("metadata", {}).get("figure_alt", ""))
    for image, description in zip(
        doctree.findall(nodes.image), descriptions, strict=False
    ):
        if description:
            image["alt"] = description


def version_local_images(app, exception):
    """Give regenerated image URLs content keys, including their full-size links."""
    if exception is not None or app.builder.format != "html":
        return
    import html
    import re
    import zlib
    from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

    root = Path(app.outdir).resolve()
    checksums = {}
    attributes = re.compile(
        r"(?P<prefix>\b(?:src|href)=[\"'])(?P<url>[^\"']+)(?P<suffix>[\"'])"
    )
    for page in root.rglob("*.html"):

        def replace(match, page=page):
            parts = urlsplit(html.unescape(match["url"]))
            if parts.scheme or parts.netloc:
                return match[0]
            target = (page.parent / unquote(parts.path)).resolve()
            if (
                not target.is_relative_to(root)
                or not target.is_file()
                or target.suffix.lower()
                not in {".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif"}
            ):
                return match[0]
            if target not in checksums:
                checksums[target] = f"{zlib.crc32(target.read_bytes()):08x}"
            query = [
                (key, value) for key, value in parse_qsl(parts.query) if key != "v"
            ]
            query.append(("v", checksums[target]))
            url = urlunsplit(parts._replace(query=urlencode(query)))
            return match["prefix"] + html.escape(url, quote=True) + match["suffix"]

        original = page.read_text()
        updated = attributes.sub(replace, original)
        if updated != original:
            page.write_text(updated)

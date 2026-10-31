"""Stable legacy URLs and visible documentation provenance."""

import json
import posixpath
import subprocess
from pathlib import Path

REDIRECTS = {
    "statistical_reference/vb_corr": "statistical_reference/internals/covariance_correction",
    "development/mgcv_optimizer_boundaries": "compatibility#guarded-boundaries",
    "development/validation_reports": "architecture#detailed-evidence",
    "development/documentation_validation": "contributing#improve-documentation",
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
REDIRECTS.update(
    json.loads(
        (Path(__file__).parents[1] / "_data/legacy_documentation.json").read_text()
    )
)


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
    app.connect("build-finished", remove_unused_utility_pages, priority=100)
    app.connect("html-collect-pages", pages)
    app.connect("html-page-context", navigation_context)
    app.connect("missing-reference", legacy_reference)
    app.connect("doctree-resolved", legacy_links)
    app.connect("builder-inited", provenance)
    app.connect("doctree-resolved", local_documentation_links)
    app.connect("doctree-resolved", notebook_image_descriptions)
    app.connect("doctree-resolved", training_anchor_alias)
    app.connect("doctree-resolved", paginate_notebook_outputs)
    app.connect("build-finished", version_local_images)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def paginate_notebook_outputs(app, doctree, docname):
    """Keep large notebook text outputs within TeX's framed-box limits."""
    if app.builder.format != "latex":
        return
    from docutils import nodes

    blocks = list(doctree.findall(nodes.literal_block))
    # nbsphinx stores text/plain outputs as raw HTML, LaTeX and text siblings.
    # Use its original text sibling so Sphinx escapes each new block correctly.
    blocks += [
        block
        for block in doctree.findall(nodes.raw)
        if block.get("format") == "latex"
        and block.astext().startswith(r"\begin{sphinxVerbatim}")
    ]
    for block in blocks:
        parent = block.parent
        while parent is not None and "nboutput" not in parent.get("classes", []):
            parent = parent.parent
        text = block.astext()
        if isinstance(block, nodes.raw):
            text_node = next(
                (
                    n for n in block.parent.children
                    if isinstance(n, nodes.raw) and n.get("format") == "text"
                ),
                None,
            )
            if text_node is None:
                continue
            text = text_node.astext()
        if parent is None or (len(text) <= 4000 and len(text.splitlines()) <= 40):
            continue
        parts = []
        while text:
            # Prefer complete lines, but also bound single very wide table rows.
            end = min(len(text), 4000)
            lines = text[:end].splitlines(keepends=True)
            if len(lines) > 40:
                end = sum(len(line) for line in lines[:40])
            elif end < len(text):
                end = text.rfind("\n", 0, end) + 1 or end
            chunk, text = text[:end], text[end:]
            part = (
                nodes.literal_block(language="none")
                if isinstance(block, nodes.raw) else block.deepcopy()
            )
            part.children = []
            part += nodes.Text(chunk)
            part.rawsource = chunk
            if parts:
                part["ids"], part["names"] = [], []
            parts.append(part)
        block.replace_self(parts)


def remove_unused_utility_pages(app, exception):
    """Remove disabled general/module indexes while preserving Sphinx search."""
    if exception is None and app.builder.format == "html":
        for filename in ("genindex.html", "py-modindex.html"):
            (Path(app.outdir) / filename).unlink(missing_ok=True)


def navigation_context(app, pagename, templatename, context, doctree):
    """Show API detail pages with concise labels in their owning section."""
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
            target = uri.split("#", 1)[0]
            path = (
                posixpath.normpath(posixpath.join(posixpath.dirname(pagename), target))
                if target
                else pagename
            ).lstrip("/")
            if not path.startswith(
                "api/generated/"
            ) and not reference.astext().startswith("nampy."):
                continue
            reference.children[:] = [nodes.Text(reference.astext().rsplit(".", 1)[-1])]
        return app.builder.render_partial(tree)["fragment"]

    context["toctree"] = navigation


def legacy_reference(app, env, node, contnode):
    """Resolve historical document and Python references to their current owners."""
    from sphinx.util.nodes import make_refnode

    source = node.get("refdoc", env.docname)
    name = node.get("reftarget", "")
    target = None
    if node.get("refdomain") == "std" and node.get("reftype") == "doc":
        old = (
            name.lstrip("/")
            if name.startswith("/")
            else posixpath.normpath(posixpath.join(posixpath.dirname(source), name))
        )
        target = REDIRECTS.get(old)
    elif node.get("refdomain") == "py":
        for old, destination in REDIRECTS.items():
            if not old.startswith("api/generated/"):
                continue
            symbol = old.removeprefix("api/generated/")
            if (
                name == symbol
                or name == symbol.rsplit(".", 1)[-1]
                or name.startswith(symbol + ".")
            ):
                target = destination
                break
    if target:
        page, _, anchor = target.partition("#")
        return make_refnode(app.builder, source, page, anchor, contnode)


def legacy_links(app, doctree, docname):
    """Keep internal links on canonical pages while preserving old external URLs."""
    from docutils import nodes

    for reference in doctree.findall(nodes.reference):
        uri = reference.get("refuri", "")
        if not uri or ":" in uri or uri.startswith("#"):
            continue
        old = posixpath.normpath(
            posixpath.join(posixpath.dirname(docname), uri.split("#", 1)[0])
        ).removesuffix(".html")
        if old not in REDIRECTS:
            continue
        page, separator, anchor = REDIRECTS[old].partition("#")
        reference["refuri"] = app.builder.get_relative_uri(docname, page) + (
            "#" + anchor if separator else ""
        )


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
            if posixpath.splitext(unquote(parts.path))[1].lower() not in {
                ".png",
                ".svg",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
            }:
                return match[0]
            target = (page.parent / unquote(parts.path)).resolve()
            if not target.is_relative_to(root) or not target.is_file():
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


def training_anchor_alias(app, doctree, docname):
    """Keep viewcode backlinks valid when Sphinx uses the public module alias."""
    if docname != "api/neural_training":
        return
    from sphinx import addnodes

    canonical = "nampy.models._neural.NeuralEstimatorBase.fit"
    alias = "nampy.models.NeuralEstimatorBase.fit"
    for signature in doctree.findall(addnodes.desc_signature):
        if canonical in signature["ids"] and alias not in signature["ids"]:
            signature["ids"].append(alias)


def local_documentation_links(app, doctree, docname):
    """Keep canonical documentation links inside the current build/version.

    Shared Markdown files use published URLs so they also work on GitHub.
    In the built docs, those URLs must not switch a preview or version to
    another site's navigation. Other hosts and explicit versions stay intact.
    """
    from urllib.parse import unquote, urlsplit

    from docutils import nodes

    prefix = "/en/latest/"
    for reference in doctree.findall(nodes.reference):
        parts = urlsplit(reference.get("refuri", ""))
        if parts.netloc != "nampy.readthedocs.io" or not parts.path.startswith(prefix):
            continue
        path = unquote(parts.path[len(prefix) :])
        target = path[:-5] if path.endswith(".html") else path.rstrip("/") + "/index"
        if target == "/index":
            target = "index"
        if target not in app.env.found_docs:
            continue
        uri = app.builder.get_relative_uri(docname, target)
        if parts.query:
            uri += "?" + parts.query
        if parts.fragment:
            uri += "#" + parts.fragment
        reference["refuri"] = uri
        reference["internal"] = True

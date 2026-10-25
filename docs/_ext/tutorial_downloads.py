"""Offer notebook exports for the walkthroughs authored in reStructuredText."""

import base64
import mimetypes
import posixpath
from pathlib import Path

import nbformat
from docutils import nodes
from sphinx import addnodes
from sphinx.util.docutils import SphinxRole

WALKTHROUGHS = {
    "tutorials/first_gam",
    "tutorials/gam_nam",
    "tutorials/changing_variability",
    "tutorials/modeling_workflows",
    "in_depth/tree_effects",
    "in_depth/quantile_calibration",
}


class DownloadReference(nodes.reference):
    """A link that explicitly requests a file download in the browser."""


def visit_download(self, node):
    self.body.append(self.starttag(node, "a", "", href=node["refuri"], download=""))


def depart_download(self, node):
    self.body.append("</a>")


class NotebookDownload(SphinxRole):
    def run(self):
        target = posixpath.normpath(
            posixpath.join(posixpath.dirname(self.env.docname), self.text)
        )
        if target not in WALKTHROUGHS:
            return [], [self.inliner.reporter.error(f"Unknown notebook: {target}")]
        return [
            DownloadReference("", "Download notebook", refuri="notebook:" + target)
        ], []


def export_notebook(app, doctree, docname):
    if app.builder.name not in {"html", "dirhtml", "singlehtml"}:
        for reference in list(doctree.findall(nodes.reference)):
            if isinstance(reference, DownloadReference) or (
                isinstance(reference, addnodes.download_reference)
                and reference.get("reftarget", "").endswith(".ipynb")
            ):
                reference.parent.remove(reference)
        return
    for reference in doctree.findall(nodes.reference):
        uri = reference.get("refuri", "")
        if uri.startswith("notebook:"):
            reference["refuri"] = download_url(docname, uri.removeprefix("notebook:"))
            reference["classes"].extend(["reference", "download"])
            reference["download"] = ""
    if docname not in WALKTHROUGHS:
        return
    cells = []

    def visit(node, level=1):
        if isinstance(node, nodes.section):
            for child in node.children:
                visit(child, level + 1)
        elif isinstance(node, nodes.title):
            cells.append(
                nbformat.v4.new_markdown_cell(
                    "#" * max(1, level - 1) + " " + node.astext()
                )
            )
        elif isinstance(node, nodes.literal_block):
            language = node.get("language", "")
            if language in {"python", "py", "python3"}:
                cells.append(nbformat.v4.new_code_cell(node.astext()))
            else:
                cells.append(
                    nbformat.v4.new_markdown_cell(
                        f"```{language}\n{node.astext()}\n```"
                    )
                )
        elif isinstance(node, nodes.image):
            uri = node.get("uri", "")
            source = Path(app.srcdir) / uri.lstrip("/")
            if not source.is_file():
                source = Path(app.srcdir) / posixpath.dirname(docname) / uri
            if source.is_file():
                name = source.name
                mime = mimetypes.guess_type(name)[0] or "image/png"
                cell = nbformat.v4.new_markdown_cell(
                    f"![{node.get('alt', '')}](attachment:{name})"
                )
                cell["attachments"] = {
                    name: {mime: base64.b64encode(source.read_bytes()).decode()}
                }
                cells.append(cell)
        elif isinstance(node, nodes.table):
            cells.append(
                nbformat.v4.new_markdown_cell(
                    app.builder.render_partial(node)["fragment"]
                )
            )
        elif isinstance(
            node,
            (
                nodes.paragraph,
                nodes.bullet_list,
                nodes.enumerated_list,
                nodes.math_block,
            ),
        ):
            if node.astext() != "Download notebook":
                text = node.astext()
                if isinstance(node, nodes.math_block):
                    text = "$$\n" + text + "\n$$"
                cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            for child in node.children:
                if isinstance(child, nodes.Element):
                    visit(child, level)

    visit(doctree)
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    target = Path(app.outdir) / "_downloads/notebooks" / (docname + ".ipynb")
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    title = next(doctree.findall(nodes.title), None)
    if title is not None:
        paragraph = nodes.paragraph()
        paragraph += DownloadReference(
            "",
            "Download notebook",
            refuri=download_url(docname, docname),
            classes=["reference", "download"],
            download="",
        )
        title.parent.insert(title.parent.index(title) + 1, paragraph)


def download_url(docname, target):
    return posixpath.relpath(
        "_downloads/notebooks/" + target + ".ipynb", posixpath.dirname(docname) or "."
    )


def exclude_epub_downloads(app, config):
    """Ignore exports left in a reused EPUB output directory as well."""
    config.epub_exclude_files = list(config.epub_exclude_files) + [
        "_downloads/notebooks/" + docname + ".ipynb" for docname in sorted(WALKTHROUGHS)
    ]


def setup(app):
    app.add_node(DownloadReference, html=(visit_download, depart_download))
    app.connect("config-inited", exclude_epub_downloads)
    app.add_role("notebook-download", NotebookDownload())
    app.connect("doctree-resolved", export_notebook)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

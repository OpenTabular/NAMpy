"""Notebook exports keep executable cells and their illustrated explanations."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from docutils import nodes
from docutils.utils import new_document

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "tutorial_downloads", ROOT / "docs/_ext/tutorial_downloads.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_walkthrough_export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "figure.png").write_bytes(b"test-image")
    document = new_document("walkthrough")
    section = nodes.section()
    section += nodes.title("", "An illustrated walkthrough")
    section += nodes.paragraph("", "Explain the fitted curve.")
    section += nodes.literal_block("", "print(1 + 2)", language="python")
    section += nodes.image(uri="figure.png", alt="Fitted curve")
    document += section
    app = SimpleNamespace(
        srcdir=source,
        outdir=tmp_path / "html",
        builder=SimpleNamespace(format="html", name="html"),
    )
    module.export_notebook(app, document, "tutorials/first_gam")
    exported = app.outdir / "_downloads/notebooks/tutorials/first_gam.ipynb"
    notebook = nbformat.read(exported, as_version=4)
    nbformat.validate(notebook)
    assert [cell.source for cell in notebook.cells if cell.cell_type == "code"] == [
        "print(1 + 2)"
    ]
    assert any(cell.get("attachments") for cell in notebook.cells)
    links = list(document.findall(nodes.reference))
    assert links[0]["refuri"] == "../_downloads/notebooks/tutorials/first_gam.ipynb"


def test_card_download_target():
    document = new_document("gallery")
    document += nodes.reference(
        "", "Download notebook", refuri="notebook:tutorials/first_gam"
    )
    app = SimpleNamespace(builder=SimpleNamespace(format="html", name="html"))
    module.export_notebook(app, document, "tutorials/index")
    reference = next(document.findall(nodes.reference))
    assert reference["refuri"] == "../_downloads/notebooks/tutorials/first_gam.ipynb"
    assert "download" in reference["classes"]


@pytest.mark.parametrize("builder", ["epub", "latex"])
def test_ebook_builds_have_no_notebook_downloads(tmp_path, builder):
    document = new_document("walkthrough")
    document += module.DownloadReference(
        "", "Download notebook", refuri="notebook:tutorials/first_gam"
    )
    document += module.addnodes.download_reference(
        "", "Download notebook", reftarget="tutorial.ipynb"
    )
    document += nodes.paragraph("", "Keep the walkthrough content.")
    app = SimpleNamespace(
        outdir=tmp_path,
        builder=SimpleNamespace(
            format="html" if builder == "epub" else "latex", name=builder
        ),
    )
    module.export_notebook(app, document, "tutorials/first_gam")
    assert not list(document.findall(nodes.reference))
    assert "Keep the walkthrough content." in document.astext()
    assert not list(tmp_path.rglob("*.ipynb"))


def test_epub_excludes_stale_exports():
    config = SimpleNamespace(epub_exclude_files=["existing-file"])
    module.exclude_epub_downloads(None, config)
    assert "existing-file" in config.epub_exclude_files
    assert set(config.epub_exclude_files[1:]) == {
        f"_downloads/notebooks/{name}.ipynb" for name in module.WALKTHROUGHS
    }

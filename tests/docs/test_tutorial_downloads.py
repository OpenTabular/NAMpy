"""Notebook exports keep executable cells and their illustrated explanations."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nbformat
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
        srcdir=source, outdir=tmp_path / "html", builder=SimpleNamespace(format="html")
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
    app = SimpleNamespace(builder=SimpleNamespace(format="html"))
    module.export_notebook(app, document, "tutorials/index")
    reference = next(document.findall(nodes.reference))
    assert reference["refuri"] == "../_downloads/notebooks/tutorials/first_gam.ipynb"
    assert "download" in reference["classes"]

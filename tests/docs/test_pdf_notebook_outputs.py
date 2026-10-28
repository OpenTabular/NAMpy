"""Long notebook evidence must survive PDF pagination without losing text."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from docutils import nodes
from docutils.utils import new_document

paginate = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "docs/_ext/site_integrity.py")
)["paginate_notebook_outputs"]


@pytest.mark.parametrize("text", ["warning details\n" * 900, "x" * 12001, "\n" * 2000])
def test_large_notebook_output_is_split_without_losing_evidence(text):
    tree = new_document("notebook")
    output = nodes.container(classes=["nboutput"])
    nested = nodes.container(classes=["output_area"])
    block = nodes.literal_block(text, text, ids=["evidence"], names=["evidence"])
    nested += block
    output += nested
    tree += output
    paginate(SimpleNamespace(builder=SimpleNamespace(format="latex")), tree, "notebook")
    blocks = list(tree.findall(nodes.literal_block))
    assert len(blocks) > 1
    assert "".join(block.astext() for block in blocks) == text
    assert all(len(block.astext()) <= 4000 for block in blocks)
    assert all(len(block.astext().splitlines()) <= 40 for block in blocks)
    assert blocks[0]["ids"] == ["evidence"]
    assert all(block["ids"] == [] and block["names"] == [] for block in blocks[1:])


@pytest.mark.parametrize("format", ["html", "epub", "latex"])
def test_short_outputs_and_code_are_preserved(format):
    tree = new_document("notebook")
    output = nodes.container(classes=["nboutput"])
    short = nodes.literal_block("value", "value")
    code = nodes.literal_block("", "print('x')\n" * 1000)
    output += short
    tree += output
    tree += code
    paginate(SimpleNamespace(builder=SimpleNamespace(format=format)), tree, "notebook")
    assert list(tree.findall(nodes.literal_block)) == [short, code]


@pytest.mark.parametrize("format", ["html", "epub"])
def test_other_formats_preserve_large_output(format):
    tree = new_document("notebook")
    output = nodes.container(classes=["nboutput"])
    block = nodes.literal_block("", "evidence\n" * 1000)
    output += block
    tree += output
    paginate(SimpleNamespace(builder=SimpleNamespace(format=format)), tree, "notebook")
    assert list(tree.findall(nodes.literal_block)) == [block]


def test_nbsphinx_raw_output_uses_original_unescaped_text():
    tree = new_document("notebook")
    output = nodes.container(classes=["nboutput"])
    text = "warning {eta} = \u03b7_1 & 100%\n" * 300
    raw_html = nodes.raw("", "<pre>HTML</pre>", format="html")
    raw_latex = nodes.raw("", r"\begin{sphinxVerbatim}escaped\end{sphinxVerbatim}", format="latex")
    raw_text = nodes.raw("", text, format="text")
    output.extend([raw_html, raw_latex, raw_text])
    tree += output
    paginate(SimpleNamespace(builder=SimpleNamespace(format="latex")), tree, "notebook")
    blocks = list(tree.findall(nodes.literal_block))
    assert len(blocks) > 1
    assert "".join(block.astext() for block in blocks) == text
    assert all(len(block.astext()) <= 4000 for block in blocks)
    assert all(len(block.astext().splitlines()) <= 40 for block in blocks)
    assert raw_html.parent is output and raw_text.parent is output
    assert raw_latex not in output.children

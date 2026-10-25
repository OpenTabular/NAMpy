"""Shared Markdown links must keep readers within the current docs build."""

import posixpath
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from docutils import nodes
from docutils.utils import new_document

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("source", ["contributing", "development/notebooks/report"])
def test_published_links_resolve_within_the_current_build(source):
    rewrite = runpy.run_path(str(ROOT / "docs/_ext/site_integrity.py"))[
        "local_documentation_links"
    ]
    tree = new_document("links")
    local = nodes.reference(
        "", "Compatibility",
        refuri="https://nampy.readthedocs.io/en/latest/compatibility.html?navigation=stable#guarded-boundaries",
    )
    preserved = [
        "https://nampy.readthedocs.io/en/stable/compatibility.html",
        "https://nampy.readthedocs.io/en/latest/unknown.html",
        "https://github.com/OpenTabular/NAMpy",
        "../compatibility.html",
    ]
    tree += local
    for url in preserved:
        tree += nodes.reference("", "Other link", refuri=url)
    app = SimpleNamespace(
        env=SimpleNamespace(found_docs={"compatibility", "index"}),
        builder=SimpleNamespace(
            get_relative_uri=lambda source, target: posixpath.relpath(
                target + ".html", posixpath.dirname(source) or "."
            )
        ),
    )
    rewrite(app, tree, source)
    expected = app.builder.get_relative_uri(source, "compatibility")
    assert local["refuri"] == expected + "?navigation=stable#guarded-boundaries"
    assert local["internal"] is True
    assert [node["refuri"] for node in list(tree.findall(nodes.reference))[1:]] == preserved

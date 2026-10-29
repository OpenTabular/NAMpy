"""Every retained document and source module has a visible navigation owner."""

import json
import posixpath
import runpy
from pathlib import Path
from types import SimpleNamespace

from docutils import nodes

ROOT = Path(__file__).resolve().parents[2]


def test_source_navigation_includes_nested_modules_and_expands_active_package():
    extension = runpy.run_path(str(ROOT / "docs/_ext/source_navigation.py"))
    modules = {
        "nampy.gam.families.inverse_gaussian": ("source", {}, {}, "module"),
        "nampy.neural.configs.nam_config": ("source", {}, {}, "module"),
        "unavailable": False,
    }
    tree = extension["module_tree"](modules)
    current = "_modules/nampy/gam/families/inverse_gaussian"

    def uri(source, target):
        return posixpath.relpath(target + ".html", posixpath.dirname(source))

    sidebar = extension["source_menu"](tree, current, uri)
    assert 'aria-current="page">inverse_gaussian</a>' in sidebar
    assert "<details open><summary>Statistical engine</summary>" in sidebar
    assert "<details><summary>Neural engine</summary>" in sidebar
    assert "unavailable" not in sidebar
    for name in modules:
        if modules[name]:
            assert uri(current, extension["source_page"](name)) in sidebar


def test_deleted_duplicate_sources_retain_canonical_redirects():
    aliases = json.loads((ROOT / "docs/_data/legacy_documentation.json").read_text())
    assert aliases
    for old, target in aliases.items():
        assert not (ROOT / "docs" / (old + ".rst")).exists(), old
        canonical = target.split("#", 1)[0]
        assert canonical not in aliases, (old, target)
        assert (ROOT / "docs" / (canonical + ".rst")).is_file(), target


def test_old_document_reference_uses_the_canonical_control_anchor():
    extension = runpy.run_path(str(ROOT / "docs/_ext/site_integrity.py"))
    app = SimpleNamespace(
        builder=SimpleNamespace(get_relative_uri=lambda a, b: b + ".html")
    )
    node = nodes.Element(
        refdoc="statistical_reference/index",
        refdomain="std",
        reftype="doc",
        reftarget="controls/epsilon",
    )
    result = extension["legacy_reference"](
        app,
        SimpleNamespace(docname="statistical_reference/index"),
        node,
        nodes.Text("coefficient tolerance"),
    )
    assert (
        result["refuri"]
        == "statistical_reference/controls/coefficient_fitting.html#control-epsilon"
    )


def test_html_audit_collects_sidebar_caption_and_source_links():
    extension = runpy.run_path(str(ROOT / "docs/scripts/check_documentation.py"))
    page = extension["Page"](
        '<nav class="wy-menu-vertical"><p><a href="api/index.html">Reference</a></p>'
        "<ul><li><details><summary>Statistical engine</summary><ul><li>"
        '<a href="_modules/nampy/gam/families/inverse_gaussian.html">inverse_gaussian</a>'
        '</li></ul></details></li></ul></nav><a href="outside.html">Outside</a>'
    )
    assert page.nav_links == [
        "api/index.html",
        "_modules/nampy/gam/families/inverse_gaussian.html",
    ]


def test_successful_html_build_removes_stale_utility_pages(tmp_path):
    extension = runpy.run_path(str(ROOT / "docs/_ext/site_integrity.py"))
    removed = ("genindex.html", "py-modindex.html", "search.html", "searchindex.js")
    for filename in (*removed, "index.html"):
        (tmp_path / filename).write_text("stale content")
    app = SimpleNamespace(outdir=tmp_path, builder=SimpleNamespace(format="html"))
    cleanup = extension["remove_unused_utility_pages"]
    cleanup(app, RuntimeError("build failed"))
    assert all((tmp_path / filename).exists() for filename in removed)
    cleanup(app, None)
    assert all(not (tmp_path / filename).exists() for filename in removed)
    assert (tmp_path / "index.html").read_text() == "stale content"
    cleanup(app, None)  # An incremental build need not contain stale files.

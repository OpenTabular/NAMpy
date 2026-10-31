"""Every retained document and source module has a visible navigation owner."""

import json
import posixpath
import runpy
from pathlib import Path
from types import SimpleNamespace

from docutils import nodes

ROOT = Path(__file__).resolve().parents[2]


def test_source_index_includes_nested_modules_without_a_global_menu():
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

    index = extension["source_menu"](tree, current, uri)
    assert 'aria-current="page">nampy.gam.families.inverse_gaussian</a>' in index
    assert '<span id="package-nampy-gam">nampy.gam</span>' in index
    assert '<span id="package-nampy-neural">nampy.neural</span>' in index
    assert "unavailable" not in index
    assert "<details" not in index
    for name in modules:
        if modules[name]:
            assert uri(current, extension["source_page"](name)) in index
    app = SimpleNamespace(
        env=SimpleNamespace(_viewcode_modules=modules),
        builder=SimpleNamespace(get_relative_uri=uri),
    )
    for page in ("index", "user_guide/survival", "_modules/index", current):
        context = {"body": "original body"}
        extension["page_context"](app, page, "page.html", context, None)
        assert context == {"body": "original body"}
    context = {"body": "source landing"}
    extension["page_context"](app, "development/source_code", "page.html", context, None)
    assert context["body"].startswith('source landing<ul class="module-source-index">')
    assert "nampy.gam.families.inverse_gaussian" in context["body"]
    assert "module_navigation" not in context



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


def test_successful_html_build_preserves_search_and_removes_only_unused_indexes(tmp_path):
    extension = runpy.run_path(str(ROOT / "docs/_ext/site_integrity.py"))
    removed = ("genindex.html", "py-modindex.html")
    retained = ("search.html", "searchindex.js", "index.html")
    for filename in (*removed, *retained):
        (tmp_path / filename).write_text("stale content")
    app = SimpleNamespace(outdir=tmp_path, builder=SimpleNamespace(format="html"))
    cleanup = extension["remove_unused_utility_pages"]
    cleanup(app, RuntimeError("build failed"))
    assert all((tmp_path / filename).exists() for filename in removed)
    cleanup(app, None)
    assert all(not (tmp_path / filename).exists() for filename in removed)
    assert all((tmp_path / filename).read_text() == "stale content" for filename in retained)
    cleanup(app, None)  # An incremental build need not contain stale files.


def test_search_uses_the_themes_form_and_source_has_one_development_entry():
    assert not (ROOT / "docs/_templates/searchbox.html").exists()
    source = (ROOT / "docs/_ext/site_integrity.py").read_text()
    assert "disable_search" not in source
    index = (ROOT / "docs/index.rst").read_text()
    assert index.count("Browse source code <development/source_code>") == 1
    layout = (ROOT / "docs/_templates/layout.html").read_text()
    assert "module_navigation" not in layout
    assert "module-code-navigation" not in layout

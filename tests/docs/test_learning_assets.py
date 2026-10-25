"""Check that published visuals and provenance support the claims beside them."""

import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import nbformat

ROOT = Path(__file__).resolve().parents[2]


def test_teaching_figures_are_retained_and_described():
    minimum = {
        "01_getting_started": 5,
        "02_statistical_gams": 4,
        "03_neural_models": 2,
        "04_distributional_models": 4,
        "05_real_data_classification": 3,
        "group_smooths": 3,
        "smoothness_selection": 1,
        "derivatives_contrasts": 3,
        "gam_diagnostics": 3,
        "overdispersed_counts": 1,
        "tensor_interactions": 3,
        "regional_smoothing": 2,
        "correlated_errors": 1,
        "neighborhood_cv": 2,
        "adaptive_smoothing": 1,
        "spherical_smoothing": 1,
        "functional_inputs": 1,
        "curvature_constraints": 1,
        "gamma_response": 1,
        "beta_response": 1,
        "tweedie_response": 1,
        "ordered_response": 1,
        "gamma_location_scale": 1,
    }
    for name, count in minimum.items():
        book = nbformat.read(ROOT / "docs/notebooks" / (name + ".ipynb"), 4)
        images = 0
        for cell in book.cells:
            for output in cell.get("outputs", []):
                if any(
                    mime in output.get("data", {})
                    for mime in ["image/png", "image/svg+xml"]
                ):
                    images += 1
                    alt = cell.metadata.get("figure_alt", "")
                    assert len(alt) > 25, (name, cell.source[:50])
                    assert ".png" not in alt
        assert images >= count, (name, images)


def test_detailed_figures_contain_claimed_panels_from_current_examples():
    manifest = json.loads((ROOT / "docs/_generated/detailed_assets.json").read_text())
    assert manifest["tree"]["panels"] == ["age", "income", "region"]
    assert all(count >= 3 for count in manifest["tree"]["lines"])
    assert len(manifest["quantile"]["panels"]) == 2
    assert "calibration" in manifest["quantile"]["panels"][0].lower()
    assert "age" in manifest["quantile"]["panels"][1].lower()
    assert manifest["quantile"]["lines"] == [2, 6]
    assert manifest["centering"] == "empirical training rows"
    for name, expected in manifest["source_sha256"].items():
        assert (
            hashlib.sha256((ROOT / "examples" / name).read_bytes()).hexdigest()
            == expected
        )


def test_source_state_tracks_code_without_hashing_its_own_outputs(
    tmp_path, monkeypatch
):
    module = runpy.run_path(str(ROOT / "docs/scripts/execution_support.py"))
    monkeypatch.setattr(module["subprocess"], "check_output", lambda *a, **k: "a" * 40)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    directory = tmp_path / "docs/notebooks"
    directory.mkdir(parents=True)
    path = directory / "example.ipynb"
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")])
    nbformat.write(book, path)
    before = module["source_state"](tmp_path)
    book.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="output changed")
    ]
    nbformat.write(book, path)
    assert module["source_state"](tmp_path)["source_sha256"] == before["source_sha256"]
    book.cells[0].source = "x = 2"
    nbformat.write(book, path)
    assert module["source_state"](tmp_path)["source_sha256"] != before["source_sha256"]
    assert len(before["revision"]) == 40 and before["source_manifest"]


def test_display_cleanup_preserves_results_and_meaningful_warnings():
    module = runpy.run_path(str(ROOT / "docs/scripts/execution_support.py"))
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("example()")])
    book.cells[0].outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text="GPU available: False\ndependency.py: FutureWarning: `torch.jit.script` is deprecated.\n  warnings.warn(\npackage/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated.\nTest R2: 0.9\nConvergenceWarning: investigate fit\n",
        ),
        nbformat.v4.new_output(
            "execute_result",
            data={"text/plain": "GAMSummary(large repr)"},
            execution_count=1,
        ),
        nbformat.v4.new_output(
            "display_data",
            data={"text/html": "<table><tr><td>Result</td></tr></table>"},
        ),
    ]
    module["clean_outputs"](book)
    assert len(book.cells[0].outputs) == 2
    stream = book.cells[0].outputs[0].text
    assert (
        "GPU available" not in stream
        and "Test R2" in stream
        and "ConvergenceWarning" in stream
    )
    assert "text/html" in book.cells[0].outputs[1].data


def test_authored_image_descriptions_reach_the_doctree(tmp_path):
    from docutils import nodes

    module = runpy.run_path(str(ROOT / "docs/_ext/site_integrity.py"))
    directory = tmp_path / "notebooks"
    directory.mkdir()
    cell = nbformat.v4.new_code_cell(
        "plot()",
        metadata={
            "figure_alt": "Known and fitted conditional standard deviation on the response scale"
        },
    )
    cell.outputs = [nbformat.v4.new_output("display_data", data={"image/png": "AA=="})]
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), directory / "example.ipynb")
    tree = nodes.document("", "")
    image = nodes.image(uri="generated-filename.png")
    tree += image
    module["notebook_image_descriptions"](
        SimpleNamespace(srcdir=tmp_path), tree, "notebooks/example"
    )
    assert image["alt"] == cell.metadata.figure_alt


def test_worked_component_row_is_separate_from_the_csv_table():
    from docutils import nodes
    from docutils.core import publish_doctree

    source = (ROOT / "docs/_generated/home_gam.inc").read_text()
    tree = publish_doctree(source, settings_overrides={"halt_level": 2})
    assert any(
        "For the first displayed row" in paragraph.astext()
        for paragraph in tree.findall(nodes.paragraph)
    )


def test_image_urls_change_when_regenerated_bytes_change(tmp_path):
    import zlib

    module = runpy.run_path(str(ROOT / "docs/_ext/site_integrity.py"))
    image = tmp_path / "plot.png"
    image.write_bytes(b"first generated plot")
    page = tmp_path / "index.html"
    page.write_text(
        '<a href="plot.png"><img src="plot.png"></a><img src="https://example.com/external.png">'
    )
    app = SimpleNamespace(outdir=tmp_path, builder=SimpleNamespace(format="html"))
    module["version_local_images"](app, None)
    first = f"{zlib.crc32(image.read_bytes()):08x}"
    assert page.read_text().count("plot.png?v=" + first) == 2
    assert 'src="https://example.com/external.png"' in page.read_text()
    image.write_bytes(b"new generated plot")
    module["version_local_images"](app, None)
    second = f"{zlib.crc32(image.read_bytes()):08x}"
    assert second != first and first not in page.read_text()
    assert page.read_text().count("plot.png?v=" + second) == 2

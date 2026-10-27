"""Reference inventory and generation must stay aligned with runtime exports."""

import importlib
import runpy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs/_ext"))
references = importlib.import_module("neighbor_reference")


def test_extension_generator_is_idempotent_and_discovers_all_symbols(tmp_path):
    references.generate(tmp_path)
    first = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    references.generate(tmp_path)
    second = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert first == second
    for name in ["compile_design", "CompiledDesign", "DesignEvaluation", "DesignTerm"]:
        assert (tmp_path / f"statistical_reference/extensions/{name}.rst").exists()
    assert (
        "10000"
        in (tmp_path / "statistical_reference/estimators/QuantileGAM.rst").read_text()
    )


def test_new_runtime_symbols_require_authored_examples(monkeypatch, tmp_path):
    from nampy.gam import extensions

    monkeypatch.setattr(extensions, "__all__", [*extensions.__all__, "FutureFeature"])
    with pytest.raises(ValueError, match="inventory differs"):
        references.generate(tmp_path)


def test_example_manifest_publication_preserves_existing_json_on_failure(tmp_path):
    import json

    runner = runpy.run_path(
        str(Path(__file__).parents[2] / "docs/scripts/check_reference_examples.py")
    )
    write = runner["write_json_atomic"]
    path = tmp_path / "records.json"
    write(path, {"status": "passed"})
    with pytest.raises(TypeError):
        write(path, {"invalid": object()})
    assert json.loads(path.read_text()) == {"status": "passed"}
    assert list(tmp_path.iterdir()) == [path]

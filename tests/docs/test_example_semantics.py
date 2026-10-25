"""Check documented behavior beyond whether examples execute without errors."""

import ast
import re
import runpy
import textwrap
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest

from nampy.contracts import AdditivePrediction, AdditiveTermMetadata

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "filename, marker",
    [
        ("example_qnam.py", "# Fit QNAM"),
        ("example_ensemble.py", "# Models"),
    ],
)
def test_detailed_examples_reserve_disjoint_test_rows(filename, marker):
    path = ROOT / "examples" / filename
    namespace = runpy.run_path(str(path))
    prefix = path.read_text().split("def main():", 1)[1].split(marker, 1)[0]
    exec(textwrap.dedent(prefix), namespace)
    splits = [set(namespace[name].index) for name in ["X_train", "X_val", "X_test"]]
    assert all(splits)
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    assert set.union(*splits) == set(namespace["X"].index)


def test_introductory_regression_renders_real_effect_figures(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    page = (ROOT / "docs/examples/fit_models.rst").read_text()
    source = (
        page.split("Predict class probabilities", 1)[0]
        + page[page.index("Inspect multiple outputs") :]
    )
    blocks = re.findall(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", source, re.M
    )
    namespace = {}
    for block in blocks:
        # Two epochs exercise the same prediction/plot contracts without
        # testing convergence or imposing a model-quality threshold.
        code = textwrap.dedent(block).replace("max_epochs=20", "max_epochs=2")
        exec(code, namespace)
    assert isinstance(namespace["X_test"], pd.DataFrame)
    assert np.asarray(namespace["predictions"]).shape == (len(namespace["X_test"]),)
    figures = namespace["effects"]
    assert figures and all(figure.axes for figure in figures)
    assert namespace["multi_predictions"].shape == (len(namespace["X_test"]), 2)
    assert all(
        np.asarray(value).shape == (len(namespace["X_test"]), 2)
        for value in namespace["components"].terms.values()
    )
    plt.close("all")


@pytest.mark.parametrize("kind", ["main", "contextual", "global"])
def test_validation_report_checks_observed_term_kinds(kind):
    notebook = nbformat.read(
        ROOT / "docs/development/notebooks/04_distributional_models.ipynb", 4
    )
    source = next(
        cell.source
        for cell in notebook.cells
        if "def demonstrated_term_kinds" in cell.source
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "demonstrated_term_kinds"
    )
    values = np.arange(4.0)
    components = AdditivePrediction(
        values,
        values,
        {"term": values},
        0.0,
        "neural",
        term_metadata={"term": AdditiveTermMetadata(("x",), kind=kind)},
    )
    estimator = SimpleNamespace(predict_components=lambda _X: components)
    namespace = {
        "np": np,
        "X": pd.DataFrame({"x": values}),
        "regression_models": {"example": estimator},
        "classification_models": {},
        "lss_models": {},
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<report>", "exec"),
        namespace,
    )
    assert namespace["demonstrated_term_kinds"]("example") == {kind}
    components.terms["term"] = values + 100
    with pytest.raises((ValueError, AssertionError)):
        namespace["demonstrated_term_kinds"]("example")


@pytest.mark.parametrize(
    "document, section, checks",
    [
        (
            "models/snam.rst",
            None,
            "assert isinstance(group_norms, dict) and group_norms\nassert set(selected) <= set(group_norms)\nassert figures",
        ),
        (
            "models/qnam.rst",
            None,
            "assert quantiles.shape == (len(X_test), 3)\nassert len(ax.lines) == 3",
        ),
        (
            "user_guide/custom_models.rst",
            "Build an attentive model",
            "assert np.isfinite(predictions).all()\nassert len(predictions) == len(X_test)",
        ),
    ],
)
def test_model_guide_snippets_run_in_clean_sessions(
    tmp_path, document, section, checks
):
    import os
    import subprocess
    import sys

    source = (ROOT / "docs" / document).read_text()
    if section:
        source = source.split(section, 1)[1]
    if document == "user_guide/custom_models.rst":
        source = source.split("Important Notes", 1)[0]
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", source)
    if document != "user_guide/custom_models.rst":
        blocks = blocks[:1]
    code = "\n".join(textwrap.dedent(block) for block in blocks).replace(
        "max_epochs=20", "max_epochs=2"
    )
    script = tmp_path / "documented_example.py"
    script.write_text(code + "\n" + checks + "\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

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
@pytest.mark.parametrize("task", ["regression", "classification", "distributional"])
@pytest.mark.parametrize("status", ["Passed", "Failed", "Not run", "Unsupported"])
def test_validation_report_checks_observed_term_kinds(kind, task, status):
    notebook = nbformat.read(
        ROOT / "docs/development/notebooks/04_distributional_models.ipynb", 4
    )
    source = next(
        cell.source
        for cell in notebook.cells
        if "def demonstrated_term_kinds" in cell.source
    )
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"task_status", "demonstrated_term_kinds"}
    ]
    values = np.arange(4.0)
    components = AdditivePrediction(
        values,
        values,
        {"term": values},
        0.0,
        "neural",
        term_metadata={"term": AdditiveTermMetadata(("x",), kind=kind)},
    )
    calls = []

    def predict_components(_X):
        calls.append(_X)
        return components

    estimator = SimpleNamespace(predict_components=predict_components)
    report = pd.DataFrame([{"architecture": "example", "status": status}])
    empty_report = report.iloc[:0]
    namespace = {
        "np": np,
        "X": pd.DataFrame({"x": values}),
        "architectures": lambda: {
            "example": SimpleNamespace(
                supports=lambda capability: capability == task and status != "Unsupported"
            )
        },
        "regression_models": {"example": estimator} if task == "regression" else {},
        "classification_models": {"example": estimator} if task == "classification" else {},
        "lss_models": {"example": estimator} if task == "distributional" else {},
        "regression_report": report if task == "regression" else empty_report,
        "classification_report": report if task == "classification" else empty_report,
        "lss_report": report if task == "distributional" else empty_report,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "<report>", "exec"),
        namespace,
    )
    if status != "Passed":
        assert namespace["demonstrated_term_kinds"]("example") == set()
        assert not calls
        return
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


@pytest.mark.parametrize("name", ["correlated_errors", "tweedie_response"])
def test_statistical_tutorial_fits_converge_without_warning(name):
    """The published datasets must pass both smoothing and coefficient checks."""
    import warnings

    from nampy.gam.diagnostics import build_optimizer_trace

    book = nbformat.read(ROOT / "docs/notebooks" / (name + ".ipynb"), 4)
    fitted_names = (
        ("ordinary", "correlated") if name == "correlated_errors" else ("model",)
    )
    namespace = {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Smoothing optimisation did not converge.*"
        )
        for cell in book.cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, name, "exec"), namespace)
            if all(key in namespace for key in fitted_names):
                break
    for key in fitted_names:
        model = namespace[key]
        diagnostic = build_optimizer_trace(model)["fit"]
        assert diagnostic["converged"], (name, key, diagnostic)
        assert model.fit_result().core.converged, (name, key)


def test_readme_nam_example_matches_homepage():
    """The shared figure must have the same fit and truth overlay in both places."""
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", readme, re.S)
    start = next(i for i, block in enumerate(blocks) if "from nampy.models import NAMRegressor" in block)
    readme_code = blocks[start] + "\n" + blocks[start + 1]
    homepage = (ROOT / "docs/_includes/model_showcases.inc").read_text().split("Neural Additive Model (NAM)", 1)[1]
    block = re.search(r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", homepage, re.M)[1]
    assert ast.dump(ast.parse(readme_code)) == ast.dump(ast.parse(textwrap.dedent(block)))


def test_readme_distributional_example_has_matching_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.S)
    namespace = {}
    for block in blocks[:3]:
        exec(compile(block, "README", "exec"), namespace)
    example = next(block for block in blocks if "mean_log_likelihood =" in block)
    exec(compile(example.replace("max_epochs=150", "max_epochs=2"), "README", "exec"), namespace)
    assert len(namespace["X_test"]) == len(namespace["y_test"])
    assert np.isfinite(namespace["mean_log_likelihood"])

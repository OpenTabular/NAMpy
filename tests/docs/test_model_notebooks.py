"""Structural checks for the generated public model notebooks."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from nampy.neural.registry import architectures

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"

ARCHITECTURE_NOTEBOOKS = {
    "linreg": "02_linreg.ipynb",
    "nam": "03_nam.ipynb",
    "snam": "04_snam.ipynb",
    "sian": "05_sian.ipynb",
    "gpnam": "06_gpnam.ipynb",
    "igann": "07_igann.ipynb",
    "nbm": "08_nbm.ipynb",
    "spam": "09_spam.ipynb",
    "nbm_spam": "10_nbm_spam.ipynb",
    "treenam": "11_treenam.ipynb",
    "ensemble_treenam": "12_ensemble_treenam.ipynb",
    "nodegam": "13_nodegam.ipynb",
    "natt": "14_natt.ipynb",
    "namformer": "15_namformer.ipynb",
    "qnam": "16_qnam.ipynb",
    "spline_nam": "17_spline_nam.ipynb",
}

OTHER_NOTEBOOKS = {
    "00_overview.ipynb",
    "01_gam.ipynb",
    "18_neural_ensemble.ipynb",
}

MODEL_SPECIFIC_TOKENS = {
    "linreg": "intercept",
    "nam": "feature_layer",
    "snam": "group_lasso_lambda",
    "sian": "interaction_selection_table",
    "gpnam": "basis_transform",
    "igann": "training_history",
    "nbm": 'featurizer="conv1d"',
    "spam": "local_term_importance",
    "nbm_spam": "num_subnets",
    "treenam": "use_hard_routing_in_eval",
    "ensemble_treenam": "num_estimators",
    "nodegam": "pretrain_epochs",
    "natt": "n_heads",
    "namformer": "interaction_importance",
    "qnam": "distributional_kwargs",
    "spline_nam": "learn_knots",
}


def _load(filename: str) -> dict:
    return json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))


def _source(payload: dict, cell_type: str | None = None) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell_type is None or cell["cell_type"] == cell_type
    )


def test_checked_in_notebooks_match_the_generator():
    generator = runpy.run_path(str(ROOT / "tests/docs/model_notebook_generator.py"))
    generated = {
        "00_overview.ipynb": generator["overview_notebook"](),
        "01_gam.ipynb": generator["gam_notebook"](),
        **{
            spec["filename"]: generator["neural_notebook"](spec)
            for spec in generator["MODEL_SPECS"]
        },
        "18_neural_ensemble.ipynb": generator["neural_ensemble_notebook"](),
    }
    assert {filename: _load(filename) for filename in generated} == generated
    assert (NOTEBOOKS / "README.md").read_text(encoding="utf-8") == generator[
        "README"
    ]


def test_every_registered_architecture_has_one_notebook():
    assert set(architectures()) == set(ARCHITECTURE_NOTEBOOKS)
    expected = set(ARCHITECTURE_NOTEBOOKS.values()) | OTHER_NOTEBOOKS
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == expected


@pytest.mark.parametrize(
    "filename", sorted(set(ARCHITECTURE_NOTEBOOKS.values()) | OTHER_NOTEBOOKS)
)
def test_notebook_is_clean_valid_json_with_compilable_code(filename):
    payload = _load(filename)
    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    assert payload["cells"]

    ids = [cell["id"] for cell in payload["cells"]]
    assert len(ids) == len(set(ids))
    assert "$$" in _source(payload, "markdown")

    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{filename}:cell-{index}", "exec")


@pytest.mark.parametrize(
    "filename", sorted(set(ARCHITECTURE_NOTEBOOKS.values()) | OTHER_NOTEBOOKS)
)
def test_notebook_executes_its_default_no_training_path(filename):
    """Execute cells in one namespace, like a fresh notebook kernel.

    The examples intentionally keep ``RUN_TRAINING`` false. This checks every
    import, constructor, parameter name, and unguarded public call without
    making documentation tests train 19 models.
    """

    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(_load(filename)["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{filename}:cell-{index}", "exec"), namespace)


@pytest.mark.parametrize("architecture,filename", ARCHITECTURE_NOTEBOOKS.items())
def test_neural_notebook_covers_shared_and_specific_api(architecture, filename):
    source = _source(_load(filename))
    assert "model.model." not in source
    for public_method in (
        ".fit(",
        ".predict(",
        ".score(",
        ".evaluate(",
        ".predict_components(",
    ):
        assert public_method in source
    assert MODEL_SPECIFIC_TOKENS[architecture] in source


def test_gam_notebook_covers_adapter_and_raw_backend_api():
    source = _source(_load("01_gam.ipynb"))
    for token in (
        "GAMRegressor",
        "GAMClassifier",
        "from nampy.gam import GAM",
        ".fit(",
        ".predict(",
        ".score(",
        ".evaluate(",
        ".predict_components(",
        ".standard_errors(",
        ".lpmatrix(",
        ".parity_snapshot(",
    ):
        assert token in source


def test_independent_ensemble_is_distinguished_from_joint_tree_ensemble():
    source = _source(_load("18_neural_ensemble.ipynb"))
    assert "NeuralEnsemble" in source
    assert "EnsembleTreeNAM" in source
    assert "predict_component_uncertainty" in source
    assert "bootstrap=True" in source

"""Structural checks for the generated public model notebooks."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import pytest

from nampy.neural.registry import architectures

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "docs/notebooks"
GENERATOR = ROOT / "docs/generate_notebooks.py"

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
    generator = runpy.run_path(str(GENERATOR))
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


@pytest.mark.parametrize("filename", sorted(ARCHITECTURE_NOTEBOOKS.values()))
def test_neural_notebook_executes_its_fitted_visual_story(filename):
    """The documented fit, explanation, and plotting calls stay executable."""
    namespace = {
        "__name__": "__notebook__",
        "RUN_TRAINING": True,
        "display": lambda *args, **kwargs: None,
    }
    for index, cell in enumerate(_load(filename)["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{filename}:cell-{index}", "exec"), namespace)

    assert namespace["components"].terms
    assert not namespace["importance"].empty
    namespace["plt"].close("all")


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
    for section in (
        "## Model in one view",
        "## Fit the model",
        "## Model-specific controls",
        "## Supported task variants",
        "## References",
    ):
        assert section in source
    for public_explanation in (".explain_terms(", ".term_importance("):
        assert public_explanation in source
    if architecture != "qnam":
        assert ".plot_terms(" in source
    for verbose_heading in (
        "## Learning goals",
        "## Practical workflow",
        "## When to use this model",
        "## Task variants and limits",
    ):
        assert verbose_heading not in source
    assert "task_variants" in source


def test_every_neural_model_has_substantive_theory_guidance_and_references():
    generator = runpy.run_path(str(GENERATOR))
    filenames = {spec["filename"] for spec in generator["MODEL_SPECS"]}

    for mapping_name in (
        "MODEL_THEORY",
        "MODEL_REFERENCES",
        "MODEL_TASK_VARIANTS",
    ):
        assert set(generator[mapping_name]) == filenames

    for filename, theory in generator["MODEL_THEORY"].items():
        assert len(theory.split()) >= 65, filename
        assert generator["MODEL_REFERENCES"][filename].strip().startswith("-")


@pytest.mark.parametrize(
    "filename", sorted(set(ARCHITECTURE_NOTEBOOKS.values()) | OTHER_NOTEBOOKS)
)
def test_every_notebook_has_a_useful_visual_surface(filename):
    source = _source(_load(filename))
    assert "plt.subplots" in source
    assert "plt.show()" in source


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
        ".explain_terms(",
        ".term_importance(",
        ".standard_errors(",
        ".lpmatrix(",
        ".parity_snapshot(",
    ):
        assert token in source


def test_gam_notebook_covers_theory_and_supported_modeling_surface():
    source = _source(_load("01_gam.ipynb"))
    normalized_source = source.lower()
    for token in (
        "Wood (2011)",
        "Pya and Wood (2015)",
        "10.1111/j.1467-9868.2010.00749.x",
        "10.1007/s11222-013-9448-7",
        "penalized likelihood",
        "effective degrees of freedom",
        "PIRLS",
        "bs='cr'",
        "bs='cs'",
        "bs='cc'",
        "bs='ps'",
        "bs='tp'",
        "bs='ts'",
        "te(",
        "ti(",
        "bs='re'",
        "bs='fs'",
        "bs='sz'",
        '"outer_newton"',
        '"bfgs"',
        '"efs"',
        '"optim"',
        '"fixed"',
        '"gaussian"',
        '"poisson"',
        '"binomial"',
        '"gamma"',
        '"negative_binomial"',
        '"tweedie"',
        '"beta"',
        'family="gaulss"',
        'family="gammals"',
        "offset(log_exposure)",
        "bs='mpi'",
        ".k_check(",
        ".gam_check(",
        ".concurvity(",
        ".vcov(",
        "stats.probplot(",
        'k_diagnostics["k_index"]',
        'concurvity["values"]',
    ):
        assert token.lower() in normalized_source
    assert "## Learning goals" not in source
    assert "## Practical workflow and limits" not in source


def test_gam_notebook_uses_one_coherent_energy_story():
    source = _source(_load("01_gam.ipynb"))
    for column in (
        '"demand"',
        '"events"',
        '"alert"',
        '"cost"',
        '"efficiency"',
        '"temperature"',
        '"humidity"',
        '"site"',
        '"exposure"',
        '"dose"',
    ):
        assert column in source


def test_gam_notebook_executes_its_short_training_story():
    """The primary fitted examples run; expensive galleries remain opt-in."""
    namespace = {
        "__name__": "__notebook__",
        "RUN_TRAINING": True,
        "RUN_EXTENDED_FITS": False,
        "display": lambda *args, **kwargs: None,
    }
    for index, cell in enumerate(_load("01_gam.ipynb")["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"01_gam.ipynb:cell-{index}", "exec"), namespace)

    assert namespace["predictions"].shape == (240,)
    assert namespace["probabilities"].shape == (240, 2)
    assert namespace["Xp"].shape[0] == 240
    namespace["components"].validate_additive_reconstruction()


@pytest.mark.skipif(
    os.environ.get("NAMPY_RUN_EXTENDED_DOC_FITS") != "1",
    reason="set NAMPY_RUN_EXTENDED_DOC_FITS=1 for the three-minute GAM gallery",
)
def test_gam_notebook_executes_its_extended_visual_gallery():
    """Optimizer, family, multi-predictor, and shape galleries stay runnable."""
    namespace = {
        "__name__": "__notebook__",
        "RUN_TRAINING": False,
        "RUN_EXTENDED_FITS": True,
        "display": lambda *args, **kwargs: None,
    }
    for index, cell in enumerate(_load("01_gam.ipynb")["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"01_gam.ipynb:cell-{index}", "exec"), namespace)

    assert len(namespace["fitted_optimizers"]) == 4
    assert len(namespace["fitted_families"]) == 7
    assert namespace["mean_and_scale"].shape == (240, 2)
    assert namespace["shape_derivative"].derivative.shape == (240,)
    namespace["plt"].close("all")


def test_independent_ensemble_is_distinguished_from_joint_tree_ensemble():
    source = _source(_load("18_neural_ensemble.ipynb"))
    assert "NeuralEnsemble" in source
    assert "EnsembleTreeNAM" in source
    assert "predict_component_uncertainty" in source
    assert "bootstrap=True" in source
    assert "explain_additive_prediction" in source
    assert "term_importance_table" in source
    assert ".plot_terms(" in source


def test_independent_ensemble_notebook_executes_its_fitted_visual_story():
    namespace = {
        "__name__": "__notebook__",
        "RUN_TRAINING": True,
        "display": lambda *args, **kwargs: None,
    }
    filename = "18_neural_ensemble.ipynb"
    for index, cell in enumerate(_load(filename)["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{filename}:cell-{index}", "exec"), namespace)

    assert namespace["uncertainty"].n_estimators == 3
    assert not namespace["importance"].empty
    namespace["plt"].close("all")


def test_docs_notebooks_are_the_only_notebook_collection():
    """Sphinx owns the generated notebooks without a second root collection."""
    docs = ROOT / "docs"
    assert not list((ROOT / "notebooks").glob("*.ipynb"))
    assert list((docs / "notebooks").glob("*.ipynb"))

    index = (docs / "index.rst").read_text(encoding="utf-8")
    tutorial_index = (docs / "notebook_tutorials.rst").read_text(encoding="utf-8")
    assert "notebook_tutorials" in index
    assert "single canonical notebook collection" in " ".join(tutorial_index.split())
    expected = set(ARCHITECTURE_NOTEBOOKS.values()) | OTHER_NOTEBOOKS
    for filename in expected:
        assert f"notebooks/{Path(filename).stem}" in tutorial_index


def test_sphinx_build_regenerates_notebooks_before_rendering():
    conf = (ROOT / "docs/conf.py").read_text(encoding="utf-8")
    assert "generate_notebooks.py" in conf
    assert '"nbsphinx"' in conf
    assert "nbsphinx_execute = \"never\"" in conf


def test_sphinx_catalogs_and_indexes_do_not_duplicate_tutorial_code():
    """Catalog, API, and routing pages stay distinct from recipes/tutorials."""
    docs = ROOT / "docs"
    routing_pages = (
        docs / "api/index.rst",
        docs / "examples/index.rst",
        docs / "models/index.rst",
        docs / "user_guide.rst",
    )
    for path in routing_pages:
        assert ".. code-block:: python" not in path.read_text(encoding="utf-8"), path

    assert not (docs / "examples/custom_model.rst").exists()
    user_guide = (docs / "user_guide.rst").read_text(encoding="utf-8")
    assert "user_guide/custom_models" in user_guide


def test_standalone_examples_declare_their_non_tutorial_role():
    source = (ROOT / "examples/README.md").read_text(encoding="utf-8")
    assert "terminal-runnable verification scripts" in source
    assert "notebooks/" in source
    for path in sorted((ROOT / "examples").glob("example_*.py")):
        assert f"`{path.name}`" in source

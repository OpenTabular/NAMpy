"""Reference generation must work before autosummary creates its output folder."""

import runpy
import sys
from pathlib import Path

# Match the Sphinx extension import environment for standalone run_path calls.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/_ext"))


def test_reference_generation_from_clean_checkout(tmp_path):
    extension = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/_ext/public_reference.py")
    )
    generated = tmp_path / "docs/api/generated"
    assert not generated.exists()

    extension["generate"](tmp_path)

    assert not generated.exists(), (
        "The include generator must not recreate legacy pages"
    )
    page = tmp_path / "docs/_generated/reference/LinRegRegressor.inc"
    reference = tmp_path / "docs/_generated/reference/LinRegRegressor.inc"
    assert reference.is_file()
    from nampy import models

    pages = [
        tmp_path / f"docs/_generated/reference/{name}.inc"
        for name in models.__all__
        if (tmp_path / f"docs/_generated/reference/{name}.inc").is_file()
    ]
    assert pages
    for generated_page in pages:
        name = generated_page.stem
        text = (tmp_path / f"docs/_generated/reference/{name}.inc").read_text()
        if name.endswith(("Regressor", "Classifier", "LSS")) and name not in {
            "GAMRegressor",
            "GAMClassifier",
            "GAMLSS",
        }:
            assert "Full shared fit contract" in text
        assert "See the shared training contract for this fit control." not in text
    for name in [
        "NAMRegressor",
        "NAMClassifier",
        "NAMLSS",
        "GPNAMRegressor",
        "IGANNRegressor",
    ]:
        text = (tmp_path / f"docs/_generated/reference/{name}.inc").read_text()
        params = getattr(models, name)().get_params(deep=False)
        for parameter in ["lr", "weight_decay", "numerical_preprocessing"]:
            if parameter in params:
                default = extension["literal"](params[parameter])
                assert f"``{parameter}``\n     - ``{default}``\n     - " in text
        assert "Native solvers such as IGANN" in text
    classifier = (tmp_path / "docs/_generated/reference/NAMClassifier.inc").read_text()
    assert "Component extraction describes all actual logits" in classifier
    lss = (tmp_path / "docs/_generated/reference/NAMLSS.inc").read_text()
    assert "family_.param_names" in lss
    assert "predict_distribution" in lss
    assert "Normal, Gamma, Student-t, beta, negative-binomial, lognormal and Weibull" in lss
    statistical_lss = (tmp_path / "docs/_generated/reference/GAMLSS.inc").read_text()
    assert "predict_distribution" in statistical_lss
    assert "nampy.distributions.DistributionPrediction" in statistical_lss
    native = (tmp_path / "docs/_generated/reference/IGANNRegressor.inc").read_text()
    assert (
        "native boosting for squared-error regression and binary classification"
        in native
    )
    assert "model.fit(X, y, max_epochs=" not in native
    native_lss = (tmp_path / "docs/_generated/reference/IGANNLSS.inc").read_text()
    assert "This LSS task uses gradient training" in native_lss
    assert "Full shared fit contract" in native_lss
    gpnam = (tmp_path / "docs/_generated/reference/GPNAMRegressor.inc").read_text()
    assert "fixed-design conjugate-gradient solver" in gpnam
    quantiles = (tmp_path / "docs/_generated/reference/QNAMLSS.inc").read_text()
    assert "pinball loss" in quantiles and "complete probability density" in quantiles
    capabilities = (
        tmp_path / "docs/_generated/reference/model_capabilities.inc"
    ).read_text()
    spline_row = capabilities.split("</models/spline_nam>", 1)[1]
    assert "Binary-classification components" not in spline_row
    assert "no LSS interface" in spline_row
    # The same builder hook also runs for repeated documentation builds.
    before = page.read_text()
    extension["generate"](tmp_path)
    assert page.read_text() == before


def test_fit_parameter_descriptions_match_the_training_contract():
    import inspect

    from nampy.models._neural import NeuralEstimatorBase
    from nampy.neural.configuration import GradientTrainingConfig
    assert set(inspect.signature(NeuralEstimatorBase.fit).parameters) == {
        "self", "X", "y", "X_val", "y_val", "offset", "offset_val",
        "sample_weight", "sample_weight_val"}
    assert "lr" in inspect.signature(GradientTrainingConfig).parameters
    assert "trainer_kwargs" in inspect.signature(GradientTrainingConfig).parameters


def test_grouped_and_variadic_parameter_docstrings():
    extension = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/_ext/public_reference.py")
    )

    def example():
        """Parameters.

        lr, weight_decay
            Optimizer configuration.
        **trainer_kwargs : object
            Additional Trainer options.
        """

    meanings = extension["descriptions"](example)
    assert meanings["lr"] == meanings["weight_decay"] == "Optimizer configuration."
    assert meanings["trainer_kwargs"] == "Additional Trainer options."


def test_schedulers_match_implementation_and_training_guide():
    from nampy.neural.configuration import GradientTrainingConfig
    guide = (Path(__file__).resolve().parents[2] / "docs/user_guide/training.rst").read_text()
    for name in GradientTrainingConfig.SCHEDULES:
        assert f"``{name}``" in guide


def test_api_overview_symbols_have_written_summaries():
    """Autosummary must show descriptions rather than empty class-name rows."""
    import importlib
    import inspect

    groups = {
        "nampy.neural.architectures": ["BaseModel", "NATT", "NAMformer"],
        "nampy.neural.task": ["TaskModule"],
        "nampy.neural.configs": ["DefaultSplineNAMConfig"],
        "nampy.gam": ["solve_fit", "FitCoreSolution"],
        "nampy.gam.inference": ["anova_gam", "AnovaGAMSingle", "AnovaGAMComparison"],
        "nampy.gam.results": ["TermFitResult", "FitResult"],
        "nampy.models": ["NeuralRegressor", "NeuralClassifier", "NeuralLSS"],
        "nampy.neural.registry": ["get_architecture"],
        "nampy.neural.objectives": [
            "RegressionObjective",
            "BinaryObjective",
            "MulticlassObjective",
            "DistributionObjective",
            "classification_objective",
        ],
    }
    for module_name, symbols in groups.items():
        module = importlib.import_module(module_name)
        for name in symbols:
            summary = inspect.getdoc(getattr(module, name))
            assert summary and summary.splitlines()[0].endswith("."), (
                module_name,
                name,
            )


def test_built_api_summary_descriptions_include_nested_text():
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/scripts/check_documentation.py")
    )
    page = module["Page"](
        '<table class="autosummary"><tr><td><p>Model</p></td><td><p>A <em>useful</em> description.</p></td></tr><tr><td>Undocumented</td><td><p> </p></td></tr></table>'
    )
    assert page.autosummary_rows == [
        ["Model", "A useful description."],
        ["Undocumented", ""],
    ]


def test_training_contract_keeps_both_viewcode_anchor_spellings():
    from docutils import nodes
    from sphinx import addnodes

    module = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/_ext/site_integrity.py")
    )
    canonical = "nampy.models._neural.NeuralEstimatorBase.fit"
    alias = "nampy.models.NeuralEstimatorBase.fit"
    tree = nodes.container()
    signature = addnodes.desc_signature(ids=[canonical])
    tree += signature
    module["training_anchor_alias"](None, tree, "api/neural_training")
    module["training_anchor_alias"](None, tree, "api/neural_training")
    assert signature["ids"] == [canonical, alias]


def test_constructor_parameter_groups_follow_the_consuming_interface():
    from public_reference import constructor_parameter_group
    assert constructor_parameter_group("preprocessing__n_knots", "nam") == "Preprocessing"
    assert constructor_parameter_group("architecture_config__n_knots", "spline_nam") == "Architecture"
    assert constructor_parameter_group("training__lr_warmup_steps", "nam") == "Training defaults"
    assert constructor_parameter_group("training__weight_decay", "nam") == "Training defaults"
    assert constructor_parameter_group("architecture_config__interactions", "nam") == "Architecture"
    assert constructor_parameter_group("objective__family", "nam") == "Objective"

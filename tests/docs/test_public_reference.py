"""Reference generation must work before autosummary creates its output folder."""

import runpy
from pathlib import Path


def test_reference_generation_from_clean_checkout(tmp_path):
    extension = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/_ext/public_reference.py")
    )
    generated = tmp_path / "docs/api/generated"
    assert not generated.exists()

    extension["generate"](tmp_path)

    page = generated / "nampy.models.LinRegRegressor.rst"
    assert ".. autoclass:: LinRegRegressor" in page.read_text()
    reference = tmp_path / "docs/_generated/reference/LinRegRegressor.inc"
    assert reference.is_file()
    pages = list(generated.glob("*.rst"))
    assert pages
    for generated_page in pages:
        name = generated_page.stem.removeprefix("nampy.models.")
        text = (tmp_path / f"docs/_generated/reference/{name}.inc").read_text()
        assert "``trainer_kwargs``\n     - Optional keyword arguments" in text
        assert "See the shared training contract for this fit control." not in text
    from nampy import models

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
    assert "Component extraction is binary-only" in classifier
    lss = (tmp_path / "docs/_generated/reference/NAMLSS.inc").read_text()
    assert "family_.param_names" in lss
    native = (tmp_path / "docs/_generated/reference/IGANNRegressor.inc").read_text()
    assert "native boosting for binary targets" in native and "boosting" in native
    assert "model.fit(X, y, max_epochs=" not in native
    native_lss = (tmp_path / "docs/_generated/reference/IGANNLSS.inc").read_text()
    assert "This LSS task uses gradient training" in native_lss
    assert "model.fit(X, y, max_epochs=" in native_lss
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
    from nampy.models._base import NeuralEstimatorBase

    extension = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/_ext/public_reference.py")
    )
    meanings = extension["descriptions"](NeuralEstimatorBase.fit)
    for name in [
        "lr",
        "lr_schedule",
        "lr_patience",
        "lr_factor",
        "weight_decay",
        "lr_warmup_steps",
        "lr_decay_steps",
        "lr_decay_factor",
    ]:
        assert name in meanings
        assert "Trainer" not in meanings[name]
    for schedule in ["plateau", "inverse_sqrt", "cosine", "warmup_cosine", "none"]:
        assert f'"{schedule}"' in meanings["lr_schedule"]
    assert "Trainer" in meanings["trainer_kwargs"]


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
    import ast
    import inspect
    import textwrap

    from nampy.models._base import NeuralEstimatorBase
    from nampy.neural.task import TaskModule

    code = ast.parse(textwrap.dedent(inspect.getsource(TaskModule.__init__)))
    allowed = next(
        {item.value for item in node.comparators[0].elts}
        for node in ast.walk(code)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "lr_schedule"
        and isinstance(node.ops[0], ast.NotIn)
        and isinstance(node.comparators[0], ast.Set)
    )
    extension = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/_ext/public_reference.py")
    )
    description = extension["descriptions"](NeuralEstimatorBase.fit)["lr_schedule"]
    guide = (
        Path(__file__).resolve().parents[2] / "docs/user_guide/training.rst"
    ).read_text()
    for name in allowed:
        assert f'"{name}"' in description
        assert f'"{name}"' in guide

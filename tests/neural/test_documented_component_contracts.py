"""Regression coverage for user-visible decomposition and extension contracts."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nampy.contracts import AdditivePrediction, AdditiveTermMetadata
from nampy.explanations import (
    center_additive_prediction,
    explain_additive_prediction,
    term_importance_table,
)
from nampy.models import GAMLSS
from nampy.neural.architectures.namformer import NAMformer


@pytest.mark.parametrize("outputs", [1, 3])
@pytest.mark.parametrize("explicit", [False, True])
def test_namformer_reports_complete_global_contribution(outputs, explicit):
    model = NAMformer(
        {},
        {"x": {"dimension": 1}, "z": {"dimension": 1}},
        num_classes=outputs,
        d_model=8,
        n_heads=2,
        n_layers=1,
        head_layer_sizes=[8],
        interactions=[("x", "z")] if explicit else None,
        feature_dropout=0.0,
    )
    model.eval()
    inputs = {"x": torch.randn(5, 1), "z": torch.randn(5, 1)}
    result = model(inputs, {})
    perturbed = model({"x": inputs["x"], "z": inputs["z"] + 5}, {})
    torch.testing.assert_close(result["x"], perturbed["x"])
    assert not torch.equal(
        result["global_transformer"], perturbed["global_transformer"]
    )
    assert "global_transformer" in result
    assert ("x:z" in result) == explicit
    total = sum(
        value for key, value in result.items() if key not in {"output", "intercept"}
    )
    torch.testing.assert_close(result["output"], total + result.get("intercept", 0))
    assert result["output"].shape == (5, outputs)


def test_parameter_identity_is_not_an_interaction():
    x = np.linspace(-1, 1, 40)
    data = pd.DataFrame({"x": x, "y": x + 0.1 * np.cos(9 * x)})
    model = GAMLSS(
        formula={"mu": "y ~ s(x, bs='cr', k=5)", "sigma": "~ s(x, bs='cr', k=5)"},
        optimize_smoothing=False,
    ).fit(data)
    parts = model.predict_components(data)
    assert {m.parameter for m in parts.term_metadata.values()} == {"mu", "sigma"}
    assert all(
        m.features == ("x",) and m.kind == "main" for m in parts.term_metadata.values()
    )
    assert term_importance_table(parts).term_type.eq("main").all()
    assert explain_additive_prediction(data[["x"]], parts).term_type.eq("main").all()
    centered = center_additive_prediction(parts)
    assert centered.term_metadata == parts.term_metadata
    centered.validate_additive_reconstruction()


def test_global_term_is_not_presented_as_a_marginal_curve():
    values = np.arange(6.0)
    parts = AdditivePrediction(
        values,
        values,
        {"global_transformer": values},
        0.0,
        "neural",
        term_metadata={
            "global_transformer": AdditiveTermMetadata(("x", "z"), kind="global")
        },
    )
    table = explain_additive_prediction(
        pd.DataFrame({"x": values, "z": values**2}), parts
    )
    assert table.term_type.eq("global").all()
    assert len(table) == 1


@pytest.mark.parametrize("num,cat", [(True, False), (False, True), (True, True)])
def test_documented_custom_attention_handles_feature_blocks(num, cat):
    source = (
        Path(__file__).resolve().parents[2] / "docs/user_guide/custom_models.rst"
    ).read_text()
    section = source.split("Build an attentive model", 1)[1].split(
        "Register the architecture", 1
    )[0]
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", section)
    code = "\n".join(
        line[3:] if line.startswith("   ") else line
        for block in blocks
        for line in block.splitlines()
    )
    namespace = {}
    exec(code.split("# 3. One declaration")[0], namespace)
    cls = namespace["AttentiveMLP"]
    model = cls(
        {"group": {"dimension": 3}} if cat else {},
        {"x": {"dimension": 2}} if num else {},
    )
    token_counts = []
    model.attention.register_forward_pre_hook(
        lambda module, args: token_counts.append(args[0].shape[1])
    )
    out = model(
        {"x": torch.randn(4, 2)} if num else {},
        {"group": torch.eye(3)[torch.tensor([0, 1, 2, 0])]} if cat else {},
    )
    assert out["output"].shape == (4, 1)
    assert token_counts == [int(num) + int(cat)]
    torch.testing.assert_close(out["global_attention"], out["output"])

    # With both residual branches zeroed, the original feature tokens must
    # survive both additions. Shape-only tests would miss a discarded residual.
    model.eval()
    with torch.no_grad():
        for parameter in model.attention.parameters():
            parameter.zero_()
        for parameter in model.ffn.parameters():
            parameter.zero_()
        for projection in model.projections:
            projection.weight.fill_(1)
            projection.bias.zero_()
        model.output_layer.weight.fill_(1)
        model.output_layer.bias.zero_()
    numerical = {"x": torch.ones(4, 2)} if num else {}
    categorical = {"group": torch.eye(3)[torch.tensor([0, 1, 2, 0])]} if cat else {}
    inputs = {**numerical, **categorical}
    tokens = torch.stack(
        [
            projection(inputs[name].float())
            for name, projection in zip(
                model.feature_names, model.projections, strict=True
            )
        ],
        dim=1,
    )
    expected = model.output_layer(tokens.mean(dim=1))
    actual = model(numerical, categorical)["output"]
    assert torch.all(expected > 0)
    torch.testing.assert_close(actual, expected)


def test_lss_evaluation_accepts_pandas_targets(tmp_path):
    from nampy.models import NAMLSS

    X = pd.DataFrame({"x": np.linspace(-1, 1, 40)})
    y = pd.Series(X.x.to_numpy() + 0.1 * np.sin(5 * X.x.to_numpy()))
    model = NAMLSS(layer_sizes=[4], dropout=0.0, numerical_preprocessing="minmax")
    model.fit(
        X,
        y,
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        checkpoint_path=tmp_path,
    )
    series_score = model.evaluate(X, y, metrics={})
    array_score = model.evaluate(X, y.to_numpy(), metrics={})
    assert series_score["NLL"] == pytest.approx(array_score["NLL"])


def test_group_norms_public_surface(tmp_path):
    from sklearn.exceptions import NotFittedError

    from nampy.models import SNAMRegressor

    model = SNAMRegressor(layer_sizes=[4], numerical_preprocessing="minmax")
    with pytest.raises(NotFittedError):
        model.group_norms()
    X = pd.DataFrame({"x": np.linspace(-1, 1, 30)})
    model.fit(
        X,
        X.x.to_numpy(),
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        checkpoint_path=tmp_path,
    )
    assert model.group_norms() == model.model.model.get_group_norms()


@pytest.mark.parametrize("estimator_name", ["NAMLSS", "QNAMLSS"])
def test_quantile_evaluation_reports_pinball_loss(estimator_name, tmp_path):
    from nampy import models

    levels = np.asarray([0.1, 0.5, 0.9])
    X = pd.DataFrame({"x": np.linspace(-1, 1, 40)})
    y = X.x.to_numpy() + 0.1 * np.sin(5 * X.x.to_numpy())
    model = getattr(models, estimator_name)(
        family="quantile",
        layer_sizes=[4],
        dropout=0.0,
        distributional_kwargs={"quantiles": levels.tolist()},
    ).fit(
        X,
        y,
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        checkpoint_path=tmp_path,
    )
    prediction = model.predict(X)
    residual = y[:, None] - prediction
    expected = np.maximum(levels * residual, (levels - 1) * residual).sum(axis=1).mean()
    scores = model.evaluate(X, y, metrics={})
    assert set(scores) == {"pinball_loss"}
    assert scores["pinball_loss"] == pytest.approx(expected, rel=1e-5)
    assert model.score(X, y) == pytest.approx(-expected, rel=1e-5)

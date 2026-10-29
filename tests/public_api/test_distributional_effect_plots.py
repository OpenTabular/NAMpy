"""Public distributional-effect plotting is family-driven and reusable."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

from nampy.neural.configuration import (
    GradientTrainingConfig,
    ObjectiveConfig,
    PreprocessingConfig,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from nampy.contracts import AdditivePrediction
from nampy.inspection.effects import (
    DistributionalEffects,
    combine_distributional_effects,
    prepare_distributional_effects,
)
from nampy.neural.families import NormalDistribution, StudentTDistribution
from nampy.plotting import (
    plot_distributional_effects,
    render_distributional_effects,
)


class _GenericFamily(torch.nn.Module):
    """A family with deliberately non-standard parameter semantics."""

    def predict_point(self, raw, transformed=False):
        assert not transformed
        return torch.tanh(raw[:, 0])

    def predict_spread(self, raw, transformed=False):
        assert not transformed
        return torch.exp(raw[:, 1])


class _FakeDistributionalEstimator:
    def __init__(self):
        model = torch.nn.Module()
        model.objective = SimpleNamespace(family=_GenericFamily())
        self.model = model

    def summarize_predictors(self, predictors):
        raw = torch.as_tensor(predictors, dtype=torch.float32)
        family = self.model.objective.family
        return (family.predict_point(raw).numpy(), family.predict_spread(raw).numpy())

    def predict_components(
        self, X, *, center=False, reference_X=None, reference_weight=None
    ):
        del reference_X, reference_weight
        assert not center
        x = np.asarray(X["x"], dtype=np.float64)
        z = np.asarray(X["z"], dtype=np.float64)
        terms = {
            "x": np.column_stack([0.4 * x, 0.1 * x]),
            "z": np.column_stack([-0.2 * z, 0.05 * z]),
        }
        intercept = np.array([0.2, -0.3])
        link = intercept + terms["x"] + terms["z"]
        return AdditivePrediction(
            link=link,
            terms=terms,
            intercept=intercept,
            backend="neural",
        )


def test_prepare_distributional_effects_uses_family_contract():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 30), "z": np.linspace(1.0, 2.0, 30)})
    effects = prepare_distributional_effects(
        _FakeDistributionalEstimator(), X, response_scale=2.0
    )

    baseline = np.tanh(0.2 - 0.2 * X["z"].mean())
    expected_point = (np.tanh(0.2 - 0.2 * X["z"].mean() + 0.4 * X["x"]) - baseline) * 2.0
    expected_spread = np.exp(-0.3 + 0.05 * X["z"].mean() + 0.1 * X["x"]) * 2.0
    np.testing.assert_allclose(
        effects.point_effects["x"][:, 0], expected_point, rtol=1e-5
    )
    np.testing.assert_allclose(
        effects.conditional_sd["x"][:, 0], expected_spread, rtol=1e-5
    )


def test_render_distributional_effects_supports_repeated_fits():
    x = np.linspace(-2.0, 2.0, 80)
    curves = np.column_stack([x, x + 0.1, x - 0.1])
    spread = np.column_stack(
        [np.full_like(x, 0.5), np.full_like(x, 0.6), np.full_like(x, 0.4)]
    )
    effects = DistributionalEffects({"x": x}, {"x": curves}, {"x": spread})
    figures = render_distributional_effects(
        effects, point_label="mean effect", fit_label="seed"
    )
    try:
        assert len(figures) == 1
        assert figures[0].axes[0].collections  # density spans and fit ribbon
        assert len(figures[0].axes[0].lines) == 3  # lower, upper, and point curves
        labels = figures[0].axes[0].get_legend_handles_labels()[1]
        assert "mean effect" in labels
        assert "seed 10–90%" in labels
    finally:
        plt.close("all")


def test_combine_distributional_effects_stacks_fit_curves():
    x = np.linspace(0.0, 1.0, 10)
    first = DistributionalEffects({"x": x}, {"x": x}, {"x": x + 1.0})
    second = DistributionalEffects({"x": x}, {"x": x + 0.2}, {"x": x + 1.2})
    combined = combine_distributional_effects([first, second])
    assert combined.n_fits == 2
    np.testing.assert_allclose(combined.point_effects["x"][:, 1], x + 0.2)


def test_public_plot_convenience_returns_figures():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24), "z": np.linspace(1.0, 2.0, 24)})
    figures = plot_distributional_effects(
        _FakeDistributionalEstimator(), X, feature_name="x"
    )
    try:
        assert len(figures) == 1
        assert figures[0].axes[0].get_title() == "x"
    finally:
        plt.close("all")


def test_lss_estimator_exposes_distributional_effect_plot(tmp_path):
    from nampy.models.linreg import LinRegLSS

    rng = np.random.default_rng(4)
    X = pd.DataFrame({"x": rng.normal(size=48), "z": rng.normal(size=48)})
    y = 1.5 * X["x"].to_numpy() + rng.normal(scale=0.2, size=len(X))
    estimator = LinRegLSS(objective=ObjectiveConfig(family='normal'), preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'), training=GradientTrainingConfig(max_epochs=2, patience=1, checkpoint_path=str(tmp_path), trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False, 'num_sanity_val_steps': 0})).fit(X, y)
    figures = estimator.plot_distributional_effects(
        X, reference_X=X, feature_names=["x"]
    )
    try:
        assert len(figures) == 1
        assert figures[0].axes[0].get_title() == "x"
    finally:
        plt.close("all")


def test_core_families_define_conditional_standard_deviation():
    normal = NormalDistribution()
    normal_params = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(
        normal.predict_spread(normal_params, transformed=True), torch.tensor([2.0])
    )

    student = StudentTDistribution(min_df=0.0)
    student_params = torch.tensor([[4.0, 1.0, 2.0]])
    torch.testing.assert_close(
        student.predict_spread(student_params, transformed=True),
        torch.tensor([2.0 * np.sqrt(2.0)], dtype=torch.float32),
    )
    undefined = student.predict_spread(
        torch.tensor([[2.0, 1.0, 2.0]]), transformed=True
    )
    assert torch.isnan(undefined).all()


def test_distributional_effect_contract_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="identical term names"):
        DistributionalEffects(
            {"x": np.arange(3)},
            {"x": np.arange(3)},
            {"z": np.arange(3)},
        )

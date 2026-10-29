"""Normalized neural likelihoods versus shared laws and statistical jets."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

from nampy.neural.families import (
    BCCGDistribution,
    BetaBinomialDistribution,
    InflatedBetaDistribution,
    ZeroAdjustedGammaDistribution,
)
from nampy.neural.families.distribution_adapter import distribution_prediction


@pytest.mark.parametrize(
    "family,raw,y",
    [
        (
            BCCGDistribution(),
            [[0.1, -0.8, 0.0], [0.4, -0.3, 0.2], [-0.2, -0.5, -0.4]],
            [1.1, 1.4, 0.8],
        ),
        (
            BetaBinomialDistribution(trials=8),
            [[0.1, 0.3], [0.4, -0.3], [-0.2, -0.5]],
            [0.0, 3.0, 8.0],
        ),
        (
            ZeroAdjustedGammaDistribution(),
            [[0.1, -0.8, 0.2], [0.4, -0.3, 0.2], [-0.2, -0.5, -0.4]],
            [0.0, 1.4, 0.8],
        ),
        (
            InflatedBetaDistribution(),
            [[0.1, 1.8, 0.2, 0.3], [0.4, 1.3, 0.2, -0.2], [-0.2, 0.5, -0.4, 0.2]],
            [0.0, 0.4, 1.0],
        ),
    ],
)
def test_normalized_loss_probability_and_autograd(family, raw, y):
    raw = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float64)
    loss = family.compute_loss(raw, y, reduction="none")
    law = distribution_prediction(family, family(raw).detach().numpy())
    assert_allclose(-loss.detach(), law.log_prob(y.numpy()), atol=2e-12, rtol=2e-12)
    assert torch.autograd.gradcheck(
        lambda value: family.compute_loss(value, y),
        (raw,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-5,
    )
    assert_allclose(family.predict_point(raw).detach(), law.mean(), atol=2e-12)
    assert_allclose(
        family.predict_spread(raw).detach(), np.sqrt(law.variance()), atol=2e-12
    )


def test_bccg_density_matches_original_gamlss_capture():
    source = pd.read_csv(
        Path(__file__).parents[1] / "reference_fixtures/bccg/probability.csv"
    )
    raw = np.column_stack([np.log(source.mu), np.log(source.sigma), source.nu])
    loss = BCCGDistribution().compute_loss(
        torch.tensor(raw), torch.tensor(source.y.to_numpy()), reduction="none"
    )
    assert_allclose(-loss.detach(), source.log_prob, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("family", ["bccg", "zaga", "beinf", "betabinomial"])
def test_fitted_gamlss_neural_objectives_persist_and_reconstruct(family, tmp_path):
    from nampy import ObjectiveConfig, SemiStructuredLSS, SemiStructuredTrainingConfig
    from nampy.neural.configs import DefaultSemiStructuredConfig

    rng = np.random.default_rng(171)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 80)})
    y = np.exp(0.2 + 0.3 * X.x.to_numpy() + rng.normal(0, 0.2, 80))
    kwargs = None
    if family == "zaga":
        y[::3] = 0
    elif family == "beinf":
        y = rng.beta(2, 4, 80)
        y[::5] = 0
        y[1::5] = 1
    elif family == "betabinomial":
        y = rng.binomial(8, 0.4, 80)
        kwargs = {"trials": 8}
    model = SemiStructuredLSS(
        objective=ObjectiveConfig(family=family, distributional_kwargs=kwargs),
        architecture_config=DefaultSemiStructuredConfig(
            structured_formula="~x", hidden_dims=(3,), representation_dim=2
        ),
        training=SemiStructuredTrainingConfig(epochs=3, random_state=2),
    ).fit(X.iloc[:60], y[:60], X_val=X.iloc[60:], y_val=y[60:])
    components = model.predict_components(X)
    raw_width = 4 if family == "beinf" else 2 if family == "betabinomial" else 3
    assert components.link.shape == (80, raw_width)
    natural = model.predict(X)
    assert natural.shape == (80, 5 if family == "beinf" else raw_width)
    assert np.isfinite(model.predict_distribution(X).log_prob(y)).all()
    assert np.isfinite(model.score(X, y))
    restored = model.load_model(model.save_model(tmp_path / "family.nampy"))
    assert_allclose(
        restored.predict_distribution(X).quantile([0.2, 0.5, 0.8]),
        model.predict_distribution(X).quantile([0.2, 0.5, 0.8]),
        rtol=0,
        atol=0,
    )


def test_inflated_beta_float32_training_loss_keeps_coupled_masses_consistent():
    family = InflatedBetaDistribution()
    raw = torch.tensor(
        [[0.1, 1.0, -0.3, 0.2], [0.2, 0.5, 0.1, -0.2], [-0.1, 0.8, 0.2, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float32)
    loss = family.compute_loss(raw, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    law = distribution_prediction(family, family(raw).detach().numpy())
    assert_allclose(
        -loss.detach(), law.log_prob(target.numpy()).mean(), atol=2e-12, rtol=2e-12
    )

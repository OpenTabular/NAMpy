"""Neural inference adapters share probability semantics without changing losses."""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError

from nampy.gam.families.gamlss import gammals, gaulss
from nampy.gam.families.gamlss.distributional import (
    BetaLocationPrecision,
    NegativeBinomialLocationDispersion,
    StudentLocationScale,
)
from nampy.gam.predict.distribution_adapter import distribution_prediction as gam_prediction
from nampy.models import LinRegLSS
from nampy.neural.configuration import (
    GradientTrainingConfig,
    ObjectiveConfig,
    PreprocessingConfig,
)
from nampy.neural.families import BetaDistribution, GammaDistribution, NegativeBinomialDistribution, NormalDistribution, StudentTDistribution
from nampy.neural.objectives.quantile import Quantile
from nampy.neural.families.distribution_adapter import distribution_prediction


@pytest.mark.parametrize(
    "family_type,gam_family",
    [(NormalDistribution, gaulss), (GammaDistribution, gammals)],
)
def test_same_law_agrees_across_backends_and_neural_loss(family_type, gam_family):
    family = family_type()
    raw = torch.tensor([[0.3, -0.7], [1.5, 0.2]], requires_grad=True)
    natural = family(raw).detach().numpy()
    if family_type is NormalDistribution:
        statistical_parameters = natural
    else:
        shape, rate = natural.astype(float).T
        statistical_parameters = np.column_stack([shape / rate, 1 / np.sqrt(shape)])
    neural = distribution_prediction(family, natural)
    statistical = gam_prediction(gam_family(), statistical_parameters)
    y = torch.tensor([0.6, 1.8])
    for name in ["log_prob", "cdf", "cdf_left", "log_cdf", "survival", "log_survival"]:
        assert_allclose(
            getattr(neural, name)(y.numpy()),
            getattr(statistical, name)(y.numpy()),
            rtol=2e-7,
        )
    assert_allclose(
        neural.quantile([0.1, 0.9]), statistical.quantile([0.1, 0.9]), rtol=2e-7
    )
    assert_array_equal(
        neural.sample(8, random_state=3), neural.sample(8, random_state=3)
    )
    loss = family.compute_loss_per_sample(raw, y)
    assert_allclose(-neural.log_prob(y.numpy()), loss.detach().numpy(), rtol=2e-6)
    loss.sum().backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize(
    "family",
    ["normal", "gamma", "studentt", "beta", "negativebinom", "lognormal", "weibull"],
)
def test_fitted_neural_estimator_transforms_once_and_honors_batching(family, tmp_path):
    rng = np.random.default_rng(6)
    X = pd.DataFrame({"x": rng.normal(size=32)})
    y = {
        "normal": rng.normal(size=32),
        "gamma": rng.gamma(2.0, size=32),
        "studentt": rng.standard_t(5.0, size=32),
        "beta": rng.beta(2.0, 3.0, size=32),
        "negativebinom": rng.poisson(2.0, size=32),
        "lognormal": rng.lognormal(-0.2, 0.4, size=32),
        "weibull": 1.2 * rng.weibull(1.8, size=32),
    }[family]
    estimator = LinRegLSS(objective=ObjectiveConfig(family=family), preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'), training=GradientTrainingConfig(max_epochs=1, patience=1, random_state=7, checkpoint_path=str(tmp_path), trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False, 'num_sanity_val_steps': 0})).fit(X, y)
    subset = X.iloc[[11, 2, 24]]
    before = estimator.predict(subset, batch_size=2)
    prediction = estimator.predict_distribution(subset, batch_size=2)
    point = prediction.parameters["loc"] if family == "studentt" else prediction.mean()
    assert_allclose(point, estimator.predict_point(subset, batch_size=2), rtol=2e-7)
    assert_array_equal(estimator.predict(subset, batch_size=2), before)
    assert_allclose(
        prediction.log_prob(y[[11, 2, 24]]).mean(),
        estimator.score(subset, y[[11, 2, 24]]),
        rtol=2e-6,
    )
    assert prediction.quantile([0.1, 0.9]).shape == (3, 2)
    assert np.isfinite(
        prediction.quantile_residuals(y[[11, 2, 24]], random_state=7).residuals
    ).all()


def test_neural_unsupported_and_unfitted_are_explicit():
    with pytest.raises(NotFittedError):
        LinRegLSS().predict_distribution([[0.0]])
    with pytest.raises(NotImplementedError, match="Quantile.*Supported families"):
        distribution_prediction(Quantile(), np.ones((2, 3)))
    from nampy.neural.families import BCCGDistribution
    from nampy.neural.families.registry import resolve_family

    family, _ = resolve_family("bccg").instantiate(np.ones(2))
    assert isinstance(family, BCCGDistribution)
    assert distribution_prediction(family, [[1.0, 0.2, 0.0]]).mean().shape == (1,)
    for name in ['bccg_lss', 'bct_lss', 'bcpe_lss']:
        with pytest.raises(ValueError, match='Unknown distribution family'):
            resolve_family(name)


def test_custom_neural_transforms_are_respected():
    family = NormalDistribution(mean_transform="exp", scale_transform="exp")
    raw = torch.tensor([[0.3, 0.8]])
    natural = family(raw).detach().numpy()
    prediction = distribution_prediction(family, natural)
    assert_allclose(prediction.mean(), natural[:, 0])
    assert_allclose(prediction.variance(), natural[:, 1] ** 2)


@pytest.mark.parametrize(
    "family", [StudentTDistribution, BetaDistribution, NegativeBinomialDistribution]
)
def test_extended_adapters_agree_between_backends_and_training_loss(family):
    instance = family()
    raw = torch.tensor(
        [[3.0, 0.2, -0.4], [4.0, 0.5, 0.2]]
        if family is StudentTDistribution
        else [[0.3, -0.7], [1.5, 0.2]],
        requires_grad=True,
    )
    natural = instance(raw).detach().numpy()
    if family is StudentTDistribution:
        # The statistical family has fixed df; compare each row at its neural df.
        statistical = [
            gam_prediction(StudentLocationScale(df=float(row[0])), [row[1:]])
            for row in natural
        ]
    elif family is BetaDistribution:
        alpha, beta = natural.astype(float).T
        statistical = gam_prediction(
            BetaLocationPrecision(),
            np.column_stack([alpha / (alpha + beta), alpha + beta]),
        )
    else:
        statistical = gam_prediction(NegativeBinomialLocationDispersion(), natural)
    prediction = distribution_prediction(instance, natural)
    y = (
        torch.tensor([0.0, 3.0])
        if family is NegativeBinomialDistribution
        else torch.tensor([0.3, 0.7])
    )
    if family is StudentTDistribution:
        for i, reference in enumerate(statistical):
            assert_allclose(
                prediction.log_prob(y.numpy())[i], reference.log_prob([y[i].item()])[0]
            )
    else:
        assert_allclose(
            prediction.log_prob(y.numpy()), statistical.log_prob(y.numpy()), atol=1e-13
        )
        assert_allclose(
            prediction.quantile([0.1, 0.9]),
            statistical.quantile([0.1, 0.9]),
            atol=1e-13,
        )
    loss = instance.compute_loss_per_sample(raw, y)
    assert_allclose(
        -prediction.log_prob(y.numpy()), loss.detach().numpy(), atol=2e-6, rtol=2e-6
    )
    loss.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_neural_beta_probability_queries_do_not_copy_training_target_clipping():
    family = BetaDistribution()
    raw = torch.tensor([[2.0, 2.0]])
    natural = family(raw).detach().numpy()
    prediction = distribution_prediction(family, natural)
    assert torch.isfinite(
        family.compute_loss_per_sample(raw, torch.tensor([0.0]))
    ).all()
    assert np.isneginf(prediction.log_prob([0.0])[0])
    assert prediction.cdf([0.0])[0] == 0


def test_clipped_neural_nb_regime_is_explicit_and_missing_rows_are_preserved():
    family = NegativeBinomialDistribution()
    with pytest.raises(NotImplementedError, match="clamped neural"):
        distribution_prediction(family, [[1e-12, 1.0]])
    prediction = distribution_prediction(family, [[1.0, 0.5], [np.nan, np.nan]])
    assert_allclose(prediction.mean(), [1.0, np.nan])

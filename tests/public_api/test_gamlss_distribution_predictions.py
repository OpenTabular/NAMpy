"""Statistical adapters preserve natural-parameter and mgcv semantics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import GAMLSS
from nampy.gam.families.gamlss import GaulssFamily, GumblsFamily, gammals, gaulss
from nampy.gam.predict.distribution_adapter import distribution_prediction


@pytest.mark.parametrize("family", ["normal", "gamma"])
def test_adapted_log_prob_matches_existing_r_gamlss_fixtures(family):
    path = Path(__file__).parents[1] / "reference_fixtures" / "gamlss" / f"{family}.csv"
    fixture = pd.read_csv(path)
    parameters = fixture[["gamlss_mu", "gamlss_sigma"]].to_numpy()
    adapted = distribution_prediction(
        gaulss() if family == "normal" else gammals(), parameters
    )
    assert_allclose(
        adapted.log_prob(fixture["y"]), fixture["gamlss_logpdf"], atol=2e-12
    )
    assert_allclose(adapted.mean(), parameters[:, 0], rtol=1e-13)


@pytest.mark.parametrize("family", ["normal", "gamma"])
def test_fitted_distribution_preserves_predictions_offsets_and_score(family):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 80)})
    frame["y"] = (
        0.3 + frame.x + rng.normal(size=80)
        if family == "normal"
        else rng.gamma(4.0, np.exp(0.3 + frame.x) / 4.0)
    )
    model = GAMLSS(
        family=family, formula={"mu": "y ~ x", "sigma": "~ 1"}, optimize_smoothing=False
    ).fit(frame)
    subset = frame.iloc[[21, 3, 55]].copy()
    offset = [np.array([0.1, -0.2, 0.3]), np.array([-0.1, 0.1, 0.2])]
    before = model.predict(subset, offset=offset)
    raw_before = model.gam_.predict(subset, type="response", offset=offset)
    prediction = model.predict_distribution(subset, offset=offset, block_size=2)
    assert_allclose(prediction.mean(), before[:, 0])
    expected_variance = before[:, 1] ** 2
    if family == "gamma":
        expected_variance *= before[:, 0] ** 2
    assert_allclose(prediction.variance(), expected_variance)
    assert_array_equal(model.predict(subset, offset=offset), before)
    assert_array_equal(
        model.gam_.predict(subset, type="response", offset=offset), raw_before
    )
    assert_allclose(
        prediction.log_prob(subset.y),
        model.gam_.family.logpdf_from_parameters(subset.y, before),
        rtol=1e-12,
    )
    training = model.predict_distribution()
    assert training.n_rows == len(frame)
    assert_allclose(
        training.log_prob(frame.y).mean(), model.score(frame, frame.y), atol=1e-12
    )
    subset.iloc[1, subset.columns.get_loc("x")] = np.nan
    missing = model.predict_distribution(subset, na_action="pass")
    assert missing.n_rows == 3
    assert np.isnan(missing.mean()[1])
    assert np.isfinite(missing.mean()[[0, 2]]).all()


def test_unfitted_or_unsupported_statistical_adapter_is_explicit():
    with pytest.raises(ValueError, match="not fitted"):
        GAMLSS().predict_distribution([[0.0]])
    with pytest.raises(NotImplementedError, match="GumblsFamily.*Supported families"):
        distribution_prediction(GumblsFamily(), np.ones((2, 2)))

    class ChangedNormal(GaulssFamily):
        pass

    with pytest.raises(NotImplementedError, match="ChangedNormal"):
        distribution_prediction(ChangedNormal(), np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        distribution_prediction(gammals(), [[1.0, np.nan]])


def test_student_adapter_preserves_existing_statistical_residual_dispatch():
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 40)})
    frame["y"] = frame.x + np.random.default_rng(7).normal(size=len(frame))
    model = GAMLSS(
        family="student_lss",
        formula={"mu": "y ~ x", "sigma": "~ 1"},
        optimize_smoothing=False,
    ).fit(frame)
    before = model.predict(frame)
    score = model.score(frame, frame.y)
    distribution = model.predict_distribution(frame)
    assert_allclose(distribution.log_prob(frame.y).mean(), score)
    assert np.isfinite(distribution.quantile_residuals(frame.y).residuals).all()
    with pytest.raises(ValueError, match="Supported residual types"):
        model.gam_.residuals(type="quantile")
    assert_array_equal(model.predict(frame), before)
    assert model.score(frame, frame.y) == score

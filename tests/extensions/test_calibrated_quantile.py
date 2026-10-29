"""qgam calFast covariance/criterion and fixed-penalty calibrated endpoints."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from nampy.gam.design import compile_design
from nampy.gam.losses.quantile_calibration import quantile_calibration_at_scale
from nampy.models.calibrated_quantile import CalibratedQuantileGAM


@pytest.mark.parametrize("k", [4, 10])
def test_calfast_profile_and_calibrated_endpoint_match_qgam(k, tmp_path):
    root = Path(__file__).parents[1] / "reference_fixtures"
    stem = "qgam_calfast_fixed" + ("" if k == 4 else f"_k{k}")
    source = pd.read_csv(root / f"{stem}_input.csv")
    profile = pd.read_csv(root / f"{stem}_profile.csv")
    endpoint = pd.read_csv(root / f"{stem}.csv")
    formula = f"~s(x,k={k})+z"
    X = source[["x", "z"]]
    design = compile_design(X, source.y, formula=formula, smoothing_params=[0.5])
    for log_scale in (-2.0, -1.0, 0.0, 1.0):
        fitted, sigma, V, VS, loss = quantile_calibration_at_scale(
            design.matrix,
            source.y.to_numpy(),
            design.penalty,
            level=0.3,
            bandwidth=0.2,
            log_scale=log_scale,
            max_iter=2000,
            tol=1e-10,
        )
        expected = profile[profile.log_sigma == log_scale]
        assert_allclose(fitted.fitted, expected.fitted, atol=2e-6, rtol=2e-6)
        assert_allclose(
            np.einsum("np,pq,nq->n", design.matrix, V, design.matrix),
            expected.posterior_variance,
            atol=2e-6,
            rtol=2e-6,
        )
        assert_allclose(
            np.einsum("np,pq,nq->n", design.matrix, VS, design.matrix),
            expected.sandwich_variance,
            atol=2e-6,
            rtol=2e-6,
        )
        assert_allclose(loss, expected.loss.iloc[0], atol=2e-6, rtol=2e-6)
    model = CalibratedQuantileGAM(
        formula=formula,
        level=0.3,
        bandwidth=0.2,
        smoothing_params=[0.5],
        log_scale_bounds=(-3, 2),
        tol=1e-10,
        calibration_tol=1e-7,
        max_iter=2000,
    ).fit(X, source.y)
    assert_allclose(
        np.log(model.loss_scale_), endpoint.log_sigma.iloc[0], atol=2e-4, rtol=2e-4
    )
    assert_allclose(model.predict(X), endpoint.fitted, atol=2e-5, rtol=2e-5)
    interval = model.predict_interval(X)
    assert_allclose(
        interval.standard_error**2, endpoint.posterior_variance, atol=2e-5, rtol=2e-5
    )
    assert interval.uncertainty_kind == "calFast_calibrated_quantile_function"
    assert interval.coverage == "approximate_pointwise_fixed_smoothing"
    restored = CalibratedQuantileGAM.load_model(
        model.save_model(tmp_path / f"calibrated-{k}.nampy")
    )
    assert_array_equal(restored.predict_interval(X).lower, interval.lower)
    assert clone(model).bandwidth == model.bandwidth
    model.set_params(level=0.4)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(X)


def test_calibrated_quantile_failed_refit_and_unsupported_routes():
    X = pd.DataFrame({"x": np.linspace(-1, 1, 40)})
    y = X.x.to_numpy() + 0.3 * np.sin(np.arange(40) * 1.7)
    model = CalibratedQuantileGAM(formula="~x", bandwidth=0.2).fit(X, y)
    before = model.predict(X)
    with pytest.raises(NotImplementedError, match="unweighted"):
        model.fit(X, y, sample_weight=np.full(40, 2.0))
    assert_array_equal(model.predict(X), before)
    with pytest.raises(ValueError, match="bandwidth"):
        CalibratedQuantileGAM(formula="~x", bandwidth=0).fit(X, y)
    with pytest.raises(NotImplementedError, match="bound"):
        CalibratedQuantileGAM(
            formula="~x", bandwidth=0.2, log_scale_bounds=(3.0, 4.0)
        ).fit(X, y)


def test_calfast_pointwise_intervals_in_independent_median_simulation():
    """A prespecified 90% pointwise coverage check, separate from R parity.

    Symmetric errors give a known linear conditional median. The exact
    binomial test uses the independent simulation count, not training rows.
    This checks one model/design regime, not a universal coverage guarantee.
    """
    from scipy.stats import binomtest

    rng = np.random.default_rng(821)
    covered = 0
    replications = 100
    query = pd.DataFrame({"x": [0.0]})
    for _ in range(replications):
        X = pd.DataFrame({"x": rng.uniform(-1, 1, 120)})
        y = 0.4 + 0.6 * X.x.to_numpy() + rng.normal(0, 0.5, len(X))
        model = CalibratedQuantileGAM(
            formula="~x",
            level=0.5,
            bandwidth=0.1,
            log_scale_bounds=(-4, 2),
        ).fit(X, y)
        interval = model.predict_interval(query, level=0.9)
        covered += bool(interval.lower[0] <= 0.4 <= interval.upper[0])
    assert binomtest(covered, replications, p=0.9).pvalue > 0.001, covered

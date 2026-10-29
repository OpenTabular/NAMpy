"""Baseline-aware mgcv uncertainty, source diagnostics and Cox subject bootstrap."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import CoxGAM
from nampy.survival import SurvivalTarget, bootstrap_survival


@pytest.mark.parametrize("stratified", [False, True])
def test_cox_baseline_coefficient_se_and_diagnostics_match_mgcv(stratified, tmp_path):
    root = Path(__file__).parents[1] / "reference_fixtures"
    data = pd.read_csv(root / "cox_fixed_input.csv")
    reference = pd.read_csv(root / "cox_uncertainty.csv")
    residuals = pd.read_csv(root / "cox_diagnostics.csv")
    reference, residuals = (
        reference[reference.stratified == stratified],
        residuals[residuals.stratified == stratified],
    )
    X, y = data[["x", "z"]], SurvivalTarget(data.time, data.event)
    fit_options = {
        "offset": data.offset.to_numpy(),
        "strata": data.stratum if stratified else None,
    }
    model = CoxGAM(formula="~x+z", tol=1e-8).fit(X, y, **fit_options)
    query = pd.DataFrame({"x": [-0.5, 0.2, 0.8], "z": [0.1, -0.3, 0.7]})
    options = {"offset": np.zeros(3), "strata": [1, 2, 1] if stratified else None}
    assert_allclose(
        model.predict_standard_error(query, times=[0.5, 1, 2], **options).ravel(),
        reference.se,
        rtol=2e-7,
        atol=2e-7,
    )
    interval = model.predict_interval(query, times=[0.5, 1, 2], **options)
    assert interval.uncertainty_kind == "cox_baseline_and_coefficient"
    assert_allclose(interval.estimate.ravel(), reference.survival, rtol=2e-7, atol=2e-7)
    diagnostic = model.diagnostics(X, y, **fit_options)
    assert_allclose(
        diagnostic.schoenfeld,
        residuals[["schoenfeld1", "schoenfeld2"]],
        rtol=2e-7,
        atol=2e-7,
        equal_nan=True,
    )
    assert_allclose(
        diagnostic.cumulative_score,
        residuals[["score1", "score2"]],
        rtol=2e-7,
        atol=2e-7,
    )
    if stratified:
        # Source postproc indexes a stratum-local r into global h for fitted
        # values. Public response prediction uses the correct stratum baseline.
        residuals = pd.read_csv(root / "cox_diagnostics_consistent.csv")
    assert_allclose(diagnostic.martingale, residuals.martingale, rtol=2e-7, atol=2e-7)
    assert_allclose(diagnostic.deviance, residuals.deviance, rtol=2e-7, atol=2e-7)
    restored = CoxGAM.load_model(model.save_model(tmp_path / "cox-inference.nampy"))
    assert_array_equal(
        restored.predict_interval(query, times=[0.5, 1, 2], **options).lower,
        interval.lower,
    )


@pytest.mark.parametrize("stratified", [False, True])
def test_cox_subject_bootstrap_refits_baseline_and_preserves_stratum_context(
    stratified,
):
    root = Path(__file__).parents[1] / "reference_fixtures"
    data = pd.read_csv(root / "cox_fixed_input.csv")
    X, y = data[["x", "z"]], SurvivalTarget(data.time, data.event)
    strata = data.stratum.to_numpy() if stratified else None
    offset = data.offset.to_numpy()
    model = CoxGAM(formula="~x+z", tol=1e-8).fit(X, y, strata=strata, offset=offset)
    query = X.iloc[:2]
    plan = np.stack([np.arange(60), np.roll(np.arange(60), 3)])
    kwargs = (
        {
            "fit_context": {"strata": strata},
            "prediction_context": {"strata": strata[:2]},
        }
        if stratified
        else {}
    )
    result = bootstrap_survival(
        model,
        X,
        y,
        query,
        resample_indices=plan,
        offset=offset,
        prediction_offset=np.zeros(2),
        **kwargs,
    )
    assert not result.failures
    for indices, prediction in zip(plan, result.predictions, strict=True):
        reference = CoxGAM(formula="~x+z", tol=1e-8).fit(
            X.iloc[indices],
            SurvivalTarget(y.time[indices], y.event[indices]),
            offset=offset[indices],
            strata=strata[indices] if stratified else None,
        )
        assert_array_equal(
            prediction.survival([0.5, 1]),
            reference.predict_distribution(
                query, offset=np.zeros(2), strata=strata[:2] if stratified else None
            ).survival([0.5, 1]),
        )
    assert np.isfinite(
        result.percentile_interval(grid=[0.5, 1], estimand="rmst").lower
    ).all()
    assert np.isfinite(result.simultaneous_interval(grid=[0.5, 1]).upper).all()

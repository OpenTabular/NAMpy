"""Fitted lag marginals, query invariance, persistence and joint contrasts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from nampy import GAMRegressor
from nampy.gam.inference.functionals import linear_functional_interval
from nampy.gam.structured import DistributedLagBasis, exposure_history


def test_fitted_linear_lag_basis_replays_dlnm_reference_and_missing_boundaries(
    tmp_path,
):
    fixture = json.loads(
        (
            Path(__file__).parents[1] / "reference_fixtures/extensions/neighbors.json"
        ).read_text()
    )
    d = fixture["inputs"]
    history = np.asarray(d["history"])
    transformer = DistributedLagBasis(
        max_lag=3, exposure_formula="~exposure+I(exposure**2)", lag_formula="~lag"
    ).fit(history)
    actual = transformer.transform(history)
    assert_allclose(actual, fixture["upstream"]["cross_basis"], atol=1e-13)
    assert_allclose(transformer.transform(history[:1]), actual[:1], atol=0, rtol=0)
    restored = transformer.load_model(transformer.save_model(tmp_path / "lag.nampy"))
    assert_array_equal(restored.transform(history), actual)
    assert transformer.exposure_design_.matrix.shape[0] == 0
    assert transformer.exposure_design_._compiled.design_matrix.shape[0] == 0
    assert clone(transformer).get_params() == transformer.get_params()
    h = exposure_history(d["series"], max_lag=3, groups=d["groups"])
    missing = ~transformer.complete_rows(h)
    assert np.isnan(transformer.transform(h)[missing]).all()
    assert np.isnan(transformer.transform(np.full((2, 4), np.nan))).all()
    with pytest.raises(ValueError, match="max_lag"):
        transformer.transform(np.ones((2, 3)))
    transformer.set_params(k=4)
    with pytest.raises(NotFittedError):
        transformer.transform(history)


def test_fitted_spline_history_basis_penalty_and_cumulative_covariance(tmp_path):
    rng = np.random.default_rng(751)
    H = rng.normal(size=(150, 5))
    transformer = DistributedLagBasis(
        max_lag=4, exposure_formula="~s(exposure,k=4)", lag_formula="~lag"
    ).fit(H)
    B = transformer.transform(H)
    S = transformer.penalty(exposure_weight=0.2, lag_weight=0.5)
    assert S.shape == (B.shape[1], B.shape[1])
    assert np.linalg.eigvalsh(S).min() > -1e-10
    assert_allclose(transformer.transform(H[:4]), B[:4], atol=0, rtol=0)
    # A full-rank linear marginal avoids redundant lag coordinates for the
    # integration check; nonlinear design/penalty persistence is tested above.
    linear = DistributedLagBasis(max_lag=4).fit(H)
    X = pd.DataFrame(linear.transform(H), columns=linear.get_feature_names_out())
    y = 0.4 + 0.3 * X.iloc[:, 0].to_numpy() + rng.normal(0, 0.2, len(H))
    model = GAMRegressor(formula="~dl_0", optimize_smoothing=False).fit(X, y)
    contrast = linear.cumulative_contrast(
        [-0.5, 0, 0.5], 0.0, coefficient_indices=[1], n_coefficients=2
    )
    result = linear_functional_interval(model, contrast, interval="pointwise")
    beta = model.gam_.fit_result().coef_full
    assert_allclose(result.estimate, contrast @ beta, atol=0, rtol=0)
    assert_allclose(
        result.standard_error,
        np.sqrt(np.einsum("np,pq,nq->n", contrast, model.gam_.vcov(), contrast)),
        atol=1e-14,
    )
    assert result.estimate[1] == 0 and result.standard_error[1] == 0
    restored = transformer.load_model(transformer.save_model(tmp_path / "spline.nampy"))
    assert_array_equal(restored.transform(H), B)
    assert_array_equal(restored.penalty(), transformer.penalty())
    with pytest.raises(ValueError):
        linear.cumulative_contrast([1], 0, coefficient_indices=[1, 1], n_coefficients=2)

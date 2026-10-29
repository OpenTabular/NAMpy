"""qgam residual-density gradient and adaptive bandwidth reference."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

from nampy.gam.losses.quantile_bandwidth import (
    _shash_value_gradient,
    estimate_quantile_bandwidth,
)


def test_shash_gradient_and_bandwidth_match_qgam():
    root = Path(__file__).parents[1] / "reference_fixtures"
    source = pd.read_csv(root / "qgam_bandwidth_input.csv")
    expected = pd.read_csv(root / "qgam_bandwidth.csv")
    parameter = np.array([0.1, 0.2, 0.15, -0.1])
    assert_allclose(
        _shash_value_gradient(parameter, source.residual.to_numpy())[1],
        approx_derivative(
            lambda p: np.array(
                [_shash_value_gradient(p, source.residual.to_numpy())[0]]
            ),
            parameter,
        ).ravel(),
        rtol=2e-6,
        atol=2e-6,
    )
    for row in expected.itertuples():
        bandwidth, error, parameters = estimate_quantile_bandwidth(
            source.residual.to_numpy(),
            level=row.level,
            variance=source.variance.iloc[0],
            effective_dimension=5,
        )
        assert_allclose(bandwidth, row.bandwidth, rtol=5e-4, atol=5e-5)
        assert_allclose(error, row.error, rtol=5e-4, atol=5e-5)
        # The nuisance SHASH parameters approach a flat limiting ridge on
        # these residuals. Compare the bandwidth estimand, not nonidentified
        # individual location/skew/tail coordinates. Both optimizers terminate.
        assert (
            _shash_value_gradient(parameters, source.residual.to_numpy())[0]
            <= _shash_value_gradient(
                np.array([row.p1, row.p2, row.p3, row.p4]), source.residual.to_numpy()
            )[0]
            + 1e-6
        )


def test_automatic_bandwidth_initial_gaussian_variance_penalty_edf_and_fitted_model():
    from nampy import CalibratedQuantileGAM
    from nampy.gam import GAM
    from nampy.gam.losses.quantile_bandwidth import initial_quantile_dimension

    root = Path(__file__).parents[1] / "reference_fixtures"
    data = pd.read_csv(root / "qgam_calfast_auto_nonlinear_input.csv")
    expected = pd.read_csv(root / "qgam_adaptive_bandwidth.csv")
    X = data[["x", "z"]]
    initial = GAM(
        formula="~s(x,k=4)+z", smoothing_method="reml", optimize_smoothing=True
    ).fit(X, data.y)
    assert_allclose(
        initial.fit_result().scale, expected.variance.iloc[0], rtol=2e-5, atol=2e-6
    )
    assert_allclose(
        initial_quantile_dimension(initial),
        expected.dimension.iloc[0],
        rtol=2e-5,
        atol=2e-6,
    )
    residual = (data.y - initial.predict(X)) / np.sqrt(initial.fit_result().scale)
    assert_allclose(residual, expected.residual, rtol=2e-5, atol=2e-5)
    model = CalibratedQuantileGAM(
        formula="~s(x,k=4)+z",
        level=0.3,
        bandwidth=None,
        optimize_smoothing=True,
        log_scale_bounds=(-4, 1),
    ).fit(X, data.y)
    assert model.bandwidth is None
    assert_allclose(model.bandwidth_, expected.bandwidth.iloc[0], rtol=5e-4, atol=5e-5)
    assert np.isfinite(model.predict_interval(X.iloc[:3]).lower).all()


def test_multilevel_fits_share_initial_preparation_and_preserve_caller(monkeypatch):
    from sklearn.exceptions import NotFittedError

    from nampy import CalibratedQuantileGAM

    root = Path(__file__).parents[1] / "reference_fixtures"
    data = pd.read_csv(root / "qgam_calfast_auto_nonlinear_input.csv")
    X = data[["x", "z"]]
    starter = CalibratedQuantileGAM(
        formula="~s(x,k=4)+z",
        bandwidth=None,
        smoothing_params=[0.02],
        log_scale_bounds=(-5, 2),
    )
    original = starter._prepare_bandwidth
    calls = []

    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(starter, "_prepare_bandwidth", prepare)
    models = starter.fit_levels(X, data.y, levels=[0.2, 0.5, 0.8])
    assert len(calls) == 1
    assert tuple(models) == (0.2, 0.5, 0.8)
    for level, model in models.items():
        assert model.level == level and model.bandwidth is None
        assert np.isfinite(model.predict(X)).all()
        model.predict_components(X).validate_additive_reconstruction()
    import pytest

    with pytest.raises(NotFittedError):
        starter.predict(X)

"""predict(type="iterms") crossings: families, structured terms, filters.

Upstream reference: mgcv/R/mgcv.r::predict.gam — "iterms" adds intercept
uncertainty to smooth-term standard errors; terms=/exclude= filter columns.
Compared through tests/mgcv_parity_utils._run_mgcv_predict_on_newdata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import (
    _make_binomial_data,
    _make_gaussian_data,
    _make_poisson_data,
    _run_mgcv_predict_on_newdata,
)

pytestmark = [pytest.mark.surface_output]


def _numeric_matrix(value) -> np.ndarray:
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]

    def _coerce(item):
        return np.nan if item is None or item == "NA" else float(item)

    return np.vectorize(_coerce, otypes=[np.float64])(arr)


def _newdata(data: pd.DataFrame, step: int = 17) -> pd.DataFrame:
    newdata = data.iloc[3::step].drop(columns=["y"]).copy()
    return newdata.reset_index(drop=True)


def _fs_data(seed=451, n=132) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.6, 1.6, size=n)
    row = np.arange(n)
    f = np.asarray(["a", "b", "c"])[row % 3]
    offsets = {"a": 0.4, "b": -0.2, "c": 0.1}
    y = (
        0.5 * np.sin(1.4 * x0)
        + np.asarray([offsets[v] for v in f])
        + rng.normal(scale=0.15, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "f": pd.Categorical(f)})


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "atol"),
    [
        pytest.param(
            "gaussian_fixed",
            "gaussian",
            'y ~ x1 + s(x0, bs="cr", k=8, sp=0.9)',
            _make_gaussian_data,
            "fixed",
            2e-8,
            id="gaussian_fixed",
        ),
        pytest.param(
            "poisson_reml",
            "poisson",
            'y ~ x1 + s(x0, bs="cr", k=8)',
            _make_poisson_data,
            "REML",
            2e-6,
            id="poisson_reml",
        ),
        pytest.param(
            "binomial_reml",
            "binomial",
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
            _make_binomial_data,
            "REML",
            2e-6,
            id="binomial_reml",
        ),
        pytest.param(
            "gaussian_fs_reml",
            "gaussian",
            'y ~ s(f, x0, bs="fs", k=5, xt="cr")',
            _fs_data,
            "REML",
            5e-5,
            id="gaussian_fs_reml",
        ),
        pytest.param(
            "gaussian_te_reml",
            "gaussian",
            'y ~ te(x0, x1, bs=["cr", "ps"], k=[5, 5])',
            _make_gaussian_data,
            "REML",
            5e-5,
            id="gaussian_te_reml",
        ),
    ],
)
def test_iterms_predictions_and_se_match_mgcv(
    case_id, family, formula, data_factory, method, atol
):
    """type="iterms" fit and SE columns match mgcv across families/terms."""
    data = data_factory()
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=(method != "fixed"),
        smoothing_method=method,
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = _newdata(data)
    actual, actual_se = gam.predict(newdata, type="iterms", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family=family,
        method=method,
        type="iterms",
        return_se=True,
        optimizer=None if method == "fixed" else "newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual),
        _numeric_matrix(expected["pred"]),
        atol=atol,
        rtol=atol,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual_se),
        _numeric_matrix(expected["se"]),
        atol=10.0 * atol,
        rtol=10.0 * atol,
    )


def test_iterms_with_unconditional_covariance_matches_mgcv():
    """iterms SE under Vc (unconditional=TRUE) matches mgcv."""
    formula = 'y ~ x1 + s(x0, bs="cr", k=8)'
    data = _make_gaussian_data()
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = _newdata(data)
    actual, actual_se = gam.predict(
        newdata,
        type="iterms",
        return_se=True,
        unconditional=True,
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="iterms",
        return_se=True,
        unconditional=True,
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual),
        _numeric_matrix(expected["pred"]),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual_se),
        _numeric_matrix(expected["se"]),
        atol=2e-5,
        rtol=2e-5,
    )


def test_iterms_with_exclude_filter_matches_mgcv():
    """exclude= composes with type="iterms" exactly as in mgcv."""
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    data = _make_gaussian_data()
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = _newdata(data)

    reference = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="iterms",
        return_se=True,
        optimizer="newton",
        allow_live_run=True,
    )
    names = reference["term_names"]
    excluded = str(names[0] if isinstance(names, list) else names)

    actual, actual_se = gam.predict(
        newdata, type="iterms", exclude=[excluded], return_se=True
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="iterms",
        return_se=True,
        exclude=[excluded],
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual),
        _numeric_matrix(expected["pred"]),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual_se),
        _numeric_matrix(expected["se"]),
        atol=2e-5,
        rtol=2e-5,
    )


def test_iterms_with_terms_filter_matches_mgcv():
    """terms= composes with type="iterms" exactly as in mgcv."""
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    data = _make_gaussian_data()
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = _newdata(data)

    reference = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="iterms",
        return_se=True,
        optimizer="newton",
        allow_live_run=True,
    )
    names = reference["term_names"]
    kept = str(names[-1] if isinstance(names, list) else names)

    actual, actual_se = gam.predict(
        newdata, type="iterms", terms=[kept], return_se=True
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="iterms",
        return_se=True,
        terms=[kept],
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual),
        _numeric_matrix(expected["pred"]),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual_se),
        _numeric_matrix(expected["se"]),
        atol=2e-5,
        rtol=2e-5,
    )

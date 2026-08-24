"""Newdata robustness: column order/extras, dtypes, one-row data, guards.

Upstream reference: mgcv/R/mgcv.r::predict.gam builds the prediction frame by
formula variable name, so column order and extra columns are irrelevant and
missing variables are an explicit error.  nampy mirrors this in
nampy/gam/data.py::coerce_formula_predict_inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.families.test_general_family_mgcv_parity import _gaulss_data
from tests.mgcv_parity_utils import (
    _make_gaussian_data,
    _run_mgcv_predict_on_newdata,
)

pytestmark = [pytest.mark.surface_output]

_FIXED_FORMULA = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'


def _fixed_gaussian_fit(data: pd.DataFrame) -> GAM:
    return GAM(
        family="gaussian",
        formula=_FIXED_FORMULA,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)


def _factor_offset_data(seed=471, n=120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    row = np.arange(n)
    f = np.asarray(["a", "b", "c"])[row % 3]
    off = 0.1 * np.sin(2.0 * x0)
    effects = {"a": 0.3, "b": -0.1, "c": 0.2}
    y = (
        0.5 * np.sin(1.5 * x0)
        + np.asarray([effects[v] for v in f])
        + off
        + rng.normal(scale=0.12, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "f": pd.Categorical(f), "off": off})


def test_reordered_and_extra_newdata_columns_match_canonical_and_mgcv():
    """Column order and extra columns do not change predictions (as in mgcv)."""
    data = _make_gaussian_data()
    gam = _fixed_gaussian_fit(data)
    newdata = data.iloc[2::15].drop(columns=["y"]).reset_index(drop=True)

    canonical, canonical_se = gam.predict(newdata, type="link", return_se=True)

    shuffled = newdata[["x1", "x0"]].copy()
    shuffled["unrelated"] = np.arange(len(shuffled), dtype=np.float64)
    reordered, reordered_se = gam.predict(shuffled, type="link", return_se=True)
    np.testing.assert_allclose(reordered, canonical, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reordered_se, canonical_se, rtol=0.0, atol=0.0)

    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _FIXED_FORMULA,
        family="gaussian",
        method="fixed",
        type="link",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        canonical,
        np.asarray(expected["pred"], dtype=np.float64).ravel(),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        canonical_se,
        np.asarray(expected["se"], dtype=np.float64).ravel(),
        atol=1e-10,
        rtol=1e-10,
    )


def test_single_row_newdata_matches_mgcv():
    """A one-row prediction frame keeps exact parity."""
    data = _make_gaussian_data()
    gam = _fixed_gaussian_fit(data)
    newdata = data.iloc[[7]].drop(columns=["y"]).reset_index(drop=True)
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _FIXED_FORMULA,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual,
        np.asarray(expected["pred"], dtype=np.float64).ravel(),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_se,
        np.asarray(expected["se"], dtype=np.float64).ravel(),
        atol=1e-10,
        rtol=1e-10,
    )


def test_plain_string_factor_column_matches_categorical_newdata():
    """Character factor columns coerce like mgcv's character->factor rule."""
    data = _factor_offset_data()
    formula = 'y ~ f + offset(off) + s(x0, bs="cr", k=8, sp=0.9)'
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data.iloc[1::11].drop(columns=["y"]).reset_index(drop=True)
    canonical = gam.predict(newdata, type="link")

    as_strings = newdata.copy()
    as_strings["f"] = np.asarray([str(v) for v in as_strings["f"]], dtype=object)
    np.testing.assert_allclose(
        gam.predict(as_strings, type="link"), canonical, rtol=0.0, atol=0.0
    )

    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="link",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        canonical,
        np.asarray(expected["pred"], dtype=np.float64).ravel(),
        atol=1e-10,
        rtol=1e-10,
    )


def test_missing_formula_column_raises_key_error():
    """Missing prediction variables are an explicit error, as in mgcv."""
    data = _make_gaussian_data()
    gam = _fixed_gaussian_fit(data)
    newdata = data.iloc[:5].drop(columns=["y", "x1"]).reset_index(drop=True)
    with pytest.raises(KeyError, match="missing formula columns"):
        gam.predict(newdata, type="link")


def test_missing_offset_column_raises_key_error():
    """A formula offset column must be present on newdata."""
    data = _factor_offset_data()
    formula = 'y ~ f + offset(off) + s(x0, bs="cr", k=8, sp=0.9)'
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data.iloc[:5].drop(columns=["y", "off"]).reset_index(drop=True)
    with pytest.raises(KeyError, match="missing formula offset column"):
        gam.predict(newdata, type="link")


def test_numeric_nan_defaults_to_na_pass_while_inf_remains_invalid():
    """predict.gam's default pass restores NaN rows; Inf remains invalid."""
    data = _make_gaussian_data()
    gam = _fixed_gaussian_fit(data)
    newdata = data.iloc[:5].drop(columns=["y"]).reset_index(drop=True)
    newdata.loc[newdata.index[2], "x0"] = np.nan
    predicted = gam.predict(newdata, type="link")
    assert predicted.shape == (5,)
    assert np.isnan(predicted[2])
    assert np.isfinite(np.delete(predicted, 2)).all()

    newdata.loc[newdata.index[2], "x0"] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        gam.predict(newdata, type="link")


def test_formula_mode_requires_dataframe_newdata():
    """Formula-based prediction rejects raw arrays explicitly."""
    data = _make_gaussian_data()
    gam = _fixed_gaussian_fit(data)
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        gam.predict(data[["x0", "x1"]].to_numpy(), type="link")


def test_multi_predictor_offset_lists_on_newdata_match_mgcv():
    """Per-linear-predictor offsets carry to newdata like mgcv (gaulss)."""
    data = _gaulss_data()
    rng = np.random.default_rng(473)
    data = data.copy()
    data["off1"] = 0.05 * np.sin(2.0 * data["x"].to_numpy(dtype=np.float64))
    data["off2"] = 0.04 * np.cos(1.5 * data["x"].to_numpy(dtype=np.float64))
    del rng
    formula = [
        'y ~ s(x, bs="cr", k=6) + offset(off1)',
        "~ 1 + offset(off2)",
    ]
    gam = GAM(
        family="gaulss",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="ML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = data.iloc[3::9].drop(columns=["y"]).reset_index(drop=True)
    actual = np.asarray(gam.predict(newdata, type="link"), dtype=np.float64)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaulss",
        method="ML",
        type="link",
        allow_live_run=True,
    )
    expected_values = np.asarray(expected["pred"], dtype=np.float64)
    if expected_values.ndim == 1:
        expected_values = expected_values.reshape(actual.shape)
    np.testing.assert_allclose(actual, expected_values, atol=5e-6, rtol=5e-6)

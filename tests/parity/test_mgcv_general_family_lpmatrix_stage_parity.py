from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam._model_state import _coerce_feature_matrix, _predictor_designs
from nampy.gam.predict.general import (
    build_general_lpmatrix,
    general_family_prediction_layout,
)
from tests.families.test_general_family_mgcv_parity import (
    _gaulss_by_data,
    _general_newdata,
)
from tests.mgcv_parity_utils import _fit_nampy_model, _run_mgcv_predict_on_newdata

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]

_LPMATRIX_STAGE_CASES = [
    pytest.param(
        "gaulss_numeric_by",
        "gaulss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gaulss_by_data,
        "ML",
        5e-6,
        id="gaulss_numeric_by",
    ),
]

_MGCV_LPMATRIX_STAGE_CASES = list(_LPMATRIX_STAGE_CASES)


def _predictor_block_without_intercept(pred, X_new_np: np.ndarray) -> np.ndarray:
    return np.asarray(pred.build_new_matrix(X_new_np), dtype=np.float64)


def _expected_predictor_block(expected_lpmatrix: np.ndarray, pred, sl: slice) -> np.ndarray:
    block = np.asarray(expected_lpmatrix[:, sl], dtype=np.float64)
    if getattr(pred, "prediction_has_intercept", pred.has_intercept):
        np.testing.assert_allclose(
            block[:, :1],
            np.ones((block.shape[0], 1), dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )
        return block[:, 1:]
    return block


def _term_prediction_parameterization(term, X_new_np: np.ndarray) -> np.ndarray:
    metadata = dict(getattr(term, "metadata", {}) or {})
    if bool(metadata.get("expose_raw_prediction_basis", False)):
        return np.asarray(
            term.prediction_parameterization_matrix(X_new_np), dtype=np.float64
        )
    return np.asarray(term.predict_matrix(X_new_np), dtype=np.float64)


def _gaulss_factor_data(seed=241, n=120):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.25, 1.25, n)
    f = np.asarray(["a", "b", "c"], dtype=object)[np.arange(n) % 3]
    mu = 0.25 + np.sin(np.pi * x)
    mu = mu + np.where(f == "b", 0.35, 0.0) - np.where(f == "c", 0.2, 0.0)
    sigma = np.exp(-0.35 + 0.15 * x)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x": x, "f": f})


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "_atol"),
    _LPMATRIX_STAGE_CASES,
)
def test_general_family_lpmatrix_layout_matches_public_prediction_surface(
    case_id,
    family,
    formula,
    data_factory,
    method,
    _atol,
):
    """Verify that general family lpmatrix layout matches public prediction surface."""
    del case_id, _atol
    data = data_factory()
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(data, formula, family, method)
    X_new_np = _coerce_feature_matrix(gam, newdata, none_is_training=False)

    layout = general_family_prediction_layout(gam, X_new_np)
    actual = np.asarray(gam.predict(newdata, type="lpmatrix"), dtype=np.float64)
    direct = np.asarray(build_general_lpmatrix(gam, X_new_np), dtype=np.float64)

    np.testing.assert_allclose(layout.lpmatrix, actual, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(direct, actual, atol=1e-12, rtol=1e-12)

    for pred, sl in zip(
        _predictor_designs(gam), layout.predictor_slices, strict=True
    ):
        expected_block = np.asarray(actual[:, sl], dtype=np.float64)
        actual_block = _predictor_block_without_intercept(pred, X_new_np)
        if getattr(pred, "prediction_has_intercept", pred.has_intercept):
            np.testing.assert_allclose(
                expected_block[:, :1],
                np.ones((expected_block.shape[0], 1), dtype=np.float64),
                atol=1e-12,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                actual_block,
                expected_block[:, 1:],
                atol=1e-12,
                rtol=1e-12,
            )
        else:
            np.testing.assert_allclose(
                actual_block,
                expected_block,
                atol=1e-12,
                rtol=1e-12,
            )


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method", "pred_atol"),
    _MGCV_LPMATRIX_STAGE_CASES,
)
def test_general_family_single_term_predictor_blocks_match_mgcv_lpmatrix_slices(
    case_id,
    family,
    formula,
    data_factory,
    method,
    pred_atol,
):
    """
    Verify that general family single term predictor blocks match mgcv lpmatrix slices.
    """
    del case_id
    data = data_factory()
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(data, formula, family, method)
    X_new_np = _coerce_feature_matrix(gam, newdata, none_is_training=False)

    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family=family,
        method=method,
        type="lpmatrix",
        return_se=False,
    )
    expected_lpmatrix = np.asarray(expected["pred"], dtype=np.float64)
    actual_lpmatrix = np.asarray(gam.predict(newdata, type="lpmatrix"), dtype=np.float64)
    tol = max(1e-8, float(pred_atol))

    np.testing.assert_allclose(
        actual_lpmatrix,
        expected_lpmatrix,
        atol=tol,
        rtol=tol,
    )

    predictors = _predictor_designs(gam)
    predictor_slices = tuple(gam.gam_result_.compiled_model.predictor_full_slices)
    first_pred = predictors[0]
    first_slice = predictor_slices[0]
    assert len(first_pred.compiled_terms) == 1

    actual_first_term = _term_prediction_parameterization(
        first_pred.compiled_terms[0],
        X_new_np,
    )
    expected_first_term = _expected_predictor_block(
        expected_lpmatrix,
        first_pred,
        first_slice,
    )
    np.testing.assert_allclose(
        actual_first_term,
        expected_first_term,
        atol=tol,
        rtol=tol,
    )

    for pred, sl in zip(predictors[1:], predictor_slices[1:], strict=True):
        assert len(pred.compiled_terms) == 0
        expected_block = np.asarray(expected_lpmatrix[:, sl], dtype=np.float64)
        assert getattr(pred, "prediction_has_intercept", pred.has_intercept)
        np.testing.assert_allclose(
            expected_block,
            np.ones_like(expected_block, dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )


def test_general_family_factor_level_lpmatrix_matches_mgcv():
    """Verify that general family factor level lpmatrix matches mgcv."""
    data = _gaulss_factor_data()
    newdata = _general_newdata(data)
    newdata["f"] = pd.Series(["a", "b", "c"] * 11)[: len(newdata)].to_numpy(dtype=object)
    formula = ['y ~ f + s(x, bs="cr", k=6)', "~ 1"]
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaulss",
        method="ML",
        type="lpmatrix",
        return_se=False,
    )
    actual = np.asarray(gam.predict(newdata, type="lpmatrix"), dtype=np.float64)

    first_predictor = _predictor_designs(gam)[0]
    assert [term.label for term in first_predictor.compiled_terms] == ["f[b]", "f[c]", 's(x, bs="cr", k=6)']
    np.testing.assert_allclose(
        actual,
        np.asarray(expected["pred"], dtype=np.float64),
        atol=5e-6,
        rtol=5e-6,
    )


def test_general_family_lpmatrix_rejects_numeric_na_newdata_explicitly():
    """Verify that general family lpmatrix rejects numeric na new-data explicitly."""
    data = _gaulss_factor_data(seed=245)
    gam = _fit_nampy_model(
        data,
        ['y ~ f + s(x, bs="cr", k=6)', "~ 1"],
        "gaulss",
        "ML",
    )
    bad_newdata = data.head(5).copy()
    bad_newdata.loc[bad_newdata.index[0], "x"] = np.nan

    with pytest.raises(ValueError, match="X contains NaN or Inf"):
        gam.predict(bad_newdata, type="lpmatrix")

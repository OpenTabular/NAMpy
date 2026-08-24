from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.inference.summary import summary_gam
from tests.families.test_general_family_mgcv_parity import (
    _gaulss_by_data,
    _general_newdata,
)
from tests.mgcv_parity_utils import _fit_nampy_model, _run_mgcv_predict_on_newdata

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]

_FORMULA = [
    'y ~ x + s(z, bs="cr", k=6, sp=0.8)',
    '~ z + s(x, bs="cr", k=6, sp=0.7)',
]


def test_general_family_inference_uses_predictor_aware_term_labels():
    """Formula-list parametric and smooth rows use mgcv's later-LP suffix."""
    data = _gaulss_by_data(seed=270, n=120)
    gam = _fit_nampy_model(data, _FORMULA, "gaulss", "fixed")

    summary = summary_gam(gam)
    assert list(summary.pterms_table["label"]) == ["x", "z.1"]
    assert list(summary.s_table["label"]) == ["s(z)", "s.1(x)"]
    anova = gam.anova()
    assert list(anova.parametric_table["label"]) == ["x", "z.1"]
    assert list(anova.smooth_table["label"]) == ["s(z)", "s.1(x)"]


def test_general_family_terms_filter_values_labels_and_se_match_mgcv():
    data = _gaulss_by_data(seed=271, n=120)
    newdata = _general_newdata(data, n=19)
    selected_terms = ["z.1", "s.1(x)"]
    gam = _fit_nampy_model(data, _FORMULA, "gaulss", "fixed")
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _FORMULA,
        family="gaulss",
        method="fixed",
        type="terms",
        return_se=True,
        terms=selected_terms,
    )

    actual, actual_se = gam.predict(
        newdata,
        type="terms",
        return_se=True,
        terms=selected_terms,
    )

    assert expected["term_names"] == selected_terms
    assert np.asarray(actual).shape == (len(newdata), len(selected_terms))
    np.testing.assert_allclose(actual, expected["pred"], atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(actual_se, expected["se"], atol=5e-7, rtol=5e-7)


@pytest.mark.parametrize(
    ("pred_type", "return_se"),
    [("link", True), ("response", True), ("lpmatrix", False)],
)
def test_general_family_terms_filter_prediction_surfaces_match_mgcv(
    pred_type,
    return_se,
):
    data = _gaulss_by_data(seed=272, n=120)
    newdata = _general_newdata(data, n=19)
    selected_terms = ["s.1(x)"]
    gam = _fit_nampy_model(data, _FORMULA, "gaulss", "fixed")
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _FORMULA,
        family="gaulss",
        method="fixed",
        type=pred_type,
        return_se=return_se,
        terms=selected_terms,
    )

    actual = gam.predict(
        newdata,
        type=pred_type,
        return_se=return_se,
        terms=selected_terms,
    )

    if return_se:
        actual_fit, actual_se = actual
        np.testing.assert_allclose(actual_fit, expected["pred"], atol=5e-7, rtol=5e-7)
        np.testing.assert_allclose(actual_se, expected["se"], atol=5e-7, rtol=5e-7)
    else:
        np.testing.assert_allclose(actual, expected["pred"], atol=5e-7, rtol=5e-7)


def test_general_family_exclude_filter_terms_values_labels_and_se_match_mgcv():
    data = _gaulss_by_data(seed=273, n=120)
    newdata = _general_newdata(data, n=19)
    excluded_terms = ["x", "s.1(x)"]
    gam = _fit_nampy_model(data, _FORMULA, "gaulss", "fixed")
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _FORMULA,
        family="gaulss",
        method="fixed",
        type="terms",
        return_se=True,
        exclude=excluded_terms,
    )

    actual, actual_se = gam.predict(
        newdata,
        type="terms",
        return_se=True,
        exclude=excluded_terms,
    )

    assert expected["term_names"] == ["z.1", "s(z)"]
    np.testing.assert_allclose(actual, expected["pred"], atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(actual_se, expected["se"], atol=5e-7, rtol=5e-7)


@pytest.mark.parametrize(
    ("filter_name", "filter_values", "warning_text"),
    [
        ("terms", ["x", "missing"], "non-existent terms requested - ignoring"),
        (
            "exclude",
            ["x", "missing"],
            "non-existent exclude terms requested - ignoring",
        ),
    ],
)
def test_general_family_unknown_filter_matches_mgcv_zeroing_and_output_shape(
    filter_name,
    filter_values,
    warning_text,
):
    data = _gaulss_by_data(seed=274, n=120)
    newdata = _general_newdata(data, n=19)
    gam = _fit_nampy_model(data, _FORMULA, "gaulss", "fixed")
    filter_kwargs = {filter_name: filter_values}
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        _FORMULA,
        family="gaulss",
        method="fixed",
        type="terms",
        return_se=True,
        **filter_kwargs,
    )

    with pytest.warns(UserWarning, match=warning_text):
        actual, actual_se = gam.predict(
            newdata,
            type="terms",
            return_se=True,
            **filter_kwargs,
        )

    assert expected["term_names"] == ["x", "z.1", "s(z)", "s.1(x)"]
    assert np.asarray(actual).shape == (len(newdata), 4)
    np.testing.assert_allclose(actual, expected["pred"], atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(actual_se, expected["se"], atol=5e-7, rtol=5e-7)

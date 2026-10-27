"""Prediction batching must preserve array-like offsets and iterable filters."""

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM


@pytest.fixture
def fitted_gam():
    x = np.linspace(0.0, 1.0, 60)
    data = pd.DataFrame(
        {"x": x, "z": np.cos(8 * x), "y": np.sin(5 * x) + 0.2 * x}
    )
    model = GAM(
        formula="y ~ z + s(x, bs='cr', k=6)",
        smoothing_params=[0.8],
        optimize_smoothing=False,
    ).fit(data=data)
    return model, data.iloc[:9].copy()


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("missing", [False, True])
def test_prediction_accepts_offset_vectors_as_python_sequences(
    fitted_gam, container, missing
):
    model, data = fitted_gam
    offset = np.linspace(-0.4, 0.5, len(data))
    if missing:
        offset[3] = np.nan
    actual = model.predict(
        data, type="link", offset=container(offset), return_se=True, block_size=2
    )
    baseline, baseline_se = model.predict(data, type="link", return_se=True)
    expected = baseline + offset
    expected_se = baseline_se.copy()
    expected_se[np.isnan(offset)] = np.nan
    np.testing.assert_allclose(actual[0], expected, equal_nan=True)
    np.testing.assert_allclose(actual[1], expected_se, equal_nan=True)


@pytest.mark.parametrize("filter_name", ["terms", "exclude"])
@pytest.mark.parametrize("prediction_type", ["link", "terms", "lpmatrix"])
def test_prediction_materializes_term_filter_iterators_once(
    fitted_gam, filter_name, prediction_type
):
    model, data = fitted_gam
    options = {"type": prediction_type, "block_size": 2, "return_se": True}
    expected = model.predict(data, **options, **{filter_name: ["s(x)"]})
    actual = model.predict(data, **options, **{filter_name: iter(["s(x)"])})
    if isinstance(expected, tuple):
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, reference)
    else:
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("filter_name", ["terms", "exclude"])
def test_guaranteed_prediction_preserves_filter_iterators(fitted_gam, filter_name):
    model, data = fitted_gam
    options = {"block_size": 2, "newdata_guaranteed": True, "return_se": True}
    expected = model.predict(data, **options, **{filter_name: ["s(x)"]})
    actual = model.predict(data, **options, **{filter_name: iter(["s(x)"])})
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference)

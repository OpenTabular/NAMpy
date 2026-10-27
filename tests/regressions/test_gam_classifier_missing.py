"""A missing class probability must not silently become the first class."""

import numpy as np
import pandas as pd
import pytest

from nampy.models import GAMClassifier


@pytest.mark.parametrize("classes", [[0, 1], ["no", "yes"], [False, True]])
@pytest.mark.parametrize("missing_in", ["feature", "offset"])
def test_classifier_preserves_missing_predictions(classes, missing_in):
    rng = np.random.default_rng(130)
    x = rng.uniform(-1, 1, 80)
    y = np.asarray(classes)[rng.binomial(1, 1 / (1 + np.exp(-2 * x)))]
    data = pd.DataFrame({"x": x})
    model = GAMClassifier(
        formula="y ~ s(x, bs='cr', k=6)",
        optimize_smoothing=False,
        smoothing_params=[0.8],
    ).fit(data, y)
    newdata = data.iloc[:7].copy()
    baseline = model.predict(newdata)
    offset = np.zeros(len(newdata))
    if missing_in == "feature":
        newdata.loc[3, "x"] = np.nan
    else:
        offset[3] = np.nan

    probabilities = model.predict_proba(newdata, offset=offset)
    assert np.isnan(probabilities[3]).all()
    prediction = model.predict(newdata, offset=offset, block_size=2)
    assert pd.isna(prediction[3])
    retained = [0, 1, 2, 4, 5, 6]
    np.testing.assert_array_equal(prediction[retained], baseline[retained])
    omitted = model.predict(newdata, offset=offset, na_action="omit")
    np.testing.assert_array_equal(omitted, baseline[retained])
    assert omitted.dtype == baseline.dtype

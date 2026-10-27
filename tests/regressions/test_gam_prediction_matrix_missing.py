"""Missing matrix-functional locations/weights follow predict.gam row policies."""

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM


@pytest.fixture(params=["ps", "cr", "tp"])
def matrix_gam(request):
    rng = np.random.default_rng(51)
    locations = rng.uniform(0, 1, (60, 3))
    weights = rng.uniform(0.2, 1.0, (60, 3))
    data = pd.DataFrame(
        {
            "X": list(locations),
            "L": list(weights),
            "y": (weights * np.sin(5 * locations)).sum(axis=1),
        }
    )
    model = GAM(
        formula=f"y ~ s(X, by=L, bs='{request.param}', k=6)",
        smoothing_params=[1.0],
        optimize_smoothing=False,
    ).fit(data=data)
    return model, data.iloc[:5].copy()


@pytest.mark.parametrize("column", ["X", "L"])
@pytest.mark.parametrize("action", ["pass", "omit", "exclude"])
@pytest.mark.parametrize("prediction_type", ["link", "terms", "lpmatrix"])
def test_matrix_missing_values_restore_or_omit_whole_rows(
    matrix_gam, column, action, prediction_type
):
    model, data = matrix_gam
    options = {"return_se": True, "block_size": 2, "type": prediction_type}
    expected = model.predict(data.iloc[[0, 1, 3, 4]], **options)
    values = data.at[2, column].copy()
    values[1] = np.nan
    data.at[2, column] = values
    actual = model.predict(data, na_action=action, **options)
    references = expected if isinstance(expected, tuple) else (expected,)
    results = actual if isinstance(actual, tuple) else (actual,)
    for result, reference in zip(results, references, strict=True):
        if action == "pass":
            assert result.shape == (5, *reference.shape[1:])
            assert np.isnan(result[2]).all()
            result = result[[0, 1, 3, 4]]
        np.testing.assert_allclose(result, reference)


@pytest.mark.parametrize("column", ["X", "L"])
@pytest.mark.parametrize("options", [{"na_action": "fail"}, {"newdata_guaranteed": True}])
def test_matrix_missing_values_reject_complete_data_promises(matrix_gam, column, options):
    model, data = matrix_gam
    values = data.at[2, column].copy()
    values[1] = np.nan
    data.at[2, column] = values
    with pytest.raises(ValueError, match="missing|complete"):
        model.predict(data, **options)

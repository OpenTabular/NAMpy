"""Censor metadata must survive response lifecycle operations without guessing rows."""

import pickle

import numpy as np
import pytest

from nampy.gam.response import (
    CensoredResponse,
    normalize_censored_response,
    response_values,
)


def test_censor_pairing_survives_permutations_duplicates_and_pickle():
    raw = np.array([[2, -np.inf], [2, 3], [2, np.inf], [3, 3]], dtype=float)
    response = normalize_censored_response(raw)
    raw[:] = 99
    subset = response.subset([2, 0, 2, 1])
    restored = pickle.loads(pickle.dumps(subset))
    np.testing.assert_array_equal(restored.values, [2, 2, 2, 2])
    np.testing.assert_array_equal(restored.bounds, [np.inf, -np.inf, np.inf, 3])
    assert not restored.values.flags.writeable
    assert not restored.bounds.flags.writeable
    assert response_values(restored) is restored.values
    with pytest.raises((TypeError, ValueError)):
        np.asarray(restored, dtype=float)


@pytest.mark.parametrize("bad", [[[1, np.nan]], [[np.inf, 2]], [[1, 2, 3]], 1])
def test_censored_response_rejects_invalid_shapes_and_nonfinite_primary(bad):
    with pytest.raises(ValueError):
        normalize_censored_response(bad)


def test_vector_normalization_resets_censoring_and_keeps_original_bound_order():
    exact = normalize_censored_response([2, -1])
    np.testing.assert_array_equal(exact.values, exact.bounds)
    interval = CensoredResponse([3], [1])
    assert interval.values[0] == 3 and interval.bounds[0] == 1


@pytest.mark.parametrize(
    "column,value", [("y", np.inf), ("bound", np.nan), ("x", np.inf)]
)
def test_infinite_censor_bounds_do_not_relax_primary_or_predictor_validation(
    column, value
):
    import pandas as pd

    from nampy.gam import GAM
    from nampy.gam.families.cnorm import CensoredNormalFamily

    data = pd.DataFrame({"x": np.linspace(0, 1, 12), "y": np.linspace(1, 2, 12)})
    data["bound"] = np.inf
    data.loc[0, column] = value
    model = GAM(
        formula="cbind(y, bound) ~ s(x, bs='cr', k=5)",
        family=CensoredNormalFamily(theta=1),
        smoothing_params=[0.2],
        optimize_smoothing=False,
    )
    with pytest.raises(ValueError):
        model.fit(data)

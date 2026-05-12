import numpy as np
import pytest

from nampy.models import NAMRegressor
from nampy.models.sklearn_regressor import _coerce_single_output_regression_target


def test_single_output_regression_target_accepts_1d_and_single_column():
    y_1d = np.array([1.0, 2.0, 3.0])
    y_column = np.array([[1.0], [2.0], [3.0]])

    assert _coerce_single_output_regression_target(y_1d, "y").shape == (3,)
    assert np.array_equal(
        _coerce_single_output_regression_target(y_column, "y"),
        y_1d,
    )


def test_regressor_fit_rejects_multi_output_targets_before_training():
    X = np.ones((4, 2))
    y = np.ones((4, 2))
    model = NAMRegressor()

    with pytest.raises(ValueError, match="Multi-output regression is not supported"):
        model.fit(X, y)


def test_regressor_fit_rejects_multi_output_validation_targets_before_training():
    X = np.ones((4, 2))
    y = np.ones(4)
    X_val = np.ones((2, 2))
    y_val = np.ones((2, 2))
    model = NAMRegressor()

    with pytest.raises(ValueError, match="y_val"):
        model.fit(X, y, X_val=X_val, y_val=y_val)

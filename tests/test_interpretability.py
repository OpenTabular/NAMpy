import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nampy.models.spline_nam import SplineNAMRegressor
from nampy.utils.interpretability import (
    feature_importance,
    plot_interactions,
    plot_terms,
    predict_terms,
)


def _prediction_dict():
    return {
        "prediction": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "terms": {
            "x1": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
            "x2": torch.tensor([[2.0], [2.0], [3.0], [5.0]]),
            "x1:x2": torch.tensor([[1.0], [0.5], [0.0], [-0.5]]),
        },
        "intercept": torch.tensor([0.1]),
        "regularization": {"smooth": torch.tensor(1.0)},
        "extras": {},
    }


class DummyEstimator:
    def __init__(self):
        self.model = object()
        self.data_module = object()
        self.feature_names_in_ = np.asarray(["x1", "x2"], dtype=object)

    def _predict(self, X):
        return _prediction_dict()


def test_predict_terms_filters_auxiliary_outputs_and_can_return_frame():
    estimator = DummyEstimator()
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 2.0, 3.0, 4.0]})

    terms = predict_terms(estimator, X)
    frame = predict_terms(
        estimator,
        X,
        include_prediction=True,
        include_intercept=True,
        as_frame=True,
    )

    assert set(terms) == {"x1", "x2", "x1:x2"}
    assert frame.columns.tolist() == ["prediction", "x1", "x2", "x1:x2", "intercept"]
    assert frame.shape == (4, 5)
    assert np.allclose(frame["intercept"], 0.1)


def test_feature_importance_returns_sorted_normalized_terms():
    estimator = DummyEstimator()
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 2.0, 3.0, 4.0]})

    importance = feature_importance(estimator, X, method="variance")

    assert np.isclose(importance["importance"].sum(), 1.0)
    assert importance["importance"].is_monotonic_decreasing
    assert set(importance["term"]) == {"x1", "x2", "x1:x2"}


def test_plot_terms_and_interactions_return_figures():
    estimator = DummyEstimator()
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 2.0, 3.0, 4.0]})

    fig_terms, _ = plot_terms(estimator, X, terms=["x1", "x2"])
    fig_interactions, _ = plot_interactions(estimator, X, num_bins=3)

    assert fig_terms is not None
    assert fig_interactions is not None
    plt.close(fig_terms)
    plt.close(fig_interactions)


def test_sklearn_wrappers_expose_generic_interpretability_methods():
    estimator = SplineNAMRegressor()
    estimator.model = object()
    estimator.data_module = object()
    estimator.feature_names_in_ = np.asarray(["x1", "x2"], dtype=object)
    estimator._predict = lambda X: _prediction_dict()
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 2.0, 3.0, 4.0]})

    terms = estimator.predict_terms(X)
    importance = estimator.feature_importance(X)

    assert set(terms) == {"x1", "x2", "x1:x2"}
    assert importance.iloc[0]["importance"] > 0.0

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

import nampy
from nampy import models
from nampy.contracts import AdditivePrediction
from nampy.models.ensemble import NeuralEnsemble


class _SeededAdditiveRegressor(BaseEstimator):
    _estimator_type = "regressor"

    def fit(self, X, y, random_state=0, **kwargs):
        del X, y, kwargs
        self.intercept_ = float(random_state)
        return self

    def predict(self, X):
        values = np.asarray(X)[:, 0]
        return values + self.intercept_

    def predict_components(
        self,
        X,
        *,
        center=False,
        reference_X=None,
        reference_weight=None,
    ):
        del center, reference_X, reference_weight
        values = np.asarray(X)[:, 0]
        prediction = self.predict(X)
        return AdditivePrediction(
            response=prediction,
            link=prediction,
            terms={"x": values},
            intercept=self.intercept_,
            backend="neural",
        )


class _RecordingRegressor(BaseEstimator):
    _estimator_type = "regressor"

    def fit(self, X, y, random_state=0, sample_weight=None, **kwargs):
        del kwargs
        self.X_ = np.asarray(X).copy()
        self.y_ = np.asarray(y).copy()
        self.sample_weight_ = np.asarray(sample_weight).copy()
        self.random_state_ = random_state
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_independent_ensemble_averages_components_and_reports_uncertainty():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    ensemble = NeuralEnsemble(
        _SeededAdditiveRegressor(),
        n_estimators=3,
        random_state=10,
        n_jobs=1,
    ).fit(X, y)

    np.testing.assert_allclose(ensemble.predict(X), X[:, 0] + 11.0)
    uncertainty = ensemble.predict_component_uncertainty(X)
    uncertainty.mean.validate_additive_reconstruction()
    assert uncertainty.n_estimators == 3
    assert uncertainty.intercept_std > 0
    np.testing.assert_allclose(uncertainty.term_std["x"], 0.0)
    assert models.NeuralEnsemble is NeuralEnsemble
    assert nampy.NeuralEnsemble is NeuralEnsemble


def test_generic_ensemble_bootstraps_rows_and_aligned_fit_channels():
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)
    weights = np.arange(10, 16)
    ensemble = NeuralEnsemble(
        _RecordingRegressor(),
        n_estimators=2,
        random_state=7,
        n_jobs=1,
        bootstrap=True,
    ).fit(X, y, sample_weight=weights)

    for index, member in enumerate(ensemble.estimators_):
        expected_indices = np.random.default_rng(7 + index).integers(0, 6, size=6)
        np.testing.assert_array_equal(member.X_, X[expected_indices])
        np.testing.assert_array_equal(member.y_, y[expected_indices])
        np.testing.assert_array_equal(member.sample_weight_, weights[expected_indices])

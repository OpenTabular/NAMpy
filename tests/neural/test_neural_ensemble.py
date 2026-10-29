from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import nampy
from nampy import models
from nampy.contracts import AdditivePrediction, AdditiveTermMetadata
from nampy.models.ensemble import NeuralEnsemble


class _SeededAdditiveRegressor(BaseEstimator):
    _estimator_type = "regressor"

    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        del X, y, kwargs
        self.intercept_ = float(self.random_state)
        return self

    def predict(self, X, *, offset=None, batch_size=None):
        values = np.asarray(X)[:, 0]
        return values + self.intercept_ + (0. if offset is None else np.asarray(offset).reshape(-1))

    def predict_components(
        self,
        X,
        *,
        center=False,
        offset=None,
        batch_size=None,
        reference_X=None,
        reference_weight=None,
    ):
        del center, reference_X, reference_weight
        values = np.asarray(X)[:, 0]
        prediction = self.predict(X, offset=offset, batch_size=batch_size)
        return AdditivePrediction(
            link=prediction,
            terms={"x": values},
            intercept=self.intercept_,
            backend="neural",
            offset=offset,
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

    def predict(self, X, *, offset=None, batch_size=None):
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
    np.testing.assert_allclose(ensemble.predict(X, offset=np.ones(len(X)), batch_size=2), X[:, 0] + 12.0)
    shifted = ensemble.predict_components(X, offset=np.ones(len(X)), batch_size=2)
    shifted.validate_additive_reconstruction()
    uncertainty = ensemble.predict_component_uncertainty(X)
    uncertainty.mean.validate_additive_reconstruction()
    assert uncertainty.n_estimators == 3
    assert uncertainty.intercept_std > 0
    np.testing.assert_allclose(uncertainty.term_std["x"], 0.0)
    assert models.NeuralEnsemble is NeuralEnsemble
    assert nampy.NeuralEnsemble is NeuralEnsemble


def test_ensemble_aligns_sparse_member_terms_over_their_union():
    components = [
        AdditivePrediction(
            link=np.array([5.0, 7.0]),
            terms={"x": np.array([1.0, 2.0]), "x:y": np.array([3.0, 4.0])},
            intercept=1.0,
            backend="neural",
        ),
        AdditivePrediction(
            link=np.array([14.0, 16.0]),
            terms={"y": np.array([5.0, 6.0]), "x": np.array([7.0, 8.0])},
            intercept=2.0,
            backend="neural",
        ),
    ]

    uncertainty = NeuralEnsemble._aggregate_components(components)

    assert tuple(uncertainty.mean.terms) == ("x", "x:y", "y")
    np.testing.assert_allclose(uncertainty.mean.terms["x:y"][:, 0], [1.5, 2.0])
    np.testing.assert_allclose(uncertainty.mean.terms["y"][:, 0], [2.5, 3.0])
    np.testing.assert_allclose(uncertainty.term_std["x:y"][:, 0], [1.5, 2.0])
    np.testing.assert_allclose(uncertainty.term_std["y"][:, 0], [2.5, 3.0])
    uncertainty.mean.validate_additive_reconstruction()


def test_ensemble_combines_constant_and_rowwise_predictor_intercepts():
    constant = np.array([1.0, 2.0])
    rowwise = np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    parts = [
        AdditivePrediction(
            link=np.broadcast_to(constant, rowwise.shape), terms={},
            intercept=constant, backend="neural",
        ),
        AdditivePrediction(
            link=rowwise, terms={}, intercept=rowwise, backend="neural",
        ),
    ]
    result = NeuralEnsemble._aggregate_components(parts)
    np.testing.assert_allclose(result.mean.intercept, (constant + rowwise) / 2)
    result.mean.validate_additive_reconstruction()


def test_sparse_ensemble_rejects_conflicting_term_meanings():
    parts = [AdditivePrediction(
        link=np.zeros(3), terms={}, intercept=0, backend="neural",
    )]
    for feature in ("x", "z"):
        parts.append(AdditivePrediction(
            link=np.ones(3), terms={"term": np.ones(3)}, intercept=0,
            backend="neural", term_metadata={"term": AdditiveTermMetadata((feature,))},
        ))
    with pytest.raises(ValueError, match="inconsistent metadata"):
        NeuralEnsemble._aggregate_components(parts)


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

"""Independent ensembles for fitted neural additive estimators."""

from __future__ import annotations

import copy
import random
from numbers import Integral

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import _safe_indexing

from ..contracts import AdditivePrediction, EnsembleAdditivePrediction


class NeuralEnsemble(BaseEstimator):
    """Fit independent replicas of a neural classifier or regressor.

    Each member owns its preprocessing and fitted model. Predictions are
    averaged on the response scale, while additive terms and links are
    averaged on the link scale.
    """

    def __init__(
        self,
        estimator,
        n_estimators: int = 5,
        random_state: int = 101,
        n_jobs: int | None = 1,
        bootstrap: bool = False,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self._estimator_type = getattr(estimator, "_estimator_type", None)

    def __sklearn_tags__(self):
        return copy.deepcopy(self.estimator.__sklearn_tags__())

    def fit(self, X, y, **fit_params):
        if not isinstance(self.n_estimators, Integral) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if self._estimator_type not in {"classifier", "regressor"}:
            raise TypeError(
                "NeuralEnsemble requires a classifier or regressor estimator; "
                "distributional/LSS ensembles need family-specific aggregation."
            )
        base_seed = int(fit_params.pop("random_state", self.random_state))
        base_estimator = self.estimator

        def fit_member(index):
            member_seed = base_seed + index
            random.seed(member_seed)
            np.random.seed(member_seed)
            try:
                import torch

                torch.manual_seed(member_seed)
            except ImportError:  # pragma: no cover - neural estimators require torch
                pass
            member = clone(base_estimator)
            member_fit_params = dict(fit_params)
            member_fit_params["random_state"] = member_seed
            member_X = X
            member_y = y
            if self.bootstrap:
                generator = np.random.default_rng(member_seed)
                indices = generator.integers(0, len(y), size=len(y))
                member_X = _safe_indexing(X, indices)
                member_y = _safe_indexing(y, indices)
                for name in ("offset", "sample_weight"):
                    if member_fit_params.get(name) is not None:
                        member_fit_params[name] = _safe_indexing(
                            member_fit_params[name], indices
                        )
            return member.fit(member_X, member_y, **member_fit_params)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_member)(index) for index in range(int(self.n_estimators))
        )
        if self._estimator_type == "classifier":
            classes = [tuple(member.classes_) for member in self.estimators_]
            if any(current != classes[0] for current in classes[1:]):
                raise RuntimeError("Ensemble members learned inconsistent classes.")
            self.classes_ = np.asarray(classes[0])
        return self

    def _members(self):
        members = getattr(self, "estimators_", None)
        if not members:
            raise NotFittedError("Call fit before using NeuralEnsemble.")
        return members

    def predict_proba(self, X):
        if self._estimator_type != "classifier":
            raise AttributeError("predict_proba is only available for classifiers.")
        probabilities = np.stack(
            [member.predict_proba(X) for member in self._members()], axis=0
        )
        return probabilities.mean(axis=0)

    def predict(self, X):
        if self._estimator_type == "classifier":
            probabilities = self.predict_proba(X)
            return self.classes_[np.argmax(probabilities, axis=1)]
        predictions = np.stack(
            [member.predict(X) for member in self._members()], axis=0
        )
        return predictions.mean(axis=0)

    def score(self, X, y, sample_weight=None):
        if self._estimator_type == "classifier":
            return float(
                accuracy_score(y, self.predict(X), sample_weight=sample_weight)
            )
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

    @staticmethod
    def _aggregate_components(components):
        keys = tuple(components[0].terms)
        if any(tuple(component.terms) != keys for component in components[1:]):
            raise RuntimeError("Ensemble members returned inconsistent additive terms.")
        responses = np.stack([component.response for component in components], axis=0)
        links = np.stack([component.link for component in components], axis=0)
        intercepts = np.stack(
            [np.asarray(component.intercept) for component in components], axis=0
        )
        term_arrays = {
            name: np.stack([component.terms[name] for component in components], axis=0)
            for name in keys
        }
        first = components[0]
        mean = AdditivePrediction(
            response=responses.mean(axis=0),
            link=links.mean(axis=0),
            terms={name: values.mean(axis=0) for name, values in term_arrays.items()},
            intercept=intercepts.mean(axis=0),
            backend=first.backend,
            offset=first.offset,
        )
        intercept_std = intercepts.std(axis=0)
        if np.asarray(intercept_std).size == 1:
            intercept_std = float(np.asarray(intercept_std).reshape(-1)[0])
        return EnsembleAdditivePrediction(
            mean=mean,
            response_std=responses.std(axis=0),
            link_std=links.std(axis=0),
            term_std={name: values.std(axis=0) for name, values in term_arrays.items()},
            intercept_std=intercept_std,
            n_estimators=len(components),
        )

    def predict_component_uncertainty(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ) -> EnsembleAdditivePrediction:
        """Return mean components and between-member standard deviations."""
        components = [
            member.predict_components(
                X,
                center=center,
                reference_X=reference_X,
                reference_weight=reference_weight,
            )
            for member in self._members()
        ]
        return self._aggregate_components(components)

    def predict_components(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ) -> AdditivePrediction:
        """Return the mean link-scale additive decomposition."""
        return self.predict_component_uncertainty(
            X,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
        ).mean

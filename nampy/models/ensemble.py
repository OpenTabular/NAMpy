"""Independent ensembles for fitted neural additive estimators."""

from __future__ import annotations

import copy
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

    def fit(
        self,
        X,
        y,
        *,
        X_val=None,
        y_val=None,
        offset=None,
        offset_val=None,
        sample_weight=None,
        sample_weight_val=None,
    ):
        fit_params = {
            name: value
            for name, value in locals().copy().items()
            if name not in {"self", "X", "y"} and value is not None
        }
        if not isinstance(self.n_estimators, Integral) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if self._estimator_type not in {"classifier", "regressor"}:
            raise TypeError(
                "NeuralEnsemble requires a classifier or regressor estimator; "
                "distributional/LSS ensembles need family-specific aggregation."
            )
        base_seed = int(self.random_state)
        base_estimator = self.estimator

        def fit_member(index):
            member_seed = base_seed + index
            member = clone(base_estimator)
            member_fit_params = dict(fit_params)
            if hasattr(member, "_resolved_training"):
                from dataclasses import replace

                member.set_params(
                    training=replace(
                        member._resolved_training(y), random_state=member_seed
                    )
                )
            elif "random_state" in member.get_params(deep=False):
                member.set_params(random_state=member_seed)
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

        members = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_member)(index) for index in range(int(self.n_estimators))
        )
        if self._estimator_type == "classifier":
            classes = [tuple(member.classes_) for member in members]
            if any(current != classes[0] for current in classes[1:]):
                raise RuntimeError("Ensemble members learned inconsistent classes.")
            self.classes_ = np.asarray(classes[0])
        self.estimators_ = members
        return self

    def set_params(self, **parameters):
        if not parameters:
            return self
        replacement = clone(self)
        BaseEstimator.set_params(replacement, **parameters)
        replacement._estimator_type = getattr(
            replacement.estimator, "_estimator_type", None
        )
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def _members(self):
        members = getattr(self, "estimators_", None)
        if not members:
            raise NotFittedError("Call fit before using NeuralEnsemble.")
        return members

    def capabilities(self):
        """Expose component prediction support without inventing ensemble Fisher."""
        from types import MappingProxyType

        from ..contracts import CapabilityStatus

        source = self.estimators_[0] if getattr(self, "estimators_", ()) else self.estimator
        result = dict(source.capabilities())
        for name in result:
            if name.startswith("fisher.") or name == "uncertainty.coefficients":
                result[name] = CapabilityStatus("unsupported", "Ensemble parameter inference has no declared joint model.")
        return MappingProxyType(result)

    def predict_link(self, X, *, offset=None, batch_size=None):
        """Average member additive predictors in their common output coordinates."""
        return np.mean([
            member.predict_link(X, offset=offset, batch_size=batch_size)
            for member in self._members()
        ], axis=0)

    def predict_proba(self, X, *, offset=None, batch_size=None):
        if self._estimator_type != "classifier":
            raise AttributeError("predict_proba is only available for classifiers.")
        probabilities = np.stack(
            [member.predict_proba(X, offset=offset, batch_size=batch_size) for member in self._members()], axis=0
        )
        return probabilities.mean(axis=0)

    def predict(self, X, *, offset=None, batch_size=None):
        if self._estimator_type == "classifier":
            probabilities = self.predict_proba(X, offset=offset, batch_size=batch_size)
            return self.classes_[np.argmax(probabilities, axis=1)]
        predictions = np.stack(
            [member.predict(X, offset=offset, batch_size=batch_size) for member in self._members()], axis=0
        )
        return predictions.mean(axis=0)

    def score(self, X, y, sample_weight=None, *, offset=None, batch_size=None):
        if self._estimator_type == "classifier":
            return float(
                accuracy_score(y, self.predict(X, offset=offset, batch_size=batch_size), sample_weight=sample_weight)
            )
        return float(r2_score(y, self.predict(X, offset=offset, batch_size=batch_size), sample_weight=sample_weight))

    @staticmethod
    def _aggregate_components(components):
        # Sparse/discovery-based architectures can learn different term supports
        # across independent replicas.  A term omitted by a member contributes
        # exactly zero for that member, so aggregate over the ordered union.  This
        # also lets term_std reflect between-member selection instability.
        if not components:
            raise ValueError("At least one component record is required.")
        first = components[0]
        for part in components[1:]:
            if (
                part.backend != first.backend
                or part.predictor_names != first.predictor_names
            ):
                raise ValueError(
                    "Ensemble components must use the same predictor coordinates."
                )
            if (part.offset is None) != (first.offset is None) or (
                part.offset is not None
                and not np.array_equal(part.offset, first.offset, equal_nan=True)
            ):
                raise ValueError(
                    "Ensemble components must use identical prediction offsets."
                )
        metadata = {}
        for part in components:
            for name, value in part.term_metadata.items():
                if name in metadata and metadata[name] != value:
                    raise ValueError(f"Ensemble term {name!r} has inconsistent metadata.")
                metadata[name] = value
        keys = tuple(
            dict.fromkeys(name for component in components for name in component.terms)
        )
        links = np.stack([component.link for component in components], axis=0)
        intercepts = np.stack(
            np.broadcast_arrays(
                *[np.asarray(component.intercept) for component in components]
            ),
            axis=0,
        )
        term_arrays = {}
        for name in keys:
            template = next(
                np.asarray(component.terms[name])
                for component in components
                if name in component.terms
            )
            term_arrays[name] = np.stack(
                [
                    np.asarray(component.terms[name])
                    if name in component.terms
                    else np.zeros_like(template)
                    for component in components
                ],
                axis=0,
            )
        first = components[0]
        mean = AdditivePrediction(
            link=links.mean(axis=0),
            terms={name: values.mean(axis=0) for name, values in term_arrays.items()},
            intercept=intercepts.mean(axis=0),
            backend=first.backend,
            predictor_names=first.predictor_names,
            term_metadata=metadata,
            offset=first.offset,
        )
        intercept_std = intercepts.std(axis=0)
        if np.asarray(intercept_std).size == 1:
            intercept_std = float(np.asarray(intercept_std).reshape(-1)[0])
        return EnsembleAdditivePrediction(
            mean=mean,
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
        offset=None,
        batch_size=None,
        reference_X=None,
        reference_weight=None,
    ) -> EnsembleAdditivePrediction:
        """Return mean components and between-member standard deviations."""
        components = [
            member.predict_components(
                X,
                center=center,
                offset=offset,
                batch_size=batch_size,
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
        offset=None,
        batch_size=None,
        reference_X=None,
        reference_weight=None,
    ) -> AdditivePrediction:
        """Return the mean link-scale additive decomposition."""
        return self.predict_component_uncertainty(
            X,
            center=center,
            offset=offset,
            batch_size=batch_size,
            reference_X=reference_X,
            reference_weight=reference_weight,
        ).mean

"""Shared lifecycle for scalar, fixed-penalty statistical loss estimators."""

import warnings
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from nampy.contracts import AdditivePrediction, AdditiveTermMetadata
from nampy.gam.design import compile_design, observation_weights, response_vector

from ..persistence import load_estimator, save_estimator
from ._data import feature_frame


class _LossGAMBase(RegressorMixin, BaseEstimator, ABC):
    """Shared compiled-design lifecycle for fixed-penalty loss estimators."""

    def __init__(
        self,
        formula=None,
        *,
        level=0.5,
        smoothing_params=None,
        k=10,
        basis="tp",
        fit_intercept=True,
        max_iter=200,
        tol=1e-9,
    ):
        self.formula = formula
        self.level = level
        self.smoothing_params = smoothing_params
        self.k = k
        self.basis = basis
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    @abstractmethod
    def _solve(self, design, response, penalty, weights):
        """Return LossFit for the supplied identified design and fixed penalty."""
        raise NotImplementedError

    @abstractmethod
    def _loss(self, residual):
        """Return one unweighted scoring loss per observation, shape (n,)."""
        raise NotImplementedError

    def fit(self, X=None, y=None, *, data=None, sample_weight=None, offset=None):
        """Fit; X is (n,d) or a DataFrame, y is (n,), returns self.

        A response-containing formula may omit y and use X or ``data``.
        ``sample_weight`` is finite nonnegative (n,), default ones; ``offset``
        is a known additive (n,) vector, default zero. Formula offsets are
        supported. Rows with missing or infinite values are rejected.
        Nonconvergence issues a ConvergenceWarning and is recorded explicitly.
        """
        design = compile_design(
            X,
            y,
            data=data,
            formula=self.formula,
            k=self.k,
            basis=self.basis,
            fit_intercept=self.fit_intercept,
            smoothing_params=self.smoothing_params,
            sample_weight=sample_weight,
            offset=offset,
        )
        if design.response is None:
            raise ValueError("y or a formula response is required for fitting.")
        solution = self._solve(
            design.matrix,
            design.response - design.training_offset,
            design.penalty,
            design.sample_weight,
        )
        if not solution.converged:
            warnings.warn(
                "Asymmetric GAM did not converge; inspect fit_result_.",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.design_ = design
        self.coef_ = solution.coefficients.copy()
        self.penalty_ = design.penalty
        self.smoothing_params_ = design.smoothing_params.copy()
        self.fit_result_ = solution
        self.n_features_in_ = len(design.feature_names)
        return self

    def lpmatrix(self, X):
        """Return the fitted basis evaluated at X (n,d), as float (n,p)."""
        check_is_fitted(self, "coef_")
        return self.design_.evaluate(X).matrix

    def predict(self, X, *, offset=None):
        """Return the conditional expectile/quantile as float (n,).

        X is a DataFrame or (n,d) array. ``offset`` is an optional (n,) vector;
        formula offsets are evaluated from X. Stored training-only offsets are
        not reused for new data, following the GAM prediction convention.
        """
        check_is_fitted(self, "coef_")
        evaluation = self.design_.evaluate(X, offset=offset)
        return evaluation.matrix @ self.coef_ + evaluation.offset

    def predict_components(self, X, *, offset=None):
        """Return predictor decomposition with (n,1) link and term arrays."""
        check_is_fitted(self, "coef_")
        evaluation = self.design_.evaluate(X, offset=offset)
        B = evaluation.matrix
        terms = {}
        metadata = {}
        for term in self.design_.terms:
            idx = term.coefficient_indices
            key = term.label if term.label not in terms else term.term_id
            terms[key] = (B[:, idx] @ self.coef_[idx])[:, None]
            features = term.features
            metadata[key] = AdditiveTermMetadata(
                features=features, kind="interaction" if len(features) > 1 else "main"
            )
        intercept = float(np.sum(self.coef_[self.design_.intercept_indices]))
        prediction = B @ self.coef_ + evaluation.offset
        off = evaluation.offset
        return AdditivePrediction(
            link=prediction,
            terms=terms,
            intercept=intercept,
            offset=off,
            backend="gam",
            term_metadata=metadata,
        )

    def plot_terms(self, X, *, term_features=None, rug=False, figsize=None):
        """Plot observed one-dimensional term contributions without intervals.

        ``X`` is a DataFrame (n,d); ``term_features`` is an optional mapping
        from term labels to numeric X columns, default inferred for single
        feature terms. ``rug`` is bool, default False; ``figsize`` is an
        optional (width,height) tuple in inches. Returns a list of Matplotlib
        Figure objects, one per rendered page. Multivariate terms are skipped
        unless explicitly mapped; this plots contributions at the supplied
        rows rather than varying other predictors to form partial dependence.
        """
        from nampy.plotting import prepared_from_contributions, render_term_plots

        components = self.predict_components(X)
        X = feature_frame(X, self.design_.feature_names)
        mapping = {
            name: info.features[0]
            for name, info in components.term_metadata.items()
            if len(info.features) == 1
        }
        mapping.update(term_features or {})
        prepared = prepared_from_contributions(
            X, components.terms, term_features=mapping
        )
        return render_term_plots(prepared, rug=rug, figsize=figsize)

    def score(self, X, y, sample_weight=None):
        """Negative mean model loss; y is (n,) or (n,1), weights optional (n,).

        Targets are normalized before subtraction; mismatched lengths and
        multi-output targets are rejected. Larger scores indicate better fit.
        """
        prediction = self.predict(X)
        target = response_vector(y, size=len(prediction))
        weights = observation_weights(sample_weight, len(prediction))
        return -float(np.average(self._loss(target - prediction), weights=weights))

    def save_model(self, path):
        """Save this estimator to a trusted NAMpy pickle artifact."""
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        """Load a trusted NAMpy artifact of this estimator class."""
        return load_estimator(path, cls)

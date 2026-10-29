"""Persisted marginal bases and explicit history layout for distributed lags."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from ..design import compile_design
from .lags import cross_basis, cross_basis_penalty, cumulative_lag_matrix


class DistributedLagBasis(TransformerMixin, BaseEstimator):
    """Fitted exposure/lag bases with fixed lag order 0,...,max_lag.

    Marginal formulas use ``exposure`` and ``lag`` respectively. The exposure
    intercept is removed to avoid its redundant product with a model intercept;
    the lag intercept is retained. Identification, knots and penalties belong to
    the existing compiler. Missing histories remain missing and are never padded.
    This transformer fits no regression coefficients or smoothing parameters.
    """

    def __init__(
        self,
        *,
        max_lag: int,
        exposure_formula="~exposure",
        lag_formula="~1",
        k=5,
        basis="cr",
    ):
        self.max_lag = max_lag
        self.exposure_formula = exposure_formula
        self.lag_formula = lag_formula
        self.k = k
        self.basis = basis

    def set_params(self, **params):
        result = super().set_params(**params)
        if params:
            for key in tuple(self.__dict__):
                if key.endswith("_"):
                    del self.__dict__[key]
        return result

    def _history(self, history):
        value = np.asarray(history, dtype=float)
        if (
            isinstance(self.max_lag, bool)
            or int(self.max_lag) != self.max_lag
            or self.max_lag < 0
        ):
            raise ValueError("max_lag must be a nonnegative integer.")
        if (
            value.ndim != 2
            or value.shape[1] != self.max_lag + 1
            or not len(value)
            or np.isinf(value).any()
        ):
            raise ValueError(
                "History must be nonempty (n, max_lag+1), ordered lag 0 first, with no infinity."
            )
        return value

    def fit(self, X, y=None):
        staged = type(self)(**self.get_params(deep=False))
        staged._fit(X)
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        return self

    def _fit(self, history):
        H = self._history(history)
        complete = np.isfinite(H).all(axis=1)
        if not complete.any():
            raise ValueError("At least one complete training history is required.")
        for formula in (self.exposure_formula, self.lag_formula):
            if (
                not isinstance(formula, str)
                or not formula.lstrip().startswith("~")
                or "offset(" in formula.replace(" ", "")
            ):
                raise ValueError(
                    "Marginal formulas must be one-sided and cannot contain offsets."
                )
        exposure = compile_design(
            pd.DataFrame({"exposure": H[complete].ravel()}),
            formula=self.exposure_formula,
            k=self.k,
            basis=self.basis,
            smoothing_params=1.0,
        )
        lag = compile_design(
            pd.DataFrame({"lag": np.arange(self.max_lag + 1, dtype=float)}),
            formula=self.lag_formula,
            k=self.k,
            basis=self.basis,
            smoothing_params=1.0,
        )
        indices = np.setdiff1d(
            np.arange(exposure.matrix.shape[1]), exposure.intercept_indices
        )
        if not len(indices):
            raise ValueError("The exposure basis must contain a nonconstant column.")
        self.exposure_indices_ = indices
        self.exposure_penalty_ = exposure.penalty[np.ix_(indices, indices)]
        self.lag_penalty_ = lag.penalty.copy()
        self.lag_basis_ = lag.matrix.copy()
        self.exposure_design_ = exposure.for_prediction()
        self.lag_design_ = lag.for_prediction()
        self.n_features_in_ = self.max_lag + 1
        self.n_features_out_ = len(indices) * self.lag_basis_.shape[1]

    def _require_fit(self):
        if not hasattr(self, "exposure_design_"):
            raise NotFittedError("Fit DistributedLagBasis before transformation.")

    def _exposure_basis(self, values):
        return self.exposure_design_.evaluate(
            pd.DataFrame({"exposure": values})
        ).matrix[:, self.exposure_indices_]

    def complete_rows(self, history):
        """Mask complete histories; leading/group-boundary missing lags stay missing."""
        return np.isfinite(self._history(history)).all(axis=1)

    def transform(self, X):
        self._require_fit()
        H = self._history(X)
        if not self.complete_rows(H).any():
            return np.full((len(H), self.n_features_out_), np.nan)
        return cross_basis(
            H, exposure_basis=self._exposure_basis, lag_basis=self.lag_basis_
        )

    def penalty(self, *, exposure_weight=1.0, lag_weight=1.0):
        """Fixed tensor penalty, in the exact transformed column order."""
        self._require_fit()
        return cross_basis_penalty(
            self.exposure_penalty_,
            self.lag_penalty_,
            exposure_weight=exposure_weight,
            lag_weight=lag_weight,
        )

    def cumulative_contrast(
        self, exposure, reference, *, coefficient_indices=None, n_coefficients=None
    ):
        """Constant-path link contrasts; optionally embed in full model coordinates.

        Use the returned matrix with linear_functional_interval to retain the
        full cross-lag/term covariance. This is not a causal effect or a neural
        coefficient interval; nonlinear response summaries require paired draws.
        """
        self._require_fit()
        matrix = cumulative_lag_matrix(
            exposure,
            reference,
            n_lags=self.max_lag + 1,
            exposure_basis=self._exposure_basis,
            lag_basis=self.lag_basis_,
        )
        if coefficient_indices is None and n_coefficients is None:
            return matrix
        indices = np.asarray(coefficient_indices)
        if (
            not isinstance(n_coefficients, Integral)
            or isinstance(n_coefficients, bool)
            or int(n_coefficients) != n_coefficients
            or n_coefficients < 1
            or indices.shape != (matrix.shape[1],)
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= n_coefficients)
        ):
            raise ValueError(
                "Supply unique full-coordinate coefficient indices and a positive n_coefficients."
            )
        full = np.zeros((len(matrix), int(n_coefficients)))
        full[:, indices] = matrix
        return full

    def get_feature_names_out(self, input_features=None):
        self._require_fit()
        if input_features is not None and len(input_features) != self.n_features_in_:
            raise ValueError("History feature names must match the fitted lag count.")
        return np.asarray(
            [f"dl_{i}" for i in range(self.n_features_out_)], dtype=object
        )

    def save_model(self, path):
        """Persist fitted marginal bases using the shared trusted artifact format."""
        from ...persistence import save_estimator

        self._require_fit()
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        from ...persistence import load_estimator

        return load_estimator(path, cls)

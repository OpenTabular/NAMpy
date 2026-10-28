"""Sklearn-style adapters around the mgcv-parity :class:`nampy.gam.GAM`.

The adapters add zero numerics: they store constructor arguments verbatim
(sklearn clone contract), build a raw ``GAM`` inside ``fit``, and delegate.
Unlike the raw ``GAM`` (fixed smoothing by default, mirroring the low-level
core), the adapters default to automatic REML smoothing selection — the
behavior users expect from mgcv's ``gam()``. The raw class stays available
for full control (``estimator.gam_``).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ClassifierTags, RegressorTags

from ._gam import _GAMAdapterBase


class GAMRegressor(_GAMAdapterBase):
    """Regression adapter around the mgcv-parity GAM backend.

    Defaults to automatic REML smoothing selection (mgcv's ``gam()``
    behavior); pass ``optimize_smoothing=False`` with ``smoothing_params``
    for fixed smoothing. The fitted raw model is available as ``gam_``.
    """

    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def _validate_family_role(self, family) -> None:
        name = str(getattr(family, "name", "")).lower()
        n_predictors = int(getattr(family, "n_linear_predictors", 1))
        if (n_predictors != 1 or name in {"binomial", "quasibinomial", "ocat"}
                or getattr(family, "family_class", "") == "general"):
            raise ValueError(
                "GAMRegressor requires a single-predictor regression family; "
                f"got {name!r} with {n_predictors} linear predictor(s). Use "
                "GAMClassifier for binary binomial models or GAMLSS for "
                "multi-parameter distributional models."
            )

    def fit(self, X, y=None, *, data=None, sample_weight=None, offset=None):
        """Fit the GAM. Array mode: ``fit(X, y)``. Formula mode: pass
        ``formula=`` to the constructor and a DataFrame as ``X`` or ``data``."""
        return self._fit_gam(
            X, y, data=data, sample_weight=sample_weight, offset=offset
        )

    def predict(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Response-scale prediction."""
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        return self.gam_.predict(
            X,
            type="response",
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction."""
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

    def evaluate(self, X, y_true, metrics=None):
        """Evaluate with a ``{name: metric_fn}`` dict (default: MSE)."""
        if metrics is None:
            metrics = {"Mean Squared Error": mean_squared_error}
        predictions = self.predict(X)
        return {
            metric_name: metric_func(y_true, predictions)
            for metric_name, metric_func in metrics.items()
        }


class GAMClassifier(_GAMAdapterBase):
    """Binary-classification adapter around the mgcv-parity GAM backend.

    Fits a binomial GAM on 0/1-encoded labels; ``classes_`` records the
    original labels. Only binary targets are supported.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        formula=None,
        family="binomial",
        k=10,
        basis="tp",
        fit_intercept=True,
        optimize_smoothing=True,
        smoothing_method="reml",
        smoothing_optimizer="outer_newton",
        smoothing_params=None,
        select=False,
        knots=None,
        min_sp=None,
        para_pen=None,
        constraints=None,
        drop_intercept=None,
        covariance="bayes",
        score_gamma=1.0,
        max_irls_iter=200,
        irls_tol=1e-7,
        sp_log_bounds=(-80.0, 20.0),
        scale=0.0,
        control=None,
        nei=None,
        coefficient_optimizer="newton",
        optim_method=None,
    ):
        super().__init__(
            formula=formula,
            family=family,
            k=k,
            basis=basis,
            fit_intercept=fit_intercept,
            optimize_smoothing=optimize_smoothing,
            smoothing_method=smoothing_method,
            smoothing_optimizer=smoothing_optimizer,
            smoothing_params=smoothing_params,
            select=select,
            knots=knots,
            min_sp=min_sp,
            para_pen=para_pen,
            constraints=constraints,
            drop_intercept=drop_intercept,
            covariance=covariance,
            score_gamma=score_gamma,
            max_irls_iter=max_irls_iter,
            irls_tol=irls_tol,
            sp_log_bounds=sp_log_bounds,
            scale=scale,
            control=control,
            nei=nei,
            coefficient_optimizer=coefficient_optimizer,
            optim_method=optim_method,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

    def _validate_family_role(self, family) -> None:
        name = str(getattr(family, "name", "")).lower()
        n_predictors = int(getattr(family, "n_linear_predictors", 1))
        if name not in {"binomial", "quasibinomial"} or n_predictors != 1:
            raise ValueError(
                "GAMClassifier supports binary binomial and quasibinomial families only; "
                f"got {name!r} with {n_predictors} linear predictor(s)."
            )

    def fit(self, X, y, *, sample_weight=None, offset=None):
        target = np.asarray(y)
        if target.ndim > 1 and not (target.ndim == 2 and target.shape[1] == 1):
            raise ValueError(
                "GAMClassifier requires one binary class label per observation. "
                "For grouped success/failure counts use nampy.gam.GAM(family='binomial')."
            )
        y = np.asarray(y).ravel()
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "GAMClassifier supports binary targets only; "
                f"got {len(self.classes_)} classes."
            )
        y01 = self._label_encoder.transform(y).astype(np.float64)
        return self._fit_gam(X, y01, sample_weight=sample_weight, offset=offset)

    def predict_proba(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Class probabilities; columns follow ``self.classes_``."""
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        p1 = np.asarray(
            self.gam_.predict(
                X,
                type="response",
                offset=offset,
                block_size=block_size,
                newdata_guaranteed=newdata_guaranteed,
                na_action=na_action,
            ),
            dtype=np.float64,
        )
        return np.column_stack([1.0 - p1, p1])

    def predict(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Predicted class labels; missing rows under na_action='pass' carry NaN."""
        p1 = self.predict_proba(
            X,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )[:, 1]
        labels = self.classes_[(p1 >= 0.5).astype(int)]
        missing = np.isnan(p1)
        if np.any(missing):
            # Object dtype preserves string, integer and boolean class labels
            # while representing an unavailable prediction as a real NaN.
            labels = labels.astype(object)
            labels[missing] = np.nan
        return labels

    def decision_function(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Link-scale (logit) decision values."""
        return self.predict_link(
            X,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels."""
        return float(accuracy_score(y, self.predict(X), sample_weight=sample_weight))

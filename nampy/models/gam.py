"""Sklearn-style adapters around the mgcv-parity :class:`nampy.gam.GAM`.

The adapters add zero numerics: they store constructor arguments verbatim
(sklearn clone contract), build a raw ``GAM`` inside ``fit``, and delegate.
Unlike the raw ``GAM`` (fixed smoothing by default, mirroring the low-level
core), the adapters default to automatic REML smoothing selection — the
behavior users expect from mgcv's ``gam()``. The raw class stays available
for full control (``estimator.gam_``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ClassifierTags, RegressorTags

from ..contracts import AdditivePrediction, FeatureSchema
from ..gam import GAM

AdapterT = TypeVar("AdapterT", bound="_GAMAdapterBase")

_GAM_CONSTRUCTOR_PARAMS = (
    "k",
    "basis",
    "fit_intercept",
    "optimize_smoothing",
    "smoothing_method",
    "smoothing_optimizer",
    "smoothing_params",
    "select",
    "knots",
    "min_sp",
    "drop_intercept",
    "covariance",
    "score_gamma",
    "max_irls_iter",
    "irls_tol",
    "sp_log_bounds",
    "scale",
    "control",
    "nei",
    "coefficient_optimizer",
    "optim_method",
)


class _GAMAdapterBase(BaseEstimator):
    """Shared plumbing for the GAM adapters. Not part of the public API."""

    def __init__(
        self,
        formula=None,
        family=None,
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
        self.formula = formula
        self.family = family
        self.k = k
        self.basis = basis
        self.fit_intercept = fit_intercept
        self.optimize_smoothing = optimize_smoothing
        self.smoothing_method = smoothing_method
        self.smoothing_optimizer = smoothing_optimizer
        self.smoothing_params = smoothing_params
        self.select = select
        self.knots = knots
        self.min_sp = min_sp
        self.drop_intercept = drop_intercept
        self.covariance = covariance
        self.score_gamma = score_gamma
        self.max_irls_iter = max_irls_iter
        self.irls_tol = irls_tol
        self.sp_log_bounds = sp_log_bounds
        self.scale = scale
        self.control = control
        self.nei = nei
        self.coefficient_optimizer = coefficient_optimizer
        self.optim_method = optim_method

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _resolved_family(self):
        return self.family

    def _resolved_formula(self):
        return self.formula

    def _validate_family_role(self, family) -> None:
        """Validate the fitted family's public estimator role."""
        return None

    def _build_gam(self) -> GAM:
        hparams = {name: getattr(self, name) for name in _GAM_CONSTRUCTOR_PARAMS}
        if self.formula is not None:
            hparams["formula"] = self._resolved_formula()
        return GAM(family=self._resolved_family(), **hparams)

    def _fit_gam(self, X, y, *, data=None, sample_weight=None, offset=None):
        if self.formula is not None:
            frame = data if data is not None else X
            self.schema_ = None
            self.gam_ = self._build_gam()
            self._validate_family_role(self.gam_.family)
            self.gam_.fit(data=frame, y=y, sample_weight=sample_weight, offset=offset)
        else:
            if X is None:
                raise ValueError("X is required when no formula is given.")
            self.schema_ = FeatureSchema.from_data(
                X if hasattr(X, "columns") else np.asarray(X)
            )
            self.gam_ = self._build_gam()
            self._validate_family_role(self.gam_.family)
            self.gam_.fit(X, y, sample_weight=sample_weight, offset=offset)

        self.n_features_in_ = (
            self.schema_.n_features if self.schema_ is not None else None
        )
        if self.schema_ is not None:
            self.feature_names_in_ = np.asarray(
                self.schema_.feature_names, dtype=object
            )
        return self

    def _check_fitted(self):
        if getattr(self, "gam_", None) is None:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

    def _validate_X(self, X):
        if X is not None and getattr(self, "schema_", None) is not None:
            self.schema_.validate(X if hasattr(X, "columns") else np.asarray(X))
        return X

    # ------------------------------------------------------------------
    # Shared prediction surface
    # ------------------------------------------------------------------

    def predict_link(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Linear-predictor (link-scale) prediction."""
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        return self.gam_.predict(
            X,
            type="link",
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

    def predict_components(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ) -> AdditivePrediction:
        """Per-term link-scale contributions as a backend-neutral result."""
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        vals = self.gam_.predict_terms(
            X,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

        link = np.asarray(vals["output"], dtype=np.float64)
        response = vals.get("response")
        if response is None:
            response = self.gam_.family.inverse_link(link)

        intercept_val = vals.get("intercept")
        if intercept_val is None:
            intercept: float | np.ndarray = 0.0
        else:
            intercept_arr = np.asarray(intercept_val, dtype=np.float64)
            intercept = (
                float(intercept_arr) if intercept_arr.size == 1 else intercept_arr
            )

        reserved = {"output", "response", "intercept", "offset"}
        label_map = self._term_label_map()
        terms = {
            label_map.get(key, key): np.asarray(value, dtype=np.float64)
            for key, value in vals.items()
            if key not in reserved
        }
        offset_arr = vals.get("offset")

        return AdditivePrediction(
            response=np.asarray(response, dtype=np.float64),
            link=link,
            terms=terms,
            intercept=intercept,
            backend="gam",
            offset=None if offset_arr is None else np.asarray(offset_arr),
        )

    def explain_terms(self, X=None, *, max_bins: int = 64, offset=None):
        """Return a binned additive-term table shared with neural estimators."""
        from ..explanations import explain_additive_prediction

        if X is None:
            raise ValueError("X is required to construct an explanation table.")
        return explain_additive_prediction(
            X, self.predict_components(X, offset=offset), max_bins=max_bins
        )

    def term_importance(self, X=None, *, offset=None):
        """Return mean absolute link-scale contribution by additive term."""
        from ..explanations import term_importance_table

        return term_importance_table(self.predict_components(X, offset=offset))

    def interaction_importance(self, X=None, *, offset=None):
        """Return the interaction-only subset of :meth:`term_importance`."""
        table = self.term_importance(X, offset=offset)
        return table.loc[table["term_type"] == "interaction"].reset_index(drop=True)

    def _term_label_map(self) -> dict[str, str]:
        """Map opaque term ids to human-readable term labels when unique."""
        gam_result = getattr(self.gam_, "gam_result_", None)
        compiled_model = None if gam_result is None else gam_result.compiled_model
        if compiled_model is None:
            return {}
        ids = [term.term_id for term in compiled_model.compiled_terms]
        labels = [term.label for term in compiled_model.compiled_terms]
        if len(set(labels)) != len(labels):
            return {}
        return dict(zip(ids, labels, strict=True))

    def standard_errors(
        self,
        X=None,
        type="response",
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
        unconditional=False,
    ):
        """Pointwise prediction standard errors."""
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        _, se = self.gam_.predict(
            X,
            return_se=True,
            type=type,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
            unconditional=unconditional,
        )
        return se

    def lpmatrix(
        self,
        X,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Linear-predictor matrix for new data."""
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        return self.gam_.lpmatrix(
            X,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

    def summary(self, **kwargs):
        """Print and return the mgcv-style model summary."""
        self._check_fitted()
        return self.gam_.summary(**kwargs)

    def plot(self, **kwargs):
        """Render mgcv-style term plots (delegates to ``GAM.plot``).

        Same family of term-contribution plots as the neural wrappers'
        ``plot_terms``; both go through the shared renderer.
        """
        self._check_fitted()
        return self.gam_.plot(**kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path) -> Path:
        """Persist this adapter in the versioned NAMpy pickle format.

        Pickle artifacts are executable Python objects and must only be loaded
        from trusted sources.
        """
        destination = Path(path)
        payload = {
            "format": "nampy-estimator",
            "version": 1,
            "estimator_class": type(self).__name__,
            "estimator": self,
        }
        with destination.open("wb") as handle:
            pickle.dump(payload, handle)
        return destination

    @classmethod
    def load_model(cls: type[AdapterT], path: str | Path) -> AdapterT:
        """Load a version-1 adapter artifact written by :meth:`save_model`."""
        source = Path(path)
        with source.open("rb") as handle:
            loaded: object = pickle.load(handle)
        if (
            not isinstance(loaded, dict)
            or loaded.get("format") != "nampy-estimator"
            or loaded.get("version") != 1
        ):
            raise ValueError(f"{source} is not a supported NAMpy estimator artifact.")
        adapter = loaded.get("estimator")
        if not isinstance(adapter, cls):
            raise TypeError(
                f"{source} contains {type(adapter).__name__}, not {cls.__name__}."
            )
        return adapter


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
        if n_predictors != 1 or name in {"binomial", "ocat"}:
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
        if name != "binomial" or n_predictors != 1:
            raise ValueError(
                "GAMClassifier supports binary binomial families only; "
                f"got {name!r} with {n_predictors} linear predictor(s)."
            )

    def fit(self, X, y, *, sample_weight=None, offset=None):
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
        """Predicted class labels (original label dtype)."""
        p1 = self.predict_proba(
            X,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )[:, 1]
        return self.classes_[(p1 >= 0.5).astype(int)]

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

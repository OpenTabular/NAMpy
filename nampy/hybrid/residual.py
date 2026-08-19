"""Frozen-GAM baseline plus a neural residual correction, on the link scale.

Two-stage fit: an mgcv-parity GAM is fitted first (exact mgcv semantics),
then a neural additive model is trained on the SAME response with the GAM's
link-scale prediction as a fixed per-sample offset — the network learns only
what the smooth additive baseline missed. Predictions compose as

    response = inverse_link(eta_gam + eta_neural)

The composite is NOT an mgcv model: the GAM stage alone is mgcv-exact, the
correction stage is a neural fit, and the sum has no mgcv counterpart. It
never enters the parity test suites.

Supported families: gaussian (identity link), poisson (log link, Poisson NLL
on the composed linear predictor), binomial (logit link, classifier stage).
Formulas with ``offset(...)`` terms are rejected: stored formula offsets
apply only when predicting on training rows, so composite predictions on new
data would silently drop them (see ``gam/fit/offsets.py``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch.nn as nn
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.utils import ClassifierTags, RegressorTags

from ..api import AdditivePrediction, Capabilities
from ..gam import GAM
from ..models.classifier import NeuralClassifier
from ..models.regressor import NeuralRegressor

ResidualT = TypeVar("ResidualT", bound="_GAMResidualBase")


def _poisson_loss():
    return nn.PoissonNLLLoss(log_input=True)


# family -> (required neural base class, loss factory for the neural stage)
_REGRESSOR_FAMILIES = {
    "gaussian": (NeuralRegressor, None),
    "poisson": (NeuralRegressor, _poisson_loss),
}
_CLASSIFIER_FAMILIES = {
    "binomial": (NeuralClassifier, None),
}


class _GAMResidualBase(BaseEstimator):
    """Shared two-stage machinery. Not part of the public API."""

    _family_table: dict = {}

    def __init__(self, formula, neural, *, family, gam_kwargs=None):
        self.formula = formula
        self.neural = neural
        self.family = family
        self.gam_kwargs = gam_kwargs

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _validate_configuration(self):
        family = str(self.family).lower()
        if family not in self._family_table:
            raise ValueError(
                f"{type(self).__name__} supports families "
                f"{sorted(self._family_table)}; got {family!r}."
            )
        expected_base, loss_factory = self._family_table[family]
        if not isinstance(self.neural, expected_base):
            raise TypeError(
                f"family={family!r} requires an unfitted "
                f"{expected_base.__name__} subclass; got "
                f"{type(self.neural).__name__}."
            )
        if "offset(" in str(self.formula):
            raise ValueError(
                "Formulas with offset(...) terms are not supported: stored "
                "formula offsets apply only when predicting on training "
                "rows, so composite predictions on new data would silently "
                "drop them."
            )
        return family, loss_factory

    def fit(
        self, data, y=None, *, neural_features, val_data=None, neural_fit_kwargs=None
    ):
        """Fit the GAM on ``data`` via the formula, then the correction.

        Parameters
        ----------
        data : pandas.DataFrame
            Training table for the formula-mode GAM fit (must contain the
            response and every formula column).
        y : ignored
            Present for sklearn tooling (cross-validation) compatibility;
            the response always comes from the formula column in ``data``.
        neural_features : sequence of str
            Columns of ``data`` fed to the neural correction stage.
        val_data : pandas.DataFrame, optional
            Explicit validation rows (same columns as ``data``); the GAM
            link prediction on these rows becomes the validation offset.
        neural_fit_kwargs : dict, optional
            Extra keyword arguments for the neural stage's ``fit`` (epochs,
            batch size, trainer flags, ...).
        """
        family, loss_factory = self._validate_configuration()

        neural_features = list(neural_features)
        if not neural_features:
            raise ValueError("neural_features must name at least one column.")
        missing = [name for name in neural_features if name not in data.columns]
        if missing:
            raise ValueError(f"neural_features not found in data: {missing}")

        gam_params = {
            "optimize_smoothing": True,
            "smoothing_method": "reml",
        }
        gam_params.update(dict(self.gam_kwargs or {}))
        self.gam_ = GAM(formula=self.formula, family=family, **gam_params)
        self.gam_.fit(data=data)

        # Link-scale baseline on the training rows (X=None applies any
        # stored training-row semantics exactly).
        eta = np.asarray(self.gam_.predict(None, type="link"), dtype=np.float64)
        y = np.asarray(self.gam_.y_)

        fit_kwargs = dict(neural_fit_kwargs or {})
        if val_data is not None:
            response_name = self.gam_.formula_response_name_
            if response_name is None or response_name not in val_data.columns:
                raise ValueError(
                    "val_data must contain the formula response column "
                    f"{response_name!r}."
                )
            fit_kwargs["X_val"] = val_data[neural_features]
            fit_kwargs["y_val"] = np.asarray(val_data[response_name])
            fit_kwargs["offset_val"] = np.asarray(
                self.gam_.predict(val_data, type="link"), dtype=np.float64
            )
        if loss_factory is not None:
            fit_kwargs["loss_fct"] = loss_factory()

        # Clone: the passed estimator is a hyperparameter template and is
        # never mutated (required for cross-validation).
        self.neural_ = clone(self.neural)
        self.neural_features_ = neural_features
        self.neural_.fit(data[neural_features], y, offset=eta, **fit_kwargs)
        return self

    def _check_fitted(self):
        neural_fitted = (
            getattr(getattr(self, "neural_", None), "model", None) is not None
        )
        if getattr(self, "gam_", None) is None or not neural_fitted:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _neural_link(self, X) -> np.ndarray:
        output = self.neural_._predict(X[self.neural_features_])["output"]
        return output.squeeze(-1).cpu().numpy().astype(np.float64)

    def predict_link(self, X) -> np.ndarray:
        """Composite link-scale prediction ``eta_gam + eta_neural``."""
        self._check_fitted()
        eta_gam = np.asarray(self.gam_.predict(X, type="link"), dtype=np.float64)
        return eta_gam + self._neural_link(X)

    def _predict_response(self, X) -> np.ndarray:
        eta = self.predict_link(X)
        return np.asarray(self.gam_.family.inverse_link(eta), dtype=np.float64)

    def _gam_label_map(self) -> dict[str, str]:
        compiled_model = getattr(self.gam_, "compiled_model_", None)
        if compiled_model is None:
            return {}
        ids = [term.term_id for term in compiled_model.compiled_terms]
        labels = [term.label for term in compiled_model.compiled_terms]
        if len(set(labels)) != len(labels):
            return {}
        return dict(zip(ids, labels, strict=True))

    def predict_components(self, X) -> AdditivePrediction:
        """Merged per-term contributions with ``gam:``/``nn:`` prefixes."""
        self._check_fitted()
        gam_components = self.gam_.predict_terms(X)
        neural_dict = self.neural_._predict(X[self.neural_features_])

        reserved = {"output", "response", "intercept", "offset"}
        label_map = self._gam_label_map()
        terms: dict[str, np.ndarray] = {}
        for key, value in gam_components.items():
            if key not in reserved:
                label = label_map.get(key, key)
                terms[f"gam:{label}"] = np.asarray(value, dtype=np.float64)
        for key, value in neural_dict.items():
            if key in reserved:
                continue
            terms[f"nn:{key}"] = value.detach().cpu().numpy().astype(np.float64)

        eta = np.asarray(
            gam_components["output"], dtype=np.float64
        ) + self._neural_link(X)
        response = np.asarray(
            self.gam_.family.inverse_link(eta), dtype=np.float64
        )

        intercept_val = gam_components.get("intercept")
        intercept = 0.0 if intercept_val is None else float(intercept_val)

        return AdditivePrediction(
            response=response,
            link=eta,
            terms=terms,
            intercept=intercept,
            backend="hybrid",
        )

    # ------------------------------------------------------------------
    # Plotting / capabilities / persistence
    # ------------------------------------------------------------------

    def _gam_term_features(self) -> dict[str, str]:
        """Map prefixed 1-d gam term names to their raw feature column."""
        compiled_model = getattr(self.gam_, "compiled_model_", None)
        if compiled_model is None:
            return {}
        feature_names = getattr(self.gam_, "formula_feature_columns_", None)
        if not feature_names:
            n = int(np.asarray(self.gam_.X_).shape[1])
            feature_names = [f"x{index}" for index in range(n)]

        mapping: dict[str, str] = {}
        for term in compiled_model.compiled_terms:
            indices = list(term.feature_info.feature_indices)
            if len(indices) != 1:
                continue
            mapping[f"gam:{term.label}"] = str(feature_names[int(indices[0])])
        return mapping

    def plot(self, X, *, rug=None, pages=0, figsize=None):
        """Plot merged 1-d term contributions via the shared renderer."""
        from ..plotting import prepared_from_contributions, render_term_plots

        components = self.predict_components(X)
        term_features = self._gam_term_features()
        for name in components.terms:
            if name.startswith("nn:"):
                column = name.split(":", 1)[1]
                if column in getattr(X, "columns", ()):
                    term_features[name] = column
        prepared = prepared_from_contributions(
            X, components.terms, term_features=term_features
        )
        return render_term_plots(prepared, rug=rug, pages=pages, figsize=figsize)

    def evaluate(self, X, y_true, metrics=None):
        """Evaluate with a ``{name: metric_fn}`` dict on composite predictions."""
        if metrics is None:
            metrics = self._default_metrics()
        predictions = self.predict(X)
        return {
            metric_name: metric_func(y_true, predictions)
            for metric_name, metric_func in metrics.items()
        }

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_predict_proba=isinstance(self, GAMResidualClassifier),
            supports_standard_errors=False,
            supports_lpmatrix=False,
            supports_term_contributions=True,
        )

    def save_model(self, path: str | Path) -> Path:
        """Persist the composite (GAM stage + neural stage) to one file."""
        destination = Path(path)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)
        return destination

    @classmethod
    def load_model(cls: type[ResidualT], path: str | Path) -> ResidualT:
        """Load a composite previously written by :meth:`save_model`."""
        source = Path(path)
        with source.open("rb") as handle:
            loaded: object = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"{source} contains {type(loaded).__name__}, not {cls.__name__}."
            )
        return loaded


class GAMResidualRegressor(_GAMResidualBase):
    """mgcv-parity GAM baseline plus a neural residual correction.

    ``family="gaussian"`` (identity link) or ``"poisson"`` (log link; the
    neural stage trains with a Poisson NLL on the composed linear
    predictor). The composite is NOT an mgcv model; the GAM stage alone is
    mgcv-exact and available as ``gam_``.
    """

    _estimator_type = "regressor"
    _family_table = _REGRESSOR_FAMILIES

    def __init__(self, formula, neural, *, family="gaussian", gam_kwargs=None):
        super().__init__(formula, neural, family=family, gam_kwargs=gam_kwargs)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def predict(self, X):
        """Response-scale composite prediction."""
        return self._predict_response(X)

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction."""
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

    def _default_metrics(self):
        return {"Mean Squared Error": mean_squared_error}


class GAMResidualClassifier(_GAMResidualBase):
    """Binary GAM baseline plus a neural logit-scale correction.

    The formula response must be binary 0/1 (binomial GAM stage);
    ``classes_`` is ``[0, 1]``. The composite is NOT an mgcv model.
    """

    _estimator_type = "classifier"
    _family_table = _CLASSIFIER_FAMILIES

    def __init__(self, formula, neural, *, family="binomial", gam_kwargs=None):
        super().__init__(formula, neural, family=family, gam_kwargs=gam_kwargs)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

    def fit(
        self, data, y=None, *, neural_features, val_data=None, neural_fit_kwargs=None
    ):
        super().fit(
            data,
            neural_features=neural_features,
            val_data=val_data,
            neural_fit_kwargs=neural_fit_kwargs,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities; columns follow ``self.classes_``."""
        p1 = self._predict_response(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Predicted 0/1 labels."""
        p1 = self._predict_response(X)
        return self.classes_[(p1 >= 0.5).astype(int)]

    def decision_function(self, X) -> np.ndarray:
        """Composite link-scale (logit) decision values."""
        return self.predict_link(X)

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels."""
        return float(
            accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        )

    def _default_metrics(self):
        return {"Accuracy": accuracy_score}

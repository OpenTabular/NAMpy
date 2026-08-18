"""Frozen-GAM baseline plus neural correction, composed on the link scale.

Two-stage fit: an mgcv-parity GAM is fitted first (exact mgcv semantics),
then a neural additive model is trained on the SAME response with the GAM's
link-scale prediction as a fixed per-sample offset — so the network learns
only what the smooth additive baseline missed. Predictions compose as

    response = inverse_link(eta_gam + eta_neural)

This composite is NOT an mgcv model. The GAM stage alone is mgcv-exact; the
correction stage is a neural fit, and the sum has no mgcv counterpart. It
must never be compared in parity suites.

v1 scope: gaussian (identity link) and binomial (logit link) families;
formulas with ``offset(...)`` terms are rejected (predict-time offsets with
explicit new data would silently drop them, see gam/fit/offsets.py).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..api import AdditivePrediction, Capabilities
from ..gam import GAM
from ..models.sklearn_classifier import SklearnBaseClassifier
from ..models.sklearn_regressor import SklearnBaseRegressor

_SUPPORTED_FAMILIES = {
    "gaussian": SklearnBaseRegressor,
    "binomial": SklearnBaseClassifier,
}


class GAMPlusNeural:
    """Frozen GAM baseline with an offset-trained neural correction.

    Parameters
    ----------
    formula : str
        mgcv-style formula for the GAM stage, including the response
        (e.g. ``"y ~ s(x0) + s(x1)"``). ``offset(...)`` terms are rejected.
    neural : estimator
        An UNFITTED NAMpy neural estimator: a regressor subclass for
        ``family="gaussian"``, a classifier subclass for
        ``family="binomial"``.
    family : str, default "gaussian"
        ``"gaussian"`` (identity link) or ``"binomial"`` (logit link).
    gam_kwargs : dict, optional
        Extra hyperparameters for the GAM stage (defaults: automatic REML
        smoothing selection).
    """

    def __init__(self, formula, neural, *, family="gaussian", gam_kwargs=None):
        family = str(family).lower()
        if family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                "GAMPlusNeural supports families "
                f"{sorted(_SUPPORTED_FAMILIES)}; got {family!r}."
            )
        expected_base = _SUPPORTED_FAMILIES[family]
        if not isinstance(neural, expected_base):
            raise TypeError(
                f"family={family!r} requires an unfitted "
                f"{expected_base.__name__} subclass; got "
                f"{type(neural).__name__}."
            )
        if getattr(neural, "model", None) is not None:
            raise ValueError("The neural estimator must be unfitted.")
        if "offset(" in str(formula):
            raise ValueError(
                "Formulas with offset(...) terms are not supported by "
                "GAMPlusNeural: stored formula offsets apply only when "
                "predicting on training rows, so composite predictions on "
                "new data would silently drop them."
            )

        self.formula = str(formula)
        self.family = family
        self.neural_ = neural
        self.gam_kwargs = dict(gam_kwargs or {})
        self.gam_ = None
        self.neural_features_ = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data, *, neural_features, neural_fit_kwargs=None):
        """Fit the GAM on ``data`` via the formula, then the correction.

        Parameters
        ----------
        data : pandas.DataFrame
            Training table for the formula-mode GAM fit (must contain the
            response and every formula column).
        neural_features : sequence of str
            Columns of ``data`` fed to the neural correction. Required and
            explicit — the correction sees only these features.
        neural_fit_kwargs : dict, optional
            Extra keyword arguments for the neural estimator's ``fit``
            (epochs, batch size, trainer flags, ...).
        """
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
        gam_params.update(self.gam_kwargs)
        self.gam_ = GAM(formula=self.formula, family=self.family, **gam_params)
        self.gam_.fit(data=data)

        # Link-scale baseline on the training rows (X=None applies any
        # stored training-row semantics exactly).
        eta = np.asarray(self.gam_.predict(None, type="link"), dtype=np.float64)
        y = np.asarray(self.gam_.y_)

        self.neural_features_ = neural_features
        fit_kwargs = dict(neural_fit_kwargs or {})
        self.neural_.fit(data[neural_features], y, offset=eta, **fit_kwargs)
        return self

    def _check_fitted(self):
        if self.gam_ is None or getattr(self.neural_, "model", None) is None:
            raise ValueError(
                "This GAMPlusNeural instance is not fitted yet. "
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

    def predict(self, X, *, type="response"):
        """Composite prediction on the response (default) or link scale."""
        if type not in {"response", "link"}:
            raise ValueError("type must be 'response' or 'link'.")
        eta = self.predict_link(X)
        if type == "link":
            return eta
        return np.asarray(self.gam_.family.inverse_link(eta), dtype=np.float64)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities (binomial family only)."""
        if self.family != "binomial":
            raise AttributeError(
                "predict_proba is only available for family='binomial'."
            )
        p1 = self.predict(X, type="response")
        return np.column_stack([1.0 - p1, p1])

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
        gam_components = self.gam_.predict_feature_vals(X)
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
    # Scoring / capabilities / plotting / persistence
    # ------------------------------------------------------------------

    def score(self, X, y):
        """R^2 for gaussian; accuracy against 0/1 labels for binomial."""
        from sklearn.metrics import accuracy_score, r2_score

        if self.family == "binomial":
            p1 = self.predict(X, type="response")
            return float(accuracy_score(np.asarray(y), (p1 >= 0.5).astype(int)))
        return float(r2_score(np.asarray(y), self.predict(X)))

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_predict_proba=self.family == "binomial",
            supports_standard_errors=False,
            supports_lpmatrix=False,
            supports_term_contributions=True,
        )

    def plot(self, X, *, rug=None, pages=0, figsize=None):
        """Plot merged 1-d term contributions via the shared renderer."""
        from ..plotting import prepared_from_contributions, render_term_plots

        components = self.predict_components(X)
        # Strip the backend prefixes for frame-column lookup, keeping the
        # prefixed name as the axis label via disambiguation when both
        # backends contribute a term for the same column.
        terms = {}
        for name, values in components.terms.items():
            bare = name.split(":", 1)[1]
            key = bare if bare not in terms else name
            terms[key] = values
        prepared = prepared_from_contributions(X, terms)
        return render_term_plots(prepared, rug=rug, pages=pages, figsize=figsize)

    def save_model(self, path: str | Path) -> Path:
        """Persist the composite (GAM stage + neural stage) to one file."""
        destination = Path(path)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)
        return destination

    @classmethod
    def load_model(cls, path: str | Path) -> GAMPlusNeural:
        """Load a composite previously written by :meth:`save_model`."""
        source = Path(path)
        with source.open("rb") as handle:
            loaded: object = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"{source} contains {type(loaded).__name__}, not {cls.__name__}."
            )
        return loaded

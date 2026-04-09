# models/gam.py
import numpy as np

from ..basemodels.gam import GAM
from ..gam.families import make_gam_family
from ..gam.formula import parse_gam_formula
from ..gam.parity import (
    assert_parity_snapshot_close,
    build_parity_snapshot,
    compare_parity_snapshots,
    load_parity_snapshot,
    save_parity_snapshot,
)
from ..gam.parity.snapshots import _coerce_snapshot_arrays


class GAMRegressor:
    """
    User-facing wrapper for classical GAM regression.
    """

    def __init__(self, family="gaussian", **kwargs):
        self.family = make_gam_family(family)
        self.model = GAM(
            family=self.family,
            **kwargs,
        )

    def fit(
        self,
        X=None,
        y=None,
        data=None,
        formula=None,
        optimize_smoothing=None,
        smoothing_method=None,
        offset=None,
        knots=None,
        min_sp=None,
        drop_intercept=None,
        **kwargs,
    ):
        self.model.fit(
            X=X,
            y=y,
            data=data,
            formula=formula,
            offset=offset,
            knots=knots,
            min_sp=min_sp,
            drop_intercept=drop_intercept,
            optimize_smoothing=optimize_smoothing,
            smoothing_method=smoothing_method,
        )
        return self

    def fit_result(self, include_covariances=True):
        return self.model.fit_result(include_covariances=include_covariances)

    def parity_snapshot(self, X=None, include_covariances=False):
        return build_parity_snapshot(
            self.model,
            X=X,
            include_covariances=include_covariances,
        )

    def save_parity_snapshot(self, path, X=None, include_covariances=False):
        snap = self.parity_snapshot(X=X, include_covariances=include_covariances)
        save_parity_snapshot(snap, path)

    def compare_to_parity_snapshot(
        self,
        path_or_snapshot,
        X=None,
        include_covariances=False,
        atol=1e-6,
        rtol=1e-6,
    ):
        actual = _coerce_snapshot_arrays(
            self.parity_snapshot(X=X, include_covariances=include_covariances)
        )
        expected = (
            load_parity_snapshot(path_or_snapshot)
            if isinstance(path_or_snapshot, str)
            else path_or_snapshot
        )
        expected = _coerce_snapshot_arrays(expected)
        return compare_parity_snapshots(actual, expected, atol=atol, rtol=rtol)

    def assert_parity_snapshot_close(
        self,
        path_or_snapshot,
        X=None,
        include_covariances=False,
        atol=1e-6,
        rtol=1e-6,
    ):
        actual = _coerce_snapshot_arrays(
            self.parity_snapshot(X=X, include_covariances=include_covariances)
        )
        expected = (
            load_parity_snapshot(path_or_snapshot)
            if isinstance(path_or_snapshot, str)
            else path_or_snapshot
        )
        expected = _coerce_snapshot_arrays(expected)
        return assert_parity_snapshot_close(actual, expected, atol=atol, rtol=rtol)

    def predict(self, X, return_se=False, cov=None, type="response", offset=None):
        return self.model.predict(
            X,
            return_se=return_se,
            cov=cov,
            type=type,
            offset=offset,
        )

    def predict_feature_vals(self, X, offset=None):
        return self.model.predict_feature_vals(X, offset=offset)

    def lpmatrix(self, X):
        return self.model.lpmatrix(X)

    def summary(self):
        return self.model.summary()

    def plot(self, X=None, y=None, n_cols=2, figsize=None):
        return self.model.plot(X=X, y=y, n_cols=n_cols, figsize=figsize)

    def evaluate(self, X, y, metrics=None, offset=None):
        y = np.asarray(y, dtype=np.float64).ravel()
        preds = np.asarray(self.predict(X, offset=offset), dtype=np.float64).ravel()

        if preds.shape[0] != y.shape[0]:
            raise ValueError("Prediction length does not match y length.")

        default_metrics = {
            "MSE": lambda yt, yp: float(np.mean((yt - yp) ** 2)),
            "MAE": lambda yt, yp: float(np.mean(np.abs(yt - yp))),
            "RMSE": lambda yt, yp: float(np.sqrt(np.mean((yt - yp) ** 2))),
            "R2": lambda yt, yp: float(
                1.0 - np.sum((yt - yp) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-15)
            ),
        }

        if metrics is None:
            metrics = default_metrics

        out = {}
        for name, fn in metrics.items():
            out[name] = fn(y, preds)
        return out

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path, device="cpu"):
        self.model.load_model(path, device=device)
        self.family = self.model.family
        return self


class GAMClassifier:
    """
    Binary GAM classifier using a binomial-logit family.

    Notes
    -----
    - Supports binary classification only.
    - Arbitrary two-class labels are internally encoded to {0, 1}.
    - `predict_proba()` returns probabilities for [class_0, class_1].
    - `decision_function()` returns the additive predictor (log-odds / logit).
    """

    def __init__(self, family="binomial", threshold=0.5, **kwargs):
        fam = make_gam_family(family)
        if fam.name != "binomial":
            raise ValueError("GAMClassifier currently supports only family='binomial'.")

        self.family = fam
        self.threshold = float(threshold)

        self.model = GAM(
            family=self.family,
            **kwargs,
        )

        self.classes_ = None
        self.class_to_index_ = None

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------

    def _encode_labels(self, y):
        y = np.asarray(y).ravel()
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional for classification.")

        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                f"GAMClassifier currently supports binary classification only; "
                f"got {classes.shape[0]} unique classes."
            )

        self.classes_ = classes
        self.class_to_index_ = {
            classes[0]: 0.0,
            classes[1]: 1.0,
        }

        y_enc = np.array([self.class_to_index_[v] for v in y], dtype=np.float64)
        return y_enc

    def _check_is_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("Classifier is not fitted.")

    # ------------------------------------------------------------------
    # Fitting / prediction
    # ------------------------------------------------------------------

    def fit(
        self,
        X=None,
        y=None,
        data=None,
        formula=None,
        optimize_smoothing=None,
        smoothing_method=None,
        offset=None,
        knots=None,
        min_sp=None,
        drop_intercept=None,
        **kwargs,
    ):
        if formula is not None and y is None:
            data_obj = data if data is not None else X
            parsed = parse_gam_formula(formula)

            if parsed.response_name is None:
                raise ValueError(
                    "Formula-based classification requires a response in the formula "
                    "if `y` is not supplied explicitly."
                )
            if data_obj is None or not hasattr(data_obj, "__getitem__"):
                raise ValueError(
                    "Formula-based classification requires `data` as a DataFrame "
                    "(or pass the DataFrame as `X`)."
                )
            if parsed.response_name not in data_obj.columns:
                raise KeyError(
                    f"Response column {parsed.response_name!r} not found in data."
                )

            y_raw = np.asarray(data_obj[parsed.response_name]).ravel()
        else:
            y_raw = y

        y_enc = self._encode_labels(y_raw)
        self.model.fit(
            X=X,
            y=y_enc,
            data=data,
            formula=formula,
            offset=offset,
            knots=knots,
            min_sp=min_sp,
            drop_intercept=drop_intercept,
            optimize_smoothing=optimize_smoothing,
            smoothing_method=smoothing_method,
        )
        return self

    def fit_result(self, include_covariances=True):
        return self.model.fit_result(include_covariances=include_covariances)

    def parity_snapshot(self, X=None, include_covariances=False):
        return build_parity_snapshot(
            self.model,
            X=X,
            include_covariances=include_covariances,
        )

    def save_parity_snapshot(self, path, X=None, include_covariances=False):
        snap = self.parity_snapshot(X=X, include_covariances=include_covariances)
        save_parity_snapshot(snap, path)

    def compare_to_parity_snapshot(
        self,
        path_or_snapshot,
        X=None,
        include_covariances=False,
        atol=1e-6,
        rtol=1e-6,
    ):
        actual = _coerce_snapshot_arrays(
            self.parity_snapshot(X=X, include_covariances=include_covariances)
        )
        expected = (
            load_parity_snapshot(path_or_snapshot)
            if isinstance(path_or_snapshot, str)
            else path_or_snapshot
        )
        expected = _coerce_snapshot_arrays(expected)
        return compare_parity_snapshots(actual, expected, atol=atol, rtol=rtol)

    def assert_parity_snapshot_close(
        self,
        path_or_snapshot,
        X=None,
        include_covariances=False,
        atol=1e-6,
        rtol=1e-6,
    ):
        actual = _coerce_snapshot_arrays(
            self.parity_snapshot(X=X, include_covariances=include_covariances)
        )
        expected = (
            load_parity_snapshot(path_or_snapshot)
            if isinstance(path_or_snapshot, str)
            else path_or_snapshot
        )
        expected = _coerce_snapshot_arrays(expected)
        return assert_parity_snapshot_close(actual, expected, atol=atol, rtol=rtol)

    def decision_function(self, X, return_se=False, cov=None, offset=None):
        self._check_is_fitted()
        return self.model.predict(
            X,
            return_se=return_se,
            cov=cov,
            type="link",
            offset=offset,
        )

    def predict_proba(self, X, return_se=False, cov=None, offset=None):
        self._check_is_fitted()

        if not return_se:
            p1 = np.asarray(
                self.model.predict(
                    X,
                    return_se=False,
                    cov=cov,
                    type="response",
                    offset=offset,
                ),
                dtype=np.float64,
            ).ravel()
            p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
            return np.column_stack([1.0 - p1, p1])

        p1, se = self.model.predict(
            X,
            return_se=True,
            cov=cov,
            type="response",
            offset=offset,
        )
        p1 = np.asarray(p1, dtype=np.float64).ravel()
        se = np.asarray(se, dtype=np.float64).ravel()
        p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - p1, p1]), se

    def predict(self, X, offset=None):
        self._check_is_fitted()
        p1 = self.predict_proba(X, offset=offset)[:, 1]
        pred01 = (p1 >= self.threshold).astype(int)
        return self.classes_[pred01]

    def predict_feature_vals(self, X, offset=None):
        self._check_is_fitted()
        return self.model.predict_feature_vals(X, offset=offset)

    def lpmatrix(self, X):
        self._check_is_fitted()
        return self.model.lpmatrix(X)

    def summary(self):
        self._check_is_fitted()
        return self.model.summary()

    def plot(self, X=None, y=None, n_cols=2, figsize=None):
        self._check_is_fitted()
        return self.model.plot(X=X, y=y, n_cols=n_cols, figsize=figsize)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X, y, metrics=None, offset=None):
        self._check_is_fitted()

        y = np.asarray(y).ravel()
        if np.any(~np.isin(y, self.classes_)):
            raise ValueError("y contains labels not seen during fit().")

        y01 = np.array([self.class_to_index_[v] for v in y], dtype=np.float64)
        proba = self.predict_proba(X, offset=offset)[:, 1]
        preds = (proba >= self.threshold).astype(float)

        def logloss(yt, p):
            p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
            yt = np.asarray(yt, dtype=np.float64)
            return float(-np.mean(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p)))

        default_metrics = {
            "accuracy": lambda yt, yp: float(np.mean(yt == yp)),
            "brier": lambda yt, p: float(np.mean((p - yt) ** 2)),
            "logloss": logloss,
        }

        if metrics is None:
            metrics = default_metrics

        out = {}
        for name, fn in metrics.items():
            lname = str(name).lower()
            if lname in {"accuracy", "acc"}:
                out[name] = fn(y01, preds)
            else:
                out[name] = fn(y01, proba)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path):
        payload = {
            "model": self.model,
            "classes_": self.classes_,
            "threshold": self.threshold,
        }
        import pickle

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_model(self, path, device="cpu"):
        import pickle

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.family = self.model.family
        self.classes_ = payload["classes_"]
        self.threshold = float(payload.get("threshold", 0.5))
        self.class_to_index_ = {
            self.classes_[0]: 0.0,
            self.classes_[1]: 1.0,
        }
        return self

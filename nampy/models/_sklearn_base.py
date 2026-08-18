from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any, TypeVar

from pretab.preprocessor import Preprocessor
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..neural.training.engine import predict_raw as _engine_predict_raw
from ..neural.training.engine import run_training
from ._sklearn_data import prepare_fit_features, prepare_predict_features

EstimatorT = TypeVar("EstimatorT", bound="NeuralEstimatorBase")


def _preprocessor_defaults() -> dict[str, Any]:
    parameters = inspect.signature(Preprocessor.__init__).parameters
    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _normalize_preprocessor_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if normalized.get("categorical_preprocessing") in ("one_hot", "one-hot"):
        normalized["categorical_preprocessing"] = "one-hot"
    if normalized.get("numerical_preprocessing") == "normalization":
        normalized["numerical_preprocessing"] = "minmax"
    return normalized


class NeuralEstimatorBase(BaseEstimator):
    """Base class for shared NAMpy sklearn estimator behavior."""

    def fit(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        """Train the model on ``X``/``y``; optionally validate on a held-out set.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like
            The target values; task-specific shape and dtype rules apply.
        val_size : float, default=0.2
            Proportion for the automatic validation split when ``X_val`` is
            None; ignored otherwise.
        X_val, y_val : optional
            Explicit validation data; must be provided together.
        max_epochs : int, default=100
            Maximum number of training epochs.
        random_state : int, default=101
            Seed for the automatic train/validation split.
        batch_size : int, default=128
            Samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data each epoch.
        patience : int, default=15
            Early-stopping patience on the monitored metric.
        monitor : str, default="val_loss"
            Metric monitored by early stopping and checkpoint selection.
        mode : str, default="min"
            Whether the monitored metric is minimized or maximized.
        lr, lr_patience, factor, weight_decay
            Optimizer and LR-scheduler settings.
        checkpoint_path : str, default="model_checkpoints"
            Root directory for per-fit checkpoint subdirectories.
        dataloader_kwargs : dict, optional
            Extra kwargs for the torch DataLoader.
        **trainer_kwargs
            Additional kwargs for PyTorch Lightning's Trainer.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        X = prepare_fit_features(self, X)
        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if X_val is not None:
            X_val = prepare_predict_features(self, X_val)

        y, y_val, plan = self._build_training_plan(y, y_val)

        run_training(
            self,
            X,
            y,
            plan,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            max_epochs=max_epochs,
            random_state=random_state,
            batch_size=batch_size,
            shuffle=shuffle,
            patience=patience,
            monitor=monitor,
            mode=mode,
            lr=lr,
            lr_patience=lr_patience,
            factor=factor,
            weight_decay=weight_decay,
            checkpoint_path=checkpoint_path,
            dataloader_kwargs=dataloader_kwargs,
            trainer_kwargs=trainer_kwargs,
        )
        return self

    def _build_training_plan(self, y, y_val):
        """Return ``(y, y_val, TrainingPlan)`` for this task. Task-specific."""
        raise NotImplementedError

    def _predict(self, X):
        """Run inference and return the raw forward-output dict."""
        if getattr(self, "model", None) is None or (
            getattr(self, "data_module", None) is None
        ):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
        X = prepare_predict_features(self, X)
        return _engine_predict_raw(self, X)

    def predict_feature_vals(self, X):
        """Return the raw forward-output dict (per-feature contributions)."""
        return self._predict(X)

    def _split_output_components(self, pred_dict):
        """Split a forward-output dict into (terms, intercept) numpy parts."""
        terms: dict[str, Any] = {}
        intercept: float | Any = 0.0
        for key, value in pred_dict.items():
            if key == "output":
                continue
            array = value.detach().cpu().numpy()
            if key == "intercept":
                intercept = float(array[0]) if array.size == 1 else array
            else:
                terms[key] = array
        return terms, intercept

    def _plot_series_labels(self, n_series: int):
        """Labels for the per-output plot lines; None means a single line."""
        return None

    def plot(self, X, y_true, feature_name=None, plot_interactions=False):
        """Plot per-feature effects (and optionally interaction heatmaps).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data for generating predictions.
        y_true : np.ndarray
            True target values shown as scatter behind the curves.
        feature_name : str, optional
            Specific feature to plot; all numerical features when None.
        plot_interactions : bool, optional
            Whether to also plot pairwise feature interactions.
        """
        from ._plotting import plot_feature_effects

        plot_feature_effects(
            self,
            X,
            y_true,
            feature_name=feature_name,
            plot_interactions=plot_interactions,
        )

    def plot_terms(self, X, *, rug=None, pages=0, figsize=None):
        """Render 1-d term-contribution curves via the shared renderer.

        Builds the same prepared-data schema the GAM plotting pipeline uses,
        so GAM, neural, and hybrid term plots share one renderer. Only
        single-column numeric main effects are drawn; interaction terms use
        :meth:`plot` with ``plot_interactions=True``.
        """
        import numpy as np

        from ..plotting import render_term_plots

        components = self.predict_components(X)
        frame = X if hasattr(X, "columns") else None

        pd_list = []
        for name, values in components.terms.items():
            if ":" in name:
                continue
            if frame is None or name not in frame.columns:
                continue
            try:
                x_raw = np.asarray(frame[name], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            contrib = np.asarray(values, dtype=np.float64).reshape(len(x_raw), -1)
            if contrib.shape[1] != 1:
                continue
            order = np.argsort(x_raw)
            pd_list.append(
                {
                    "kind": "1d",
                    "plot_me": True,
                    "x": x_raw[order],
                    "fit": contrib[order, 0],
                    "raw": x_raw,
                    "xlab": name,
                    "ylab": f"f({name})",
                    "main": "",
                    "scheme": 0,
                }
            )

        if not pd_list:
            raise ValueError("No numeric 1-d terms available to plot.")

        prepared = {
            "pd": pd_list,
            "ylim": None,
            "partial_resids": False,
            "by_resids": False,
            "shift": 0.0,
            "trans": lambda values: values,
            "jit": False,
            "select": None,
            "scale": False,
            "rug_default": len(pd_list[0]["raw"]) <= 10000,
        }
        return render_term_plots(prepared, rug=rug, pages=pages, figsize=figsize)

    def save_model(self, path: str | Path) -> Path:
        """Persist this estimator, including fitted state, to ``path``.

        The file uses Python's pickle protocol and must only be loaded from a
        trusted source.
        """
        destination = Path(path)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)
        return destination

    @classmethod
    def load_model(cls: type[EstimatorT], path: str | Path) -> EstimatorT:
        """Load an estimator previously written by :meth:`save_model`."""
        source = Path(path)
        with source.open("rb") as handle:
            loaded: object = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"{source} contains {type(loaded).__name__}, not {cls.__name__}."
            )
        return loaded

    def _initialize_estimator_parameters(self, config_class, kwargs):
        config_names = set(getattr(config_class, "__dataclass_fields__", {}))
        preprocessor_defaults = _preprocessor_defaults()
        preprocessor_names = set(preprocessor_defaults)

        flat_kwargs = {}
        nested_preprocessor_kwargs = {}
        for name, value in kwargs.items():
            if name.startswith("preprocessor__"):
                nested_preprocessor_kwargs[name.split("__", 1)[1]] = value
            else:
                flat_kwargs[name] = value

        unknown_flat = set(flat_kwargs) - config_names - preprocessor_names
        unknown_nested = set(nested_preprocessor_kwargs) - preprocessor_names
        if unknown_flat or unknown_nested:
            unknown = sorted(unknown_flat | unknown_nested)
            valid = sorted(config_names | preprocessor_names)
            raise TypeError(
                f"Unexpected parameter(s) {unknown} for {self.__class__.__name__}. "
                f"Valid parameters are {valid}."
            )

        config_inputs = {
            name: value for name, value in flat_kwargs.items() if name in config_names
        }
        explicit_preprocessor = {
            name: value
            for name, value in flat_kwargs.items()
            if name in preprocessor_names
        }
        explicit_preprocessor.update(nested_preprocessor_kwargs)
        explicit_preprocessor = _normalize_preprocessor_params(explicit_preprocessor)

        self._config_param_names = tuple(sorted(config_names))
        self._preprocessor_param_names = tuple(sorted(preprocessor_names))
        self.config = config_class(**config_inputs)
        self.config_kwargs = {
            name: getattr(self.config, name) for name in self._config_param_names
        }
        self._preprocessor_kwargs = dict(preprocessor_defaults)
        self._preprocessor_kwargs.update(explicit_preprocessor)
        self._provided_preprocessor_kwargs = explicit_preprocessor
        self._rebuild_preprocessor()

    def _rebuild_preprocessor(self):
        self.preprocessor = Preprocessor(**self._preprocessor_kwargs)

    def get_params(self, deep=True):
        params = dict(self.config_kwargs)
        for name, value in self._preprocessor_kwargs.items():
            if name not in params:
                params[name] = value
            elif params[name] != value:
                params[f"preprocessor__{name}"] = value

        if deep:
            params.update(
                {
                    f"preprocessor__{name}": value
                    for name, value in self._preprocessor_kwargs.items()
                }
            )
        return params

    def set_params(self, **parameters):
        config_updates = {}
        preprocessor_updates = {}

        for name, value in parameters.items():
            if name.startswith("preprocessor__"):
                preprocessor_name = name.split("__", 1)[1]
                if preprocessor_name not in self._preprocessor_param_names:
                    raise ValueError(
                        f"Invalid parameter {name!r} for {self.__class__.__name__}."
                    )
                preprocessor_updates[preprocessor_name] = value
                continue

            owns_config = name in self._config_param_names
            owns_preprocessor = name in self._preprocessor_param_names
            if not owns_config and not owns_preprocessor:
                valid = sorted(
                    set(self._config_param_names) | set(self._preprocessor_param_names)
                )
                raise ValueError(
                    f"Invalid parameter {name!r} for {self.__class__.__name__}. "
                    f"Valid parameters are {valid}."
                )
            if owns_config:
                config_updates[name] = value
            if owns_preprocessor:
                preprocessor_updates[name] = value

        for name, value in config_updates.items():
            setattr(self.config, name, value)
            self.config_kwargs[name] = value

        if preprocessor_updates:
            self._preprocessor_kwargs.update(
                _normalize_preprocessor_params(preprocessor_updates)
            )
            self._rebuild_preprocessor()

        return self

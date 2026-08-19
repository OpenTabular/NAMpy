from __future__ import annotations

import inspect
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pretab.preprocessor import Preprocessor
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..neural.data.datamodule import NAMpyDataModule
from ..neural.task import TaskModule
from ._data import prepare_fit_features, prepare_predict_features

EstimatorT = TypeVar("EstimatorT", bound="NeuralEstimatorBase")


@dataclass
class TrainingPlan:
    """Task-specific wiring for the shared neural training engine.

    Attributes
    ----------
    datamodule_regression : bool
        Label dtype switch for :class:`NAMpyDataModule` (float32 vs long).
    taskmodel_kwargs : dict
        Task wiring forwarded to :class:`TaskModule` (``num_classes``,
        ``task`` or ``lss``/``family``).
    stratify : array-like or None
        Stratification labels for the automatic train/validation split.
    """

    datamodule_regression: bool
    taskmodel_kwargs: dict[str, Any] = field(default_factory=dict)
    stratify: Any = None


def build_callbacks(name: str, *, monitor, mode, patience, checkpoint_path):
    """Build the EarlyStopping/ModelCheckpoint pair for one fit.

    Both callbacks honour the user's ``monitor``/``mode``. Each fit writes into
    its own subdirectory so concurrent or successive fits never clobber each
    other's checkpoints.
    """
    early_stop_callback = EarlyStopping(
        monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
    )
    run_dir = Path(checkpoint_path) / f"{name}-{uuid4().hex[:8]}"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        dirpath=str(run_dir),
        filename="best_model",
    )
    return early_stop_callback, checkpoint_callback


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
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs=None,
        offset=None,
        offset_val=None,
        loss_fct=None,
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
        lr, lr_patience, lr_factor, weight_decay
            Optional fit-time overrides for the estimator's optimizer and
            LR-scheduler configuration. ``None`` uses the corresponding
            constructor/config value.
        checkpoint_path : str, default="model_checkpoints"
            Root directory for per-fit checkpoint subdirectories.
        dataloader_kwargs : dict, optional
            Extra kwargs for the torch DataLoader.
        offset, offset_val : array-like, optional
            Per-sample additive offsets on the prediction (link) scale for
            the training and explicit-validation rows. Not supported for
            LSS tasks. Note that ``predict`` does NOT apply a stored offset;
            callers composing models must add offsets themselves.
        loss_fct : callable, optional
            Custom loss ``fn(preds, targets)`` replacing the task default
            (regression tasks only; e.g. ``nn.PoissonNLLLoss(log_input=True)``
            for count responses composed on the log-link scale).
        **trainer_kwargs
            Additional kwargs for PyTorch Lightning's Trainer.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        if "factor" in trainer_kwargs:
            raise TypeError(
                "fit() uses 'lr_factor'; the ambiguous 'factor' name is not "
                "forwarded to the Lightning Trainer."
            )
        X = prepare_fit_features(self, X)
        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if X_val is not None:
            X_val = prepare_predict_features(self, X_val)

        self._fit_loss_fct = loss_fct
        y, y_val, plan = self._build_training_plan(y, y_val)

        lr = self.config.lr if lr is None else lr
        lr_patience = self.config.lr_patience if lr_patience is None else lr_patience
        lr_factor = self.config.lr_factor if lr_factor is None else lr_factor
        weight_decay = (
            self.config.weight_decay if weight_decay is None else weight_decay
        )

        self._run_training(
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
            lr_factor=lr_factor,
            weight_decay=weight_decay,
            checkpoint_path=checkpoint_path,
            dataloader_kwargs=dataloader_kwargs,
            trainer_kwargs=trainer_kwargs,
            offset=offset,
            offset_val=offset_val,
        )
        return self

    def _build_training_plan(self, y, y_val):
        """Return ``(y, y_val, TrainingPlan)`` for this task. Task-specific."""
        raise NotImplementedError

    def _run_training(
        self,
        X,
        y,
        plan: TrainingPlan,
        *,
        val_size,
        X_val,
        y_val,
        max_epochs,
        random_state,
        batch_size,
        shuffle,
        patience,
        monitor,
        mode,
        lr,
        lr_patience,
        lr_factor,
        weight_decay,
        checkpoint_path,
        dataloader_kwargs,
        trainer_kwargs,
        offset=None,
        offset_val=None,
    ) -> None:
        """Fit this estimator in place: data module, TaskModule, Trainer, reload."""
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        self.data_module = NAMpyDataModule(
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=plan.datamodule_regression,
            **dataloader_kwargs,
        )
        self.data_module.setup_data(
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            stratify=plan.stratify,
            offset=offset,
            offset_val=offset_val,
        )

        self.model = TaskModule(
            model_class=self.base_model,
            config=self.config,
            cat_feature_info=self.data_module.cat_feature_info,
            num_feature_info=self.data_module.num_feature_info,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            weight_decay=weight_decay,
            **plan.taskmodel_kwargs,
        )

        early_stop_callback, checkpoint_callback = build_callbacks(
            type(self).__name__,
            monitor=monitor,
            mode=mode,
            patience=patience,
            checkpoint_path=checkpoint_path,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            **trainer_kwargs,
        )
        trainer.fit(self.model, self.data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])

    def _predict_raw(self, X) -> dict[str, torch.Tensor]:
        """Run inference on prepared features and return the forward-output dict.

        ``X`` must already have passed through the estimator's feature
        preparation. The model's train/eval state is restored afterwards.
        """
        cat_tensor_dict, num_tensor_dict = self.data_module.preprocess_test_data(X)

        device = next(self.model.parameters()).device
        cat_tensor_dict = {
            key: tensor.to(device) for key, tensor in cat_tensor_dict.items()
        }
        num_tensor_dict = {
            key: tensor.to(device) for key, tensor in num_tensor_dict.items()
        }

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self.model(
                    num_features=num_tensor_dict, cat_features=cat_tensor_dict
                )
        finally:
            self.model.train(was_training)

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
        return self._predict_raw(X)

    def _split_output_components(self, pred_dict):
        """Split a forward-output dict into (terms, intercept) numpy parts."""
        from ..neural.contracts import is_penalty_key

        terms: dict[str, Any] = {}
        intercept: float | Any = 0.0
        for key, value in pred_dict.items():
            if key == "output" or is_penalty_key(key):
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
        """Legacy density-style view of per-feature effects.

        Draws each feature's contribution curve over a density-shaded
        background with the (centered) targets scattered behind it. Kept for
        back-compat; prefer :meth:`plot_terms` for term-contribution curves
        via the shared renderer, and :meth:`plot_interactions` for pairwise
        interaction heatmaps.

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

    def plot_interactions(self, X):
        """Plot binned heatmaps for the fitted pairwise interaction terms.

        Renders one heatmap (per output series) for every ``":"``-keyed
        interaction term in the model's forward output.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data for generating predictions.
        """
        from ._plotting import plot_interaction_heatmaps

        plot_interaction_heatmaps(self, X)

    def plot_terms(self, X, *, rug=None, pages=0, figsize=None):
        """Render 1-d term-contribution curves via the shared renderer.

        Builds the same prepared-data schema the GAM plotting pipeline uses,
        so GAM and neural term plots share one renderer. Only single-column
        numeric main effects are drawn; interaction terms use
        :meth:`plot_interactions`.
        """
        from ..plotting import prepared_from_contributions, render_term_plots

        components = self.predict_components(X)
        prepared = prepared_from_contributions(X, components.terms)
        return render_term_plots(prepared, rug=rug, pages=pages, figsize=figsize)

    def save_model(self, path: str | Path) -> Path:
        """Persist this estimator in the versioned NAMpy pickle format.

        Pickle artifacts are executable Python objects and must only be loaded
        from trusted sources. The envelope makes incompatible formats fail
        explicitly instead of producing obscure unpickle errors.
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
    def load_model(cls: type[EstimatorT], path: str | Path) -> EstimatorT:
        """Load a version-1 estimator artifact written by :meth:`save_model`."""
        source = Path(path)
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        if (
            not isinstance(loaded, dict)
            or loaded.get("format") != "nampy-estimator"
            or loaded.get("version") != 1
        ):
            raise ValueError(f"{source} is not a supported NAMpy estimator artifact.")
        estimator = loaded.get("estimator")
        if not isinstance(estimator, cls):
            raise TypeError(
                f"{source} contains {type(estimator).__name__}, not {cls.__name__}."
            )
        return estimator

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

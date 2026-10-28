from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pretab.preprocessor import Preprocessor
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.class_weight import compute_sample_weight

from ..neural.training import run_training, seed_neural_runtime
from ._data import prepare_fit_features, prepare_predict_features
from ._persistence import load_estimator, save_estimator


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
        max_steps: int = -1,
        max_time=None,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        early_stopping_steps: int | None = None,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        lr_schedule: str | None = None,
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_decay_factor: float = 0.2,
        weight_decay: float | None = None,
        optimizer=None,
        optimizer_kwargs=None,
        warm_start=False,
        average_checkpoints: bool = False,
        n_last_checkpoints: int = 5,
        pretrain_epochs: int = 0,
        pretrain_max_steps: int = -1,
        pretraining_ratio: float = 0.15,
        pretraining_noise: float = 0.1,
        pretraining_feature_mask: bool = True,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs=None,
        offset=None,
        offset_val=None,
        sample_weight=None,
        sample_weight_val=None,
        class_weight=None,
        sampling_strategy=None,
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
            Maximum number of Lightning training epochs. Architectures with a
            native optimizer, such as IGANN, use their model-specific stage
            count instead (``n_estimators`` for IGANN).
        max_steps : int, default=-1
            Optional global optimizer-step limit. ``-1`` leaves it unlimited.
        max_time : optional
            Lightning-compatible wall-clock limit, such as ``"00:30:00"``.
        random_state : int, default=101
            Seed for Python, NumPy, Torch, the automatic train/validation split,
            and data-loader ordering. Architectures may expose a more specific
            seed, such as GPNAM's ``rff_random_state``.
        batch_size : int, default=128
            Samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data each epoch.
        patience : int, default=15
            Early-stopping patience on the monitored metric.
        early_stopping_steps : int, optional
            When provided, replace epoch-check patience with optimizer-step
            distance since the last monitored improvement.
        monitor : str, default="val_loss"
            Metric monitored by early stopping and checkpoint selection.
        mode : str, default="min"
            Whether the monitored metric is minimized or maximized.
        lr : float, optional
            Optimizer learning rate. None uses the estimator configuration.
        lr_patience : int, optional
            Plateau scheduler patience in monitored epochs; None uses configuration.
        lr_factor : float, optional
            Multiplicative learning-rate reduction for the plateau scheduler.
        lr_schedule : str, optional
            Learning-rate schedule: "plateau", "inverse_sqrt", "cosine",
            "warmup_cosine", or "none". None uses the estimator configuration.
        weight_decay : float, optional
            Optimizer weight-decay coefficient. None uses configuration.
        lr_warmup_steps : int, optional
            Number of optimizer steps for linear learning-rate warmup.
        lr_decay_steps : int, optional
            Step-decay interval, or total cosine horizon. A zero cosine horizon
            is inferred from the trainer's estimated optimizer steps.
        lr_decay_factor : float, optional
            Multiplicative factor for the step-decay scheduler.
        optimizer : str or optimizer class, optional
            Uses the architecture default when omitted. Supported names are
            ``"adam"``, ``"adamw"``, ``"adagrad"``, ``"sgd"``, and optional
            ``"qhadam"``; a compatible optimizer class is also accepted.
        optimizer_kwargs : dict, optional
            Extra optimizer constructor arguments.
        warm_start : bool or path, default=False
            Reuse the current fitted weights, or load weights from a Lightning
            checkpoint. This is weights-only; optimizer state is not resumed.
        average_checkpoints : bool, default=False
            Load the arithmetic mean of the latest epoch-end floating model
            states after fitting.
        n_last_checkpoints : int, default=5
            Number of recent epoch-end states retained for averaging.
        pretrain_epochs : int, default=0
            Run optional masked reconstruction before supervised fitting.
            Currently supported only by NodeGAM architectures.
        pretrain_max_steps : int, default=-1
            Maximum reconstruction-pretraining optimizer steps; -1 is unlimited.
        pretraining_ratio : float, default=0.15
            Masked reconstruction fraction for NodeGAM pretraining.
        pretraining_noise : float, default=0.1
            Noise level for the NodeGAM reconstruction objective.
        pretraining_feature_mask : bool, default=True
            Whether to mask source features during NodeGAM pretraining.
        checkpoint_path : str, default="model_checkpoints"
            Root directory for per-fit checkpoint subdirectories.
        dataloader_kwargs : dict, optional
            Extra kwargs for the torch DataLoader.
        offset, offset_val : array-like, optional
            Per-sample additive offsets on the prediction (link) scale for
            the training and explicit-validation rows. Not supported for
            LSS tasks. Note that ``predict`` does NOT apply a stored offset;
            callers composing models must add offsets themselves.
        sample_weight, sample_weight_val : array-like, optional
            Non-negative per-row training and explicit-validation loss weights.
            Supported by regression, classification, and distributional objectives.
        class_weight : dict or "balanced", optional
            Class weights combined multiplicatively with ``sample_weight``.
        sampling_strategy : {None, "balanced"}, default=None
            For classifiers, optionally draw inverse-frequency balanced
            training batches. This is independent from loss weighting.
        loss_fct : callable, optional
            Custom loss ``fn(preds, targets)`` replacing the task default
            (regression tasks only; e.g. ``nn.PoissonNLLLoss(log_input=True)``
            for count responses composed on the log-link scale).
        **trainer_kwargs : object
            Additional kwargs for PyTorch Lightning's Trainer. These do not
            apply to architectures using native training.

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
        seed_neural_runtime(random_state)
        if pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be non-negative.")
        if pretrain_max_steps != -1 and pretrain_max_steps < 1:
            raise ValueError("pretrain_max_steps must be -1 or at least 1.")
        X = prepare_fit_features(self, X)
        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if X_val is not None:
            X_val = prepare_predict_features(self, X_val)

        raw_y = np.asarray(y)
        self._fit_loss_fct = loss_fct
        y, y_val, plan = self._build_training_plan(y, y_val)
        if class_weight is not None:
            if plan.datamodule_regression:
                raise ValueError("class_weight is only supported for classifiers.")
            class_weights = compute_sample_weight(class_weight, raw_y.reshape(-1))
            if sample_weight is None:
                sample_weight = class_weights
            else:
                sample_weight = np.asarray(sample_weight).reshape(-1) * class_weights
        if sampling_strategy is not None and plan.datamodule_regression:
            raise ValueError("sampling_strategy is only supported for classifiers.")
        lr = self.config.lr if lr is None else lr
        lr_patience = self.config.lr_patience if lr_patience is None else lr_patience
        lr_factor = self.config.lr_factor if lr_factor is None else lr_factor
        lr_schedule = (
            getattr(self.config, "lr_schedule", "plateau")
            if lr_schedule is None
            else lr_schedule
        )
        weight_decay = (
            self.config.weight_decay if weight_decay is None else weight_decay
        )
        optimizer = (
            getattr(self.config, "optimizer", "adam")
            if optimizer is None
            else optimizer
        )

        warm_start_state = None
        if warm_start:
            if isinstance(warm_start, (str, Path)):
                payload = torch.load(warm_start, map_location="cpu", weights_only=False)
                warm_start_state = payload.get("state_dict", payload)
            elif getattr(self, "model", None) is not None:
                warm_start_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                raise ValueError(
                    "warm_start=True requires an already fitted estimator or a "
                    "checkpoint path."
                )

        run_training(
            self,
            X,
            y,
            plan,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            max_epochs=max_epochs,
            max_steps=max_steps,
            max_time=max_time,
            random_state=random_state,
            batch_size=batch_size,
            shuffle=shuffle,
            patience=patience,
            early_stopping_steps=early_stopping_steps,
            monitor=monitor,
            mode=mode,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_factor=lr_decay_factor,
            weight_decay=weight_decay,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            warm_start_state=warm_start_state,
            average_checkpoints=average_checkpoints,
            n_last_checkpoints=n_last_checkpoints,
            pretrain_epochs=pretrain_epochs,
            pretrain_max_steps=pretrain_max_steps,
            pretraining_ratio=pretraining_ratio,
            pretraining_noise=pretraining_noise,
            pretraining_feature_mask=pretraining_feature_mask,
            checkpoint_path=checkpoint_path,
            dataloader_kwargs=dataloader_kwargs,
            trainer_kwargs=trainer_kwargs,
            offset=offset,
            offset_val=offset_val,
            sample_weight=sample_weight,
            sample_weight_val=sample_weight_val,
            sampling_strategy=sampling_strategy,
        )
        return self

    def _build_training_plan(self, y, y_val):
        """Return ``(y, y_val, TrainingPlan)`` for this task. Task-specific."""
        raise NotImplementedError

    def _record_model_metadata(self) -> None:
        """Expose backend-independent complexity and fitted architecture state."""
        parameters = tuple(self.model.parameters())
        self.n_parameters_total_ = int(
            sum(parameter.numel() for parameter in parameters)
        )
        self.n_parameters_trainable_ = int(
            sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
        )
        architecture = self.model.model
        for name in getattr(architecture, "estimator_fitted_attributes", ()):
            setattr(self, name, getattr(architecture, name))

    def _predict_raw(
        self, X, *, batch_size: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Run inference on prepared features and return the forward-output dict.

        ``X`` must already have passed through the estimator's feature
        preparation. The model's train/eval state is restored afterwards.
        """
        from ..neural.contracts import is_penalty_key

        cat_tensor_dict, num_tensor_dict = self.data_module.preprocess_tensors(X)
        tensors = tuple(cat_tensor_dict.values()) + tuple(num_tensor_dict.values())
        if not tensors:
            raise ValueError("Prediction requires at least one transformed feature.")
        n_rows = int(tensors[0].shape[0])
        if batch_size is None:
            batch_size = n_rows or 1
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        device = next(self.model.parameters()).device

        was_training = self.model.training
        self.model.eval()
        collected: dict[str, list[torch.Tensor]] = {}
        constants: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for start in range(0, n_rows, batch_size):
                    stop = min(start + batch_size, n_rows)
                    cat_batch = {
                        key: tensor[start:stop].to(device)
                        for key, tensor in cat_tensor_dict.items()
                    }
                    num_batch = {
                        key: tensor[start:stop].to(device)
                        for key, tensor in num_tensor_dict.items()
                    }
                    result = self.model(num_features=num_batch, cat_features=cat_batch)
                    for key, value in result.items():
                        is_batched = (
                            key != "intercept"
                            and not is_penalty_key(key)
                            and value.ndim > 0
                            and value.shape[0] == stop - start
                        )
                        if is_batched:
                            collected.setdefault(key, []).append(value)
                        elif key not in constants:
                            constants[key] = value
        finally:
            self.model.train(was_training)

        merged = {key: torch.cat(values, dim=0) for key, values in collected.items()}
        merged.update(constants)
        return merged

    def _predict(self, X, *, batch_size: int | None = None):
        """Run inference and return the raw forward-output dict."""
        if getattr(self, "model", None) is None or (
            getattr(self, "data_module", None) is None
        ):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
        X = prepare_predict_features(self, X)
        return self._predict_raw(X, batch_size=batch_size)

    def basis_transform(self, X, *, batch_size: int | None = None):
        """Return an optional architecture's fixed design matrix."""
        if getattr(self, "model", None) is None or self.data_module is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        architecture = self.model.model
        transform = getattr(architecture, "linear_design", None)
        if transform is None:
            raise NotImplementedError(
                f"{type(architecture).__name__} does not expose a fixed basis."
            )
        X = prepare_predict_features(self, X)
        cat, num = self.data_module.preprocess_tensors(X)
        tensors = tuple(cat.values()) + tuple(num.values())
        n_rows = int(tensors[0].shape[0])
        if batch_size is None:
            batch_size = n_rows or 1
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer.")
        device = next(architecture.parameters()).device
        blocks = []
        with torch.no_grad():
            for start in range(0, n_rows, int(batch_size)):
                stop = min(start + int(batch_size), n_rows)
                blocks.append(
                    transform(
                        {
                            key: value[start:stop].to(device)
                            for key, value in num.items()
                        },
                        {
                            key: value[start:stop].to(device)
                            for key, value in cat.items()
                        },
                    ).cpu()
                )
        return torch.cat(blocks).numpy()

    def local_term_importance(
        self,
        X,
        *,
        target: int = 0,
        top_k: int = 10,
        batch_size: int | None = None,
    ):
        """Return architecture-specific local source-term expansions.

        This is currently implemented by SPAM. Other architectures fail
        explicitly rather than substituting generic additive importance for a
        model-specific polynomial expansion.
        """
        if getattr(self, "model", None) is None or self.data_module is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        architecture = self.model.model
        importance = getattr(architecture, "local_term_importance", None)
        if importance is None:
            raise NotImplementedError(
                f"{type(architecture).__name__} has no local term expansion."
            )
        X = prepare_predict_features(self, X)
        cat, num = self.data_module.preprocess_tensors(X)
        tensors = tuple(cat.values()) + tuple(num.values())
        n_rows = int(tensors[0].shape[0])
        if batch_size is None:
            batch_size = n_rows or 1
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer.")
        device = next(architecture.parameters()).device
        rows = []
        was_training = architecture.training
        architecture.eval()
        try:
            with torch.no_grad():
                for start in range(0, n_rows, int(batch_size)):
                    stop = min(start + int(batch_size), n_rows)
                    rows.extend(
                        importance(
                            {
                                key: value[start:stop].to(device)
                                for key, value in num.items()
                            },
                            {
                                key: value[start:stop].to(device)
                                for key, value in cat.items()
                            },
                            target=target,
                            top_k=top_k,
                        )
                    )
        finally:
            architecture.train(was_training)
        return rows

    def basis_metadata(self) -> dict:
        """Return fitted fixed-basis metadata when the architecture provides it."""
        if getattr(self, "model", None) is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        metadata = getattr(self.model.model, "basis_metadata", None)
        if metadata is None:
            raise NotImplementedError(
                f"{type(self.model.model).__name__} has no fixed-basis metadata."
            )
        return metadata()

    def model_complexity(self) -> dict[str, int]:
        """Return generic and optional architecture-specific complexity counts."""
        if not hasattr(self, "n_parameters_total_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        result = {
            "total_parameters": self.n_parameters_total_,
            "trainable_parameters": self.n_parameters_trainable_,
        }
        architecture_metadata = getattr(self.model.model, "complexity_metadata", None)
        if architecture_metadata is not None:
            result.update(architecture_metadata())
        return result

    def group_norms(self) -> dict[str, float]:
        """Return fitted term-group parameter norms for sparse architectures.

        Raises ``NotImplementedError`` when the architecture has no group
        penalty, and ``NotFittedError`` before fitting.
        """
        if getattr(self, "model", None) is None:
            raise NotFittedError("Fit the estimator before requesting group norms.")
        method = getattr(self.model.model, "get_group_norms", None)
        if method is None:
            raise NotImplementedError("This architecture does not expose group norms.")
        return dict(method())

    def training_history(self) -> dict:
        """Return architecture-native fit history when one is available."""
        if getattr(self, "model", None) is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        history = getattr(self.model.model, "training_history_", None)
        if history is None:
            raise NotImplementedError(
                f"{type(self.model.model).__name__} has no native training history."
            )
        return dict(history)

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
                intercept = float(array.item()) if array.size == 1 else array
            else:
                terms[key] = array
        return terms, intercept

    def _maybe_center_components(
        self,
        prediction,
        *,
        center,
        reference_X=None,
        reference_weight=None,
    ):

        from ..contracts import AdditiveTermMetadata
        from ..explanations import _term_features, center_additive_prediction

        feature_names = tuple(str(name) for name in self.feature_names_in_)
        metadata = {}
        for term in prediction.terms:
            features = _term_features(term, feature_names)
            kind: Literal["main", "interaction", "global", "contextual"] = (
                "interaction" if len(features) > 1 else "main"
            )
            if term in getattr(self.model.model, "global_term_names", ()):
                features, kind = feature_names, "global"
            elif any(
                name == term or term.startswith(f"{name}[")
                for name in getattr(self.model.model, "contextual_feature_names", ())
            ):
                features = tuple(self.model.model.contextual_feature_names)
                kind = "contextual"
            metadata[term] = AdditiveTermMetadata(features=features, kind=kind)
        prediction = replace(prediction, term_metadata=metadata)
        if not center and reference_X is None:
            return prediction

        reference = None
        if reference_X is not None:
            reference = self.predict_components(reference_X)
        return center_additive_prediction(
            prediction,
            reference=reference,
            sample_weight=reference_weight,
        )

    def center_components(self, X, *, reference_X=None, reference_weight=None):
        """Return contributions centered on ``X`` or a reference dataset."""
        return self.predict_components(
            X,
            center=True,
            reference_X=reference_X,
            reference_weight=reference_weight,
        )

    def _plot_series_labels(self, n_series: int):
        """Labels for the per-output plot lines; None means a single line."""
        return None

    def plot_interactions(
        self, X, *, num_bins=30, slice_bins=3, max_slices=9, pages=0, figsize=None
    ):
        """Return interaction heatmaps and conditioned higher-order slices."""
        from ..plotting import prepared_from_interactions, render_term_plots

        frame = prepare_predict_features(self, X)
        components = self.predict_components(frame)
        width = np.asarray(components.link).reshape(len(frame), -1).shape[1]
        prepared = prepared_from_interactions(
            frame,
            components.terms,
            term_features={
                name: metadata.features
                for name, metadata in components.term_metadata.items()
            },
            series_labels=self._plot_series_labels(width),
            num_bins=num_bins,
            slice_bins=slice_bins,
            max_slices=max_slices,
        )
        return render_term_plots(prepared, pages=pages, figsize=figsize)

    def plot_terms(
        self,
        X,
        *,
        center=False,
        reference_X=None,
        reference_weight=None,
        rug=None,
        density=True,
        density_bins=30,
        density_color="#D73027",
        pages=0,
        figsize=None,
    ):
        """Render 1-d term-contribution curves via the shared renderer.

        Builds the same prepared-data schema the GAM plotting pipeline uses,
        so GAM and neural term plots share one renderer. The default reddish
        shading shows the empirical feature density behind each effect. Set
        ``density=False`` to remove it or ``rug=True`` to add observation
        marks. Only single-column numeric main effects are drawn; interaction
        terms use :meth:`plot_interactions`.
        """
        from ..plotting import prepared_from_contributions, render_term_plots

        X = prepare_predict_features(self, X)
        components = self.predict_components(
            X,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
        )
        prepared = prepared_from_contributions(X, components.terms)
        return render_term_plots(
            prepared,
            rug=rug,
            density=density,
            density_bins=density_bins,
            density_color=density_color,
            pages=pages,
            figsize=figsize,
        )

    def explain_terms(
        self,
        X,
        *,
        max_bins: int = 64,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ):
        """Return a binned, backend-neutral additive-term explanation table."""
        from ..explanations import explain_additive_prediction

        return explain_additive_prediction(
            X,
            self.predict_components(
                X,
                center=center,
                reference_X=reference_X,
                reference_weight=reference_weight,
            ),
            max_bins=max_bins,
        )

    def term_importance(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ):
        """Return mean absolute link-scale contribution by additive term."""
        from ..explanations import term_importance_table

        return term_importance_table(
            self.predict_components(
                X,
                center=center,
                reference_X=reference_X,
                reference_weight=reference_weight,
            )
        )

    def interaction_importance(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
    ):
        """Return the interaction-only subset of :meth:`term_importance`."""
        table = self.term_importance(
            X,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
        )
        return table.loc[table["term_type"] == "interaction"].reset_index(drop=True)

    def save_model(self, path):
        """Save this estimator to a trusted NAMpy pickle artifact."""
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        """Load a trusted NAMpy artifact of this estimator class."""
        return load_estimator(path, cls)

    def _initialize_estimator_parameters(self, config_class, kwargs):
        self._config_class = config_class
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
        if getattr(type(self), "architecture_name", None) is None:
            params.update(model=self.base_model, config=self._config_class)
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
        if getattr(type(self), "architecture_name", None) is None:
            model = parameters.pop("model", self.base_model)
            config = parameters.pop("config", self._config_class)
            if config is not self._config_class:
                retained = {
                    name: value
                    for name, value in self.config_kwargs.items()
                    if name in getattr(config, "__dataclass_fields__", {})
                }
                retained.update(
                    {
                        f"preprocessor__{name}": value
                        for name, value in self._preprocessor_kwargs.items()
                    }
                )
                self._initialize_estimator_parameters(config, retained)
            self.base_model = model
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

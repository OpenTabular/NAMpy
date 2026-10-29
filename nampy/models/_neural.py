from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from pretab.preprocessor import Preprocessor
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..neural.configuration import (
    FixedDesignTrainingConfig,
    GradientTrainingConfig,
    NativeTrainingConfig,
    ObjectiveConfig,
    PreprocessingConfig,
)
from ..persistence import load_estimator, save_estimator
from ._data import prepare_predict_features


class NeuralEstimatorBase(BaseEstimator):
    """Base class for shared NAMpy sklearn estimator behavior."""

    def capabilities(self):
        """Return immutable operation declarations refined by fitted state."""
        from ..neural.capabilities import estimator_capabilities

        return estimator_capabilities(self)

    def fit(
        self,
        X,
        y,
        *,
        X_val=None,
        y_val=None,
        offset=None,
        offset_val=None,
        sample_weight=None,
        sample_weight_val=None,
    ):
        """Fit transactionally using constructor-owned training configuration."""
        from ..neural.training.session import fit_estimator

        warm_start_state = None
        if (
            isinstance(self.training, GradientTrainingConfig)
            and self.training.warm_start
        ):
            warm = self.training.warm_start
            if isinstance(warm, (str, Path)):
                payload = torch.load(warm, map_location="cpu", weights_only=False)
                warm_start_state = payload.get("state_dict", payload)
            elif self.model is not None:
                warm_start_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
        staged = type(self)(**self.get_params(deep=False))
        fit_estimator(
            staged,
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            offset=offset,
            offset_val=offset_val,
            sample_weight=sample_weight,
            sample_weight_val=sample_weight_val,
            warm_start_state=warm_start_state,
        )
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        return self

    def _prepare_fit(self, y):
        pass

    def _finish_fit(self):
        pass

    def _build_training_plan(self, y, y_val):
        """Return ``(y, y_val, TrainingPlan)`` for this task. Task-specific."""
        raise NotImplementedError

    def _record_model_metadata(self) -> None:
        """Expose backend-independent complexity and fitted architecture state."""
        self.predictor_names_ = tuple(
            f"predictor[{i}]" for i in range(self.model.objective.output_dim)
        )
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

    def _require_neural_fit(self):
        if self.model is None or getattr(self, "fitted_preprocessing_", None) is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call fit first."
            )

    def _predict(self, X, *, batch_size=None, offset=None):
        """Validate the fitted schema and return raw architecture predictions."""
        from ..neural.prediction import predict_raw

        self._require_neural_fit()
        X = prepare_predict_features(self, X)
        from ..neural.offsets import add_predictor_offset

        result = predict_raw(
            self.model, self.fitted_preprocessing_, X, batch_size=batch_size
        )
        result["output"] = add_predictor_offset(result["output"], offset)
        return result

    def predict_link(self, X, *, batch_size=None, offset=None):
        """Return additive raw predictors with explicit row and predictor axes."""
        return self._predict(X, batch_size=batch_size, offset=offset)["output"].cpu().numpy()

    def basis_transform(self, X, *, batch_size: int | None = None):
        """Return an optional architecture's fixed design matrix."""
        from ..neural.prediction import basis_transform

        self._require_neural_fit()
        X = prepare_predict_features(self, X)
        return basis_transform(
            self.model.model, self.fitted_preprocessing_, X, batch_size=batch_size
        )

    def fisher_information(
        self, X, *, offset=None, context=None, config=None, batch_size=None
    ):
        """Mean likelihood Fisher information on the supplied reference inputs.

        Known offsets must be supplied again for new rows when fitting used
        offsets. Diagnostics evaluate the deterministic fitted model and retain
        neither reference observations nor training execution state.
        """
        from ..diagnostics.fisher import FisherDiagnostics, normalize_fisher_diagnostics
        from ..neural.fisher import estimate_fisher_information
        from ..neural.offsets import predictor_offset

        self._require_neural_fit()
        X = prepare_predict_features(self, X)
        if offset is None and getattr(self, "fisher_offset_required_", False):
            raise ValueError(
                "Fisher diagnostics require explicit offsets because fitting used offsets."
            )
        validated_offset = predictor_offset(
            offset, len(X), self.model.objective.output_dim
        )
        if config is None:
            config = (
                normalize_fisher_diagnostics(self.fisher_diagnostics)
                or FisherDiagnostics()
            )
        elif not isinstance(config, FisherDiagnostics):
            raise TypeError("config must be FisherDiagnostics or None.")
        return estimate_fisher_information(
            self.model,
            self.fitted_preprocessing_,
            X,
            config,
            offset=validated_offset,
            context=context,
            batch_size=batch_size,
            population_size=(
                len(X)
                if config.input_sampling == "without_replacement"
                and config.population_size is None
                else config.population_size
            ),
            include_frozen=getattr(self, "native_training_info_", None) is not None,
        )

    def local_term_importance(self, X, *, target=0, top_k=10, batch_size=None):
        """Return an architecture-owned local source-term expansion.

        SPAM supports polynomial expansions. Architectures without a local
        expansion reject this operation explicitly.
        """
        from ..neural.prediction import local_term_importance

        self._require_neural_fit()
        X = prepare_predict_features(self, X)
        return local_term_importance(
            self.model.model,
            self.fitted_preprocessing_,
            X,
            target=target,
            top_k=top_k,
            batch_size=batch_size,
        )

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
        return cast(dict, metadata())

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

    def _maybe_center_components(
        self,
        prediction,
        *,
        center,
        reference_X=None,
        reference_weight=None,
        reference_offset=None,
    ):

        from ..contracts import AdditiveTermMetadata
        from ..diagnostics.contributions import (
            _term_features,
            center_additive_prediction,
        )

        feature_names = tuple(str(name) for name in self.feature_names_in_)
        metadata = {}
        declared = getattr(self.model.model, "component_metadata", {})
        for term in prediction.terms:
            if term in declared:
                metadata[term] = declared[term]
                continue
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
        if reference_offset is not None and reference_X is None:
            raise ValueError("reference_offset requires reference_X.")
        if not center and reference_X is None:
            return prediction

        reference = None
        if reference_X is not None:
            reference = self.predict_components(reference_X, offset=reference_offset)
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
        from ..diagnostics.contributions import explain_additive_prediction

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
        from ..diagnostics.contributions import term_importance_table

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

    def _artifact_state(self):
        from ..neural.fitted import neural_artifact_state

        return neural_artifact_state(self)

    def save_model(self, path):
        """Save this estimator to a trusted NAMpy pickle artifact."""
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        """Load a trusted NAMpy artifact of this estimator class."""
        return load_estimator(path, cls)

    def _initialize_estimator_parameters(
        self,
        config_class,
        *,
        architecture_config=None,
        objective=None,
        training=None,
        preprocessing=None,
        fisher_diagnostics=False,
    ):
        from ..diagnostics.fisher import normalize_fisher_diagnostics

        normalize_fisher_diagnostics(fisher_diagnostics)
        self.fisher_diagnostics = fisher_diagnostics
        self._config_class = config_class
        self.architecture_config = (
            config_class() if architecture_config is None else architecture_config
        )
        if not isinstance(self.architecture_config, config_class):
            raise TypeError(f"architecture_config must be {config_class.__name__}.")
        self.objective = ObjectiveConfig() if objective is None else objective
        if not isinstance(self.objective, ObjectiveConfig):
            raise TypeError("objective must be ObjectiveConfig.")
        if getattr(self, "_estimator_type", None) in {"regressor", "classifier"} and (
            self.objective.family is not None
            or self.objective.distributional_kwargs is not None
        ):
            raise ValueError("Distribution family settings require an LSS estimator.")
        self.training = training
        if training is not None and not isinstance(
            training,
            (GradientTrainingConfig, FixedDesignTrainingConfig, NativeTrainingConfig),
        ):
            raise TypeError("training must select a supported training configuration.")
        self.preprocessing = (
            PreprocessingConfig() if preprocessing is None else preprocessing
        )
        if not isinstance(self.preprocessing, PreprocessingConfig):
            raise TypeError("preprocessing must be PreprocessingConfig.")
        from copy import deepcopy

        defaults: dict[str, Any] = {}
        architecture_name = getattr(type(self), "architecture_name", None)
        if architecture_name is not None:
            from ..neural.registry import get_architecture

            defaults.update(get_architecture(architecture_name).preprocessor_defaults)
        if getattr(self, "_estimator_type", None) == "classifier":
            defaults["task"] = "classification"
        if getattr(self, "_estimator_type", None) == "survival":
            defaults.update(
                numerical_preprocessing="none", categorical_preprocessing="one-hot"
            )
        self.preprocessor = Preprocessor(
            **(defaults | deepcopy(self.preprocessing.parameters))
        )

    def _resolved_training(self, y):
        from ..neural.training.routes import resolve_training

        return resolve_training(
            self.base_model,
            self._config_class,
            self.objective,
            self.training,
            self._estimator_type
            if hasattr(self, "_estimator_type")
            else "distributional",
            y,
        )

    def get_params(self, deep=True):
        from dataclasses import fields, is_dataclass

        params = {
            name: getattr(self, name)
            for name in (
                "architecture_config",
                "objective",
                "training",
                "preprocessing",
                "fisher_diagnostics",
            )
        }
        if getattr(type(self), "architecture_name", None) is None:
            params.update(model=self.base_model, config_class=self._config_class)
        if deep:
            for group in (
                "architecture_config",
                "objective",
                "training",
                "preprocessing",
                "fisher_diagnostics",
            ):
                value = params[group]
                entries: Iterable[tuple[str, Any]]
                if isinstance(value, PreprocessingConfig):
                    entries = value.parameters.items()
                elif is_dataclass(value):
                    entries = ((f.name, getattr(value, f.name)) for f in fields(value))
                else:
                    entries = ()
                params.update({f"{group}__{name}": item for name, item in entries})
        return params

    def set_params(self, **parameters):

        if not parameters:
            return self
        params = self.get_params(deep=False)
        nested: dict[str, dict[str, Any]] = {}
        for name, value in parameters.items():
            group, separator, field = name.partition("__")
            if group not in params:
                raise ValueError(f"Unknown estimator parameter {group!r}.")
            if separator:
                nested.setdefault(group, {})[field] = value
            else:
                params[group] = value
        for group, updates in nested.items():
            record = params[group]
            if isinstance(record, PreprocessingConfig):
                params[group] = PreprocessingConfig(**(record.parameters | updates))
            elif group == "fisher_diagnostics" and isinstance(record, bool):
                from ..diagnostics.fisher import FisherDiagnostics

                params[group] = FisherDiagnostics(**updates)
            elif record is None and group == "training":
                from dataclasses import fields

                from ..neural.training.routes import resolve_training

                record = resolve_training(
                    params.get("model", self.base_model),
                    params.get("config_class", self._config_class),
                    params["objective"] or ObjectiveConfig(),
                    None,
                    getattr(self, "_estimator_type", "distributional"),
                    None,
                )
                if not set(updates) <= {field.name for field in fields(record)}:
                    record = GradientTrainingConfig()
                params[group] = replace(record, **updates)
            else:
                params[group] = replace(record, **updates)
        replacement = type(self)(**params)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

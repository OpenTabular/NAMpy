"""Neural survival task orchestration using existing registered architectures."""

import numpy as np

from ..contracts import AdditivePrediction
from ..neural.configuration import GradientTrainingConfig, SurvivalObjectiveConfig
from ..neural.prediction import split_output_components
from ..neural.survival import (
    CensoredParametricObjective,
    CoxObjective,
    LongitudinalPiecewiseObjective,
    PiecewiseExponentialObjective,
)
from ..neural.training import TrainingPlan
from ..survival import PiecewiseExponentialDesign, SurvivalPrediction
from ..survival.contracts import validate_weights
from ..survival.data import validate_covariates
from ..survival.likelihood import (
    observed_log_likelihood,
    parameter_offsets,
    parametric_distribution,
)
from ..survival.longitudinal import LongitudinalSurvivalTarget, subject_weight_sum
from ._neural import NeuralEstimatorBase


class NeuralSurvival(NeuralEstimatorBase):
    """Gradient survival task with interval hazards or latent-time parameters.

    ``likelihood="cox"`` uses one unstratified Breslow baseline, right censoring
    without delayed entry, unit case weights and complete-cohort optimization.
    Ordinary batch_size controls do not split Cox risk sets.
    """

    _estimator_type = "survival"

    def __init__(
        self,
        model,
        config_class,
        *,
        architecture_config=None,
        objective=None,
        training=None,
        preprocessing=None,
        fisher_diagnostics=False,
    ):
        if not isinstance(objective, SurvivalObjectiveConfig):
            raise TypeError("Neural survival requires SurvivalObjectiveConfig.")
        self._initialize_estimator_parameters(
            config_class,
            architecture_config=architecture_config,
            objective=objective,
            training=training,
            preprocessing=preprocessing,
            fisher_diagnostics=fisher_diagnostics,
        )
        self.base_model = model
        self.model = None
        self.data_module = None

    def _resolved_training(self, y):
        if self.training is not None and not isinstance(
            self.training, GradientTrainingConfig
        ):
            raise NotImplementedError(
                "Survival requires gradient training; native/fixed routes are unsupported."
            )
        training = super()._resolved_training(y)
        if {
            "sampler",
            "batch_sampler",
            "drop_last",
        } & training.dataloader_kwargs.keys():
            raise NotImplementedError(
                "Survival custom samplers and drop_last are unsupported."
            )
        if training.pretrain_epochs or (
            training.warm_start and not isinstance(training.warm_start, bool)
        ):
            raise NotImplementedError(
                "Survival pretraining and external warm-start checkpoints are unsupported."
            )
        options = training.trainer_kwargs
        if self.objective.likelihood == "cox" and any(
            options.get(name, default) != default
            for name, default in (("limit_train_batches", 1.0), ("limit_val_batches", 1.0), ("overfit_batches", 0))
        ):
            raise NotImplementedError("Cox training requires complete training and validation cohorts.")
        if options.get("accumulate_grad_batches", 1) != 1:
            raise NotImplementedError(
                "Survival gradient accumulation is not yet validated."
            )
        if (
            options.get("devices", 1) != 1
            or options.get("num_nodes", 1) != 1
            or "strategy" in options
        ):
            raise NotImplementedError(
                "Distributed survival training is not yet supported."
            )
        if self.preprocessor.numerical_preprocessing not in {"none", "standardization"}:
            raise NotImplementedError(
                "Survival supports unsupervised none/standardization preprocessing only."
            )
        if (
            self.preprocessor.categorical_preprocessing not in {"one-hot", "int"}
            or self.preprocessor.feature_preprocessing
        ):
            raise NotImplementedError(
                "Survival supports unsupervised one-hot/int categorical preprocessing without per-feature overrides."
            )
        return training

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
        longitudinal = isinstance(y, LongitudinalSurvivalTarget)
        if longitudinal:
            if self.objective.likelihood != "piecewise_exponential":
                raise NotImplementedError("Longitudinal fitting supports piecewise hazards only.")
            if not isinstance(y_val, LongitudinalSurvivalTarget) or X_val is None:
                raise NotImplementedError("Longitudinal neural fitting requires explicit disjoint whole-subject validation trajectories.")
            if isinstance(self.training, GradientTrainingConfig) and self.training.warm_start:
                raise NotImplementedError("Longitudinal warm starts are unsupported.")
        elif isinstance(y_val, LongitudinalSurvivalTarget):
            raise ValueError("Training and validation must use matching survival observation contracts.")
        weights = validate_weights(y, sample_weight, require_event=True, allow_longitudinal=True)
        if len(X) != len(y):
            raise ValueError("X and SurvivalTarget must have the same subject count.")
        self._offset(offset, len(y), likelihood=self.objective.likelihood)
        for features in (X, X_val):
            if features is None:
                continue
            validate_covariates(features)
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if y_val is not None:
            validate_weights(y_val, sample_weight_val, allow_longitudinal=True)
            if len(X_val) != len(y_val):
                raise ValueError("Validation features and target must align.")
            self._offset(offset_val, len(y_val), likelihood=self.objective.likelihood)
            # Auto-generated positional IDs are local to each target. Explicit
            # identifiers are checked by the user-facing split when supplied.
            if y.has_explicit_ids and y_val.has_explicit_ids:
                if np.isin(y.subject_id, y_val.subject_id).any():
                    raise ValueError("Training and validation subject IDs overlap.")
        if (
            self.model is not None
            and isinstance(self.training, GradientTrainingConfig)
            and self.training.warm_start
        ):
            if self.likelihood_ != self.objective.likelihood:
                raise ValueError(
                    "Warm starts require an unchanged survival likelihood."
                )
            if (
                self.likelihood_ == "piecewise_exponential"
                and tuple(self.cuts_) != self.objective.cuts
            ):
                raise ValueError("Warm starts require an unchanged cut grid.")
        fitted = super().fit(
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            offset=offset,
            offset_val=offset_val,
            sample_weight=weights,
            sample_weight_val=sample_weight_val,
        )
        fitted.offset_required_ = offset is not None
        return fitted

    def _offset(self, offset, n, *, likelihood=None):
        likelihood = (getattr(self, "likelihood_", self.objective.likelihood)
                      if likelihood is None else likelihood)
        if likelihood not in {"piecewise_exponential", "cox"}:
            return parameter_offsets(offset, n)
        if offset is None:
            return np.zeros((n, 1))
        values = np.asarray(offset, dtype=float)
        if values.shape not in {(n,), (n, 1)} or not np.isfinite(values).all():
            raise ValueError(
                "offset must be a finite scalar log-rate offset per subject."
            )
        return values.reshape(n, 1)

    def _build_training_plan(self, y, y_val):
        self.likelihood_ = self.objective.likelihood
        if self.likelihood_ == "cox":
            self.n_outputs_ = 1
            self.n_subjects_ = len(y)
            self.parameter_names_ = ("log_relative_hazard",)
            objective = CoxObjective()

            def pack_cohort(target):
                packed = np.column_stack([target.time, target.event, target.entry])
                objective.validate_training_targets(packed, np.ones(len(target)))
                return packed

            return pack_cohort(y), None if y_val is None else pack_cohort(y_val), TrainingPlan(objective)
        if self.likelihood_ in {"weibull", "lognormal"}:
            self.n_outputs_ = 2
            self.n_subjects_ = len(y)
            self.parameter_names_ = (
                ("scale", "shape")
                if self.likelihood_ == "weibull"
                else ("loc", "scale")
            )

            def pack(target):
                return np.column_stack([target.time, target.event, target.entry])

            return (
                pack(y),
                None if y_val is None else pack(y_val),
                TrainingPlan(CensoredParametricObjective(self.likelihood_)),
            )
        design = PiecewiseExponentialDesign(y, self.objective.cuts)
        self.cuts_ = design.cuts
        self.n_outputs_ = len(self.cuts_) - 1
        self.n_subjects_ = y.n_subjects if isinstance(y, LongitudinalSurvivalTarget) else len(y)
        packed = np.concatenate([design.exposure, design.events], axis=1)
        val = None
        if y_val is not None:
            validation = PiecewiseExponentialDesign(y_val, self.cuts_)
            val = np.concatenate([validation.exposure, validation.events], axis=1)
        if isinstance(y, LongitudinalSurvivalTarget):
            packed = np.column_stack([packed, y.normalization])
            val = np.column_stack([val, y_val.normalization])
            return packed, val, TrainingPlan(LongitudinalPiecewiseObjective(self.n_outputs_))
        return packed, val, TrainingPlan(PiecewiseExponentialObjective(self.n_outputs_))

    def _finish_fit(self):
        self._record_model_metadata()
        if self.likelihood_ == "cox":
            from ..neural.prediction import predict_raw
            from ..survival.cox import baseline_from_predictors

            data = self.data_module
            eta = predict_raw(self.model, data, data.X_train)["output"].numpy()[:, 0].astype(float)
            eta += self._offset(data.offset_train, len(eta))[:, 0]
            # Center the unidentifiable Cox intercept before exponentiation;
            # store the same reference for every subsequent curve prediction.
            self.baseline_reference_ = float(eta.max())
            time, event, _ = np.asarray(data.y_train).T
            self.baseline_times_, self.baseline_cumulative_hazard_ = baseline_from_predictors(
                eta - self.baseline_reference_, time, event.astype(bool), np.zeros(len(eta), dtype=int)
            )[0]


    def distribution_fit_status(self):
        """Training completion is distinct from statistical optimizer convergence."""
        from ..contracts import DistributionFitStatus

        self._require_neural_fit()
        return DistributionFitStatus("neural", "completed",
            "Gradient survival training completed; this is not a convergence guarantee.")

    def clone_for_survival_bootstrap(self, y, *, training_randomness, seed,
                                     checkpoint_path, has_validation):
        """Clone grouped configuration; isolate checkpoints and validation rows."""
        from dataclasses import replace

        from sklearn.base import clone

        self._require_neural_fit()
        if (self.objective.likelihood != self.likelihood_
                or (self.likelihood_ == "piecewise_exponential" and tuple(self.objective.cuts) != tuple(self.cuts_))):
            raise ValueError("Bootstrap objective/cuts must match the published fit; refit configuration changes first.")
        if not has_validation:
            raise NotImplementedError(
                "Neural subject bootstrap requires a disjoint external validation cohort; "
                "automatic splitting of duplicate bootstrap subjects is unsupported.")
        # Resolve constructor configuration without accessing the staged
        # preprocessor, which is deliberately absent from published fit state.
        training = super()._resolved_training(y)
        if not isinstance(training, GradientTrainingConfig):
            raise NotImplementedError("Survival bootstrap requires gradient training.")
        if training.warm_start:
            raise NotImplementedError("Bootstrap refits require fresh training; warm starts are unsupported.")
        if training != self.training_config_:
            raise ValueError("Bootstrap training configuration must match the published fit; refit changes first.")
        options = dict(training.trainer_kwargs)
        if (any(name in options for name in ("callbacks", "plugins", "profiler"))
                or not isinstance(options.get("logger", False), bool)):
            raise NotImplementedError("Bootstrap refits do not reuse caller-owned trainer execution objects.")
        options["default_root_dir"] = str(checkpoint_path)
        effective_seed = training.random_state if training_randomness == "fixed" else seed
        if (isinstance(effective_seed, (bool, np.bool_))
                or not isinstance(effective_seed, (int, np.integer))
                or not 0 <= effective_seed < 2**32):
            raise ValueError("Bootstrap training needs an explicit integer seed in [0,2**32).")
        training = replace(training, random_state=effective_seed,
                           checkpoint_path=str(checkpoint_path), trainer_kwargs=options)
        estimator = clone(self).set_params(training=training)
        return estimator, ("fixed_seed" if training_randomness == "fixed" else "independent_seed"), effective_seed

    def predict_link(self, X, *, offset=None, batch_size=None):
        """Interval log hazards or declared latent-parameter predictors."""
        self._require_neural_fit()
        validate_covariates(X)
        if getattr(self, "offset_required_", False) and offset is None:
            raise ValueError(
                "Prediction requires explicit new-subject log-rate offsets."
            )
        eta = self._predict(X, batch_size=batch_size)["output"].cpu().numpy()
        return eta + self._offset(offset, len(eta))

    predictor_backend = "neural"

    def survival_time_grid(self):
        """Fitted piecewise grid for shared trajectory prediction."""
        self._require_neural_fit()
        if self.likelihood_ != "piecewise_exponential":
            raise NotImplementedError("Time-varying parametric survival paths are unsupported.")
        return self.cuts_.copy()

    def predict_distribution(self, X, *, offset=None, batch_size=None):
        return self.survival_from_predictors(
            self.predict_link(X, offset=offset, batch_size=batch_size))

    def survival_from_predictors(self, predictors):
        """Evaluate fitted objective transforms, without training or preprocessing."""
        self._require_neural_fit()
        eta = np.asarray(predictors)
        if eta.ndim != 2 or eta.shape[1] != self.n_outputs_ or not np.isfinite(eta).all():
            raise ValueError(f"predictors must be finite (n_subjects,{self.n_outputs_}).")
        if self.likelihood_ == "cox":
            with np.errstate(over="raise", invalid="raise"):
                values = np.exp(eta - self.baseline_reference_) * self.baseline_cumulative_hazard_[None, :]
            return SurvivalPrediction(cuts=self.baseline_times_, cumulative_hazard_steps=values, backend="neural")
        if self.likelihood_ in {"weibull", "lognormal"}:
            import torch

            assert self.model is not None
            parameters = (
                self.model.objective.transform(torch.as_tensor(eta)).cpu().numpy()
            )
            return SurvivalPrediction(
                distribution=parametric_distribution(self.likelihood_, parameters),
                backend="neural",
            )
        return SurvivalPrediction(np.exp(eta), self.cuts_, backend="neural")

    def sample_survival_predictions(self, X, *, offset=None, n_draws=1000,
                                    covariance="bayes", seed=None, standard_draws=None):
        """Single gradient fits have no validated coefficient covariance route."""
        self._require_neural_fit()
        raise NotImplementedError(
            "Neural survival coefficient intervals are unsupported; use replicated-fit "
            "SurvivalPrediction snapshots for explicitly labeled between-fit variability.")

    def sample_survival_effect_predictions(self, X, **kwargs):
        """Single neural fits have no joint coefficient effect-inference route."""
        self._require_neural_fit()
        raise NotImplementedError("Neural survival coefficient effect-contrast intervals are unsupported.")

    def predict(self, X, *, times, offset=None, conditional_on=None, batch_size=None):
        return self.predict_survival(
            X,
            times=times,
            offset=offset,
            conditional_on=conditional_on,
            batch_size=batch_size,
        )

    def predict_survival(
        self, X, *, times, offset=None, conditional_on=None, batch_size=None
    ):
        return self.predict_distribution(
            X, offset=offset, batch_size=batch_size
        ).survival(times, conditional_on=conditional_on)

    def predict_cumulative_hazard(
        self, X, *, times, offset=None, conditional_on=None, batch_size=None
    ):
        return self.predict_distribution(
            X, offset=offset, batch_size=batch_size
        ).cumulative_hazard(times, conditional_on=conditional_on)

    def predict_hazard(self, X, *, times, offset=None, batch_size=None):
        return self.predict_distribution(
            X, offset=offset, batch_size=batch_size
        ).hazard(times)

    def predict_components(
        self,
        X,
        *,
        offset=None,
        center=False,
        reference_X=None,
        reference_weight=None,
        batch_size=None,
    ):
        from ..diagnostics.contributions import center_additive_prediction

        self._require_neural_fit()
        validate_covariates(X)
        if getattr(self, "offset_required_", False) and offset is None:
            raise ValueError(
                "Prediction requires explicit new-subject log-rate offsets."
            )
        output = self._predict(X, batch_size=batch_size)
        eta = output["output"].cpu().numpy()
        known = self._offset(offset, len(eta))
        terms, intercept = split_output_components(output)
        components = AdditivePrediction(
            link=eta + known,
            terms=terms,
            intercept=intercept,
            offset=known,
            predictor_names=(
                self.parameter_names_
                if self.likelihood_ != "piecewise_exponential"
                else tuple(f"log_hazard[{j}]" for j in range(self.n_outputs_))
            ),
            backend="neural",
        )
        components = self._maybe_center_components(
            components,
            center=False,
        )
        if not center and reference_X is None:
            return components
        reference = (
            None
            if reference_X is None
            else self.predict_components(
                reference_X,
                offset=self._offset(None, len(reference_X)),
                batch_size=batch_size,
            )
        )
        return center_additive_prediction(
            components, reference=reference, sample_weight=reference_weight
        )

    def score(self, X, y, sample_weight=None, *, offset=None):
        self._require_neural_fit()
        weights = validate_weights(y, sample_weight, allow_longitudinal=True)
        if self.likelihood_ == "cox":
            import torch

            objective = CoxObjective()
            targets = np.column_stack([y.time, y.event, y.entry])
            objective.validate_training_targets(targets, weights)
            eta = torch.as_tensor(self.predict_link(X, offset=offset))
            return -float(objective.compute_loss(eta, torch.as_tensor(targets)))
        if self.likelihood_ in {"weibull", "lognormal"}:
            distribution = self.predict_distribution(X, offset=offset).distribution
            return float(
                weights @ observed_log_likelihood(y, distribution) / weights.sum()
            )
        design = PiecewiseExponentialDesign(y, self.cuts_)
        eta = self.predict_link(X, offset=offset)
        return float(weights @ design.log_likelihood(eta) / subject_weight_sum(y, weights))

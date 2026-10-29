# lss.py
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import torch

from ..distributions.contracts import DistributionFitStatus
from ..neural.families.registry import family_name_for_instance, resolve_family
from ..neural.objectives import DistributionObjective
from ..neural.prediction import split_output_components
from ..neural.training import TrainingPlan
from ..neural.training.common import _training_tensor
from ._neural import NeuralEstimatorBase


class NeuralLSS(NeuralEstimatorBase):
    """Shared estimator interface for neural distributional models."""

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

    def __sklearn_tags__(self):
        # Distribution-parameter output: neither a classifier nor a regressor.
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def _prepare_fit(self, y):
        self.training_completion_ = None
        self._distribution_fit_status_ = DistributionFitStatus(
            "neural", "unknown", "Training has not completed."
        )
        family = self.objective.family
        if family is None:
            architecture = getattr(type(self), "architecture_name", None)
            if architecture is not None:
                from ..neural.registry import get_architecture

                family = get_architecture(architecture).default_family
        self.family_, self.distributional_kwargs_ = resolve_family(
            family or "normal"
        ).instantiate(y, deepcopy(self.objective.distributional_kwargs))

    def _finish_fit(self):
        completion = self.training_completion_
        if completion is None:
            native = getattr(self, "native_training_info_", None) or {}
            if (
                native.get("finished") is True
                and native.get("optimizer_steps", 0) > 0
                and np.isfinite(native.get("best_val_loss", np.nan))
            ):
                self._distribution_fit_status_ = DistributionFitStatus(
                    "neural",
                    "completed",
                    "Native full-cohort training completed; this is not statistical convergence.",
                )
            return
        if completion is not None:
            state: Literal["failed", "completed"]
            if not completion["finished"] or completion["interrupted"]:
                state, message = "failed", "Training was interrupted or did not finish."
            elif getattr(self, "n_optimizer_steps_", 0) < 1:
                state, message = (
                    "failed",
                    "Training finished without an optimizer step.",
                )
            elif any(
                not np.isfinite(value)
                for key, value in self.final_training_metrics_.items()
                if key in {"train_loss", "train_loss_epoch", "val_loss"}
            ):
                state, message = "failed", "Training reported a nonfinite loss."
            else:
                state, message = (
                    "completed",
                    "Training completed; this is not a statistical convergence claim.",
                )
            self._distribution_fit_status_ = DistributionFitStatus(
                "neural", state, message
            )

    def distribution_fit_status(self):
        """Report training completion for shared candidate-family comparison.

        Returns
        -------
        nampy.distributions.DistributionFitStatus
            Completed training is distinct from statistical convergence.
            Unfitted estimators and training routes without completion metadata
            remain unknown, and are excluded from automatic ranking.
        """
        return getattr(
            self,
            "_distribution_fit_status_",
            DistributionFitStatus(
                "neural", "unknown", "No training-completion evidence is available."
            ),
        )

    def _build_training_plan(self, y, y_val):
        if self.objective.loss_fct is not None:
            raise ValueError(
                "Custom loss_fct is only supported for regression tasks; "
                "LSS losses come from the distribution family."
            )
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        encoder = getattr(self.family_, "encode_targets", None)
        if encoder is not None:
            y = encoder(y)
            y_val = None if y_val is None else encoder(y_val)
        plan = TrainingPlan(objective=DistributionObjective(self.family_))
        return y, y_val, plan

    def predict(self, X, raw=False, *, batch_size=None, offset=None):
        predictions = self._predict(X, batch_size=batch_size, offset=offset)["output"]

        if not raw:
            # Some distribution transforms depend on learned family parameters
            # (for example ordinal cutpoints), so their result can require a
            # gradient even though the backbone prediction path is in eval mode.
            assert self.model is not None
            with torch.no_grad():
                transformed = self.model.objective.transform(predictions)
            return transformed.cpu().numpy()

        # Convert predictions to NumPy array and return
        else:
            return predictions.cpu().numpy()

    def predict_distribution(self, X, *, batch_size=None, offset=None):
        """Return conditional distributions for supported scalar neural LSS.

        Parameters
        ----------
        X : DataFrame or ndarray
            Prediction rows, using the fitted feature schema.
        batch_size : int, optional
            Number of rows per inference batch, as in ``predict``.

        Returns
        -------
        nampy.distributions.DistributionPrediction
            NumPy inference for normal, gamma, studentt, beta, negativebinom,
            lognormal and weibull
            on detached natural parameters. Clamped negative-binomial regimes
            are explicitly unsupported. Other
            families raise ``NotImplementedError``; their training, prediction
            and scoring remain available. Draws exclude training uncertainty.
        """
        from ..neural.families.distribution_adapter import distribution_prediction

        parameters = self.predict(X, batch_size=batch_size, offset=offset)
        return distribution_prediction(self.family_, parameters)

    def predict_quantiles(self, X, probabilities, *, batch_size=None, offset=None):
        """Return (n,q) response quantiles without requiring a complete density.

        Quantile-loss families expose only their explicitly fitted levels;
        interpolation, extrapolation and crossing repair are unsupported. Other
        families delegate to their shared probability law's quantile operation.
        Requested levels retain their supplied order, including duplicates.
        """
        from ..neural.families.distribution_adapter import distribution_prediction
        from ..neural.objectives.quantile import Quantile

        probs = np.atleast_1d(np.asarray(probabilities, dtype=float))
        if (
            probs.ndim != 1
            or not np.isfinite(probs).all()
            or np.any((probs < 0) | (probs > 1))
        ):
            raise ValueError("probabilities must be a finite vector in [0,1].")
        parameters = self.predict(X, batch_size=batch_size, offset=offset)
        if isinstance(self.family_, Quantile):
            fitted = np.asarray(self.family_.quantiles)
            indices = []
            for probability in probs:
                match = np.flatnonzero(
                    np.isclose(
                        fitted, probability, rtol=float(8 * np.finfo(float).eps), atol=0
                    )
                )
                if len(match) != 1:
                    raise NotImplementedError(
                        "Quantile predictions require an unambiguous explicitly fitted level; interpolation is unsupported."
                    )
                indices.append(int(match[0]))
            return parameters[:, indices]
        return distribution_prediction(self.family_, parameters).quantile(probs)

    def predict_point(self, X, *, batch_size=None, offset=None):
        """Return the family-defined conditional point prediction."""
        predictions = self._predict(X, batch_size=batch_size, offset=offset)["output"]
        with torch.no_grad():
            point = self.family_.predict_point(predictions, transformed=False)
        return point.cpu().numpy()

    def score(self, X, y, sample_weight=None, *, offset=None):
        """Return the negative mean family loss for ``y``.

        Higher is better, so sklearn model-selection utilities can rank fits.
        Quantile families sum pinball loss over requested levels and average
        over rows; likelihood families use NLL.
        """
        raw_pred = self._predict(X, offset=offset)["output"]
        if isinstance(y, pd.Series):
            y = y.to_numpy(copy=True)
        if isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.to_numpy()
        encoder = getattr(self.family_, "encode_targets", None)
        if encoder is not None:
            y = encoder(y)
        with torch.no_grad():
            y_tensor = torch.as_tensor(y, device=raw_pred.device)
            weight_tensor = (
                None
                if sample_weight is None
                else torch.as_tensor(sample_weight, device=raw_pred.device)
            )
            assert self.model is not None
            nll = self.model.objective.compute_loss(
                raw_pred, y_tensor, sample_weight=weight_tensor
            )
        return -float(nll.detach().cpu().item())

    def _plot_series_labels(self, n_series: int):
        param_names = getattr(getattr(self, "family_", None), "param_names", None)
        if param_names is not None:
            return list(param_names)[:n_series]
        return [f"Param {i + 1}" for i in range(n_series)]

    def predict_components(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
        reference_offset=None,
        batch_size=None,
        offset=None,
    ):
        """Per-term contributions on the raw parameter scale.

        ``link`` holds the raw multi-column network output, and each term
        a matching multi-column contribution;
        additivity holds on the raw (link) scale.
        """
        from ..contracts import AdditivePrediction

        pred_dict = self._predict(X, batch_size=batch_size, offset=offset)
        raw = pred_dict["output"]
        terms, intercept = split_output_components(pred_dict)
        prediction = AdditivePrediction(
            link=raw.cpu().numpy(),
            predictor_names=tuple(self.family_.param_names),
            terms=terms,
            intercept=intercept,
            offset=0.0 if offset is None else offset,
            backend="neural",
        )
        return self._maybe_center_components(
            prediction,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
            reference_offset=reference_offset,
        )

    @property
    def predictor_point_summary(self):
        self._require_neural_fit()
        return self.family_.point_summary

    def summarize_predictors(self, predictors):
        """Evaluate family-defined point and conditional SD in raw coordinates."""
        if self.model is None:
            raise ValueError("Fit the estimator before summarizing predictors.")
        family = self.model.objective.family
        device = next(self.model.parameters()).device
        raw = torch.as_tensor(
            np.asarray(predictors), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            point = family.predict_point(raw, transformed=False)
            try:
                spread = family.predict_spread(raw, transformed=False)
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"{type(family).__name__} does not provide a conditional standard deviation."
                ) from error
        return point.cpu().numpy().reshape(-1), spread.cpu().numpy().reshape(-1)

    def plot_distributional_effects(
        self,
        X,
        *,
        reference_X=None,
        reference_weight=None,
        response_scale=1.0,
        feature_name=None,
        feature_names=None,
        **plot_kwargs,
    ):
        """Plot point effects and conditional spread for additive terms.

        Effects are computed through the fitted family's ``predict_point`` and
        ``predict_spread`` methods.  No parameter ordering or family formula is
        encoded in the plotting layer.  ``response_scale`` converts effects
        and standard deviations back to target units after fitting a linearly
        standardized response.

        Returns
        -------
        list of matplotlib.figure.Figure
            Figures containing the requested one-dimensional terms.
        """
        from ..plotting import plot_distributional_effects

        return plot_distributional_effects(
            self,
            X,
            reference_X=reference_X,
            reference_weight=reference_weight,
            response_scale=response_scale,
            feature_name=feature_name,
            feature_names=feature_names,
            **plot_kwargs,
        )

    def evaluate(self, X, y_true, metrics=None, distribution_family=None):
        """
        Evaluate the fitted distributional model.

        The family loss is computed from raw network outputs. Likelihood
        families report ``NLL``; quantile families report ``pinball_loss``
        summed over requested levels and averaged over rows.
        Other metrics use transformed parameters from ``family(raw_predictions)``.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Input samples.
        y_true : array-like
            True targets. Can be shape (n,), (n,1), or (n,K) for multivariate families
            (e.g. Dirichlet).
        metrics : dict, optional
            Mapping metric_name -> callable(y_true, transformed_predictions).
            If None, uses `get_default_metrics(...)`.
        distribution_family : str, optional
            Family name override. If None, inferred from ``self.family_``.

        Returns
        -------
        scores : dict
            Metric values, including ``pinball_loss`` for quantile families or
            ``NLL`` for likelihood families.
        """
        # Basic fitted check
        if self.model is None or getattr(self, "fitted_preprocessing_", None) is None:
            raise ValueError("The model or data module has not been fitted yet.")
        if getattr(self, "family_", None) is None:
            raise ValueError(
                "No distribution family found. Fit the model with a valid family first."
            )

        # Infer family name if not provided
        if distribution_family is None:
            distribution_family = family_name_for_instance(self.family_)

        distribution_family = str(distribution_family).lower()

        # Default metrics if none provided
        if metrics is None:
            metrics = self.get_default_metrics(distribution_family)

        # ------------------------------------------------------------------
        # Single forward pass for raw predictions, then transform once
        # ------------------------------------------------------------------
        pred_dict = self._predict(X)  # returns dict of torch tensors
        raw_pred = pred_dict["output"]  # torch.Tensor on model device

        # Label the objective by its mathematical meaning.
        scores = {}
        with torch.no_grad():
            target = (
                y_true.to_numpy(copy=True)
                if isinstance(y_true, (pd.Series, pd.DataFrame))
                else y_true
            )
            y_tensor = _training_tensor(target, device=raw_pred.device)
            nll = self.model.objective.compute_loss(raw_pred, y_tensor)
            loss_name = (
                "pinball_loss"
                if family_name_for_instance(self.family_) == "quantile"
                else "NLL"
            )
            scores[loss_name] = float(nll.detach().cpu().item())

            # Transformed predictions for all other metrics
            transformed_pred = self.model.objective.transform(raw_pred)

        predictions_transformed = transformed_pred.detach().cpu().numpy()

        # ------------------------------------------------------------------
        # Compute user/default metrics on transformed predictions
        # ------------------------------------------------------------------
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = float(metric_func(y_true, predictions_transformed))

        return scores

    def get_default_metrics(self, distribution_family):
        """Return metrics declared by the resolved distribution family."""
        from ..neural.families.default_metrics import default_metrics_for

        metadata = resolve_family(distribution_family)
        return default_metrics_for(metadata.metric_profile, self.family_)

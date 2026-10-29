"""Differentiable interval-hazard likelihood with subject-level reduction.

References
----------
- Cox, D. R. (1972). Regression Models and Life-Tables.
  https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
- Breslow, N. (1974). Covariance Analysis of Censored Survival Data.
  https://www.jstor.org/stable/2529620
- Laird, N., and Olivier, D. (1981). Covariance Analysis of Censored Survival Data Using
  Log-Linear Analysis Techniques.
  https://doi.org/10.1080/01621459.1981.10477634
- Katzman, J. L., et al. (2018). DeepSurv: personalized treatment recommender system
  using a Cox proportional hazards deep neural network.
  https://doi.org/10.1186/s12874-018-0482-1

DeepSurv is neural Cox background; NAMpy uses its registered architectures and the
locally declared cohort, censoring and tie restrictions.
"""

import numpy as np
import torch

from ..survival import SurvivalTarget
from ..survival.likelihood import validate_parametric_fit
from .families.continuous import LogNormalDistribution, WeibullDistribution
from .objectives import NeuralObjective


class SurvivalObjective(NeuralObjective):
    """Shared unsupervised preprocessing and subject-normalized survival loss."""

    kind = "survival"

    def __init__(self, n_intervals):
        super().__init__(n_intervals)
        self.mean_weights = {"train": 1.0, "val": 1.0}

    def preprocessing_targets(self, targets):
        return None

    def configure_data(self, data):
        for stage, y, weights in (
            ("train", data.y_train, data.sample_weight_train),
            ("val", data.y_val, data.sample_weight_val),
        ):
            weights = np.ones(len(y)) if weights is None else np.asarray(weights)
            if not len(y) or weights.sum() <= 0:
                raise ValueError(
                    "Training and validation need positive total subject weight."
                )
            self.mean_weights[stage] = float(weights.mean())
            if stage == "train":
                self.validate_training_targets(np.asarray(y), weights)

    def validate_training_targets(self, targets, weights):
        raise NotImplementedError

    def compute_stage_loss(self, predictions, targets, sample_weight=None, *, stage):
        targets = self.prepare_targets(predictions, targets)
        losses = self.loss_values(predictions, targets)
        if sample_weight is not None:
            losses = losses * sample_weight.reshape(-1).to(losses)
        return losses.mean() / self.mean_weights[stage]


class PiecewiseExponentialObjective(SurvivalObjective):
    """Log hazards; packed target columns are exposures followed by events."""

    def configure_data(self, data):
        from .offsets import predictor_offset

        # A known subject log-rate offset shifts every interval log hazard.
        # Materialize that task-specific mapping before generic batch validation;
        # unrelated multi-predictor objectives retain their exact-width contract.
        for stage in ("train", "val"):
            offset = getattr(data, f"offset_{stage}", None)
            if offset is not None:
                n_rows = len(getattr(data, f"y_{stage}"))
                value = predictor_offset(offset, n_rows, 1)
                setattr(
                    data,
                    f"offset_{stage}",
                    np.repeat(value, self.output_dim, axis=1),
                )
        super().configure_data(data)

    def validate_training_targets(self, targets, weights):
        exposure, events = np.split(targets, 2, axis=1)
        if np.any(weights @ exposure <= 0):
            raise ValueError(
                "Training subjects need positive-weight exposure in every interval."
            )
        if np.any(weights @ events <= 0):
            raise NotImplementedError(
                "Zero-event interval baseline boundary fits are unsupported."
            )

    def prepare_targets(self, predictions, targets):
        if predictions.ndim != 2 or targets.shape != (
            len(predictions),
            2 * self.output_dim,
        ):
            raise ValueError(
                "Survival tensor targets must pack aligned exposure and event columns."
            )
        if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
            raise ValueError("Survival predictions and targets must be finite.")
        return targets.to(dtype=predictions.dtype)

    def loss_values(self, predictions, targets):
        exposure, events = targets.split(self.output_dim, dim=1)
        # No evaluation of exp(eta) outside the observed at-risk intervals.
        active_eta = torch.where(
            exposure > 0, predictions, torch.zeros_like(predictions)
        )
        return (exposure * torch.exp(active_eta) - events * predictions).sum(dim=1)

    def transform(self, predictions):
        return torch.exp(predictions)


class LongitudinalPiecewiseObjective(PiecewiseExponentialObjective):
    """Record-additive likelihood divided by total subject case weight.

    The last packed column allocates 1/n_records_for_subject to each record.
    Explicit disjoint validation trajectories prevent subject leakage. Random
    record minibatches have the same full-cohort normalization, rather than
    treating the number of records as the number of independent patients.
    """

    def configure_data(self, data):
        super().configure_data(data)
        for stage, y, weights in (("train", data.y_train, data.sample_weight_train),
                                  ("val", data.y_val, data.sample_weight_val)):
            allocation = np.asarray(y)[:, -1]
            if not np.isfinite(allocation).all() or np.any((allocation <= 0) | (allocation > 1)):
                raise ValueError("Invalid subject normalization allocation.")
            weights = np.ones(len(y)) if weights is None else np.asarray(weights)
            self.mean_weights[stage] = float(np.mean(weights*allocation))

    def validate_training_targets(self, targets, weights):
        return super().validate_training_targets(targets[:, :-1], weights)

    def prepare_targets(self, predictions, targets):
        if targets.ndim != 2 or targets.shape[1] != 2*self.output_dim+1:
            raise ValueError("Longitudinal targets require exposure/events and subject allocation.")
        return super().prepare_targets(predictions, targets[:, :-1])


class CensoredParametricObjective(SurvivalObjective):
    """Two-predictor latent time law; packed target columns time/event/entry."""

    target_dtype = torch.float64

    def __init__(self, likelihood):
        super().__init__(2)
        self.likelihood = likelihood
        self.family: WeibullDistribution | LogNormalDistribution
        if likelihood == "weibull":
            self.family = WeibullDistribution(
                scale_transform=torch.exp, shape_transform=torch.exp, eps=0.0
            )
        elif likelihood == "lognormal":
            self.family = LogNormalDistribution(scale_transform=torch.exp, eps=0.0)
        else:
            raise NotImplementedError(
                f"Unsupported parametric survival likelihood: {likelihood}."
            )

    def validate_training_targets(self, targets, weights):
        validate_parametric_fit(
            SurvivalTarget(targets[:, 0], targets[:, 1], targets[:, 2]), weights
        )

    def prepare_targets(self, predictions, targets):
        if predictions.shape != (len(targets), 2) or targets.shape != (
            len(predictions),
            3,
        ):
            raise ValueError(
                "Parametric survival needs two predictors and time/event/entry targets."
            )
        if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
            raise ValueError("Survival predictions and targets must be finite.")
        return targets

    def loss_values(self, predictions, targets):
        # Preserve observed times and evaluate log-tail differences in float64.
        eta = predictions.to(dtype=targets.dtype)
        time, event, entry = targets.unbind(dim=1)
        log_t = time.log()
        positive = entry > 0
        if self.likelihood == "weibull":
            shape = eta[:, 1].exp()
            u = shape * (log_t - eta[:, 0])
            log_ratio = torch.zeros_like(time)
            near = positive & (entry > 0.5 * time)
            far = positive & ~near
            log_ratio[near] = torch.log1p((entry[near] - time[near]) / time[near])
            log_ratio[far] = entry[far].log() - log_t[far]
            fraction = torch.ones_like(time)
            fraction[positive] = -torch.expm1(shape[positive] * log_ratio[positive])
            return u.exp() * fraction - event * (eta[:, 1] - log_t + u)
        z = (log_t - eta[:, 0]) * (-eta[:, 1]).exp()
        log_density = -0.5 * np.log(2 * np.pi) - eta[:, 1] - 0.5 * z.square() - log_t
        value = torch.where(event.bool(), log_density, torch.special.log_ndtr(-z))
        za = (entry[positive].log() - eta[positive, 0]) * (-eta[positive, 1]).exp()
        normalizer = torch.zeros_like(time)
        normalizer[positive] = torch.special.log_ndtr(-za)
        return normalizer - value

    def transform(self, predictions):
        return self.family(predictions)


class CoxObjective(SurvivalObjective):
    """Unstratified full-cohort Cox loss with Breslow ties.

    Risk sets cannot be split into independent minibatches. The configured
    data module therefore emits one complete cohort for each stage, regardless
    of the ordinary training batch size. Delayed entry and case weights are
    explicitly unsupported.
    """

    target_dtype = torch.float64

    def __init__(self):
        super().__init__(1)
        self.cohort_sizes = {}

    def configure_data(self, data):
        super().configure_data(data)
        for stage in ("train", "val"):
            y = np.asarray(getattr(data, f"y_{stage}"))
            weights = getattr(data, f"sample_weight_{stage}")
            self.validate_training_targets(y, np.ones(len(y)) if weights is None else weights)
            self.cohort_sizes[stage] = len(y)
        data.batch_size = max(self.cohort_sizes.values())

    def validate_training_targets(self, targets, weights):
        target = SurvivalTarget(targets[:, 0], targets[:, 1], targets[:, 2])
        if np.any(target.entry):
            raise NotImplementedError("Neural Cox delayed entry is unsupported.")
        if np.any(np.asarray(weights) != 1):
            raise NotImplementedError("Neural Cox case weights are unsupported.")
        if not target.event.any():
            raise ValueError("Each Cox training/validation cohort needs an event.")

    def prepare_targets(self, predictions, targets):
        if predictions.shape != (len(targets), 1) or targets.shape != (len(predictions), 3):
            raise ValueError("Cox requires one predictor and time/event/entry targets.")
        if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
            raise ValueError("Cox predictors and targets must be finite.")
        return targets

    def loss_values(self, predictions, targets):
        """Per-row partial log-likelihood terms; censored rows contribute zero."""
        eta = predictions[:, 0].to(targets.dtype)
        time, event, _ = targets.unbind(dim=1)
        order = torch.argsort(time, descending=True, stable=True)
        ordered_time, ordered_eta = time[order], eta[order]
        _, counts = torch.unique_consecutive(ordered_time, return_counts=True)
        # A tied event uses the denominator after every event/censor tie entered.
        ends = counts.cumsum(0) - 1
        denominators = torch.repeat_interleave(
            torch.logcumsumexp(ordered_eta, dim=0)[ends], counts
        )
        values = torch.zeros_like(eta)
        values[order] = event[order] * (denominators - ordered_eta)
        return values

    def compute_stage_loss(self, predictions, targets, sample_weight=None, *, stage):
        if len(predictions) != self.cohort_sizes[stage]:
            raise ValueError("Cox loss requires the complete stage cohort in one batch.")
        return super().compute_stage_loss(predictions, targets, sample_weight, stage=stage)

    def transform(self, predictions):
        return predictions.exp()

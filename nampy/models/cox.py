"""Static-covariate, fixed-smoothing Cox fitting in identified GAM designs."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..contracts import AdditivePrediction, AdditiveTermMetadata, DistributionFitStatus
from ..gam.design import compile_design
from ..gam.fit.solvers.cox import fit_cox
from ..gam.losses.cox import (
    cox_baseline,
    cox_baseline_uncertainty,
    cox_partial_likelihood,
)
from ..persistence import load_estimator, save_estimator
from ..survival import SurvivalPrediction, SurvivalTarget
from ..survival.data import validate_covariates


class CoxGAM(BaseEstimator):
    """Penalized Cox partial likelihood with mgcv's tied-risk-set convention.

    Static covariates, right censoring and optional strata are supported.
    Fixed smoothing remains the default. Automatic REML uses the existing
    general-family solver (Newton/BFGS; stratified automatic fits require BFGS). No overall intercept (it cancels from partial likelihood).
    Delayed entry, case weights, neural Cox and continuous hazard density are
    unsupported. Survival SE includes baseline and coefficient uncertainty;
    subject bootstrap re-estimates both and supports RMST intervals.
    ``score`` is per-subject partial likelihood, not a full event-time density.
    """

    def capabilities(self):
        """Describe Cox operations while preserving partial-likelihood scope."""
        from ..gam.capabilities import estimator_capabilities

        return estimator_capabilities(self, task="cox")

    _estimator_type = "survival"

    def __init__(
        self,
        formula=None,
        *,
        k=10,
        basis="tp",
        smoothing_params=None,
        knots=None,
        max_iter=200,
        tol=1e-9,
        optimize_smoothing=False,
        smoothing_optimizer="outer_newton",
        control=None,
    ):
        self.formula = formula
        self.k = k
        self.basis = basis
        self.smoothing_params = smoothing_params
        self.knots = knots
        self.max_iter = max_iter
        self.tol = tol
        self.optimize_smoothing = optimize_smoothing
        self.smoothing_optimizer = smoothing_optimizer
        self.control = control

    def _target(self, target, strata, *, require_event=True):
        if not isinstance(target, SurvivalTarget):
            raise NotImplementedError(
                "Cox requires one static record per subject; longitudinal risk sets are unsupported."
            )
        if np.any(target.entry > 0):
            raise NotImplementedError("Delayed-entry Cox risk sets are unsupported.")
        labels = (
            np.zeros(len(target), dtype=int) if strata is None else np.asarray(strata)
        )
        if labels.shape != (len(target),) or pd.isna(labels).any():
            raise ValueError("strata must be aligned nonmissing labels.")
        codes, levels = pd.factorize(labels, sort=False)
        if require_event and any(
            not target.event[codes == i].any() for i in range(len(levels))
        ):
            raise NotImplementedError("Each Cox stratum needs observed events.")
        return codes, tuple(levels)

    def fit(self, X, y, *, strata=None, offset=None):
        staged = type(self)(**self.get_params(deep=False))
        staged._fit(X, y, strata=strata, offset=offset)
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        return self

    def _fit(self, X, y, *, strata=None, offset=None):
        from .._offsets import predictor_offset

        values = predictor_offset(offset, len(X), 1)
        offset = None if values is None else values[:, 0]
        validate_covariates(X)
        if len(X) != len(y):
            raise ValueError("Cox covariates and targets must align.")
        codes, levels = self._target(y, strata)
        if self.formula is not None and not self.formula.lstrip().startswith("~"):
            raise ValueError(
                "Cox formula must be a right-hand-side formula; event times belong to SurvivalTarget."
            )
        design = compile_design(
            X,
            y.time,
            formula=self.formula,
            k=self.k,
            basis=self.basis,
            fit_intercept=True,
            smoothing_params=self.smoothing_params,
            knots=self.knots,
            offset=offset,
        )
        # Compile conventional contrasts with an intercept, then remove only
        # the overall constant coordinate, as mgcv's drop.intercept contract.
        free = np.setdiff1d(np.arange(design.matrix.shape[1]), design.intercept_indices)
        matrix = design.matrix[:, free]
        penalty, known = design.penalty[np.ix_(free, free)], design.training_offset
        if matrix.shape[1] == 0:
            raise NotImplementedError(
                "Use at least one identifiable Cox predictor; intercept-only fits are unsupported."
            )
        gam = None
        if self.optimize_smoothing:
            if len(levels) > 1 and self.smoothing_optimizer != "bfgs":
                raise NotImplementedError(
                    "Stratified automatic Cox currently requires bfgs; the strict "
                    "Newton coefficient endpoint has not passed the reference convergence gate."
                )
            from ..gam import GAM
            from ..gam.control import gam_control
            from ..gam.families.cox import CoxPartialLikelihood
            from ..gam.fit.capabilities import validate_family_smoothing_optimizer
            from ..gam.fit.solvers.cox import CoxSolution

            family = CoxPartialLikelihood(y.event, codes)
            validate_family_smoothing_optimizer(family, self.smoothing_optimizer)
            gam = GAM(
                formula=self.formula,
                family=family,
                k=self.k,
                basis=self.basis,
                drop_intercept=True,
                fit_intercept=False,
                optimize_smoothing=True,
                smoothing_method="reml",
                smoothing_optimizer=self.smoothing_optimizer,
                smoothing_params=self.smoothing_params,
                knots=self.knots,
                control=gam_control(self.control, maxit=self.max_iter),
                irls_tol=self.tol,
            ).fit(X, y.time, offset=offset)
            result = gam.fit_result()
            from ..gam.design import CompiledDesign

            design = CompiledDesign.from_fitted_gam(gam, X, offset=offset)
            free = np.arange(design.matrix.shape[1])
            matrix, known = design.matrix, design.training_offset
            if result.core.converged is False or result.core.failed_step:
                raise RuntimeError(
                    f"Cox general-family coefficient fitting failed: converged={result.core.converged}, failed_step={result.core.failed_step}, reason={result.core.failure_reason}."
                )
            from ..gam.diagnostics.fit_status import distribution_fit_status

            status = distribution_fit_status(gam)
            if not status.eligible:
                raise RuntimeError(f"Cox outer smoothing failed: {status.message}")
            beta = result.coef_full
            loglik = cox_partial_likelihood(
                beta, matrix, y.time, y.event, codes, known
            )[0]
            solution = CoxSolution(beta, gam.vcov(), loglik, 0)
        else:
            solution = fit_cox(
                matrix,
                penalty,
                y.time,
                y.event,
                codes,
                known,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        baseline = cox_baseline(
            matrix, solution.coefficients, y.time, y.event, codes, known
        )
        baseline_uncertainty = cox_baseline_uncertainty(
            matrix, solution.coefficients, y.time, y.event, codes, known
        )
        # Publish only after successful likelihood, covariance and baseline work.
        self.design_ = design
        if gam is not None:
            self.gam_ = gam
        self.smoothing_params_ = (
            gam.smoothing_params.copy()
            if gam is not None
            else design.smoothing_params.copy()
        )
        self.free_indices_ = free
        self.coef_ = solution.coefficients
        self.covariance_ = solution.covariance
        self.strata_ = levels
        self.baseline_ = baseline
        self.baseline_uncertainty_ = baseline_uncertainty
        self.predictor_names_ = ("log_relative_hazard",)
        self.offset_required_ = offset is not None
        self.partial_log_likelihood_ = solution.partial_log_likelihood
        self.n_subjects_ = len(y)
        self.n_features_in_ = len(design.feature_names)
        return self

    def _evaluation(self, X, offset):
        if not hasattr(self, "coef_"):
            raise NotFittedError("Fit CoxGAM before prediction.")
        if self.offset_required_ and offset is None:
            raise ValueError("Cox prediction requires explicit new-subject offsets.")
        validate_covariates(X)
        if not len(X):
            raise ValueError("Cox prediction requires nonempty rows.")
        from .._offsets import predictor_offset

        values = predictor_offset(offset, len(X), 1)
        return self.design_.evaluate(X, offset=None if values is None else values[:, 0])

    def predict_link(self, X, *, offset=None):
        evaluation = self._evaluation(X, offset)
        return (evaluation.matrix[:, self.free_indices_] @ self.coef_ + evaluation.offset)[:, None]

    def predict_components(self, X, *, offset=None):
        evaluation = self._evaluation(X, offset)
        terms, metadata = {}, {}
        for term in self.design_.terms:
            name = term.label if term.label not in terms else term.term_id
            keep = np.isin(self.free_indices_, term.coefficient_indices)
            terms[name] = (
                evaluation.matrix[:, self.free_indices_[keep]] @ self.coef_[keep]
            )[:, None]
            metadata[name] = AdditiveTermMetadata(
                features=term.features,
                kind="interaction" if len(term.features) > 1 else "main",
            )
        return AdditivePrediction(
            link=self.predict_link(X, offset=offset),
            terms=terms,
            intercept=0.0,
            offset=evaluation.offset,
            backend="gam",
            term_metadata=metadata,
            predictor_names=("log_relative_hazard",),
        )

    def predict_distribution(self, X, *, strata=None, offset=None):
        eta = self.predict_link(X, offset=offset)[:, 0]
        if strata is None:
            if len(self.strata_) != 1:
                raise ValueError("Stratified Cox prediction requires explicit strata.")
            labels = np.repeat(self.strata_[0], len(eta))
        else:
            labels = np.asarray(strata)
        if labels.shape != eta.shape or pd.isna(labels).any():
            raise ValueError("Prediction strata must align with rows.")
        if not set(labels) <= set(self.strata_):
            raise NotImplementedError("Unseen Cox strata have no fitted baseline.")
        lookup = {label: i for i, label in enumerate(self.strata_)}
        codes = [lookup[label] for label in labels]
        # Restrict to the common follow-up support of the requested strata.
        horizon = min(self.baseline_[i][0][-1] for i in set(codes))
        cuts = np.unique(
            np.r_[
                0.0,
                horizon,
                np.concatenate(
                    [
                        self.baseline_[i][0][self.baseline_[i][0] <= horizon]
                        for i in set(codes)
                    ]
                ),
            ]
        )
        values = np.empty((len(eta), len(cuts)))
        with np.errstate(over="raise", invalid="raise"):
            for row, code in enumerate(codes):
                times, baseline = self.baseline_[code]
                values[row] = baseline[
                    np.searchsorted(times, cuts, side="right") - 1
                ] * np.exp(eta[row])
        return SurvivalPrediction(
            cuts=cuts, cumulative_hazard_steps=values, backend="gam"
        )

    def predict(self, X, *, times, strata=None, offset=None, conditional_on=None):
        return self.predict_distribution(X, strata=strata, offset=offset).survival(
            times, conditional_on=conditional_on
        )

    def predict_standard_error(self, X, *, times, strata=None, offset=None):
        """mgcv coxpred survival SE, including baseline and coefficient terms."""
        prediction = self.predict_distribution(X, strata=strata, offset=offset)
        grid = prediction._times(times)
        matrix = self._evaluation(X, offset).matrix[:, self.free_indices_]
        eta = self.predict_link(X, offset=offset)[:, 0]
        labels = (
            np.repeat(self.strata_[0], len(matrix))
            if strata is None
            else np.asarray(strata)
        )
        from ..gam.inference.cox import prediction_standard_error

        return prediction_standard_error(
            matrix, eta, prediction.survival(grid), grid, labels, self.strata_,
            self.baseline_, self.baseline_uncertainty_, self.covariance_,
        )

    def predict_interval(self, X, *, times, strata=None, offset=None, level=0.95):
        """Approximate pointwise Wald survival limits, including baseline error."""
        from scipy.stats import norm

        from ..survival.uncertainty import SurvivalCurveInterval

        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie in (0,1).")
        prediction = self.predict_distribution(X, strata=strata, offset=offset)
        grid = prediction._times(times)
        estimate = prediction.survival(grid)
        se = self.predict_standard_error(X, times=grid, strata=strata, offset=offset)
        critical = float(norm.ppf((1 + level) / 2))
        return SurvivalCurveInterval(
            grid,
            estimate,
            np.maximum(estimate - critical * se, 0),
            np.minimum(estimate + critical * se, 1),
            np.zeros(len(estimate)),
            level,
            0,
            "cox_baseline_and_coefficient",
            "approximate_pointwise_wald",
            se,
            critical,
        )

    def diagnostics(self, X, y, *, strata=None, offset=None):
        """Source Schoenfeld/cumulative-score and event-time residual data."""
        from ..gam.diagnostics.cox import cox_diagnostics

        evaluation = self._evaluation(X, offset)
        if len(X) != len(y):
            raise ValueError("Cox diagnostics must align with target rows.")
        codes, levels = self._target(y, strata)
        if not set(levels) <= set(self.strata_):
            raise ValueError("Diagnostic strata must be known fitted labels.")
        lookup = {label: i for i, label in enumerate(self.strata_)}
        codes = np.array([lookup[levels[code]] for code in codes])
        return cox_diagnostics(
            evaluation.matrix[:, self.free_indices_],
            self.coef_,
            y,
            codes,
            evaluation.offset,
            self.baseline_,
        )

    def score(self, X, y, *, strata=None, offset=None):
        evaluation = self._evaluation(X, offset)
        if len(X) != len(y):
            raise ValueError("Cox covariates and targets must align.")
        codes, levels = self._target(y, strata, require_event=False)
        if not set(levels) <= set(self.strata_):
            raise ValueError("Scoring strata must be known fitted labels.")
        return float(
            cox_partial_likelihood(
                self.coef_,
                evaluation.matrix[:, self.free_indices_],
                y.time,
                y.event,
                codes,
                evaluation.offset,
            )[0]
            / len(y)
        )

    def distribution_fit_status(self):
        return DistributionFitStatus(
            "gam",
            "converged" if hasattr(self, "coef_") else "unknown",
            "Cox partial likelihood; full probability-density scoring is unsupported.",
        )

    def set_params(self, **params):
        if not params:
            return self
        current = self.get_params(deep=False)
        if params.keys() - current.keys():
            raise ValueError(
                f"Unknown Cox parameters: {sorted(params.keys() - current.keys())}"
            )
        replacement = type(self)(**(current | params))
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    survival_bootstrap_context_names = ("strata",)

    def clone_for_survival_bootstrap(
        self, y, *, training_randomness, seed, checkpoint_path, has_validation
    ):
        """Fresh static Cox refit; baseline and any smoothing are re-estimated."""
        from sklearn.base import clone

        if not hasattr(self, "coef_"):
            raise NotFittedError("Fit CoxGAM before bootstrap.")
        if has_validation or training_randomness != "fixed":
            raise NotImplementedError(
                "Cox bootstrap has no validation/training-randomness route."
            )
        return clone(self), "not_applicable", None

    def save_model(self, path):
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        return load_estimator(path, cls)

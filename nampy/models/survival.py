"""Statistical survival composition without changes to the Poisson fitter.

References
----------
- Laird, N., and Olivier, D. (1981). Covariance Analysis of Censored Survival Data Using
  Log-Linear Analysis Techniques.
  https://doi.org/10.1080/01621459.1981.10477634
- Cox, D. R. (1972). Regression Models and Life-Tables.
  https://doi.org/10.1111/j.2517-6161.1972.tb00899.x

Piecewise-exponential, parametric and partial-likelihood routes retain the separate
estimands documented by NAMpy.
"""

import re
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..persistence import load_estimator, save_estimator
from ..survival import (
    PiecewiseExponentialDesign,
    SurvivalComponents,
    SurvivalPrediction,
)
from ..survival.contracts import validate_weights
from ..survival.data import validate_covariates
from ..survival.likelihood import (
    observed_log_likelihood,
    parameter_offsets,
    parametric_distribution,
    validate_parametric_fit,
)
from ..survival.longitudinal import LongitudinalSurvivalTarget, subject_weight_sum
from ._data import prepare_fit_features, prepare_predict_features
from .gam import GAMRegressor


class SurvivalGAM(BaseEstimator):
    """Survival likelihoods fitted with the existing Poisson or joint engine.

    ``formula`` is a right-hand-side formula for static covariates, e.g.
    ``'~ s(age, k=6)'``. baseline='factor' keeps the interval-factor baseline;
    baseline='smooth' uses a penalized spline at right interval endpoints with
    k basis functions (k must not exceed the interval count). Integration still
    uses constant hazards within each interval; this is not continuous-hazard
    quadrature or Cox fitting.
    None selects linear effects of the supplied columns. Weibull/lognormal
    use a parameter-name mapping of RHS formulas (second parameter defaults
    to an intercept), no cuts, and fixed smoothing or automatic joint REML
    with outer_newton/bfgs. Targets must be
    SurvivalTarget or, for piecewise fitting, LongitudinalSurvivalTarget.
    Right censoring and delayed entry are supported; use the separate CoxGAM
    for static fixed-smoothing Cox models. Other censoring modes, separate ML/GCV criteria and censored EFS are unsupported.
    """

    def capabilities(self):
        """Describe supported survival operations and their conditions."""
        from ..gam.capabilities import estimator_capabilities

        return estimator_capabilities(self, task="survival")

    _estimator_type = "survival"

    def __init__(
        self,
        cuts=None,
        *,
        formula=None,
        likelihood="piecewise_exponential",
        baseline="factor",
        temporal_formula=None,
        optimize_smoothing=True,
        smoothing_method="reml",
        smoothing_optimizer="outer_newton",
        smoothing_params=None,
        k=10,
        basis="tp",
        control=None,
        knots=None,
        irls_tol=1e-7,
    ):
        self.cuts = cuts
        self.formula = formula
        self.likelihood = likelihood
        self.baseline = baseline
        self.temporal_formula = temporal_formula
        self.optimize_smoothing = optimize_smoothing
        self.smoothing_method = smoothing_method
        self.smoothing_optimizer = smoothing_optimizer
        self.smoothing_params = smoothing_params
        self.k = k
        self.basis = basis
        self.control = control
        self.knots = knots
        self.irls_tol = irls_tol

    def fit(self, X, y, *, sample_weight=None, offset=None):
        staged = type(self)(**self.get_params(deep=False))
        staged._fit(X, y, sample_weight=sample_weight, offset=offset)
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        return self

    def _fit(self, X, y, *, sample_weight, offset):
        if self.baseline not in {"factor", "smooth"}:
            raise NotImplementedError("Piecewise baseline must be factor or smooth.")
        if self.temporal_formula is not None and (self.likelihood != "piecewise_exponential" or self.baseline != "smooth"):
            raise NotImplementedError("Temporal effects require piecewise_exponential with a smooth baseline.")
        if self.likelihood in {"weibull", "lognormal"}:
            if isinstance(y, LongitudinalSurvivalTarget):
                raise NotImplementedError("Longitudinal Weibull/lognormal parameter paths are unsupported; use piecewise_exponential.")
            if self.baseline != "factor":
                raise NotImplementedError("Parametric survival does not use a piecewise baseline representation.")
            self._fit_parametric(X, y, sample_weight=sample_weight, offset=offset)
            return
        if self.likelihood != "piecewise_exponential":
            raise NotImplementedError(
                "Supported survival likelihoods are piecewise_exponential, weibull and lognormal."
            )
        design = PiecewiseExponentialDesign(y, self.cuts)
        weights = design.validate_fit(sample_weight, saturated_baseline=self.baseline == "factor")
        if self.baseline == "smooth" and (isinstance(self.k, (bool, np.bool_))
                or not isinstance(self.k, (int, np.integer)) or not 3 <= self.k <= len(design.cuts)-1):
            raise ValueError("Smooth baseline requires integer k>=3 and at least k distinct interval endpoints.")
        frame = prepare_fit_features(self, X)
        self._validate_frame(frame)
        if len(frame) != len(y):
            raise ValueError("X and SurvivalTarget must have the same subject count.")
        known_offset = self._offset(offset, len(y))
        if self.formula is None:
            if any(not name.isidentifier() for name in frame.columns):
                raise ValueError(
                    "Nonidentifier feature names need an explicit formula."
                )
            rhs = " + ".join(frame.columns) or "1"
        else:
            if not isinstance(
                self.formula, str
            ) or not self.formula.lstrip().startswith("~"):
                raise ValueError(
                    "formula must be a right-hand-side formula starting with '~'."
                )
            rhs = self.formula.lstrip()[1:].strip()
            if not rhs:
                raise ValueError("formula must contain a right-hand side.")
            if re.search(r"\boffset\s*\(", rhs):
                raise NotImplementedError(
                    "Use the explicit offset argument for known log-rate offsets."
                )
        expanded_rows, interval = np.nonzero(design.exposure > 0)
        expanded = frame.iloc[expanded_rows].reset_index(drop=True)
        expanded["survival_interval"] = pd.Categorical(
            interval, categories=np.arange(len(design.cuts) - 1)
        )
        events = design.events[expanded_rows, interval]
        expanded["survival_event"] = events
        if self.baseline == "smooth":
            if re.search(r"\bsurvival_(time|interval)\b", rhs):
                raise NotImplementedError("Explicit time-dependent covariate terms require a temporal-effect contract; only a smooth baseline is supported here.")
            # Right-endpoint evaluation defines constant hazards per interval.
            # Fitted spline knots/centering are reused at prediction, never refit.
            expanded["survival_time"] = design.cuts[1:][interval]
            baseline = f"s(survival_time,k={int(self.k)})"
        else:
            baseline = "factor(survival_interval)"
        if self.temporal_formula is not None:
            temporal = self.temporal_formula
            if not isinstance(temporal, str) or not temporal.lstrip().startswith("~"):
                raise ValueError("temporal_formula must be a right-hand-side formula.")
            temporal = temporal.lstrip()[1:].strip()
            from ..gam.formula import ParsedSmoothTerm, parse_gam_formula
            parsed_temporal = parse_gam_formula("~"+temporal)
            temporal_terms = parsed_temporal.predictors[0].terms
            if (not temporal_terms or any(not isinstance(term, ParsedSmoothTerm)
                    or term.kind != "ti" or "survival_time" not in term.features
                    or not (set(term.features)-{"survival_time"}) <= set(frame.columns)
                    or len(term.features) < 2 for term in temporal_terms)):
                raise NotImplementedError("Temporal effects currently require ti(survival_time, covariate, ...) interaction terms.")
            if (not re.search(r"\bsurvival_time\b", temporal)
                    or re.search(r"\boffset\s*\(|\bsurvival_(interval|event)\b", temporal)):
                raise NotImplementedError("Temporal effects use survival_time and source covariates; offsets and interval/event columns are unsupported.")
            rhs += f" + {temporal}"
        formula = f"survival_event ~ {baseline} + {rhs}"
        self.poisson_ = GAMRegressor(
            formula=formula,
            family="poisson",
            optimize_smoothing=self.optimize_smoothing,
            smoothing_method=self.smoothing_method,
            smoothing_optimizer=self.smoothing_optimizer,
            smoothing_params=self.smoothing_params,
            k=self.k,
            basis=self.basis,
            control=self.control,
            knots=self.knots,
            irls_tol=self.irls_tol,
        ).fit(
            expanded,
            sample_weight=weights[expanded_rows],
            offset=np.log(design.exposure[expanded_rows, interval])
            + known_offset[expanded_rows],
        )
        self.predictor_names_ = tuple(f"log_hazard[{i}]" for i in range(len(design.cuts) - 1))
        self.gam_ = self.poisson_.gam_
        fitted = self.gam_.fit_result()
        if fitted.core.converged is False or fitted.core.failed_step:
            raise RuntimeError(
                "Piecewise Poisson coefficient fitting did not converge."
            )
        if not np.isfinite(fitted.coef_full).all():
            raise RuntimeError("Piecewise Poisson fit produced nonfinite coefficients.")
        self.cuts_ = design.cuts
        self.baseline_ = self.baseline
        self.temporal_formula_ = self.temporal_formula
        self.likelihood_ = "piecewise_exponential"
        self.n_subjects_ = y.n_subjects if isinstance(y, LongitudinalSurvivalTarget) else len(y)
        self.offset_required_ = offset is not None
        self.native_poisson_log_likelihood_ = self.gam_.loglik()
        correction: float = float(
            np.sum(
                weights[expanded_rows]
                * events
                * np.log(design.exposure[expanded_rows, interval])
            )
        )
        self.survival_log_likelihood_ = self.native_poisson_log_likelihood_ - correction

    def _fit_parametric(self, X, y, *, sample_weight, offset):
        from ..gam.families.gamlss.censored import CensoredLogNormal, CensoredWeibull
        from ..gam.fit.capabilities import validate_family_smoothing_optimizer
        from .gamlss import GAMLSS

        family = CensoredWeibull() if self.likelihood == "weibull" else CensoredLogNormal()
        if self.cuts is not None:
            raise ValueError("Parametric survival does not use cuts; omit them.")
        if self.smoothing_method not in {"fixed", "reml"}:
            raise NotImplementedError(
                "Separate censored ML/LAML/GCV/UBRE/NCV criteria are unsupported; use fixed smoothing or REML."
            )
        if self.optimize_smoothing:
            if self.smoothing_method != "reml":
                raise NotImplementedError("Automatic censored smoothing requires smoothing_method='reml'.")
            validate_family_smoothing_optimizer(family, self.smoothing_optimizer)
        weights = validate_parametric_fit(y, sample_weight)
        frame = prepare_fit_features(self, X)
        self._validate_frame(frame)
        if len(frame) != len(y):
            raise ValueError("X and SurvivalTarget must have the same subject count.")
        offsets = parameter_offsets(offset, len(y))
        names = family.parameter_names
        formula = self.formula
        if formula is None:
            if any(not name.isidentifier() for name in frame.columns):
                raise ValueError(
                    "Nonidentifier feature names need an explicit formula."
                )
            formula = {
                names[0]: "~ " + (" + ".join(frame.columns) or "1"),
                names[1]: "~ 1",
            }
        elif isinstance(formula, str):
            formula = {names[0]: formula, names[1]: "~ 1"}
        if not isinstance(formula, Mapping) or set(formula) != set(names):
            raise ValueError(f"Parametric formula keys must be {names}.")
        formulas = {}
        for index, name in enumerate(names):
            value = formula[name]
            if (
                not isinstance(value, str)
                or not value.lstrip().startswith("~")
                or not value.lstrip()[1:].strip()
            ):
                raise ValueError(
                    "Survival parameter formulas must be right-hand-side formulas starting with '~'."
                )
            if re.search(r"\boffset\s*\(", value):
                raise NotImplementedError(
                    "Use the explicit two-predictor offset matrix."
                )
            formulas[name] = ("survival_time " if index == 0 else "") + value.lstrip()
        data = frame.copy()
        data["survival_time"] = y.time
        self.parametric_ = GAMLSS(
            formula=formulas,
            family=family,
            optimize_smoothing=self.optimize_smoothing,
            smoothing_method=self.smoothing_method,
            smoothing_optimizer=self.smoothing_optimizer,
            smoothing_params=self.smoothing_params,
            k=self.k,
            basis=self.basis,
            knots=self.knots,
            irls_tol=self.irls_tol,
        )
        self.parametric_.control = self.control
        self.parametric_.fit(
            data,
            sample_weight=weights,
            offset=offsets,
            context={"event": y.event, "entry": y.entry},
        )
        self.predictor_names_ = tuple(self.parametric_.predictor_names_)
        self.gam_ = self.parametric_.gam_
        fitted = self.gam_.fit_result()
        if (
            fitted.core.converged is False
            or fitted.core.failed_step
            or not np.isfinite(fitted.coef_full).all()
        ):
            raise RuntimeError(
                "Censored parametric fitting did not converge to finite coefficients. "
                f"{fitted.core.failure_reason or 'The coefficient backend reported a failed or nonfinite fit.'}"
            )
        self.likelihood_ = self.likelihood
        self.parameter_names_ = tuple(names)
        self.n_subjects_ = y.n_subjects if isinstance(y, LongitudinalSurvivalTarget) else len(y)
        self.offset_required_ = offset is not None
        self.survival_log_likelihood_ = self.gam_.loglik()

    @staticmethod
    def _validate_frame(frame):
        if set(frame.columns) & {
            "survival_event",
            "survival_interval",
            "survival_time",
        }:
            raise ValueError("X contains reserved survival design column names.")
        validate_covariates(frame)

    @staticmethod
    def _offset(offset, n):
        if offset is None:
            return np.zeros(n)
        from .._offsets import predictor_offset

        return predictor_offset(offset, n, 1)[:, 0]

    def _prediction_data(self, X, offset):
        if not hasattr(self, "poisson_"):
            raise NotFittedError("Fit SurvivalGAM before prediction.")
        frame = prepare_predict_features(self, X)
        self._validate_frame(frame)
        if self.offset_required_ and offset is None:
            raise ValueError(
                "Prediction requires explicit new-subject log-rate offsets."
            )
        known = self._offset(offset, len(frame))
        k = len(self.cuts_) - 1
        rows = np.repeat(np.arange(len(frame)), k)
        interval = np.tile(np.arange(k), len(frame))
        expanded = frame.iloc[rows].reset_index(drop=True)
        expanded["survival_interval"] = pd.Categorical(
            interval, categories=np.arange(k)
        )
        if self.baseline_ == "smooth":
            expanded["survival_time"] = self.cuts_[1:][interval]
        return expanded, known[rows], rows, interval

    def predict_link(self, X, *, offset=None):
        """Interval log hazards or the declared latent-parameter predictors."""
        if getattr(self, "likelihood_", None) in {"weibull", "lognormal"}:
            frame = prepare_predict_features(self, X)
            self._validate_frame(frame)
            if self.offset_required_ and offset is None:
                raise ValueError(
                    "Prediction requires explicit new-subject parameter offsets."
                )
            return self.parametric_.predict_link(
                frame, offset=parameter_offsets(offset, len(frame))
            )
        frame, known, rows, _ = self._prediction_data(X, offset)
        eta = self.poisson_.predict_link(frame, offset=known)
        k = len(self.cuts_) - 1
        return np.asarray(eta).reshape(len(rows) // k, k)

    predictor_backend = "gam"

    def survival_time_grid(self):
        """Fitted piecewise grid for shared trajectory evaluation."""
        if not hasattr(self, "likelihood_"):
            raise NotFittedError("Fit before trajectory prediction.")
        if self.likelihood_ != "piecewise_exponential":
            raise NotImplementedError("Time-varying parametric survival paths are unsupported.")
        return self.cuts_.copy()

    def predict_distribution(self, X, *, offset=None):
        return self.survival_from_predictors(self.predict_link(X, offset=offset))

    def distribution_fit_status(self):
        """Report existing coefficient/outer convergence without changing fitting."""
        from ..gam.diagnostics.fit_status import distribution_fit_status

        return distribution_fit_status(getattr(self, "gam_", None))

    def clone_for_survival_bootstrap(self, y, *, training_randomness, seed,
                                     checkpoint_path, has_validation):
        """Fresh statistical configuration; ordinary compilation/smoothing rerun."""
        from sklearn.base import clone

        if (self.likelihood != self.likelihood_
                or (self.likelihood_ == "piecewise_exponential" and
                    (not np.array_equal(self.cuts, self.cuts_) or self.baseline != self.baseline_
                     or self.temporal_formula != self.temporal_formula_))):
            raise ValueError("Bootstrap likelihood/cuts must match the published fit; refit configuration changes first.")
        if has_validation:
            raise NotImplementedError("Statistical survival bootstrap does not use validation data.")
        if training_randomness != "fixed":
            raise NotImplementedError("Independent neural training randomness is not a statistical bootstrap route.")
        return clone(self), "not_applicable", None

    def survival_from_predictors(self, predictors):
        """Evaluate hypothetical fitted-layout predictors through the shared laws."""
        if not hasattr(self, "likelihood_"):
            raise NotFittedError("Fit SurvivalGAM before predictor evaluation.")
        eta = np.asarray(predictors, dtype=float)
        width = 2 if self.likelihood_ in {"weibull", "lognormal"} else len(self.cuts_)-1
        if eta.ndim != 2 or eta.shape[1] != width or not np.isfinite(eta).all():
            raise ValueError(f"predictors must be finite (n_subjects,{width}).")
        if self.likelihood_ in {"weibull", "lognormal"}:
            parameters = self.gam_.family.distribution_parameters_from_eta(eta)
            return SurvivalPrediction(
                distribution=parametric_distribution(self.likelihood_, parameters),
                backend="gam",
            )
        return SurvivalPrediction(np.exp(eta), self.cuts_, backend="gam")

    def _survival_coefficient_design(self, X, offset):
        """Adapt input rows to the statistical joint sampler."""
        from ..gam.inference.survival import coefficient_design

        mean = self.predict_link(X, offset=offset)
        piecewise = self.likelihood_ == "piecewise_exponential"
        if piecewise:
            frame, _, _, _ = self._prediction_data(X, offset)
            matrix = self.poisson_.lpmatrix(frame)
        else:
            frame = prepare_predict_features(self, X)
            matrix = self.parametric_.lpmatrix(frame)
        design, beta = coefficient_design(
            self.gam_, matrix, n_rows=len(mean), n_predictors=mean.shape[1], piecewise=piecewise
        )
        return mean, design, beta

    def _survival_coefficient_draws(self, *, covariance, n_draws, seed, standard_draws):
        from ..gam.inference.survival import survival_coefficient_draws

        return survival_coefficient_draws(
            self.gam_, covariance=covariance, n_draws=n_draws, seed=seed,
            standard_draws=standard_draws,
        )

    def sample_survival_predictions(
        self, X, *, offset=None, n_draws=1000, covariance="bayes", seed=None,
        standard_draws=None,
    ):
        """Joint coefficient draws transformed to shared survival snapshots.

        Full cross-term/predictor covariance and fixed offsets are retained.
        Unsupported coefficient layouts raise; no fitting state is modified.
        """
        mean, design, beta = self._survival_coefficient_design(X, offset)
        draws = self._survival_coefficient_draws(covariance=covariance,
            n_draws=n_draws, seed=seed, standard_draws=standard_draws)
        return tuple(self.survival_from_predictors(
            mean+np.einsum("nkp,p->nk", design, draw-beta)) for draw in draws)

    def _survival_term_designs(self, X, design, components, beta):
        """Use compiled term coefficient identities, checking public decomposition."""
        from ..gam.model_state import _term_full_coefficient_indices
        from ..gam.predict.terms import _prediction_term_groups

        piecewise = self.likelihood_ == "piecewise_exponential"
        labels = self.poisson_._term_label_map() if piecewise else {}
        result = {name: np.zeros_like(design) for name in components.terms}
        mapped = set()
        # predict.gam groups all categorical/interaction columns belonging to
        # one formula term; public contributions use the first block's identity.
        for group in _prediction_term_groups(self.gam_):
            term = group["blocks"][0]
            columns = np.unique(np.concatenate([
                _term_full_coefficient_indices(self.gam_, block) for block in group["blocks"]]))
            if piecewise:
                key = labels.get(str(term.term_id), str(term.term_id))
                if key in result:
                    result[key][:, :, columns] = design[:, :, columns]
                    mapped.add(key)
                continue
            targets = tuple(term.predictor_indices) or (term.predictor_index,)
            for k in targets:
                parameter = self.parameter_names_[k]
                key = f"{parameter}:{term.label}"
                if key not in result:
                    key = f"{parameter}:{term.term_id}"
                if key in result:
                    result[key][:, k, columns] = design[:, k, columns]
                    mapped.add(key)
        mismatched = [name for name, matrix in result.items() if not np.allclose(
            np.einsum("nkp,p->nk", matrix, beta), components.terms[name], atol=1e-10, rtol=1e-10)]
        if mapped != result.keys() or mismatched:
            raise NotImplementedError(
                "Survival term simulation requires an exact compiled/public term mapping; "
                f"unmapped={sorted(result.keys()-mapped)}, mismatched={mismatched}.")
        return result

    def sample_survival_effect_predictions(
        self, X, *, terms, reference_X=None, reference_weight=None, offset=None,
        reference_offset=None, covariance="bayes", n_draws=1000, seed=None,
        standard_draws=None,
    ):
        """Paired reference-plus-term predictions from the same coefficient draws.

        Reference centering is recomputed in every joint draw on fixed reference
        rows/weights. It does not resample the reference population or refit.
        """
        from ..diagnostics.contributions import (
            _weighted_term_mean,
            center_additive_prediction,
        )
        from ..diagnostics.survival import _subject_components
        from ..survival.uncertainty import SurvivalEffectSamples

        mean, design, beta = self._survival_coefficient_design(X, offset)
        current = _subject_components(self.predict_components(X, offset=offset), len(mean))
        matrices = self._survival_term_designs(X, design, current, beta)
        if reference_X is None:
            if reference_offset is not None:
                raise ValueError("reference_offset requires reference_X.")
            reference, ref_matrices = current, matrices
        else:
            ref_mean, ref_design, _ = self._survival_coefficient_design(reference_X, reference_offset)
            reference = _subject_components(self.predict_components(reference_X, offset=reference_offset), len(ref_mean))
            ref_matrices = self._survival_term_designs(reference_X, ref_design, reference, beta)
        if set(reference.terms) != set(current.terms):
            raise NotImplementedError("Reference and prediction term layouts must agree.")
        shifts = {name: _weighted_term_mean(value, reference_weight)
                  for name, value in ref_matrices.items()}
        centered = center_additive_prediction(current, reference=reference, sample_weight=reference_weight)
        baseline = np.array(np.broadcast_to(centered.intercept, mean.shape), dtype=float)
        if centered.offset is not None:
            baseline += centered.offset
        base_design = design-sum(matrices.values())+sum(shifts.values())
        draws = self._survival_coefficient_draws(covariance=covariance,
            n_draws=n_draws, seed=seed, standard_draws=standard_draws)
        references: list[SurvivalPrediction] = []
        selected: dict[str, list[SurvivalPrediction]] = {name: [] for name in terms}
        if not terms or set(terms)-matrices.keys():
            raise ValueError("Select available survival effect terms.")
        for draw in draws:
            delta = draw-beta
            eta = baseline+np.einsum("nkp,p->nk", base_design, delta)
            references.append(self.survival_from_predictors(eta))
            for name in terms:
                contribution = (centered.terms[name]
                    + np.einsum("nkp,p->nk", matrices[name]-shifts[name], delta))
                selected[name].append(self.survival_from_predictors(eta+contribution))
        return SurvivalEffectSamples(tuple(references),
                                     {name: tuple(values) for name, values in selected.items()})

    def predict_components(self, X, *, offset=None):
        if getattr(self, "likelihood_", None) in {"weibull", "lognormal"}:
            self.predict_link(
                X, offset=offset
            )  # Validate schema, domain and required offsets.
            frame = prepare_predict_features(self, X)
            return self.parametric_.predict_components(
                frame, offset=parameter_offsets(offset, len(frame))
            )
        frame, known, rows, interval = self._prediction_data(X, offset)
        prediction = self.poisson_.predict_components(frame, offset=known)
        prediction = replace(prediction, predictor_names=("log_hazard",))
        return SurvivalComponents(prediction, rows, interval, self.cuts_[1:][interval])

    def predict(self, X, *, times, offset=None, conditional_on=None):
        return self.predict_survival(
            X, times=times, offset=offset, conditional_on=conditional_on
        )

    def predict_survival(self, X, *, times, offset=None, conditional_on=None):
        return self.predict_distribution(X, offset=offset).survival(
            times, conditional_on=conditional_on
        )

    def predict_cumulative_hazard(self, X, *, times, offset=None, conditional_on=None):
        return self.predict_distribution(X, offset=offset).cumulative_hazard(
            times, conditional_on=conditional_on
        )

    def predict_hazard(self, X, *, times, offset=None):
        return self.predict_distribution(X, offset=offset).hazard(times)

    def score(self, X, y, sample_weight=None, *, offset=None):
        if not hasattr(self, "likelihood_"):
            raise NotFittedError("Fit SurvivalGAM before scoring.")
        weights = validate_weights(y, sample_weight, allow_longitudinal=True)
        if self.likelihood_ in {"weibull", "lognormal"}:
            distribution = self.predict_distribution(X, offset=offset).distribution
            return float(
                weights @ observed_log_likelihood(y, distribution) / weights.sum()
            )
        design = PiecewiseExponentialDesign(y, self.cuts_)
        eta = self.predict_link(X, offset=offset)
        return float(weights @ design.log_likelihood(eta) / subject_weight_sum(y, weights))

    def set_params(self, **params):
        if not params:
            return self
        unknown = params.keys() - self.get_params(deep=False).keys()
        if unknown:
            raise ValueError(f"Unknown survival parameters: {sorted(unknown)}")
        replacement = type(self)(**(self.get_params(deep=False) | params))
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def _artifact_state(self):
        if not hasattr(self, "gam_"):
            raise NotFittedError("Only fitted survival estimators can be saved.")
        return dict(self.__dict__)

    def save_model(self, path):
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        return load_estimator(path, cls)

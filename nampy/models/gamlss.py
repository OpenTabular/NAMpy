"""First-class sklearn-style GAMLSS estimator backed by :class:`nampy.gam.GAM`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..contracts import AdditivePrediction
from ..gam.families.gamlss import GamlssFamily, gammals, gaulss
from .gam import _GAMAdapterBase

_FAMILY_ALIASES = {
    "normal": gaulss,
    "gaussian": gaulss,
    "gaulss": gaulss,
    "gamma": gammals,
    "gammals": gammals,
}


def _resolve_gamlss_family(family) -> GamlssFamily:
    if isinstance(family, GamlssFamily):
        return family
    key = str(family).lower()
    try:
        factory = _FAMILY_ALIASES[key]
    except KeyError:
        supported = ", ".join(sorted(_FAMILY_ALIASES))
        raise ValueError(
            f"Unknown GAMLSS family {family!r}. Supported aliases: {supported}."
        ) from None
    return factory()


class GAMLSS(_GAMAdapterBase):
    """Generalized additive location, scale, and shape estimator.

    ``predict`` returns natural distribution parameters in columns named by
    ``parameter_names_``. Use ``predict(raw=True)`` or ``predict_link`` for the
    additive linear predictors. In formula mode, a named formula mapping is
    preferred, for example ``{"mu": "y ~ s(x)", "sigma": "~ s(x)"}``.
    """

    def __init__(
        self,
        formula=None,
        family="normal",
        k=10,
        basis="tp",
        fit_intercept=True,
        optimize_smoothing=True,
        smoothing_method="reml",
        smoothing_optimizer="outer_newton",
        smoothing_params=None,
        select=False,
        knots=None,
        min_sp=None,
        drop_intercept=None,
        covariance="bayes",
        score_gamma=1.0,
        max_irls_iter=200,
        irls_tol=1e-7,
        sp_log_bounds=(-80.0, 20.0),
    ):
        super().__init__(
            formula=formula,
            family=family,
            k=k,
            basis=basis,
            fit_intercept=fit_intercept,
            optimize_smoothing=optimize_smoothing,
            smoothing_method=smoothing_method,
            smoothing_optimizer=smoothing_optimizer,
            smoothing_params=smoothing_params,
            select=select,
            knots=knots,
            min_sp=min_sp,
            drop_intercept=drop_intercept,
            covariance=covariance,
            score_gamma=score_gamma,
            max_irls_iter=max_irls_iter,
            irls_tol=irls_tol,
            sp_log_bounds=sp_log_bounds,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def _resolved_family(self):
        return _resolve_gamlss_family(self.family)

    def _validate_family_role(self, family) -> None:
        if not isinstance(family, GamlssFamily):
            raise ValueError(
                "GAMLSS requires a multi-predictor GamlssFamily; "
                f"got {type(family).__name__}."
            )
        if int(family.n_linear_predictors) < 2:
            raise ValueError("GAMLSS families must define at least two predictors.")

    def _resolved_formula(self):
        family = self._resolved_family()
        names = tuple(family.parameter_names)
        formula = self.formula
        if isinstance(formula, Mapping):
            missing = set(names) - set(formula)
            extra = set(formula) - set(names)
            if missing or extra:
                raise ValueError(
                    "GAMLSS formula keys must exactly match the family parameters "
                    f"{names}; missing={sorted(missing)}, extra={sorted(extra)}."
                )
            formulas = [formula[name] for name in names]
        elif isinstance(formula, Sequence) and not isinstance(formula, str):
            formulas = list(formula)
            if len(formulas) != len(names):
                raise ValueError(
                    f"GAMLSS family {family.name!r} expects {len(names)} formulas, "
                    f"got {len(formulas)}."
                )
        else:
            raise TypeError(
                "GAMLSS formula must be a parameter-name mapping or an ordered "
                "sequence of formulas."
            )
        if any(not isinstance(value, str) for value in formulas):
            raise TypeError("Every GAMLSS formula must be a string.")
        for value in formulas[1:]:
            lhs, separator, _ = value.partition("~")
            if not separator or lhs.strip():
                raise ValueError(
                    "Only the first GAMLSS formula may contain a response; "
                    "secondary parameter formulas must be one-sided ('~ ...')."
                )
        return formulas

    def fit(self, X, y=None, *, data=None, sample_weight=None, offset=None):
        """Fit distribution parameters in array or multi-formula mode."""
        family = self._resolved_family()
        if offset is not None:
            if not isinstance(offset, (list, tuple)):
                raise ValueError(
                    "GAMLSS offset must be a list/tuple with one entry per "
                    "distribution parameter."
                )
            if len(offset) != int(family.n_linear_predictors):
                raise ValueError(
                    f"GAMLSS family {family.name!r} expects "
                    f"{family.n_linear_predictors} offsets, got {len(offset)}."
                )
        result = self._fit_gam(
            X, y, data=data, sample_weight=sample_weight, offset=offset
        )
        self.parameter_names_ = tuple(self.gam_.family.parameter_names)
        return result

    def predict(self, X=None, raw=False, offset=None):
        """Return natural parameters, or raw linear predictors with ``raw=True``."""
        eta = np.asarray(self.predict_link(X, offset=offset), dtype=np.float64)
        if raw:
            return eta
        return self.gam_.family.distribution_parameters_from_eta(eta)

    def predict_point(self, X=None, offset=None):
        """Return the family-defined conditional mean prediction."""
        parameters = self.predict(X, offset=offset)
        return self.gam_.family.point_prediction_from_parameters(parameters)

    def standard_errors(self, X=None, type="response", offset=None):
        """Pointwise errors on the natural-parameter or link scale."""
        if type == "link":
            return super().standard_errors(X, type="link", offset=offset)
        if type != "response":
            raise ValueError("type must be 'response' or 'link'.")
        self._check_fitted()
        self._validate_X(X)
        eta, eta_se = self.gam_.predict(
            X, return_se=True, type="link", offset=offset
        )
        jacobian = self.gam_.family.distribution_parameter_jacobian(eta)
        return np.abs(jacobian) * np.asarray(eta_se, dtype=np.float64)

    def score(self, X, y, sample_weight=None):
        """Return weighted mean log likelihood (higher is better)."""
        logpdf = self.gam_.family.logpdf_from_parameters(y, self.predict(X))
        if sample_weight is None:
            return float(np.mean(logpdf))
        weights = np.asarray(sample_weight, dtype=np.float64).ravel()
        if weights.shape != logpdf.shape:
            raise ValueError(
                f"sample_weight must have shape {logpdf.shape}, got {weights.shape}."
            )
        return float(np.average(logpdf, weights=weights))

    def evaluate(self, X, y_true, metrics=None):
        """Evaluate natural parameters (default: mean negative log likelihood)."""
        parameters = self.predict(X)
        if metrics is None:
            logpdf = self.gam_.family.logpdf_from_parameters(y_true, parameters)
            return {"Negative Log-Likelihood": float(-np.mean(logpdf))}
        return {
            metric_name: metric_func(y_true, parameters)
            for metric_name, metric_func in metrics.items()
        }

    def predict_components(self, X=None, offset=None) -> AdditivePrediction:
        """Return zero-padded per-parameter contributions on the link scale."""
        self._check_fitted()
        self._validate_X(X)
        vals = self.gam_.predict_terms(X, offset=offset)
        link = np.asarray(vals["output"], dtype=np.float64)
        response = self.gam_.family.distribution_parameters_from_eta(link)
        n_samples, n_parameters = link.shape

        compiled = self.gam_.gam_result_.require_compiled_model()
        predictor_index = {
            str(predictor.name): index
            for index, predictor in enumerate(compiled.predictors)
        }
        label_map = {
            (str(term.predictor_name), str(term.term_id)): str(term.label)
            for term in compiled.compiled_terms
        }

        terms = {}
        reserved = {"output", "response", "intercept", "offset"}
        for raw_key, value in vals.items():
            if raw_key in reserved:
                continue
            predictor_name, separator, term_id = str(raw_key).partition(":")
            if not separator or predictor_name not in predictor_index:
                raise RuntimeError(f"Unexpected multi-predictor term key {raw_key!r}.")
            index = predictor_index[predictor_name]
            contribution = np.zeros((n_samples, n_parameters), dtype=np.float64)
            contribution[:, index] = np.asarray(value, dtype=np.float64)
            label = label_map.get((predictor_name, term_id), term_id)
            public_key = f"{self.parameter_names_[index]}:{label}"
            if public_key in terms:
                public_key = f"{self.parameter_names_[index]}:{term_id}"
            terms[public_key] = contribution

        intercept = np.zeros(n_parameters, dtype=np.float64)
        raw_intercept = vals.get("intercept", {})
        if isinstance(raw_intercept, Mapping):
            for predictor_name, value in raw_intercept.items():
                intercept[predictor_index[str(predictor_name)]] = float(value)

        offset_array = None
        raw_offset = vals.get("offset")
        if isinstance(raw_offset, Mapping):
            offset_array = np.zeros((n_samples, n_parameters), dtype=np.float64)
            for predictor_name, value in raw_offset.items():
                offset_array[:, predictor_index[str(predictor_name)]] = np.asarray(
                    value, dtype=np.float64
                )

        return AdditivePrediction(
            response=response,
            link=link,
            terms=terms,
            intercept=intercept,
            backend="gam",
            offset=offset_array,
        )


__all__ = ["GAMLSS"]

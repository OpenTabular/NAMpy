"""First-class sklearn-style GAMLSS estimator backed by :class:`nampy.gam.GAM`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..contracts import AdditivePrediction, AdditiveTermMetadata
from ..gam.families.gamlss import GamlssFamily
from ..gam.families.registry import make_distributional_family
from ..gam.predict.distributional import distributional_standard_errors
from ._gam import _GAMAdapterBase


def _resolve_gamlss_family(family) -> GamlssFamily:
    result = make_distributional_family(family)
    if not isinstance(result, GamlssFamily):
        raise ValueError(
            f"Unknown GAMLSS family {family!r}: a general distribution family is required."
        )
    return result


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
        if int(family.n_linear_predictors) < 1:
            raise ValueError("GAMLSS families must define at least one predictor.")

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
            if len(formulas) < len(names):
                raise ValueError(
                    f"GAMLSS family {family.name!r} expects {len(names)} formulas "
                    "or more when extra components have numeric predictor labels, "
                    f"got {len(formulas)}."
                )
        else:
            raise TypeError(
                "GAMLSS formula must be a parameter-name mapping or an ordered "
                "sequence of formulas."
            )
        if any(not isinstance(value, str) for value in formulas):
            raise TypeError("Every GAMLSS formula must be a string.")
        for value in formulas[1 : len(names)]:
            lhs, separator, _ = value.partition("~")
            if not separator or lhs.strip():
                raise ValueError(
                    "Only the first GAMLSS formula may contain a response; "
                    "secondary parameter formulas must be one-sided ('~ ...')."
                )
        if len(formulas) > len(names):
            from ..gam.formula import get_numeric_response_labels

            for value in formulas[len(names) :]:
                labels = get_numeric_response_labels(value)
                if len(labels) < 2:
                    raise ValueError(
                        "Extra GAMLSS formula components must use a numeric "
                        "multi-predictor lhs such as '1 + 2 ~ s(x)'."
                    )
        return formulas

    def fit(self, X, y=None, *, data=None, sample_weight=None, offset=None, context=None):
        """Fit distribution parameters; context supplies explicit observation vectors.

        For betabinomial_lss use context={"trials": trials}. Counts are successes,
        never proportions or success/failure pairs. Context is separate from
        sample_weight and is copied into this fit's isolated family state.
        """
        family = self._resolved_family()
        from ..distributions.observations import ObservationTarget
        if isinstance(y, ObservationTarget):
            if not getattr(family, "supports_observation_target", False):
                raise NotImplementedError("This family does not support ObservationTarget.")
            if context is not None:
                raise ValueError("ObservationTarget already supplies observation context.")
            encoded = y.encoded
            y = encoded[:, 0]
            context = {"observation_upper":encoded[:,1], "observation_code":encoded[:,2]}
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
            X, y, data=data, sample_weight=sample_weight, offset=offset, context=context
        )
        fitted_family = self.gam_.family
        self.predictor_names_ = tuple(fitted_family.parameter_names)
        self.parameter_names_ = tuple(
            getattr(fitted_family, "distribution_parameter_names", ())
            or fitted_family.parameter_names
        )
        return result

    def predict(
        self,
        X=None,
        raw=False,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Return natural parameters, or raw linear predictors with ``raw=True``."""
        eta = np.asarray(
            self.predict_link(
                X,
                offset=offset,
                block_size=block_size,
                newdata_guaranteed=newdata_guaranteed,
                na_action=na_action,
            ),
            dtype=np.float64,
        )
        if raw:
            return eta
        return self.gam_.family.distribution_parameters_from_eta(eta)

    def predict_distribution(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
        context=None,
    ):
        """Return conditional distributions for supported scalar GAMLSS families.

        Parameters
        ----------
        X : DataFrame or ndarray, optional
            Prediction rows; omitted X uses the fitted observations.
        offset : list or tuple, optional
            One known offset per linear predictor, as in ``predict``.
        block_size : int, optional
            Rows processed per statistical prediction block.
        newdata_guaranteed : bool, default=False
            Use the existing validated-newdata prediction contract.
        na_action : str, default="pass"
            Existing prediction missing-row policy. Pass preserves missing
            rows as NaN in the returned distribution.
        context : mapping of str to array, optional
            Explicit row-aligned observation context. Beta-binomial requires
            {"trials": trials} for every supplied X, even if it has the same
            length as training data. Omitted X reuses fit context. Context
            families currently require na_action="pass".

        Returns
        -------
        nampy.distributions.DistributionPrediction
            NumPy probability queries and outcome draws conditional on fitted
            parameters for families declaring a probability adapter in the
            statistical family registry.
            Other families raise ``NotImplementedError``. Existing
            ``predict`` output and mgcv fitting semantics are unchanged.
        """
        from ..gam.predict.distribution_adapter import distribution_prediction

        parameters = self.predict(
            X,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )
        context = self._prediction_context(X, context, na_action)
        return distribution_prediction(self.gam_.family, parameters, context=context)

    def _prediction_context(self, X, context, na_action="pass"):
        """Reuse fit context only for omitted X; never infer it from row count."""
        self._check_fitted()
        names = tuple(getattr(self.gam_.family, "prediction_context_names", getattr(self.gam_.family, "context_names", ())))
        if names and na_action != "pass":
            raise NotImplementedError(
                "Context-dependent predictions currently require na_action='pass' to preserve row alignment."
            )
        if context is None and names and X is None:
            provider = getattr(self.gam_.family, "fit_context", None)
            if not callable(provider):
                raise NotImplementedError(
                    "This context-dependent family does not expose fitted observation context; supply context explicitly."
                )
            return provider(self.gam_.n_samples_)
        return context

    def predict_point(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
        context=None,
    ):
        """Return the family-defined point summary (mean or median)."""
        self._check_fitted()
        if getattr(self.gam_.family, "context_names", ()):
            distribution = self.predict_distribution(
                X, offset=offset, block_size=block_size,
                newdata_guaranteed=newdata_guaranteed, na_action=na_action,
                context=context,
            )
            if getattr(self.gam_.family, "point_summary", "mean") == "median":
                return distribution.quantile(0.5)[:, 0]
            return distribution.mean()
        if context is not None and len(context):
            raise ValueError(f"{self.gam_.family.name} does not accept observation context.")
        parameters = self.predict(
            X,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )
        return self.gam_.family.point_prediction_from_parameters(parameters)

    def standard_errors(
        self,
        X=None,
        type="response",
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
        unconditional=False,
    ):
        """Pointwise errors on the natural-parameter or link scale."""
        if type == "link":
            return super().standard_errors(
                X,
                type="link",
                offset=offset,
                block_size=block_size,
                newdata_guaranteed=newdata_guaranteed,
                na_action=na_action,
                unconditional=unconditional,
            )
        if type != "response":
            raise ValueError("type must be 'response' or 'link'.")
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        return distributional_standard_errors(
            self.gam_, X, cov=self.covariance, offset=offset,
            block_size=block_size, newdata_guaranteed=newdata_guaranteed,
            na_action=na_action, unconditional=unconditional,
        )

    def score(self, X, y, sample_weight=None, *, context=None):
        """Return weighted mean log likelihood (higher is better)."""
        self._check_fitted()
        if getattr(self.gam_.family, "context_names", ()) or context is not None:
            logpdf = self.predict_distribution(X, context=context).log_prob(y)
        else:
            logpdf = self.gam_.family.logpdf_from_parameters(y, self.predict(X))
        if sample_weight is None:
            return float(np.mean(logpdf))
        weights = np.asarray(sample_weight, dtype=np.float64).ravel()
        if weights.shape != logpdf.shape:
            raise ValueError(
                f"sample_weight must have shape {logpdf.shape}, got {weights.shape}."
            )
        return float(np.average(logpdf, weights=weights))

    def distribution_fit_status(self):
        """Report existing coefficient/outer convergence for family comparison.

        Returns
        -------
        nampy.distributions.DistributionFitStatus
            Backend-owned convergence evidence. Missing status is unknown;
            failed coefficient or smoothing convergence is ineligible.
        """
        from ..gam.diagnostics.fit_status import distribution_fit_status

        return distribution_fit_status(getattr(self, "gam_", None))

    def evaluate(self, X, y_true, metrics=None, *, context=None):
        """Evaluate natural parameters (default: mean negative log likelihood)."""
        parameters = self.predict(X)
        distribution = None
        if getattr(self.gam_.family, "context_names", ()) or context is not None:
            distribution = self.predict_distribution(X, context=context)
        if metrics is None:
            logpdf = (distribution.log_prob(y_true) if distribution is not None
                      else self.gam_.family.logpdf_from_parameters(y_true, parameters))
            return {"Negative Log-Likelihood": float(-np.mean(logpdf))}
        return {
            metric_name: metric_func(y_true, parameters)
            for metric_name, metric_func in metrics.items()
        }

    @property
    def predictor_point_summary(self):
        self._check_fitted()
        return getattr(self.gam_.family, "point_summary", "mean")

    def summarize_predictors(self, predictors):
        """Evaluate the family point summary and SD through its probability law."""
        from ..gam.predict.distribution_adapter import distribution_prediction

        self._check_fitted()
        parameters = self.gam_.family.distribution_parameters_from_eta(
            np.asarray(predictors, dtype=float)
        )
        distribution = distribution_prediction(self.gam_.family, parameters)
        point = (distribution.quantile(0.5)[:, 0]
                 if self.predictor_point_summary == "median" else distribution.mean())
        return point, np.sqrt(distribution.variance())

    def predict_components(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ) -> AdditivePrediction:
        """Return contributions in the actual additive predictor coordinates."""
        self._check_fitted()
        if not newdata_guaranteed:
            self._validate_X(X)
        vals = self.gam_.predict_terms(
            X,
            offset=offset,
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )
        eta = np.asarray(vals["output"], dtype=np.float64)
        family = self.gam_.family
        link = eta
        n_samples, n_parameters = link.shape
        names = tuple(family.parameter_names)
        compiled = self.gam_.gam_result_.require_compiled_model()
        predictor_index = {f"eta{index + 1}": index for index in range(n_parameters)}
        label_map = {}
        feature_map = {}
        for term in compiled.compiled_terms:
            targets = tuple(getattr(term, "predictor_indices", ())) or (
                int(getattr(term, "predictor_index", 0)),
            )
            for target in targets:
                label_map[(f"eta{int(target) + 1}", str(term.term_id))] = str(
                    term.label
                )
                features = tuple(term.metadata.get("source_variables", term.feature_info.feature_names))
                by = term.by_variable_info.name
                feature_map[(f"eta{int(target) + 1}", str(term.term_id))] = tuple(
                    dict.fromkeys((*features, *((by,) if by else ())))
                )

        terms = {}
        metadata = {}
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
            public_key = f"{names[index]}:{label}"
            if public_key in terms:
                public_key = f"{names[index]}:{term_id}"
            terms[public_key] = contribution
            features = feature_map.get((predictor_name, term_id), ())
            metadata[public_key] = AdditiveTermMetadata(
                features=features,
                parameter=names[index],
                kind="interaction" if len(features) > 1 else "main",
            )

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
            link=link,
            predictor_names=names,
            terms=terms,
            intercept=intercept,
            backend="gam",
            offset=offset_array,
            term_metadata=metadata,
        )


__all__ = ["GAMLSS"]

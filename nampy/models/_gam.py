"""Shared public-adapter lifecycle for likelihood GAM estimators."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from .._data import feature_frame
from ..contracts import AdditivePrediction, AdditiveTermMetadata, FeatureSchema
from ..gam import GAM
from ..persistence import load_estimator, save_estimator

_GAM_CONSTRUCTOR_PARAMS = (
    "k",
    "basis",
    "fit_intercept",
    "optimize_smoothing",
    "smoothing_method",
    "smoothing_optimizer",
    "smoothing_params",
    "select",
    "knots",
    "min_sp",
    "para_pen",
    "constraints",
    "drop_intercept",
    "covariance",
    "score_gamma",
    "max_irls_iter",
    "irls_tol",
    "sp_log_bounds",
    "scale",
    "control",
    "nei",
    "coefficient_optimizer",
    "optim_method",
    "fisher_diagnostics",
)


class _GAMAdapterBase(BaseEstimator):
    """Shared plumbing for the GAM adapters. Not part of the public API."""

    def capabilities(self):
        """Discover declared statistical operations without evaluating rows."""
        from ..gam.capabilities import estimator_capabilities

        return estimator_capabilities(self, task=getattr(self, "_prediction_kind", "regression"))

    def __init__(
        self,
        formula=None,
        family=None,
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
        para_pen=None,
        constraints=None,
        drop_intercept=None,
        covariance="bayes",
        score_gamma=1.0,
        max_irls_iter=200,
        irls_tol=1e-7,
        sp_log_bounds=(-80.0, 20.0),
        scale=0.0,
        control=None,
        nei=None,
        coefficient_optimizer="newton",
        optim_method=None,
        fisher_diagnostics=False,
    ):
        self.formula = formula
        self.family = family
        self.k = k
        self.basis = basis
        self.fit_intercept = fit_intercept
        self.optimize_smoothing = optimize_smoothing
        self.smoothing_method = smoothing_method
        self.smoothing_optimizer = smoothing_optimizer
        self.smoothing_params = smoothing_params
        self.select = select
        self.knots = knots
        self.min_sp = min_sp
        self.para_pen = para_pen
        self.constraints = constraints
        self.drop_intercept = drop_intercept
        self.covariance = covariance
        self.score_gamma = score_gamma
        self.max_irls_iter = max_irls_iter
        self.irls_tol = irls_tol
        self.sp_log_bounds = sp_log_bounds
        self.scale = scale
        self.control = control
        self.nei = nei
        self.coefficient_optimizer = coefficient_optimizer
        self.optim_method = optim_method
        from ..diagnostics.fisher import normalize_fisher_diagnostics

        normalize_fisher_diagnostics(fisher_diagnostics)
        self.fisher_diagnostics = fisher_diagnostics

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _resolved_family(self):
        return self.family

    def _resolved_formula(self):
        return self.formula

    def _validate_family_role(self, family) -> None:
        """Validate the fitted family's public estimator role."""
        return None

    def _build_gam(self) -> GAM:
        hparams = {name: getattr(self, name) for name in _GAM_CONSTRUCTOR_PARAMS}
        if self.formula is not None:
            hparams["formula"] = self._resolved_formula()
        return GAM(family=self._resolved_family(), **hparams)

    def _fit_gam(self, X, y, *, data=None, sample_weight=None, offset=None, context=None):
        gam = self._build_gam()
        self._validate_family_role(gam.family)
        frame = data if data is not None else X
        offset = self._backend_offset(frame, offset, gam=gam)
        if self.formula is not None:
            frame = data if data is not None else X
            schema = None
            gam.fit(data=frame, y=y, sample_weight=sample_weight, offset=offset, context=context)
        else:
            if X is None:
                raise ValueError("X is required when no formula is given.")
            schema = FeatureSchema.from_data(
                X if hasattr(X, "columns") else np.asarray(X)
            )
            gam.fit(X, y, sample_weight=sample_weight, offset=offset, context=context)

        self.gam_ = gam
        self.predictor_names_ = tuple(getattr(gam.family, "parameter_names", ()) or ("predictor",))
        self.schema_ = schema
        for name in ("fisher_config_", "fisher_history_", "fisher_information_"):
            setattr(self, name, getattr(gam, name))

        self.n_features_in_ = (
            self.schema_.n_features if self.schema_ is not None else None
        )
        if self.schema_ is not None:
            self.feature_names_in_ = np.asarray(
                self.schema_.feature_names, dtype=object
            )
        return self

    def _backend_offset(self, X, offset, *, gam=None, allow_missing=False):
        """Translate the shared row/predictor axes at the backend boundary."""
        from .._offsets import predictor_offset
        from ..gam.model_state import _predictor_full_indices

        if X is None:
            raise ValueError("Explicit input rows are required.")
        gam = self.gam_ if gam is None else gam
        count = (
            len(_predictor_full_indices(gam))
            if getattr(gam, "_fitted", False)
            else int(getattr(gam.family, "n_linear_predictors", 1))
        )
        values = predictor_offset(offset, len(X), count, allow_missing=allow_missing)
        if values is None:
            return None
        if getattr(gam.family, "family_class", "") == "general":
            return list(values.T)
        return values[:, 0]

    def _check_fitted(self):
        if getattr(self, "gam_", None) is None:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

    def fisher_information(self, X, *, offset=None, context=None, config=None,
                           batch_size=None):
        """Evaluate mean likelihood information on explicit reference rows."""
        self._check_fitted()
        self._validate_X(X)
        from ..diagnostics.fisher import FisherDiagnostics, normalize_fisher_diagnostics
        resolved = config if config is not None else normalize_fisher_diagnostics(self.fisher_diagnostics) or FisherDiagnostics()
        if not isinstance(resolved, FisherDiagnostics):
            raise TypeError("config must be FisherDiagnostics or None.")
        return self.gam_.fisher_information(
            X, offset=self._backend_offset(X, offset), context=context,
            config=resolved, return_diagonal=resolved.record_diagonal != "never",
            batch_size=batch_size,
        )

    def _validate_X(self, X):
        if X is None:
            raise ValueError("Explicit input rows are required.")
        schema = getattr(self, "schema_", None)
        if schema is not None:
            schema.validate(X if hasattr(X, "columns") else np.asarray(X))
        return X

    # ------------------------------------------------------------------
    # Shared prediction surface
    # ------------------------------------------------------------------

    def predict_link(
        self,
        X,
        offset=None,
        *,
        batch_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Linear-predictor (link-scale) prediction."""
        self._check_fitted()
        if X is None or not newdata_guaranteed:
            self._validate_X(X)
        values = np.asarray(self.gam_.predict(
            X,
            type="link",
            offset=self._backend_offset(X, offset, allow_missing=True),
            block_size=batch_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        ))
        return values[:, None] if values.ndim == 1 else values

    def predict_components(
        self,
        X,
        offset=None,
        *,
        batch_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ) -> AdditivePrediction:
        """Per-term link-scale contributions as a backend-neutral result."""
        self._check_fitted()
        if X is None or not newdata_guaranteed:
            self._validate_X(X)
        vals = self.gam_.predict_terms(
            X,
            offset=self._backend_offset(X, offset, allow_missing=True),
            block_size=batch_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

        link = np.asarray(vals["output"], dtype=np.float64)
        intercept_val = vals.get("intercept")
        if intercept_val is None:
            intercept: float | np.ndarray = 0.0
        else:
            intercept_arr = np.asarray(intercept_val, dtype=np.float64)
            intercept = (
                float(intercept_arr) if intercept_arr.size == 1 else intercept_arr
            )

        reserved = {"output", "response", "intercept", "offset"}
        label_map = self._term_label_map()
        terms = {
            label_map.get(key, key): np.asarray(value, dtype=np.float64)
            for key, value in vals.items()
            if key not in reserved
        }
        offset_arr = vals.get("offset")

        metadata = {}
        compiled = self.gam_.gam_result_.require_compiled_model()
        for term in compiled.compiled_terms:
            key = label_map.get(str(term.term_id), str(term.term_id))
            if key not in terms:
                continue
            features = tuple(term.metadata.get("source_variables", term.feature_info.feature_names))
            by = term.by_variable_info.name
            features = tuple(dict.fromkeys((*features, *((by,) if by else ()))))
            metadata[key] = AdditiveTermMetadata(
                features=features, kind="interaction" if len(features) > 1 else "main")

        return AdditivePrediction(
            link=link,
            terms=terms,
            intercept=intercept,
            backend="gam",
            term_metadata=metadata,
            offset=None if offset_arr is None else np.asarray(offset_arr),
        )

    def explain_terms(self, X=None, *, max_bins: int = 64, offset=None):
        """Return a binned additive-term table shared with neural estimators."""
        from ..diagnostics.contributions import explain_additive_prediction

        if X is None:
            raise ValueError("X is required to construct an explanation table.")
        return explain_additive_prediction(
            X, self.predict_components(X, offset=offset), max_bins=max_bins
        )

    def term_importance(self, X=None, *, offset=None):
        """Return mean absolute link-scale contribution by additive term."""
        from ..diagnostics.contributions import term_importance_table

        return term_importance_table(self.predict_components(X, offset=offset))

    def interaction_importance(self, X=None, *, offset=None):
        """Return the interaction-only subset of :meth:`term_importance`."""
        table = self.term_importance(X, offset=offset)
        return table.loc[table["term_type"] == "interaction"].reset_index(drop=True)

    def _term_label_map(self) -> dict[str, str]:
        """Map opaque term ids to human-readable term labels when unique."""
        gam_result = getattr(self.gam_, "gam_result_", None)
        compiled_model = None if gam_result is None else gam_result.compiled_model
        if compiled_model is None:
            return {}
        ids = [term.term_id for term in compiled_model.compiled_terms]
        labels = [term.label for term in compiled_model.compiled_terms]
        if len(set(labels)) != len(labels):
            return {}
        return dict(zip(ids, labels, strict=True))

    def standard_errors(
        self,
        X=None,
        type="response",
        offset=None,
        *,
        batch_size=None,
        newdata_guaranteed=False,
        na_action="pass",
        unconditional=False,
    ):
        """Pointwise prediction standard errors."""
        self._check_fitted()
        if X is None or not newdata_guaranteed:
            self._validate_X(X)
        _, se = self.gam_.predict(
            X,
            return_se=True,
            type=type,
            offset=self._backend_offset(X, offset, allow_missing=True),
            block_size=batch_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
            unconditional=unconditional,
        )
        return se

    def lpmatrix(
        self,
        X,
        *,
        batch_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):
        """Linear-predictor matrix for new data."""
        self._check_fitted()
        if X is None or not newdata_guaranteed:
            self._validate_X(X)
        return self.gam_.lpmatrix(
            X,
            block_size=batch_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

    def plot_terms(
        self, X, *, offset=None, rug=False, density=True, pages=0, figsize=None
    ):
        """Render observed one-dimensional contributions with the shared renderer."""
        from ..plotting import prepared_from_contributions, render_term_plots

        components = self.predict_components(X, offset=offset)
        names = (
            self.schema_.feature_names
            if self.schema_ is not None
            else self.gam_.formula_used_columns_
        )
        frame = feature_frame(X, names)
        labels = self._term_label_map()
        mapping = {
            labels.get(term.term_id, term.term_id): term.feature_info.feature_names[0]
            for term in self.gam_.gam_result_.compiled_model.compiled_terms
            if len(term.feature_info.feature_names) == 1
        }
        prepared = prepared_from_contributions(
            frame, components.terms, term_features=mapping
        )
        return render_term_plots(
            prepared, rug=rug, density=density, pages=pages, figsize=figsize
        )

    def summary(self, **kwargs):
        """Print and return the mgcv-style model summary."""
        self._check_fitted()
        return self.gam_.summary(**kwargs)

    def plot(self, **kwargs):
        """Render mgcv-style term plots (delegates to ``GAM.plot``).

        Same family of term-contribution plots as the neural wrappers'
        ``plot_terms``; both go through the shared renderer.
        """
        self._check_fitted()
        return self.gam_.plot(**kwargs)

    def qq(self, **kwargs):
        """Plot residual quantiles; delegates to ``GAM.qq``."""
        self._check_fitted()
        return self.gam_.qq(**kwargs)

    def plot_check(self, **kwargs):
        """Draw model-check panels; delegates to ``GAM.plot_check``."""
        self._check_fitted()
        return self.gam_.plot_check(**kwargs)

    def vis(self, **kwargs):
        """Plot a conditional prediction surface; delegates to ``GAM.vis``."""
        self._check_fitted()
        return self.gam_.vis(**kwargs)

    def derivative(self, X=None, **kwargs):
        """Return a smooth derivative and its Bayesian standard error."""
        self._check_fitted()
        return self.gam_.derivative(X, **kwargs)

    def plot_derivative(self, X=None, **kwargs):
        """Plot a smooth derivative with a pointwise Bayesian band."""
        self._check_fitted()
        return self.gam_.plot_derivative(X, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path):
        """Save this estimator to a trusted NAMpy pickle artifact."""
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        """Load a trusted NAMpy artifact of this estimator class."""
        return load_estimator(path, cls)

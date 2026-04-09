# basemodels/gam.py
"""User-facing classical GAM backend built on the general smooth-model core."""
import pickle

import numpy as np
import pandas as pd

from ..configs.gam_config import DefaultGAMConfig
from ..gam.families import make_gam_family
from ..gam.model import _GAMDataMixin, _GAMSolveMixin, _GAMSpecsMixin
from ..gam.parity import build_parity_snapshot


class GAM(_GAMDataMixin, _GAMSpecsMixin, _GAMSolveMixin):
    """
    User-facing classical GAM backend built on the general smooth-model core.
    """

    def __init__(
        self,
        cat_feature_info=None,
        num_feature_info=None,
        num_classes: int = 1,
        config: DefaultGAMConfig = DefaultGAMConfig(),
        family=None,
        **kwargs,
    ):
        self.hparams = {
            k: v
            for k, v in kwargs.items()
            if k not in ("cat_feature_info", "num_feature_info")
        }

        self.k = int(self.hparams.get("k", getattr(config, "k", 10)))
        self.basis = self.hparams.get("basis", getattr(config, "basis", "tp"))
        self.fit_intercept = bool(
            self.hparams.get("fit_intercept", getattr(config, "fit_intercept", True))
        )
        self.max_irls_iter = int(
            self.hparams.get("max_irls_iter", getattr(config, "max_irls_iter", 100))
        )
        self.irls_tol = float(
            self.hparams.get("irls_tol", getattr(config, "irls_tol", 1e-11))
        )
        self.max_step_halving = int(
            self.hparams.get(
                "max_step_halving", getattr(config, "max_step_halving", 25)
            )
        )
        self.smoothing_params = self.hparams.get(
            "smoothing_params", getattr(config, "smoothing_params", None)
        )
        self.optimize_smoothing = bool(
            self.hparams.get(
                "optimize_smoothing",
                getattr(config, "optimize_smoothing", False),
            )
        )
        self.smoothing_method = str(
            self.hparams.get(
                "smoothing_method",
                getattr(config, "smoothing_method", "fixed"),
            )
        ).lower()
        self.smoothing_optimizer = str(
            self.hparams.get(
                "smoothing_optimizer",
                getattr(config, "smoothing_optimizer", "lbfgsb"),
            )
        ).lower()
        self.sp_log_bounds = tuple(
            self.hparams.get("sp_log_bounds", config.sp_log_bounds)
        )
        self.score_gamma = float(
            self.hparams.get("score_gamma", getattr(config, "score_gamma", 1.0))
        )
        self.covariance = str(
            self.hparams.get("covariance", getattr(config, "covariance", "bayes"))
        ).lower()

        self.select = bool(self.hparams.get("select", getattr(config, "select", False)))

        self.main_effects = bool(
            self.hparams.get("main_effects", getattr(config, "main_effects", True))
        )
        self.tensor_terms = self.hparams.get(
            "tensor_terms", getattr(config, "tensor_terms", None)
        )

        self.knots = self.hparams.get("knots", getattr(config, "knots", None))
        self.min_sp = self.hparams.get("min_sp", getattr(config, "min_sp", None))
        self.drop_intercept = self.hparams.get(
            "drop_intercept", getattr(config, "drop_intercept", None)
        )

        self.family = make_gam_family(family)

        self._device = "cpu"

        self.formula = self.hparams.get("formula", getattr(config, "formula", None))

        self.formula_ = None
        self.formula_mode_ = False
        self.formula_response_name_ = None
        self.formula_preprocess_state_ = None

        # mirrored fitted attributes
        self.feature_names = None
        self.X_ = None
        self.y_ = None
        self.prior_weights_ = None
        self.offset_train_ = None
        self.offset_predict_default_ = None
        self.predictor_designs = None
        self.n_samples_ = None
        self.design_ = None
        self.Z = None
        self.ZTZ = None
        self.n_coef_ = None
        self.term_blocks_ = None
        self.penalty_blocks_ = None
        self.n_smoothing_params_ = None
        self.smoothing_fixed_mask_ = None
        self.smoothing_override_values_ = None
        self.smoothing_override_modes_ = None
        self.min_sp_ = None
        self.reparam_state_ = None
        self.sl_blocks_ = None
        self.intercept_ = None
        self.coef_ = None
        self.coef_full_ = None
        self.beta = None
        self.edf_ = None
        self.trace_H_ = None
        self.scale_ = None
        self.rss_ = None
        self.deviance_ = None
        self.edf_by_term_ = None
        self.Vp_ = None
        self.Vf_ = None
        self._fitted = False
        self._optim_method = None
        self._optim_result = None
        self._optim_trace = None
        self._optim_used_gradient = None
        self._optim_used_hessian = None
        self.smoothing_score_ = None
        self.result_ = None
        self.side_condition_reports_ = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, path, device="cpu"):
        with open(path, "rb") as f:
            other = pickle.load(f)
        self.__dict__.update(other.__dict__)
        self._device = device
        return self

    def get_device(self):
        return self._device

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def formula_used_columns_(self):
        if self.formula_preprocess_state_ is None:
            return None
        used_columns = self.formula_preprocess_state_.get("used_columns")
        if used_columns is None:
            return None
        return list(used_columns)

    @property
    def formula_offset_name_(self):
        if self.formula_preprocess_state_ is None:
            return None
        return self.formula_preprocess_state_.get("offset_name")

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X=None,
        y=None,
        optimize_smoothing=None,
        smoothing_method=None,
        data=None,
        formula=None,
        offset=None,
        sample_weight=None,
        min_sp=None,
        knots=None,
        drop_intercept=None,
    ):
        formula = self.formula if formula is None else formula
        knots = self.knots if knots is None else knots
        min_sp = self.min_sp if min_sp is None else min_sp
        drop_intercept = (
            self.drop_intercept if drop_intercept is None else drop_intercept
        )

        if formula is not None:
            if data is None:
                if isinstance(X, pd.DataFrame):
                    data = X
                    X = None
                else:
                    raise ValueError(
                        "Formula-based fitting requires `data` as a pandas DataFrame "
                        "(or pass the DataFrame as `X`)."
                    )

            (
                parsed,
                predictor_specs,
                X_np,
                feature_names,
                y_out,
                used_cols,
                offset_formula,
                preprocess_state,
            ) = self._prepare_formula_inputs(
                data=data,
                formula=formula,
                y=y,
                knots=knots,
                drop_intercept=drop_intercept,
            )

            self.formula_ = parsed
            self.formula_mode_ = True
            self.formula_response_name_ = parsed.response_name
            self.formula_preprocess_state_ = preprocess_state

            fit_intercept = bool(parsed.predictors[0].intercept)
            self.fit_intercept = fit_intercept
            y_use = y_out

            offset_arg = self._coerce_optional_offset(offset, len(X_np))
            offset_use = self._combine_offsets(offset_formula, offset_arg)

            # mgcv-like prediction semantics:
            # remember only formula offsets for default prediction.
            predict_offset_default = (
                None
                if offset_formula is None
                else np.asarray(offset_formula, dtype=np.float64).copy()
            )
        else:
            X_np, feature_names = self._coerce_X(X)
            predictor_specs = self._make_predictor_specs(feature_names, knots=knots)
            fit_intercept = self.fit_intercept
            y_use = y

            self.formula_ = None
            self.formula_mode_ = False
            self.formula_response_name_ = None
            self.formula_preprocess_state_ = None

            # Separate fit-time offset is used in fitting, but not remembered by default
            # for prediction, matching mgcv's documented behaviour.
            offset_use = self._coerce_optional_offset(offset, len(X_np))
            predict_offset_default = None

        if y_use is None:
            raise ValueError(
                "y must be supplied, or a formula with a response column must be provided."
            )

        sw_use = None
        if sample_weight is not None:
            if isinstance(sample_weight, str):
                if formula is None or data is None:
                    raise ValueError(
                        "sample_weight as a column name requires formula fitting with `data`."
                    )
                if sample_weight not in data.columns:
                    raise ValueError(
                        f"sample_weight column {sample_weight!r} not found in data."
                    )
                sw_use = np.asarray(
                    data[sample_weight].to_numpy(), dtype=np.float64
                ).ravel()
            else:
                sw_use = np.asarray(sample_weight, dtype=np.float64).ravel()
            if sw_use.shape[0] != len(y_use):
                raise ValueError(
                    "sample_weight must have the same length as y "
                    f"({len(y_use)}), got {sw_use.shape[0]}."
                )

        self.predictor_specs = predictor_specs
        self.fit_intercept = fit_intercept
        self.min_sp = min_sp
        self.result_ = None
        self.side_condition_reports_ = None

        from ..gam.fit.orchestrator import fit_model_core

        fit_model_core(
            X=X_np,
            feature_names=feature_names,
            y=y_use,
            offset=offset_use,
            optimize_smoothing=optimize_smoothing,
            smoothing_method=smoothing_method,
            model=self,
            sample_weight=sw_use,
        )

        # Default prediction offset follows mgcv semantics:
        # formula offset(...) is remembered; separate fit-time offset is not.
        self.offset_predict_default_ = (
            None
            if predict_offset_default is None
            else np.asarray(predict_offset_default, dtype=np.float64).copy()
        )
        return self

    def fit_result(self, include_covariances=True):
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if self.result_ is None:
            self.result_ = self._build_fit_result()
        return (
            self.result_ if include_covariances else self.result_.without_covariances()
        )

    def _select_cov(self, cov):
        from ..gam.fit.covariance import select_covariance_matrix

        return select_covariance_matrix(self, cov=cov)

    def parity_snapshot(self, X=None, include_covariances=False):
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return build_parity_snapshot(self, X=X, include_covariances=include_covariances)

    def predict(self, X=None, return_se=False, cov=None, type="response", offset=None):
        from ..gam.predict import predict_values

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        if X is None:
            offset_use = None
            if offset is not None:
                offset_use = self._coerce_optional_offset(offset, self.X_.shape[0])
            return predict_values(
                X=None,
                return_se=return_se,
                cov=cov,
                type=type,
                offset=offset_use,
                model=self,
            )

        if self.formula_mode_:
            X_np, _, offset_formula = self._coerce_formula_predict_inputs(X)
            offset_use = self._combine_offsets(
                offset_formula,
                self._coerce_optional_offset(offset, X_np.shape[0]),
            )
        else:
            X_np, _ = self._coerce_X(X)
            offset_use = self._coerce_optional_offset(offset, X_np.shape[0])

        return predict_values(
            X=X_np,
            return_se=return_se,
            cov=cov,
            type=type,
            offset=offset_use,
            model=self,
        )

    def predict_feature_vals(self, X=None, offset=None):
        from ..gam.predict import predict_values

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            offset_use = None
            if offset is not None:
                offset_use = self._coerce_optional_offset(offset, self.X_.shape[0])
            X_use = None
        elif self.formula_mode_:
            X_np, _, offset_formula = self._coerce_formula_predict_inputs(X)
            offset_use = self._combine_offsets(
                offset_formula,
                self._coerce_optional_offset(offset, X_np.shape[0]),
            )
            X_use = X_np
        else:
            X_np, _ = self._coerce_X(X)
            offset_use = self._coerce_optional_offset(offset, X_np.shape[0])
            X_use = X_np

        eta = predict_values(model=self, X=X_use, type="link", offset=offset_use)
        terms = predict_values(model=self, X=X_use, type="terms", offset=offset_use)
        out = {"output": eta}
        if self.family.name != "gaussian":
            out["response"] = predict_values(
                model=self, X=X_use, type="response", offset=offset_use
            )
        for j, tb in enumerate(self.term_blocks_):
            out[tb.term_id] = terms[:, j]
        if self.fit_intercept:
            out["intercept"] = np.array(self.intercept_, dtype=np.float64)
        if offset_use is not None:
            out["offset"] = np.asarray(offset_use, dtype=np.float64)
        return out

    def lpmatrix(self, X):
        from ..gam.predict import build_lpmatrix

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        if self.formula_mode_:
            X_np, _, _ = self._coerce_formula_predict_inputs(X)
        else:
            X_np, _ = self._coerce_X(X)

        return build_lpmatrix(self, X_new=X_np)

    def plot(self, X=None, y=None, n_cols=2, figsize=None):
        from ..gam.diagnostics import plot_gam_terms

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            return plot_gam_terms(self, X=None, n_cols=n_cols, figsize=figsize)

        if self.formula_mode_:
            X_np, _, _ = self._coerce_formula_predict_inputs(X)
        else:
            X_np, _ = self._coerce_X(X)

        return plot_gam_terms(self, X=X_np, n_cols=n_cols, figsize=figsize)

    def summary(self):
        from ..gam.diagnostics import print_summary

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return print_summary(self)

    def residuals(self, type="deviance"):
        from ..gam.diagnostics import residuals_gam

        return residuals_gam(self, type=type)

    def concurvity(self, full=True):
        from ..gam.diagnostics import concurvity

        return concurvity(self, full=full)

    def k_check(self, subsample=5000, n_rep=400, seed=None):
        from ..gam.diagnostics import k_check

        return k_check(self, subsample=subsample, n_rep=n_rep, seed=seed)

    def gam_check(self, *, type="deviance", k_sample=5000, k_rep=200, seed=None):
        from ..gam.diagnostics import gam_check

        return gam_check(
            self,
            type=type,
            k_sample=k_sample,
            k_rep=k_rep,
            seed=seed,
        )

    def sp_vcov(self, edge_correct=True, reg=1e-3):
        from ..gam.smoothing_selection import sp_vcov

        return sp_vcov(self, edge_correct=edge_correct, reg=reg)

    def gam_vcomp(self, *, rescale=False, conf_lev=0.95):
        from ..gam.smoothing_selection import gam_vcomp

        return gam_vcomp(self, rescale=rescale, conf_lev=conf_lev)

    def one_se_rule(self, candidate_indices=None):
        from ..gam.smoothing_selection import one_se_rule

        return one_se_rule(self, candidate_indices=candidate_indices)

    def anova(self, *models, dispersion=None, test=None, freq=False):
        from ..gam.inference import anova_gam

        return anova_gam(
            self,
            *models,
            dispersion=dispersion,
            test=test,
            freq=freq,
        )

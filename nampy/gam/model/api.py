"""User-facing mgcv-aligned GAM model."""

import pickle

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .._model_state import (
    _coef_full,
    _cov_bayes,
    _cov_freq,
    _cov_unconditional,
    _edf2,
    _edf_total,
    _fit_result,
    _fit_scale,
    _fit_state,
    _fitted_eta,
    _fitted_mu,
    _intercept,
    _predictor_full_slices,
    _term_blocks_seq,
)
from ..data import (
    coerce_formula_predict_inputs,
    coerce_optional_offset,
    coerce_X,
    combine_offsets,
    copy_offset,
)
from ..families import make_gam_family
from ..fit.offsets import coerce_offset_array
from ..fit.result_builders import copy_fit_result
from ..parity import build_parity_snapshot
from ..smoothing_selection.criteria.gaussian_reml_algebra import (
    gaussian_reml_weighted_degrees_and_log_weight_term,
)
from ..specs.modeling import make_predictor_specs, prepare_formula_inputs


class GAM:
    """
    User-facing classical GAM backend built on the general smooth-model core.
    """

    def __init__(
        self,
        cat_feature_info=None,
        num_feature_info=None,
        num_classes: int = 1,
        family=None,
        **kwargs,
    ):
        self.hparams = {
            k: v
            for k, v in kwargs.items()
            if k not in ("cat_feature_info", "num_feature_info")
        }

        self.k = int(self.hparams.get("k", 10))
        self.basis = self.hparams.get("basis", "tp")
        self.fit_intercept = bool(self.hparams.get("fit_intercept", True))
        self.max_irls_iter = int(self.hparams.get("max_irls_iter", 200))
        self.irls_tol = float(self.hparams.get("irls_tol", 1e-7))
        self.max_step_halving = int(self.hparams.get("max_step_halving", 25))
        self.smoothing_params = self.hparams.get("smoothing_params", None)
        self.optimize_smoothing = bool(self.hparams.get("optimize_smoothing", False))
        self.smoothing_method = str(
            self.hparams.get("smoothing_method", "fixed")
        ).lower()
        self.smoothing_optimizer = str(
            self.hparams.get("smoothing_optimizer", "outer_newton")
        ).lower()
        self.sp_log_bounds = tuple(self.hparams.get("sp_log_bounds", (-80.0, 20.0)))
        self.score_gamma = float(self.hparams.get("score_gamma", 1.0))
        self.covariance = str(self.hparams.get("covariance", "bayes")).lower()
        self.select = bool(self.hparams.get("select", False))
        self.main_effects = bool(self.hparams.get("main_effects", True))
        self.tensor_terms = self.hparams.get("tensor_terms", None)
        self.knots = self.hparams.get("knots", None)
        self.min_sp = self.hparams.get("min_sp", None)
        self.drop_intercept = self.hparams.get("drop_intercept", None)
        self.nei = self.hparams.get("nei", None)

        self.family = make_gam_family(family)

        self._device = "cpu"

        self.formula = self.hparams.get("formula", None)

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
        self.n_samples_ = None
        self.smoothing_fixed_mask_ = None
        self.smoothing_override_values_ = None
        self.smoothing_override_modes_ = None
        self.min_sp_ = None
        self.reparam_state_ = None
        self.sl_blocks_ = None
        self._fitted = False
        self._optim_method = None
        self._optim_result = None
        self._optim_trace = None
        self._optim_used_gradient = None
        self._optim_used_hessian = None
        self.smoothing_score_ = None
        self.gam_result_ = None
        self.fit_core_solution_ = None
        self.side_condition_reports_ = None
        self._coef_reduced_to_full_idx = None
        self.compiled_model_ = None
        self._edf_by_term_fit_ = None

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
    def formula_feature_columns_(self):
        if self.formula_preprocess_state_ is None:
            return None
        feature_columns = self.formula_preprocess_state_.get("feature_columns")
        if feature_columns is None:
            feature_columns = self.formula_preprocess_state_.get("used_columns")
        if feature_columns is None:
            return None
        return list(feature_columns)

    @property
    def formula_offset_name_(self):
        offset_names = self.formula_offset_names_
        if offset_names is None or len(offset_names) != 1:
            return None
        return offset_names[0]

    @property
    def formula_offset_names_(self):
        if self.formula_preprocess_state_ is None:
            return None
        offset_names = self.formula_preprocess_state_.get("offset_names", None)
        if offset_names is not None:
            return tuple(offset_names)
        offset_name = self.formula_preprocess_state_.get("offset_name", None)
        return None if offset_name is None else (offset_name,)

    def _coerce_api_offset(self, offset, n_rows):
        if str(getattr(self.family, "family_class", "")).lower() == "general":
            return coerce_offset_array(offset, n_rows, name="offset")
        return coerce_optional_offset(offset, n_rows)

    def _general_family_offset_list(self):
        offset = getattr(self, "offset_train_", None)
        if offset is None:
            return None
        n_pred = len(_predictor_full_slices(self))
        if isinstance(offset, (list, tuple)):
            out = [
                None if off is None else np.asarray(off, dtype=np.float64)
                for off in offset
            ]
        else:
            out = [np.asarray(offset, dtype=np.float64)]
        if len(out) < n_pred:
            out.extend([None] * (n_pred - len(out)))
        return out

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
        nei=None,
    ):
        formula = self.formula if formula is None else formula
        knots = self.knots if knots is None else knots
        min_sp = self.min_sp if min_sp is None else min_sp
        drop_intercept = (
            self.drop_intercept if drop_intercept is None else drop_intercept
        )
        self.nei = self.nei if nei is None else nei

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
            ) = prepare_formula_inputs(
                self,
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

            offset_arg = self._coerce_api_offset(offset, len(X_np))
            offset_use = combine_offsets(offset_formula, offset_arg)

            # mgcv-like prediction semantics:
            # remember only formula offsets for default prediction.
            predict_offset_default = (
                None if offset_formula is None else copy_offset(offset_formula)
            )
        else:
            X_np, feature_names = coerce_X(self, X)
            predictor_specs = make_predictor_specs(self, feature_names, knots=knots)
            fit_intercept = self.fit_intercept
            y_use = y

            self.formula_ = None
            self.formula_mode_ = False
            self.formula_response_name_ = None
            self.formula_preprocess_state_ = None

            # Separate fit-time offset is used in fitting, but not remembered by default
            # for prediction, matching mgcv's documented behaviour.
            offset_use = self._coerce_api_offset(offset, len(X_np))
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
        self.gam_result_ = None
        self.side_condition_reports_ = None
        self._edf_by_term_fit_ = None

        from ..fit import fit_model_core

        fit_model_core(
            X=X_np,
            feature_names=feature_names,
            y=y_use,
            fit_offset=offset_use,
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
            else copy_offset(predict_offset_default)
        )
        return self

    def fit_result(self, include_covariances=True):
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if self.gam_result_ is None:
            from ..fit.result_builders import build_gam_result

            self.gam_result_ = build_gam_result(self)
        return copy_fit_result(
            self.gam_result_.fit_summary,
            include_covariances=include_covariances,
        )

    def _select_cov(self, cov):
        from ..fit import select_covariance_matrix

        return select_covariance_matrix(self, cov=cov)

    def _resolve_ml_reml_scoring_backend(self, method="reml"):
        from ..fit.capabilities import resolve_ml_reml_scoring_backend

        return resolve_ml_reml_scoring_backend(self, method=method)

    def _expand_smoothing_params_from_log(self, log_free_sp):
        from ..fit.smoothing_params import expand_smoothing_params_from_log

        return expand_smoothing_params_from_log(self, log_free_sp)

    def _has_tensor_terms(self) -> bool:
        for tb in _term_blocks_seq(self):
            if str(getattr(tb, "term_type", "")).lower() in {
                "tensor_smooth",
                "tensor_interaction",
            }:
                return True
        return False

    def parity_snapshot(self, X=None, include_covariances=False):
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return build_parity_snapshot(self, X=X, include_covariances=include_covariances)

    def predict(
        self,
        X=None,
        return_se=False,
        cov=None,
        type="response",
        offset=None,
    ):
        from ..predict import predict_values

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        if X is None:
            offset_use = None
            if offset is not None:
                offset_use = self._coerce_api_offset(offset, self.X_.shape[0])
            return predict_values(
                X=None,
                return_se=return_se,
                cov=cov,
                type=type,
                offset=offset_use,
                model=self,
            )

        if self.formula_mode_:
            X_np, _, offset_formula = coerce_formula_predict_inputs(self, X)
            offset_use = combine_offsets(
                offset_formula,
                self._coerce_api_offset(offset, X_np.shape[0]),
            )
        else:
            X_np, _ = coerce_X(self, X)
            offset_use = self._coerce_api_offset(offset, X_np.shape[0])

        return predict_values(
            X=X_np,
            return_se=return_se,
            cov=cov,
            type=type,
            offset=offset_use,
            model=self,
        )

    def predict_feature_vals(self, X=None, offset=None):
        from ..predict import predict_values

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            offset_use = None
            if offset is not None:
                offset_use = self._coerce_api_offset(offset, self.X_.shape[0])
            X_use = None
        elif self.formula_mode_:
            X_np, _, offset_formula = coerce_formula_predict_inputs(self, X)
            offset_use = combine_offsets(
                offset_formula,
                self._coerce_api_offset(offset, X_np.shape[0]),
            )
            X_use = X_np
        else:
            X_np, _ = coerce_X(self, X)
            offset_use = self._coerce_api_offset(offset, X_np.shape[0])
            X_use = X_np

        eta = predict_values(model=self, X=X_use, type="link", offset=offset_use)
        terms = predict_values(model=self, X=X_use, type="terms", offset=offset_use)
        out = {"output": eta}
        if self.family.name != "gaussian":
            out["response"] = predict_values(
                model=self, X=X_use, type="response", offset=offset_use
            )
        for j, tb in enumerate(_term_blocks_seq(self)):
            out[tb.term_id] = terms[:, j]
        if self.fit_intercept:
            out["intercept"] = np.array(_intercept(self), dtype=np.float64)
        if offset_use is not None:
            out["offset"] = np.asarray(offset_use, dtype=np.float64)
        return out

    def lpmatrix(self, X):
        from ..predict import build_lpmatrix

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        if self.formula_mode_:
            X_np, _, _ = coerce_formula_predict_inputs(self, X)
        else:
            X_np, _ = coerce_X(self, X)

        return build_lpmatrix(self, X_new=X_np)

    def plot(self, X=None, y=None, n_cols=2, figsize=None):
        from ..diagnostics import plot_gam_terms

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            return plot_gam_terms(self, X=None, n_cols=n_cols, figsize=figsize)

        if self.formula_mode_:
            return plot_gam_terms(self, X=X, n_cols=n_cols, figsize=figsize)

        X_np, _ = coerce_X(self, X)
        return plot_gam_terms(self, X=X_np, n_cols=n_cols, figsize=figsize)

    def summary(self):
        from ..diagnostics import print_summary

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return print_summary(self)

    def residuals(self, type="deviance"):
        from ..diagnostics import residuals_gam

        return residuals_gam(self, type=type)

    def concurvity(self, full=True):
        from ..diagnostics import concurvity

        return concurvity(self, full=full)

    def k_check(self, subsample=5000, n_rep=400, seed=None):
        from ..diagnostics import k_check

        return k_check(self, subsample=subsample, n_rep=n_rep, seed=seed)

    def gam_check(self, *, type="deviance", k_sample=5000, k_rep=200, seed=None):
        from ..diagnostics import gam_check

        return gam_check(
            self,
            type=type,
            k_sample=k_sample,
            k_rep=k_rep,
            seed=seed,
        )

    def sp_vcov(self, edge_correct=True, reg=1e-3):
        from ..smoothing_selection import sp_vcov

        return sp_vcov(self, edge_correct=edge_correct, reg=reg)

    def vcov(
        self,
        *,
        sandwich: bool = False,
        freq: bool = False,
        dispersion: float | None = None,
        unconditional: bool = False,
    ):
        """
        Extract mgcv-style coefficient covariance matrix.

        Mirrors mgcv ``vcov.gam`` in mgcv/R/mgcv.r.
        """
        Vp = (
            None
            if _cov_bayes(self) is None
            else np.asarray(_cov_bayes(self), dtype=np.float64)
        )
        Vf = (
            None
            if _cov_freq(self) is None
            else np.asarray(_cov_freq(self), dtype=np.float64)
        )
        Vc = _cov_unconditional(self)
        Vc = None if Vc is None else np.asarray(Vc, dtype=np.float64)
        fit_state = _fit_state(self)
        fit_scale = float(_fit_scale(self))
        coef_full = np.asarray(_coef_full(self), dtype=np.float64)
        edf_total = float(_edf_total(self))
        if sandwich:
            B2 = np.zeros_like(Vp) if freq else (Vp - Vf)
            X = np.asarray(fit_state.X, dtype=np.float64)
            m = float(X.shape[0])
            m = m / (m - edf_total)

            family_class = str(getattr(self.family, "family_class", "")).lower()
            if family_class == "general":
                if not hasattr(self.family, "sandwich"):
                    raise RuntimeError(
                        "Sandwich covariance matrix is not available for this general family."
                    )
                jj = [
                    np.arange(sl.start, sl.stop, dtype=int)
                    for sl in _predictor_full_slices(self)
                ]
                offset = self._general_family_offset_list()
                fill = np.asarray(
                    self.family.sandwich(
                        np.asarray(self.y_, dtype=np.float64),
                        X,
                        jj,
                        coef_full,
                        (
                            None
                            if self.prior_weights_ is None
                            else np.asarray(self.prior_weights_, dtype=np.float64)
                        ),
                        offset=offset,
                    ),
                    dtype=np.float64,
                )
                vc = m * Vp @ fill @ Vp + B2
            elif family_class == "extended":
                raise RuntimeError(
                    "Sandwich covariance matrix is not implemented for this extended family."
                )
            else:
                eta = np.asarray(_fitted_eta(self), dtype=np.float64).ravel()
                mu = np.asarray(_fitted_mu(self), dtype=np.float64).ravel()
                w = (
                    self.family.mu_eta(eta)
                    * (np.asarray(self.y_, dtype=np.float64).ravel() - mu)
                    / (fit_scale * self.family.variance(mu))
                )
                WX = np.asarray(w[:, None] * X, dtype=np.float64)
                vc = m * Vp @ (WX.T @ WX) @ Vp + B2

            vc = np.asarray(vc, dtype=np.float64).copy()
            if dispersion is not None:
                vc *= float(dispersion) / fit_scale
            return vc

        if freq:
            vc = Vf
        else:
            vc = Vc if unconditional and Vc is not None else Vp

        if vc is None:
            raise RuntimeError("Requested covariance matrix is not available.")

        vc = np.asarray(vc, dtype=np.float64).copy()
        if dispersion is not None:
            vc *= float(dispersion) / fit_scale
        return vc

    def _loglik_effective_df(self) -> float:
        """
        mgcv-style effective df used by ``logLik.gam`` / AIC / BIC.

        Mirrors mgcv ``logLik.gam`` in mgcv/R/mgcv.r.
        """
        family_class = str(getattr(self.family, "family_class", "")).lower()
        sc_p = (
            0.0
            if family_class == "general"
            else (1.0 if getattr(self.family, "known_scale", None) is None else 0.0)
        )
        p = float(_edf_total(self)) + sc_p
        edf2 = _edf2(self)
        if edf2 is not None:
            p = float(np.sum(np.asarray(edf2, dtype=np.float64))) + sc_p
        np_max = float(len(np.asarray(_coef_full(self), dtype=np.float64))) + sc_p
        p = min(p, np_max)
        n_theta = getattr(self.family, "n_theta", None)
        if family_class == "extended" and n_theta is not None:
            p += float(n_theta)
        return p

    def _loglik_value_and_effective_df(self) -> tuple[float, float]:
        """
        mgcv ``logLik.gam`` uses two df notions:

        - value uses ``sum(edf) + scale.estimated``
        - attr(df) uses ``sum(edf2) + scale.estimated`` when available
        """
        family_class = str(getattr(self.family, "family_class", "")).lower()
        sc_p = (
            0.0
            if family_class == "general"
            else (1.0 if getattr(self.family, "known_scale", None) is None else 0.0)
        )
        p_val = float(_edf_total(self)) + sc_p
        p_df = p_val
        edf2 = _edf2(self)
        if edf2 is not None:
            p_df = float(np.sum(np.asarray(edf2, dtype=np.float64))) + sc_p
        np_max = float(len(np.asarray(_coef_full(self), dtype=np.float64))) + sc_p
        if p_df > np_max:
            p_df = np_max
        n_theta = getattr(self.family, "n_theta", None)
        if family_class == "extended" and n_theta is not None:
            p_df += float(n_theta)
        return p_val, p_df

    def _object_aic(self) -> float | None:
        """
        Final ``object$aic`` before ``logLik.gam`` post-processing.

        For Gaussian fits, mirror ``gam.fit3.r`` raw-family AIC plus the later
        ``mgcv.r`` `+ 2*sum(object$edf)` update exactly.
        """
        fit_result = _fit_result(self)
        if fit_result is None:
            return None

        family_name = str(getattr(self.family, "name", "")).lower()
        weights = (
            np.ones_like(np.asarray(self.y_, dtype=np.float64), dtype=np.float64)
            if self.prior_weights_ is None
            else np.asarray(self.prior_weights_, dtype=np.float64)
        )
        positive = weights > 0.0
        nobs = float(np.sum(weights[positive]))
        if nobs <= 0.0:
            return None

        if family_name == "gaussian":
            dev = float(getattr(fit_result, "deviance", np.nan))
            if not np.isfinite(dev) or dev <= 0.0:
                return None
            n_row = float(len(weights))
            n_true = getattr(self, "n_true_", None)
            n_eff = (
                None
                if n_true is None
                or not np.isfinite(float(n_true))
                or float(n_true) <= 0.0
                else float(n_true)
            )
            nobs, sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
                weights,
                n_row,
                mp=0.0,
                n_effective_total=n_eff,
            )
            if not np.isfinite(nobs) or nobs <= 0.0 or not np.isfinite(sum_log_scaled):
                return None
            raw_aic = (
                nobs * (np.log(dev / nobs * 2.0 * np.pi) + 1.0)
                + 2.0
                - float(sum_log_scaled)
            )
            return float(raw_aic + 2.0 * float(_edf_total(self)))

        y = np.asarray(self.y_, dtype=np.float64)
        mu = np.asarray(self.predict(X=None, type="response"), dtype=np.float64)
        raw_aic = None

        if family_name == "poisson":
            mu = np.clip(mu, np.finfo(np.float64).tiny, None)
            raw_aic = -2.0 * float(
                np.sum(weights * (y * np.log(mu) - mu - gammaln(y + 1.0)))
            )
        elif family_name == "binomial":
            mu = np.clip(mu, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)
            n = weights
            y_count = np.rint(n * y)
            n_round = np.rint(n)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_comb = (
                    gammaln(n_round + 1.0)
                    - gammaln(y_count + 1.0)
                    - gammaln(n_round - y_count + 1.0)
                )
                log_p = (
                    log_comb
                    + y_count * np.log(mu)
                    + (n_round - y_count) * np.log(1.0 - mu)
                )
                scale = np.where(n > 0.0, weights / np.maximum(n, 1.0), 0.0)
            raw_aic = -2.0 * float(np.sum(scale * log_p))
        elif family_name == "gamma":
            scale = None
            optim_result = getattr(self, "_optim_result", None)
            joint_log_phi = (
                None
                if optim_result is None
                else getattr(optim_result, "joint_log_phi", None)
            )
            if (
                joint_log_phi is not None
                and np.isfinite(float(joint_log_phi))
                and str(getattr(self, "_optim_method", "")).lower() in {"reml", "ml"}
            ):
                scale = float(np.exp(float(joint_log_phi)))
            if scale is None:
                fit_method = str(getattr(self, "smoothing_method", "")).lower()
                if fit_method == "fixed" and getattr(self.family, "known_scale", None) is None:
                    from .._model_state import _coef_column_offset
                    from ..smoothing_selection.criteria.pirls.value import (
                        _solve_gamma_profile_scale,
                    )
                    from ..smoothing_selection.reparam import _static_penalty_null_dim

                    penalty = float(
                        getattr(fit_result, "penalty_quadratic", 0.0) or 0.0
                    )
                    mp = float(
                        _static_penalty_null_dim(self) + _coef_column_offset(self)
                    )
                    init_scale = _fit_scale(self)
                    if init_scale is None or not np.isfinite(float(init_scale)):
                        init_scale = 1.0
                    # mgcv/R/gam.fit3.r::gam.fit3 uses `reml.scale`
                    # rather than the reported Pearson `sig2` in
                    # stats::Gamma()$aic when fixed sp are fitted through
                    # the REML path.
                    scale = float(
                        _solve_gamma_profile_scale(
                            self,
                            y,
                            float(getattr(fit_result, "deviance", np.nan)) + penalty,
                            mp=mp,
                            method="REML",
                            init_scale=float(init_scale),
                        )
                    )
                if scale is None:
                    scale = _fit_scale(self)
            if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0.0:
                return None
            disp = float(scale)
            shape = 1.0 / disp
            mu = np.clip(mu, np.finfo(np.float64).tiny, None)
            y = np.clip(y, np.finfo(np.float64).tiny, None)
            raw_aic = (
                -2.0
                * float(
                    np.sum(
                        weights
                        * (
                            (shape - 1.0) * np.log(y)
                            - y / (mu * disp)
                            - gammaln(shape)
                            - shape * np.log(mu * disp)
                        )
                    )
                )
                + 2.0
            )

        if raw_aic is not None:
            return float(raw_aic + 2.0 * float(_edf_total(self)))

        return None

    def loglik(self) -> float:
        """
        Unpenalized fitted log-likelihood at penalized MLE.

        Mirrors mgcv ``logLik.gam`` value semantics.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        object_aic = self._object_aic()
        if object_aic is not None:
            p_val, _p_df = self._loglik_value_and_effective_df()
            return float(p_val - object_aic / 2.0)

        if getattr(self.family, "family_class", "") == "general":
            fit_result = _fit_result(self)
            if fit_result is not None and fit_result.loglik is not None:
                # General-family fits already store the mgcv-shaped unpenalized
                # log-likelihood in fit space. Recomputing from exported
                # coefficients is wrong when public coefficients have been
                # mapped to a prediction parameterization.
                return float(fit_result.loglik)
            X = np.asarray(_fit_state(self).X, dtype=np.float64)
            jj = [
                np.arange(sl.start, sl.stop, dtype=int)
                for sl in _predictor_full_slices(self)
            ]
            weights = (
                np.ones_like(np.asarray(self.y_, dtype=np.float64), dtype=np.float64)
                if self.prior_weights_ is None
                else np.asarray(self.prior_weights_, dtype=np.float64)
            )
            ll = self.family.ll(
                np.asarray(self.y_, dtype=np.float64),
                X,
                jj,
                np.asarray(_coef_full(self), dtype=np.float64),
                weights,
                offset=self._general_family_offset_list(),
                deriv=0,
            )
            return float(ll["l"])

        y = np.asarray(self.y_, dtype=np.float64)
        mu = np.asarray(self.predict(X=None, type="response"), dtype=np.float64)
        weights = (
            None
            if self.prior_weights_ is None
            else np.asarray(self.prior_weights_, dtype=np.float64)
        )
        dev = float(self.family.deviance(y, mu, weights=weights))
        fit_scale = _fit_scale(self)
        if (
            fit_scale is not None
            and np.isfinite(float(fit_scale))
            and float(fit_scale) > 0.0
        ):
            scale = float(fit_scale)
        elif getattr(self.family, "known_scale", None) is None:
            scale_est = self.family.estimate_dispersion(
                y,
                mu,
                edf=float(_edf_total(self)),
                weights=weights,
            )
            scale = max(float(scale_est), np.finfo(np.float64).tiny)
        else:
            scale = float(self.family.known_scale)
        sat = float(
            self.family.saturated_loglik(
                y,
                weights=weights,
                n=len(y),
                scale=scale,
            )
        )
        return float(sat - dev / (2.0 * scale))

    def aic(self) -> float:
        """mgcv-style conditional AIC based on effective df."""
        object_aic = self._object_aic()
        if object_aic is not None:
            p_val, p_df = self._loglik_value_and_effective_df()
            return float(object_aic + 2.0 * (p_df - p_val))
        return float(-2.0 * self.loglik() + 2.0 * self._loglik_effective_df())

    def bic(self) -> float:
        """BIC using mgcv-style effective df."""
        n_obs = float(len(np.asarray(self.y_, dtype=np.float64)))
        return float(-2.0 * self.loglik() + np.log(n_obs) * self._loglik_effective_df())

    def gam_vcomp(self, *, rescale=True, conf_lev=0.95):
        from ..smoothing_selection import gam_vcomp

        return gam_vcomp(self, rescale=rescale, conf_lev=conf_lev)

    def one_se_rule(self, candidate_indices=None):
        from ..smoothing_selection import one_se_rule

        return one_se_rule(self, candidate_indices=candidate_indices)

    def anova(self, *models, dispersion=None, test=None, freq=False):
        from ..inference import anova_gam

        return anova_gam(
            self,
            *models,
            dispersion=dispersion,
            test=test,
            freq=freq,
        )

"""User-facing mgcv-aligned GAM model."""

import pickle
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ..control import gam_control
from ..data import (
    coerce_formula_predict_inputs,
    coerce_optional_offset,
    coerce_X,
    combine_offsets,
    copy_offset,
)
from ..diagnostics import (
    concurvity,
    gam_check,
    k_check,
    plot_gam,
    print_summary,
    residuals_gam,
    smooth_derivative,
)
from ..families import clone_gam_family, make_gam_family
from ..fit import fit_model_core, select_covariance_matrix
from ..fit.covariance import select_prediction_covariance_matrix
from ..fit.offsets import coerce_offset_array
from ..fit.result_builders import build_gam_result, copy_fit_result
from ..fit.selection import gam_vcomp, one_se_rule, sp_vcov
from ..fit.selection.criteria.ml_reml import resolve_ml_reml_scoring_backend
from ..fit.smoothing_params import expand_smoothing_params_from_log
from ..inference import anova_gam
from ..inference.anova import _edf1_vector
from ..inference.loglik import (
    loglik_effective_df,
    loglik_gam,
    loglik_value_and_effective_df,
    object_aic,
)
from ..model_state import (
    _coef_full,
    _cov_bayes,
    _cov_freq,
    _cov_unconditional,
    _edf_total,
    _fit_scale,
    _fit_state,
    _fitted_eta,
    _fitted_mu,
    _intercept,
    _predictor_full_indices,
    _term_blocks_seq,
)
from ..predict import (
    predict_values,
    prediction_guaranteed_skip_contract,
)
from ..predict.terms import _prediction_term_groups
from ..results.snapshots import build_snapshot
from ..specs.modeling import make_predictor_specs, prepare_formula_inputs
from .persistence import gam_pickle_state, restore_gam_pickle_state
from .session import FitSession


def _normalize_prediction_na_action(value):
    if value is None:
        return None
    key = str(value).lower().replace("_", ".")
    if key.startswith("na."):
        key = key[3:]
    if key not in {"pass", "omit", "exclude", "fail"}:
        raise ValueError(
            "na_action must be one of {'pass', 'omit', 'exclude', 'fail', None}."
        )
    return key


def _normalize_prediction_block_size(value, *, n_rows, explicit_newdata, pred_type):
    if str(pred_type).lower() == "lpmatrix":
        return max(int(n_rows), 1)
    if value is None:
        return 1000 if explicit_newdata else max(int(n_rows), 1)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("block_size must be an integer or None.")
    size = int(value)
    return max(int(n_rows), 1) if size < 1 else size


def _slice_prediction_value(value, rows):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [None if item is None else np.asarray(item)[rows] for item in value]
    return np.asarray(value)[rows]


def _prediction_complete_rows(model, X):
    if isinstance(X, pd.DataFrame):
        if bool(getattr(model, "formula_mode_", False)):
            required = list(getattr(model, "formula_used_columns_", ()) or ())
            columns = [name for name in required if name in X.columns]
        else:
            columns = list(X.columns)
        if not columns:
            return np.ones(len(X), dtype=bool)
        return ~X[columns].isna().any(axis=1).to_numpy(dtype=bool)
    array = np.asarray(X)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return ~pd.isna(array).any(axis=1)


def _prediction_offset_complete_rows(offset, n_rows):
    complete = np.ones(int(n_rows), dtype=bool)
    if offset is None:
        return complete
    values = offset if isinstance(offset, (list, tuple)) else (offset,)
    for value in values:
        if value is None:
            continue
        array = np.asarray(value).reshape(-1)
        if array.shape != (int(n_rows),):
            raise ValueError(
                f"offset must have shape ({int(n_rows)},), got {array.shape}."
            )
        complete &= ~pd.isna(array)
    return complete


def _concatenate_prediction_blocks(blocks):
    first = blocks[0]
    if isinstance(first, tuple):
        return tuple(
            _concatenate_prediction_blocks([block[index] for block in blocks])
            for index in range(len(first))
        )
    return np.concatenate([np.asarray(block) for block in blocks], axis=0)


def _empty_prediction_result(value):
    if isinstance(value, tuple):
        return tuple(_empty_prediction_result(item) for item in value)
    return np.asarray(value)[:0]


def _restore_prediction_na_rows(value, retained_rows, n_rows):
    if isinstance(value, tuple):
        return tuple(
            _restore_prediction_na_rows(item, retained_rows, n_rows) for item in value
        )
    array = np.asarray(value)
    out = np.full((int(n_rows), *array.shape[1:]), np.nan, dtype=np.float64)
    out[np.asarray(retained_rows, dtype=int)] = np.asarray(array, dtype=np.float64)
    return out


_GAM_HPARAM_KEYS = frozenset(
    {
        "k",
        "basis",
        "fit_intercept",
        "max_irls_iter",
        "irls_tol",
        "max_step_halving",
        "smoothing_params",
        "optimize_smoothing",
        "smoothing_method",
        "smoothing_optimizer",
        "sp_log_bounds",
        "score_gamma",
        "covariance",
        "select",
        "knots",
        "xt",
        "min_sp",
        "drop_intercept",
        "formula",
        # Read from hparams outside the constructor:
        "apply_side_conditions",  # fit/design_setup.py
        "trace",  # fit/solvers/general_family/fixed_smoothing.py
        "positive_transform",  # transformed-coefficient solver
        "softplus_beta",  # transformed-coefficient solver
        "softplus_threshold",  # transformed-coefficient solver
        "start",  # transformed-coefficient solver
        "ar1_rho",  # generic Gaussian-identity residual correlation
        "ar_start",  # starts of independent AR1 sections
        "scale",
        "control",
        "nei",
        "coefficient_optimizer",
        "optim_method",
    }
)


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
        unknown = sorted(set(self.hparams) - _GAM_HPARAM_KEYS)
        if unknown:
            # Unknown arguments must fail loudly: silently swallowing them
            # previously made unported mgcv arguments (paraPen=, absorb.cons=,
            # H=, gamma=, ...) no-ops instead of errors.
            raise TypeError(f"Unknown GAM argument(s): {unknown}")

        self.k = int(self.hparams.get("k", 10))
        self.basis = self.hparams.get("basis", "tp")
        self.fit_intercept = bool(self.hparams.get("fit_intercept", True))
        self.control = gam_control(self.hparams.get("control", None))
        self.max_irls_iter = int(self.hparams.get("max_irls_iter", self.control.maxit))
        self.irls_tol = float(self.hparams.get("irls_tol", self.control.epsilon))
        self.max_step_halving = int(
            self.hparams.get("max_step_halving", self.control.mgcv_half)
        )
        self.ar1_rho = float(self.hparams.get("ar1_rho", 0.0))
        if not -1.0 < self.ar1_rho < 1.0:
            raise ValueError("ar1_rho must be strictly between -1 and 1.")
        self.ar_start = self.hparams.get("ar_start", None)
        self.ar_start_ = None
        self.ar1_standardized_residuals_ = None
        self.smoothing_params = self.hparams.get("smoothing_params", None)
        self.optimize_smoothing = bool(self.hparams.get("optimize_smoothing", False))
        self.smoothing_method = str(
            self.hparams.get("smoothing_method", "fixed")
        ).lower()
        self.smoothing_optimizer = str(
            self.hparams.get("smoothing_optimizer", "outer_newton")
        ).lower()
        self._smoothing_optimizer_user_supplied = "smoothing_optimizer" in self.hparams
        self.coefficient_optimizer = str(
            self.hparams.get("coefficient_optimizer", "newton")
        ).lower()
        if self.coefficient_optimizer not in {"newton", "bfgs"}:
            raise ValueError("coefficient_optimizer must be 'newton' or 'bfgs'.")
        self.optim_method = self.hparams.get("optim_method", None)
        self.sp_log_bounds = tuple(self.hparams.get("sp_log_bounds", (-80.0, 20.0)))
        self.score_gamma = float(self.hparams.get("score_gamma", 1.0))
        self.covariance = str(self.hparams.get("covariance", "bayes")).lower()
        self.select = bool(self.hparams.get("select", False))
        self.knots = self.hparams.get("knots", None)
        self.xt = self.hparams.get("xt", None)
        self.min_sp = self.hparams.get("min_sp", None)
        self.scale = float(self.hparams.get("scale", 0.0))
        if not np.isfinite(self.scale):
            raise ValueError("scale must be finite.")
        self.nei = self.hparams.get("nei", None)
        self.drop_intercept = self.hparams.get("drop_intercept", None)
        self.positive_transform = str(
            self.hparams.get("positive_transform", "exp")
        ).lower()
        if self.positive_transform not in {"exp", "softplus"}:
            raise ValueError("positive_transform must be 'exp' or 'softplus'.")
        self.softplus_beta = float(
            self.hparams.get("softplus_beta", self.control.scam_b_notexp)
        )
        self.softplus_threshold = float(
            self.hparams.get("softplus_threshold", self.control.scam_threshold_notexp)
        )
        self.start = self.hparams.get("start", None)

        resolved_family = make_gam_family(family)
        self._family_template = clone_gam_family(resolved_family)
        self.family = clone_gam_family(self._family_template)
        self._apply_configured_scale()

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

    def _apply_configured_scale(self):
        """Apply upstream ``scale=`` semantics to the per-fit family clone."""
        family_name = str(getattr(self.family, "name", "")).lower()
        fixed_one = family_name in {
            "binomial",
            "poisson",
            "negbin",
            "betar",
            "ocat",
        }
        if self.scale > 0.0:
            self.family.known_scale = float(self.scale)
        elif fixed_one:
            self.family.known_scale = 1.0
        else:
            self.family.known_scale = None
        self._optim_method = None
        self._optim_result = None
        self._optim_trace = None
        self._optim_used_gradient = None
        self._optim_used_hessian = None
        self.smoothing_score_ = None
        self.gam_result_ = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def __getstate__(self):
        return gam_pickle_state(self)

    def __setstate__(self, state):
        restore_gam_pickle_state(self, state)

    def save_model(self, path):
        destination = Path(path)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)
        return destination

    @classmethod
    def load_model(cls, path, device="cpu"):
        source = Path(path)
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"{source} contains {type(loaded).__name__}, not {cls.__name__}."
            )
        loaded._device = device
        return loaded

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
        n_pred = len(_predictor_full_indices(self))
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
    ):
        """Fit transactionally with fresh solver and mutable-family state."""
        session = FitSession.begin(self)
        session.working_model._fit_in_place(
            X=X,
            y=y,
            optimize_smoothing=optimize_smoothing,
            smoothing_method=smoothing_method,
            data=data,
            formula=formula,
            offset=offset,
            sample_weight=sample_weight,
            min_sp=min_sp,
            knots=knots,
            drop_intercept=drop_intercept,
        )
        session.commit_to(self)
        return self

    def _fit_in_place(
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
        self._apply_configured_scale()
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

            # `drop_intercept` is applied during formula intent extraction, so
            # the resolved predictor spec -- not the raw parsed formula -- is
            # the canonical owner of the intercept policy.
            fit_intercept = bool(predictor_specs[0].has_intercept)
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

        if self.ar_start is not None:
            if isinstance(self.ar_start, str):
                if formula is None or data is None or self.ar_start not in data.columns:
                    raise ValueError(
                        "ar_start as a column name requires formula data containing that column."
                    )
                ar_start = np.asarray(data[self.ar_start], dtype=bool).reshape(-1)
            else:
                ar_start = np.asarray(self.ar_start, dtype=bool).reshape(-1)
            if ar_start.shape != (len(y_use),):
                raise ValueError(
                    f"ar_start must have shape ({len(y_use)},), got {ar_start.shape}."
                )
            self.ar_start_ = ar_start.copy()
        else:
            self.ar_start_ = None

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
        if self.gam_result_ is None or self.gam_result_.fit_summary is None:
            self.gam_result_ = build_gam_result(self, prefer_cached_summary=False)
        return copy_fit_result(
            self.gam_result_.require_fit_summary(),
            include_covariances=include_covariances,
        )

    def _select_cov(self, cov):

        return select_covariance_matrix(self, cov=cov)

    def _resolve_ml_reml_scoring_backend(self, method="reml"):

        return resolve_ml_reml_scoring_backend(self, method=method)

    def _expand_smoothing_params_from_log(self, log_free_sp):

        return expand_smoothing_params_from_log(self, log_free_sp)

    def _has_tensor_terms(self) -> bool:
        for tb in _term_blocks_seq(self):
            if str(getattr(tb, "term_type", "")).lower() in {
                "tensor_smooth",
                "tensor_interaction",
                "tensor_t2",
            }:
                return True
        return False

    def parity_snapshot(self, X=None, include_covariances=False):
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return build_snapshot(self, X=X, include_covariances=include_covariances)

    def predict(
        self,
        X=None,
        return_se=False,
        cov=None,
        type="response",
        offset=None,
        terms=None,
        exclude=None,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
        unconditional=False,
        iterms_type=1,
    ):

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if bool(unconditional) and cov is not None:
            raise ValueError("cov and unconditional=True cannot be used together.")
        if bool(newdata_guaranteed) and X is None:
            raise ValueError("newdata_guaranteed=True requires explicit newdata.")
        cov_use = cov
        unconditional_core = bool(unconditional)
        if unconditional_core:
            cov_use = select_prediction_covariance_matrix(self, unconditional=True)
            unconditional_core = False
        skip_term_ids = frozenset()
        allowed_missing_features = frozenset()
        if bool(newdata_guaranteed):
            skip_term_ids, allowed_missing_features = (
                prediction_guaranteed_skip_contract(self, terms=terms, exclude=exclude)
            )

        if X is None:
            offset_use = None
            if offset is not None:
                offset_use = self._coerce_api_offset(offset, self.X_.shape[0])
            return predict_values(
                X=None,
                return_se=return_se,
                cov=cov_use,
                type=type,
                offset=offset_use,
                model=self,
                terms=terms,
                exclude=exclude,
                unconditional=unconditional_core,
                iterms_type=iterms_type,
                skip_term_ids=skip_term_ids,
            )

        n_original = len(X)
        action = (
            None
            if bool(newdata_guaranteed)
            else _normalize_prediction_na_action(na_action)
        )
        complete = _prediction_complete_rows(self, X)
        complete &= _prediction_offset_complete_rows(offset, n_original)
        if bool(newdata_guaranteed) and not bool(np.all(complete)):
            raise ValueError(
                "newdata_guaranteed=True requires complete prediction data."
            )
        if action == "fail" and not bool(np.all(complete)):
            raise ValueError("missing values in prediction data")
        restore_missing = action == "pass" and not bool(np.all(complete))
        if action in {"pass", "omit", "exclude"}:
            retained_rows = np.flatnonzero(complete)
            X_input = (
                X.iloc[retained_rows]
                if isinstance(X, pd.DataFrame)
                else np.asarray(X)[retained_rows]
            )
            offset_input = _slice_prediction_value(offset, retained_rows)
        else:
            retained_rows = np.arange(n_original, dtype=int)
            X_input = X
            offset_input = offset
        allow_missing_numeric = action is None

        if len(retained_rows) == 0:
            probe_offset = None
            if offset is not None:
                if isinstance(offset, (list, tuple)):
                    probe_offset = [
                        None if item is None else np.zeros(1, dtype=np.float64)
                        for item in offset
                    ]
                else:
                    probe_offset = np.zeros(1, dtype=np.float64)
            probe = predict_values(
                X=np.asarray(self.X_)[:1],
                return_se=return_se,
                cov=cov_use,
                type=type,
                offset=probe_offset,
                model=self,
                terms=terms,
                exclude=exclude,
                unconditional=unconditional_core,
                iterms_type=iterms_type,
                skip_term_ids=skip_term_ids,
                allow_missing_numeric=allow_missing_numeric,
            )
            empty = _empty_prediction_result(probe)
            return (
                _restore_prediction_na_rows(empty, retained_rows, n_original)
                if restore_missing
                else empty
            )

        if self.formula_mode_:
            X_np, _, offset_formula = coerce_formula_predict_inputs(
                self,
                X_input,
                allowed_missing_features=allowed_missing_features,
                allow_missing_numeric=allow_missing_numeric,
            )
            offset_use = combine_offsets(
                offset_formula,
                self._coerce_api_offset(offset_input, X_np.shape[0]),
            )
        else:
            X_np, _ = coerce_X(
                self, X_input, allow_missing_numeric=allow_missing_numeric
            )
            offset_use = self._coerce_api_offset(offset_input, X_np.shape[0])

        size = _normalize_prediction_block_size(
            block_size,
            n_rows=X_np.shape[0],
            explicit_newdata=True,
            pred_type=type,
        )
        blocks = []
        for start in range(0, X_np.shape[0], size):
            stop = min(start + size, X_np.shape[0])
            blocks.append(
                predict_values(
                    X=X_np[start:stop],
                    return_se=return_se,
                    cov=cov_use,
                    type=type,
                    offset=_slice_prediction_value(offset_use, slice(start, stop)),
                    model=self,
                    terms=terms,
                    exclude=exclude,
                    unconditional=unconditional_core,
                    iterms_type=iterms_type,
                    skip_term_ids=skip_term_ids,
                    allow_missing_numeric=allow_missing_numeric,
                )
            )
        result = _concatenate_prediction_blocks(blocks)
        if restore_missing:
            result = _restore_prediction_na_rows(result, retained_rows, n_original)
        return result

    def predict_terms(
        self,
        X=None,
        offset=None,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        prediction_args = {
            "block_size": block_size,
            "newdata_guaranteed": newdata_guaranteed,
            "na_action": na_action,
        }
        eta = self.predict(X, type="link", offset=offset, **prediction_args)
        terms = self.predict(X, type="terms", offset=offset, **prediction_args)
        out = {"output": eta}
        if self.family.name != "gaussian":
            out["response"] = self.predict(
                X, type="response", offset=offset, **prediction_args
            )
        groups = _prediction_term_groups(self)
        is_multi_predictor = len(_predictor_full_indices(self)) > 1
        for index, group in enumerate(groups):
            term = group["blocks"][0]
            term_key = str(getattr(term, "term_id", "") or group["label"])
            if is_multi_predictor:
                targets = tuple(getattr(term, "predictor_indices", ())) or (
                    int(getattr(term, "predictor_index", 0)),
                )
                for target in targets:
                    out[f"eta{int(target) + 1}:{term_key}"] = terms[:, index]
            else:
                out[term_key] = terms[:, index]

        if is_multi_predictor:
            compiled = self.gam_result_.require_compiled_model()
            coef_full = np.asarray(_coef_full(self), dtype=np.float64)
            intercepts = {
                f"eta{index + 1}": 0.0
                for index in range(len(compiled.predictor_full_indices))
            }
            for component_index, (predictor, predictor_slice) in enumerate(
                zip(
                    compiled.predictors,
                    compiled.predictor_full_slices,
                    strict=True,
                )
            ):
                if bool(predictor.prediction_has_intercept):
                    value = float(coef_full[int(predictor_slice.start)])
                    targets = tuple(
                        int(target) - 1
                        for target in (
                            (getattr(predictor, "metadata", {}) or {}).get(
                                "lpi", (component_index + 1,)
                            )
                            or (component_index + 1,)
                        )
                    )
                    for target in targets:
                        intercepts[f"eta{target + 1}"] += value
            if any(value != 0.0 for value in intercepts.values()):
                out["intercept"] = intercepts
        else:
            if self.fit_intercept:
                out["intercept"] = np.array(_intercept(self), dtype=np.float64)

        has_formula_offset = any(
            name is not None for name in (self.formula_offset_names_ or ())
        )
        if offset is not None or has_formula_offset:
            eta_array = np.asarray(eta, dtype=np.float64)
            reconstruction = np.zeros_like(eta_array, dtype=np.float64)
            if is_multi_predictor:
                for name, value in out.get("intercept", {}).items():
                    reconstruction[:, int(str(name)[3:]) - 1] += float(value)
                for key, value in out.items():
                    if key in {"output", "response", "intercept", "offset"}:
                        continue
                    predictor_name, separator, _term_id = str(key).partition(":")
                    if separator:
                        reconstruction[:, int(predictor_name[3:]) - 1] += np.asarray(
                            value, dtype=np.float64
                        )
                residual = eta_array - reconstruction
                out["offset"] = {
                    f"eta{index + 1}": residual[:, index]
                    for index in range(residual.shape[1])
                }
            else:
                reconstruction = reconstruction + float(out.get("intercept", 0.0))
                for key, value in out.items():
                    if key not in {"output", "response", "intercept", "offset"}:
                        reconstruction += np.asarray(value, dtype=np.float64)
                out["offset"] = eta_array - reconstruction
        return out

    def lpmatrix(
        self,
        X,
        *,
        block_size=None,
        newdata_guaranteed=False,
        na_action="pass",
    ):

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        return self.predict(
            X,
            type="lpmatrix",
            block_size=block_size,
            newdata_guaranteed=newdata_guaranteed,
            na_action=na_action,
        )

    def plot(self, **kwargs):
        """mgcv ``plot.gam``-shaped term plots (mgcv/R/plots.r:1271-1565).

        Accepts the ported plot.gam arguments (``residuals``, ``rug``, ``se``,
        ``pages``, ``select``, ``scale``, ``n``, ``n2``, ``n3``, ``theta``,
        ``phi``, ``jit``, ``xlab``, ``ylab``, ``main``, ``ylim``, ``xlim``,
        ``too_far``, ``shade_col``, ``shift``, ``trans``, ``se_with_mean``,
        ``unconditional``, ``by_resids``, ``scheme``, ``figsize``) and returns
        the prepared plot-data list with the matplotlib figures attached,
        mirroring upstream's invisible ``pd`` return.
        """

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return plot_gam(self, **kwargs)

    def summary(self, *, dispersion=None, freq=False, re_test=True):
        """
        mgcv ``summary.gam``-shaped summary.

        Prints the ``print.summary.gam`` layout and returns the structured
        :class:`~nampy.gam.inference.summary.GAMSummary` object
        (mgcv/R/mgcv.r:3858-4068).
        """

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return print_summary(self, dispersion=dispersion, freq=freq, re_test=re_test)

    def residuals(self, type="deviance", *, setseed=None):

        return residuals_gam(self, type=type, setseed=setseed)

    def ar1_standardized_residuals(self):
        """Return AR(1)-standardized response residuals.

        The values are the response residuals after applying the same square
        root correlation transform used by the fitted observation contract. They
        are available only after fitting with non-zero ``ar1_rho``.
        """

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if self.ar1_rho == 0.0 or self.ar1_standardized_residuals_ is None:
            raise RuntimeError(
                "AR(1)-standardized residuals require a fit with non-zero ar1_rho."
            )
        return np.asarray(self.ar1_standardized_residuals_, dtype=np.float64).copy()

    def derivative(self, X=None, smooth_number=1, deriv=1):
        """Return a term-owned univariate smooth derivative and Bayesian SE."""
        X_use = None
        if X is not None:
            if self.formula_mode_:
                X_use, _, _ = coerce_formula_predict_inputs(self, X)
            else:
                X_use, _ = coerce_X(self, X)
        return smooth_derivative(
            self, X=X_use, smooth_number=smooth_number, deriv=deriv
        )

    def concurvity(self, full=True):

        return concurvity(self, full=full)

    def k_check(self, subsample=5000, n_rep=400, seed=None):

        return k_check(self, subsample=subsample, n_rep=n_rep, seed=seed)

    def gam_check(self, *, type="deviance", k_sample=5000, k_rep=200, seed=None):

        return gam_check(
            self,
            type=type,
            k_sample=k_sample,
            k_rep=k_rep,
            seed=seed,
        )

    def sp_vcov(self, edge_correct=True, reg=1e-3):

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
            if Vp is None or Vf is None:
                raise RuntimeError(
                    "Sandwich covariance requires Bayesian and frequentist covariance."
                )
            B2 = np.zeros_like(Vp) if freq else (Vp - Vf)
            if fit_state is None:
                raise RuntimeError("Sandwich covariance requires fitted solver state.")
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
                    np.asarray(indices, dtype=int)
                    for indices in _predictor_full_indices(self)
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
            selected_vc = Vf
        else:
            selected_vc = Vc if unconditional and Vc is not None else Vp

        if selected_vc is None:
            raise RuntimeError("Requested covariance matrix is not available.")

        vc = np.asarray(selected_vc, dtype=np.float64).copy()
        if dispersion is not None:
            vc *= float(dispersion) / fit_scale
        return vc

    def _loglik_effective_df(self) -> float:
        return loglik_effective_df(self)

    def _loglik_value_and_effective_df(self) -> tuple[float, float]:
        return loglik_value_and_effective_df(self)

    def _object_aic(self) -> float | None:
        return object_aic(self)

    def loglik(self) -> float:
        """Unpenalized fitted log-likelihood at penalized MLE (mgcv logLik.gam)."""
        return loglik_gam(self)

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

    def edf1(self) -> np.ndarray:
        """
        Per-coefficient upper-bound EDF ``2*diag(F) - rowSums(F*F')``.

        Mirrors mgcv ``object$edf1`` (mgcv/R/gam.fit3.r:1022 /
        mgcv/R/gam.fit4.r:1713).
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        return np.asarray(_edf1_vector(self), dtype=np.float64)

    def gam_vcomp(self, *, rescale=True, conf_lev=0.95):

        return gam_vcomp(self, rescale=rescale, conf_lev=conf_lev)

    def one_se_rule(self, candidate_indices=None):

        return one_se_rule(self, candidate_indices=candidate_indices)

    def anova(self, *models, dispersion=None, test=None, freq=False):

        return anova_gam(
            self,
            *models,
            dispersion=dispersion,
            test=test,
            freq=freq,
        )

# basemodels/gam.py
import pickle
import warnings

import numpy as np
import pandas as pd
import torch

from ..configs.gam_config import DefaultGAMConfig
from ..gam.families import make_gam_family
from ..gam.formula import (
    apply_drop_intercept,
    compile_predictor_specs_from_formula,
    extract_formula_data,
    parse_gam_formula,
)
from ..gam.formula.preprocess import (
    apply_formula_preprocess_to_new_data,
    preprocess_formula_predictor_specs,
)
from ..gam.parity import build_parity_snapshot
from ..gam.specs import LinearPredictorSpec, TermSpec
from .basemodel import BaseModel


class GAM(BaseModel):
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
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

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

        self.register_buffer("_device_ref", torch.tensor(0.0), persistent=False)

        self.formula = self.hparams.get("formula", getattr(config, "formula", None))

        self.formula_ = None
        self.formula_mode_ = False
        self.formula_response_name_ = None
        self.formula_used_columns_ = None
        self.formula_preprocess_state_ = None
        self.formula_offset_name_ = None

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
        self._primary_reparam_sp_index_per_term_ = None
        self.min_sp_ = None
        self._reparam_meta = None
        self._reparam_rand_blocks_ = None
        self._reparam_sp_groups_ = None
        self._penalty_ranks = None
        self._penalty_logdet_plus_fixed = None
        self.X_fix_ = None
        self.rank_X_fix_ = None
        self._fix_pivot_keep = None
        self.Z_rand_ = None
        self.n_rand_ = None
        self.ZtZ_rand_ = None
        self.rand_dims_per_term_ = None
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
        self.to(device)
        return self

    def get_device(self):
        return self._device_ref.device

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------

    def _coerce_X(self, X):
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_np = self._dataframe_to_feature_matrix(X)
        else:
            X_np = np.asarray(X)
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)
            if X_np.ndim != 2:
                raise ValueError("X must be a 2D array or DataFrame.")

            feature_names = [f"x{i}" for i in range(X_np.shape[1])]

            # Preserve mixed/object arrays if the caller supplied them.
            if X_np.dtype != object:
                X_num = np.asarray(X_np, dtype=np.float64)
                if not np.isfinite(X_num).all():
                    raise ValueError("X contains NaN or Inf")
                X_np = X_num

        return X_np, feature_names

    def _coerce_optional_offset(self, offset, n_rows, *, name="offset"):
        if offset is None:
            return None
        out = np.asarray(offset, dtype=np.float64).ravel()
        if out.shape != (int(n_rows),):
            raise ValueError(
                f"{name} must have shape ({int(n_rows)},), got {out.shape}."
            )
        if not np.isfinite(out).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return out

    def _combine_offsets(self, *offsets):
        arrs = [np.asarray(o, dtype=np.float64) for o in offsets if o is not None]
        if not arrs:
            return None
        out = np.zeros_like(arrs[0], dtype=np.float64)
        for arr in arrs:
            if arr.shape != out.shape:
                raise ValueError(
                    f"Offset arrays must all have the same shape, got {out.shape} and {arr.shape}."
                )
            out = out + arr
        return out

    def _knots_for_feature(self, feature_name, *, knots=None):
        knots = self.knots if knots is None else knots
        if knots is None:
            return None
        if isinstance(knots, dict):
            return knots.get(str(feature_name), None)
        return knots

    def _knots_for_features(self, feature_names, *, knots=None):
        knots = self.knots if knots is None else knots
        if knots is None:
            return None
        if isinstance(knots, dict):
            vals = [knots.get(str(f), None) for f in feature_names]
            return None if all(v is None for v in vals) else vals
        return knots

    def _dataframe_to_feature_matrix(
        self, X_df: pd.DataFrame, *, allow_missing_non_numeric=False
    ):
        non_numeric = [
            c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])
        ]

        if len(non_numeric) == 0:
            X_np = X_df.to_numpy(dtype=np.float64)
            if not np.isfinite(X_np).all():
                raise ValueError("X contains NaN or Inf")
            return X_np

        for c in X_df.columns:
            s = X_df[c]
            if pd.api.types.is_numeric_dtype(s):
                vals = np.asarray(s, dtype=np.float64)
                if not np.isfinite(vals).all():
                    raise ValueError(f"Numeric column {c!r} contains NaN or Inf.")
            else:
                if s.isna().any() and not bool(allow_missing_non_numeric):
                    raise ValueError(
                        f"Non-numeric column {c!r} contains missing values, "
                        "which are not currently supported in fitting."
                    )

        return X_df.to_numpy(dtype=object)

    def _make_tensor_term(self, spec, *, knots=None):
        if hasattr(spec, "fit") and callable(spec.fit):
            return spec

        if (
            isinstance(spec, (tuple, list))
            and len(spec) >= 2
            and not isinstance(spec, dict)
        ):
            features = list(spec)
            return TermSpec(
                kind="smooth",
                features=tuple(str(f) for f in features),
                by_variable=None,
                basis_options={
                    "special": "te",
                    "bs": self.basis,
                    "k": self.k,
                    "knots": self._knots_for_features(features, knots=knots),
                    "select": bool(self.select),
                },
                smoothing_id=None,
                label=f"te({', '.join(map(str, features))})",
                metadata={},
            )

        if not isinstance(spec, dict):
            raise TypeError(
                "Each tensor term specification must be either a term object, "
                "a tuple/list of features, or a dict."
            )

        features = spec.get("features", None)
        if features is None:
            raise ValueError(
                "Tensor term dicts must contain a 'features' entry, "
                "e.g. {'features': ('x1', 'x2')}"
            )

        kind = str(spec.get("kind", "te")).lower()
        k = spec.get("k", self.k)
        basis = spec.get("basis", self.basis)
        label = spec.get("label", f"{kind}({', '.join(map(str, features))})")
        smoothing_id = spec.get("smoothing_id", None)
        metadata = spec.get("metadata", None)
        by = spec.get("by", None)
        mc = spec.get("mc", None)
        full = bool(spec.get("full", False))
        ord_ = spec.get("ord", None)
        fixed = bool(spec.get("fixed", spec.get("fx", False)))
        select = bool(spec.get("select", self.select))
        sp = spec.get("sp", None)
        term_knots = spec.get("knots", self._knots_for_features(features, knots=knots))

        if kind == "ti" and mc is not None:
            mc_vals = [bool(v) for v in ([mc] if np.isscalar(mc) else mc)]
            if len(mc_vals) == 0:
                raise ValueError("mc must not be empty.")
            if not any(mc_vals):
                warnings.warn(
                    f"{label}: all ti() marginal constraints are turned off (mc={mc_vals}). "
                    "This leaves the term without identifiability constraints unless the rest "
                    "of the model provides them.",
                    stacklevel=2,
                )

        if kind == "t2":
            return TermSpec(
                kind="smooth",
                features=tuple(str(f) for f in features),
                by_variable=by,
                basis_options={
                    "special": kind,
                    "bs": basis,
                    "k": k,
                    "full": full,
                    "ord": ord_,
                    "fx": fixed,
                    "select": select,
                    "sp": sp,
                    "knots": term_knots,
                },
                smoothing_id=(None if smoothing_id is None else str(smoothing_id)),
                label=label,
                metadata=dict(metadata or {}),
            )

        return TermSpec(
            kind="smooth",
            features=tuple(str(f) for f in features),
            by_variable=by,
            basis_options={
                "special": kind,
                "bs": basis,
                "k": k,
                "mc": mc,
                "fx": fixed,
                "select": select,
                "sp": sp,
                "knots": term_knots,
            },
            smoothing_id=(None if smoothing_id is None else str(smoothing_id)),
            label=label,
            metadata=dict(metadata or {}),
        )

    def _make_predictor_specs(self, feature_names, *, knots=None):
        terms = []

        if self.main_effects:
            basis = str(self.basis).lower()
            main_terms = []

            for name in feature_names:
                term_knots = self._knots_for_feature(name, knots=knots)

                if basis in {"cr", "cs", "cc"}:
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            basis_options={
                                "special": "s",
                                "bs": basis,
                                "k": self.k,
                                "sp": None,
                                "fx": False,
                                "select": bool(self.select),
                                "knots": term_knots,
                            },
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        )
                    )
                elif basis == "ps":
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            basis_options={
                                "special": "s",
                                "bs": "ps",
                                "k": self.k,
                                "m": None,
                                "sp": None,
                                "fx": False,
                                "select": bool(self.select),
                                "knots": term_knots,
                            },
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        )
                    )
                elif basis in {"tp", "ts"}:
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            basis_options={
                                "special": "s",
                                "bs": basis,
                                "k": self.k,
                                "m": None,
                                "sp": None,
                                "fx": False,
                                "select": bool(self.select),
                                "knots": term_knots,
                                "xt": None,
                            },
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        )
                    )
                elif basis == "gp":
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            basis_options={
                                "special": "s",
                                "bs": "gp",
                                "k": self.k,
                                "m": None,
                                "sp": None,
                                "fx": False,
                                "select": bool(self.select),
                                "pc": None,
                                "knots": term_knots,
                                "xt": None,
                            },
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        ),
                    )
                elif basis == "re":
                    main_terms.append(
                        TermSpec(
                            kind="smooth",
                            features=(str(name),),
                            by_variable=None,
                            basis_options={
                                "special": "s",
                                "bs": "re",
                                "sp": None,
                                "xt": None,
                            },
                            smoothing_id=None,
                            label=name,
                            metadata={},
                        ),
                    )
                else:
                    raise NotImplementedError(
                        f"Automatic main-effect construction currently supports "
                        f"basis in {{'cr','cs','cc','ps','tp','ts','gp','re'}}, got {self.basis!r}."
                    )

            terms.extend(main_terms)

        has_full_te = False
        has_t2 = False
        if self.tensor_terms is not None:
            for spec in self.tensor_terms:
                term = self._make_tensor_term(spec, knots=knots)
                if getattr(term, "term_type", None) == "tensor_smooth":
                    has_full_te = True
                if getattr(term, "term_type", None) == "tensor_anova":
                    has_t2 = True
                terms.append(term)

        if self.main_effects and has_full_te:
            warnings.warn(
                "Model contains both main-effect smooths and full te() tensor-product terms. "
                "This is identifiable via side conditions, but is typically less stable and less "
                "interpretable than a ti() ANOVA-style decomposition.",
                stacklevel=2,
            )

        if self.main_effects and has_t2:
            warnings.warn(
                "Model contains both separate main-effect smooths and t2() terms. "
                "t2() already contains ANOVA-style lower-order components, so this combination "
                "can create strong overlap in the current framework.",
                stacklevel=2,
            )

        return [LinearPredictorSpec(name="eta", terms=terms)]

    def _prepare_formula_inputs(
        self, data, formula, y=None, knots=None, drop_intercept=None
    ):
        parsed = parse_gam_formula(formula)
        parsed = apply_drop_intercept(parsed, drop_intercept=drop_intercept)

        predictor_specs = compile_predictor_specs_from_formula(
            parsed,
            default_k=self.k,
            default_basis=self.basis,
            default_select=self.select,
            knots=knots,
        )

        predictor_specs, data_work, preprocess_state = (
            preprocess_formula_predictor_specs(
                parsed=parsed,
                predictor_specs=predictor_specs,
                data=data,
            )
        )

        X_np, feature_names, y_out, used_cols, offset_out = extract_formula_data(
            data=data_work,
            parsed=parsed,
            predictor_specs=predictor_specs,
            y=y,
        )
        return (
            parsed,
            predictor_specs,
            X_np,
            feature_names,
            y_out,
            used_cols,
            offset_out,
            preprocess_state,
        )

    def _coerce_formula_predict_inputs(self, X):
        if not self.formula_mode_:
            X_np, feature_names = self._coerce_X(X)
            return X_np, feature_names, None

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Prediction for formula-based GAMs currently requires a pandas DataFrame."
            )

        X_work = apply_formula_preprocess_to_new_data(X, self.formula_preprocess_state_)

        missing = [c for c in self.formula_used_columns_ if c not in X_work.columns]
        if missing:
            raise KeyError(f"Prediction data is missing formula columns: {missing}")

        X_df = X_work[self.formula_used_columns_]
        X_np = self._dataframe_to_feature_matrix(X_df, allow_missing_non_numeric=True)

        offset = None
        if self.formula_offset_name_ is not None:
            if self.formula_offset_name_ not in X_work.columns:
                raise KeyError(
                    f"Prediction data is missing formula offset column: {self.formula_offset_name_!r}"
                )
            if not pd.api.types.is_numeric_dtype(X_work[self.formula_offset_name_]):
                raise NotImplementedError(
                    "Current formula-based prediction supports numeric offsets only. "
                    f"Offset column {self.formula_offset_name_!r} is non-numeric."
                )
            offset = X_work[self.formula_offset_name_].to_numpy(dtype=np.float64)

        return X_np, list(X_df.columns), offset

    def _coerce_feature_matrix(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        return X

    def _coerce_offset(self, offset, n_rows):
        from ..gam.fit.offsets import coerce_offset_array

        return coerce_offset_array(offset, n_rows)

    def _prediction_offset(self, X, offset):
        from ..gam.fit.offsets import resolve_prediction_offset

        return resolve_prediction_offset(self, X, offset)

    def _uses_closed_form_solver(self):
        return bool(getattr(self.family, "supports_closed_form_solve", False))

    def _uses_pirls_solver(self):
        return bool(getattr(self.family, "supports_pirls", False))

    def _available_fit_backends(self):
        from ..gam.fit.backends import available_fit_backends

        return available_fit_backends(self)

    def _resolve_fit_backend(self):
        from ..gam.fit.backends import resolve_fit_backend

        return resolve_fit_backend(self)

    def _supports_smoothing_method(self, method):
        from ..gam.smoothing_selection.optimize import supports_smoothing_method

        return supports_smoothing_method(self, method)

    def _resolve_smoothing_method(self, method):
        from ..gam.smoothing_selection.optimize import resolve_smoothing_method

        return resolve_smoothing_method(self, method)

    def _needs_exact_gaussian_reparameterization(self):
        return (
            self._uses_closed_form_solver()
            and self._can_use_exact_gaussian_ml_reml()
            and any(
                bool(getattr(self.family, attr, False))
                for attr in ("supports_ml", "supports_reml", "supports_laml")
            )
        )

    def _can_use_exact_gaussian_ml_reml(self):
        from ..gam.smoothing_selection.reparam import can_use_exact_gaussian_ml_reml

        return can_use_exact_gaussian_ml_reml(self)

    def _can_use_simple_ml_reml_structure(self):
        from ..gam.smoothing_selection.reparam import can_use_simple_ml_reml_structure

        return can_use_simple_ml_reml_structure(self)

    def _resolve_ml_reml_scoring_backend(self, method="reml"):
        from ..gam.smoothing_selection.criteria import resolve_ml_reml_scoring_backend

        return resolve_ml_reml_scoring_backend(self, method=method)

    def _raise_ml_reml_backend_error(self, method):
        method = str(method).lower()
        backend = self._resolve_ml_reml_scoring_backend(method=method)
        if backend is not None:
            return

        if not bool(getattr(self.family, f"supports_{method}", False)):
            raise NotImplementedError(
                f"Automatic smoothing selection with method={method!r} is not "
                f"supported for family={self.family.name!r}."
            )

        if not self._can_use_simple_ml_reml_structure():
            raise NotImplementedError(
                f"Automatic smoothing selection with method={method!r} is not "
                "currently available for this model configuration. "
                "The current ML/REML backend requires each penalized term to "
                "contribute exactly one primary smooth penalty, plus at most "
                "one null-space penalty on the same term. General overlapping "
                "multi-penalty structures must currently use 'fixed', 'gcv', "
                "or 'ubre' where available."
            )

        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            f"supported for family={self.family.name!r}."
        )

    def _has_tensor_terms(self):
        if self.term_blocks_ is None:
            return False
        return any(
            tb.term_type
            in {
                "tensor_smooth",
                "tensor_interaction",
                "tensor_anova",
            }
            for tb in self.term_blocks_
        )

    def _resolve_min_sp(self, min_sp):
        if self.n_smoothing_params_ is None:
            raise RuntimeError("Design has not been compiled yet.")

        if min_sp is None:
            return np.zeros(self.n_smoothing_params_, dtype=np.float64)

        arr = np.asarray(min_sp, dtype=np.float64).ravel()
        if arr.size == 0 and self.n_smoothing_params_ == 0:
            return np.empty((0,), dtype=np.float64)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("min_sp values must be finite and >= 0.")

        if arr.shape == (self.n_smoothing_params_,):
            return arr.copy()

        if arr.shape == (len(self.penalty_blocks_),):
            out = np.zeros(self.n_smoothing_params_, dtype=np.float64)
            for val, pb in zip(arr, self.penalty_blocks_):
                out[pb.smoothing_index] = max(out[pb.smoothing_index], float(val))
            return out

        raise ValueError(
            f"min_sp must have shape ({self.n_smoothing_params_},) for underlying smoothing "
            f"parameters or ({len(self.penalty_blocks_)},) for total penalties, got {arr.shape}."
        )

    def _resolve_smoothing_params(self, n_smoothing_params):
        sp = self.smoothing_params
        if sp is None:
            sp = np.ones(n_smoothing_params, dtype=np.float64)
        else:
            sp = np.asarray(sp, dtype=np.float64)
            if sp.ndim == 0:
                sp = np.full(n_smoothing_params, float(sp), dtype=np.float64)
            if sp.shape != (n_smoothing_params,):
                raise ValueError(
                    f"smoothing_params must have shape ({n_smoothing_params},), got {sp.shape}"
                )
            sp = sp.copy()

        fixed_mask = np.zeros(n_smoothing_params, dtype=bool)

        override_modes = getattr(self.design_, "smoothing_override_modes", None)
        override_values = getattr(self.design_, "smoothing_override_values", None)

        if override_modes is not None:
            if len(override_modes) != n_smoothing_params:
                raise ValueError(
                    "CompiledPredictor smoothing_override_modes has incompatible length."
                )
            if override_values is None:
                override_values = np.full(n_smoothing_params, np.nan, dtype=np.float64)
            override_values = np.asarray(override_values, dtype=np.float64)
            if override_values.shape != (n_smoothing_params,):
                raise ValueError(
                    "CompiledPredictor smoothing_override_values has incompatible shape."
                )

            for i, mode in enumerate(override_modes):
                if mode is None:
                    continue

                if mode == "fixed":
                    val = float(override_values[i])
                    if not np.isfinite(val) or val < 0:
                        raise ValueError(
                            f"Fixed smoothing parameter override at index {i} "
                            f"must be finite and >= 0, got {val}."
                        )
                    sp[i] = val
                    fixed_mask[i] = True
                elif mode == "estimate":
                    fixed_mask[i] = False
                    if (not np.isfinite(sp[i])) or (sp[i] <= 0):
                        sp[i] = 1.0
                else:
                    raise ValueError(
                        f"Unknown smoothing override mode {mode!r} at index {i}."
                    )

        min_sp = (
            np.zeros(n_smoothing_params, dtype=np.float64)
            if self.min_sp_ is None
            else np.asarray(self.min_sp_, dtype=np.float64)
        )

        if min_sp.shape != (n_smoothing_params,):
            raise ValueError(
                f"min_sp must have shape ({n_smoothing_params},), got {min_sp.shape}"
            )

        if np.any(fixed_mask & (sp < min_sp)):
            raise ValueError(
                "Fixed smoothing parameters must satisfy the configured min_sp lower bounds."
            )

        sp = np.maximum(sp, min_sp)
        free_mask = ~fixed_mask

        if np.any(~np.isfinite(sp[free_mask])) or np.any(sp[free_mask] <= 0):
            raise ValueError("All free smoothing parameters must be finite and > 0.")

        if np.any(~np.isfinite(sp[fixed_mask])) or np.any(sp[fixed_mask] < 0):
            raise ValueError("All fixed smoothing parameters must be finite and >= 0.")

        self.smoothing_fixed_mask_ = fixed_mask
        self.smoothing_override_modes_ = (
            None if override_modes is None else list(override_modes)
        )
        return sp

    def _n_free_smoothing_params(self):
        from ..gam.smoothing_selection.optimize import n_free_smoothing_params

        return n_free_smoothing_params(self)

    def _expand_smoothing_params_from_log(self, log_free_sp):
        from ..gam.smoothing_selection.optimize import expand_smoothing_params_from_log

        return expand_smoothing_params_from_log(self, log_free_sp)

    def _compile_designs(self, X, feature_names):
        from ..gam.constraints.identifiability import apply_global_side_conditions
        from ..gam.design.compiler import compile_predictor_designs

        compiled = compile_predictor_designs(
            X=X,
            feature_names=feature_names,
            predictor_specs=self.predictor_specs,
        )

        if bool(self.hparams.get("apply_side_conditions", True)):
            adjusted = []
            reports = []
            for d in compiled:
                d_adj, rep = apply_global_side_conditions(
                    d,
                    fit_intercept=self.fit_intercept,
                    tol=float(self.hparams.get("side_condition_tol", 1e-10)),
                    warn=True,
                )
                adjusted.append(d_adj)
                reports.append(rep)
            self.predictor_designs = adjusted
            self.side_condition_reports_ = reports
        else:
            self.predictor_designs = compiled
            self.side_condition_reports_ = None

        self.family.validate_predictor_count(len(self.predictor_designs))

        if len(self.predictor_designs) != 1:
            raise NotImplementedError(
                "Current core supports one active linear predictor in fitting. "
                "The predictor architecture is in place for later extension "
                "to multiple distribution parameters."
            )

        self.design_ = self.predictor_designs[0]
        self.Z = self.design_.design_matrix
        self.ZTZ = self.Z.T @ self.Z
        self.n_coef_ = self.design_.n_coef
        self.term_blocks_ = self.design_.compiled_terms
        self.penalty_blocks_ = self.design_.compiled_penalties
        self.n_smoothing_params_ = self.design_.n_smoothing_params
        self.min_sp_ = self._resolve_min_sp(self.min_sp)
        self.smoothing_params = self._resolve_smoothing_params(self.n_smoothing_params_)

        if self._needs_exact_gaussian_reparameterization():
            self._build_gaussian_reparameterized_system()
        else:
            self._reparam_meta = None
            self._reparam_rand_blocks_ = None
            self._reparam_sp_groups_ = None
            self._penalty_ranks = None
            self._penalty_logdet_plus_fixed = None
            self.X_fix_ = None
            self.rank_X_fix_ = None
            self._fix_pivot_keep = None
            self.Z_rand_ = None
            self.n_rand_ = None
            self.ZtZ_rand_ = None
            self.rand_dims_per_term_ = None
            self._primary_reparam_sp_index_per_term_ = None

    def _one_penalty_per_term_matrices(self):
        penalties = []
        for tb in self.term_blocks_:
            matches = [
                pb for pb in self.penalty_blocks_ if pb.coef_slice == tb.coef_slice
            ]
            if len(matches) != 1:
                raise NotImplementedError(
                    "Current PIRLS path assumes one penalty per term. "
                    "Multi-penalty terms are planned for later phases."
                )
            penalties.append(matches[0].matrix)
        return penalties

    def _assemble_penalty_matrix(self, smoothing_params):
        smoothing_params = np.asarray(smoothing_params, dtype=np.float64)
        if smoothing_params.shape != (self.n_smoothing_params_,):
            raise ValueError(
                f"Expected {self.n_smoothing_params_} smoothing parameters, "
                f"got shape {smoothing_params.shape}."
            )

        P = np.zeros((self.n_coef_, self.n_coef_), dtype=np.float64)
        for pb in self.penalty_blocks_:
            sl = pb.coef_slice
            lam = float(smoothing_params[pb.smoothing_index])
            P[sl, sl] += lam * pb.matrix
        return P

    def _solve_gaussian_given_smoothing(self, y, smoothing_params):
        from ..gam.fit.solvers.gaussian_exact import solve_gaussian_fit

        return solve_gaussian_fit(self, y, smoothing_params)

    def gcv_score(self, y, log_smoothing_params):
        from ..gam.smoothing_selection.criteria import gcv_score_gaussian

        return gcv_score_gaussian(self, y, log_smoothing_params)

    def _criterion_gcv_gaussian(self, y, log_sp):
        from ..gam.smoothing_selection.criteria import criterion_gcv_gaussian

        return criterion_gcv_gaussian(self, y, log_sp)

    def _build_gaussian_reparameterized_system(self):
        from ..gam.smoothing_selection.reparam import (
            build_gaussian_reparameterized_system,
        )

        return build_gaussian_reparameterized_system(self)

    def _build_penalty_reparameterized_system(self):
        from ..gam.smoothing_selection.reparam import (
            build_penalty_reparameterized_system,
        )

        return build_penalty_reparameterized_system(self)

    def _criterion_ml_reml_exact(self, y, log_sp, method):
        from ..gam.smoothing_selection.criteria import criterion_ml_reml_exact

        return criterion_ml_reml_exact(self, y, log_sp, method)

    def _criterion_ml_reml(self, y, log_sp, method):
        from ..gam.smoothing_selection.criteria import criterion_ml_reml

        return criterion_ml_reml(self, y, log_sp, method)

    def _solve_pirls_given_smoothing(self, y, smoothing_params):
        from ..gam.fit.solvers.pirls import solve_pirls_fit

        return solve_pirls_fit(self, y, smoothing_params)

    def _criterion_gcv_pirls(self, y, log_sp):
        from ..gam.smoothing_selection.criteria import criterion_gcv_pirls

        return criterion_gcv_pirls(self, y, log_sp)

    def _criterion_ubre_pirls(self, y, log_sp):
        from ..gam.smoothing_selection.criteria import criterion_ubre_pirls

        return criterion_ubre_pirls(self, y, log_sp)

    def _criterion(self, y, log_sp, method="gcv"):
        from ..gam.smoothing_selection.criteria import criterion_value

        return criterion_value(self, y, log_sp, method=method)

    def _criterion_gradient(self, y, log_sp, method="gcv"):
        from ..gam.smoothing_selection.criteria import criterion_gradient

        return criterion_gradient(self, y, log_sp, method=method)

    def _criterion_hessian(self, y, log_sp, method="gcv"):
        from ..gam.smoothing_selection.criteria import criterion_hessian

        return criterion_hessian(self, y, log_sp, method=method)

    def optimize_smoothing_params(
        self,
        y,
        initial_smoothing_params=None,
        method="gcv",
        optimizer="lbfgsb",
    ):
        from ..gam.smoothing_selection.optimize import optimize_smoothing_params

        return optimize_smoothing_params(
            self,
            y=y,
            initial_smoothing_params=initial_smoothing_params,
            method=method,
            optimizer=optimizer,
        )

    def _build_fit_result(self):
        from ..gam.fit.results import GAMFitResult, TermFitResult

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        term_results = []
        for i, tb in enumerate(self.term_blocks_):
            sp_vals = [float(self.smoothing_params[j]) for j in tb.smoothing_indices]

            deleted = []
            if tb.deleted_columns is not None:
                deleted = [
                    int(v) for v in np.asarray(tb.deleted_columns, dtype=int).tolist()
                ]

            kept = []
            if tb.kept_columns is not None:
                kept = [int(v) for v in np.asarray(tb.kept_columns, dtype=int).tolist()]

            term_results.append(
                TermFitResult(
                    label=tb.label,
                    term_type=tb.term_type,
                    basis_name=tb.basis_name,
                    coef_slice=(int(tb.coef_slice.start), int(tb.coef_slice.stop)),
                    n_coef=int(tb.coef_slice.stop - tb.coef_slice.start),
                    edf=(
                        float(self.edf_by_term_[i])
                        if self.edf_by_term_ is not None
                        else None
                    ),
                    smoothing_indices=[int(v) for v in tb.smoothing_indices],
                    smoothing_ids=list(tb.smoothing_ids),
                    smoothing_values=sp_vals,
                    deleted_columns=deleted,
                    kept_columns=kept,
                    metadata=dict(tb.metadata),
                )
            )

        return GAMFitResult(
            family_name=self.family.name,
            link_name=self.family.link_name,
            criterion_name=self._optim_method,
            criterion_value=self.smoothing_score_,
            coef_full=np.asarray(self.coef_full_, dtype=np.float64).copy(),
            intercept=float(self.intercept_),
            smoothing_params=np.asarray(self.smoothing_params, dtype=np.float64).copy(),
            edf_total=float(self.edf_),
            edf_by_term=np.asarray(self.edf_by_term_, dtype=np.float64).copy(),
            trace_H=float(self.trace_H_),
            scale=float(self.scale_),
            rss=None if self.rss_ is None else float(self.rss_),
            deviance=float(self.deviance_),
            cov_bayes=(
                None
                if self.Vp_ is None
                else np.asarray(self.Vp_, dtype=np.float64).copy()
            ),
            cov_freq=(
                None
                if self.Vf_ is None
                else np.asarray(self.Vf_, dtype=np.float64).copy()
            ),
            side_condition_reports=(
                None
                if self.side_condition_reports_ is None
                else list(self.side_condition_reports_)
            ),
            term_results=term_results,
            metadata={
                "n_samples": int(self.n_samples_),
                "n_coef": int(self.n_coef_),
                "fit_intercept": bool(self.fit_intercept),
                "covariance_default": str(self.covariance),
                "score_gamma": float(self.score_gamma),
                "has_offset": bool(self.offset_train_ is not None),
            },
        )

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
            self.formula_used_columns_ = list(used_cols)
            self.formula_offset_name_ = parsed.predictors[0].offset_name
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
            self.formula_used_columns_ = None
            self.formula_offset_name_ = None
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
        from ..gam.predict import predict_term_contributions

        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            offset_use = None
            if offset is not None:
                offset_use = self._coerce_optional_offset(offset, self.X_.shape[0])
            return predict_term_contributions(self, X=None, offset=offset_use)

        if self.formula_mode_:
            X_np, _, offset_formula = self._coerce_formula_predict_inputs(X)
            offset_use = self._combine_offsets(
                offset_formula,
                self._coerce_optional_offset(offset, X_np.shape[0]),
            )
        else:
            X_np, _ = self._coerce_X(X)
            offset_use = self._coerce_optional_offset(offset, X_np.shape[0])

        return predict_term_contributions(self, X=X_np, offset=offset_use)

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

# gam/model/gam_solve.py
"""Solver dispatch, smoothing resolution, design compilation, and fit-result helpers."""
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .._model_state import _require_fitted


class _GAMSolveMixin:
    # ------------------------------------------------------------------
    # Solver / backend dispatch
    # ------------------------------------------------------------------

    def _uses_closed_form_solver(self):
        return bool(getattr(self.family, "supports_closed_form_solve", False))

    def _uses_pirls_solver(self):
        return bool(getattr(self.family, "supports_pirls", False))

    def _available_fit_backends(self):
        from ..fit.backends import available_fit_backends

        return available_fit_backends(self)

    def _resolve_fit_backend(self):
        from ..fit.backends import resolve_fit_backend

        return resolve_fit_backend(self)

    def _supports_smoothing_method(self, method):
        from ..smoothing_selection.optimize import supports_smoothing_method

        return supports_smoothing_method(self, method)

    def _resolve_smoothing_method(self, method):
        from ..smoothing_selection.optimize import resolve_smoothing_method

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
        from ..smoothing_selection.reparam import can_use_exact_gaussian_ml_reml

        return can_use_exact_gaussian_ml_reml(self)

    def _can_use_simple_ml_reml_structure(self):
        from ..smoothing_selection.reparam import can_use_simple_ml_reml_structure

        return can_use_simple_ml_reml_structure(self)

    def _resolve_ml_reml_scoring_backend(self, method="reml"):
        from ..smoothing_selection.criteria import resolve_ml_reml_scoring_backend

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

    # ------------------------------------------------------------------
    # Smoothing parameter resolution
    # ------------------------------------------------------------------

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
        elif isinstance(sp, Mapping):
            out = np.ones(n_smoothing_params, dtype=np.float64)
            group_map = (
                {}
                if getattr(self, "design_", None) is None
                else dict(self.design_.metadata.get("s_id_to_sp_indices", {}) or {})
            )
            unknown = sorted(str(key) for key in sp.keys() if str(key) not in group_map)
            if unknown:
                raise ValueError(
                    f"Unknown smoothing id(s) in smoothing_params: {unknown}."
                )

            for key, value in sp.items():
                indices = list(group_map[str(key)])
                vals = np.asarray(value, dtype=np.float64).ravel()
                if vals.ndim == 0 or vals.size == 1:
                    if len(indices) != 1:
                        raise ValueError(
                            f"smoothing_params[{key!r}] must provide {len(indices)} "
                            "values for this multi-penalty smoothing id."
                        )
                    out[indices[0]] = float(vals.reshape(-1)[0])
                    continue
                if vals.shape != (len(indices),):
                    raise ValueError(
                        f"smoothing_params[{key!r}] must have shape ({len(indices)},), "
                        f"got {vals.shape}."
                    )
                out[np.asarray(indices, dtype=int)] = vals
            sp = out
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
        from ..smoothing_selection.optimize import n_free_smoothing_params

        return n_free_smoothing_params(self)

    def _expand_smoothing_params_from_log(self, log_free_sp):
        from ..smoothing_selection.optimize import expand_smoothing_params_from_log

        return expand_smoothing_params_from_log(self, log_free_sp)

    # ------------------------------------------------------------------
    # Design compilation
    # ------------------------------------------------------------------

    def _compile_designs(self, X, feature_names):
        from ..constraints.identifiability import apply_global_side_conditions
        from ..design.compiler import compile_predictor_designs

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

        if len(self.predictor_designs) == 1:
            self.design_ = self.predictor_designs[0]
            self.Z = self.design_.design_matrix
            self.ZTZ = self.Z.T @ self.Z
            self.n_coef_ = self.design_.n_coef
            self.term_blocks_ = self.design_.compiled_terms
            self.penalty_blocks_ = self.design_.compiled_penalties
            self.n_smoothing_params_ = self.design_.n_smoothing_params
            self._predictor_full_slices_ = [
                slice(
                    0,
                    int(self.design_.n_coef) + (1 if bool(self.design_.has_intercept) else 0),
                )
            ]
            self._coef_reduced_to_full_idx = np.arange(
                int(self.n_coef_), dtype=int
            ) + (1 if bool(self.design_.has_intercept) else 0)
        else:
            global_terms = []
            global_penalties = []
            combined_blocks = []
            combined_map = {}
            override_modes = []
            override_values = []
            predictor_full_slices = []
            reduced_to_full = []
            coef_shift = 0
            sp_shift = 0
            full_shift = 0

            for pred in self.predictor_designs:
                combined_blocks.append(np.asarray(pred.design_matrix, dtype=np.float64))

                if bool(pred.has_intercept):
                    reduced_to_full.extend(
                        list(
                            np.arange(
                                full_shift + 1,
                                full_shift + 1 + int(pred.n_coef),
                                dtype=int,
                            )
                        )
                    )
                    predictor_full_slices.append(
                        slice(full_shift, full_shift + int(pred.n_coef) + 1)
                    )
                    full_shift += int(pred.n_coef) + 1
                else:
                    reduced_to_full.extend(
                        list(np.arange(full_shift, full_shift + int(pred.n_coef), dtype=int))
                    )
                    predictor_full_slices.append(slice(full_shift, full_shift + int(pred.n_coef)))
                    full_shift += int(pred.n_coef)

                for term in pred.compiled_terms:
                    global_terms.append(
                        replace(
                            term,
                            coef_slice=slice(
                                coef_shift + int(term.coef_slice.start),
                                coef_shift + int(term.coef_slice.stop),
                            ),
                            smoothing_indices=[
                                sp_shift + int(v) for v in getattr(term, "smoothing_indices", [])
                            ],
                            smoothing_ids=[
                                f"{pred.name}:{sid}" if sid is not None else None
                                for sid in getattr(term, "smoothing_ids", [])
                            ],
                        )
                    )
                for pb in pred.compiled_penalties:
                    global_penalties.append(
                        replace(
                            pb,
                            coef_slice=slice(
                                coef_shift + int(pb.coef_slice.start),
                                coef_shift + int(pb.coef_slice.stop),
                            ),
                            smoothing_index=sp_shift + int(pb.smoothing_index),
                            smoothing_id=(
                                None
                                if pb.smoothing_id is None
                                else f"{pred.name}:{pb.smoothing_id}"
                            ),
                        )
                    )
                for sid, idxs in (pred.metadata.get("s_id_to_sp_indices", {}) or {}).items():
                    combined_map[f"{pred.name}:{sid}"] = [sp_shift + int(i) for i in idxs]
                override_modes.extend(list(pred.smoothing_override_modes or []))
                if pred.smoothing_override_values is not None:
                    override_values.extend(
                        list(np.asarray(pred.smoothing_override_values, dtype=np.float64))
                    )
                coef_shift += int(pred.n_coef)
                sp_shift += int(pred.n_smoothing_params)

            self.design_ = SimpleNamespace(
                metadata={"s_id_to_sp_indices": combined_map},
                smoothing_override_modes=list(override_modes),
                smoothing_override_values=np.asarray(override_values, dtype=np.float64),
            )
            self.Z = (
                np.column_stack(combined_blocks)
                if combined_blocks
                else np.empty((X.shape[0], 0), dtype=np.float64)
            )
            self.ZTZ = self.Z.T @ self.Z
            self.n_coef_ = coef_shift
            self.term_blocks_ = tuple(global_terms)
            self.penalty_blocks_ = tuple(global_penalties)
            self.n_smoothing_params_ = sp_shift
            self._predictor_full_slices_ = predictor_full_slices
            self._coef_reduced_to_full_idx = np.asarray(reduced_to_full, dtype=int)

        self.min_sp_ = self._resolve_min_sp(self.min_sp)
        self.smoothing_params = self._resolve_smoothing_params(self.n_smoothing_params_)

        if self._needs_exact_gaussian_reparameterization():
            self._build_gaussian_reparameterized_system()
            self.sl_blocks_ = (
                None
                if self.reparam_state_ is None
                else list(self.reparam_state_.sl_blocks or [])
            )
        else:
            self.reparam_state_ = None
            self.sl_blocks_ = None

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

    # ------------------------------------------------------------------
    # Solver entry points
    # ------------------------------------------------------------------

    def _solve_gaussian_given_smoothing(self, y, smoothing_params):
        from ..fit.solvers.gaussian_exact import solve_gaussian_fit

        return solve_gaussian_fit(
            self,
            y,
            smoothing_params,
            weights=self.prior_weights_,
        )

    def gcv_score(self, y, log_smoothing_params):
        from ..smoothing_selection.criteria import gcv_score_gaussian

        return gcv_score_gaussian(self, y, log_smoothing_params)

    def _criterion_gcv_gaussian(self, y, log_sp):
        from ..smoothing_selection.criteria import criterion_gcv_gaussian

        return criterion_gcv_gaussian(self, y, log_sp)

    def _build_gaussian_reparameterized_system(self):
        from ..smoothing_selection.reparam import (
            build_gaussian_reparameterized_system,
        )

        return build_gaussian_reparameterized_system(self)

    def _build_penalty_reparameterized_system(self):
        from ..smoothing_selection.reparam import (
            build_penalty_reparameterized_system,
        )

        return build_penalty_reparameterized_system(self)

    def _criterion_ml_reml_exact(self, y, log_sp, method):
        from ..smoothing_selection.criteria import criterion_ml_reml_exact

        return criterion_ml_reml_exact(self, y, log_sp, method)

    def _criterion_ml_reml(self, y, log_sp, method):
        from ..smoothing_selection.criteria import criterion_ml_reml

        return criterion_ml_reml(self, y, log_sp, method)

    def _solve_pirls_given_smoothing(self, y, smoothing_params):
        from ..fit.solvers.pirls import solve_pirls_fit

        return solve_pirls_fit(
            self,
            y,
            smoothing_params,
            weights=self.prior_weights_,
        )

    def _criterion_gcv_pirls(self, y, log_sp):
        from ..smoothing_selection.criteria import criterion_gcv_pirls

        return criterion_gcv_pirls(self, y, log_sp)

    def _criterion_ubre_pirls(self, y, log_sp):
        from ..smoothing_selection.criteria import criterion_ubre_pirls

        return criterion_ubre_pirls(self, y, log_sp)

    def _criterion(self, y, log_sp, method="gcv"):
        from ..smoothing_selection.criteria import criterion_value

        return criterion_value(self, y, log_sp, method=method)

    def _criterion_gradient(self, y, log_sp, method="gcv"):
        from ..smoothing_selection.criteria import criterion_gradient

        return criterion_gradient(self, y, log_sp, method=method)

    def _criterion_hessian(self, y, log_sp, method="gcv"):
        from ..smoothing_selection.criteria import criterion_hessian

        return criterion_hessian(self, y, log_sp, method=method)

    def optimize_smoothing_params(
        self,
        y,
        initial_smoothing_params=None,
        method="gcv",
        optimizer="lbfgsb",
    ):
        from ..smoothing_selection.optimize import optimize_smoothing_params

        return optimize_smoothing_params(
            self,
            y=y,
            initial_smoothing_params=initial_smoothing_params,
            method=method,
            optimizer=optimizer,
        )

    # ------------------------------------------------------------------
    # Fit result assembly
    # ------------------------------------------------------------------

    def _build_fit_result(self):
        from ..fit.postprocess.gaussian_smoothness_postprocess import (
            refresh_gaussian_ml_reml_score_from_fit_state,
        )
        from ..fit.results import GAMFitResult, TermFitResult

        _require_fitted(self)
        if str(getattr(self, "_optim_method", "")).lower() in {"reml", "ml"}:
            refresh_gaussian_ml_reml_score_from_fit_state(self, self.y_)

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

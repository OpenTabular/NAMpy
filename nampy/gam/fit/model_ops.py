"""Internal model-facing fit helpers used by the GAM facade and engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from .._model_state import (
    _coef_full,
    _compiled_metadata,
    _compiled_model,
    _cov_bayes,
    _cov_freq,
    _deviance,
    _edf_by_term,
    _edf_total,
    _fit_core_solution,
    _fit_scale,
    _fit_summary,
    _intercept,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
    _predictor_designs,
    _require_fitted,
    _rss,
    _term_blocks_seq,
    _trace_H,
)
from ..results import GAMResult


def uses_closed_form_solver(model):
    return bool(getattr(model.family, "supports_closed_form_solve", False))


def uses_pirls_solver(model):
    return bool(getattr(model.family, "supports_pirls", False))


def can_use_exact_gaussian_ml_reml(model):
    from ..smoothing_selection.reparam import can_use_exact_gaussian_ml_reml

    return can_use_exact_gaussian_ml_reml(model)


def can_use_simple_ml_reml_structure(model):
    from ..smoothing_selection.reparam import can_use_simple_ml_reml_structure

    return can_use_simple_ml_reml_structure(model)


def needs_exact_gaussian_reparameterization(model):
    return uses_closed_form_solver(model) and can_use_exact_gaussian_ml_reml(model) and any(
        bool(getattr(model.family, attr, False))
        for attr in ("supports_ml", "supports_reml", "supports_laml")
    )


def resolve_ml_reml_scoring_backend(model, method="reml"):
    from ..selection import resolve_ml_reml_scoring_backend as _resolve

    return _resolve(model, method=method)


def raise_ml_reml_backend_error(model, method):
    method = str(method).lower()
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend is not None:
        return
    if not bool(getattr(model.family, f"supports_{method}", False)):
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            f"supported for family={model.family.name!r}."
        )
    if not can_use_simple_ml_reml_structure(model):
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            "currently available for this model configuration. "
            "The current ML/REML backend still rejects penalty layouts "
            "with null-space penalties coupling disconnected primary "
            "penalty components. Use 'fixed', 'gcv', or 'ubre' where "
            "available for those cases."
        )
    raise NotImplementedError(
        f"Automatic smoothing selection with method={method!r} is not "
        f"supported for family={model.family.name!r}."
    )


def supports_smoothing_method(model, method):
    from ..selection import supports_smoothing_method as _supports

    return _supports(model, method)


def resolve_smoothing_method(model, method):
    from ..selection import resolve_smoothing_method as _resolve

    return _resolve(model, method)


def resolve_min_sp(model, min_sp):
    n_smoothing_params = _n_smoothing_params(model)
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    if n_smoothing_params == 0 and _compiled_model(model) is None:
        raise RuntimeError("Design has not been compiled yet.")

    if min_sp is None:
        return np.zeros(n_smoothing_params, dtype=np.float64)

    arr = np.asarray(min_sp, dtype=np.float64).ravel()
    if arr.size == 0 and n_smoothing_params == 0:
        return np.empty((0,), dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("min_sp values must be finite and >= 0.")

    if arr.shape == (n_smoothing_params,):
        return arr.copy()
    if arr.shape == (len(penalty_blocks),):
        out = np.zeros(n_smoothing_params, dtype=np.float64)
        for val, pb in zip(arr, penalty_blocks):
            out[pb.smoothing_index] = max(out[pb.smoothing_index], float(val))
        return out
    raise ValueError(
        f"min_sp must have shape ({n_smoothing_params},) for underlying smoothing "
        f"parameters or ({len(penalty_blocks)},) for total penalties, got {arr.shape}."
    )


def resolve_smoothing_params(model, n_smoothing_params):
    sp = model.smoothing_params
    if sp is None:
        sp = np.ones(n_smoothing_params, dtype=np.float64)
    elif isinstance(sp, Mapping):
        out = np.ones(n_smoothing_params, dtype=np.float64)
        group_map = dict(_compiled_metadata(model).get("s_id_to_sp_indices", {}) or {})
        unknown = sorted(str(key) for key in sp.keys() if str(key) not in group_map)
        if unknown:
            raise ValueError(f"Unknown smoothing id(s) in smoothing_params: {unknown}.")
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
    compiled_model = _compiled_model(model)
    override_modes = (
        None if compiled_model is None else getattr(compiled_model, "smoothing_override_modes", None)
    )
    override_values = (
        None if compiled_model is None else getattr(compiled_model, "smoothing_override_values", None)
    )
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
                raise ValueError(f"Unknown smoothing override mode {mode!r} at index {i}.")

    min_sp = (
        np.zeros(n_smoothing_params, dtype=np.float64)
        if model.min_sp_ is None
        else np.asarray(model.min_sp_, dtype=np.float64)
    )
    if min_sp.shape != (n_smoothing_params,):
        raise ValueError(f"min_sp must have shape ({n_smoothing_params},), got {min_sp.shape}")
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

    model.smoothing_fixed_mask_ = fixed_mask
    model.smoothing_override_modes_ = None if override_modes is None else list(override_modes)
    return sp


def n_free_smoothing_params(model):
    from ..selection import n_free_smoothing_params as _count

    return _count(model)


def expand_smoothing_params_from_log(model, log_free_sp):
    from ..smoothing_selection.optimize import expand_smoothing_params_from_log

    return expand_smoothing_params_from_log(model, log_free_sp)


def compile_designs(model, X, feature_names):
    from ..compiler.compile_model import compile_model

    compiled_model = compile_model(
        X=X,
        feature_names=feature_names,
        predictor_specs=model.predictor_specs,
        fit_intercept=model.fit_intercept,
        apply_side_conditions=bool(model.hparams.get("apply_side_conditions", True)),
        side_condition_tol=float(model.hparams.get("side_condition_tol", 1e-10)),
    )
    model.compiled_model_ = compiled_model
    model.side_condition_reports_ = (
        None
        if compiled_model.side_condition_reports is None
        else list(compiled_model.side_condition_reports)
    )
    model.family.validate_predictor_count(len(_predictor_designs(model)))
    model._coef_reduced_to_full_idx = np.asarray(
        compiled_model.coef_reduced_to_full_idx,
        dtype=int,
    )
    model.min_sp_ = resolve_min_sp(model, model.min_sp)
    model.smoothing_params = resolve_smoothing_params(model, _n_smoothing_params(model))

    if needs_exact_gaussian_reparameterization(model):
        build_gaussian_reparameterized_system(model)
        model.sl_blocks_ = (
            None if model.reparam_state_ is None else list(model.reparam_state_.sl_blocks or [])
        )
    else:
        model.reparam_state_ = None
        model.sl_blocks_ = None


def one_penalty_per_term_matrices(model):
    penalties = []
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    for tb in _term_blocks_seq(model):
        matches = [pb for pb in penalty_blocks if pb.coef_slice == tb.coef_slice]
        if not matches:
            penalties.append(np.zeros((tb.basis_train.shape[1], tb.basis_train.shape[1])))
            continue
        P_term = np.zeros_like(np.asarray(matches[0].matrix, dtype=np.float64))
        for pb in matches:
            P_term += np.asarray(pb.matrix, dtype=np.float64)
        penalties.append(P_term)
    return penalties


def assemble_penalty_matrix(model, smoothing_params):
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64)
    n_coef = _n_coef(model)
    n_smoothing_params = _n_smoothing_params(model)
    if smoothing_params.shape != (n_smoothing_params,):
        raise ValueError(
            f"Expected {n_smoothing_params} smoothing parameters, "
            f"got shape {smoothing_params.shape}."
        )
    P = np.zeros((n_coef, n_coef), dtype=np.float64)
    for pb in _penalty_blocks_seq(model):
        sl = pb.coef_slice
        lam = float(smoothing_params[pb.smoothing_index])
        P[sl, sl] += lam * pb.matrix
    return P


def build_gaussian_reparameterized_system(model):
    from ..smoothing_selection.reparam import build_gaussian_reparameterized_system

    return build_gaussian_reparameterized_system(model)


def build_penalty_reparameterized_system(model):
    from ..smoothing_selection.reparam import build_penalty_reparameterized_system

    return build_penalty_reparameterized_system(model)


def solve_gaussian_given_smoothing(model, y, smoothing_params):
    from ..engine import solve_gaussian_fit

    return solve_gaussian_fit(
        model,
        y,
        smoothing_params,
        weights=model.prior_weights_,
    )


def solve_pirls_given_smoothing(model, y, smoothing_params):
    from ..engine import solve_pirls_fit

    return solve_pirls_fit(
        model,
        y,
        smoothing_params,
        weights=model.prior_weights_,
    )


def criterion_value(model, y, log_sp, method="gcv"):
    from ..selection import criterion_value as _criterion_value

    return _criterion_value(model, y, log_sp, method=method)


def criterion_gradient(model, y, log_sp, method="gcv"):
    from ..selection import criterion_gradient as _criterion_gradient

    return _criterion_gradient(model, y, log_sp, method=method)


def criterion_hessian(model, y, log_sp, method="gcv"):
    from ..selection import criterion_hessian as _criterion_hessian

    return _criterion_hessian(model, y, log_sp, method=method)


def optimize_smoothing_params(model, y, initial_smoothing_params=None, method="gcv", optimizer="lbfgsb"):
    from ..selection import optimize_smoothing_params as _optimize

    return _optimize(
        model,
        y=y,
        initial_smoothing_params=initial_smoothing_params,
        method=method,
        optimizer=optimizer,
    )


def build_fit_result(model):
    from ..fit.postprocess.gaussian_smoothness_postprocess import (
        refresh_gaussian_ml_reml_score_from_fit_state,
    )
    from ..results.fit_result import GAMFitResult, TermFitResult

    _require_fitted(model)
    if str(getattr(model, "_optim_method", "")).lower() in {"reml", "ml"}:
        refresh_gaussian_ml_reml_score_from_fit_state(model, model.y_)

    compiled_model = _compiled_model(model)
    if compiled_model is None:
        raise RuntimeError("Model has no compiled model.")
    fit_core_solution = _fit_core_solution(model)
    if fit_core_solution is None:
        raise RuntimeError("Model has no fitted core solution.")

    term_results = []
    edf_by_term = np.asarray(_edf_by_term(model), dtype=np.float64).copy()
    for i, tb in enumerate(_term_blocks_seq(model)):
        sp_vals = [float(model.smoothing_params[j]) for j in tb.smoothing_indices]
        deleted = []
        if tb.deleted_columns is not None:
            deleted = [int(v) for v in np.asarray(tb.deleted_columns, dtype=int).tolist()]
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
                edf=(float(edf_by_term[i]) if i < edf_by_term.size else None),
                smoothing_indices=[int(v) for v in tb.smoothing_indices],
                smoothing_ids=list(tb.smoothing_ids),
                smoothing_values=sp_vals,
                deleted_columns=deleted,
                kept_columns=kept,
                metadata=dict(tb.metadata),
            )
        )

    return GAMFitResult(
        family_name=model.family.name,
        link_name=model.family.link_name,
        criterion_name=model._optim_method,
        criterion_value=model.smoothing_score_,
        coef_full=np.asarray(_coef_full(model), dtype=np.float64).copy(),
        intercept=float(_intercept(model)),
        smoothing_params=np.asarray(model.smoothing_params, dtype=np.float64).copy(),
        edf_total=float(_edf_total(model)),
        edf_by_term=edf_by_term,
        trace_H=float(_trace_H(model)),
        scale=float(_fit_scale(model)),
        rss=None if _rss(model) is None else float(_rss(model)),
        deviance=float(_deviance(model)),
        cov_bayes=(
            None if _cov_bayes(model) is None else np.asarray(_cov_bayes(model), dtype=np.float64).copy()
        ),
        cov_freq=(
            None if _cov_freq(model) is None else np.asarray(_cov_freq(model), dtype=np.float64).copy()
        ),
        side_condition_reports=(
            None if model.side_condition_reports_ is None else list(model.side_condition_reports_)
        ),
        term_results=term_results,
        metadata={
            "n_samples": int(model.n_samples_),
            "n_coef": int(compiled_model.n_coef),
            "fit_intercept": bool(model.fit_intercept),
        },
    )


def copy_fit_result(result, *, include_covariances=True):
    cov_bayes = result.cov_bayes
    cov_freq = result.cov_freq
    if include_covariances:
        cov_bayes = (
            None if cov_bayes is None else np.asarray(cov_bayes, dtype=np.float64).copy()
        )
        cov_freq = (
            None if cov_freq is None else np.asarray(cov_freq, dtype=np.float64).copy()
        )
    else:
        cov_bayes = None
        cov_freq = None

    term_results = [
        replace(
            term,
            smoothing_indices=list(term.smoothing_indices),
            smoothing_ids=list(term.smoothing_ids),
            smoothing_values=list(term.smoothing_values),
            deleted_columns=list(term.deleted_columns),
            kept_columns=list(term.kept_columns),
            metadata=dict(term.metadata),
        )
        for term in result.term_results
    ]
    return replace(
        result,
        coef_full=np.asarray(result.coef_full, dtype=np.float64).copy(),
        smoothing_params=np.asarray(result.smoothing_params, dtype=np.float64).copy(),
        edf_by_term=np.asarray(result.edf_by_term, dtype=np.float64).copy(),
        cov_bayes=cov_bayes,
        cov_freq=cov_freq,
        side_condition_reports=(
            None
            if result.side_condition_reports is None
            else [dict(report) for report in result.side_condition_reports]
        ),
        term_results=term_results,
        metadata=dict(result.metadata),
    )


def build_gam_result(model):
    compiled_model = _compiled_model(model)
    if compiled_model is None:
        raise RuntimeError("Model has no compiled model.")
    fit_core_solution = _fit_core_solution(model)
    if fit_core_solution is None:
        raise RuntimeError("Model has no fitted core solution.")
    fit_summary = _fit_summary(model)
    if fit_summary is None:
        fit_summary = build_fit_result(model)
    return GAMResult(
        compiled_model=compiled_model,
        fit_core_solution=fit_core_solution,
        fit_summary=fit_summary,
    )

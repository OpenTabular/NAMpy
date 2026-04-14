"""Shared model lifecycle checks and canonical fitted-state accessors."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import qr


def _require_fitted(obj: Any) -> None:
    if not getattr(obj, "_fitted", False):
        raise RuntimeError("Model is not fitted.")


def _require_design(model: Any) -> None:
    if _compiled_model(model) is None:
        raise RuntimeError("Model has no compiled design; fit the model first.")


def _gam_result(obj: Any):
    return getattr(obj, "gam_result_", None)


def _compiled_model(obj: Any):
    result = _gam_result(obj)
    if result is not None:
        return result.compiled_model
    return getattr(obj, "compiled_model_", None)


def _fit_summary(obj: Any):
    result = _gam_result(obj)
    if result is not None:
        return result.fit_summary
    return None


def _fit_core_solution(obj: Any):
    result = _gam_result(obj)
    if result is not None:
        return result.fit_core_solution
    return getattr(obj, "fit_core_solution_", None)


def _fit_state(obj: Any):
    fit_core_solution = _fit_core_solution(obj)
    if fit_core_solution is not None:
        return fit_core_solution.fit_state
    return None


def _fit_result(obj: Any):
    fit_core_solution = _fit_core_solution(obj)
    if fit_core_solution is not None:
        return fit_core_solution.fit_result
    return None


def _penalized_system(obj: Any):
    fit_core_solution = _fit_core_solution(obj)
    if fit_core_solution is not None:
        return fit_core_solution.penalized_system
    return None


def _fit_intercept(obj: Any) -> bool:
    return bool(getattr(obj, "fit_intercept", False))


def _coef_column_offset(obj: Any) -> int:
    return 1 if _fit_intercept(obj) else 0


def _term_blocks_seq(obj: Any):
    compiled_model = _compiled_model(obj)
    blocks = (
        None
        if compiled_model is None
        else getattr(compiled_model, "compiled_terms", None)
    )
    if not blocks:
        return ()
    return blocks


def _penalty_blocks_seq(obj: Any):
    compiled_model = _compiled_model(obj)
    blocks = (
        None
        if compiled_model is None
        else getattr(compiled_model, "compiled_penalties", None)
    )
    if not blocks:
        return ()
    return blocks


def _predictor_designs(obj: Any):
    compiled_model = _compiled_model(obj)
    if compiled_model is None:
        return ()
    return getattr(compiled_model, "predictors", ())


def _predictor_full_slices(obj: Any):
    compiled_model = _compiled_model(obj)
    if compiled_model is None:
        return ()
    return getattr(compiled_model, "predictor_full_slices", ())


def _n_coef(obj: Any) -> int:
    compiled_model = _compiled_model(obj)
    if compiled_model is None:
        return 0
    return int(getattr(compiled_model, "n_coef", 0))


def _n_smoothing_params(obj: Any) -> int:
    compiled_model = _compiled_model(obj)
    if compiled_model is None:
        return 0
    return int(getattr(compiled_model, "n_smoothing_params", 0))


def _compiled_metadata(obj: Any) -> dict[str, Any]:
    compiled_model = _compiled_model(obj)
    if compiled_model is None:
        return {}
    return dict(getattr(compiled_model, "metadata", {}) or {})


def _design_matrix(obj: Any):
    compiled_model = _compiled_model(obj)
    if compiled_model is not None:
        design = getattr(compiled_model, "design_matrix", None)
        if design is not None:
            return design
    return None


def _coef_full(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None:
        coef_full = getattr(fit_result, "coef_full", None)
        if coef_full is not None:
            return coef_full
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        coef_full = getattr(fit_summary, "coef_full", None)
        if coef_full is not None:
            return coef_full
    return getattr(obj, "coef_full_", None)


def _coef(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None:
        coef = getattr(fit_result, "beta", None)
        if coef is not None:
            return coef
    coef_full = _coef_full(obj)
    if coef_full is None:
        return getattr(obj, "coef_", None)
    coef_full = np.asarray(coef_full, dtype=np.float64).ravel()
    return coef_full[_coef_column_offset(obj) :]


def _intercept(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None:
        intercept = getattr(fit_result, "intercept", None)
        if intercept is not None:
            return intercept
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        intercept = getattr(fit_summary, "intercept", None)
        if intercept is not None:
            return intercept
    return getattr(obj, "intercept_", None)


def _fit_scale(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        scale = getattr(fit_summary, "scale", None)
        if scale is not None:
            return scale
    fit_result = _fit_result(obj)
    if fit_result is not None:
        scale = getattr(fit_result, "scale", None)
        if scale is not None:
            return scale
    return getattr(obj, "scale_", None)


def _edf_total(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        edf_total = getattr(fit_summary, "edf_total", None)
        if edf_total is not None:
            return edf_total
    fit_result = _fit_result(obj)
    if fit_result is not None:
        edf_total = getattr(fit_result, "edf", None)
        if edf_total is not None:
            return edf_total
    return getattr(obj, "edf_", None)


def _edf_by_term(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        edf = getattr(fit_summary, "edf_by_term", None)
        if edf is not None:
            return edf
    edf = getattr(obj, "_edf_by_term_fit_", None)
    if edf is not None:
        return edf
    return getattr(obj, "edf_by_term_", None)


def _trace_H(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        trace_H = getattr(fit_summary, "trace_H", None)
        if trace_H is not None:
            return trace_H
    fit_result = _fit_result(obj)
    if fit_result is not None:
        trace_H = getattr(fit_result, "trace_H", None)
        if trace_H is not None:
            return trace_H
    return getattr(obj, "trace_H_", None)


def _rss(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        return getattr(fit_summary, "rss", None)
    fit_result = _fit_result(obj)
    if fit_result is not None:
        return getattr(fit_result, "rss", None)
    return getattr(obj, "rss_", None)


def _deviance(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None:
        deviance = getattr(fit_summary, "deviance", None)
        if deviance is not None:
            return deviance
    fit_result = _fit_result(obj)
    if fit_result is not None:
        deviance = getattr(fit_result, "deviance", None)
        if deviance is not None:
            return deviance
    return getattr(obj, "deviance_", None)


def _cov_bayes(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None and getattr(fit_summary, "cov_bayes", None) is not None:
        return fit_summary.cov_bayes
    fit_result = _fit_result(obj)
    if fit_result is not None and getattr(fit_result, "cov_bayes", None) is not None:
        return fit_result.cov_bayes
    return getattr(obj, "Vp_", None)


def _cov_freq(obj: Any):
    fit_summary = _fit_summary(obj)
    if fit_summary is not None and getattr(fit_summary, "cov_freq", None) is not None:
        return fit_summary.cov_freq
    fit_result = _fit_result(obj)
    if fit_result is not None and getattr(fit_result, "cov_freq", None) is not None:
        return fit_result.cov_freq
    return getattr(obj, "Vf_", None)


def _cov_unconditional(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None and getattr(fit_result, "cov_unconditional", None) is not None:
        return fit_result.cov_unconditional
    return getattr(obj, "Vc_", None)


def _edf2(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None and getattr(fit_result, "edf2", None) is not None:
        return fit_result.edf2
    return getattr(obj, "edf2_", None)


def _H_coef(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None and getattr(fit_result, "H_coef", None) is not None:
        return fit_result.H_coef
    return getattr(obj, "_H_coef", None)


def _fitted_eta(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None and getattr(fit_result, "eta", None) is not None:
        return fit_result.eta
    return getattr(obj, "_fitted_eta", None)


def _fitted_mu(obj: Any):
    fit_result = _fit_result(obj)
    if fit_result is not None and getattr(fit_result, "mu", None) is not None:
        return fit_result.mu
    return getattr(obj, "_fitted_mu", None)


def _summary_R(obj: Any):
    fit_state = _fit_state(obj)
    if fit_state is None:
        return getattr(obj, "_summary_R_", None)
    X = getattr(fit_state, "X", None)
    w = getattr(fit_state, "working_weights", None)
    if X is None or w is None:
        return getattr(obj, "_summary_R_", None)
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).ravel()
    if X.ndim != 2 or w.shape[0] != X.shape[0]:
        return getattr(obj, "_summary_R_", None)
    _, R_nat = qr(np.sqrt(np.clip(w, 0.0, None))[:, None] * X, mode="economic")
    return R_nat


def _coerce_feature_matrix(model: Any, X, *, none_is_training: bool = False):
    """Coerce user features to a 2D array; optional training-matrix default."""
    if none_is_training and X is None:
        return model.X_
    from .data import coerce_feature_matrix

    return coerce_feature_matrix(model, X)

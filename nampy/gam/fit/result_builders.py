"""Fit-result assembly and GAMResult synchronization helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .._model_state import (
    _compiled_model,
    _edf_by_term,
    _fit_core_solution,
    _fit_summary,
    _require_fitted,
    _term_blocks_seq,
)
from ..results import GAMResult


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
        core=fit_core_solution.fit_result,
        smoothing_params=np.asarray(model.smoothing_params, dtype=np.float64).copy(),
        side_condition_reports=(
            None
            if compiled_model.side_condition_reports is None
            else list(compiled_model.side_condition_reports)
        ),
        term_results=term_results,
        metadata={
            "n_samples": int(model.n_samples_),
            "n_coef": int(compiled_model.n_coef),
            "fit_intercept": bool(model.fit_intercept),
        },
    )


def copy_fit_result(result, *, include_covariances=True):
    def _copy_or_none(value):
        return None if value is None else np.asarray(value, dtype=np.float64).copy()

    core = result.core
    core_copy = replace(
        core,
        coef_full=np.asarray(core.coef_full, dtype=np.float64).copy(),
        edf_by_term=np.asarray(core.edf_by_term, dtype=np.float64).copy(),
        cov_bayes=_copy_or_none(core.cov_bayes) if include_covariances else None,
        cov_freq=_copy_or_none(core.cov_freq) if include_covariances else None,
        cov_unconditional=(
            _copy_or_none(core.cov_unconditional) if include_covariances else None
        ),
        edf2=_copy_or_none(core.edf2),
    )

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
        core=core_copy,
        smoothing_params=np.asarray(result.smoothing_params, dtype=np.float64).copy(),
        side_condition_reports=(
            None
            if result.side_condition_reports is None
            else [dict(report) for report in result.side_condition_reports]
        ),
        term_results=term_results,
        metadata=dict(result.metadata),
    )


def build_gam_result(model, *, prefer_cached_summary=True):
    compiled_model = _compiled_model(model)
    if compiled_model is None:
        raise RuntimeError("Model has no compiled model.")
    fit_core_solution = _fit_core_solution(model)
    if fit_core_solution is None:
        raise RuntimeError("Model has no fitted core solution.")
    fit_summary = _fit_summary(model) if prefer_cached_summary else None
    if fit_summary is None:
        fit_summary = build_fit_result(model)
    return GAMResult(
        compiled_model=compiled_model,
        fit_core_solution=fit_core_solution,
        fit_summary=fit_summary,
    )


def sync_gam_result(model):
    if _compiled_model(model) is None or _fit_core_solution(model) is None:
        model.gam_result_ = None
        return None
    current = model.gam_result_
    summary = build_fit_result(model)
    model.gam_result_ = replace(current, fit_summary=summary)
    return model.gam_result_

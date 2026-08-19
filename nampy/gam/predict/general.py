"""Prediction helpers for multi-predictor general-family GAMs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .._model_state import (
    _coef_full,
    _predictor_designs,
    _predictor_full_slices,
    _term_blocks_seq,
)
from ..data import coerce_optional_offset
from ..fit.offsets import resolve_prediction_offset
from ..fit.parameterization import prediction_parameterization_map
from ..linalg import symmetrize_matrix


@dataclass(frozen=True)
class _GeneralPredictionLayout:
    Z_new: np.ndarray
    Xp_blocks: list[np.ndarray]
    predictor_slices: tuple[slice, ...]
    jj: list[np.ndarray]
    lpmatrix: np.ndarray


def _prediction_parameterization_wider(tb) -> bool:
    metadata = dict(getattr(tb, "metadata", {}) or {})
    expose_raw = bool(metadata.get("expose_raw_prediction_basis", False))
    wider = metadata.get("prediction_parameterization_wider", None)
    if wider is None:
        return expose_raw
    return expose_raw or bool(wider)


def general_family_prediction_offset(model, X_np, offset):
    n_rows = model.n_samples_ if X_np is None else int(X_np.shape[0])
    n_pred = len(_predictor_full_slices(model))
    offset_value = resolve_prediction_offset(model, X_np, offset)
    if offset_value is None:
        return None
    if isinstance(offset_value, (list, tuple)):
        out = [
            (
                None
                if off_i is None
                else coerce_optional_offset(off_i, n_rows, name=f"offset[{i}]")
            )
            for i, off_i in enumerate(offset_value)
        ]
    else:
        out = [coerce_optional_offset(offset_value, n_rows)]
    if len(out) < n_pred:
        out.extend([None] * (n_pred - len(out)))
    return out


def general_family_prediction_layout(model, X_np):
    Z_blocks = []
    Xp_blocks = []
    predictors = _predictor_designs(model)
    predictor_slices = tuple(_predictor_full_slices(model))
    for pred in predictors:
        use_training_prediction_matrix = any(
            bool(getattr(tb, "metadata", {}).get("expose_raw_prediction_basis"))
            for tb in getattr(pred, "compiled_terms", ())
        )
        predict_has_intercept = bool(
            getattr(pred, "prediction_has_intercept", pred.has_intercept)
        )
        Zp = (
            np.asarray(pred.build_new_matrix(model.X_), dtype=np.float64)
            if X_np is None and use_training_prediction_matrix
            else np.asarray(pred.design_matrix, dtype=np.float64)
            if X_np is None
            else np.asarray(pred.build_new_matrix(X_np), dtype=np.float64)
        )
        Z_blocks.append(Zp)
        if predict_has_intercept:
            Xp_blocks.append(
                np.column_stack([np.ones(Zp.shape[0], dtype=np.float64), Zp])
            )
        else:
            Xp_blocks.append(Zp)
    Z_new = (
        np.column_stack(Z_blocks)
        if Z_blocks
        else np.empty((model.n_samples_ if X_np is None else len(X_np), 0))
    )
    lpmatrix = (
        np.column_stack(Xp_blocks)
        if Xp_blocks
        else np.empty((Z_new.shape[0], 0), dtype=np.float64)
    )
    return _GeneralPredictionLayout(
        Z_new=Z_new,
        Xp_blocks=Xp_blocks,
        predictor_slices=predictor_slices,
        jj=[np.arange(sl.start, sl.stop, dtype=int) for sl in predictor_slices],
        lpmatrix=lpmatrix,
    )


def general_family_link_prediction_with_offset(model, layout, offset):
    eta_cols = []
    off_list = None if offset is None else list(offset)
    coef_full = np.asarray(_coef_full(model), dtype=np.float64)
    for k, (Xp, sl) in enumerate(
        zip(layout.Xp_blocks, layout.predictor_slices, strict=True)
    ):
        eta_k = Xp @ np.asarray(coef_full[sl], dtype=np.float64)
        if off_list is not None and k < len(off_list) and off_list[k] is not None:
            eta_k = eta_k + np.asarray(off_list[k], dtype=np.float64)
        eta_cols.append(np.asarray(eta_k, dtype=np.float64))
    return np.column_stack(eta_cols) if eta_cols else np.empty((0, 0), dtype=np.float64)


def build_general_lpmatrix(model, X_new=None):
    return general_family_prediction_layout(model, X_new).lpmatrix


def _general_family_covariance_for_prediction(model, cov):
    V = model._select_cov(cov)
    if V is None:
        return None
    P = prediction_parameterization_map(model)
    if P is None:
        return np.asarray(V, dtype=np.float64)
    return symmetrize_matrix(np.asarray(P, dtype=np.float64) @ np.asarray(V, dtype=np.float64) @ np.asarray(P, dtype=np.float64).T)


def predict_general_values(
    model,
    X=None,
    return_se=False,
    cov=None,
    type="response",
    offset=None,
    terms=None,
    exclude=None,
):
    if terms is not None or exclude is not None:
        raise NotImplementedError(
            "terms/exclude prediction filters are not yet supported for "
            "multi-predictor general-family models."
        )
    pred_type = str(type).lower()
    if pred_type not in {"response", "link", "terms", "iterms", "lpmatrix"}:
        raise ValueError(
            "type must be one of {'response', 'link', 'terms', 'iterms', 'lpmatrix'}"
        )
    if pred_type == "iterms":
        warnings.warn(
            "type iterms not available for multiple predictor cases; reset to terms",
            stacklevel=2,
        )
        pred_type = "terms"
    from .predictions import (
        _group_standard_error_rows,
        _group_term_contribution,
        _prediction_term_groups,
    )

    if pred_type == "terms" and any(
        _prediction_parameterization_wider(tb)
        for tb in _term_blocks_seq(model)
    ):
        raise NotImplementedError(
            "type='terms' is not supported for general-family models whose "
            "prediction parameterization is wider than the fitted coefficient space."
        )

    offset_list = general_family_prediction_offset(model, X, offset)
    layout = general_family_prediction_layout(model, X)

    eta = general_family_link_prediction_with_offset(model, layout, offset_list)
    Z_new = layout.Z_new

    if pred_type == "lpmatrix":
        return layout.lpmatrix
    if pred_type == "terms":
        groups = _prediction_term_groups(model)
        terms = np.column_stack(
            [_group_term_contribution(model, Z_new, group) for group in groups]
        )
        if not return_se:
            return terms
        V = _general_family_covariance_for_prediction(model, cov)
        ses = []
        for group in groups:
            Xi, sl_full = _group_standard_error_rows(
                model,
                layout.lpmatrix,
                group,
                type=pred_type,
            )
            if sl_full is None:
                var = np.einsum("ij,jk,ik->i", Xi, V, Xi)
            elif isinstance(sl_full, np.ndarray):
                Vi = V[np.ix_(sl_full, sl_full)]
                var = np.einsum("ij,jk,ik->i", Xi, Vi, Xi)
            else:
                Vi = V[sl_full, sl_full]
                var = np.einsum("ij,jk,ik->i", Xi, Vi, Xi)
            ses.append(np.sqrt(np.maximum(var, 0.0)))
        return terms, np.column_stack(ses)
    if pred_type == "link":
        if not return_se:
            return eta
        V = _general_family_covariance_for_prediction(model, cov)
        se_cols = []
        for Xp, sl in zip(
            layout.Xp_blocks, layout.predictor_slices, strict=True
        ):
            Vk = V[sl, sl]
            var = np.einsum("ij,jk,ik->i", Xp, Vk, Xp)
            se_cols.append(np.sqrt(np.maximum(var, 0.0)))
        return eta, np.column_stack(se_cols)

    coef_full = np.asarray(_coef_full(model), dtype=np.float64)
    family_predict = getattr(model.family, "predict", None)
    if callable(family_predict):
        out = family_predict(
            eta=eta,
            X=layout.lpmatrix,
            jj=layout.jj,
            coef=coef_full,
            offset=offset_list,
            se=return_se,
            Vb=(
                None
                if not return_se
                else _general_family_covariance_for_prediction(model, cov)
            ),
        )
        return out

    if return_se:
        raise NotImplementedError(
            f"{model.family.__class__.__name__} does not yet implement predictive standard errors."
        )
    return np.asarray(
        model.family.predict_fitted(
            layout.lpmatrix,
            layout.jj,
            coef_full,
            offset=offset_list,
        ),
        dtype=np.float64,
    )

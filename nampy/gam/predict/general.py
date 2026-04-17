"""Prediction helpers for multi-predictor general-family GAMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._model_state import (
    _coef,
    _coef_full,
    _predictor_designs,
    _predictor_full_slices,
    _term_blocks_seq,
)
from ..data import coerce_optional_offset
from ..fit.offsets import resolve_prediction_offset


@dataclass(frozen=True)
class _GeneralPredictionLayout:
    Z_new: np.ndarray
    Xp_blocks: list[np.ndarray]
    predictor_slices: tuple[slice, ...]
    jj: list[np.ndarray]
    lpmatrix: np.ndarray


def general_family_prediction_offset(model, X_np, offset):
    n_rows = model.n_samples_ if X_np is None else int(X_np.shape[0])
    n_pred = len(_predictor_full_slices(model))
    if offset is None:
        offset_vec = resolve_prediction_offset(model, X_np, None)
        if offset_vec is None:
            return None
        return [offset_vec] + [None] * max(n_pred - 1, 0)
    if isinstance(offset, (list, tuple)):
        out = []
        for i, off_i in enumerate(offset):
            out.append(
                None
                if off_i is None
                else coerce_optional_offset(off_i, n_rows, name=f"offset[{i}]")
            )
        if len(out) < n_pred:
            out.extend([None] * (n_pred - len(out)))
        return out
    offset_vec = coerce_optional_offset(offset, n_rows)
    return [offset_vec] + [None] * max(n_pred - 1, 0)


def general_family_prediction_layout(model, X_np):
    Z_blocks = []
    Xp_blocks = []
    predictors = _predictor_designs(model)
    predictor_slices = tuple(_predictor_full_slices(model))
    for pred in predictors:
        Zp = (
            np.asarray(pred.design_matrix, dtype=np.float64)
            if X_np is None
            else np.asarray(pred.build_new_matrix(X_np), dtype=np.float64)
        )
        Z_blocks.append(Zp)
        if bool(pred.has_intercept):
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


def general_family_prediction_blocks(model, X_np):
    layout = general_family_prediction_layout(model, X_np)
    return layout.Z_new, layout.Xp_blocks


def general_family_link_prediction_with_offset(model, layout, offset):
    eta_cols = []
    off_list = None if offset is None else list(offset)
    coef_full = np.asarray(_coef_full(model), dtype=np.float64)
    for k, (Xp, sl) in enumerate(zip(layout.Xp_blocks, layout.predictor_slices)):
        eta_k = Xp @ np.asarray(coef_full[sl], dtype=np.float64)
        if off_list is not None and k < len(off_list) and off_list[k] is not None:
            eta_k = eta_k + np.asarray(off_list[k], dtype=np.float64)
        eta_cols.append(np.asarray(eta_k, dtype=np.float64))
    return np.column_stack(eta_cols) if eta_cols else np.empty((0, 0), dtype=np.float64)


def build_general_lpmatrix(model, X_new=None):
    return general_family_prediction_layout(model, X_new).lpmatrix


def predict_general_values(
    model, X=None, return_se=False, cov=None, type="response", offset=None
):
    type = str(type).lower()
    if type not in {"response", "link", "terms", "lpmatrix"}:
        raise ValueError(
            "type must be one of {'response', 'link', 'terms', 'lpmatrix'}"
        )

    offset_list = general_family_prediction_offset(model, X, offset)
    layout = general_family_prediction_layout(model, X)
    eta = general_family_link_prediction_with_offset(model, layout, offset_list)
    Z_new = layout.Z_new

    if type == "lpmatrix":
        return layout.lpmatrix
    if type == "terms":
        beta = np.asarray(_coef(model), dtype=np.float64)
        terms = np.column_stack(
            [
                Z_new[:, tb.coef_slice] @ beta[tb.coef_slice]
                for tb in _term_blocks_seq(model)
            ]
        )
        if not return_se:
            return terms
        V = model._select_cov(cov)
        ses = []
        full_idx = np.asarray(model._coef_reduced_to_full_idx, dtype=int)
        for tb in _term_blocks_seq(model):
            idx = full_idx[tb.coef_slice]
            Xi = np.zeros((Z_new.shape[0], V.shape[0]), dtype=np.float64)
            Xi[:, idx] = Z_new[:, tb.coef_slice]
            var = np.einsum("ij,jk,ik->i", Xi, V, Xi)
            ses.append(np.sqrt(np.maximum(var, 0.0)))
        return terms, np.column_stack(ses)
    if type == "link":
        if not return_se:
            return eta
        V = model._select_cov(cov)
        se_cols = []
        for Xp, sl in zip(layout.Xp_blocks, layout.predictor_slices):
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
            Vb=None if not return_se else model._select_cov(cov),
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

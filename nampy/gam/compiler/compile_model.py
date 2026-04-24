"""Compile declarative predictor specs into a fit-facing compiled model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

from ..constraints.identifiability import apply_global_side_conditions
from ..linalg import upper_triangular_rrank
from .compile_predictors import compile_predictors
from .structures import CompiledModel


def _block_diagonal_matrix(blocks: list[np.ndarray]) -> np.ndarray:
    total_rows = int(sum(block.shape[0] for block in blocks))
    total_cols = int(sum(block.shape[1] for block in blocks))
    out = np.zeros((total_rows, total_cols), dtype=np.float64)
    row_start = 0
    col_start = 0
    for block in blocks:
        n_rows = int(block.shape[0])
        n_cols = int(block.shape[1])
        row_stop = row_start + n_rows
        col_stop = col_start + n_cols
        out[row_start:row_stop, col_start:col_stop] = np.asarray(
            block, dtype=np.float64
        )
        row_start = row_stop
        col_start = col_stop
    return out


def _full_predictor_matrix(predictor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Z_fit = np.asarray(predictor.design_matrix, dtype=np.float64)
    pred_blocks = []
    for term in predictor.compiled_terms:
        use_raw = bool(getattr(term, "metadata", {}).get("expose_raw_prediction_basis"))
        if use_raw:
            block = np.asarray(
                term.prediction_parameterization_matrix(X), dtype=np.float64
            )
        else:
            block = np.asarray(term.predict_matrix(X), dtype=np.float64)
        pred_blocks.append(block)
    Z_pred = (
        np.column_stack(pred_blocks)
        if pred_blocks
        else np.empty((X.shape[0], 0), dtype=np.float64)
    )
    predict_has_intercept = bool(
        getattr(predictor, "prediction_has_intercept", predictor.has_intercept)
    )
    if bool(predictor.has_intercept):
        ones = np.ones((Z_fit.shape[0], 1), dtype=np.float64)
        X_fit = np.column_stack([ones, Z_fit])
        X_pred = np.column_stack([ones, Z_pred]) if predict_has_intercept else Z_pred
        return X_fit, X_pred
    return Z_fit, Z_pred


def _fit_to_prediction_parameterization_map(
    X_fit: np.ndarray,
    X_pred: np.ndarray,
    *,
    tol: float = 1e-10,
) -> np.ndarray | None:
    X_fit = np.asarray(X_fit, dtype=np.float64)
    X_pred = np.asarray(X_pred, dtype=np.float64)
    if X_fit.shape[0] != X_pred.shape[0]:
        raise ValueError(
            "fit/prediction matrices must share row count, got "
            f"{X_fit.shape} and {X_pred.shape}."
        )
    if X_fit.shape[1] == 0 and X_pred.shape[1] == 0:
        return None
    if X_pred.shape[1] == 0:
        raise ValueError(
            "Prediction-parameterization matrix has zero width but fit matrix does not."
        )
    if X_fit.shape == X_pred.shape and np.allclose(
        X_fit, X_pred, atol=max(float(tol), 1e-8), rtol=max(float(tol), 1e-8)
    ):
        return None
    # Mirror mgcv/R/mgcv.r construction of G$P:
    # qr(Xp, LAPACK=TRUE) -> Rrank(R) -> triangular solve on QtX -> restore pivots.
    Q, R, piv = scipy_qr(X_pred, mode="economic", pivoting=True)
    p_pred = int(R.shape[1])
    rank = upper_triangular_rrank(R, tol=np.finfo(np.float64).eps ** 0.9)
    QtX = np.asarray(Q.T @ X_fit, dtype=np.float64)[:rank, :]
    if rank < p_pred:
        R1 = np.asarray(R[:rank, :], dtype=np.float64)
        Qr, Rr, _pivot_r = scipy_qr(R1.T, mode="full", pivoting=True)
        G0 = solve_triangular(
            np.asarray(Rr[:rank, :rank], dtype=np.float64).T,
            QtX,
            lower=True,
            check_finite=False,
        )
        P_pivot = Qr @ np.vstack(
            [
                np.asarray(G0, dtype=np.float64),
                np.zeros((p_pred - rank, X_fit.shape[1]), dtype=np.float64),
            ]
        )
    else:
        P_pivot = solve_triangular(
            np.asarray(R[:rank, :rank], dtype=np.float64),
            QtX,
            lower=False,
            check_finite=False,
        )

    P = np.zeros((p_pred, X_fit.shape[1]), dtype=np.float64)
    P[np.asarray(piv, dtype=int), :] = np.asarray(P_pivot, dtype=np.float64)
    recon = X_pred @ P
    if not np.allclose(recon, X_fit, atol=1e-8, rtol=1e-8):
        raise RuntimeError(
            "Failed to recover mgcv-style fit->prediction parameterization transform."
        )
    if P.shape[0] == P.shape[1] and np.allclose(
        P, np.eye(P.shape[0], dtype=np.float64), atol=1e-12, rtol=1e-12
    ):
        return None
    return np.asarray(P, dtype=np.float64)


def _predictor_parameterization_map(
    predictor,
    X: np.ndarray,
) -> np.ndarray | None:
    X_fit_full, X_pred_full = _full_predictor_matrix(predictor, X)
    return _fit_to_prediction_parameterization_map(X_fit_full, X_pred_full)


def _model_parameterization_map(
    compiled_predictors,
    X: np.ndarray,
) -> np.ndarray | None:
    blocks: list[np.ndarray] = []
    any_nontrivial = False
    for predictor in compiled_predictors:
        P_pred = _predictor_parameterization_map(predictor, X)
        if P_pred is None:
            width = int(predictor.n_coef) + int(bool(predictor.has_intercept))
            P_pred = np.eye(width, dtype=np.float64)
        else:
            any_nontrivial = True
        blocks.append(np.asarray(P_pred, dtype=np.float64))
    if not any_nontrivial:
        return None
    return _block_diagonal_matrix(blocks)


def compile_model(
    X: np.ndarray,
    feature_names: list[str],
    predictor_specs,
    *,
    fit_intercept: bool,
    apply_side_conditions: bool = True,
    side_condition_tol: float = 1e-10,
):
    compiled_predictors = compile_predictors(
        X=X,
        feature_names=feature_names,
        predictor_specs=predictor_specs,
    )

    reports = None
    if apply_side_conditions:
        adjusted = []
        reports = []
        for predictor in compiled_predictors:
            predictor_adj, report = apply_global_side_conditions(
                predictor,
                fit_intercept=bool(predictor.has_intercept),
                tol=side_condition_tol,
                warn=True,
            )
            adjusted.append(predictor_adj)
            reports.append(report)
        compiled_predictors = adjusted

    pred_param_map = _model_parameterization_map(compiled_predictors, X)
    predictor_prediction_widths = [
        int(_full_predictor_matrix(predictor, X)[1].shape[1])
        for predictor in compiled_predictors
    ]

    if len(compiled_predictors) == 1:
        predictor = compiled_predictors[0]
        metadata = dict(getattr(predictor, "metadata", {}) or {})
        if pred_param_map is not None:
            metadata["fit_to_prediction_parameterization_map"] = np.asarray(
                pred_param_map, dtype=np.float64
            )
        return CompiledModel(
            predictors=(predictor,),
            design_matrix=np.asarray(predictor.design_matrix, dtype=np.float64),
            compiled_terms=tuple(predictor.compiled_terms),
            compiled_penalties=tuple(predictor.compiled_penalties),
            metadata=metadata,
            n_coef=int(predictor.n_coef),
            n_smoothing_params=int(predictor.n_smoothing_params),
            predictor_full_slices=(slice(0, predictor_prediction_widths[0]),),
            coef_reduced_to_full_idx=np.arange(int(predictor.n_coef), dtype=int)
            + (1 if bool(predictor.has_intercept) else 0),
            smoothing_override_modes=list(predictor.smoothing_override_modes or []),
            smoothing_override_values=(
                None
                if predictor.smoothing_override_values is None
                else np.asarray(
                    predictor.smoothing_override_values, dtype=np.float64
                ).copy()
            ),
            side_condition_reports=(
                None if reports is None else tuple(dict(report) for report in reports)
            ),
            fit_to_prediction_parameterization_map=(
                None
                if pred_param_map is None
                else np.asarray(pred_param_map, dtype=np.float64)
            ),
        )

    global_terms = []
    global_penalties = []
    combined_blocks = []
    combined_map = {}
    override_modes = []
    override_values = []
    predictor_full_slices = []
    reduced_to_full = []
    term_shift = 0
    coef_shift = 0
    sp_shift = 0
    full_shift = 0

    for predictor_index, predictor in enumerate(compiled_predictors):
        combined_blocks.append(np.asarray(predictor.design_matrix, dtype=np.float64))
        pred_full_width = int(predictor_prediction_widths[predictor_index])

        if bool(predictor.has_intercept):
            reduced_to_full.extend(
                list(
                    np.arange(
                        full_shift + 1,
                        full_shift + 1 + int(predictor.n_coef),
                        dtype=int,
                    )
                )
            )
            predictor_full_slices.append(
                slice(full_shift, full_shift + pred_full_width)
            )
            full_shift += pred_full_width
        else:
            reduced_to_full.extend(
                list(
                    np.arange(full_shift, full_shift + int(predictor.n_coef), dtype=int)
                )
            )
            predictor_full_slices.append(
                slice(full_shift, full_shift + pred_full_width)
            )
            full_shift += pred_full_width

        for term in predictor.compiled_terms:
            global_terms.append(
                replace(
                    term,
                    coef_slice=slice(
                        coef_shift + int(term.coef_slice.start),
                        coef_shift + int(term.coef_slice.stop),
                    ),
                    smoothing_indices=[
                        sp_shift + int(value)
                        for value in getattr(term, "smoothing_indices", [])
                    ],
                    smoothing_ids=[
                        f"{predictor.name}:{sid}" if sid is not None else None
                        for sid in getattr(term, "smoothing_ids", [])
                    ],
                )
            )
        for penalty in predictor.compiled_penalties:
            global_penalties.append(
                replace(
                    penalty,
                    coef_slice=slice(
                        coef_shift + int(penalty.coef_slice.start),
                        coef_shift + int(penalty.coef_slice.stop),
                    ),
                    term_index=(
                        term_shift + int(penalty.term_index)
                        if int(penalty.term_index) >= 0
                        else int(penalty.term_index)
                    ),
                    smoothing_index=sp_shift + int(penalty.smoothing_index),
                    smoothing_id=(
                        None
                        if penalty.smoothing_id is None
                        else f"{predictor.name}:{penalty.smoothing_id}"
                    ),
                )
            )
        for smoothing_id, indices in (
            predictor.metadata.get("s_id_to_sp_indices", {}) or {}
        ).items():
            combined_map[f"{predictor.name}:{smoothing_id}"] = [
                sp_shift + int(index) for index in indices
            ]
        override_modes.extend(list(predictor.smoothing_override_modes or []))
        if predictor.smoothing_override_values is not None:
            override_values.extend(
                list(np.asarray(predictor.smoothing_override_values, dtype=np.float64))
            )
        term_shift += len(predictor.compiled_terms)
        coef_shift += int(predictor.n_coef)
        sp_shift += int(predictor.n_smoothing_params)

    return CompiledModel(
        predictors=tuple(compiled_predictors),
        design_matrix=(
            np.column_stack(combined_blocks)
            if combined_blocks
            else np.empty((X.shape[0], 0), dtype=np.float64)
        ),
        compiled_terms=tuple(global_terms),
        compiled_penalties=tuple(global_penalties),
        metadata={
            "s_id_to_sp_indices": combined_map,
            **(
                {}
                if pred_param_map is None
                else {
                    "fit_to_prediction_parameterization_map": np.asarray(
                        pred_param_map, dtype=np.float64
                    )
                }
            ),
        },
        n_coef=coef_shift,
        n_smoothing_params=sp_shift,
        predictor_full_slices=tuple(predictor_full_slices),
        coef_reduced_to_full_idx=np.asarray(reduced_to_full, dtype=int),
        smoothing_override_modes=list(override_modes),
        smoothing_override_values=np.asarray(override_values, dtype=np.float64),
        side_condition_reports=(
            None if reports is None else tuple(dict(report) for report in reports)
        ),
        fit_to_prediction_parameterization_map=(
            None
            if pred_param_map is None
            else np.asarray(pred_param_map, dtype=np.float64)
        ),
    )


__all__ = ["compile_model"]

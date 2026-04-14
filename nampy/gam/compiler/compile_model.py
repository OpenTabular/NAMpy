"""Compile declarative predictor specs into an engine-facing compiled model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..constraints.identifiability import apply_global_side_conditions
from .compile_predictors import compile_predictors
from .structures import CompiledModel


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
                fit_intercept=fit_intercept,
                tol=side_condition_tol,
                warn=True,
            )
            adjusted.append(predictor_adj)
            reports.append(report)
        compiled_predictors = adjusted

    if len(compiled_predictors) == 1:
        predictor = compiled_predictors[0]
        return CompiledModel(
            predictors=(predictor,),
            design_matrix=np.asarray(predictor.design_matrix, dtype=np.float64),
            compiled_terms=tuple(predictor.compiled_terms),
            compiled_penalties=tuple(predictor.compiled_penalties),
            metadata=dict(getattr(predictor, "metadata", {}) or {}),
            n_coef=int(predictor.n_coef),
            n_smoothing_params=int(predictor.n_smoothing_params),
            predictor_full_slices=(
                slice(
                    0,
                    int(predictor.n_coef) + (1 if bool(predictor.has_intercept) else 0),
                ),
            ),
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
        )

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

    for predictor in compiled_predictors:
        combined_blocks.append(np.asarray(predictor.design_matrix, dtype=np.float64))

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
                slice(full_shift, full_shift + int(predictor.n_coef) + 1)
            )
            full_shift += int(predictor.n_coef) + 1
        else:
            reduced_to_full.extend(
                list(
                    np.arange(full_shift, full_shift + int(predictor.n_coef), dtype=int)
                )
            )
            predictor_full_slices.append(
                slice(full_shift, full_shift + int(predictor.n_coef))
            )
            full_shift += int(predictor.n_coef)

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
        metadata={"s_id_to_sp_indices": combined_map},
        n_coef=coef_shift,
        n_smoothing_params=sp_shift,
        predictor_full_slices=tuple(predictor_full_slices),
        coef_reduced_to_full_idx=np.asarray(reduced_to_full, dtype=int),
        smoothing_override_modes=list(override_modes),
        smoothing_override_values=np.asarray(override_values, dtype=np.float64),
        side_condition_reports=(
            None if reports is None else tuple(dict(report) for report in reports)
        ),
    )


__all__ = ["compile_model"]

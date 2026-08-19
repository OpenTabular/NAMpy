"""Compiler-owned predictor assembly from constructed terms."""

from __future__ import annotations

import numpy as np

from ..penalties import (
    default_penalty_id,
    merge_smoothing_override,
    normalize_penalty_spec,
    penalty_id_for_local_index,
)
from ..specs import LinearPredictorSpec, PenaltyGroupSpec
from .construct import construct_smooth
from .factory import instantiate_predictor_terms
from .linked_basis import attach_shared_basis_metadata
from .structures import CompiledPenalty, CompiledPredictor, CompiledTerm


def compile_predictors(
    X: np.ndarray,
    feature_names: list[str],
    predictor_specs: list[LinearPredictorSpec],
):
    predictor_specs = [
        linked_spec
        for pred_spec in predictor_specs
        for linked_spec in attach_shared_basis_metadata(
            [pred_spec], X=X, feature_names=feature_names
        )
    ]
    predictor_specs = instantiate_predictor_terms(predictor_specs)

    designs = []
    for pred_spec in predictor_specs:
        pred_metadata = dict(getattr(pred_spec, "metadata", {}) or {})
        raw_group_specs = pred_metadata.get("penalty_group_specs", []) or []
        penalty_group_specs: dict[str, PenaltyGroupSpec] = {}
        for spec in raw_group_specs:
            if isinstance(spec, PenaltyGroupSpec):
                group = PenaltyGroupSpec(
                    smoothing_id=str(spec.smoothing_id),
                    term_ids=list(spec.term_ids),
                    labels=list(spec.labels),
                    sp_count=spec.sp_count,
                    sp_indices=list(spec.sp_indices),
                )
            else:
                group = PenaltyGroupSpec(
                    smoothing_id=str(spec["smoothing_id"]),
                    term_ids=list(spec.get("term_ids", [])),
                    labels=list(spec.get("labels", [])),
                    sp_count=spec.get("sp_count"),
                    sp_indices=list(spec.get("sp_indices", [])),
                )
            penalty_group_specs[group.smoothing_id] = group

        term_blocks = []
        penalty_blocks = []
        design_blocks = []
        smoothing_id_map: dict[str, int] = {}
        smoothing_override_by_id: dict[str, dict | None] = {}
        start = 0

        for term_like in pred_spec.terms:
            smooth = construct_smooth(
                term_like,
                X=X,
                feature_names=feature_names,
                absorb_cons=True,
                apply_by=True,
                null_space_penalty=False,
            )
            base_term = smooth.compiled_term
            B = np.asarray(base_term.basis_train, dtype=np.float64)
            d = B.shape[1]
            sl = slice(start, start + d)
            term_meta = dict(base_term.metadata)
            term_meta["constructor_metadata"] = dict(base_term.constructor_metadata)
            term_blocks.append(
                CompiledTerm(
                    label=base_term.label,
                    coef_slice=sl,
                    basis_train=B,
                    predict_fn=base_term.predict_fn,
                    predict_coefficient_map=base_term.predict_coefficient_map,
                    basis_transform=base_term.basis_transform,
                    coefficient_maps=tuple(base_term.coefficient_maps),
                    feature_info=base_term.feature_info,
                    by_variable_info=base_term.by_variable_info,
                    side_condition_policy=base_term.side_condition_policy,
                    kept_columns=np.arange(d, dtype=int),
                    deleted_columns=np.array([], dtype=int),
                    smoothing_indices=[],
                    smoothing_ids=[],
                    n_penalties=0,
                    term_type=str(base_term.term_type),
                    basis_name=str(base_term.basis_name),
                    term_id=base_term.term_id,
                    smoothing_group_id=base_term.smoothing_group_id,
                    penalty_specs=tuple(smooth.penalty_specs),
                    constructor_metadata=dict(base_term.constructor_metadata),
                    metadata=term_meta,
                )
            )
            start += d

        for term_index, tb in enumerate(term_blocks):
            B = np.asarray(tb.basis_train, dtype=np.float64)
            d = B.shape[1]
            if d > 0:
                design_blocks.append(B)

            penalty_defs = [normalize_penalty_spec(p) for p in tb.penalty_specs]
            raw_sid_counts: dict[str, int] = {}
            for pdef in penalty_defs:
                sid_raw = getattr(pdef, "smoothing_id", None)
                if sid_raw is None:
                    continue
                sid_key = str(sid_raw)
                raw_sid_counts[sid_key] = raw_sid_counts.get(sid_key, 0) + 1
            term_smoothing_indices: list[int] = []
            term_smoothing_ids: list[str | None] = []
            for j, pdef in enumerate(penalty_defs):
                P = np.asarray(pdef.matrix, dtype=np.float64)
                sid = pdef.smoothing_id
                if sid is None:
                    if tb.smoothing_group_id is not None:
                        sid = (
                            str(tb.smoothing_group_id)
                            if len(penalty_defs) <= 1
                            else penalty_id_for_local_index(
                                tb.smoothing_group_id, j, n_penalties=len(penalty_defs)
                            )
                        )
                    else:
                        sid = default_penalty_id(
                            pred_name=pred_spec.name,
                            term=tb,
                            term_label=tb.label,
                            coef_start=int(tb.coef_slice.start),
                            local_penalty_index=j,
                            n_penalties=len(penalty_defs),
                        )
                sid = str(sid)
                first_smoothing_id_occurrence = sid not in smoothing_id_map
                if first_smoothing_id_occurrence:
                    smoothing_id_map[sid] = len(smoothing_id_map)
                sp_idx = smoothing_id_map[sid]
                if first_smoothing_id_occurrence:
                    smoothing_override_by_id[sid] = merge_smoothing_override(
                        smoothing_override_by_id.get(sid, None),
                        pdef.sp_mode,
                        pdef.sp_value,
                        smoothing_id=sid,
                        label=tb.label,
                    )
                penalty_blocks.append(
                    CompiledPenalty(
                        label=tb.label,
                        coef_slice=tb.coef_slice,
                        matrix=P,
                        smoothing_index=sp_idx,
                        term_index=term_index,
                        smoothing_id=sid,
                        kind=str(pdef.kind),
                        rank=pdef.rank,
                        null_space_dim=pdef.null_space_dim,
                        is_null_space_penalty=bool(pdef.is_null_space_penalty),
                        sp_mode=pdef.sp_mode,
                        sp_value=pdef.sp_value,
                        metadata=dict(pdef.metadata),
                    )
                )
                term_smoothing_indices.append(sp_idx)
                term_smoothing_ids.append(sid)

            tb.smoothing_indices = term_smoothing_indices
            tb.smoothing_ids = term_smoothing_ids
            tb.n_penalties = len(penalty_defs)

            group_id = (
                None
                if getattr(tb, "smoothing_group_id", None) is None
                else str(tb.smoothing_group_id)
            )
            if group_id is not None:
                linked_group = penalty_group_specs.get(group_id)
                if linked_group is None:
                    linked_group = PenaltyGroupSpec(smoothing_id=group_id)
                    penalty_group_specs[group_id] = linked_group
                if linked_group.sp_count is None:
                    linked_group.sp_count = len(term_smoothing_indices)
                elif linked_group.sp_count < len(term_smoothing_indices):
                    raise ValueError(
                        f"Linked smoothing id {group_id!r} cannot have more smoothing "
                        f"parameters than its defining term ({linked_group.sp_count}); got "
                        f"{len(term_smoothing_indices)}."
                    )
                if not linked_group.sp_indices:
                    linked_group.sp_indices = list(term_smoothing_indices)
                elif linked_group.sp_indices[: len(term_smoothing_indices)] != list(
                    term_smoothing_indices
                ):
                    raise ValueError(
                        f"Linked smoothing id {group_id!r} resolved to inconsistent "
                        f"smoothing indices {linked_group.sp_indices} and "
                        f"{term_smoothing_indices}."
                    )
                if tb.term_id not in linked_group.term_ids:
                    linked_group.term_ids.append(str(tb.term_id))
                if tb.label not in linked_group.labels:
                    linked_group.labels.append(str(tb.label))

        matrix_train = (
            np.column_stack(design_blocks)
            if design_blocks
            else np.empty((X.shape[0], 0), dtype=np.float64)
        )
        n_sp = len(smoothing_id_map)
        override_modes: list[str | None] = [None] * n_sp
        override_values = np.full(n_sp, np.nan, dtype=np.float64)
        for sid, spec in smoothing_override_by_id.items():
            if spec is None:
                continue
            idx = smoothing_id_map[sid]
            override_modes[idx] = spec["mode"]
            if spec["mode"] == "fixed":
                override_values[idx] = float(spec["value"])

        term_index_map = {tb.term_id: i for i, tb in enumerate(term_blocks)}
        pred_metadata["penalty_group_specs"] = list(penalty_group_specs.values())
        pred_metadata["s_id_to_sp_indices"] = {
            sid: list(group.sp_indices)
            for sid, group in penalty_group_specs.items()
            if group.sp_indices
        }
        designs.append(
            CompiledPredictor(
                name=pred_spec.name,
                design_matrix=matrix_train,
                compiled_terms=tuple(term_blocks),
                compiled_penalties=tuple(penalty_blocks),
                smoothing_parameter_map=smoothing_id_map,
                has_intercept=bool(pred_spec.has_intercept),
                term_index_map=term_index_map,
                side_condition_Q=np.eye(start, dtype=np.float64),
                n_coef=start,
                n_smoothing_params=len(smoothing_id_map),
                smoothing_override_modes=override_modes,
                smoothing_override_values=override_values,
                metadata=pred_metadata,
            )
        )
    return designs


__all__ = ["compile_predictors"]

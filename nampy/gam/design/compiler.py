"""
Stage 4 of the GAM fit pipeline: predictor compilation.

Assembles fitted ConstructedTerm objects (stage 3) into a CompiledPredictor:
assigns global coefficient slices, registers smoothing-parameter ids, normalises
penalty specs, and records the initial (identity) basis_transform on every term.

The output of this stage is handed to stage 5 (apply_global_side_conditions)
which applies predictor-wide identifiability constraints and updates basis_transform
on each term to reflect the final constructed-space-to-fitted coefficient map.
"""

from __future__ import annotations

import numpy as np

from ..penalties import (
    default_penalty_id,
    merge_smoothing_override,
    normalize_penalty_spec,
)
from ..runtime.factory import instantiate_predictor_terms
from ..specs import LinearPredictorSpec, PenaltyGroupSpec
from .constructors import construct_terms
from .linked_basis import attach_shared_basis_metadata
from .structures import CompiledPenalty, CompiledPredictor, CompiledTerm


def compile_predictor_designs(
    X: np.ndarray, feature_names: list[str], predictor_specs: list[LinearPredictorSpec]
):
    """
    Run stages 2–4 for a list of LinearPredictorSpecs and return CompiledPredictors.

    Internally this function drives:
      - Stage 2: instantiate runtime terms from TermSpecs
        (via ``instantiate_predictor_terms``)
      - Stage 3: fit each runtime term and wrap it in a ConstructedTerm
        (via ``construct_terms``)
      - Stage 4: assemble ConstructedTerms into a CompiledPredictor, assign
        coefficient slices, smoothing-parameter ids, and initial basis_transforms

    The returned CompiledPredictors have ``basis_transform = I`` on every term,
    where that identity acts on the stage-3 constructed-term coefficient space.
    Stage 5 (``apply_global_side_conditions``) must be called to produce the
    final predictor with canonical constructed-space coefficient transforms.
    """
    predictor_specs = attach_shared_basis_metadata(
        predictor_specs, X=X, feature_names=feature_names
    )
    predictor_specs = instantiate_predictor_terms(predictor_specs)

    designs = []
    for pred_spec in predictor_specs:
        pred_metadata = dict(getattr(pred_spec, "metadata", {}) or {})
        raw_group_specs = pred_metadata.get("penalty_group_specs", []) or []
        penalty_group_specs = {}
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
        smoothing_id_map = {}
        smoothing_override_by_id = {}
        start = 0

        for term_like in pred_spec.terms:
            constructed = construct_terms(
                term_like,
                X=X,
                feature_names=feature_names,
                absorb_cons=True,
                apply_by=True,
                null_space_penalty=False,
            )
            for smooth in constructed:
                B = np.asarray(smooth.train_design_matrix, dtype=np.float64)
                d = B.shape[1]
                sl = slice(start, start + d)
                penalty_defs = [normalize_penalty_spec(p) for p in smooth.penalty_specs]
                term_smoothing_indices = []
                term_smoothing_ids = []
                for j, pdef in enumerate(penalty_defs):
                    P = np.asarray(pdef.matrix, dtype=np.float64)
                    sid = pdef.smoothing_id
                    if sid is None:
                        sid = default_penalty_id(
                            pred_name=pred_spec.name,
                            term=smooth,
                            term_label=smooth.label,
                            coef_start=start,
                            local_penalty_index=j,
                            n_penalties=len(penalty_defs),
                        )
                    sid = str(sid)
                    if sid not in smoothing_id_map:
                        smoothing_id_map[sid] = len(smoothing_id_map)
                    sp_idx = smoothing_id_map[sid]
                    smoothing_override_by_id[sid] = merge_smoothing_override(
                        smoothing_override_by_id.get(sid, None),
                        pdef.sp_mode,
                        pdef.sp_value,
                        smoothing_id=sid,
                        label=smooth.label,
                    )
                    penalty_blocks.append(
                        CompiledPenalty(
                            label=smooth.label,
                            coef_slice=sl,
                            matrix=P,
                            smoothing_index=sp_idx,
                            term_index=len(term_blocks),
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

                group_id = (
                    None
                    if getattr(smooth, "smoothing_id", None) is None
                    else str(smooth.smoothing_id)
                )
                if group_id is not None:
                    group = penalty_group_specs.get(group_id)
                    if group is None:
                        group = PenaltyGroupSpec(smoothing_id=group_id)
                        penalty_group_specs[group_id] = group
                    if group.sp_count is None:
                        group.sp_count = len(term_smoothing_indices)
                    elif group.sp_count != len(term_smoothing_indices):
                        raise ValueError(
                            f"Linked smoothing id {group_id!r} expects {group.sp_count} "
                            f"smoothing parameters, got {len(term_smoothing_indices)}."
                        )
                    if not group.sp_indices:
                        group.sp_indices = list(term_smoothing_indices)
                    elif group.sp_indices != list(term_smoothing_indices):
                        raise ValueError(
                            f"Linked smoothing id {group_id!r} resolved to inconsistent "
                            f"smoothing indices {group.sp_indices} and "
                            f"{term_smoothing_indices}."
                        )
                    if smooth.term_id not in group.term_ids:
                        group.term_ids.append(str(smooth.term_id))
                    if smooth.label not in group.labels:
                        group.labels.append(str(smooth.label))

                term_meta = dict(smooth.metadata)
                term_meta["constructor_metadata"] = dict(smooth.constructor_metadata)
                term_blocks.append(
                    CompiledTerm(
                        label=smooth.label,
                        coef_slice=sl,
                        smooth=smooth,
                        basis_train=B,
                        basis_transform=np.eye(d, dtype=np.float64),
                        kept_columns=np.arange(d, dtype=int),
                        deleted_columns=np.array([], dtype=int),
                        smoothing_indices=term_smoothing_indices,
                        smoothing_ids=term_smoothing_ids,
                        n_penalties=len(penalty_defs),
                        term_type=str(smooth.term_type),
                        basis_name=str(smooth.basis_name),
                        term_id=smooth.term_id,
                        smoothing_group_id=smooth.smoothing_id,
                        metadata=term_meta,
                    )
                )
                design_blocks.append(B)
                start += d

        matrix_train = (
            np.column_stack(design_blocks)
            if design_blocks
            else np.empty((X.shape[0], 0), dtype=np.float64)
        )
        n_sp = len(smoothing_id_map)
        override_modes = [None] * n_sp
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


__all__ = ["compile_predictor_designs"]

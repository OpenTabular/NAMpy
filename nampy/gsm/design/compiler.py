from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..smooths.materialize import materialize_predictor_specs
from ..smooths.ids import attach_shared_basis_metadata
from ..smooths.construct import smooth_con
from .objects import PenaltyDefinition, PenaltyBlock, TermBlock, PredictorDesign


def _coerce_penalty_definition(obj, *, default_kind="smooth"):
    if isinstance(obj, PenaltyDefinition):
        out = obj
    else:
        out = PenaltyDefinition(matrix=np.asarray(obj, dtype=np.float64), kind=default_kind)

    P = np.asarray(out.matrix, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(
            f"Penalty matrices must be square 2D arrays, got shape={P.shape}."
        )

    if out.sp_mode not in {None, "fixed", "estimate"}:
        raise ValueError(
            "PenaltyDefinition.sp_mode must be one of {None, 'fixed', 'estimate'}."
        )

    if out.sp_mode == "fixed":
        if out.sp_value is None:
            raise ValueError("Fixed penalty overrides require sp_value.")
        if not np.isfinite(float(out.sp_value)) or float(out.sp_value) < 0:
            raise ValueError("Fixed sp_value must be finite and >= 0.")
    elif out.sp_value is not None:
        raise ValueError(
            "sp_value should be None unless sp_mode == 'fixed'."
        )

    return PenaltyDefinition(
        matrix=P,
        smoothing_id=out.smoothing_id,
        kind=str(out.kind),
        rank=out.rank,
        null_space_dim=out.null_space_dim,
        is_null_space_penalty=bool(out.is_null_space_penalty),
        sp_mode=out.sp_mode,
        sp_value=(None if out.sp_value is None else float(out.sp_value)),
        metadata=dict(out.metadata),
    )


def _default_penalty_id(pred_name, term, term_label, coef_start, local_penalty_index, n_penalties):
    base_id = getattr(term, "smoothing_id", None)
    if base_id is not None:
        base_id = str(base_id)
        if n_penalties <= 1:
            return base_id
        return f"{base_id}::{local_penalty_index}"

    return f"__auto__:{pred_name}:{term_label}:{coef_start}:{local_penalty_index}"


def _merge_smoothing_override(existing, mode, value, *, smoothing_id, label):
    if mode is None:
        return existing

    candidate = {
        "mode": str(mode),
        "value": (None if value is None else float(value)),
        "labels": [str(label)],
    }

    if existing is None:
        return candidate

    if existing["mode"] != candidate["mode"]:
        raise ValueError(
            f"Conflicting term-level sp overrides for smoothing_id={smoothing_id!r}: "
            f"got both {existing['mode']!r} and {candidate['mode']!r}."
        )

    if candidate["mode"] == "fixed":
        if not np.isclose(existing["value"], candidate["value"], atol=0.0, rtol=0.0):
            raise ValueError(
                f"Conflicting fixed term-level sp overrides for smoothing_id={smoothing_id!r}: "
                f"{existing['value']} vs {candidate['value']}."
            )

    existing["labels"].append(str(label))
    return existing


def compile_predictor_designs(X, feature_names, predictor_specs):
    """
    Fit all smooths and compile design / penalty metadata.

    Construction flow
    -----------------
    predictor specs
      -> shared-id basis setup metadata
      -> runtime term materialization
      -> smoothCon-style wrapper
      -> compiled design / penalty blocks
    """
    predictor_specs = attach_shared_basis_metadata(
        predictor_specs, X=X, feature_names=feature_names
    )
    predictor_specs = materialize_predictor_specs(predictor_specs)

    designs = []

    for pred_spec in predictor_specs:
        term_blocks = []
        penalty_blocks = []
        design_blocks = []

        smoothing_id_map = {}
        smoothing_override_by_id = {}

        start = 0

        for term_like in pred_spec.terms:
            constructed = smooth_con(
                term_like,
                X=X,
                feature_names=feature_names,
                absorb_cons=True,
                apply_by=True,
                null_space_penalty=False,
            )

            for smooth in constructed:
                B = np.asarray(smooth.X, dtype=np.float64)
                if B.ndim != 2:
                    raise ValueError(
                        f"Smooth {smooth.label!r} produced a non-2D basis matrix "
                        f"with shape={B.shape}."
                    )

                d = B.shape[1]
                sl = slice(start, start + d)

                penalty_defs = [
                    _coerce_penalty_definition(p) for p in smooth.penalty_definitions
                ]

                term_smoothing_indices = []
                term_smoothing_ids = []

                for j, pdef in enumerate(penalty_defs):
                    P = np.asarray(pdef.matrix, dtype=np.float64)
                    if P.shape != (d, d):
                        raise ValueError(
                            f"Penalty for smooth {smooth.label!r} has shape {P.shape}, "
                            f"but the smooth basis has width {d}."
                        )

                    sid = pdef.smoothing_id
                    if sid is None:
                        sid = _default_penalty_id(
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

                    smoothing_override_by_id[sid] = _merge_smoothing_override(
                        smoothing_override_by_id.get(sid, None),
                        pdef.sp_mode,
                        pdef.sp_value,
                        smoothing_id=sid,
                        label=smooth.label,
                    )

                    penalty_blocks.append(
                        PenaltyBlock(
                            label=smooth.label,
                            coef_slice=sl,
                            matrix=P,
                            smoothing_index=sp_idx,
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

                term_meta = dict(smooth.metadata)
                term_meta["constructor_metadata"] = dict(smooth.constructor_metadata)

                term_blocks.append(
                    TermBlock(
                        label=smooth.label,
                        coef_slice=sl,
                        smooth=smooth,
                        basis_train=B,
                        basis_transform=np.eye(d, dtype=np.float64),
                        original_n_coef=d,
                        kept_columns=np.arange(d, dtype=int),
                        deleted_columns=np.array([], dtype=int),
                        smoothing_indices=term_smoothing_indices,
                        smoothing_ids=term_smoothing_ids,
                        n_penalties=len(penalty_defs),
                        term_type=str(smooth.term_type),
                        basis_name=str(smooth.basis_name),
                        by_variable=smooth.by_variable,
                        term_id=smooth.smoothing_id,
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

        designs.append(
            PredictorDesign(
                name=pred_spec.name,
                term_blocks=term_blocks,
                penalty_blocks=penalty_blocks,
                matrix_train=matrix_train,
                n_coef=start,
                n_smoothing_params=len(smoothing_id_map),
                smoothing_id_map=smoothing_id_map,
                smoothing_override_modes=override_modes,
                smoothing_override_values=override_values,
                metadata=dict(getattr(pred_spec, "metadata", {}) or {}),
            )
        )

    return designs

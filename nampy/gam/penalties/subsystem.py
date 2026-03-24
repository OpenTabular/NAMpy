"""
Canonical penalty metadata and normalisation subsystem (architecture section 3.4).

Every PenaltySpec that leaves a runtime term must pass through
``normalize_penalty_spec`` before reaching the compiler or fitting code.
That function is the single source of truth for:

  - symmetrisation of the penalty matrix
  - rank and null-space dimension computation
  - validation of sp_mode / sp_value consistency

Higher-level helpers (``make_penalty_spec``, ``build_null_space_selection_spec``)
build on ``normalize_penalty_spec`` so callers never need to construct raw
PenaltySpec objects and call normalisation separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..design.structures import PenaltySpec
from .algebra import null_space_penalty_from_penalty, symmetrize_penalty


def penalty_rank_null_dim(P: np.ndarray, tol: float = 1e-10) -> tuple[int, int]:
    """Return (rank, null_space_dim) for a square penalty matrix."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Penalty matrices must be square.")
    if P.shape[0] == 0:
        return 0, 0
    evals = np.linalg.eigvalsh(symmetrize_penalty(P))
    tol_eff = tol * max(1.0, float(np.max(np.abs(evals))))
    rank = int(np.sum(evals > tol_eff))
    return rank, int(P.shape[0] - rank)


def normalize_penalty_spec(obj: PenaltySpec | np.ndarray, *, default_kind: str = "smooth") -> PenaltySpec:
    if isinstance(obj, PenaltySpec):
        spec = obj
        P = np.asarray(spec.matrix, dtype=np.float64)
    else:
        spec = PenaltySpec(matrix=np.asarray(obj, dtype=np.float64), kind=default_kind)
        P = np.asarray(spec.matrix, dtype=np.float64)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"Penalty matrices must be square 2D arrays, got shape={P.shape}.")
    P = symmetrize_penalty(P)
    rank, null_dim = penalty_rank_null_dim(P)

    if spec.sp_mode not in {None, "fixed", "estimate"}:
        raise ValueError("PenaltySpec.sp_mode must be one of {None, 'fixed', 'estimate'}.")
    if spec.sp_mode == "fixed":
        if spec.sp_value is None:
            raise ValueError("Fixed penalty overrides require sp_value.")
        if not np.isfinite(float(spec.sp_value)) or float(spec.sp_value) < 0:
            raise ValueError("Fixed sp_value must be finite and >= 0.")
    elif spec.sp_value is not None:
        raise ValueError("sp_value should be None unless sp_mode == 'fixed'.")

    return PenaltySpec(
        matrix=P,
        smoothing_id=spec.smoothing_id,
        kind=str(spec.kind),
        rank=(rank if spec.rank is None else spec.rank),
        null_space_dim=(null_dim if spec.null_space_dim is None else spec.null_space_dim),
        is_null_space_penalty=bool(spec.is_null_space_penalty),
        sp_mode=spec.sp_mode,
        sp_value=(None if spec.sp_value is None else float(spec.sp_value)),
        metadata=dict(spec.metadata),
    )


def make_penalty_spec(
    *,
    matrix: np.ndarray,
    smoothing_id: str | None,
    kind: str,
    sp_mode: str | None = None,
    sp_value: float | None = None,
    is_null_space_penalty: bool = False,
    metadata: dict[str, Any] | None = None,
) -> PenaltySpec:
    return normalize_penalty_spec(
        PenaltySpec(
            matrix=np.asarray(matrix, dtype=np.float64),
            smoothing_id=smoothing_id,
            kind=str(kind),
            is_null_space_penalty=bool(is_null_space_penalty),
            sp_mode=sp_mode,
            sp_value=sp_value,
            metadata=dict(metadata or {}),
        )
    )


def build_null_space_selection_spec(
    *,
    main_penalty: np.ndarray,
    smoothing_id: str | None,
    tol: float = 1e-10,
    metadata: dict[str, Any] | None = None,
) -> PenaltySpec | None:
    S0, meta = null_space_penalty_from_penalty(main_penalty, tol=tol)
    if int(meta["rank"]) <= 0:
        return None
    return make_penalty_spec(
        matrix=S0,
        smoothing_id=smoothing_id,
        kind="null_space",
        sp_mode=None,
        sp_value=None,
        is_null_space_penalty=True,
        metadata=dict(metadata or {}),
    )


def merge_smoothing_override(existing, mode, value, *, smoothing_id: str, label: str):
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


def default_penalty_id(pred_name: str, term, term_label: str, coef_start: int, local_penalty_index: int, n_penalties: int) -> str:
    base_id = getattr(term, "smoothing_id", None)
    if base_id is not None:
        base_id = str(base_id)
        return base_id if n_penalties <= 1 else f"{base_id}::{local_penalty_index}"
    return f"__auto__:{pred_name}:{term_label}:{coef_start}:{local_penalty_index}"


__all__ = [
    "penalty_rank_null_dim",
    "normalize_penalty_spec",
    "make_penalty_spec",
    "build_null_space_selection_spec",
    "merge_smoothing_override",
    "default_penalty_id",
]

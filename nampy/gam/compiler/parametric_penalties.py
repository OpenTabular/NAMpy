"""Parametric penalty assembly matching mgcv/R/mgcv.r::parametricPenalty."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from .structures import CompiledPenalty


def _unpack(spec):
    if isinstance(spec, Mapping):
        options = dict(spec)
        rank = options.pop("rank", None)
        sp = options.pop("sp", None)
        L = options.pop("L", None)
        if "penalties" in options:
            penalties = options.pop("penalties")
            if options:
                raise ValueError("para_pen cannot mix 'penalties' and named matrices")
        else:
            penalties = list(options.values())
    else:
        penalties, rank, sp, L = spec, None, None, None
    arr = np.asarray(penalties, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.size == 0:
        arr = np.empty((0, 0, 0))
    if arr.ndim != 3:
        raise ValueError("para_pen penalties must be square matrices")
    return list(arr), rank, sp, L


def add_parametric_penalties(predictor, para_pen):
    """Prepend penalties, preserving original parametric term assignments."""
    if para_pen is None:
        return predictor
    if not isinstance(para_pen, Mapping):
        raise TypeError(
            "para_pen must map parametric term labels to penalty specifications"
        )
    groups = {}
    for index, term in enumerate(predictor.compiled_terms):
        if term.term_type != "parametric":
            continue
        label = str(term.metadata.get("formula_term", term.label))
        groups.setdefault(label, []).append((index, term))
    blocks, modes, values, ids = [], [], [], {}
    records = []
    for label, terms in groups.items():
        if label not in para_pen or para_pen[label] is None:
            continue
        matrices, rank, sp, L = _unpack(para_pen[label])
        count = len(matrices)
        if rank is not None:
            rank = np.atleast_1d(np.asarray(rank, dtype=float))
            if rank.shape != (count,):
                raise ValueError("'rank' has wrong length in para_pen")
            if not np.isfinite(rank).all() or np.any(rank != np.floor(rank)):
                raise ValueError("para_pen ranks must be finite integers")
        if count == 0:
            continue
        start, stop = terms[0][1].coef_slice.start, terms[-1][1].coef_slice.stop
        width = stop - start
        if width != sum(t.basis_train.shape[1] for _, t in terms):
            raise ValueError(
                "Parametric term coefficients must form a contiguous block"
            )
        L = np.eye(count) if L is None else np.asarray(L, dtype=float)
        if L.ndim != 2 or L.shape[0] != count or not np.isfinite(L).all():
            raise ValueError("L has wrong dimension or nonfinite values in para_pen")
        # A shared one-hot map is exactly the existing linked-penalty contract.
        if not (np.all((L == 0) | (L == 1)) and np.all(L.sum(axis=1) == 1)):
            raise NotImplementedError("para_pen L currently requires one-hot rows")
        nsp = L.shape[1]
        if np.any(L.sum(axis=0) == 0):
            raise ValueError("para_pen L contains an unused smoothing parameter")
        sp = (
            np.full(nsp, -1.0)
            if sp is None
            else np.atleast_1d(np.asarray(sp, dtype=float))
        )
        if sp.shape != (nsp,) or not np.isfinite(sp).all():
            raise ValueError("'sp' dimension wrong or nonfinite in para_pen")
        base = len(modes)
        for j, value in enumerate(sp):
            name = label if nsp == 1 else f"{label}{j + 1}"
            ids[f"para_pen:{name}"] = base + j
            modes.append("fixed" if value >= 0 else None)
            values.append(float(value) if value >= 0 else np.nan)
        ranks = []
        for j, matrix in enumerate(matrices):
            if matrix.shape != (width, width) or not np.isfinite(matrix).all():
                raise ValueError(
                    "a parametric penalty has wrong dimension or nonfinite values"
                )
            if not np.allclose(matrix, matrix.T, atol=1e-14, rtol=1e-14):
                raise ValueError("a parametric penalty must be symmetric")
            ev = np.linalg.eigvalsh(matrix)
            rank_j = (
                int(np.sum(ev > np.max(ev) * np.finfo(float).eps * 10))
                if rank is None
                else int(np.atleast_1d(rank)[j])
            )
            if rank_j < 0 or rank_j > width:
                raise ValueError(
                    "para_pen rank must lie between zero and coefficient count"
                )
            ranks.append(rank_j)
            idx = base + int(np.argmax(L[j]))
            sid = next(k for k, v in ids.items() if v == idx)
            blocks.append(
                CompiledPenalty(
                    label=label,
                    coef_slice=slice(start, stop),
                    matrix=matrix.copy(),
                    smoothing_index=idx,
                    term_index=terms[0][0],
                    smoothing_id=sid,
                    kind="parametric",
                    rank=rank_j,
                    null_space_dim=width - rank_j,
                    sp_mode=modes[idx],
                    sp_value=values[idx] if modes[idx] else None,
                    metadata={"para_pen": True, "term_indices": [i for i, _ in terms]},
                )
            )
        records.append(
            {
                "label": label,
                "rank": ranks,
                "L": L.copy(),
                "sp": sp.copy(),
                "coef_slice": (start, stop),
            }
        )
    if not blocks:
        return predictor
    shift = len(modes)
    penalties = tuple(blocks) + tuple(
        replace(pb, smoothing_index=pb.smoothing_index + shift)
        for pb in predictor.compiled_penalties
    )
    terms = []
    for term in predictor.compiled_terms:
        terms.append(
            replace(term, smoothing_indices=[i + shift for i in term.smoothing_indices])
        )
    metadata = dict(predictor.metadata)
    metadata["para_pen"] = records
    metadata["n_para_pen"] = len(blocks)
    metadata["penalty_group_specs"] = [
        replace(group, sp_indices=[i + shift for i in group.sp_indices])
        for group in metadata.get("penalty_group_specs", [])
    ]
    metadata["s_id_to_sp_indices"] = {
        k: [i + shift for i in v]
        for k, v in metadata.get("s_id_to_sp_indices", {}).items()
    }
    ids.update({k: v + shift for k, v in predictor.smoothing_parameter_map.items()})
    old_values = predictor.smoothing_override_values
    return replace(
        predictor,
        compiled_terms=tuple(terms),
        compiled_penalties=penalties,
        smoothing_parameter_map=ids,
        n_smoothing_params=predictor.n_smoothing_params + shift,
        smoothing_override_modes=modes + list(predictor.smoothing_override_modes or []),
        smoothing_override_values=np.r_[
            values, [] if old_values is None else old_values
        ],
        metadata=metadata,
    )

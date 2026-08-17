"""Shared helpers for structural mgcv parity characterization tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nampy.gam.model.api import GAM
from nampy.gam.parity import (
    build_parity_snapshot,
    load_parity_snapshot,
    save_parity_snapshot,
)


def fit_gaussian_formula(data, formula: str, **gam_kwargs) -> GAM:
    """Fit :class:`~nampy.gam.model.api.GAM` with a formula (Gaussian family)."""
    kw = {
        "family": "gaussian",
        "formula": formula,
        "optimize_smoothing": True,
        "smoothing_method": "REML",
    }
    kw.update(gam_kwargs)
    m = GAM(**kw)
    m.fit(data=data)
    return m


def design_structure(model: GAM) -> dict[str, Any]:
    """Structural summary of the training design (no full numeric dumps)."""
    assert getattr(model, "_fitted", False)
    compiled = model.compiled_model_
    assert compiled is not None
    design = np.asarray(compiled.design_matrix, dtype=np.float64)
    n = int(design.shape[0])
    blocks: list[dict[str, Any]] = []
    for tb in compiled.compiled_terms:
        B = np.asarray(design[:, tb.coef_slice], dtype=np.float64)
        pbs = [
            pb for pb in compiled.compiled_penalties if pb.coef_slice == tb.coef_slice
        ]
        meta = dict(tb.metadata or {})
        cmeta = dict(getattr(tb, "constructor_metadata", {}) or {})
        blocks.append(
            {
                "label": tb.label,
                "term_type": tb.term_type,
                "basis_name": tb.basis_name,
                "train_shape": tuple(B.shape),
                "coef_width": int(B.shape[1]),
                "n_penalties": len(pbs),
                "penalty_shapes": [
                    tuple(np.asarray(pb.matrix, dtype=np.float64).shape) for pb in pbs
                ],
                "smoothing_ids": list(tb.smoothing_ids),
                "has_parametric_expansion_meta": "parametric_expansion" in meta,
                "runtime_by_name": cmeta.get("runtime_by_name"),
                "runtime_constraint_kind": cmeta.get("runtime_constraint_kind"),
                "by_handling": cmeta.get("by_handling"),
            }
        )
    return {
        "n_samples": n,
        "n_coef": int(compiled.n_coef),
        "n_smoothing_params": int(compiled.n_smoothing_params),
        "term_blocks": blocks,
    }


def smoothing_parameter_map_structure(model: GAM) -> dict[str, Any]:
    """How penalties map to global smoothing indices (linked ids share an index)."""
    compiled = model.compiled_model_
    assert compiled is not None
    if len(compiled.predictors) == 1:
        pmap = dict(compiled.predictors[0].smoothing_parameter_map)
    else:
        pmap = {}
        sp_shift = 0
        for predictor in compiled.predictors:
            for key, value in predictor.smoothing_parameter_map.items():
                pmap[f"{predictor.name}:{key}"] = sp_shift + int(value)
            sp_shift += int(predictor.n_smoothing_params)
    penalties = []
    for pb in compiled.compiled_penalties:
        penalties.append(
            {
                "label": pb.label,
                "smoothing_index": int(pb.smoothing_index),
                "smoothing_id": pb.smoothing_id,
            }
        )
    return {
        "smoothing_parameter_map": pmap,
        "n_smoothing_params": int(compiled.n_smoothing_params),
        "penalty_smoothing_indices": [p["smoothing_index"] for p in penalties],
        "penalty_smoothing_ids": [p["smoothing_id"] for p in penalties],
    }


def predict_type_shapes(model: GAM, X) -> dict[str, tuple[int, ...]]:
    """Shapes of ``predict`` / ``lpmatrix`` outputs on ``X``."""
    r = np.asarray(model.predict(X=X, type="response")).shape
    lk = np.asarray(model.predict(X=X, type="link")).shape
    tr = np.asarray(model.predict(X=X, type="terms")).shape
    lp = np.asarray(model.lpmatrix(X)).shape
    return {"response": r, "link": lk, "terms": tr, "lpmatrix": lp}


def term_contribution_shapes(model: GAM, X=None) -> dict[str, Any]:
    terms = np.asarray(model.predict(X=X, type="terms"))
    return {"terms": tuple(terms.shape)}


def parity_snapshot_structure(snap: dict) -> dict[str, Any]:
    """Shape-only view of a parity snapshot (fit scalars + prediction array shapes)."""
    fit = snap.get("fit", {})
    preds = snap.get("predictions", {})
    out: dict[str, Any] = {
        "fit": {
            "coef_full_shape": tuple(np.asarray(fit.get("coef_full")).shape),
            "smoothing_params_shape": tuple(
                np.asarray(fit.get("smoothing_params")).shape
            ),
            "edf_by_term_shape": tuple(np.asarray(fit.get("edf_by_term")).shape),
            "intercept": float(fit.get("intercept", 0.0)),
            "edf_total": float(fit.get("edf_total", 0.0)),
            "n_term_results": len(fit.get("term_results", []) or []),
        },
        "predictions": {
            k: tuple(np.asarray(preds.get(k)).shape)
            for k in ("response", "link", "terms", "lpmatrix")
        },
    }
    tr = fit.get("term_results") or []
    out["fit"]["term_summaries"] = [
        {
            "label": t.get("label"),
            "basis_name": t.get("basis_name"),
            "term_type": t.get("term_type"),
            "n_coef": int(t.get("n_coef", 0)),
            "smoothing_ids": list(t.get("smoothing_ids", [])),
        }
        for t in tr
    ]
    if "parity" in snap:
        out["parity_keys"] = list(snap["parity"].keys())
        cv = snap["parity"].get("criterion_view")
        if cv is not None:
            out["criterion_view_keys"] = sorted(cv.keys())
    return out


def assert_parity_structure_equal(actual_struct: dict, expected_struct: dict) -> None:
    """Assert two :func:`parity_snapshot_structure` dicts match."""
    assert actual_struct == expected_struct


def roundtrip_parity_snapshot(model: GAM, tmp_path: Path, *, X=None) -> None:
    """Save/load JSON and compare structural snapshot equality."""
    snap0 = build_parity_snapshot(model, X=X, include_covariances=False)
    path = tmp_path / "snap.json"
    save_parity_snapshot(snap0, path)
    snap1 = load_parity_snapshot(path)
    assert_parity_structure_equal(
        parity_snapshot_structure(snap0), parity_snapshot_structure(snap1)
    )


__all__ = [
    "assert_parity_structure_equal",
    "design_structure",
    "fit_gaussian_formula",
    "parity_snapshot_structure",
    "predict_type_shapes",
    "roundtrip_parity_snapshot",
    "smoothing_parameter_map_structure",
    "term_contribution_shapes",
]

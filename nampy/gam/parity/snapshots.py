"""Parity snapshot helpers.

Core fit serialization is intentionally kept semantic-free:
- ``fit_result.to_dict()`` reflects the model's fit state only.
- parity-only recomputations live under the top-level ``parity`` section.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..smoothing_selection.criteria import (
    criterion_ml_reml,
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_pirls,
    resolve_ml_reml_scoring_backend,
)


def _coerce_snapshot_arrays(snapshot):
    out = dict(snapshot)
    fit = dict(out.get("fit", {}))
    preds = dict(out.get("predictions", {}))

    for key in (
        "coef_full",
        "smoothing_params",
        "edf_by_term",
        "cov_bayes",
        "cov_freq",
    ):
        if key in fit and fit[key] is not None:
            fit[key] = np.asarray(fit[key], dtype=np.float64)

    for key in ("response", "link", "terms", "lpmatrix"):
        if key in preds and preds[key] is not None:
            preds[key] = np.asarray(preds[key], dtype=np.float64)

    out["fit"] = fit
    out["predictions"] = preds
    return out


def _get_core(model):
    if (
        hasattr(model, "_fitted")
        and hasattr(model, "fit_result")
        and hasattr(model, "predict")
        and hasattr(model, "lpmatrix")
    ):
        return model
    if hasattr(model, "core_") and model.core_ is not None:
        return model.core_
    if hasattr(model, "model") and hasattr(model.model, "core_") and model.model.core_ is not None:
        return model.model.core_
    if (
        hasattr(model, "model")
        and model.model is not None
        and hasattr(model.model, "_fitted")
        and hasattr(model.model, "fit_result")
        and hasattr(model.model, "predict")
        and hasattr(model.model, "lpmatrix")
    ):
        return model.model
    raise TypeError(
        "Expected a fitted GAM-like object exposing fit/predict/lpmatrix APIs "
        "directly or via `.core_` / `.model`."
    )


def _build_parity_criterion_view(core, fit_dict):
    criterion_name = fit_dict.get("criterion_name", None)
    view = {
        "criterion_name": criterion_name,
        "stored_criterion_value": fit_dict.get("criterion_value", None),
        "recomputed_criterion_value": None,
        "recomputed_criterion_source": None,
        "criterion_backend": None,
    }

    if (
        criterion_name is None
        or str(fit_dict.get("family_name", "")).lower() != "gaussian"
        or str(criterion_name).lower() not in {"ml", "reml", "laml"}
    ):
        return view

    fixed_mask = (
        np.zeros(core.n_smoothing_params_, dtype=bool)
        if core.smoothing_fixed_mask_ is None
        else np.asarray(core.smoothing_fixed_mask_, dtype=bool)
    )
    free_vals = np.asarray(core.smoothing_params[~fixed_mask], dtype=np.float64)
    log_free = np.log(free_vals) if free_vals.size > 0 else np.empty((0,), dtype=np.float64)
    branch_method = "REML" if str(criterion_name).lower() in {"reml", "laml"} else "ML"
    backend = resolve_ml_reml_scoring_backend(core, method=str(criterion_name).lower())
    view["criterion_backend"] = backend

    joint_s2 = getattr(core, "_gaussian_reml_sigma2_opt_", None)
    if joint_s2 is not None and np.isfinite(float(joint_s2)):
        score_joint = getattr(core, "smoothing_score_", None)
        candidate = (
            float(score_joint)
            if score_joint is not None and np.isfinite(float(score_joint))
            else None
        )
        source = "smoothing_score"
        if candidate is None and log_free.size > 0:
            try:
                log_s2 = float(np.log(max(float(joint_s2), 1e-300)))
                jm = "LAML" if str(criterion_name).lower() == "laml" else branch_method
                candidate = float(
                    criterion_ml_reml_gaussian_dynamic_joint(
                        core,
                        core.y_,
                        log_free,
                        log_s2,
                        method=jm,
                    )
                )
                source = "gaussian_dynamic_joint"
            except Exception:
                candidate = None
                source = None
    elif backend in {"gaussian_dynamic", "pirls_laplace_dynamic"}:
        candidate = float(criterion_ml_reml(core, core.y_, log_free, method=branch_method))
        source = "criterion_ml_reml"
    else:
        candidate = float(criterion_ml_reml_pirls(core, core.y_, log_free, method=branch_method))
        source = "criterion_ml_reml_pirls"

    view["recomputed_criterion_value"] = candidate
    view["recomputed_criterion_source"] = source
    return view


def build_parity_snapshot(model, X=None, include_covariances=False):
    core = _get_core(model)

    if not getattr(core, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    fit_result = core.fit_result(include_covariances=include_covariances)
    fit_dict = fit_result.to_dict(include_covariances=include_covariances)
    parity_view = _build_parity_criterion_view(core, fit_dict)

    predict_api = model if (hasattr(model, "predict") and hasattr(model, "lpmatrix")) else core

    if X is None:
        response = core.predict(X=None, type="response")
        link = core.predict(X=None, type="link")
        terms = core.predict(X=None, type="terms")
        lpmatrix = core.lpmatrix(core.X_)
    else:
        response = predict_api.predict(X=X, type="response")
        link = predict_api.predict(X=X, type="link")
        terms = predict_api.predict(X=X, type="terms")
        lpmatrix = predict_api.lpmatrix(X)

    return {
        "fit": fit_dict,
        "predictions": {
            "response": np.asarray(response, dtype=np.float64).tolist(),
            "link": np.asarray(link, dtype=np.float64).tolist(),
            "terms": np.asarray(terms, dtype=np.float64).tolist(),
            "lpmatrix": np.asarray(lpmatrix, dtype=np.float64).tolist(),
        },
        "parity": {
            "criterion_view": parity_view,
        },
    }


def save_parity_snapshot(snapshot, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def load_parity_snapshot(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    return _coerce_snapshot_arrays(snap)


__all__ = [
    "build_parity_snapshot",
    "save_parity_snapshot",
    "load_parity_snapshot",
    "_coerce_snapshot_arrays",
    "_get_core",
]

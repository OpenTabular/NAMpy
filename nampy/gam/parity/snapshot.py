import json
from pathlib import Path

import numpy as np

from ..smoothness.criteria import criterion_ml_reml_pirls


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


def build_parity_snapshot(model, X=None, include_covariances=False):
    core = _get_core(model)

    if not getattr(core, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    fit_result = core.fit_result(include_covariances=include_covariances)
    fit_dict = fit_result.to_dict(include_covariances=include_covariances)

    criterion_name = fit_dict.get("criterion_name", None)
    if (
        criterion_name is not None
        and str(fit_dict.get("family_name", "")).lower() == "gaussian"
        and str(criterion_name).lower() in {"ml", "reml", "laml"}
    ):
        fixed_mask = (
            np.zeros(core.n_smoothing_params_, dtype=bool)
            if core.smoothing_fixed_mask_ is None
            else np.asarray(core.smoothing_fixed_mask_, dtype=bool)
        )
        free_vals = np.asarray(core.smoothing_params[~fixed_mask], dtype=np.float64)
        log_free = (
            np.log(free_vals)
            if free_vals.size > 0
            else np.empty((0,), dtype=np.float64)
        )
        branch_method = "REML" if str(criterion_name).lower() in {"reml", "laml"} else "ML"
        fit_dict["criterion_value"] = float(
            criterion_ml_reml_pirls(core, core.y_, log_free, method=branch_method)
        )

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

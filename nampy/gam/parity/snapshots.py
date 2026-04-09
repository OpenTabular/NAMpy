"""Parity snapshot helpers.

Core fit serialization is intentionally kept semantic-free:
- ``fit_result.to_dict()`` reflects the model's fit state only.
- parity-only recomputations live under the top-level ``parity`` section.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.linalg import qr

from ..smoothing_selection.criteria import (
    criterion_ml_reml,
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_exact_joint,
    criterion_ml_reml_pirls,
    resolve_ml_reml_scoring_backend,
)
from ..smoothing_selection.postfit import optimizer_endpoint_diagnostics


def _intercept_offset(model) -> int:
    """Return 1 if model was fit with an intercept column in X, else 0."""
    return 1 if bool(getattr(model, "fit_intercept", False)) else 0


def _coerce_snapshot_arrays(snapshot):
    out = dict(snapshot)
    fit = dict(out.get("fit", {}))
    preds = dict(out.get("predictions", {}))
    parity = dict(out.get("parity", {}))
    diagnostics = dict(parity.get("diagnostics", {}))

    for key in (
        "coef_full",
        "smoothing_params",
        "log_smoothing_params",
        "edf_by_term",
        "cov_bayes",
        "cov_freq",
        "outer_grad",
        "outer_hess",
    ):
        if key in fit and fit[key] is not None:
            fit[key] = np.asarray(fit[key], dtype=np.float64)

    for key in ("response", "link", "terms", "lpmatrix", "se_response", "se_link"):
        if key in preds and preds[key] is not None:
            preds[key] = np.asarray(preds[key], dtype=np.float64)

    if diagnostics.get("concurvity_full", None) is not None:
        diagnostics["concurvity_full"] = np.asarray(
            diagnostics["concurvity_full"], dtype=np.float64
        )
    pairwise = diagnostics.get("concurvity_pairwise", None)
    if pairwise is not None:
        pairwise = dict(pairwise)
        for key, vals in list(pairwise.items()):
            if key == "labels" or vals is None:
                continue
            pairwise[key] = np.asarray(vals, dtype=np.float64)
        diagnostics["concurvity_pairwise"] = pairwise
    if diagnostics.get("sp_vcov", None) is not None:
        diagnostics["sp_vcov"] = np.asarray(diagnostics["sp_vcov"], dtype=np.float64)
    endpoint_block = diagnostics.get("optimizer_endpoint", None)
    if endpoint_block is not None:
        endpoint_block = dict(endpoint_block)
        for key in (
            "log_smoothing_params",
            "bounds",
            "gradient",
            "projected_gradient",
            "hessian",
            "hessian_eigenvalues",
            "at_lower_bound",
            "at_upper_bound",
        ):
            vals = endpoint_block.get(key, None)
            if vals is not None:
                endpoint_block[key] = np.asarray(vals, dtype=np.float64)
        diagnostics["optimizer_endpoint"] = endpoint_block
    if diagnostics.get("one_se_rule", None) is not None:
        diagnostics["one_se_rule"] = np.asarray(
            diagnostics["one_se_rule"], dtype=np.float64
        )
    if diagnostics.get("gam_vcomp", None) is not None:
        diagnostics["gam_vcomp"] = np.asarray(
            diagnostics["gam_vcomp"], dtype=np.float64
        )
    residuals_block = diagnostics.get("residuals", None)
    if residuals_block is not None:
        residuals_block = dict(residuals_block)
        for key in (
            "response",
            "working",
            "pearson",
            "scaled_pearson",
            "deviance",
        ):
            vals = residuals_block.get(key, None)
            if vals is not None:
                residuals_block[key] = np.asarray(vals, dtype=np.float64)
        diagnostics["residuals"] = residuals_block
    k_check_block = diagnostics.get("k_check", None)
    if k_check_block is not None:
        k_check_block = dict(k_check_block)
        labels = k_check_block.get("labels", None)
        if labels is not None and not isinstance(labels, list):
            k_check_block["labels"] = [labels]
        vals = k_check_block.get("values", None)
        if vals is not None:
            k_check_block["values"] = np.asarray(vals, dtype=np.float64)
        diagnostics["k_check"] = k_check_block
    for key in ("smooth_cov_bayes", "smooth_cov_freq"):
        block = diagnostics.get(key, None)
        if block is not None:
            block = dict(block)
            labels = block.get("labels", None)
            if labels is not None and not isinstance(labels, list):
                block["labels"] = [labels]
            mats = block.get("blocks", None)
            if mats is not None:
                block["blocks"] = [np.asarray(m, dtype=np.float64) for m in mats]
            diagnostics[key] = block
    edf1_block = diagnostics.get("smooth_edf1", None)
    if edf1_block is not None:
        edf1_block = dict(edf1_block)
        labels = edf1_block.get("labels", None)
        if labels is not None and not isinstance(labels, list):
            edf1_block["labels"] = [labels]
        if edf1_block.get("values", None) is not None:
            edf1_block["values"] = np.asarray(edf1_block["values"], dtype=np.float64)
        diagnostics["smooth_edf1"] = edf1_block
    sti = diagnostics.get("smooth_test_inputs", None)
    if sti is not None:
        sti = dict(sti)
        labels = sti.get("labels", None)
        if labels is not None and not isinstance(labels, list):
            sti["labels"] = [labels]
        for key in ("coef_blocks", "r_blocks"):
            mats = sti.get(key, None)
            if mats is not None:
                sti[key] = [np.asarray(m, dtype=np.float64) for m in mats]
        for key in ("edf", "edf1"):
            vals = sti.get(key, None)
            if vals is not None:
                sti[key] = np.asarray(vals, dtype=np.float64)
        if sti.get("residual_df", None) is not None:
            sti["residual_df"] = float(sti["residual_df"])
        diagnostics["smooth_test_inputs"] = sti
    sfs = diagnostics.get("smooth_function_space", None)
    if sfs is not None:
        sfs = dict(sfs)
        labels = sfs.get("labels", None)
        if labels is not None and not isinstance(labels, list):
            sfs["labels"] = [labels]
        for key in ("fitted", "variance_diag"):
            vals = sfs.get(key, None)
            if vals is not None:
                if not isinstance(vals, list) or (
                    len(sfs.get("labels", [])) == 1
                    and vals
                    and not isinstance(vals[0], (list, tuple))
                ):
                    vals = [vals]
                sfs[key] = [np.asarray(v, dtype=np.float64) for v in vals]
        diagnostics["smooth_function_space"] = sfs
    for key in ("anova_parametric", "anova_smooth"):
        block = diagnostics.get(key, None)
        if block is not None:
            block = dict(block)
            labels = block.get("labels", None)
            if labels is not None and not isinstance(labels, list):
                block["labels"] = [labels]
            if block.get("values", None) is not None:
                block["values"] = np.asarray(block["values"], dtype=np.float64)
            diagnostics[key] = block

    out["fit"] = fit
    out["predictions"] = preds
    parity["diagnostics"] = diagnostics
    out["parity"] = parity
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
    if (
        hasattr(model, "model")
        and hasattr(model.model, "core_")
        and model.model.core_ is not None
    ):
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


def _normalize_mgcv_term_label(label):
    if label is None:
        return None
    text = str(label)
    text = re.sub(r",\s*bs\s*=\s*(\"[^\"]*\"|'[^']*'|[^,)]+)", "", text)
    text = re.sub(r",\s*k\s*=\s*[^,)]+", "", text)
    return text


def _residual_df_from_snapshot_state(core) -> float:
    H = np.asarray(core._H_coef, dtype=np.float64)
    edf1 = 2.0 * np.diag(H) - np.sum(H * H.T, axis=1)
    intercept_df = 1.0 if bool(getattr(core, "fit_intercept", False)) else 0.0
    return float(core.n_samples_) - intercept_df - float(np.sum(edf1))


def _build_parity_criterion_view(core, fit_dict):
    criterion_name = fit_dict.get("criterion_name", None)
    view = {
        "criterion_name": criterion_name,
        "stored_criterion_value": fit_dict.get("criterion_value", None),
        "recomputed_criterion_value": None,
        "recomputed_criterion_source": None,
        "criterion_backend": None,
        "profiled_criterion_value": None,
        "joint_criterion_value": None,
        "joint_log_sigma2": None,
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
    log_free = (
        np.log(free_vals) if free_vals.size > 0 else np.empty((0,), dtype=np.float64)
    )
    branch_method = "REML" if str(criterion_name).lower() in {"reml", "laml"} else "ML"
    backend = resolve_ml_reml_scoring_backend(core, method=str(criterion_name).lower())
    view["criterion_backend"] = backend
    score_joint = getattr(core, "smoothing_score_", None)
    joint_s2 = getattr(core, "_gaussian_reml_sigma2_opt_", None)
    if joint_s2 is not None and np.isfinite(float(joint_s2)):
        view["joint_log_sigma2"] = float(np.log(max(float(joint_s2), 1e-300)))
    if backend in {"gaussian_exact", "gaussian_dynamic"} and log_free.size > 0:
        try:
            view["profiled_criterion_value"] = float(
                criterion_ml_reml(core, core.y_, log_free, method=branch_method)
            )
        except Exception:
            view["profiled_criterion_value"] = None
    if backend == "gaussian_exact" and log_free.size > 0 and joint_s2 is not None:
        try:
            view["joint_criterion_value"] = float(
                criterion_ml_reml_gaussian_exact_joint(
                    core,
                    core.y_,
                    log_free,
                    float(np.log(max(float(joint_s2), 1e-300))),
                    method=(
                        "LAML"
                        if str(criterion_name).lower() == "laml"
                        else branch_method
                    ),
                )
            )
        except Exception:
            view["joint_criterion_value"] = None
    elif backend == "gaussian_dynamic" and log_free.size > 0 and joint_s2 is not None:
        try:
            view["joint_criterion_value"] = float(
                criterion_ml_reml_gaussian_dynamic_joint(
                    core,
                    core.y_,
                    log_free,
                    float(np.log(max(float(joint_s2), 1e-300))),
                    method=(
                        "LAML"
                        if str(criterion_name).lower() == "laml"
                        else branch_method
                    ),
                )
            )
        except Exception:
            view["joint_criterion_value"] = None
    if score_joint is not None and np.isfinite(float(score_joint)):
        view["recomputed_criterion_value"] = float(score_joint)
        view["recomputed_criterion_source"] = "smoothing_score"
        return view

    if joint_s2 is not None and np.isfinite(float(joint_s2)):
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
    elif backend in {"gaussian_exact", "gaussian_dynamic", "pirls_laplace_dynamic"}:
        candidate = float(
            criterion_ml_reml(core, core.y_, log_free, method=branch_method)
        )
        source = "criterion_ml_reml"
    else:
        candidate = float(
            criterion_ml_reml_pirls(core, core.y_, log_free, method=branch_method)
        )
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
    if hasattr(core.family, "theta"):
        fit_dict["family_theta"] = float(core.family.theta)
    else:
        fit_dict["family_theta"] = None
    fit_dict["estimate_theta"] = bool(getattr(core.family, "estimate_theta", False))
    if fit_dict.get("smoothing_params", None) is not None:
        sp = np.asarray(fit_dict["smoothing_params"], dtype=np.float64)
        fit_dict["log_smoothing_params"] = np.log(np.maximum(sp, 1e-300)).tolist()
    parity_view = _build_parity_criterion_view(core, fit_dict)

    predict_api = (
        model if (hasattr(model, "predict") and hasattr(model, "lpmatrix")) else core
    )

    if X is None:
        response = core.predict(X=None, type="response")
        link = core.predict(X=None, type="link")
        terms = core.predict(X=None, type="terms")
        lpmatrix = core.lpmatrix(core.X_)
        _, se_response = core.predict(X=None, type="response", return_se=True)
        _, se_link = core.predict(X=None, type="link", return_se=True)
    else:
        response = predict_api.predict(X=X, type="response")
        link = predict_api.predict(X=X, type="link")
        terms = predict_api.predict(X=X, type="terms")
        lpmatrix = predict_api.lpmatrix(X)
        _, se_response = predict_api.predict(X=X, type="response", return_se=True)
        _, se_link = predict_api.predict(X=X, type="link", return_se=True)

    diagnostics = {}

    smooth_labels = []
    smooth_cov_bayes = []
    smooth_cov_freq = []
    smooth_edf1_vals = []
    edf1_vec = None
    if len(getattr(core, "term_blocks_", ()) or ()) > 0:
        try:
            H = np.asarray(core._H_coef, dtype=np.float64)
            edf1_vec = 2.0 * np.diag(H) - np.sum(H * H.T, axis=1)
        except Exception:
            edf1_vec = None
        x_off = _intercept_offset(core)
        for tb in getattr(core, "term_blocks_", ()) or ():
            if str(getattr(tb, "term_type", "")) == "parametric":
                continue
            sl = tb.coef_slice
            # sl indexes coef_ (no intercept); Vp_, Vf_, H include the intercept column
            x_sl = slice(sl.start + x_off, sl.stop + x_off)
            smooth_labels.append(_normalize_mgcv_term_label(tb.label))
            if getattr(core, "Vp_", None) is not None:
                smooth_cov_bayes.append(
                    np.asarray(core.Vp_[x_sl, x_sl], dtype=np.float64)
                )
            if getattr(core, "Vf_", None) is not None:
                smooth_cov_freq.append(
                    np.asarray(core.Vf_[x_sl, x_sl], dtype=np.float64)
                )
            if edf1_vec is not None:
                smooth_edf1_vals.append(float(np.sum(edf1_vec[x_sl])))

    diagnostics["smooth_cov_bayes"] = (
        None
        if len(smooth_cov_bayes) == 0
        else {
            "labels": list(smooth_labels),
            "blocks": [m.tolist() for m in smooth_cov_bayes],
        }
    )
    diagnostics["smooth_cov_freq"] = (
        None
        if len(smooth_cov_freq) == 0
        else {
            "labels": list(smooth_labels),
            "blocks": [m.tolist() for m in smooth_cov_freq],
        }
    )
    diagnostics["smooth_edf1"] = (
        None
        if len(smooth_edf1_vals) == 0
        else {
            "labels": list(smooth_labels),
            "values": np.asarray(smooth_edf1_vals, dtype=np.float64).tolist(),
        }
    )
    if len(smooth_labels) > 0:
        X_full = np.asarray(core.fit_state_.X, dtype=np.float64)
        R_full = getattr(core, "_summary_R_", None)
        if R_full is None:
            W_full = np.asarray(core.fit_state_.working_weights, dtype=np.float64)
            weighted_X = np.sqrt(np.clip(W_full, 0.0, None))[:, None] * X_full
            _, R_full = qr(weighted_X, mode="economic")
        diagnostics["smooth_test_inputs"] = {
            "labels": list(smooth_labels),
            "coef_blocks": [
                np.asarray(core.coef_[tb.coef_slice], dtype=np.float64).tolist()
                for tb in getattr(core, "term_blocks_", ()) or ()
                if str(getattr(tb, "term_type", "")) != "parametric"
            ],
            "r_blocks": [
                np.asarray(
                    R_full[
                        :,
                        int(tb.coef_slice.start)
                        + _intercept_offset(core) : int(tb.coef_slice.stop)
                        + _intercept_offset(core),
                    ],
                    dtype=np.float64,
                ).tolist()
                for tb in getattr(core, "term_blocks_", ()) or ()
                if str(getattr(tb, "term_type", "")) != "parametric"
            ],
            "edf": np.asarray(
                [
                    float(core.edf_by_term_[i])
                    for i, tb in enumerate(getattr(core, "term_blocks_", ()) or ())
                    if str(getattr(tb, "term_type", "")) != "parametric"
                ],
                dtype=np.float64,
            ).tolist(),
            "edf1": np.asarray(smooth_edf1_vals, dtype=np.float64).tolist(),
            "residual_df": float(core.n_samples_)
            - (1.0 if bool(getattr(core, "fit_intercept", False)) else 0.0)
            - float(core.edf_),
        }
        diagnostics["smooth_function_space"] = {
            "labels": list(smooth_labels),
            "fitted": [
                np.asarray(terms[:, i], dtype=np.float64).tolist()
                for i, tb in enumerate(getattr(core, "term_blocks_", ()) or ())
                if str(getattr(tb, "term_type", "")) != "parametric"
            ],
            "variance_diag": [
                np.asarray(
                    np.sum(
                        (
                            np.asarray(
                                lpmatrix[
                                    :,
                                    int(tb.coef_slice.start) : int(tb.coef_slice.stop),
                                ],
                                dtype=np.float64,
                            )
                            @ np.asarray(
                                core.Vp_[tb.coef_slice, tb.coef_slice], dtype=np.float64
                            )
                        )
                        * np.asarray(
                            lpmatrix[
                                :,
                                int(tb.coef_slice.start) : int(tb.coef_slice.stop),
                            ],
                            dtype=np.float64,
                        ),
                        axis=1,
                    ),
                    dtype=np.float64,
                ).tolist()
                for tb in getattr(core, "term_blocks_", ()) or ()
                if str(getattr(tb, "term_type", "")) != "parametric"
            ],
        }
    else:
        diagnostics["smooth_test_inputs"] = None
        diagnostics["smooth_function_space"] = None
    try:
        conc = predict_api.concurvity(full=True)
        diagnostics["concurvity_labels"] = [
            _normalize_mgcv_term_label(v) for v in conc["labels"]
        ]
        diagnostics["concurvity_full"] = np.asarray(
            conc["values"], dtype=np.float64
        ).tolist()
    except Exception:
        diagnostics["concurvity_labels"] = None
        diagnostics["concurvity_full"] = None

    try:
        conc = predict_api.concurvity(full=False)
        diagnostics["concurvity_pairwise"] = {
            "labels": [_normalize_mgcv_term_label(v) for v in conc["labels"]],
            **{
                name: np.asarray(mat, dtype=np.float64).tolist()
                for name, mat in conc["values"].items()
            },
        }
    except Exception:
        diagnostics["concurvity_pairwise"] = None

    try:
        V = predict_api.sp_vcov(edge_correct=False)
        diagnostics["sp_vcov"] = (
            None if V is None else np.asarray(V, dtype=np.float64).tolist()
        )
    except Exception:
        diagnostics["sp_vcov"] = None

    try:
        endpoint_diag = optimizer_endpoint_diagnostics(core)
        diagnostics["optimizer_endpoint"] = endpoint_diag
    except Exception:
        diagnostics["optimizer_endpoint"] = None

    try:
        vc = predict_api.gam_vcomp(rescale=False)
        if vc is not None and isinstance(vc, dict) and "vc" in vc:
            diagnostics["gam_vcomp"] = np.asarray(vc["vc"], dtype=np.float64).tolist()
            diagnostics["gam_vcomp_names"] = list(vc.get("names", []))
        else:
            diagnostics["gam_vcomp"] = None
            diagnostics["gam_vcomp_names"] = None
    except Exception:
        diagnostics["gam_vcomp"] = None
        diagnostics["gam_vcomp_names"] = None

    try:
        one_se = predict_api.one_se_rule()
        diagnostics["one_se_rule"] = np.asarray(one_se, dtype=np.float64).tolist()
    except Exception:
        diagnostics["one_se_rule"] = None

    try:
        diagnostics["residuals"] = {
            "response": np.asarray(
                predict_api.residuals(type="response"), dtype=np.float64
            ).tolist(),
            "working": np.asarray(
                predict_api.residuals(type="working"), dtype=np.float64
            ).tolist(),
            "pearson": np.asarray(
                predict_api.residuals(type="pearson"), dtype=np.float64
            ).tolist(),
            "scaled_pearson": np.asarray(
                predict_api.residuals(type="scaled.pearson"), dtype=np.float64
            ).tolist(),
            "deviance": np.asarray(
                predict_api.residuals(type="deviance"), dtype=np.float64
            ).tolist(),
        }
    except Exception:
        diagnostics["residuals"] = None

    try:
        k_tab = predict_api.k_check(subsample=120, n_rep=8, seed=0)
        diagnostics["k_check"] = (
            None
            if k_tab is None
            else {
                "labels": [_normalize_mgcv_term_label(v) for v in k_tab.index.tolist()],
                "values": k_tab[["k_prime", "edf", "k_index", "p_value"]]
                .to_numpy(dtype=np.float64)
                .tolist(),
            }
        )
    except Exception:
        diagnostics["k_check"] = None

    try:
        anova_res = predict_api.anova(freq=False)
        diagnostics["anova_parametric"] = {
            "labels": [
                _normalize_mgcv_term_label(v)
                for v in anova_res.parametric_table["label"].tolist()
            ],
            "values": anova_res.parametric_table[["df", "wald_stat", "p_value"]]
            .to_numpy(dtype=np.float64)
            .tolist(),
        }
        diagnostics["anova_smooth"] = {
            "labels": [
                _normalize_mgcv_term_label(v)
                for v in anova_res.smooth_table["label"].tolist()
            ],
            "values": anova_res.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]]
            .to_numpy(dtype=np.float64)
            .tolist(),
        }
    except Exception:
        diagnostics["anova_parametric"] = None
        diagnostics["anova_smooth"] = None

    return {
        "fit": fit_dict,
        "predictions": {
            "response": np.asarray(response, dtype=np.float64).tolist(),
            "link": np.asarray(link, dtype=np.float64).tolist(),
            "terms": np.asarray(terms, dtype=np.float64).tolist(),
            "lpmatrix": np.asarray(lpmatrix, dtype=np.float64).tolist(),
            "se_response": np.asarray(se_response, dtype=np.float64).tolist(),
            "se_link": np.asarray(se_link, dtype=np.float64).tolist(),
        },
        "parity": {
            "criterion_view": parity_view,
            "diagnostics": diagnostics,
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

"""
General-family fixed-smoothing backend using mgcv-style ``gam.fit5``.

Mirrors mgcv ``gam.fit5`` / ``gam.fit5.post.proc`` from ``mgcv/R/gam.fit4.r``
for multi-linear-predictor GAMLSS-style families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._model_state import _penalty_blocks_seq, _predictor_designs
from ..model_ops import expand_smoothing_params_from_log
from ..state import FitCoreSolution
from .gam_fit5 import GamFit5Control, gam_fit5, gam_fit5_post_proc


@dataclass
class _GeneralPredictorLayout:
    X_full: np.ndarray
    jj: list[np.ndarray]
    reduced_to_full_idx: np.ndarray
    predictor_full_slices: list[slice]


def _build_general_predictor_layout(model) -> _GeneralPredictorLayout:
    blocks = []
    jj: list[np.ndarray] = []
    predictor_full_slices: list[slice] = []
    reduced_to_full: list[int] = []
    full_start = 0
    reduced_start = 0

    for pred in _predictor_designs(model):
        Z = np.asarray(pred.design_matrix, dtype=np.float64)
        if bool(pred.has_intercept):
            Xp = np.column_stack([np.ones(Z.shape[0], dtype=np.float64), Z])
            local_idx = np.arange(full_start, full_start + Z.shape[1] + 1, dtype=int)
            reduced_to_full.extend(
                list(
                    np.arange(
                        full_start + 1,
                        full_start + 1 + Z.shape[1],
                        dtype=int,
                    )
                )
            )
        else:
            Xp = Z
            local_idx = np.arange(full_start, full_start + Z.shape[1], dtype=int)
            reduced_to_full.extend(
                list(np.arange(full_start, full_start + Z.shape[1], dtype=int))
            )
        blocks.append(Xp)
        jj.append(local_idx)
        predictor_full_slices.append(slice(full_start, full_start + Xp.shape[1]))
        full_start += Xp.shape[1]
        reduced_start += Z.shape[1]

    X_full = np.column_stack(blocks) if blocks else np.empty((model.n_samples_, 0))
    return _GeneralPredictorLayout(
        X_full=np.asarray(X_full, dtype=np.float64),
        jj=jj,
        reduced_to_full_idx=np.asarray(reduced_to_full, dtype=int),
        predictor_full_slices=predictor_full_slices,
    )


def _build_general_penalty_matrix(
    model, smoothing_params, layout
) -> tuple[np.ndarray, list[np.ndarray]]:
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64).ravel()
    p_full = layout.X_full.shape[1]
    St = np.zeros((p_full, p_full), dtype=np.float64)
    S_blocks: list[np.ndarray] = []

    full_idx = layout.reduced_to_full_idx
    for pb in _penalty_blocks_seq(model):
        S_full = np.zeros((p_full, p_full), dtype=np.float64)
        idx = full_idx[pb.coef_slice]
        S_full[np.ix_(idx, idx)] = np.asarray(pb.matrix, dtype=np.float64)
        S_blocks.append(S_full)
        St += float(smoothing_params[pb.smoothing_index]) * S_full

    return St, S_blocks


def _offset_list(model, n_pred: int):
    offset = getattr(model, "offset_train_", None)
    if offset is None:
        return None
    if isinstance(offset, (list, tuple)):
        return list(offset)
    return [np.asarray(offset, dtype=np.float64)] + [None] * (n_pred - 1)


def _general_fit_score_type_name(method: str) -> str:
    method_l = str(method).lower()
    if method_l in {"reml", "laml"}:
        return "REML"
    if method_l == "ml":
        return "ML"
    return method_l.upper()


def _record_outer_derivative_mode(model, *, gradient_source=None, hessian_source=None):
    info = dict(getattr(model, "_general_fit5_outer_derivative_info", {}) or {})
    family = getattr(model, "family", None)
    family_name = str(getattr(family, "name", "")).lower()
    supports_analytic = bool(
        getattr(family, "supports_analytic_outer_derivatives", False)
    )
    supports_analytic_hessian = _supports_analytic_outer_hessian(family)
    if gradient_source is not None:
        info["gradient_source"] = str(gradient_source)
    if hessian_source is not None:
        info["hessian_source"] = str(hessian_source)
    info["supports_analytic_outer_derivatives"] = supports_analytic
    info["penalty_logdet_source"] = "analytic"
    info["uses_exact_penalty_logdet"] = True
    if (
        not supports_analytic
        and not supports_analytic_hessian
        and family_name in {"gevlss", "shashlss"}
    ):
        info["fallback_reason"] = (
            "fully analytic outer Hessian is not exposed by this family; "
            "use analytic gradient and finite-difference Hessian fallback"
        )
    else:
        info.pop("fallback_reason", None)
    model._general_fit5_outer_derivative_info = info


def _supports_analytic_outer_gradient(family) -> bool:
    return bool(
        getattr(family, "supports_analytic_outer_derivatives", False)
        or getattr(family, "supports_analytic_outer_gradient", False)
    )


def _supports_analytic_outer_hessian(family) -> bool:
    return bool(
        getattr(family, "supports_analytic_outer_derivatives", False)
        or getattr(family, "supports_analytic_outer_hessian", False)
    )


def _run_general_fit5(
    model,
    y,
    smoothing_params,
    *,
    weights=None,
    deriv=2,
    score_type=None,
):
    from ...smoothing_selection.reparam import _stable_penalty_logdet_derivatives

    layout = _build_general_predictor_layout(model)
    St, S_blocks = _build_general_penalty_matrix(model, smoothing_params, layout)
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.clip(smoothing_params, 1e-300, None))
    # mgcv's gam.fit5 consumes rp$ldetS / rp$ldet1 / rp$ldet2 produced upstream
    # by ldetS()/gam.reparam(); use the canonical reparameterization helper
    # directly rather than carrying a local forwarding wrapper.
    ldetS, ldetS1, ldetS2 = _stable_penalty_logdet_derivatives(
        model, smoothing_params, order=2
    )
    evals = np.linalg.eigvalsh(0.5 * (St + St.T))
    tol = max(np.max(evals), 0.0) * np.finfo(np.float64).eps ** 0.75
    Mp = int(St.shape[0] - np.count_nonzero(evals > tol))
    ctl = GamFit5Control(
        maxit=int(getattr(model, "max_irls_iter", 200)),
        epsilon=float(getattr(model, "irls_tol", 1e-7)),
        trace=bool(getattr(model, "hparams", {}).get("trace", False)),
    )
    offset_list = _offset_list(model, len(layout.jj))
    fit = gam_fit5(
        layout.X_full,
        np.asarray(y, dtype=np.float64),
        layout.jj,
        log_sp,
        St,
        S_blocks,
        ldetS=float(ldetS),
        ldetS1=np.asarray(ldetS1, dtype=np.float64),
        ldetS2=np.asarray(ldetS2, dtype=np.float64),
        family=model.family,
        weights=weights,
        offset=offset_list,
        deriv=deriv,
        score_type=_general_fit_score_type_name(
            getattr(model, "_optim_method", "REML")
            if score_type is None
            else score_type
        ),
        control=ctl,
        Mp=Mp,
    )
    return {
        "layout": layout,
        "fit": fit,
        "offset_list": offset_list,
        "smoothing_params": smoothing_params,
        "log_sp": log_sp,
    }


def criterion_ml_reml_general_fit5(model, y, log_sp, method):
    sp = expand_smoothing_params_from_log(model, log_sp)
    run = _run_general_fit5(
        model, y, sp, weights=model.prior_weights_, deriv=0, score_type=method
    )
    return float(run["fit"]["score"])


def criterion_gradient_ml_reml_general_fit5(model, y, log_sp, method):
    if not _supports_analytic_outer_gradient(model.family):
        raise NotImplementedError(
            "General-family ML/REML outer optimization requires analytic outer "
            "gradients for strict mgcv parity; finite-difference fallback removed."
        )
    _record_outer_derivative_mode(model, gradient_source="analytic")
    sp = expand_smoothing_params_from_log(model, log_sp)
    run = _run_general_fit5(
        model, y, sp, weights=model.prior_weights_, deriv=1, score_type=method
    )
    grad = run["fit"].get("score1", None)
    if grad is None:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(grad, dtype=np.float64)


def criterion_hessian_ml_reml_general_fit5(model, y, log_sp, method):
    if _supports_analytic_outer_hessian(model.family):
        _record_outer_derivative_mode(model, hessian_source="analytic")
        sp = expand_smoothing_params_from_log(model, log_sp)
        run = _run_general_fit5(
            model, y, sp, weights=model.prior_weights_, deriv=2, score_type=method
        )
        hess = run["fit"].get("score2", None)
        if hess is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.asarray(hess, dtype=np.float64)
    raise NotImplementedError(
        "General-family ML/REML outer optimization requires analytic outer "
        "Hessians for strict mgcv parity; finite-difference fallback removed."
    )


def solve_general_fit(model, y, smoothing_params, weights=None):
    run = _run_general_fit5(
        model,
        y,
        smoothing_params,
        weights=weights,
        deriv=(
            2
            if (
                len(tuple(_penalty_blocks_seq(model))) > 0
                and bool(
                    getattr(model.family, "supports_analytic_outer_derivatives", False)
                )
            )
            else 0
        ),
        score_type=getattr(model, "_optim_method", "REML"),
    )
    layout = run["layout"]
    fit = run["fit"]
    post = gam_fit5_post_proc(fit)

    eta_cols = []
    for k, sl in enumerate(layout.predictor_full_slices):
        eta_k = layout.X_full[:, sl] @ np.asarray(fit["coef"][sl], dtype=np.float64)
        if run["offset_list"] is not None and k < len(run["offset_list"]):
            off_k = run["offset_list"][k]
            if off_k is not None:
                eta_k = eta_k + np.asarray(off_k, dtype=np.float64)
        eta_cols.append(np.asarray(eta_k, dtype=np.float64))
    eta = (
        np.column_stack(eta_cols)
        if eta_cols
        else np.empty((len(y), 0), dtype=np.float64)
    )

    mu = np.asarray(model.family.predict(eta=eta), dtype=np.float64)

    RTR = np.asarray(post["R"].T @ post["R"], dtype=np.float64)
    H_coef = np.asarray(post["Vp"] @ RTR, dtype=np.float64)

    Vc = None
    if fit.get("db_drho", None) is not None and fit.get("REML2", None) is not None:
        J = np.asarray(fit["db_drho"], dtype=np.float64)
        Hsp = np.asarray(fit["REML2"], dtype=np.float64)
        if J.ndim == 2 and Hsp.ndim == 2 and Hsp.size > 0:
            try:
                Vsp = np.linalg.pinv(0.5 * (Hsp + Hsp.T), rcond=1e-10)
                Vc = np.asarray(post["Vp"] + J @ Vsp @ J.T, dtype=np.float64)
            except np.linalg.LinAlgError:
                Vc = None

    coef_full = np.asarray(fit["coef"], dtype=np.float64)
    beta = np.asarray(coef_full[layout.reduced_to_full_idx], dtype=np.float64)
    intercept = float(coef_full[0]) if layout.predictor_full_slices else 0.0

    return FitCoreSolution.from_dict(
        {
            "coef_full": coef_full,
            "intercept": intercept,
            "beta": beta,
            "eta": eta,
            "mu": mu,
            "rss": None,
            "deviance": float(-2.0 * float(fit["l"])),
            "edf": float(np.sum(np.asarray(post["edf"], dtype=np.float64))),
            "trace_H": float(np.trace(H_coef)),
            "scale": 1.0,
            "cov_bayes": np.asarray(post["Vp"], dtype=np.float64),
            "cov_freq": np.asarray(post["Ve"], dtype=np.float64),
            "cov_unconditional": Vc,
            "H_coef": H_coef,
            "edf2": np.asarray(post["edf2"], dtype=np.float64),
            "X": layout.X_full,
            "A": np.asarray(-fit["lbb"], dtype=np.float64)
            + np.asarray(fit["St_full"], dtype=np.float64),
            "A_inv": np.asarray(post["Vp"], dtype=np.float64),
            "XtWX": None,
            "P": np.asarray(fit["St_full"], dtype=np.float64),
            "penalty_matrix": np.asarray(fit["St_full"], dtype=np.float64),
            "working_weights": None,
            "fisher_weights": None,
            "working_response": None,
            "penalty_quadratic": 0.5
            * float(coef_full @ (np.asarray(fit["St_full"], dtype=np.float64) @ coef_full)),
            "loglik": float(fit["l"]),
            "offset": None,
            "log_det_XtWX_plus_penalty": float(fit["ldetHp"]),
            "converged": (len(fit.get("warn", [])) == 0),
            "iter": int(fit["iter"]),
            "failed_step": bool(len(fit.get("warn", [])) > 0),
            "failure_reason": (
                None if len(fit.get("warn", [])) == 0 else "; ".join(fit["warn"])
            ),
            "inner_trace": None,
        }
    )

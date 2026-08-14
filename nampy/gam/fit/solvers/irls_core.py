"""
Shared penalized IRLS core for fixed-smoothing GAM fits.

This consolidates old Gaussian one-step penalized LS and PIRLS loops into one
solver over the full design matrix ``X`` and full penalty matrix ``S``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..._mgcv_constants import LOG_GUARD_MIN
from ..._model_state import _fit_intercept
from ..covariance import build_bayes_and_freq_covariances
from ..linalg.stacked_qr import (
    _stacked_penalized_ls_nonneg_solution,
    build_penalized_qr_state_nonnegative,
    penalty_sqrt_rows,
    solve_gaussian_penalized_ls_stacked_qr,
)


@dataclass
class PenalizedIrlsControl:
    maxit: int = 100
    epsilon: float = 1e-7
    trace: bool = False


def _family_is_general(family: Any) -> bool:
    return str(getattr(family, "family_class", "")).lower() == "general"


def _validate_extended_family_pirls_hooks(family: Any) -> None:
    if str(getattr(family, "family_class", "")).lower() != "extended":
        return
    required = (
        "initialize_mu",
        "link",
        "inverse_link",
        "mu_eta",
        "variance",
        "deviance",
        "estimate_dispersion",
        "loglik",
        "dvar",
        "d2link",
        "Dd",
        "ls",
        "getTheta",
        "putTheta",
    )
    missing = [name for name in required if not callable(getattr(family, name, None))]
    if missing:
        raise NotImplementedError(
            "Extended family PIRLS path requires hooks: " + ", ".join(missing)
        )


def _mgcv_null_coef(X: np.ndarray, y: np.ndarray, family: Any) -> np.ndarray:
    """
    Mirror mgcv/R/mgcv.r::get.null.coef().

    The resulting coefficients are used only as the feasible anchor for
    gam.fit3's immediate-divergence step-halving checks.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must match nrow(X).")
    if X.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)

    mu_mean = np.full(y.shape[0], float(np.mean(y)), dtype=np.float64)
    eta_mean = np.asarray(family.link(mu_mean), dtype=np.float64).ravel()
    coef, *_ = np.linalg.lstsq(X, eta_mean, rcond=None)
    coef = np.asarray(coef, dtype=np.float64).ravel()
    if coef.shape != (X.shape[1],):
        out = np.zeros(X.shape[1], dtype=np.float64)
        out[: min(out.size, coef.size)] = coef[: min(out.size, coef.size)]
        coef = out
    coef[~np.isfinite(coef)] = 0.0
    return coef


def _strictly_additive_gaussian_identity(family: Any) -> bool:
    return (
        str(getattr(family, "name", "")).lower() == "gaussian"
        and str(getattr(family, "link_name", "")).lower() == "identity"
    )


def _use_exact_extended_negbin_terms(family: Any) -> bool:
    return (
        str(getattr(family, "family_class", "")).lower() == "extended"
        and str(getattr(family, "name", "")).lower() == "negbin"
        and str(getattr(family, "link_name", "")).lower() == "log"
    )


def _mgcv_poisson_identity_fisher_endpoint(family: Any) -> bool:
    """
    Match the mgcv endpoint for poisson(link="identity") fixed-SP PIRLS fits.

    In mgcv/R/gam.fit3.r this family is a noncanonical full-Newton case, but
    y == 0 rows set alpha to .Machine$double.eps and make the C pls_fit1()
    pseudo-data path numerically platform-sensitive. The reported gam() endpoint
    for these boundary fits agrees with Fisher scoring, so keep the stabilization
    scoped to this family/link pair.
    """

    return (
        str(getattr(family, "family_class", "")).lower() == "glm"
        and str(getattr(family, "name", "")).lower() == "poisson"
        and str(getattr(family, "link_name", "")).lower() == "identity"
    )


def _mgcv_effective_irls_tol(family: Any, tol: float) -> float:
    if _mgcv_poisson_identity_fisher_endpoint(family):
        return min(float(tol), 1e-11)
    return float(tol)


def irls_core(
    X: np.ndarray,
    y: np.ndarray,
    family: Any,
    S: np.ndarray,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    *,
    fit_intercept: bool = True,
    max_iter: int = 200,
    tol: float = 1e-7,
    max_step_halving: int = 25,
    coef_start: np.ndarray | None = None,
    null_coef: np.ndarray | None = None,
    etastart: np.ndarray | None = None,
    mustart: np.ndarray | None = None,
    fisher_scoring_only: bool = False,
    penalty_rank_rows: np.ndarray | None = None,
    # mgcv/R/gam.fit3.r:131: rank.tol <- .Machine$double.eps*100; the PIRLS
    # inner solve must use the same drop threshold as the gdiPK derivative
    # kernel (_MGCV_GAM_FIT3_RANK_TOL), not the looser eps**0.66 stacked-QR
    # default.
    rank_tol: float = float(np.finfo(np.float64).eps) * 100.0,
    coef_method: str = "householder",
    near_singular_null_pin: bool | Literal["auto"] = False,
    force_stacked_qr: bool = False,
    scale_reference: float | None = None,
    trace: bool = False,
) -> dict[str, Any]:
    if _family_is_general(family):
        raise NotImplementedError("General family support is not yet implemented.")
    _validate_extended_family_pirls_hooks(family)

    X = np.asarray(X, dtype=np.float64)
    y = family.validate_y(y)
    nobs, q = X.shape
    S = np.asarray(S, dtype=np.float64)
    if S.shape != (q, q):
        raise ValueError(f"S must be ({q}, {q}), got {S.shape}.")

    if weights is None:
        weights = np.ones(nobs, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.shape != (nobs,):
        raise ValueError("weights must match nrow(X).")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative.")

    if offset is None:
        offset = np.zeros(nobs, dtype=np.float64)
        offset_out = None
    else:
        offset = np.asarray(offset, dtype=np.float64).ravel()
        offset_out = offset.copy()
    if offset.shape != (nobs,):
        raise ValueError("offset must match nrow(X).")

    warnings_list: list[str] = []

    if q == 0:
        eta = offset.copy()
        mu = np.asarray(family.inverse_link(eta), dtype=np.float64)
        dev = float(family.deviance(y, mu, weights=weights))
        H_coef = np.zeros((0, 0), dtype=np.float64)
        return {
            "coef_full": np.zeros(0, dtype=np.float64),
            "coef": np.zeros(0, dtype=np.float64),
            "intercept": 0.0,
            "beta": np.zeros(0, dtype=np.float64),
            "eta": eta,
            "linear_predictor": eta,
            "mu": mu,
            "rss": float(np.sum((y - mu) ** 2)),
            "deviance": dev,
            "edf": 0.0,
            "trace_H": 0.0,
            "scale": float(getattr(family, "known_scale", 1.0) or 1.0),
            "cov_bayes": np.zeros((0, 0), dtype=np.float64),
            "cov_freq": np.zeros((0, 0), dtype=np.float64),
            "H_coef": H_coef,
            "X": X,
            "A": np.zeros((0, 0), dtype=np.float64),
            "A_inv": np.zeros((0, 0), dtype=np.float64),
            "XtWX": np.zeros((0, 0), dtype=np.float64),
            "P": S,
            "penalty_matrix": S,
            "working_weights": weights.copy(),
            "fisher_weights": weights.copy(),
            "working_response": np.zeros(nobs, dtype=np.float64),
            "penalty_quadratic": 0.0,
            "loglik": None,
            "converged": True,
            "iter": 0,
            "iterations": 0,
            "failed_step": False,
            "failure_reason": None,
            "offset": offset_out,
            "penalized_system_rank": 0,
            "dropped_column_indices": np.zeros((0,), dtype=np.int64),
            "inner_trace": [],
            "warnings": warnings_list,
        }

    penalty_sqrt, penalty_rank_template = penalty_sqrt_rows(S)
    if penalty_rank_rows is None:
        penalty_rank_rows = penalty_rank_template
    else:
        penalty_rank_rows = np.asarray(penalty_rank_rows, dtype=np.float64)

    last_stacked_qr_state = None

    def _stacked_qr_penalized_step(X_curr, w_curr, rhs_curr, *, rhs_is_weighted):
        nonlocal last_stacked_qr_state
        z_curr = np.asarray(rhs_curr, dtype=np.float64).ravel().copy()
        if rhs_is_weighted:
            positive = w_curr > 0.0
            z_weighted = z_curr.copy()
            z_curr.fill(0.0)
            z_curr[positive] = z_weighted[positive] / w_curr[positive]
        out = _stacked_penalized_ls_nonneg_solution(
            X_curr,
            z_curr,
            w_curr,
            penalty_sqrt=np.asarray(penalty_sqrt, dtype=np.float64),
            penalty_rank_rows=np.asarray(penalty_rank_rows, dtype=np.float64),
            P_dense=S,
            rank_tol=rank_tol,
            coef_method=coef_method,
            near_singular_null_pin=near_singular_null_pin,
        )
        qr_state = None
        # Under-determined stacked fits use the lstsq branch because the
        # Householder reconstruction path can fail on q > n systems even when
        # the penalized least-squares solution itself is well-defined.
        if (
            str(coef_method).lower().strip() != "lstsq"
            and penalty_sqrt.shape[0] > 0
            and penalty_rank_rows.shape[0] > 0
        ):
            qr_state = build_penalized_qr_state_nonnegative(
                X_curr,
                z_curr,
                w_curr,
                penalty_sqrt_E=np.asarray(penalty_sqrt, dtype=np.float64),
                penalty_rank_Es=np.asarray(penalty_rank_rows, dtype=np.float64),
                rS=np.asarray(penalty_sqrt, dtype=np.float64).T,
                rank_tol=rank_tol,
                reml=True,
            )
        last_stacked_qr_state = {
            "penalized_system_rank": int(getattr(out, "system_rank", X_curr.shape[1])),
            "dropped_column_indices": np.asarray(
                getattr(out, "dropped_column_indices", np.zeros((0,), dtype=np.int64)),
                dtype=np.int64,
            ),
            "penalized_qr_state": qr_state,
        }
        return np.asarray(out.coef_full, dtype=np.float64)

    def _irls_state(eta_value):
        mu_value = np.asarray(family.inverse_link(eta_value), dtype=np.float64)
        mu_eta_value = np.asarray(family.mu_eta(eta_value), dtype=np.float64)
        var_value = np.asarray(family.variance(mu_value), dtype=np.float64)
        return mu_value, mu_eta_value, var_value

    def _working_response_terms(eta_curr, mu_curr):
        if _use_exact_extended_negbin_terms(family):
            mu_curr = np.clip(
                np.asarray(mu_curr, dtype=np.float64), LOG_GUARD_MIN, None
            )
            dd = family.Dd(y, mu_curr, family.getTheta(False), weights, level=0)
            Deta = np.asarray(dd["Dmu"], dtype=np.float64) * mu_curr
            Deta2 = (
                np.asarray(dd["Dmu2"], dtype=np.float64) * (mu_curr**2)
                + np.asarray(dd["Dmu"], dtype=np.float64) * mu_curr
            )
            w_curr = 0.5 * Deta2
            wz_curr = w_curr * (eta_curr - offset) - 0.5 * Deta
            with np.errstate(divide="ignore", invalid="ignore"):
                z_curr = (eta_curr - offset) - Deta / Deta2
            good_curr = np.isfinite(z_curr) & np.isfinite(w_curr)
            rhs_is_weighted = False
            rhs_curr = np.asarray(z_curr, dtype=np.float64)
            if np.any(~good_curr):
                rhs_is_weighted = True
                good_curr = np.isfinite(w_curr) & np.isfinite(wz_curr)
                rhs_curr = np.asarray(wz_curr, dtype=np.float64)
            return {
                "good": np.asarray(good_curr, dtype=bool),
                "w": np.asarray(w_curr[good_curr], dtype=np.float64),
                "z": np.asarray(rhs_curr[good_curr], dtype=np.float64),
                "rhs_is_weighted": bool(rhs_is_weighted),
                "w_full": np.asarray(w_curr, dtype=np.float64),
                "wz_full": np.asarray(wz_curr, dtype=np.float64),
                "z_full": np.asarray(z_curr, dtype=np.float64),
                "Deta2_full": np.asarray(Deta2, dtype=np.float64),
            }

        mu_eta_curr = np.asarray(family.mu_eta(eta_curr), dtype=np.float64)
        var_curr = np.asarray(family.variance(mu_curr), dtype=np.float64)
        good_curr = (
            (weights > 0.0)
            & np.isfinite(mu_eta_curr)
            & (mu_eta_curr != 0.0)
            & np.isfinite(var_curr)
            & (var_curr > 0.0)
        )
        if not np.any(good_curr):
            return None

        y_g_curr = y[good_curr]
        mu_g_curr = mu_curr[good_curr]
        eta_g_curr = eta_curr[good_curr]
        mu_eta_g_curr = mu_eta_curr[good_curr]
        var_g_curr = var_curr[good_curr]
        weights_g_curr = weights[good_curr]
        off_g_curr = offset[good_curr]

        fisher_W_curr = weights_g_curr * (mu_eta_g_curr**2) / var_g_curr
        use_fisher_curr = bool(
            fisher_scoring_only or bool(getattr(family, "canonical_link", False))
        )
        if (
            not use_fisher_curr
            and hasattr(family, "dvar")
            and hasattr(family, "d2link")
        ):
            try:
                dvar_curr = np.asarray(family.dvar(mu_g_curr), dtype=np.float64)
                d2link_curr = np.asarray(family.d2link(mu_g_curr), dtype=np.float64)
                alpha_curr = 1.0 + (y_g_curr - mu_g_curr) * (
                    dvar_curr / var_g_curr + d2link_curr * mu_eta_g_curr
                )
                eps_alpha_curr = np.finfo(np.float64).eps
                zero_curr = alpha_curr == 0.0
                if np.any(zero_curr):
                    alpha_curr = alpha_curr.copy()
                    alpha_curr[zero_curr] = eps_alpha_curr
                w_curr = fisher_W_curr * alpha_curr
                z_curr = (eta_g_curr - off_g_curr) + (y_g_curr - mu_g_curr) / (
                    mu_eta_g_curr * alpha_curr
                )
                if np.any(~np.isfinite(w_curr)) or np.any(~np.isfinite(z_curr)):
                    use_fisher_curr = True
            except Exception:
                use_fisher_curr = True
        if use_fisher_curr:
            w_curr = fisher_W_curr
            z_curr = (eta_g_curr - off_g_curr) + (y_g_curr - mu_g_curr) / mu_eta_g_curr

        return {
            "good": good_curr,
            "w": np.asarray(w_curr, dtype=np.float64),
            "z": np.asarray(z_curr, dtype=np.float64),
            "rhs_is_weighted": False,
        }

    valid_eta = getattr(family, "valid_eta", None)
    valid_mu = getattr(family, "valid_mu", None)

    def _eta_mu_valid(eta_curr: np.ndarray, mu_curr: np.ndarray) -> bool:
        ok_eta = True if not callable(valid_eta) else bool(valid_eta(eta_curr))
        ok_mu = True if not callable(valid_mu) else bool(valid_mu(mu_curr))
        return ok_eta and ok_mu

    def _weighted_deviance(mu_curr: np.ndarray) -> float:
        # Mirror mgcv::gam.fit3(), which uses dev.resids(..., weights) at every
        # PIRLS deviance / penalized-deviance update.
        return float(family.deviance(y, mu_curr, weights=weights))

    def _weighted_loglik(mu_curr: np.ndarray, scale_curr: float) -> float | None:
        if hasattr(family, "loglik_obs"):
            return float(
                np.sum(
                    weights
                    * np.asarray(
                        family.loglik_obs(y, mu_curr, scale=scale_curr),
                        dtype=np.float64,
                    )
                )
            )
        if hasattr(family, "loglik"):
            return float(family.loglik(y, mu_curr, scale=scale_curr))
        return None

    def _estimate_gam_fit3_scale(
        mu_curr: np.ndarray,
        edf_curr: float,
    ) -> float:
        # Mirrors mgcv/R/gam.fit3.r::gam.fit3 default
        # gam.control(scale.est="fletcher") scale calculation.
        known_scale = getattr(family, "known_scale", None)
        if known_scale is not None:
            return float(known_scale)
        if hasattr(family, "estimate_dispersion"):
            scale_curr = float(
                family.estimate_dispersion(y, mu_curr, edf=edf_curr, weights=weights)
            )
        else:
            w_sum = float(np.sum(weights))
            denom = max(w_sum - float(edf_curr), np.finfo(np.float64).eps)
            scale_curr = float(_weighted_deviance(mu_curr) / denom)

        dvar = getattr(family, "dvar", None)
        if not callable(dvar):
            return scale_curr
        try:
            var = np.asarray(family.variance(mu_curr), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                s_terms = (
                    np.asarray(dvar(mu_curr), dtype=np.float64)
                    * (y - mu_curr)
                    / var
                )
                s_bar_raw = float(np.mean(s_terms))
        except Exception:
            return scale_curr
        if np.isfinite(s_bar_raw):
            s_bar = max(-0.9, s_bar_raw)
            scale_curr = scale_curr / (1.0 + s_bar)
        return float(scale_curr)

    def _recompute_step(beta_curr: np.ndarray):
        eta_curr = offset + X @ beta_curr
        mu_curr = family.inverse_link(eta_curr)
        dev_curr = _weighted_deviance(mu_curr)
        pdev_curr = dev_curr + float(beta_curr @ (S @ beta_curr))
        return eta_curr, mu_curr, dev_curr, pdev_curr

    beta = None
    if coef_start is not None:
        beta0 = np.asarray(coef_start, dtype=np.float64).ravel()
        if beta0.shape == (q,) and np.all(np.isfinite(beta0)):
            beta = beta0.copy()
    if beta is None:
        beta = np.zeros(q, dtype=np.float64)

    if etastart is not None:
        eta = np.asarray(etastart, dtype=np.float64).ravel()
        if eta.shape != (nobs,):
            raise ValueError("etastart must have length nrow(X).")
        mu = family.inverse_link(eta)
    elif mustart is not None:
        mu0 = np.asarray(mustart, dtype=np.float64).ravel()
        if mu0.shape != (nobs,):
            raise ValueError("mustart must have length nrow(X).")
        mu = mu0
        eta = family.link(mu)
    else:
        if coef_start is None:
            try:
                mu_init = family.initialize_mu(y, weights=weights)
            except TypeError:
                mu_init = family.initialize_mu(y)
            eta = family.link(mu_init)
        else:
            eta = offset + X @ beta
        mu = family.inverse_link(eta)

    if null_coef is not None:
        null_arr = np.asarray(null_coef, dtype=np.float64).ravel()
        if null_arr.shape != (q,):
            raise ValueError(f"null_coef must have shape ({q},), got {null_arr.shape}.")
        null_beta = np.where(np.isfinite(null_arr), null_arr, 0.0)
    else:
        null_beta = np.zeros_like(beta)
    null_eta = offset + X @ null_beta
    old_pdev = float(
        _weighted_deviance(family.inverse_link(null_eta)) + null_beta @ (S @ null_beta)
    )
    coefold = null_beta.copy()

    for _ in range(20):
        if (
            np.all(np.isfinite(mu))
            and np.all(np.isfinite(eta))
            and _eta_mu_valid(eta, mu)
        ):
            break
        if coef_start is not None and etastart is None and mustart is None:
            beta = 0.9 * beta + 0.1 * null_beta
            eta = offset + X @ beta
        else:
            eta = 0.9 * eta + 0.1 * null_eta
        mu = family.inverse_link(eta)
    else:
        raise RuntimeError("Cannot find valid starting values (eta/mu).")

    strictly_additive = _strictly_additive_gaussian_identity(family)
    converged = False
    failed_step = False
    failure_reason = None
    n_iter = 0
    inner_trace = []
    inner_halving_limit = max(int(max_iter), 100)

    for it in range(max_iter):
        n_iter = it + 1
        mu, mu_eta, var = _irls_state(eta)
        work_terms = _working_response_terms(eta, mu)
        if work_terms is None or not np.any(work_terms["good"]):
            failed_step = True
            failure_reason = "no_informative_observations"
            warnings_list.append(
                f"irls_core: no informative observations at iteration {n_iter}"
            )
            break
        good = np.asarray(work_terms["good"], dtype=bool)
        X_g = X[good, :]
        W = np.asarray(work_terms["w"], dtype=np.float64)
        z_work = np.asarray(work_terms["z"], dtype=np.float64)
        rhs_is_weighted = bool(work_terms.get("rhs_is_weighted", False))
        grad_good = good
        grad_w = W
        grad_z = z_work
        grad_rhs_is_weighted = rhs_is_weighted

        XtW = X_g.T * W
        XtWX = XtW @ X_g
        A = XtWX + S

        try:
            beta_prop = _stacked_qr_penalized_step(
                X_g,
                W,
                z_work,
                rhs_is_weighted=rhs_is_weighted,
            )
        except (np.linalg.LinAlgError, ValueError):
            if _use_exact_extended_negbin_terms(family):
                Deta2_full = np.asarray(work_terms["Deta2_full"], dtype=np.float64)
                good_pos = np.isfinite(Deta2_full) & (Deta2_full > 0.0)
                w_pos_full = np.zeros_like(Deta2_full, dtype=np.float64)
                w_pos_full[good_pos] = 0.5 * Deta2_full[good_pos]
                wz_pos_full = np.asarray(work_terms["wz_full"], dtype=np.float64)
                good_rhs = np.isfinite(w_pos_full) & np.isfinite(wz_pos_full)
                X_pos = X[good_rhs, :]
                W_pos = w_pos_full[good_rhs]
                z_pos = np.asarray(wz_pos_full[good_rhs], dtype=np.float64)
                rhs_is_weighted = True
            else:
                use_fisher = bool(
                    fisher_scoring_only
                    or bool(getattr(family, "canonical_link", False))
                )
                if use_fisher:
                    failed_step = True
                    failure_reason = "linear_solve_failed"
                    warnings_list.append(
                        f"irls_core: linear solve failed at iteration {n_iter}"
                    )
                    break

                mu_eta_g = mu_eta[good]
                var_g = var[good]
                y_g = y[good]
                mu_g = mu[good]
                eta_g = eta[good]
                weights_g = weights[good]
                off_g = offset[good]
                W_pos = weights_g * (mu_eta_g**2) / var_g
                eta_lin = eta_g - off_g
                wz_pos_full = (
                    W_pos * eta_lin + weights_g * mu_eta_g * (y_g - mu_g) / var_g
                )
                z_pos_full = np.asarray(
                    (eta_g - off_g) + (y_g - mu_g) / mu_eta_g, dtype=np.float64
                )
                good_pos = np.isfinite(z_pos_full) & np.isfinite(W_pos)
                if np.any(~good_pos):
                    good_pos = np.isfinite(W_pos) & np.isfinite(wz_pos_full)
                    z_pos = np.asarray(wz_pos_full[good_pos], dtype=np.float64)
                    rhs_is_weighted = True
                else:
                    z_pos = np.asarray(z_pos_full[good_pos], dtype=np.float64)
                    rhs_is_weighted = False
                X_pos = X_g[good_pos, :]
                W_pos = W_pos[good_pos]

            if W_pos.size == 0:
                failed_step = True
                failure_reason = "linear_solve_failed"
                warnings_list.append(
                    f"irls_core: linear solve failed at iteration {n_iter}"
                )
                break
            XtW = X_pos.T * W_pos
            XtWX = XtW @ X_pos
            grad_good = np.zeros_like(good, dtype=bool)
            grad_good[np.flatnonzero(good)[good_pos]] = True
            grad_w = np.asarray(W_pos, dtype=np.float64)
            grad_z = np.asarray(z_pos, dtype=np.float64)
            grad_rhs_is_weighted = rhs_is_weighted
            try:
                beta_prop = _stacked_qr_penalized_step(
                    X_pos,
                    W_pos,
                    z_pos,
                    rhs_is_weighted=rhs_is_weighted,
                )
            except (np.linalg.LinAlgError, ValueError):
                failed_step = True
                failure_reason = "linear_solve_failed"
                warnings_list.append(
                    f"irls_core: linear solve failed at iteration {n_iter}"
                )
                break

        beta_new = np.asarray(beta_prop, dtype=np.float64)
        if np.any(~np.isfinite(beta_new)):
            failed_step = True
            failure_reason = "non_finite_coefficients"
            warnings_list.append(
                f"irls_core: non-finite coefficients at iteration {n_iter}"
            )
            break

        eta_new, mu_new, dev_new, pdev_new = _recompute_step(beta_new)

        if trace:
            print(f"Deviance = {dev_new}  Iteration = {n_iter}")

        if not np.isfinite(dev_new):
            warnings_list.append("irls_core: step size truncated due to divergence")
            ii_h = 1
            while not np.isfinite(dev_new):
                if ii_h > inner_halving_limit:
                    failed_step = True
                    failure_reason = "step_halving_exhausted"
                    warnings_list.append(
                        "irls_core: inner loop 1; can't correct step size"
                    )
                    break
                ii_h += 1
                beta_new = 0.5 * (beta_new + coefold)
                eta_new, mu_new, dev_new, pdev_new = _recompute_step(beta_new)
            if failed_step:
                break

        if not _eta_mu_valid(eta_new, mu_new):
            warnings_list.append("irls_core: step size truncated: out of bounds")
            ii_h = 1
            while not _eta_mu_valid(eta_new, mu_new):
                if ii_h > inner_halving_limit:
                    failed_step = True
                    failure_reason = "step_halving_exhausted"
                    warnings_list.append(
                        "irls_core: inner loop 2; can't correct step size"
                    )
                    break
                ii_h += 1
                beta_new = 0.5 * (beta_new + coefold)
                eta_new, mu_new, dev_new, pdev_new = _recompute_step(beta_new)
            if failed_step:
                break

        # Mirror mgcv::gam.fit3() divergence threshold exactly.
        div_thresh = 10.0 * (0.1 + abs(old_pdev)) * (np.finfo(np.float64).eps ** 0.5)
        if pdev_new - old_pdev > div_thresh:
            coef_anchor = null_beta if n_iter == 1 else coefold
            ii_h = 1
            while pdev_new - old_pdev > div_thresh:
                if ii_h > 100:
                    failed_step = True
                    failure_reason = "step_halving_exhausted"
                    warnings_list.append(
                        "irls_core: inner loop 3; can't correct step size"
                    )
                    break
                ii_h += 1
                beta_new = 0.5 * (beta_new + coef_anchor)
                eta_new, mu_new, dev_new, pdev_new = _recompute_step(beta_new)
            if failed_step:
                break

        beta = beta_new
        eta = eta_new
        mu = mu_new

        theta_efs_enabled = bool(getattr(family, "estimate_theta", False)) and not bool(
            getattr(family, "_disable_theta_efs", False)
        )
        if theta_efs_enabled and hasattr(family, "estimate_theta_mle"):
            theta_efs = family.estimate_theta_mle(y, mu, weights=weights)
            if np.isfinite(theta_efs) and theta_efs > 0.0:
                family.theta = theta_efs
                dev_new = _weighted_deviance(mu)
        penalty_new = float(beta @ (S @ beta))
        pdev_new = dev_new + penalty_new

        if strictly_additive:
            converged = True
            inner_trace.append(
                {
                    "iter": int(n_iter),
                    "log_theta": (
                        None
                        if not hasattr(family, "theta")
                        else float(np.log(max(float(family.theta), LOG_GUARD_MIN)))
                    ),
                    "deviance": float(dev_new),
                    "penalized_deviance": float(pdev_new),
                    "penalized_deviance_conv": float(pdev_new),
                    "grad_inf_norm": 0.0,
                    "converged_here": True,
                }
            )
            old_pdev = pdev_new
            break

        eta_grad = (eta - offset)[grad_good]
        X_good = X[grad_good, :]
        if grad_rhs_is_weighted:
            grad = 2.0 * (X_good.T @ (grad_w * eta_grad - grad_z)) + 2.0 * (S @ beta)
        else:
            grad = 2.0 * (X_good.T @ (grad_w * (eta_grad - grad_z))) + 2.0 * (S @ beta)

        scale_ref = (
            float(scale_reference)
            if scale_reference is not None
            else (
                1.0
                if getattr(family, "known_scale", None) is not None
                else max(abs(dev_new), 1.0)
            )
        )
        pdev_conv = pdev_new
        grad_inf_norm = float(np.max(np.abs(grad)))
        converged_here = bool(
            abs(pdev_conv - old_pdev) < tol * (abs(scale_ref) + abs(pdev_conv))
            and grad_inf_norm <= tol * (abs(scale_ref) + abs(pdev_new))
        )
        inner_trace.append(
            {
                "iter": int(n_iter),
                "log_theta": (
                    None
                    if not hasattr(family, "theta")
                    else float(np.log(max(float(family.theta), LOG_GUARD_MIN)))
                ),
                "deviance": float(dev_new),
                "penalized_deviance": float(pdev_new),
                "penalized_deviance_conv": float(pdev_conv),
                "grad_inf_norm": grad_inf_norm,
                "converged_here": converged_here,
            }
        )
        old_pdev = pdev_new
        coefold = beta.copy()
        if converged_here:
            converged = True
            break

    eta = offset + X @ beta
    mu = family.inverse_link(eta)
    mu_eta = np.asarray(family.mu_eta(eta), dtype=np.float64)
    var = np.asarray(family.variance(mu), dtype=np.float64)
    good = (
        (weights > 0.0)
        & np.isfinite(mu_eta)
        & (mu_eta != 0.0)
        & np.isfinite(var)
        & (var > 0.0)
    )
    if not np.any(good):
        raise RuntimeError("No informative observations at IRLS solution.")

    mu_eta_g = mu_eta[good]
    var_g = var[good]
    y_g = y[good]
    mu_g = mu[good]
    eta_g = eta[good]
    X_g = X[good, :]
    weights_g = weights[good]
    off_g = offset[good]

    fisher_W_g = weights_g * (mu_eta_g**2) / var_g
    use_fisher = bool(
        fisher_scoring_only or bool(getattr(family, "canonical_link", False))
    )
    if not use_fisher and hasattr(family, "dvar") and hasattr(family, "d2link"):
        try:
            dvar = np.asarray(family.dvar(mu_g), dtype=np.float64)
            d2link = np.asarray(family.d2link(mu_g), dtype=np.float64)
            alpha = 1.0 + (y_g - mu_g) * (dvar / var_g + d2link * mu_eta_g)
            eps_alpha = np.finfo(np.float64).eps
            zero = alpha == 0.0
            if np.any(zero):
                alpha = alpha.copy()
                alpha[zero] = eps_alpha
            W_g = fisher_W_g * alpha
            z_g = (eta_g - off_g) + (y_g - mu_g) / (mu_eta_g * alpha)
            if np.any(~np.isfinite(W_g)) or np.any(~np.isfinite(z_g)):
                use_fisher = True
        except Exception:
            use_fisher = True
    if use_fisher:
        W_g = fisher_W_g
        z_g = (eta_g - off_g) + (y_g - mu_g) / mu_eta_g

    W = np.zeros_like(y, dtype=np.float64)
    W[good] = W_g
    fisher_W = np.zeros_like(y, dtype=np.float64)
    fisher_W[good] = fisher_W_g
    z_work = np.zeros_like(y, dtype=np.float64)
    z_work[good] = z_g

    # Mirror `mgcv/src/gdi.c::gdi1()`: the reported post-fit covariance / EDF
    # objects are built from the Fisher-weighted system (`wf`), even when the
    # PIRLS coefficient updates used full-Newton working weights (`w`).
    cov_W_g = np.asarray(fisher_W_g, dtype=np.float64)
    cov_z_g = np.asarray((eta_g - off_g) + (y_g - mu_g) / mu_eta_g, dtype=np.float64)

    XtW = X_g.T * cov_W_g
    XtWX = XtW @ X_g
    A = XtWX + S
    beta_report = np.asarray(beta, dtype=np.float64).copy()
    eta_report = np.asarray(eta, dtype=np.float64).copy()
    mu_report = np.asarray(mu, dtype=np.float64).copy()
    # `mgcv::gdi1()` receives these pre-refresh PIRLS arrays. Its internal
    # `gdiPK()` then refreshes the coefficient representative returned to R,
    # but the deviance/link/variance derivative arrays remain tied to this
    # state for the current call.
    gdi1_eta = np.asarray(eta, dtype=np.float64).copy()
    gdi1_mu = np.asarray(mu, dtype=np.float64).copy()

    # Mirror mgcv::gam.fit3(): the post-loop gdi1() call solves the current
    # weighted penalized least-squares system one more time and reports
    # oo$beta / fitted values from that solve, while the derivative and
    # covariance objects remain tied to the same working system.
    if (
        str(getattr(family, "family_class", "")).lower() == "glm"
        and np.all(np.isfinite(W_g))
        and np.all(W_g >= 0.0)
    ):
        try:
            beta_refresh = _stacked_qr_penalized_step(
                X_g,
                W_g,
                z_g,
                rhs_is_weighted=False,
            )
        except (np.linalg.LinAlgError, ValueError):
            beta_refresh = None
        if beta_refresh is not None:
            beta_refresh = np.asarray(beta_refresh, dtype=np.float64).ravel()
            if beta_refresh.shape == beta_report.shape and np.all(
                np.isfinite(beta_refresh)
            ):
                eta_refresh = offset + X @ beta_refresh
                mu_refresh = family.inverse_link(eta_refresh)
                if _eta_mu_valid(eta_refresh, mu_refresh):
                    beta_report = beta_refresh
                    eta_report = np.asarray(eta_refresh, dtype=np.float64)
                    mu_report = np.asarray(mu_refresh, dtype=np.float64)

    stacked = solve_gaussian_penalized_ls_stacked_qr(
        X_g,
        cov_z_g,
        cov_W_g,
        S,
        penalty_rank_rows=penalty_rank_rows,
        coef_method=coef_method,
        near_singular_null_pin=near_singular_null_pin,
    )
    A = np.asarray(stacked["A"], dtype=np.float64)
    XtWX = np.asarray(stacked["XtWX"], dtype=np.float64)
    A_inv = np.asarray(stacked["A_inv"], dtype=np.float64)
    H_coef = np.asarray(stacked["coef_hat_matrix"], dtype=np.float64)
    log_det_xtwx_plus_penalty = float(stacked["log_det_XtWX_plus_penalty"])
    cov_root = np.asarray(stacked["covariance_root"], dtype=np.float64)
    WX_rV = np.asarray(stacked["WX_sqrt"], dtype=np.float64) @ cov_root
    trace_H = float(np.sum(WX_rV * WX_rV))
    penalized_system_rank = int(stacked["penalized_system_rank"])
    dropped_column_indices = np.asarray(
        stacked["dropped_column_indices"], dtype=np.int64
    )
    edf = trace_H

    scale = _estimate_gam_fit3_scale(mu, edf)
    deviance = _weighted_deviance(mu)
    rss = float(np.sum((y - mu) ** 2))
    penalty_quadratic = float(beta @ (S @ beta))
    loglik = _weighted_loglik(mu, scale)

    Vp, Vf, H_coef = build_bayes_and_freq_covariances(scale, A_inv, XtWX)
    if stacked is not None:
        cov_root = np.asarray(stacked["covariance_root"], dtype=np.float64)
        Vp = np.asarray(scale * (cov_root @ cov_root.T), dtype=np.float64)
        Vp = 0.5 * (Vp + Vp.T)
    if last_stacked_qr_state is not None:
        penalized_system_rank = int(last_stacked_qr_state["penalized_system_rank"])
        dropped_column_indices = np.asarray(
            last_stacked_qr_state["dropped_column_indices"], dtype=np.int64
        )
    penalized_qr_state = (
        None
        if last_stacked_qr_state is None
        else last_stacked_qr_state.get("penalized_qr_state", None)
    )

    if fit_intercept and q > 0:
        intercept = float(beta_report[0])
        beta_term = beta_report[1:].copy()
    else:
        intercept = 0.0
        beta_term = beta_report.copy()

    if (not converged) and (not failed_step) and n_iter >= max_iter:
        warnings_list.append(
            f"irls_core: reached max_iter={max_iter} without meeting convergence criteria"
        )

    return {
        "coef_full": beta_report.copy(),
        "coef": beta_report.copy(),
        "intercept": intercept,
        "beta": beta_term,
        "eta": eta_report,
        "linear_predictor": eta_report,
        "mu": mu_report,
        "gdi1_eta": gdi1_eta,
        "gdi1_mu": gdi1_mu,
        "rss": rss,
        "deviance": deviance,
        "edf": edf,
        "trace_H": trace_H,
        "scale": scale,
        "cov_bayes": Vp,
        "cov_freq": Vf,
        "H_coef": H_coef,
        "X": X,
        "A": A,
        "A_inv": A_inv,
        "XtWX": XtWX,
        "P": S,
        "penalty_matrix": S,
        "log_det_XtWX_plus_penalty": log_det_xtwx_plus_penalty,
        "working_weights": W,
        "fisher_weights": fisher_W,
        "working_response": z_work,
        "penalty_quadratic": penalty_quadratic,
        "loglik": loglik,
        "converged": converged,
        "iter": n_iter,
        "iterations": n_iter,
        "failed_step": failed_step,
        "failure_reason": failure_reason,
        "offset": offset_out,
        "penalized_system_rank": penalized_system_rank,
        "dropped_column_indices": dropped_column_indices,
        "penalized_qr_state": penalized_qr_state,
        "inner_trace": inner_trace,
        "warnings": warnings_list,
    }


def fit_irls_from_model(
    model: Any,
    y: np.ndarray,
    smoothing_params: np.ndarray,
    *,
    attach_smoothness_postprocess: bool = True,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    from ..linalg.stacked_qr import balanced_penalty_template_sqrt_for_rank
    from ..penalized_system import build_full_design, build_full_penalty_from_blocks
    from ..postprocess.gaussian_smoothness_postprocess import (
        merge_gaussian_smoothness_into_fit_result,
    )

    y = model.family.validate_y(y)
    fi = _fit_intercept(model)
    from ..._model_state import _design_matrix, _n_coef, _penalty_blocks_seq

    Z = np.asarray(_design_matrix(model), dtype=np.float64)
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    n_coef = _n_coef(model)
    X = build_full_design(Z, fit_intercept=fi)
    S = build_full_penalty_from_blocks(
        penalty_blocks=penalty_blocks,
        smoothing_params=np.asarray(smoothing_params, dtype=np.float64).ravel(),
        fit_intercept=fi,
        n_coef=n_coef,
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        penalty_blocks,
        fit_intercept=fi,
        n_coef=int(n_coef),
    )

    out = irls_core(
        X,
        y,
        model.family,
        S,
        offset=model.offset_train_,
        weights=weights,
        fit_intercept=fi,
        max_iter=int(getattr(model, "max_irls_iter", 200)),
        tol=_mgcv_effective_irls_tol(
            model.family, float(getattr(model, "irls_tol", 1e-7))
        ),
        max_step_halving=int(getattr(model, "max_step_halving", 25)),
        null_coef=_mgcv_null_coef(X, y, model.family),
        fisher_scoring_only=_mgcv_poisson_identity_fisher_endpoint(model.family),
        penalty_rank_rows=rank_rows,
    )
    if (
        attach_smoothness_postprocess
        and str(getattr(model.family, "name", "")).lower() == "gaussian"
    ):
        sp_lin = np.asarray(smoothing_params, dtype=np.float64).ravel()
        score_type = str(getattr(model, "smoothing_method", "REML")).upper()
        out = merge_gaussian_smoothness_into_fit_result(
            out,
            model,
            y,
            sp_lin,
            score_type=score_type,
            deriv=2,
        )
    return out

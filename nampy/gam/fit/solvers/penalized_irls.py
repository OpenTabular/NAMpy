"""
Penalized IRLS coefficient solver for GAMs with fixed smoothing parameters.

Given fixed smoothing parameters, this module iterates the penalized
weighted least-squares problem until penalized deviance converges.  It is
the inner loop called by the outer smoothness-parameter optimiser.

Iteration outline
-----------------
1. Build working weights ``W`` and adjusted response ``z`` from the current
   predicted mean ``mu`` (Fisher scoring or Newton step).
2. Solve the penalized weighted least-squares subproblem via the stacked
   pivoted QR solver.
3. Apply step halving if the new penalized deviance is non-finite or
   diverges by more than a threshold relative to the previous iteration.
4. Check convergence on both penalized deviance and gradient magnitude.

Fisher vs Newton working weights
---------------------------------
Fisher scoring (positive-definite weights) is always used for canonical-link
families.  For other links, Newton weights are attempted first; if they are
negative or non-finite, the solver falls back to Fisher scoring for that step.

Gaussian + identity shortcut
------------------------------
For the Gaussian identity-link family the working weights are constant (equal
to the prior weights), so the penalized least-squares system does not change
between iterations.  A single solve is sufficient; the loop exits after the
first step.

Not yet implemented
-------------------
- Penalty reparameterisation for improved numerical stability near singular
  penalty matrices.
- Negative Newton working weights (requires a signed-weight penalized solver).
- Full post-fit smoothness-score derivatives (handled separately in
  :mod:`nampy.gam.fit.postprocess.gaussian_smoothness_postprocess`).

This solver is the Python analogue of mgcv's ``gam.fit3`` (penalized IRLS inner loop).
- Extended / general family support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..linalg.stacked_qr import STACKED_QR_RANK_TOLERANCE, pls_fit1_nonneg_w


@dataclass
class PenalizedIrlsControl:
    """
    Iteration control for the penalized IRLS solver.

    Attributes
    ----------
    maxit : int
        Maximum IRLS iterations before declaring non-convergence.
    epsilon : float
        Relative convergence tolerance for penalized deviance.
    trace : bool
        If True, print deviance and penalized deviance at each iteration.
    """

    maxit: int = 100
    epsilon: float = 1e-7
    trace: bool = False


def _family_is_extended(family: Any) -> bool:
    return str(getattr(family, "family_class", "")).lower() in (
        "extended",
        "general",
    )


def _strictly_additive_gaussian_identity(family: Any) -> bool:
    # Gaussian identity-link: working weights are constant, one solve is enough.
    return (
        str(getattr(family, "name", "")).lower() == "gaussian"
        and str(getattr(family, "link_name", "")).lower() == "identity"
    )


def fit_penalized_irls(
    X: np.ndarray,
    y: np.ndarray,
    log_smoothing_params: np.ndarray,
    penalty_sqrt_work: np.ndarray,
    penalty_sqrt_balanced: np.ndarray,
    St: np.ndarray,
    family: Any,
    *,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    control: PenalizedIrlsControl | None = None,
    start: np.ndarray | None = None,
    etastart: np.ndarray | None = None,
    mustart: np.ndarray | None = None,
    null_coef: np.ndarray | None = None,
    score_type: str = "REML",
    deriv: int = 2,
    scale_reference: float = 1.0,
    fisher_scoring_only: bool = False,
    pls_rank_tol: float = STACKED_QR_RANK_TOLERANCE,
    coef_method: str = "householder",
    near_singular_null_pin: bool | Literal["auto"] = False,
) -> dict[str, Any]:
    """
    Penalized IRLS solver for a GAM with fixed smoothing parameters.

    Iterates the penalized weighted least-squares problem until convergence or
    ``control.maxit`` iterations.  Returns coefficients and diagnostic quantities
    but does **not** compute smoothness-selection scores; use
    :func:`fit_penalized_irls_from_model` with ``attach_smoothness_postprocess=True`` for that.

    Parameters
    ----------
    X
        Model matrix, shape ``(n, q)``.
    y
        Response vector, shape ``(n,)``.
    log_smoothing_params
        Log-scale smoothing parameters; stored in the output dict for reference.
    penalty_sqrt_work
        Square-root factor ``E`` of the total penalty ``S = E'E``, used to form
        the augmented system in each penalized least-squares step.
    penalty_sqrt_balanced
        Balanced penalty square root used for numerical rank detection.
    St
        Total penalty matrix ``S = sum_k lambda_k * S_k``, shape ``(q, q)``.
        Used in quadratic-form evaluations and the convergence gradient check.
    family
        A :class:`nampy.gam.families.base.GLMFamily` instance.
    weights
        Prior observation weights (positive where informative).  Default: all ones.
    offset
        Fixed additive offset on the link scale.  Default: all zeros.
    control
        Convergence tolerances; see :class:`PenalizedIrlsControl`.
    start, etastart, mustart
        Optional starting values for coefficients, linear predictor, or mean.
    null_coef
        Coefficients for the null linear predictor used in step-halving guards.
        Default: zero vector.
    score_type, deriv
        Passed through to the output dict for downstream scoring; no derivatives
        are computed here.
    scale_reference
        Known or estimated scale used in the gradient convergence test.
    fisher_scoring_only
        If True, always use Fisher (expected-information) working weights.
    pls_rank_tol, coef_method, near_singular_null_pin
        Forwarded to :func:`pls_fit1_nonneg_w`.

    Returns
    -------
    dict
        Keys: ``coef``, ``linear_predictor`` (includes offset), ``mu``,
        ``deviance``, ``penalty_quadratic``, ``penalized_deviance``,
        ``converged``, ``iterations``, ``warnings``, ``score_type``,
        ``deriv``, ``log_smoothing_params``.  ``reml_score`` and ``gcv_score``
        are ``None`` until post-processing is applied.
    """
    # Extended / general families require a different solver; not yet implemented.
    if _family_is_extended(family):
        raise NotImplementedError(
            "Extended and general family support is not yet implemented."
        )

    if int(deriv) not in (0, 1, 2):
        raise ValueError("deriv must be 0, 1, or 2 (unsupported order).")

    ctl = control or PenalizedIrlsControl()
    X = np.asarray(X, dtype=np.float64)
    y = family.validate_y(y)
    nobs, q = X.shape
    St = np.asarray(St, dtype=np.float64)
    if St.shape != (q, q):
        raise ValueError(f"St must be ({q}, {q}), got {St.shape}.")

    penalty_sqrt_work = np.asarray(penalty_sqrt_work, dtype=np.float64)
    penalty_sqrt_balanced = np.asarray(penalty_sqrt_balanced, dtype=np.float64)

    if weights is None:
        weights = np.ones(nobs, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.shape[0] != nobs:
        raise ValueError("weights must match nrow(X).")

    if offset is None:
        offset = np.zeros(nobs, dtype=np.float64)
    else:
        offset = np.asarray(offset, dtype=np.float64).ravel()
    if offset.shape[0] != nobs:
        raise ValueError("offset must match nrow(X).")

    log_sp = np.asarray(log_smoothing_params, dtype=np.float64).ravel()

    use_fisher = bool(
        fisher_scoring_only or bool(getattr(family, "canonical_link", False))
    )

    warnings_list: list[str] = []

    # Empty model (no columns): return null-fit quantities directly.
    if q == 0:
        eta = np.zeros(nobs, dtype=np.float64) + offset
        mu = np.asarray(family.inverse_link(eta), dtype=np.float64)
        dev = float(family.deviance(y, mu))
        return {
            "coef": np.zeros(0, dtype=np.float64),
            "linear_predictor": eta,
            "mu": mu,
            "deviance": dev,
            "penalty_quadratic": 0.0,
            "penalized_deviance": dev,
            "converged": True,
            "iterations": 0,
            "warnings": warnings_list,
            "score_type": score_type,
            "deriv": int(deriv),
            "log_smoothing_params": log_sp,
            "reml_score": None,
            "edf": None,
        }

    if null_coef is None:
        null_coef = np.zeros(q, dtype=np.float64)
    else:
        null_coef = np.asarray(null_coef, dtype=np.float64).ravel()
        if null_coef.shape[0] != q:
            raise ValueError("null_coef length must match ncol(X).")

    # Compute initial linear predictor and mean; record null penalized deviance.
    if etastart is not None:
        eta = np.asarray(etastart, dtype=np.float64).ravel()
        if eta.shape[0] != nobs:
            raise ValueError("etastart must have length nrow(X).")
    elif start is not None:
        start_v = np.asarray(start, dtype=np.float64).ravel()
        if start_v.shape[0] != q:
            raise ValueError("start must have length ncol(X).")
        eta = offset + (X @ start_v)
    elif mustart is not None:
        mu0 = np.asarray(mustart, dtype=np.float64).ravel()
        eta = np.asarray(family.link(mu0), dtype=np.float64)
        if eta.shape[0] != nobs:
            raise ValueError("mustart must have length nrow(X).")
    else:
        mu0 = family.initialize_mu(y)
        eta = np.asarray(family.link(mu0), dtype=np.float64)

    mu = np.asarray(family.inverse_link(eta), dtype=np.float64)
    coef_old = null_coef.copy()
    null_eta = (X @ null_coef + offset).astype(np.float64, copy=False)
    old_pdev = float(family.deviance(y, family.inverse_link(null_eta)) + float(null_coef @ (St @ null_coef)))

    ii = 0
    while True:
        if np.all(np.isfinite(mu)) and np.all(np.isfinite(eta)):
            break
        ii += 1
        if ii > 20:
            raise RuntimeError("Cannot find valid starting values (eta/mu).")
        if start is not None:
            start = 0.9 * np.asarray(start, dtype=np.float64).ravel() + 0.1 * coef_old
        eta = 0.9 * eta + 0.1 * null_eta
        mu = np.asarray(family.inverse_link(eta), dtype=np.float64)

    strictly_additive = _strictly_additive_gaussian_identity(family)

    conv = False
    coef = np.zeros(q, dtype=np.float64)
    # Initialise "previous" state to the null fit for the first step-halving check.
    eta_old = null_eta.copy()
    n_iter = 0

    # Main penalized IRLS loop.
    for it in range(ctl.maxit):
        iter_no = it + 1
        n_iter = iter_no
        good_prior = weights > 0.0
        var_val = np.asarray(family.variance(mu), dtype=np.float64)
        if np.any(~np.isfinite(var_val[good_prior])):
            raise RuntimeError("NAs in V(mu)")
        if np.any(var_val[good_prior] <= 0.0):
            raise RuntimeError("0s in V(mu)")

        mu_eta_val = np.asarray(family.mu_eta(eta), dtype=np.float64)
        if np.any(~np.isfinite(mu_eta_val[good_prior])):
            raise RuntimeError("NAs in d(mu)/d(eta)")

        good = good_prior & (mu_eta_val != 0.0)
        if not np.any(good):
            warnings_list.append(
                f"fit_penalized_irls: no informative observations at iteration {iter_no}"
            )
            conv = False
            break

        mevg = mu_eta_val[good]
        mug = mu[good]
        yg = y[good]
        weg = weights[good]
        var_mug = var_val[good]
        eta_g = eta[good]
        off_g = offset[good]
        X_g = X[good, :]

        if use_fisher:
            # Fisher scoring: positive-definite expected-information weights.
            z = (eta_g - off_g) + (yg - mug) / mevg
            w_work = (weg * mevg**2) / var_mug
        else:
            # Newton step: observed-information weights (may be negative).
            c = yg - mug
            dvar = np.asarray(family.dvar(mug), dtype=np.float64)
            d2link = np.asarray(family.d2link(mug), dtype=np.float64)
            alpha = 1.0 + c * (dvar / var_mug + d2link * mevg)
            eps_m = np.finfo(np.float64).eps
            alpha = np.where(alpha == 0.0, eps_m, alpha)
            z = (eta_g - off_g) + (yg - mug) / (mevg * alpha)
            w_work = weg * alpha * mevg**2 / var_mug

        if np.any(~np.isfinite(w_work)) or np.any(w_work < 0.0):
            # Newton weights are indefinite; fall back to Fisher scoring.
            # (A signed-weight solver would be needed to handle the Newton branch fully.)
            z = (eta_g - off_g) + (yg - mug) / mevg
            w_work = (weg * mevg**2) / var_mug

        wy = w_work * z
        start_coef, penalty = pls_fit1_nonneg_w(
            X_g,
            z,
            w_work,
            wy,
            penalty_sqrt_E=penalty_sqrt_work,
            penalty_rank_Es=penalty_sqrt_balanced,
            rank_tol=pls_rank_tol,
            coef_method=coef_method,
            near_singular_null_pin=near_singular_null_pin,
            P_for_gauge=St,
        )

        if np.any(~np.isfinite(start_coef)):
            warnings_list.append(
                f"fit_penalized_irls: non-finite coefficients at iteration {iter_no}"
            )
            conv = False
            break

        # Update linear predictor and mean after the penalized LS step.
        eta_linear = X @ start_coef
        eta = eta_linear + offset
        mu = np.asarray(family.inverse_link(eta), dtype=np.float64)

        dev = float(family.deviance(y, mu))

        if ctl.trace:
            print(f"Deviance = {dev}  Iteration = {iter_no}")

        # Step-halve until deviance is finite.
        boundary = False
        if not np.isfinite(dev):
            ii_h = 0
            while not np.isfinite(dev):
                ii_h += 1
                if ii_h > ctl.maxit:
                    raise RuntimeError("inner loop 1: cannot correct step size")
                start_coef = 0.5 * (start_coef + coef_old)
                eta_linear = X @ start_coef
                eta = eta_linear + offset
                mu = np.asarray(family.inverse_link(eta), dtype=np.float64)
                dev = float(family.deviance(y, mu))
            boundary = True
            penalty = float(start_coef @ (St @ start_coef))
            if ctl.trace:
                print(f"Step halved: new deviance = {dev}")

        # Step-halve until eta and mu are valid (family-specific validity hooks).
        valid_eta = getattr(family, "valid_eta", None)
        valid_mu = getattr(family, "valid_mu", None)
        if callable(valid_eta) or callable(valid_mu):
            ii_h = 0
            while True:
                ok_eta = True if not callable(valid_eta) else bool(valid_eta(eta))
                ok_mu = True if not callable(valid_mu) else bool(valid_mu(mu))
                if ok_eta and ok_mu:
                    break
                ii_h += 1
                if ii_h > ctl.maxit:
                    raise RuntimeError("inner loop 2: cannot correct step size")
                start_coef = 0.5 * (start_coef + coef_old)
                eta_linear = X @ start_coef
                eta = eta_linear + offset
                mu = np.asarray(family.inverse_link(eta), dtype=np.float64)
            if ii_h > 0:
                boundary = True
                penalty = float(start_coef @ (St @ start_coef))
                dev = float(family.deviance(y, mu))
                if ctl.trace:
                    print(f"Step halved: new deviance = {dev}")

        pdev = dev + penalty

        if ctl.trace:
            print(f"penalized deviance = {pdev}")

        # Step-halve if penalized deviance diverges relative to previous iteration.
        div_thresh = 10.0 * (0.1 + abs(old_pdev)) * np.sqrt(np.finfo(np.float64).eps)
        if pdev - old_pdev > div_thresh:
            ii_h = 0
            if iter_no == 1:
                coef_old = null_coef.copy()
                eta_old = null_eta.copy()
            while pdev - old_pdev > div_thresh:
                ii_h += 1
                if ii_h > 100:
                    raise RuntimeError("inner loop 3: cannot correct step size")
                start_coef = 0.5 * (start_coef + coef_old)
                eta_linear = X @ start_coef
                eta = eta_linear + offset
                mu = np.asarray(family.inverse_link(eta), dtype=np.float64)
                dev = float(family.deviance(y, mu))
                pdev = dev + float(start_coef @ (St @ start_coef))
                if ctl.trace:
                    print(f"Step halved: new penalized deviance = {pdev}")

        # Gaussian identity-link: working weights are constant, converged after one step.
        if strictly_additive:
            conv = True
            coef = start_coef
            break

        # Convergence check: penalized deviance change and gradient magnitude.
        scale = float(scale_reference)
        if abs(pdev - old_pdev) < ctl.epsilon * (abs(scale) + abs(pdev)):
            grad = 2.0 * (X_g.T @ (w_work * ((X_g @ start_coef) - z))) + 2.0 * (St @ start_coef)
            if np.max(np.abs(grad)) > ctl.epsilon * (abs(pdev) + abs(scale)):
                old_pdev = pdev
                coef = coef_old = start_coef
                eta_old = eta.copy()
            else:
                conv = True
                coef = start_coef
                eta_old = eta.copy()
                break
        else:
            old_pdev = pdev
            coef = coef_old = start_coef
            eta_old = eta.copy()
    else:
        if not conv:
            warnings_list.append(
                f"fit_penalized_irls: reached maxit={ctl.maxit} without meeting convergence criteria"
            )

    mu = np.asarray(family.inverse_link(eta), dtype=np.float64)
    dev = float(family.deviance(y, mu))
    penalty_out = float(coef @ (St @ coef))

    return {
        "coef": coef,
        "linear_predictor": eta,
        "mu": mu,
        "deviance": dev,
        "penalty_quadratic": penalty_out,
        "penalized_deviance": dev + penalty_out,
        "converged": bool(conv),
        "iterations": int(n_iter),
        "warnings": warnings_list,
        "score_type": score_type,
        "deriv": int(deriv),
        "log_smoothing_params": log_sp,
        "reml_score": None,
        "edf": None,
    }


def fit_penalized_irls_from_model(
    model: Any,
    y: np.ndarray,
    smoothing_params: np.ndarray,
    *,
    attach_smoothness_postprocess: bool = True,
) -> dict[str, Any]:
    """
    Run :func:`fit_penalized_irls` using a fitted model's compiled design and penalty blocks.

    Builds the model matrix and total penalty from the model's compiled design,
    then calls :func:`fit_penalized_irls`.  When ``attach_smoothness_postprocess`` is True and the family
    is Gaussian, post-fit smoothness scores and derivatives are merged in via
    :func:`nampy.gam.fit.postprocess.gaussian_smoothness_postprocess.merge_gaussian_smoothness_into_fit_result`.
    """
    from ..penalized_system import (
        build_full_design,
        build_full_penalty_from_blocks,
    )
    from ..linalg.stacked_qr import (
        balanced_penalty_template_sqrt_for_rank,
        penalty_sqrt_rows_prefer_diagonal,
    )

    y = model.family.validate_y(y)
    n = int(y.shape[0])
    w = getattr(model, "prior_weights_", None)
    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).ravel()

    X = build_full_design(model.Z, fit_intercept=model.fit_intercept)
    P = build_full_penalty_from_blocks(
        penalty_blocks=model.penalty_blocks_,
        smoothing_params=np.asarray(smoothing_params, dtype=np.float64).ravel(),
        fit_intercept=model.fit_intercept,
        n_coef=model.n_coef_,
    )
    Sr, Eb_template = penalty_sqrt_rows_prefer_diagonal(P)
    Eb = balanced_penalty_template_sqrt_for_rank(
        model.penalty_blocks_,
        fit_intercept=model.fit_intercept,
        n_coef=int(model.n_coef_),
    )
    offset = model.offset_train_
    off = np.zeros(n, dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64).ravel()

    log_sp = np.log(np.maximum(np.asarray(smoothing_params, dtype=np.float64).ravel(), np.finfo(np.float64).tiny))

    from ..postprocess.gaussian_smoothness_postprocess import (
        merge_gaussian_smoothness_into_fit_result,
    )

    score_type = str(getattr(model, "smoothing_method", "REML")).upper()
    out = fit_penalized_irls(
        X,
        y,
        log_sp,
        penalty_sqrt_work=Sr,
        penalty_sqrt_balanced=Eb,
        St=P,
        family=model.family,
        weights=w,
        offset=off,
        control=PenalizedIrlsControl(
            maxit=int(getattr(model, "max_irls_iter", 100)),
            epsilon=float(getattr(model, "irls_tol", 1e-7)),
            trace=False,
        ),
        score_type=score_type,
    )
    if attach_smoothness_postprocess and str(getattr(model.family, "name", "")).lower() == "gaussian":
        sp_lin = np.asarray(smoothing_params, dtype=np.float64).ravel()
        out = merge_gaussian_smoothness_into_fit_result(
            out,
            model,
            y,
            sp_lin,
            score_type=score_type,
            deriv=2,
        )
    return out

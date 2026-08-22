"""Generic separable transformed-coefficient Newton solver.

The design acts on prediction coefficients ``beta.t`` while the roughness
penalty acts on unconstrained optimization coefficients ``beta``.  Selected
coordinates are linked by exp or SCAM's softplus ``notExp`` map.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.optimize import lsq_linear

from ...model_state import (
    _compiled_model,
    _design_matrix,
    _fit_intercept,
    _n_coef,
    _penalty_blocks_seq,
)
from ..capabilities import coefficient_transform, observation_transform
from ..penalized_system import build_full_design, build_full_penalty_from_blocks
from ..state import FitCoreSolution

_USE_MODEL_START = object()


def _prior_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    out = np.asarray(weights, dtype=np.float64).reshape(-1)
    if out.shape != (n,) or not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError(f"sample weights must be finite, non-negative, shape ({n},).")
    return out


def _valid_family_state(family, eta, mu) -> bool:
    valid_eta = getattr(family, "valid_eta", lambda value: np.all(np.isfinite(value)))
    valid_mu = getattr(family, "valid_mu", lambda value: np.all(np.isfinite(value)))
    return bool(valid_eta(eta) and valid_mu(mu))


def _penalty_root(S: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (S + S.T))
    values = np.maximum(values, 0.0)
    return np.asarray(np.sqrt(values)[:, None] * vectors.T, dtype=np.float64)


def _initialize_optimization_coefficients(
    model,
    X,
    y,
    weights,
    offset,
    S,
    transform,
    supplied_start=_USE_MODEL_START,
) -> np.ndarray:
    supplied = (
        getattr(model, "start", None)
        if supplied_start is _USE_MODEL_START
        else supplied_start
    )
    q = X.shape[1]
    if supplied is not None:
        beta = np.asarray(supplied, dtype=np.float64).reshape(-1)
        if beta.shape != (q,):
            raise ValueError(f"start must have length {q}, got {beta.size}.")
        return beta.copy()

    mu0 = np.asarray(model.family.initialize_mu(y), dtype=np.float64)
    eta0 = np.asarray(model.family.link(mu0), dtype=np.float64) - offset
    root = _penalty_root(S)
    augmented_X = np.vstack([np.sqrt(weights)[:, None] * X, root])
    augmented_y = np.concatenate([np.sqrt(weights) * eta0, np.zeros(q)])
    lower = np.full(q, -np.inf, dtype=np.float64)
    lower[transform.positive_mask] = 1e-12
    theta0 = lsq_linear(
        augmented_X,
        augmented_y,
        bounds=(lower, np.full(q, np.inf)),
        tol=1e-12,
        lsmr_tol=1e-12,
        max_iter=500,
    ).x
    # scam.fit initializes its optimization coordinates with log(beta.t),
    # including when not.exp=TRUE.
    beta = np.asarray(theta0, dtype=np.float64)
    beta[transform.positive_mask] = np.log(beta[transform.positive_mask])
    return beta


def _working_state(model, X, y, weights, offset, S, transform, beta):
    beta_t = transform.forward(beta)
    eta = np.asarray(X @ beta_t + offset, dtype=np.float64)
    mu = np.asarray(model.family.inverse_link(eta), dtype=np.float64)
    if not _valid_family_state(model.family, eta, mu):
        raise FloatingPointError("Invalid eta/mu in shape-constrained Newton state.")
    d1 = transform.derivative(beta, order=1)
    d2 = transform.derivative(beta, order=2)
    X1 = np.asarray(X * d1[None, :], dtype=np.float64)
    mu_eta = np.asarray(model.family.mu_eta(eta), dtype=np.float64)
    g_deriv = 1.0 / mu_eta
    variance = np.asarray(model.family.variance(mu), dtype=np.float64)
    w1 = weights / (variance * g_deriv**2)
    residual = y - mu
    alpha = 1.0 + residual * (
        np.asarray(model.family.dvar(mu), dtype=np.float64) / variance
        + np.asarray(model.family.d2link(mu), dtype=np.float64) / g_deriv
    )
    working_weights = w1 * alpha
    E_diag = d2 * np.asarray(X.T @ (w1 * g_deriv * residual), dtype=np.float64)
    gradient = -X1.T @ (w1 * g_deriv * residual) + S @ beta
    hessian = X1.T @ (working_weights[:, None] * X1) - np.diag(E_diag) + S
    deviance = float(model.family.deviance(y, mu, weights=weights))
    penalty = float(beta @ (S @ beta))
    return {
        "beta_t": beta_t,
        "eta": eta,
        "mu": mu,
        "d1": d1,
        "d2": d2,
        "X1": X1,
        "g_deriv": g_deriv,
        "w1": w1,
        "alpha": alpha,
        "working_weights": working_weights,
        "E_diag": E_diag,
        "gradient": np.asarray(gradient, dtype=np.float64),
        "hessian": np.asarray(0.5 * (hessian + hessian.T), dtype=np.float64),
        "deviance": deviance,
        "penalty": penalty,
        "penalized_deviance": deviance + penalty,
    }


def _newton_candidate(state, beta, S) -> tuple[np.ndarray, str]:
    hessian = np.asarray(state["hessian"], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues.size and float(np.min(eigenvalues)) > 0.0:
        threshold = float(np.max(eigenvalues)) * np.sqrt(np.finfo(np.float64).eps)
        if float(np.min(eigenvalues)) >= threshold:
            step = solve(hessian, state["gradient"], assume_a="sym")
            return np.asarray(beta - step, dtype=np.float64), "newton"
        # scam/R/scam.r switches to an SVD step when the pivoted augmented
        # system is rank deficient, retaining singular values above
        # max(d) * .Machine$double.eps^.5.
        step = np.linalg.pinv(
            hessian,
            rcond=np.sqrt(np.finfo(np.float64).eps),
            hermitian=True,
        ) @ state["gradient"]
        return np.asarray(beta - step, dtype=np.float64), "newton_svd"

    fisher = state["X1"].T @ (state["w1"][:, None] * state["X1"]) + S
    # The upstream fallback is a penalized Fisher scoring solve. Express its
    # normal equations through the current gradient to avoid pseudodata
    # cancellation while preserving the same root.
    fisher = 0.5 * (fisher + fisher.T)
    values = np.linalg.eigvalsh(fisher)
    threshold = float(np.max(values)) * np.sqrt(np.finfo(np.float64).eps)
    if float(np.min(values)) >= threshold:
        step = solve(fisher, state["gradient"], assume_a="sym")
        kind = "fisher"
    else:
        step = np.linalg.pinv(
            fisher,
            rcond=np.sqrt(np.finfo(np.float64).eps),
            hermitian=True,
        ) @ state["gradient"]
        kind = "fisher_svd"
    return np.asarray(beta - step, dtype=np.float64), kind


def solve_transformed_coefficient_fit(
    model,
    y,
    smoothing_params,
    weights=None,
    *,
    initial_coefficients=_USE_MODEL_START,
    tolerance: float | None = None,
):
    """Fit a one-predictor model with a separable coefficient transform."""
    compiled = _compiled_model(model)
    if compiled is None:
        raise RuntimeError("Transformed-coefficient fitting requires a compiled model.")
    transform = coefficient_transform(model)
    if transform.is_identity:
        raise ValueError("Transformed-coefficient backend requires a non-identity map.")
    if (
        len(compiled.predictors) > 1
        or str(getattr(model.family, "family_class", "")).lower() == "general"
    ):
        from .transformed_general import solve_transformed_general_family_fit

        return solve_transformed_general_family_fit(
            model, y, smoothing_params, weights=weights
        )
    mask = getattr(transform, "positive_mask", None)
    if mask is None:
        raise NotImplementedError(
            "The current transformed-coefficient Newton kernel requires a "
            "coordinatewise positivity mask."
        )
    mask = np.asarray(mask, dtype=bool)
    if compiled.fit_to_prediction_parameterization_map is not None:
        raise NotImplementedError(
            "Nonlinear coefficient transforms cannot yet be combined with a non-identity "
            "global prediction parameterization map."
        )

    y = model.family.validate_y(y)
    y_original = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    prior = _prior_weights(weights, n)
    X = build_full_design(_design_matrix(model), _fit_intercept(model))
    X_original = np.asarray(X, dtype=np.float64).copy()
    if mask.shape != (X.shape[1],):
        raise RuntimeError(
            f"Compiled positivity mask has shape {mask.shape}, expected {(X.shape[1],)}."
        )
    S = build_full_penalty_from_blocks(
        _penalty_blocks_seq(model),
        smoothing_params,
        _fit_intercept(model),
        _n_coef(model),
    )
    offset = (
        np.zeros(n, dtype=np.float64)
        if model.offset_train_ is None
        else np.asarray(model.offset_train_, dtype=np.float64).reshape(-1)
    )
    obs_transform = observation_transform(model)
    ar1_rho = float(getattr(model, "ar1_rho", 0.0))
    if not obs_transform.is_identity:
        if (
            str(getattr(model.family, "name", "")).lower() != "gaussian"
            or str(getattr(model.family, "link_name", "identity")).lower()
            != "identity"
        ):
            raise ValueError(
                "AR(1) residual correlation is available only for Gaussian identity models."
            )
        X, y, offset = obs_transform.transform_system(X, y, offset)
    beta = _initialize_optimization_coefficients(
        model,
        X,
        y,
        prior,
        offset,
        S,
        transform,
        supplied_start=initial_coefficients,
    )
    state = _working_state(model, X, y, prior, offset, S, transform, beta)
    trace = []
    converged = False
    max_iter = int(getattr(model, "max_irls_iter", 200))
    tol = (
        float(getattr(model, "irls_tol", 1e-7))
        if tolerance is None
        else float(tolerance)
    )
    max_halves = max(int(getattr(model, "max_step_halving", 25)), 100)

    for iteration in range(1, max_iter + 1):
        candidate, step_kind = _newton_candidate(state, beta, S)
        trial = None
        for _halvings in range(max_halves + 1):
            try:
                trial = _working_state(
                    model, X, y, prior, offset, S, transform, candidate
                )
            except (FloatingPointError, ValueError, OverflowError):
                trial = None
            threshold = (
                10.0
                * (0.1 + abs(float(state["penalized_deviance"])))
                * np.sqrt(np.finfo(np.float64).eps)
            )
            if (
                trial is not None
                and np.isfinite(trial["penalized_deviance"])
                and trial["penalized_deviance"]
                - state["penalized_deviance"]
                <= threshold
            ):
                break
            candidate = 0.5 * (candidate + beta)
        if trial is None:
            raise RuntimeError(
                "Transformed-coefficient Newton step-halving failed to find a valid state."
            )

        grad_inf = float(np.max(np.abs(trial["gradient"])))
        relative = abs(
            float(trial["penalized_deviance"] - state["penalized_deviance"])
        ) / (0.1 + abs(float(trial["penalized_deviance"])))
        gradient_limit = tol * float(np.max(np.abs(candidate + beta))) / 2.0
        converged_here = bool(relative < tol and grad_inf <= gradient_limit)
        trace.append(
            {
                "iter": iteration,
                "deviance": float(trial["deviance"]),
                "penalized_deviance": float(trial["penalized_deviance"]),
                "grad_inf_norm": grad_inf,
                "step_halvings": _halvings,
                "step_kind": step_kind,
                "converged_here": converged_here,
            }
        )
        beta = candidate
        state = trial
        if converged_here:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "Transformed-coefficient Newton solver did not converge in "
            f"{max_iter} iterations."
        )

    fit_hessian = np.asarray(state["hessian"], dtype=np.float64)
    fit_hessian_inv = np.linalg.pinv(fit_hessian, hermitian=True)
    fit_fisher_cross = state["X1"].T @ (
        state["w1"][:, None] * state["X1"]
    )
    fit_edf = float(np.trace(fit_hessian_inv @ fit_fisher_cross))
    post_state = state
    if ar1_rho != 0.0:
        # scam/R/scam.r::scam.fit.post deliberately recomputes all post-fit
        # Hessian, EDF, covariance, scale, eta and mu quantities on the
        # original (unwhitened) data after estimating beta on the AR root scale.
        post_state = _working_state(
            model,
            X_original,
            y_original,
            prior,
            offset,
            S,
            transform,
            beta,
        )
    hessian = np.asarray(post_state["hessian"], dtype=np.float64)
    hessian_inv = np.linalg.pinv(hessian, hermitian=True)
    observed_cross = post_state["X1"].T @ (
        post_state["working_weights"][:, None] * post_state["X1"]
    )
    # scam.fit.post computes F = P %*% KtILQ1R. Algebraically this is the
    # inverse observed penalized Hessian applied to the expected (Fisher)
    # data Hessian X1' W1 X1; alpha belongs to the observed Hessian only.
    fisher_cross = post_state["X1"].T @ (
        post_state["w1"][:, None] * post_state["X1"]
    )
    hat_coef = np.asarray(hessian_inv @ fisher_cross, dtype=np.float64)
    post_edf = float(np.trace(hat_coef))
    edf = fit_edf if ar1_rho != 0.0 else post_edf
    scale = float(
        model.family.estimate_dispersion(
            y_original, post_state["mu"], edf=post_edf, weights=prior
        )
    )
    cov_bayes_opt = np.asarray(scale * hessian_inv, dtype=np.float64)
    cov_freq_opt = np.asarray(
        scale * hessian_inv @ observed_cross @ hessian_inv, dtype=np.float64
    )
    beta_t = np.asarray(post_state["beta_t"], dtype=np.float64)
    # SCAM terms declare their released beta.t covariance scaling on their
    # transform block. Generic nonlinear blocks default to Jacobian transport.
    cov_bayes = transform.transport_covariance(beta, cov_bayes_opt)
    cov_freq = transform.transport_covariance(beta, cov_freq_opt)
    residual = y_original - post_state["mu"]
    if ar1_rho != 0.0:
        eta_original = X_original @ beta_t + offset
        mu_original = np.asarray(
            model.family.inverse_link(eta_original), dtype=np.float64
        )
        original_residuals = y_original - mu_original
        model.ar1_standardized_residuals_ = obs_transform.apply(original_residuals)
        eta_output = eta_original
        mu_output = mu_original
        residual_output = original_residuals
        deviance_output = float(
            model.family.deviance(y_original, mu_original, weights=prior)
        )
    else:
        model.ar1_standardized_residuals_ = None
        eta_output = post_state["eta"]
        mu_output = post_state["mu"]
        residual_output = residual
        deviance_output = float(state["deviance"])

    payload = {
        "coef_full": beta_t,
        "coef_optimization": beta,
        "positive_coefficient_mask": mask,
        "intercept": float(beta_t[0]) if _fit_intercept(model) else 0.0,
        "beta": beta_t[1:].copy() if _fit_intercept(model) else beta_t.copy(),
        "eta": eta_output,
        "mu": mu_output,
        "rss": float(np.sum(prior * residual_output**2)),
        "deviance": deviance_output,
        "edf": edf,
        "trace_H": edf,
        "scale": scale,
        "cov_bayes": cov_bayes,
        "cov_freq": cov_freq,
        "cov_unconditional": None,
        "cov_bayes_optimization": cov_bayes_opt,
        "cov_freq_optimization": cov_freq_opt,
        "H_coef": hat_coef,
        "penalty_quadratic": float(state["penalty"]),
        "loglik": float(model.family.loglik(y_original, mu_output, scale=scale)),
        "converged": converged,
        "iter": len(trace),
        "failed_step": False,
        "failure_reason": None,
        "inner_trace": trace,
        "coef_space": "prediction",
        "cov_bayes_space": "prediction",
        "cov_freq_space": "prediction",
        "X": X,
        "A": hessian,
        "A_inv": hessian_inv,
        "XtWX": observed_cross,
        "P": None,
        "penalty_matrix": S,
        "working_weights": post_state["working_weights"],
        "fisher_weights": post_state["w1"],
        "working_response": post_state["g_deriv"] * (
            y_original - post_state["mu"]
        )
        + post_state["X1"] @ beta,
        "offset": offset,
        "penalized_system_rank": int(np.linalg.matrix_rank(hessian)),
    }
    return FitCoreSolution.from_dict(payload)


def solve_shape_constrained_fit(*args, **kwargs):
    """Compatibility alias for the generic transformed-coefficient solver."""
    return solve_transformed_coefficient_fit(*args, **kwargs)


__all__ = ["solve_shape_constrained_fit", "solve_transformed_coefficient_fit"]

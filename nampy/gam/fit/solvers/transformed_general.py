"""Fixed-smoothing Newton kernel for transformed multi-predictor models."""

from __future__ import annotations

import numpy as np

from ...model_state import _compiled_model
from ..capabilities import coefficient_transform
from ..state import FitCoreSolution
from .general_family.fixed_smoothing import build_general_family_setup_state


def _penalty_root(S: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (S + S.T))
    keep = values > max(float(np.max(values)), 1.0) * np.finfo(float).eps**0.75
    if not np.any(keep):
        return np.empty((0, S.shape[0]), dtype=np.float64)
    return np.asarray(np.sqrt(values[keep])[:, None] * vectors[:, keep].T)


def _initial_optimization_coefficients(family, y, setup, weights, transform):
    beta = np.asarray(
        family.initialize(
            y,
            setup.X_full,
            setup.jj,
            offset=setup.offset_list,
            weights=weights,
            E=_penalty_root(setup.St),
        ),
        dtype=np.float64,
    ).reshape(-1)
    if beta.shape != (transform.size,):
        raise RuntimeError(
            f"General-family initializer returned {beta.size} coefficients; "
            f"the compiled transform requires {transform.size}."
        )
    mask = getattr(transform, "positive_mask", None)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        beta[mask] = np.maximum(beta[mask], 1e-8)
    return np.asarray(transform.inverse(beta), dtype=np.float64)


def _likelihood_state(family, y, X, jj, offset, weights, transform, theta):
    beta = np.asarray(transform.forward(theta), dtype=np.float64)
    likelihood = family.ll(
        y, X, jj, beta, weights, offset=offset, deriv=1
    )
    score_beta = np.asarray(likelihood["lb"], dtype=np.float64).reshape(-1)
    hessian_beta = np.asarray(likelihood["lbb"], dtype=np.float64)
    d1 = np.asarray(transform.derivative(theta, order=1), dtype=np.float64)
    d2 = np.asarray(transform.derivative(theta, order=2), dtype=np.float64)
    score_theta = d1 * score_beta
    hessian_theta = (
        d1[:, None] * hessian_beta * d1[None, :]
        + np.diag(d2 * score_beta)
    )
    return {
        "l": float(likelihood["l"]),
        "beta": beta,
        "score": score_theta,
        "hessian": hessian_theta,
    }


def solve_transformed_general_family_fit(
    model, y, smoothing_params, weights=None
):
    """Fit transformed coefficients for any ``GeneralFamily`` at fixed SP."""
    compiled = _compiled_model(model)
    if compiled is None:
        raise RuntimeError("A compiled model is required.")
    if compiled.fit_to_prediction_parameterization_map is not None:
        raise NotImplementedError(
            "Transformed general-family coefficients cannot yet be combined "
            "with a non-identity prediction parameterization map."
        )
    transform = coefficient_transform(model)
    setup = build_general_family_setup_state(
        model, smoothing_params, score_type="fixed"
    )
    if setup.X_full.shape[1] != transform.size:
        raise RuntimeError("Transform and general-family coefficient layouts differ.")
    y = np.asarray(model.family.validate_y(y), dtype=np.float64)
    prior = (
        np.ones(y.size, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if prior.shape != y.shape or np.any(prior < 0.0) or not np.all(np.isfinite(prior)):
        raise ValueError("General-family sample weights must be finite and non-negative.")

    theta = _initial_optimization_coefficients(
        model.family, y, setup, prior, transform
    )
    S = np.asarray(setup.St, dtype=np.float64)
    state = _likelihood_state(
        model.family,
        y,
        setup.X_full,
        setup.jj,
        setup.offset_list,
        prior,
        transform,
        theta,
    )
    objective = -state["l"] + 0.5 * float(theta @ S @ theta)
    trace = []
    converged = False
    tol = float(getattr(model, "irls_tol", 1e-7))
    max_iter = int(getattr(model, "max_irls_iter", 200))
    max_halving = max(int(getattr(model, "max_step_halving", 25)), 25)

    for iteration in range(1, max_iter + 1):
        gradient = -np.asarray(state["score"], dtype=np.float64) + S @ theta
        hessian = -np.asarray(state["hessian"], dtype=np.float64) + S
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian, hermitian=True) @ gradient
        candidate = theta - step
        accepted = None
        trial_objective = np.inf
        halvings = 0
        for _halvings in range(max_halving + 1):
            halvings = _halvings
            try:
                trial = _likelihood_state(
                    model.family,
                    y,
                    setup.X_full,
                    setup.jj,
                    setup.offset_list,
                    prior,
                    transform,
                    candidate,
                )
                trial_objective = -trial["l"] + 0.5 * float(candidate @ S @ candidate)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                trial = None
            if trial is not None and np.isfinite(trial_objective) and trial_objective <= objective:
                accepted = trial
                break
            candidate = 0.5 * (candidate + theta)
        if accepted is None:
            raise RuntimeError(
                "Transformed general-family Newton step failed to decrease the objective."
            )
        relative = abs(objective - trial_objective) / (0.1 + abs(trial_objective))
        theta = candidate
        state = accepted
        objective = trial_objective
        grad_inf = float(
            np.max(np.abs(-np.asarray(state["score"]) + S @ theta))
        )
        converged_here = bool(relative < tol and grad_inf < np.sqrt(tol))
        trace.append(
            {
                "iter": iteration,
                "penalized_deviance": 2.0 * objective,
                "grad_inf_norm": grad_inf,
                "step_halvings": halvings,
                "converged_here": converged_here,
            }
        )
        if converged_here:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"Transformed general-family solver did not converge in {max_iter} iterations."
        )

    beta = np.asarray(state["beta"], dtype=np.float64)
    d1 = np.asarray(transform.derivative(theta, order=1), dtype=np.float64)
    data_information = -np.asarray(state["hessian"], dtype=np.float64)
    penalized_hessian = data_information + S
    covariance_opt = np.linalg.pinv(penalized_hessian, hermitian=True)
    covariance_freq_opt = covariance_opt @ data_information @ covariance_opt
    covariance = transform.transport_covariance(theta, covariance_opt)
    covariance_freq = transform.transport_covariance(theta, covariance_freq_opt)
    influence_opt = covariance_opt @ data_information
    influence = d1[:, None] * influence_opt / d1[None, :]
    eta = np.asarray(
        model.family._stacked_eta(
            setup.X_full,
            setup.jj,
            beta,
            offset=setup.offset_list,
        ),
        dtype=np.float64,
    )
    mu = np.asarray(model.family.predict(eta=eta), dtype=np.float64)
    reduced = np.asarray(beta[setup.reduced_to_full_idx], dtype=np.float64)
    intercept = (
        float(beta[int(np.asarray(setup.predictor_full_slices[0], dtype=int)[0])])
        if setup.predictor_full_slices
        and np.asarray(setup.predictor_full_slices[0]).size
        else 0.0
    )
    sign, logdet = np.linalg.slogdet(penalized_hessian)
    return FitCoreSolution.from_dict(
        {
            "coef_full": beta,
            "coef_optimization": theta,
            "positive_coefficient_mask": np.asarray(
                getattr(transform, "positive_mask", np.zeros(transform.size)),
                dtype=bool,
            ),
            "intercept": intercept,
            "beta": reduced,
            "eta": eta,
            "mu": mu,
            "rss": None,
            "deviance": float(-2.0 * state["l"]),
            "edf": float(np.trace(influence)),
            "trace_H": float(np.trace(influence)),
            "scale": 1.0,
            "cov_bayes": covariance,
            "cov_freq": covariance_freq,
            "cov_unconditional": None,
            "cov_bayes_optimization": covariance_opt,
            "cov_freq_optimization": covariance_freq_opt,
            "H_coef": influence,
            "penalty_quadratic": float(theta @ S @ theta),
            "loglik": float(state["l"]),
            "converged": converged,
            "iter": len(trace),
            "failed_step": False,
            "failure_reason": None,
            "inner_trace": trace,
            "coef_space": "prediction",
            "cov_bayes_space": "prediction",
            "cov_freq_space": "prediction",
            "X": setup.X_full,
            "A": penalized_hessian,
            "A_inv": covariance_opt,
            "XtWX": data_information,
            "P": S,
            "penalty_matrix": S,
            "working_weights": None,
            "fisher_weights": None,
            "working_response": None,
            "offset": None,
            "log_det_XtWX_plus_penalty": (
                float(logdet) if sign > 0.0 else None
            ),
            "penalized_system_rank": int(np.linalg.matrix_rank(penalized_hessian)),
        }
    )


__all__ = ["solve_transformed_general_family_fit"]

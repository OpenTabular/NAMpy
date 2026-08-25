"""GCV/UBRE criteria for separable transformed coefficients.

The kernel is generic over the compiled coefficient transform. Its exact
first-derivative algebra follows ``scam/R/bfgs.r::gcv.ubre_grad``.
"""

from __future__ import annotations

import numpy as np

from ....model_state import (
    _fit_intercept,
    _fit_workspace,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ...capabilities import coefficient_transform, has_transformed_coefficients
from ...penalized_system import build_full_penalty_from_blocks
from ...solvers.shape_constrained import solve_transformed_coefficient_fit
from .pirls.value import expand_smoothing_params_from_log


def is_shape_constrained_model(model) -> bool:
    """Compatibility spelling for transformed-coefficient capability."""
    return has_transformed_coefficients(model)


def is_transformed_coefficient_model(model) -> bool:
    return has_transformed_coefficients(model)


def _penalty_components(model) -> list[np.ndarray]:
    n_sp = _n_smoothing_params(model)
    components = []
    for index in range(n_sp):
        selector = np.zeros(n_sp, dtype=np.float64)
        selector[index] = 1.0
        components.append(
            build_full_penalty_from_blocks(
                _penalty_blocks_seq(model),
                selector,
                _fit_intercept(model),
                _n_coef(model),
            )
        )
    return components


def _transformed_criterion_state(model, y, log_sp):
    log_sp = np.asarray(log_sp, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    compiled = getattr(getattr(model, "gam_result_", None), "compiled_model", None)
    workspace = _fit_workspace(model)
    cached = workspace.get("transformed_gcv_ubre_state", None)
    if (
        isinstance(cached, dict)
        and cached.get("compiled_id") == id(compiled)
        and np.array_equal(cached.get("y"), y)
        and np.array_equal(cached.get("log_sp"), log_sp)
    ):
        return cached

    smoothing_params = expand_smoothing_params_from_log(model, log_sp)
    solution = solve_transformed_coefficient_fit(
        model,
        y,
        smoothing_params,
        weights=getattr(model, "prior_weights_", None),
    )
    state = {
        "compiled_id": id(compiled),
        "y": y.copy(),
        "log_sp": log_sp.copy(),
        "sp": np.asarray(smoothing_params, dtype=np.float64),
        "solution": solution,
    }
    workspace.transformed_gcv_ubre_state = state
    workspace.shape_gcv_ubre_state = state
    return state


def criterion_value_transformed(model, y, log_sp, method="gcv") -> float:
    method = str(method).lower()
    state = _transformed_criterion_state(model, y, log_sp)
    solution = state["solution"]
    n = float(model.n_samples_)
    deviance = float(solution["deviance"])
    trace = float(solution["trace_H"])
    gamma = float(model.score_gamma)
    if method == "gcv":
        denominator = n - gamma * trace
        if not np.isfinite(denominator) or denominator == 0.0:
            return np.inf
        return float(n * deviance / denominator**2)
    if method in {"ubre", "aic", "ubreaic"}:
        scale = getattr(model.family, "known_scale", None)
        if scale is None:
            raise ValueError(
                "Transformed-coefficient UBRE requires a family with known scale."
            )
        return float(deviance / n - scale + 2.0 * gamma * trace * scale / n)
    raise ValueError(
        "Transformed-coefficient smoothing selection supports only GCV or UBRE/AIC."
    )


def criterion_gradient_transformed(model, y, log_sp, method="gcv") -> np.ndarray:
    """Exact ``gcv.ubre_grad`` derivative with respect to free log-SP."""
    method = str(method).lower()
    state = _transformed_criterion_state(model, y, log_sp)
    sp = state["sp"]
    solution = state["solution"]
    beta = np.asarray(solution["coef_optimization"], dtype=np.float64)
    eta = np.asarray(solution["eta"], dtype=np.float64)
    mu = np.asarray(solution["mu"], dtype=np.float64)
    X = np.asarray(solution["X"], dtype=np.float64)
    hessian_inv = np.asarray(solution["A_inv"], dtype=np.float64)
    prior = (
        np.ones_like(mu)
        if getattr(model, "prior_weights_", None) is None
        else np.asarray(model.prior_weights_, dtype=np.float64)
    )
    transform = coefficient_transform(model)
    if transform.size != beta.size:
        raise RuntimeError(
            "Compiled coefficient transform does not match fitted state."
        )
    d1 = transform.derivative(beta, order=1)
    d2 = transform.derivative(beta, order=2)
    d3 = transform.derivative(beta, order=3)
    X1 = X * d1[None, :]
    residual = np.asarray(y, dtype=np.float64) - mu
    family = model.family
    variance = np.asarray(family.variance(mu), dtype=np.float64)
    dvar = np.asarray(family.dvar(mu), dtype=np.float64)
    d2var = np.asarray(family.d2var(mu), dtype=np.float64)
    g_deriv = 1.0 / np.asarray(family.mu_eta(eta), dtype=np.float64)
    d2link = np.asarray(family.d2link(mu), dtype=np.float64)
    d3link = np.asarray(family.d3link(mu), dtype=np.float64)
    w1 = prior / (variance * g_deriv**2)
    alpha = 1.0 + residual * (dvar / variance + d2link / g_deriv)
    working_weight = w1 * alpha
    fisher_cross = X1.T @ (w1[:, None] * X1)
    transform_v = w1 * g_deriv * residual

    # Constants follow scam/R/bfgs.r::gcv.ubre_grad exactly (including the
    # released prior-weight algebra in a2).
    d2link_dlink = d2link / g_deriv
    a2 = w1**2 * (dvar * g_deriv + 2.0 * variance * d2link)
    dvar_var = dvar / variance
    alpha1 = (
        -(dvar_var + d2link_dlink) / g_deriv
        - residual
        * (dvar_var**2 + d2link_dlink**2 - d2var / variance - d3link / g_deriv)
        / g_deriv
    )

    components = _penalty_components(model)
    fixed_mask = (
        np.zeros(len(components), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_indices = np.flatnonzero(~fixed_mask)
    deviance_gradient_beta = X1.T @ (-2.0 * residual / (variance * g_deriv))
    trace = float(solution["trace_H"])
    deviance = float(solution["deviance"])
    n = float(model.n_samples_)
    gamma = float(model.score_gamma)
    gradient = np.empty(free_indices.size, dtype=np.float64)
    deviance_gradient = np.empty(free_indices.size, dtype=np.float64)
    trace_gradient = np.empty(free_indices.size, dtype=np.float64)

    for out_index, penalty_index in enumerate(free_indices):
        penalty_component = np.asarray(components[penalty_index], dtype=np.float64)
        dbeta = -sp[penalty_index] * (hessian_inv @ (penalty_component @ beta))
        deta = X1 @ dbeta
        dX1 = X * (d2 * dbeta)[None, :]
        dw1 = -a2 * deta
        dalpha = alpha1 * deta
        dworking_weight = dw1 * alpha + w1 * dalpha
        dmu = deta / g_deriv
        dg_deriv = d2link * dmu
        dtransform_v = (
            dw1 * g_deriv * residual + w1 * dg_deriv * residual - w1 * g_deriv * dmu
        )
        de_diag = (d3 * dbeta) * (X.T @ transform_v) + d2 * (X.T @ dtransform_v)
        d_hessian = (
            dX1.T @ (working_weight[:, None] * X1)
            + X1.T @ (dworking_weight[:, None] * X1)
            + X1.T @ (working_weight[:, None] * dX1)
            - np.diag(de_diag)
            + sp[penalty_index] * penalty_component
        )
        d_fisher = (
            dX1.T @ (w1[:, None] * X1)
            + X1.T @ (dw1[:, None] * X1)
            + X1.T @ (w1[:, None] * dX1)
        )
        d_trace = float(
            np.trace(
                -hessian_inv @ d_hessian @ hessian_inv @ fisher_cross
                + hessian_inv @ d_fisher
            )
        )
        d_deviance = float(deviance_gradient_beta @ dbeta)
        deviance_gradient[out_index] = d_deviance
        trace_gradient[out_index] = d_trace
        if method == "gcv":
            denominator = n - gamma * trace
            gradient[out_index] = (
                n
                * (d_deviance * denominator + 2.0 * gamma * deviance * d_trace)
                / denominator**3
            )
        elif method in {"ubre", "aic", "ubreaic"}:
            scale = getattr(model.family, "known_scale", None)
            if scale is None:
                raise ValueError(
                    "Transformed-coefficient UBRE requires a family with known scale."
                )
            gradient[out_index] = d_deviance / n + 2.0 * gamma * d_trace * scale / n
        else:
            raise ValueError(
                "Transformed-coefficient smoothing selection supports only "
                "GCV or UBRE/AIC."
            )
    state["deviance_gradient"] = deviance_gradient.copy()
    state["trace_gradient"] = trace_gradient.copy()
    return gradient


# Compatibility spellings for callers that imported the original SCAM-named
# private criterion functions. Dispatch and new code use capability names.
criterion_value_shape = criterion_value_transformed
criterion_gradient_shape = criterion_gradient_transformed


__all__ = [
    "criterion_gradient_shape",
    "criterion_gradient_transformed",
    "criterion_value_shape",
    "criterion_value_transformed",
    "is_shape_constrained_model",
    "is_transformed_coefficient_model",
]

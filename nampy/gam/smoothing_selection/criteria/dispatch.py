"""
Top-level dispatch for smoothing_selection-selection criterion value, gradient, and Hessian.

:func:`criterion_value` — scalar criterion at a given log-smoothing-parameter vector.
:func:`criterion_gradient` — gradient w.r.t. log-smoothing parameters (exact when
    available, finite-difference fallback otherwise).
:func:`criterion_hessian` — Hessian (exact or finite-difference).
:func:`criterion_infinite_sp_signal` — gradient and curvature signal used by the
    outer optimiser to detect and roll back infinite-smoothing-parameter solutions.
"""

import numpy as np

from ...fit.solvers.general_fit5 import (
    criterion_gradient_ml_reml_general_fit5,
    criterion_hessian_ml_reml_general_fit5,
)
from .gaussian import criterion_gcv_gaussian
from .gaussian_dyn import _gaussian_dynamic_reml_derivative_terms
from .ml_reml import (
    _model_has_random_effect_smooth,
    criterion_ml_reml,
    resolve_ml_reml_scoring_backend,
)
from .pirls import (
    _current_joint_negbin_eval_state,
    _is_joint_negbin_theta_model,
    criterion_gcv_pirls,
    criterion_ml_reml_pirls_frozen_negbin,
    criterion_ubre_pirls,
)
from .pirls_deriv import (
    criterion_gradient_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
)


def criterion_value(model, y, log_sp, method="gcv"):
    method = str(method).lower()
    if method == "gcv":
        if model._uses_closed_form_solver():
            return criterion_gcv_gaussian(model, y, log_sp)
        return criterion_gcv_pirls(model, y, log_sp)
    if method in {"ubre", "aic", "ubreaic"}:
        return criterion_ubre_pirls(model, y, log_sp)
    if method == "ml":
        if _is_joint_negbin_theta_model(model):
            return criterion_ml_reml_pirls_frozen_negbin(model, y, log_sp, "ML")
        return criterion_ml_reml(model, y, log_sp, "ml")
    if method in {"reml", "laml"}:
        if _is_joint_negbin_theta_model(model):
            return criterion_ml_reml_pirls_frozen_negbin(model, y, log_sp, "REML")
        return criterion_ml_reml(model, y, log_sp, method)
    raise ValueError(
        "method must be one of "
        "{'gcv', 'ubre', 'aic', 'ubreaic', 'ml', 'reml', 'laml'}"
    )


def criterion_gradient_numerical(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-5,
    eps_rel=1e-4,
    _baseline_state=None,
):
    """Centered finite-difference gradient of the smoothing criterion."""
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    if x.size == 0:
        return np.empty((0,), dtype=np.float64)

    baseline_state = _baseline_state
    if (
        baseline_state is None
        and _is_joint_negbin_theta_model(model)
        and method in {"ml", "reml", "laml"}
    ):
        baseline_state = _current_joint_negbin_eval_state(model)

    def _value_at(x_eval):
        if baseline_state is not None:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return float(
                criterion_ml_reml_pirls_frozen_negbin(
                    model,
                    y,
                    x_eval,
                    exact_method,
                    baseline_state=baseline_state,
                )
            )
        return float(criterion_value(model, y, x_eval, method=method))

    grad = np.empty_like(x)
    f0 = _value_at(x)

    if not np.isfinite(f0):
        grad.fill(np.nan)
        return grad

    for i in range(x.size):
        step = max(float(eps_abs), float(eps_rel) * (1.0 + abs(float(x[i]))))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += step
        x_minus[i] -= step

        f_plus = _value_at(x_plus)
        f_minus = _value_at(x_minus)

        if np.isfinite(f_plus) and np.isfinite(f_minus):
            grad[i] = (f_plus - f_minus) / (2.0 * step)
        elif np.isfinite(f_plus):
            grad[i] = (f_plus - f0) / step
        elif np.isfinite(f_minus):
            grad[i] = (f0 - f_minus) / step
        else:
            grad[i] = np.nan

    return grad


def criterion_gradient(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-5,
    eps_rel=1e-4,
):
    method = str(method).lower()
    # Gaussian REML/LAML stays on finite differences for criterion derivatives.
    # Exact Gaussian REML outer derivatives in dispatch were derived for an older
    # scale convention and are intentionally not used here.
    if method in {"reml", "laml"} and bool(
        getattr(model.family, "supports_closed_form_solve", False)
    ):
        return criterion_gradient_numerical(
            model,
            y,
            log_sp,
            method=method,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
        )
    if method in {"ml", "reml", "laml"}:
        backend = resolve_ml_reml_scoring_backend(model, method=method)
        if backend == "gaussian_exact":
            if _model_has_random_effect_smooth(model) and method == "ml":
                return criterion_gradient_numerical(
                    model,
                    y,
                    log_sp,
                    method=method,
                    eps_abs=eps_abs,
                    eps_rel=eps_rel,
                )
        if backend == "gaussian_dynamic" and method in {"reml", "laml"}:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            out = _gaussian_dynamic_reml_derivative_terms(
                model, y, log_sp, exact_method
            )
            if bool(out.get("valid", False)):
                return np.asarray(out["grad"], dtype=np.float64)
        if backend == "general_fit5":
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_gradient_ml_reml_general_fit5(
                model, y, log_sp, exact_method
            )
        if (
            backend == "pirls_laplace"
            and (
                getattr(model.family, "known_scale", None) is not None
                or str(getattr(model.family, "name", "")).lower() == "gamma"
            )
            and bool(
                getattr(model.family, "supports_exact_pirls_first_derivatives", False)
            )
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_gradient_ml_reml_pirls_exact(
                model, y, log_sp, exact_method
            )
    return criterion_gradient_numerical(
        model,
        y,
        log_sp,
        method=method,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
    )


def criterion_hessian_numerical(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-4,
    eps_rel=1e-3,
    _baseline_state=None,
):
    """Centered finite-difference Hessian of the smoothing criterion."""
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    baseline_state = _baseline_state
    if (
        baseline_state is None
        and _is_joint_negbin_theta_model(model)
        and method in {"ml", "reml", "laml"}
    ):
        baseline_state = _current_joint_negbin_eval_state(model)

    H = np.empty((n, n), dtype=np.float64)
    steps = np.maximum(float(eps_abs), float(eps_rel) * (1.0 + np.abs(x)))

    for j in range(n):
        h = float(steps[j])
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h

        if baseline_state is not None:
            g_plus = criterion_gradient_numerical(
                model,
                y,
                x_plus,
                method=method,
                eps_abs=max(eps_abs * 0.1, 1e-6),
                eps_rel=max(eps_rel * 0.1, 1e-5),
                _baseline_state=baseline_state,
            )
            g_minus = criterion_gradient_numerical(
                model,
                y,
                x_minus,
                method=method,
                eps_abs=max(eps_abs * 0.1, 1e-6),
                eps_rel=max(eps_rel * 0.1, 1e-5),
                _baseline_state=baseline_state,
            )
        else:
            g_plus = criterion_gradient(
                model,
                y,
                x_plus,
                method=method,
                eps_abs=max(eps_abs * 0.1, 1e-6),
                eps_rel=max(eps_rel * 0.1, 1e-5),
            )
            g_minus = criterion_gradient(
                model,
                y,
                x_minus,
                method=method,
                eps_abs=max(eps_abs * 0.1, 1e-6),
                eps_rel=max(eps_rel * 0.1, 1e-5),
            )
        H[:, j] = (g_plus - g_minus) / (2.0 * h)

    return 0.5 * (H + H.T)


def criterion_hessian(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-4,
    eps_rel=1e-3,
):
    method = str(method).lower()
    if method in {"reml", "laml"} and bool(
        getattr(model.family, "supports_closed_form_solve", False)
    ):
        return criterion_hessian_numerical(
            model,
            y,
            log_sp,
            method=method,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
        )
    if method in {"ml", "reml", "laml"}:
        backend = resolve_ml_reml_scoring_backend(model, method=method)
        if (
            backend == "pirls_laplace"
            and method in {"reml", "laml"}
            and (
                getattr(model.family, "known_scale", None) is not None
                or str(getattr(model.family, "name", "")).lower() == "gamma"
            )
            and bool(
                getattr(model.family, "supports_exact_pirls_second_derivatives", False)
            )
        ):
            return criterion_hessian_ml_reml_pirls_exact(
                model, y, log_sp, "REML"
            )
        if backend == "gaussian_dynamic" and method in {"reml", "laml"}:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            out = _gaussian_dynamic_reml_derivative_terms(
                model, y, log_sp, exact_method
            )
            if bool(out.get("valid", False)):
                return np.asarray(out["hess"], dtype=np.float64)
        if backend == "general_fit5":
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_hessian_ml_reml_general_fit5(
                model, y, log_sp, exact_method
            )
    return criterion_hessian_numerical(
        model,
        y,
        log_sp,
        method=method,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
    )


def criterion_infinite_sp_signal(model, y, log_sp, method="reml"):
    method = str(method).lower()
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    grad = np.asarray(criterion_gradient(model, y, x, method=method), dtype=np.float64)
    hess = np.asarray(criterion_hessian(model, y, x, method=method), dtype=np.float64)
    if hess.ndim != 2 or hess.shape[0] != hess.shape[1] or hess.shape[0] != n:
        dvkk = np.full(n, np.nan, dtype=np.float64)
    else:
        dvkk = np.diag(hess).astype(np.float64, copy=True)
    return grad, dvkk

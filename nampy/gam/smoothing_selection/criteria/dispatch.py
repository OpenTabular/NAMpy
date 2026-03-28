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

from .gaussian import criterion_gcv_gaussian
from .gaussian_dyn import _gaussian_dynamic_reml_derivative_terms
from .gaussian_grad import criterion_gradient_ml_reml_exact
from .ml_reml import (
    _model_has_random_effect_smooth,
    criterion_ml_reml,
    resolve_ml_reml_scoring_backend,
)
from .pirls import criterion_gcv_pirls, criterion_ubre_pirls
from .pirls_deriv import (
    criterion_gradient_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
)
from .laplace import _penalty_derivative_matrices

def criterion_value(model, y, log_sp, method="gcv"):
    method = str(method).lower()
    if method == "gcv":
        if model._uses_closed_form_solver():
            return criterion_gcv_gaussian(model, y, log_sp)
        return criterion_gcv_pirls(model, y, log_sp)
    if method in {"ubre", "aic", "ubreaic"}:
        return criterion_ubre_pirls(model, y, log_sp)
    if method == "ml":
        return criterion_ml_reml(model, y, log_sp, "ml")
    if method in {"reml", "laml"}:
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
):
    """Centered finite-difference gradient of the smoothing criterion."""
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    if x.size == 0:
        return np.empty((0,), dtype=np.float64)

    grad = np.empty_like(x)
    f0 = float(criterion_value(model, y, x, method=method))

    if not np.isfinite(f0):
        grad.fill(np.nan)
        return grad

    for i in range(x.size):
        step = max(float(eps_abs), float(eps_rel) * (1.0 + abs(float(x[i]))))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += step
        x_minus[i] -= step

        f_plus = float(criterion_value(model, y, x_plus, method=method))
        f_minus = float(criterion_value(model, y, x_minus, method=method))

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
    # Gaussian REML/LAML now uses the deviance-based scale convention that matches
    # mgcv's profiled Wood-style criterion. The older analytic derivative paths in
    # `gaussian_grad.py` / `gaussian_dyn.py` were derived for the previous scale
    # profiling and are no longer reliable for outer optimisation. Keep the scalar
    # criterion exact, but differentiate it numerically until those branches are
    # rederived under the corrected convention.
    if (
        method in {"reml", "laml"}
        and bool(getattr(model.family, "supports_closed_form_solve", False))
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
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            if _model_has_random_effect_smooth(model) and method in {"reml", "laml"}:
                out = _gaussian_dynamic_reml_derivative_terms(
                    model, y, log_sp, exact_method
                )
                if bool(out.get("valid", False)):
                    return np.asarray(out["grad"], dtype=np.float64)
            if _model_has_random_effect_smooth(model) and method == "ml":
                return criterion_gradient_numerical(
                    model,
                    y,
                    log_sp,
                    method=method,
                    eps_abs=eps_abs,
                    eps_rel=eps_rel,
                )
            return criterion_gradient_ml_reml_exact(model, y, log_sp, exact_method)
        if backend == "gaussian_dynamic" and method in {"reml", "laml"}:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            out = _gaussian_dynamic_reml_derivative_terms(model, y, log_sp, exact_method)
            if bool(out.get("valid", False)):
                return np.asarray(out["grad"], dtype=np.float64)
        if (
            backend == "pirls_laplace"
            and (
                getattr(model.family, "known_scale", None) is not None
                or str(getattr(model.family, "name", "")).lower() == "gamma"
            )
            and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_gradient_ml_reml_pirls_exact(model, y, log_sp, exact_method)
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
):
    """Centered finite-difference Hessian of the smoothing criterion."""
    x = np.asarray(log_sp, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    H = np.empty((n, n), dtype=np.float64)
    steps = np.maximum(float(eps_abs), float(eps_rel) * (1.0 + np.abs(x)))

    for j in range(n):
        h = float(steps[j])
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h

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
    if (
        method in {"reml", "laml"}
        and bool(getattr(model.family, "supports_closed_form_solve", False))
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
            and bool(getattr(model.family, "supports_exact_pirls_second_derivatives", False))
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_hessian_ml_reml_pirls_exact(model, y, log_sp, exact_method)
        if (
            backend == "gaussian_exact"
            and _model_has_random_effect_smooth(model)
            and method in {"reml", "laml"}
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            out = _gaussian_dynamic_reml_derivative_terms(
                model, y, log_sp, exact_method
            )
            if bool(out.get("valid", False)):
                return np.asarray(out["hess"], dtype=np.float64)
        if backend == "gaussian_dynamic" and method in {"reml", "laml"}:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            out = _gaussian_dynamic_reml_derivative_terms(model, y, log_sp, exact_method)
            if bool(out.get("valid", False)):
                return np.asarray(out["hess"], dtype=np.float64)
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

    backend = None
    if method in {"ml", "reml", "laml"}:
        backend = resolve_ml_reml_scoring_backend(model, method=method)

    exact_method = "REML" if method in {"reml", "laml"} else "ML"

    if (
        backend == "pirls_laplace"
        and (
            getattr(model.family, "known_scale", None) is not None
            or str(getattr(model.family, "name", "")).lower() == "gamma"
        )
        and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
        and model._can_use_simple_ml_reml_structure()
    ):
        sp = model._expand_smoothing_params_from_log(x)
        sol = model._solve_pirls_given_smoothing(y, sp)
        beta = np.asarray(sol["coef_full"], dtype=np.float64)
        A = np.asarray(sol["A"], dtype=np.float64)
        A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
        P_derivs = _penalty_derivative_matrices(model, sp)

        grad = np.asarray(
            criterion_gradient_ml_reml_pirls_exact(model, y, x, exact_method),
            dtype=np.float64,
        )
        dvkk = np.zeros(int(model.n_smoothing_params_ or 0), dtype=np.float64)
        for j, Pj in enumerate(P_derivs):
            if not np.any(Pj):
                continue
            dbeta_j = -(A_inv @ (Pj @ beta))
            dvkk[j] = float(dbeta_j @ (A @ dbeta_j))

        free_mask = (
            np.zeros(model.n_smoothing_params_, dtype=bool)
            if model.smoothing_fixed_mask_ is None
            else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
        )
        free_mask = ~free_mask
        return grad, dvkk[free_mask]

    if backend == "gaussian_exact" and model._can_use_exact_gaussian_ml_reml():
        sp = model._expand_smoothing_params_from_log(x)
        sol = model._solve_gaussian_given_smoothing(y, sp)
        beta = np.asarray(sol["coef_full"], dtype=np.float64)
        A = np.asarray(sol["A"], dtype=np.float64)
        A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
        P_derivs = _penalty_derivative_matrices(model, sp)

        grad = np.asarray(
            criterion_gradient_ml_reml_exact(model, y, x, exact_method),
            dtype=np.float64,
        )
        dvkk = np.zeros(int(model.n_smoothing_params_ or 0), dtype=np.float64)
        for j, Pj in enumerate(P_derivs):
            if not np.any(Pj):
                continue
            dbeta_j = -(A_inv @ (Pj @ beta))
            dvkk[j] = float(dbeta_j @ (A @ dbeta_j))

        free_mask = (
            np.zeros(model.n_smoothing_params_, dtype=bool)
            if model.smoothing_fixed_mask_ is None
            else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
        )
        free_mask = ~free_mask
        return grad, dvkk[free_mask]

    grad = np.asarray(criterion_gradient(model, y, x, method=method), dtype=np.float64)
    hess = np.asarray(criterion_hessian(model, y, x, method=method), dtype=np.float64)
    if hess.ndim != 2 or hess.shape[0] != hess.shape[1] or hess.shape[0] != n:
        dvkk = np.full(n, np.nan, dtype=np.float64)
    else:
        dvkk = np.diag(hess).astype(np.float64, copy=True)
    return grad, dvkk

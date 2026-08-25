"""
Top-level dispatch for smoothing-selection criterion value, gradient, and Hessian.

:func:`criterion_value` — scalar criterion at a given log-smoothing-parameter vector.
:func:`criterion_gradient` — exact gradient w.r.t. log smoothing parameters when
    the upstream-mirrored derivative path exists.
:func:`criterion_hessian` — exact Hessian when the upstream-mirrored derivative
    path exists.
"""

import numpy as np

from ....linalg import symmetrize_matrix
from ...backends import GENERAL_FAMILY_BACKEND
from ...capabilities import uses_closed_form_solver
from ...solvers.general_family.fixed_smoothing import (
    criterion_gradient_ml_reml_general_family,
    criterion_hessian_ml_reml_general_family,
)
from .gaussian import criterion_gcv_gaussian
from .gaussian_dyn import _gaussian_dynamic_reml_derivative_terms
from .ml_reml import (
    criterion_ml_reml,
    resolve_ml_reml_scoring_backend,
)
from .ncv import criterion_gradient_ncv_pirls, criterion_ncv_pirls
from .pirls import (
    criterion_gacv_pirls,
    criterion_gcv_pirls,
    criterion_gradient_gcv_ubre_pirls_exact,
    criterion_gradient_p_ml_reml_pirls_exact,
    criterion_hessian_gcv_ubre_pirls_exact,
    criterion_hessian_p_ml_reml_pirls_exact,
    criterion_p_ml_reml_pirls,
    criterion_ubre_pirls,
)
from .pirls.derivatives import (
    criterion_gradient_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
)
from .pirls.family_betar import (
    criterion_ml_reml_pirls_frozen_betar,
    is_joint_betar_theta_model,
)
from .pirls.family_negbin import (
    criterion_ml_reml_pirls_frozen_negbin,
    current_joint_negbin_eval_state,
    is_joint_negbin_theta_model,
)
from .pirls.family_ocat import (
    criterion_ml_reml_pirls_frozen_ocat,
    is_joint_ocat_theta_model,
)
from .pirls.outer_strategies import get_joint_outer_handler
from .shape import (
    criterion_gradient_transformed,
    criterion_value_transformed,
    is_transformed_coefficient_model,
)


def _normalize_criterion_method(model, method):
    method = str(method).strip().lower().replace("_", "-")
    aliases = {"p.ml": "p-ml", "p.reml": "p-reml"}
    method = aliases.get(method, method)
    if (
        method in {"p-ml", "p-reml"}
        and getattr(model.family, "known_scale", None) is not None
    ):
        return "ml" if method == "p-ml" else "reml"
    if method not in {"gcv.cp", "gacv.cp"}:
        return method

    family = getattr(model, "family", None)
    family_class = str(getattr(family, "family_class", "")).lower()

    if family_class in {"extended", "general"}:
        return "reml"
    if getattr(family, "known_scale", None) is not None:
        return "aic"
    return "gacv" if method == "gacv.cp" else "gcv"


def criterion_value(model, y, log_sp, method="gcv"):
    method = _normalize_criterion_method(model, method)
    if is_transformed_coefficient_model(model) and method in {
        "gcv",
        "ubre",
        "aic",
        "ubreaic",
    }:
        return criterion_value_transformed(model, y, log_sp, method=method)
    if method == "gcv":
        if uses_closed_form_solver(model):
            return criterion_gcv_gaussian(model, y, log_sp)
        return criterion_gcv_pirls(model, y, log_sp)
    if method == "gacv":
        return criterion_gacv_pirls(model, y, log_sp)
    if method in {"ncv", "qncv"}:
        return criterion_ncv_pirls(model, y, log_sp, qapprox=method == "qncv")
    if method in {"ubre", "aic", "ubreaic"}:
        return criterion_ubre_pirls(model, y, log_sp)
    if method == "ml":
        if is_joint_negbin_theta_model(model):
            return criterion_ml_reml_pirls_frozen_negbin(model, y, log_sp, "ML")
        if is_joint_betar_theta_model(model):
            return criterion_ml_reml_pirls_frozen_betar(model, y, log_sp, "ML")
        if is_joint_ocat_theta_model(model):
            return criterion_ml_reml_pirls_frozen_ocat(model, y, log_sp, "ML")
        handler = get_joint_outer_handler(model.family)
        if handler is not None:
            return handler.value(model, y, log_sp, "ML")
        return criterion_ml_reml(model, y, log_sp, "ml")
    if method in {"reml", "laml"}:
        if is_joint_negbin_theta_model(model):
            return criterion_ml_reml_pirls_frozen_negbin(model, y, log_sp, "REML")
        if is_joint_betar_theta_model(model):
            return criterion_ml_reml_pirls_frozen_betar(model, y, log_sp, "REML")
        if is_joint_ocat_theta_model(model):
            return criterion_ml_reml_pirls_frozen_ocat(model, y, log_sp, "REML")
        handler = get_joint_outer_handler(model.family)
        if handler is not None:
            return handler.value(model, y, log_sp, "REML")
        return criterion_ml_reml(model, y, log_sp, method)
    if method in {"p-ml", "p-reml"}:
        return criterion_p_ml_reml_pirls(model, y, log_sp, method.upper())
    raise ValueError(
        "method must be one of "
        "{'gcv', 'gacv', 'ubre', 'aic', 'ubreaic', 'ml', 'reml', "
        "'p-ml', 'p-reml', 'ncv', 'qncv', 'laml'}"
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
        and is_joint_negbin_theta_model(model)
        and method in {"ml", "reml", "laml"}
    ):
        baseline_state = current_joint_negbin_eval_state(model)

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
    method = _normalize_criterion_method(model, method)
    if is_transformed_coefficient_model(model) and method in {
        "gcv",
        "ubre",
        "aic",
        "ubreaic",
    }:
        return criterion_gradient_transformed(model, y, log_sp, method=method)
    if method in {"gcv", "gacv", "ubre", "aic", "ubreaic"}:
        return criterion_gradient_gcv_ubre_pirls_exact(model, y, log_sp, method)
    if method in {"p-ml", "p-reml"}:
        return criterion_gradient_p_ml_reml_pirls_exact(model, y, log_sp, method)
    if method in {"ncv", "qncv"}:
        return criterion_gradient_ncv_pirls(model, y, log_sp, qapprox=method == "qncv")
    if method in {"ml", "reml", "laml"}:
        handler = get_joint_outer_handler(model.family)
        if handler is not None:
            return handler.gradient(
                model,
                y,
                log_sp,
                "REML" if method in {"reml", "laml"} else "ML",
            )
        backend = resolve_ml_reml_scoring_backend(model, method=method)
        if backend in {"gaussian_exact", "gaussian_dynamic"}:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            out = _gaussian_dynamic_reml_derivative_terms(
                model, y, log_sp, exact_method
            )
            if bool(out.get("valid", False)):
                return np.asarray(out["grad"], dtype=np.float64)
            raise NotImplementedError(
                "Gaussian ML/REML/LAML outer optimisation requires exact "
                "mgcv-parity derivatives; finite-difference fallback removed."
            )
        if backend == GENERAL_FAMILY_BACKEND:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_gradient_ml_reml_general_family(
                model, y, log_sp, exact_method
            )
        if (
            backend == "pirls_laplace"
            and (
                getattr(model.family, "known_scale", None) is not None
                or str(getattr(model.family, "name", "")).lower()
                in {"gamma", "gaussian"}
            )
            and bool(
                getattr(model.family, "supports_exact_pirls_first_derivatives", False)
            )
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_gradient_ml_reml_pirls_exact(
                model, y, log_sp, exact_method
            )
        raise NotImplementedError(
            "ML/REML/LAML outer optimisation requires an exact upstream-mirrored "
            "derivative path; numerical fallback removed."
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
        and is_joint_negbin_theta_model(model)
        and method in {"ml", "reml", "laml"}
    ):
        baseline_state = current_joint_negbin_eval_state(model)

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

    return symmetrize_matrix(H)


def criterion_hessian(
    model,
    y,
    log_sp,
    method="gcv",
    eps_abs=1e-4,
    eps_rel=1e-3,
):
    method = _normalize_criterion_method(model, method)
    if method in {"gcv", "gacv", "ubre", "aic", "ubreaic"}:
        return criterion_hessian_gcv_ubre_pirls_exact(model, y, log_sp, method)
    if method in {"p-ml", "p-reml"}:
        return criterion_hessian_p_ml_reml_pirls_exact(model, y, log_sp, method)
    if method in {"ml", "reml", "laml"}:
        handler = get_joint_outer_handler(model.family)
        if handler is not None:
            return handler.hessian(
                model,
                y,
                log_sp,
                "REML" if method in {"reml", "laml"} else "ML",
            )
        backend = resolve_ml_reml_scoring_backend(model, method=method)
        if (
            backend == "pirls_laplace"
            and method in {"ml", "reml", "laml"}
            and (
                getattr(model.family, "known_scale", None) is not None
                or str(getattr(model.family, "name", "")).lower()
                in {"gamma", "gaussian"}
            )
            and bool(
                getattr(model.family, "supports_exact_pirls_second_derivatives", False)
            )
        ):
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_hessian_ml_reml_pirls_exact(model, y, log_sp, exact_method)
        if backend in {"gaussian_exact", "gaussian_dynamic"}:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            out = _gaussian_dynamic_reml_derivative_terms(
                model, y, log_sp, exact_method
            )
            if bool(out.get("valid", False)):
                return np.asarray(out["hess"], dtype=np.float64)
            raise NotImplementedError(
                "Gaussian ML/REML/LAML outer optimisation requires exact "
                "mgcv-parity Hessians; finite-difference fallback removed."
            )
        if backend == GENERAL_FAMILY_BACKEND:
            exact_method = "REML" if method in {"reml", "laml"} else "ML"
            return criterion_hessian_ml_reml_general_family(
                model, y, log_sp, exact_method
            )
        raise NotImplementedError(
            "ML/REML/LAML outer optimisation requires an exact upstream-mirrored "
            "Hessian path; numerical fallback removed."
        )
    return criterion_hessian_numerical(
        model,
        y,
        log_sp,
        method=method,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
    )

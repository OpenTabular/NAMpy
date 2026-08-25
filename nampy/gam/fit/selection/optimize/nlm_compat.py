"""Local ``stats::nlm``-compatible outer-optimization adapter.

mgcv and scam supply only a value plus, for the analytic route, a gradient to
R's ``nlm``.  This module preserves that public distinction while using
SciPy's unbounded variable-metric implementation as the local numerical core.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult, dual_annealing, minimize


_NLM_MESSAGES = {
    1: "Relative gradient is close to zero, current iterate is probably solution",
    2: "Successive iterates within tolerance, current iterate is probably solution",
    3: "Last global step failed to locate a lower point",
    4: "Iteration limit exceeded",
    5: "Maximum step size exceeded repeatedly",
}


def optimize_outer_nlm(objective, x0, *, control, finite_difference=False):
    """Run the analytic ``nlm`` or scam-only ``nlm.fd`` route."""
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    cfg = dict(control or {})
    gradtol = float(cfg.get("gradtol", 1e-6))
    steptol = float(cfg.get("steptol", 1e-4))
    iterlim = int(cfg.get("iterlim", 200))
    ndigit = int(cfg.get("ndigit", 7))
    # R's ndigit controls the assumed objective precision.  Map it to the
    # line-search tolerance without changing the unbounded log-SP geometry.
    ftol = max(10.0 ** (-max(1, min(ndigit, 15))), np.finfo(float).eps)

    invalid_value = float(np.finfo(np.float64).max / 1e100)

    def safe_fun(x):
        try:
            value = float(objective.fun(x))
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return invalid_value
        return value if np.isfinite(value) else invalid_value

    def safe_jac(x):
        try:
            value = np.asarray(objective.jac(x), dtype=np.float64)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.zeros_like(np.asarray(x, dtype=np.float64))
        return value if np.all(np.isfinite(value)) else np.zeros_like(value)

    result = minimize(
        safe_fun,
        x0,
        method="BFGS",
        jac=None if finite_difference else safe_jac,
        options={
            "gtol": gradtol,
            "maxiter": iterlim,
            "xrtol": steptol,
            "finite_diff_rel_step": np.sqrt(ftol),
        },
    )
    grad = np.asarray(
        result.jac if getattr(result, "jac", None) is not None else np.full(x0.size, np.nan),
        dtype=np.float64,
    )
    if bool(result.success):
        code = 1 if np.linalg.norm(grad, ord=np.inf) <= gradtol else 2
    elif int(getattr(result, "nit", 0)) >= iterlim:
        code = 4
    else:
        code = 3
    out = OptimizeResult(result)
    out.code = int(code)
    out.status = int(code)
    out.message = _NLM_MESSAGES[code]
    out.minimum = float(out.fun)
    out.estimate = np.asarray(out.x, dtype=np.float64).copy()
    out.gradient = grad.copy()
    out.iterations = int(getattr(out, "nit", 0))
    out.nlm_finite_difference = bool(finite_difference)
    out.outer_info = {
        "optimizer": "nlm.fd" if finite_difference else "nlm",
        "termcode": int(code),
        "conv": str(out.message),
        "iterations": int(out.iterations),
        "gradient": grad.copy(),
        "gradient_full": grad.copy(),
    }
    return out


__all__ = ["optimize_outer_nlm"]


def optimize_shape_optim(objective, x0, *, optim_method=None, factr=1e7, bounds=None):
    """SCAM's selectable ``optim(method, fd|grad)`` route."""
    if optim_method is None:
        method_name, derivative = "Nelder-Mead", "fd"
    elif isinstance(optim_method, str):
        method_name, derivative = optim_method, "fd"
    else:
        values = list(optim_method)
        method_name = str(values[0]) if values else "Nelder-Mead"
        derivative = str(values[1]).lower() if len(values) > 1 else "fd"
    allowed = {"Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN"}
    if method_name not in allowed:
        method_name = "L-BFGS-B"
    use_gradient = derivative == "grad" and method_name not in {"Nelder-Mead", "SANN"}
    x0 = np.asarray(x0, dtype=np.float64).ravel()

    def safe_fun(x):
        try:
            value = float(objective.fun(x))
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return float(np.finfo(np.float64).max / 1e100)
        return value if np.isfinite(value) else float(np.finfo(np.float64).max / 1e100)

    def safe_jac(x):
        try:
            value = np.asarray(objective.jac(x), dtype=np.float64)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.zeros_like(np.asarray(x, dtype=np.float64))
        return value if np.all(np.isfinite(value)) else np.zeros_like(value)

    if method_name == "SANN":
        finite_bounds = (
            [(-80.0, 20.0)] * x0.size
            if bounds is None
            else [
                (
                    -80.0 if not np.isfinite(lo) else float(lo),
                    20.0 if not np.isfinite(hi) else float(hi),
                )
                for lo, hi in bounds
            ]
        )
        result = dual_annealing(safe_fun, bounds=finite_bounds, x0=x0)
    else:
        options = {}
        scipy_bounds = None
        if method_name == "L-BFGS-B":
            scipy_bounds = bounds
            options = {
                "ftol": float(np.finfo(float).eps * abs(float(factr))),
                "maxcor": min(5, max(1, x0.size)),
            }
        result = minimize(
            safe_fun,
            x0,
            method=method_name,
            jac=safe_jac if use_gradient else None,
            bounds=scipy_bounds,
            options=options,
        )
    out = OptimizeResult(result)
    out.scam_optim_method = (method_name, "grad" if use_gradient else "fd")
    out.outer_info = {
        "optimizer": "optim",
        "method": method_name,
        "derivative": "grad" if use_gradient else "fd",
        "termcode": int(getattr(out, "status", 0)),
        "conv": str(getattr(out, "message", "")),
        "iterations": int(getattr(out, "nit", 0)),
    }
    return out


__all__.append("optimize_shape_optim")

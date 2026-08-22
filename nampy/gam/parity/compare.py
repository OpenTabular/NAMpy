"""
Numerical comparison of parity snapshots against reference outputs.

:func:`compare_parity_snapshots` compares a newly captured snapshot to a
saved reference (typically produced by the R mgcv package) and returns a
structured report of which quantities agree within tolerance.
"""


import numpy as np


def _as_float_array(x):
    if x is None:
        return None
    return x


def _max_abs_diff(a, b):
    a = _as_float_array(a)
    b = _as_float_array(b)
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return np.inf
    if a.shape != b.shape:
        return np.inf
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def _allclose(a, b, atol, rtol):
    a = _as_float_array(a)
    b = _as_float_array(b)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))


def compare_parity_snapshots(actual, expected, atol=1e-6, rtol=1e-6):
    a_fit = actual["fit"]
    e_fit = expected["fit"]
    a_pred = actual["predictions"]
    e_pred = expected["predictions"]

    checks = {
        "coef_full": (a_fit["coef_full"], e_fit["coef_full"]),
        "smoothing_params": (a_fit["smoothing_params"], e_fit["smoothing_params"]),
        "edf_by_term": (a_fit["edf_by_term"], e_fit["edf_by_term"]),
        "response": (a_pred["response"], e_pred["response"]),
        "link": (a_pred["link"], e_pred["link"]),
        "terms": (a_pred["terms"], e_pred["terms"]),
        "lpmatrix": (a_pred["lpmatrix"], e_pred["lpmatrix"]),
    }

    if "cov_bayes" in a_fit or "cov_bayes" in e_fit:
        checks["cov_bayes"] = (
            a_fit.get("cov_bayes", None),
            e_fit.get("cov_bayes", None),
        )
    if "cov_freq" in a_fit or "cov_freq" in e_fit:
        checks["cov_freq"] = (a_fit.get("cov_freq", None), e_fit.get("cov_freq", None))

    report = {
        "scalar_checks": {},
        "array_checks": {},
        "passed": True,
    }

    scalar_pairs = {
        "intercept": (a_fit["intercept"], e_fit["intercept"]),
        "edf_total": (a_fit["edf_total"], e_fit["edf_total"]),
        "trace_H": (a_fit["trace_H"], e_fit["trace_H"]),
        "scale": (a_fit["scale"], e_fit["scale"]),
        "rss": (a_fit["rss"], e_fit["rss"]),
        "deviance": (a_fit["deviance"], e_fit["deviance"]),
    }

    for name, (a, e) in scalar_pairs.items():
        ok = (a is None and e is None) or np.isclose(
            float(a), float(e), atol=atol, rtol=rtol
        )
        report["scalar_checks"][name] = {
            "actual": a,
            "expected": e,
            "ok": bool(ok),
            "abs_diff": (
                None if (a is None or e is None) else float(abs(float(a) - float(e)))
            ),
        }
        report["passed"] = report["passed"] and bool(ok)

    for name, (a, e) in checks.items():
        ok = _allclose(a, e, atol=atol, rtol=rtol)
        report["array_checks"][name] = {
            "ok": bool(ok),
            "shape_actual": None if a is None else list(a.shape),
            "shape_expected": None if e is None else list(e.shape),
            "max_abs_diff": _max_abs_diff(a, e),
        }
        report["passed"] = report["passed"] and bool(ok)

    return report

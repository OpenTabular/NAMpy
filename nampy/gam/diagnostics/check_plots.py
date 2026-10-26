"""Numerical panels for mgcv::gam.check, separate from its fit report."""

from __future__ import annotations

import numpy as np

from ..model_state import _fitted_eta, _fitted_mu, _require_fitted
from .qq import _normal_qq, prepare_qq_gam_data
from .residuals import residuals_gam


def _pretty_histogram_breaks(lo, hi, n):
    """R pretty.default defaults, min.n=1; R src/appl/pretty.c.

    Source: https://github.com/wch/r-source/blob/trunk/src/appl/pretty.c
    """
    cell = max(abs(lo), abs(hi))
    small = hi - lo < cell * 1.4 * max(1, n) * np.finfo(float).eps * 3
    if hi == lo == 0:
        cell, small = 1.0, True
    if small:
        if cell > 10:
            cell = 9 + cell / 10
        cell *= 0.75
    else:
        cell = hi / n - lo / n if not np.isfinite(hi - lo) else (hi - lo) / n
    cell = np.clip(cell, 2**-20 * np.finfo(float).tiny, np.finfo(float).max / 1.25)
    base = 10.0 ** np.floor(np.log10(cell))
    unit = base
    if 2 * base - cell < 1.5 * (cell - unit):
        unit = 2 * base
        if 5 * base - cell < 2.75 * (cell - unit):
            unit = 5 * base
            if 10 * base - cell < 1.5 * (cell - unit):
                unit = 10 * base
    ns, nu = np.floor(lo / unit + 1e-10), np.ceil(hi / unit - 1e-10)
    while ns * unit > lo + 1e-10 * unit:
        ns -= 1
    while nu * unit < hi - 1e-10 * unit:
        nu += 1
    if nu == ns:
        if ns >= 0:
            ns -= 1
        else:
            nu += 1
    count = int(0.5 + nu - ns)
    breaks = np.linspace(min(lo, ns * unit), max(hi, nu * unit), count + 1)
    breaks[np.abs(breaks) < 1e-14 * (breaks[-1] / count - breaks[0] / count)] = 0
    return breaks


def _r_histogram(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Histogram requires finite residuals")
    breaks = _pretty_histogram_breaks(
        finite.min(), finite.max(), int(np.ceil(np.log2(len(finite)) + 1))
    )
    widths = np.diff(breaks)
    diddle = 1e-7 * (
        np.median(widths)
        if len(breaks) > 5
        else np.ptp(finite)
        if len(breaks) <= 3
        else np.min(widths[widths > 0])
    )
    fuzzy = breaks + diddle
    fuzzy[0] = breaks[0] - diddle
    index = np.searchsorted(fuzzy, finite, side="left") - 1
    index[finite == fuzzy[0]] = 0
    counts = np.bincount(index, minlength=len(widths))
    return {
        "values": values,
        "breaks": breaks,
        "counts": counts,
        "density": counts / (len(finite) * widths),
        "mids": (breaks[1:] + breaks[:-1]) / 2,
        "xlab": "Residuals",
        "title": "Histogram of residuals",
    }


def prepare_gam_check_data(
    model,
    *,
    old_style=False,
    type="deviance",
    rep=0,
    level=0.9,
    s_rep=10,
    seed=None,
    quantile_permutations=None,
    response_draws=None,
):
    """Prepare the four gam.check plots without rerunning k.check or fitting."""
    _require_fitted(model)
    if type not in {"deviance", "pearson", "response"}:
        raise ValueError("type must be deviance, pearson or response")
    residuals = residuals_gam(model, type=type)
    qq = (
        _normal_qq(residuals, type)
        if old_style
        else prepare_qq_gam_data(
            model,
            rep=rep,
            level=level,
            s_rep=s_rep,
            type=type,
            seed=seed,
            quantile_permutations=quantile_permutations,
            response_draws=response_draws,
        )
    )
    eta = np.asarray(_fitted_eta(model))
    if eta.ndim == 2 and residuals.ndim == 1:
        eta = eta[:, 0]
    fitted = np.asarray(_fitted_mu(model))
    # Extended families may transform their response prediction (e.g. ocat).
    from ..families.family_base import ExtendedFamily

    if isinstance(model.family, ExtendedFamily):
        fitted = np.asarray(model.predict(type="response"))
    y = np.asarray(model.y_)
    if fitted.ndim == 2 and y.ndim == 1:
        fitted = fitted[:, 0]
    return {
        "qq": qq,
        "residual_vs_linear": {
            "x": eta,
            "y": residuals,
            "xlab": "linear predictor",
            "ylab": "residuals",
            "title": "Resids vs. linear pred.",
        },
        "histogram": _r_histogram(residuals),
        "response_vs_fitted": {
            "x": fitted,
            "y": y,
            "xlab": "Fitted Values",
            "ylab": "Response",
            "title": "Response vs. Fitted Values",
        },
        "type": type,
        "old_style": bool(old_style),
    }

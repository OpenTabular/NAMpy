"""
Gaussian REML / ML Laplace score algebra (Wood-style penalized likelihood).

Building blocks for the concentrated and joint Gaussian REML objectives:

- **Weighted deviance** for identity Gaussian: ``sum_i w_i (y_i - μ_i)²``.
- **Quadratic penalty** ``β' S β`` from the current penalized fit.
- **Scale estimates** from deviance or Pearson χ² with effective residual degrees
  ``n_effective - tr(A)``.
- **Saturated Gaussian log-likelihood fragment** in ``σ²`` (value and first two
  derivatives) used inside the Laplace REML score.
- **Joint (sp, log σ²) objective** uses an effective count ``ν`` and a
  ``sum(log w)`` correction when prior weights are not identically one.

The joint outer loop matches standard GAM software that optimises ``log σ²``
alongside log smoothing parameters; the profiled REML variance is
``(deviance + penalty) / (n_row - Mp)`` for the derivative formulas in
``_gaussian_dynamic_reml_derivative_terms``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def prior_weights_diagonal_from_fit(sol: Any, n: int) -> np.ndarray:
    """
    Prior / IRLS diagonal weights to use in Gaussian deviance and REML saturation terms.

    For a penalized fit, this should be the same positive weights that entered
    ``X' W X + S`` (Gaussian identity: working weights equal prior weights).
    Gaussian solves store this on ``working_weights``; P-IRLS stores the final ``W``.

    Pass the same ``sol`` that supplied ``A`` / ``coef_full`` so weights match the
    normal equations when non-unit weights are used.
    """
    n = int(n)
    if sol is None:
        return np.ones(n, dtype=np.float64)
    if isinstance(sol, Mapping):
        w = sol.get("working_weights")
    else:
        w = getattr(sol, "working_weights", None)
    if w is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size != n or not np.all(np.isfinite(w)):
        return np.ones(n, dtype=np.float64)
    return w


def gaussian_weighted_residual_sum_squares(
    y: np.ndarray,
    mu: np.ndarray,
    prior_weights: np.ndarray | None = None,
) -> float:
    """Weighted Gaussian ``deviance`` sum: ``sum_i w_i (y_i - μ_i)²``."""
    y = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape.")
    if prior_weights is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = np.asarray(prior_weights, dtype=np.float64).ravel()
        if w.shape != y.shape:
            raise ValueError("prior_weights must match y.")
    return float(np.sum(w * (y - mu) ** 2))


def quadratic_form_penalty(beta: np.ndarray, penalty_matrix: np.ndarray) -> float:
    """
    Scalar smoothness penalty ``β' S β``.

    ``penalty_matrix`` is the total penalty ``S`` at the current smoothing
    parameters (same as ``solve_gaussian_fit`` ``P_full``).
    """
    b = np.asarray(beta, dtype=np.float64).ravel()
    P = np.asarray(penalty_matrix, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("penalty_matrix must be a square matrix.")
    if b.size != P.shape[0]:
        raise ValueError("beta length must match penalty_matrix dimension.")
    return float(b @ (P @ b))


def deviance_method_scale_estimate(
    dev: float,
    tr_a: float,
    n_effective_rows: float,
    *,
    dev_extra: float = 0.0,
) -> float:
    """
    Deviance-based scale ``(dev + dev_extra) / (n_effective_rows - tr(A))``.

    ``tr_a`` is the hat-matrix trace (total EDF) from the penalized fit.
    """
    den = float(n_effective_rows) - float(tr_a)
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    num = float(dev) + float(dev_extra)
    if not np.isfinite(num) or num < 0.0:
        return float("nan")
    return float(num / den)


def pearson_method_scale_estimate(
    pearson_chi2: float,
    tr_a: float,
    n_effective_rows: float,
    *,
    pearson_extra: float = 0.0,
    fletcher: bool = False,
    y: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    dvar_over_var: np.ndarray | None = None,
) -> float:
    """
    Pearson (and optional Fletcher) scale ``(χ²_pearson + extra) / (n_eff - tr(A))``.

    If ``fletcher`` is True, pass ``y``, ``mu``, and ``dvar_over_var`` with
    ``dvar_over_var[i] = V'(μ_i)/V(μ_i)``.
    """
    den = float(n_effective_rows) - float(tr_a)
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    scale = (float(pearson_chi2) + float(pearson_extra)) / den
    if not fletcher:
        return float(scale)
    if y is None or mu is None or dvar_over_var is None:
        raise ValueError("Fletcher correction requires y, mu, and dvar_over_var.")
    y = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    dv = np.asarray(dvar_over_var, dtype=np.float64).ravel()
    if not (y.shape == mu.shape == dv.shape):
        raise ValueError("y, mu, dvar_over_var must have the same shape.")
    resid = y - mu
    s_bar = float(np.mean(dv * resid))
    s_bar = max(-0.9, s_bar)
    if not np.isfinite(s_bar):
        return float(scale)
    return float(scale / (1.0 + s_bar))


def profiled_gaussian_reml_variance(dev: float, penalty_P: float, n_row: float, mp: float) -> float:
    """
    Profiled Gaussian REML error variance ``(dev + P) / (n_row - Mp)`` for the
    concentrated Laplace REML score (matches ``_gaussian_dynamic_reml_derivative_terms``).
    """
    den = float(n_row) - float(mp)
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    num = float(dev) + float(penalty_P)
    if not np.isfinite(num) or num <= 0.0:
        return float("nan")
    return float(num / den)


def gaussian_reml_weighted_degrees_and_log_weight_term(
    weights: np.ndarray,
    n_row: float,
    mp: float,
    *,
    n_effective_total: float | None = None,
) -> tuple[float, float]:
    """
    Effective ``ν`` and scaled ``sum(log w)`` for the joint Gaussian REML objective.

    For prior weights ``w`` with ``n_w = sum(w > 0)`` and optional effective row
    count ``n_effective_total`` (defaults to ``n_row``), the doubled joint score uses::

        ν = (n_eff / n_row) * n_w - Mp
        correction = (n_eff / n_row) * sum(log w[w>0])

    Returns ``(ν, correction)`` for ``criterion_ml_reml_gaussian_dynamic_joint``.
    Set ``model.n_true_`` to supply ``n_effective_total`` when replicating software
    that replaces ``nrow`` by a user ``n`` in ML/REML bookkeeping.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    nobs = float(n_row)
    pos = w > 0.0
    n_w = float(np.sum(pos))
    if n_w <= 0.0 or not np.isfinite(nobs) or nobs <= 0.0:
        return float("nan"), float("nan")
    sum_log_w = float(np.sum(np.log(w[pos])))
    if n_effective_total is None or float(n_effective_total) <= 0.0:
        nt = nobs
    else:
        nt = float(n_effective_total)
    fac = nt / nobs
    nu = fac * n_w - float(mp)
    return float(nu), float(fac * sum_log_w)


def gaussian_reml_saturation_terms_wrt_variance(
    weights: np.ndarray, variance: float
) -> np.ndarray:
    """
    Gaussian saturated log-likelihood fragment and derivatives w.r.t. ``σ²``.

    Returns ``array([L, dL/d(σ²), d²L/d(σ²)²])`` for the i.i.d. Gaussian part that
    depends only on ``σ²`` and the prior weights (positive-weight count and
    ``sum(log w)`` on positive weights).
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    variance = float(variance)
    pos = w > 0.0
    n_w = int(np.sum(pos))
    if n_w == 0 or not np.isfinite(variance) or variance <= 0.0:
        return np.array([np.inf, np.nan, np.nan], dtype=np.float64)
    sum_log_w = float(np.sum(np.log(w[pos])))
    ls = -0.5 * n_w * np.log(2.0 * np.pi * variance) + 0.5 * sum_log_w
    d1 = -0.5 * n_w / variance
    d2 = 0.5 * n_w / (variance * variance)
    return np.array([ls, d1, d2], dtype=np.float64)


def gaussian_reml_score_from_saturated_part(
    dev: float,
    penalty_P: float,
    variance: float,
    logdet_penalized_minus_penalty: float,
    mp: float,
    gamma: float,
    reml: bool,
    saturation_loglik: float,
) -> float:
    """
    Laplace REML/ML score given the Gaussian saturation term ``L`` at ``σ²``.

    ``logdet_penalized_minus_penalty`` is ``log|X'WX + S| - log|S|_+`` (difference
    before the outer ``1/2`` factor applied here).
    """
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return float(np.inf)
    score = ((dev + penalty_P) / (2.0 * variance) - saturation_loglik) / gamma + 0.5 * float(
        logdet_penalized_minus_penalty
    )
    if reml:
        score -= float(mp) * (np.log(2.0 * np.pi * variance) / 2.0 - np.log(gamma) / 2.0)
    return float(score)


def gaussian_reml_laplace_score(
    dev: float,
    penalty_P: float,
    variance: float,
    logdet_penalized_minus_penalty: float,
    Mp: float,
    weights: np.ndarray,
    *,
    gamma: float = 1.0,
    reml: bool = True,
) -> float:
    """Full Gaussian REML/ML Laplace score using ``gaussian_reml_saturation_terms_wrt_variance``."""
    ls_vec = gaussian_reml_saturation_terms_wrt_variance(weights, variance)
    return gaussian_reml_score_from_saturated_part(
        dev,
        penalty_P,
        variance,
        logdet_penalized_minus_penalty,
        Mp,
        gamma,
        reml,
        float(ls_vec[0]),
    )

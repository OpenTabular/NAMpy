"""Joint coefficient simulation and finite-grid simultaneous inference."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..model_state import _coef_full, _cov_unconditional, _predictor_full_indices


@dataclass(frozen=True)
class FunctionalInterval:
    """Float arrays (m,) estimate, SE, lower, upper; scalar critical value.

    ``significant`` is a bool (m,) mask where the interval excludes zero.
    Simultaneous coverage refers to the supplied finite evaluation grid only.
    """

    estimate: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    critical_value: float
    significant: np.ndarray


def _distribution(model, covariance):
    model = getattr(model, "gam_", model)
    if not getattr(model, "_fitted", False):
        raise ValueError("A fitted statistical GAM is required.")
    beta = np.asarray(_coef_full(model), dtype=float)
    if covariance not in {"bayes", "freq", "unconditional"}:
        raise ValueError("covariance must be 'bayes', 'freq', or 'unconditional'.")
    V = (
        _cov_unconditional(model)
        if covariance == "unconditional"
        else model._select_cov(covariance)
    )
    if V is None:
        raise ValueError(f"Covariance {covariance!r} is unavailable for this fit.")
    V = np.asarray(V, dtype=float)
    if V.shape != (beta.size, beta.size) or not np.all(np.isfinite(V)):
        raise ValueError("Coefficient covariance must be finite (p,p).")
    tolerance = 100 * beta.size * np.finfo(float).eps * np.max(np.abs(V))
    if np.max(np.abs(V - V.T)) > tolerance:
        raise ValueError("Coefficient covariance must be symmetric.")
    V = (V + V.T) / 2
    values, vectors = np.linalg.eigh(V)
    if values.min() < -tolerance:
        raise ValueError("Coefficient covariance is not positive semidefinite.")
    # The symmetric root is invariant to eigenvector signs and ordering,
    # making shared standard draws reproducible across LAPACK builds.
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    return beta, V, root


def coefficient_draws(
    model, *, n_draws=10000, seed=None, covariance="bayes", standard_draws=None
):
    """Draw approximate joint coefficient posterior samples, float (n_draws,p).

    ``model`` is a fitted GAM or GAM adapter. ``covariance`` is 'bayes'
    (default, conditional on smoothing), 'freq', or 'unconditional'. The last
    option requires an available smoothing-adjusted covariance. ``n_draws``
    is a positive int, default 10000; ``seed`` is an optional NumPy RNG seed.
    ``standard_draws`` optionally supplies finite standard normals (n_draws,p)
    for reproducibility across implementations; it overrides n_draws/seed.
    Samples concern coefficient estimation, not new-response variation.
    """
    beta, _, root = _distribution(model, covariance)
    if standard_draws is None:
        if isinstance(n_draws, bool) or int(n_draws) != n_draws or n_draws < 1:
            raise ValueError("n_draws must be a positive integer.")
        n_draws = int(n_draws)
        Z = np.random.default_rng(seed).standard_normal((n_draws, beta.size))
    else:
        Z = np.asarray(standard_draws, dtype=float)
        if (
            Z.ndim != 2
            or Z.shape[1] != beta.size
            or not Z.shape[0]
            or not np.all(np.isfinite(Z))
        ):
            raise ValueError("standard_draws must be finite (n_draws,p).")
    return beta + Z @ root.T


def linear_functional_interval(
    model,
    matrix,
    *,
    level=0.95,
    interval="simultaneous",
    covariance="bayes",
    n_draws=10000,
    seed=None,
    standard_draws=None,
):
    """Intervals for joint linear functionals, contrasts or integrated effects.

    Parameters
    ----------
    model : fitted GAM or GAM adapter
        Supplies full coefficient vector and joint covariance.
    matrix : array-like of float, shape (m,p)
        One functional per row: prediction basis, a difference of two bases,
        a derivative basis, or weighted integrals of basis rows. Known offsets
        must be added separately; response-scale nonlinear transforms require
        transforming posterior draws rather than applying this linear formula.
    level : float, default 0.95
        Coverage in (0,1).
    interval : {'simultaneous', 'pointwise'}, default 'simultaneous'
        Simultaneous uses a simulated quantile of max absolute standardized
        joint errors, as in gratia; pointwise uses a Normal critical value.
    covariance : str, default 'bayes'
        See coefficient_draws. Full cross-term covariance is retained.
    n_draws : int, default 10000
        Monte Carlo sample count for simultaneous bands.
    seed : int or None, default None
        Reproducible NumPy RNG seed.
    standard_draws : ndarray of float or None, default None
        Optional common standard normal draws (n_draws,p).

    Returns
    -------
    FunctionalInterval
        Estimates, SEs and bounds (m,), scalar critical value and bool (m,)
        significance mask. Zero-variance rows receive degenerate intervals.
    """
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be in (0,1).")
    if interval not in {"simultaneous", "pointwise"}:
        raise ValueError("interval must be 'simultaneous' or 'pointwise'.")
    beta, V, _ = _distribution(model, covariance)
    L = np.asarray(matrix, dtype=float)
    if (
        L.ndim != 2
        or L.shape[1] != beta.size
        or not L.shape[0]
        or not np.all(np.isfinite(L))
    ):
        raise ValueError("matrix must be finite (m,p).")
    mean = L @ beta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", L, V, L), 0))
    active = se > 0
    if interval == "pointwise":
        critical = float(norm.ppf((1 + level) / 2))
    elif not active.any():
        critical = 0.0
    else:
        draws = coefficient_draws(
            model,
            n_draws=n_draws,
            seed=seed,
            covariance=covariance,
            standard_draws=standard_draws,
        )
        deviations = draws - beta
        rows = np.flatnonzero(active)
        maxima = np.zeros(len(draws))
        for start in range(0, len(rows), 1000):
            block = rows[start : start + 1000]
            z = deviations @ L[block].T / se[block]
            maxima = np.maximum(maxima, np.max(np.abs(z), axis=1))
        critical = float(np.quantile(maxima, level, method="median_unbiased"))
    lower, upper = mean - critical * se, mean + critical * se
    return FunctionalInterval(
        mean, se, lower, upper, critical, (lower > 0) | (upper < 0)
    )


def derivative_interval(
    model, X, variable, *, step=1e-5, order=1, predictor=0, **kwargs
):
    """Finite-difference derivative bands for a numeric DataFrame column.

    ``X`` is a pandas DataFrame (n,d), ``variable`` a column name, ``step``
    a positive response-predictor step (default 1e-5), and ``order`` 1 or 2
    (default 1). ``predictor`` is a zero-based int, default 0, selecting a
    distributional predictor. Other keywords pass to linear_functional_interval.
    Returns FunctionalInterval arrays (n,). The derivative is of the linear
    predictor; known offsets are excluded. Evaluations must lie in the model's
    valid predictor domain. Second differences may need a larger step.
    """
    if not np.isfinite(step) or step <= 0 or order not in (1, 2):
        raise ValueError("step must be positive and order must be 1 or 2.")
    plus, minus = X.copy(), X.copy()
    plus[variable] = plus[variable] + step
    minus[variable] = minus[variable] - step
    L = (model.lpmatrix(plus) - model.lpmatrix(minus)) / (2 * step)
    if order == 2:
        L = (
            model.lpmatrix(plus) - 2 * model.lpmatrix(X) + model.lpmatrix(minus)
        ) / step**2
    indices = _predictor_full_indices(getattr(model, "gam_", model))
    if (
        isinstance(predictor, bool)
        or not isinstance(predictor, (int, np.integer))
        or not 0 <= predictor < len(indices)
    ):
        raise ValueError("predictor must select an available linear predictor.")
    keep = np.zeros(L.shape[1], dtype=bool)
    keep[indices[predictor]] = True
    L[:, ~keep] = 0
    return linear_functional_interval(model, L, **kwargs)


def joint_edf(information, penalty):
    """Return coefficient EDF (p,), total EDF float and testing EDF (p,).

    Both inputs are finite symmetric float (p,p) matrices in the same
    identifiable, unconstrained coordinates; penalty must be PSD and
    information+penalty must be positive definite. Information may include
    cross-parameter curvature. Computes F=solve(information+S,information), edf=diag(F),
    edf1=diag(2F-F@F). Conditional on fixed smoothing; no edf2 correction.
    Singular, constrained or nonlinear coefficient maps require a separate
    treatment and are rejected rather than silently pseudoinverted.
    """
    information, S = (
        np.asarray(information, dtype=float),
        np.asarray(penalty, dtype=float),
    )
    if (
        information.ndim != 2
        or information.shape[0] != information.shape[1]
        or S.shape != information.shape
        or not information.size
    ):
        raise ValueError(
            "information and penalty must be nonempty square matrices of equal shape."
        )
    for A in (information, S):
        if not np.all(np.isfinite(A)) or not np.allclose(
            A, A.T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("information and penalty must be finite and symmetric.")
    if np.linalg.eigvalsh(S).min() < -1e-10 * max(1, np.linalg.norm(S, 2)):
        raise ValueError("penalty must be positive semidefinite.")
    np.linalg.cholesky(information + S)
    F = np.linalg.solve(information + S, information)
    edf = np.diag(F).copy()
    return edf, float(np.trace(F)), np.diag(2 * F - F @ F).copy()

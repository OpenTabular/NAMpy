#utils/distributional_metrics.py
import math
from typing import Optional

import numpy as np
from scipy.special import gammaln


_EPS = 1e-9

def _as_1d(x):
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 1:
        return x
    raise ValueError(f"Expected 1D array or shape (n,1); got shape {x.shape}")


def _safe_positive(x, eps=_EPS):
    return np.clip(np.asarray(x, dtype=float), eps, None)


def _beta_mean_from_params(y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 2 or y_pred.shape[1] < 2:
        raise ValueError(
            "Expected Beta transformed params with shape (n, 2): [alpha, beta]."
        )
    alpha = _safe_positive(y_pred[:, 0])
    beta = _safe_positive(y_pred[:, 1])
    return alpha / (alpha + beta)


def _gamma_mean_from_params(y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 2 or y_pred.shape[1] < 2:
        raise ValueError(
            "Expected Gamma transformed params with shape (n, 2): [shape, rate]."
        )
    shape = _safe_positive(y_pred[:, 0])
    rate = _safe_positive(y_pred[:, 1])
    return shape / rate


def _student_t_params(y_pred, df=None):
    """
    Accept either:
      - y_pred shape (n,3): [df, loc, scale]
      - y_pred shape (n,2): [loc, scale], with scalar df provided
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 2:
        raise ValueError(f"Expected 2D array for Student-t params; got {y_pred.shape}")

    if y_pred.shape[1] >= 3:
        nu = _safe_positive(y_pred[:, 0])
        loc = y_pred[:, 1]
        scale = _safe_positive(y_pred[:, 2])
        return nu, loc, scale

    if y_pred.shape[1] == 2 and df is not None:
        nu = np.full(y_pred.shape[0], float(df), dtype=float)
        loc = y_pred[:, 0]
        scale = _safe_positive(y_pred[:, 1])
        return nu, loc, scale

    raise ValueError(
        "Student-t metrics expect transformed params [df, loc, scale] "
        "or [loc, scale] with `df=` provided."
    )


def poisson_deviance(y_true, y_pred):
    """
    Poisson deviance.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like
        Either:
          - shape (n,) / (n,1): predicted mean/rate
          - shape (n, k): if k>=1, column 0 is interpreted as rate
    """
    y = _as_1d(y_true).astype(float)
    pred = np.asarray(y_pred, dtype=float)

    if pred.ndim == 2:
        mu = pred[:, 0]
    else:
        mu = _as_1d(pred)

    mu = _safe_positive(mu)
    if y.shape[0] != mu.shape[0]:
        raise ValueError("y_true and y_pred lengths do not match.")

    # 0 * log(0 / mu) := 0
    term = np.where(y > 0, y * np.log(np.clip(y, _EPS, None) / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))/y.shape[0]


def gamma_deviance(y_true, y_pred):
    """
    Gamma deviance.

    Parameters
    ----------
    y_true : array-like, positive
    y_pred : array-like
        Either:
          - predicted mean mu (shape (n,) or (n,1))
          - transformed Gamma params [shape, rate] (shape (n,2))
    """
    y = _safe_positive(_as_1d(y_true))
    pred = np.asarray(y_pred, dtype=float)

    if pred.ndim == 2 and pred.shape[1] >= 2:
        mu = _gamma_mean_from_params(pred)
    else:
        mu = _safe_positive(_as_1d(pred))

    if y.shape[0] != mu.shape[0]:
        raise ValueError("y_true and y_pred lengths do not match.")

    # Standard Gamma deviance for mean predictions
    return float(2.0 * np.sum((y - mu) / mu - np.log(y / mu)))


def beta_brier_score(y_true, y_pred):
    """
    Compatibility function retained from original code.

    For Beta LSS models, `y_pred` is usually transformed params [alpha, beta].
    We therefore compute MSE on the Beta mean:
        E[Y] = alpha / (alpha + beta)

    If y_pred is already a 1D mean prediction, computes ordinary MSE.
    """
    y = np.clip(_as_1d(y_true).astype(float), _EPS, 1.0 - _EPS)
    pred = np.asarray(y_pred, dtype=float)

    if pred.ndim == 2 and pred.shape[1] >= 2:
        mu = _beta_mean_from_params(pred)
    else:
        mu = _as_1d(pred).astype(float)

    if y.shape[0] != mu.shape[0]:
        raise ValueError("y_true and y_pred lengths do not match.")

    return float(np.mean((mu - y) ** 2))


def dirichlet_error(y_true, y_pred):
    """
    Mean squared error on simplex-valued predictions.

    Parameters
    ----------
    y_true : array-like, shape (n, K)
    y_pred : array-like, shape (n, K)
        Can be probabilities/simplex points OR positive concentrations.
        If rows do not sum to ~1 but are positive, we normalize row-wise.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    if y.ndim == 3 and y.shape[1] == 1:
        y = y[:, 0, :]
    if p.ndim == 3 and p.shape[1] == 1:
        p = p[:, 0, :]

    if y.ndim != 2 or p.ndim != 2 or y.shape != p.shape:
        raise ValueError(
            f"dirichlet_error expects matching 2D arrays; got {y.shape=} and {p.shape=}"
        )

    # Ensure y on simplex
    y = np.clip(y, _EPS, None)
    y = y / np.clip(y.sum(axis=1, keepdims=True), _EPS, None)

    # If predictions do not sum to 1, treat them as concentrations and convert to mean
    row_sums = p.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        p = np.clip(p, _EPS, None)
        p = p / np.clip(p.sum(axis=1, keepdims=True), _EPS, None)

    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def student_t_loss(y_true, y_pred, df=None):
    """
    Student-t negative log-likelihood (up to exact constants when df is provided per sample, exact here).

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like
        Either transformed params [df, loc, scale] or [loc, scale] with `df=...`.
    df : float, optional
        Used only when y_pred has shape (n,2).
    """
    y = _as_1d(y_true).astype(float)
    nu, loc, scale = _student_t_params(y_pred, df=df)

    if not (y.shape[0] == nu.shape[0] == loc.shape[0] == scale.shape[0]):
        raise ValueError("y_true and y_pred lengths do not match.")

    z = (y - loc) / scale
    # Exact Student-t NLL
    nll = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
        - ((nu + 1.0) / 2.0) * np.log1p((z**2) / nu)
    )
    return float(-np.mean(nll))


def negative_binomial_deviance(y_true, y_pred, alpha: Optional[float] = None):
    """
    Negative Binomial deviance under mean/dispersion parameterization.

    Supported inputs
    ----------------
    y_pred can be:
      - shape (n,) / (n,1): mean mu (requires scalar `alpha`)
      - shape (n,2): transformed params [mean, dispersion], where dispersion=alpha > 0
    """
    y = _as_1d(y_true).astype(float)
    pred = np.asarray(y_pred, dtype=float)

    if pred.ndim == 2 and pred.shape[1] >= 2:
        mu = _safe_positive(pred[:, 0])
        alpha_arr = _safe_positive(pred[:, 1])
    else:
        if alpha is None:
            raise ValueError(
                "negative_binomial_deviance requires `alpha` when y_pred is only a mean prediction."
            )
        mu = _safe_positive(_as_1d(pred))
        alpha_arr = np.full_like(mu, float(alpha), dtype=float)
        alpha_arr = _safe_positive(alpha_arr)

    if y.shape[0] != mu.shape[0]:
        raise ValueError("y_true and y_pred lengths do not match.")

    # NB2 variance: Var = mu + alpha * mu^2
    # Deviance contribution (alpha -> 0 recovers Poisson limit)
    term1 = np.where(y > 0, y * np.log(np.clip(y, _EPS, None) / mu), 0.0)
    term2 = (y + 1.0 / alpha_arr) * np.log(
        (1.0 + alpha_arr * y) / np.clip(1.0 + alpha_arr * mu, _EPS, None)
    )
    dev = 2.0 * np.sum(term1 - term2)
    return float(dev)


def inverse_gamma_loss(y_true, y_pred):
    """
    Inverse-Gamma negative log-likelihood from transformed params [shape, rate].

    Parameters
    ----------
    y_true : array-like, shape (n,), positive
    y_pred : array-like, shape (n,2)
        Transformed params [shape, rate]
    """
    y = _safe_positive(_as_1d(y_true))
    pred = np.asarray(y_pred, dtype=float)

    if pred.ndim != 2 or pred.shape[1] < 2:
        raise ValueError(
            "inverse_gamma_loss expects transformed params with shape (n,2): [shape, rate]."
        )

    shape = _safe_positive(pred[:, 0])
    rate = _safe_positive(pred[:, 1])

    if y.shape[0] != shape.shape[0]:
        raise ValueError("y_true and y_pred lengths do not match.")

    # InverseGamma(shape=a, rate=b) logpdf:
    #   a log b - gammaln(a) - (a+1) log y - b / y
    logpdf = shape * np.log(rate) - gammaln(shape) - (shape + 1.0) * np.log(y) - rate / y
    return float(-np.mean(logpdf))
"""
Covariance matrix utilities for fitted GAMs.

Two covariance estimates are provided for each fitted model:

- **Bayesian posterior covariance** ``Vp = scale * A^{-1}`` where
  ``A = X'WX + S`` is the penalized information matrix.  This is the
  covariance used by default for confidence intervals on smooth functions;
  it accounts for the smoothing-parameter uncertainty correction due to
  the penalty.

- **Frequentist sandwich covariance** ``Vf = scale * A^{-1} X'WX A^{-1}``.
  This is appropriate when the model is viewed as a regression with a
  fixed penalty rather than a prior.
"""

import numpy as np

from .._model_state import _cov_bayes, _cov_freq


def build_bayes_and_freq_covariances(scale, A_inv, XWX):
    """
    Compute Bayesian posterior and frequentist sandwich covariance matrices.

    Parameters
    ----------
    scale : float
        Dispersion / scale estimate (``sigma^2`` for Gaussian).
    A_inv : array, shape (q, q)
        Inverse of the penalized information matrix ``A = X'WX + S``.
    XWX : array, shape (q, q)
        Unpenalized information matrix ``X'WX``.

    Returns
    -------
    Vp : ndarray
        Bayesian posterior covariance ``scale * A^{-1}``.
    Vf : ndarray
        Frequentist sandwich covariance ``scale * A^{-1} X'WX A^{-T}``.
    H_coef : ndarray
        Hat matrix in coefficient space ``A^{-1} X'WX``; trace gives EDF.
    """
    scale = float(scale)
    A_inv = np.asarray(A_inv, dtype=np.float64)
    XWX = np.asarray(XWX, dtype=np.float64)

    H_coef = A_inv @ XWX
    Vp = scale * A_inv
    # mgcv enforces symmetry on the inverted penalized information matrix
    # before forming frequentist covariance (`vcov(..., freq=TRUE)` parity).
    A_inv = 0.5 * (A_inv + A_inv.T)
    Vf = scale * (A_inv @ XWX @ A_inv.T)
    return Vp, Vf, H_coef


def select_covariance_matrix(model, cov=None):
    key = (cov or model.covariance).lower()
    if key == "bayes":
        cov_bayes = _cov_bayes(model)
        return None if cov_bayes is None else np.asarray(cov_bayes, dtype=np.float64)
    if key == "freq":
        cov_freq = _cov_freq(model)
        return None if cov_freq is None else np.asarray(cov_freq, dtype=np.float64)
    raise ValueError("cov must be 'bayes' or 'freq'.")

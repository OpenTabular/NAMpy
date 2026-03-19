import numpy as np


def build_bayes_and_freq_covariances(scale, A_inv, XWX):
    """
    Build Bayes and frequentist covariance matrices from a penalized system.

    Returns
    -------
    Vp, Vf, H_coef
    """
    scale = float(scale)
    A_inv = np.asarray(A_inv, dtype=np.float64)
    XWX = np.asarray(XWX, dtype=np.float64)

    H_coef = A_inv @ XWX
    Vp = scale * A_inv
    Vf = scale * (A_inv @ XWX @ A_inv.T)
    return Vp, Vf, H_coef


def select_covariance_matrix(model, cov=None):
    key = (cov or model.covariance).lower()
    if key == "bayes":
        return model.Vp_
    if key == "freq":
        return model.Vf_
    raise ValueError("cov must be 'bayes' or 'freq'.")
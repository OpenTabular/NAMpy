import numpy as np

from .lpmatrix import _build_prediction_matrices
from .predict import predict_values


def predict_term_contributions(model, X=None, offset=None):
    """
    Build term-wise contribution dictionary used by plotting and diagnostics.

    Returns a dict with:
    - output: link-scale additive predictor
    - response: response-scale prediction (for non-Gaussian families)
    - one entry per term label
    - intercept
    - offset (when active)
    """
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    Z_new, _ = _build_prediction_matrices(model, X_new=X)
    offset_vec = model._prediction_offset(X, offset)

    eta = predict_values(model, X=X, type="link", offset=offset)
    result = {"output": eta}

    if model.family.name != "gaussian":
        result["response"] = predict_values(model, X=X, type="response", offset=offset)

    for tb in model.term_blocks_:
        result[tb.label] = Z_new[:, tb.coef_slice] @ model.coef_[tb.coef_slice]

    if model.fit_intercept:
        result["intercept"] = np.array(model.intercept_, dtype=np.float64)

    if offset_vec is not None:
        result["offset"] = np.asarray(offset_vec, dtype=np.float64)

    return result
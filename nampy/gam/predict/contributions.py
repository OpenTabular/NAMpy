import numpy as np

from .linear_predictor_matrix import _build_prediction_matrices
from .predictions import _term_contribution, predict_values


def predict_term_contributions(model, X=None, offset=None):
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    Z_new, _ = _build_prediction_matrices(model, X_new=X)
    offset_vec = model._prediction_offset(X, offset)

    eta = predict_values(model, X=X, type="link", offset=offset)
    result = {"output": eta}

    if model.family.name != "gaussian":
        result["response"] = predict_values(model, X=X, type="response", offset=offset)

    for tb in model.term_blocks_:
        result[tb.term_id] = _term_contribution(model, Z_new, tb)

    if model.fit_intercept:
        result["intercept"] = np.array(model.intercept_, dtype=np.float64)

    if offset_vec is not None:
        result["offset"] = np.asarray(offset_vec, dtype=np.float64)

    return result


__all__ = ["predict_term_contributions"]

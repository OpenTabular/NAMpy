import numpy as np


def scale_penalty(basis, penalty):
    """
    Rescale the penalty matrix based on the smoother design matrix.
    """
    basis = np.asarray(basis, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)

    X_inf_norm = max(np.sum(np.abs(basis), axis=1)) ** 2
    S_norm = np.linalg.norm(penalty, ord=1)

    if X_inf_norm <= 0 or S_norm <= 0:
        return penalty.copy()

    norm = S_norm / X_inf_norm
    return penalty / norm

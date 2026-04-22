import numpy as np

from ..linalg import (
    r_matrix_norm_one,
    numerical_rank,
    symmetric_eigen_partition,
    symmetrize_matrix,
)


def penalty_rescale_factor(basis, penalty, tol=0.0):
    basis = np.asarray(basis, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)

    # mgcv::smoothCon() scales with norm(sm$X, type="I")^2, i.e. max row sum.
    X_inf_norm = r_matrix_norm_one(basis) ** 2
    S_norm = np.linalg.norm(penalty, ord=1)

    if X_inf_norm <= tol or S_norm <= tol:
        return 1.0

    return float(S_norm / X_inf_norm)


def scale_penalty(basis, penalty):
    basis = np.asarray(basis, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)

    norm = penalty_rescale_factor(basis, penalty)
    if norm <= 0:
        return penalty.copy()
    return penalty / norm


def symmetrize_penalty(P):
    return symmetrize_matrix(P)


def penalty_eigendecomposition(P, tol=1e-10):
    return symmetric_eigen_partition(P, tol=tol, descending=False)


def null_space_penalty_from_penalty(P, tol=1e-10):
    dec = penalty_eigendecomposition(P, tol=tol)
    U0 = dec["U0"]

    if U0.shape[1] == 0:
        S0 = np.zeros_like(np.asarray(P, dtype=np.float64))
        rank = 0
        null_dim = 0
    else:
        S0 = U0 @ U0.T
        S0 = symmetrize_penalty(S0)
        rank = numerical_rank(S0, hermitian=True)
        null_dim = U0.shape[1]

    return S0, {
        "rank": rank,
        "null_space_dim": null_dim,
        "tol_eff": dec["tol_eff"],
    }


__all__ = [
    "penalty_rescale_factor",
    "scale_penalty",
    "symmetrize_penalty",
    "penalty_eigendecomposition",
    "null_space_penalty_from_penalty",
]

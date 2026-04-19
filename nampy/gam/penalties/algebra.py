import numpy as np


def penalty_rescale_factor(basis, penalty, tol=0.0):
    basis = np.asarray(basis, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)

    X_inf_norm = max(np.sum(np.abs(basis), axis=1)) ** 2
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
    P = np.asarray(P, dtype=np.float64)
    return 0.5 * (P + P.T)


def penalty_eigendecomposition(P, tol=1e-10):
    P = symmetrize_penalty(P)
    evals, U = np.linalg.eigh(P)
    idx = np.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]

    tol_eff = tol * max(1.0, np.max(np.abs(evals)) if evals.size else 1.0)
    null_mask = evals <= tol_eff
    pos_mask = ~null_mask

    return {
        "evals": evals,
        "U": U,
        "U0": U[:, null_mask],
        "U1": U[:, pos_mask],
        "d_pos": evals[pos_mask],
        "null_mask": null_mask,
        "pos_mask": pos_mask,
        "rank": int(np.sum(pos_mask)),
        "null_space_dim": int(np.sum(null_mask)),
        "tol_eff": tol_eff,
    }


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
        rank = int(np.linalg.matrix_rank(S0))
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

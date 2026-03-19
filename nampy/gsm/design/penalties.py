# gsm/design/penalties.py
import numpy as np
 
def symmetrize_penalty(P):
    P = np.asarray(P, dtype=np.float64)
    return 0.5 * (P + P.T)


def penalty_eigendecomposition(P, tol=1e-10):
    """
    Stable eigendecomposition helper for quadratic penalties.

    Returns the null-space / positive-space split of a symmetric penalty.
    """
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
    """
    Construct a selection/null-space penalty from an existing penalty matrix.

    If the original smooth penalty is S, then this returns the orthogonal projector
    onto the null space of S:
        S0 = U0 U0^T
    where columns of U0 span null(S).

    This is the standard mathematically clean way to add an extra shrinkage penalty
    that can penalize the smooth's null-space components to zero.
    """
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

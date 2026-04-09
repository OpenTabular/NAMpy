import numpy as np


def identconst(basis, penalty):
    """
    Absorb identifiability (sum-to-zero) constraint into basis/penalty.
    """
    basis = np.asarray(basis, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)

    constraint_matrix = basis.mean(axis=0).reshape(-1, 1)
    q, _ = np.linalg.qr(constraint_matrix, mode="complete")
    penalty = np.double(
        np.linalg.multi_dot([np.transpose(q[:, 1:]), penalty, q[:, 1:]])
    )
    basis = basis @ q[:, 1:]
    return basis, penalty, q[:, 1:]

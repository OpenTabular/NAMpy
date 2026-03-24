from dataclasses import dataclass

import numpy as np

from .univariate_bases import (
    bspline_design_matrix,
    pspline_difference_penalty,
    pspline_knots,
    pspline_predict_matrix,
)


@dataclass
class PSplineBasisSetup:
    feature_index: int
    feature_name: str
    basis_order: int
    penalty_order: int
    knots: np.ndarray
    basis_train: np.ndarray
    penalty: np.ndarray
    bs_dim: int
    rank: int


def build_pspline_term_setup(
    x,
    *,
    feature_index,
    feature_name,
    bs_dim,
    m,
    knots=None,
):
    x = np.asarray(x, dtype=np.float64).ravel()
    basis_order, penalty_order = (int(m[0]), int(m[1]))
    if basis_order < 0 or penalty_order < 0:
        raise ValueError("For bs='ps', m entries must be >= 0.")

    k = pspline_knots(
        x,
        bs_dim=int(bs_dim),
        basis_order=basis_order,
        supplied_knots=knots,
    )
    degree = basis_order + 1
    B = bspline_design_matrix(
        x,
        k,
        degree=degree,
        deriv=0,
        extrapolate=True,
    )
    S = pspline_difference_penalty(B.shape[1], penalty_order)
    S = 0.5 * (S + S.T)

    return PSplineBasisSetup(
        feature_index=int(feature_index),
        feature_name=str(feature_name),
        basis_order=int(basis_order),
        penalty_order=int(penalty_order),
        knots=np.asarray(k, dtype=np.float64),
        basis_train=np.asarray(B, dtype=np.float64),
        penalty=np.asarray(S, dtype=np.float64),
        bs_dim=int(B.shape[1]),
        rank=int(np.linalg.matrix_rank(S)),
    )


def predict_pspline_term(x_new, setup: PSplineBasisSetup):
    x_new = np.asarray(x_new, dtype=np.float64).ravel()
    return np.asarray(
        pspline_predict_matrix(
            x_new,
            setup.knots,
            basis_order=setup.basis_order,
            deriv=0,
        ),
        dtype=np.float64,
    )

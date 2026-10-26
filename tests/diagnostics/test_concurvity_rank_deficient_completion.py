"""mgcv's tol=0 QR keeps non-identified completion directions."""

import numpy as np


def test_concurvity_rank_deficient_completion():
    # mgcv.r::concurvity uses r=ncol(Xi), not rank(Xi), then includes
    # R[1:r,] in its estimate. Both factorizations are exact QR decompositions
    # of [Xi, Xj] with Xi=[e1,e1], Xj=e2, but the unused second Q direction
    # can either contain all of Xj or none of it.
    eye = np.eye(3)
    design = np.column_stack([eye[:, 0], eye[:, 0], eye[:, 1]])
    estimates = []
    for q in (eye, eye[:, [0, 2, 1]]):
        r = q.T @ design
        np.testing.assert_array_equal(q @ r, design)
        np.testing.assert_array_equal(np.tril(r, -1), np.zeros((3, 3)))
        np.testing.assert_array_equal(r.T @ r, design.T @ design)
        right = r[:, 2:]
        estimates.append(np.sum(right[:2] ** 2) / np.sum(right**2))
    np.testing.assert_array_equal(estimates, [1.0, 0.0])

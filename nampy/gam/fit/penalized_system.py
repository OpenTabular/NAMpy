"""
Penalized system assembly and numerically stable solvers.

Provides three utilities used across all fitting backends:

- :func:`stabilized_cholesky_solve`: Cholesky solve with automatic jitter
  schedule for near-singular systems.
- :func:`build_full_design`: Prepend intercept column to term design matrix.
- :func:`build_full_penalty_from_blocks`: Assemble the total penalty matrix
  ``sum_k lambda_k * S_k`` from per-term penalty blocks.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def stabilized_cholesky_solve(A, b, jitter_schedule=None):
    if jitter_schedule is None:
        jitter_schedule = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
    last_err = None
    eye = np.eye(A.shape[0], dtype=np.float64)
    for jitter in jitter_schedule:
        try:
            A_work = A if jitter == 0.0 else A + jitter * eye
            cA, loA = cho_factor(A_work, check_finite=False)
            x = cho_solve((cA, loA), b, check_finite=False)
            return x, cA, loA, jitter
        except np.linalg.LinAlgError as err:
            last_err = err
    raise np.linalg.LinAlgError(
        f"Unable to Cholesky-solve penalized system. Last error: {last_err}"
    )


def build_full_design(Z, fit_intercept):
    Z = np.asarray(Z, dtype=np.float64)
    if fit_intercept:
        return np.column_stack([np.ones(Z.shape[0], dtype=np.float64), Z])
    return Z


def build_full_penalty_from_blocks(
    penalty_blocks, smoothing_params, fit_intercept, n_coef
):
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64)
    offset = 1 if fit_intercept else 0
    total_dim = offset + int(n_coef)

    P_full = np.zeros((total_dim, total_dim), dtype=np.float64)
    for pb in penalty_blocks:
        if pb.smoothing_index < 0 or pb.smoothing_index >= len(smoothing_params):
            raise IndexError(
                f"Penalty block smoothing_index={pb.smoothing_index} is out of bounds "
                f"for {len(smoothing_params)} smoothing parameters."
            )
        lam = float(smoothing_params[pb.smoothing_index])
        sl = pb.coef_slice
        full_sl = slice(offset + sl.start, offset + sl.stop)

        P = np.asarray(pb.matrix, dtype=np.float64)
        width = sl.stop - sl.start
        if P.shape != (width, width):
            raise ValueError(
                f"Penalty block for term {pb.label!r} has shape {P.shape}, "
                f"but its coefficient slice width is {width}."
            )

        P_full[full_sl, full_sl] += lam * P

    return P_full


def coerce_fit_offset(offset, n_rows):
    if offset is None:
        return None
    out = np.asarray(offset, dtype=np.float64).ravel()
    if out.shape != (int(n_rows),):
        raise ValueError(f"offset must have shape ({int(n_rows)},), got {out.shape}.")
    if not np.isfinite(out).all():
        raise ValueError("offset contains NaN or Inf")
    return out

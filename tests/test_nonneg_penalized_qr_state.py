"""Parity checks for nonnegative penalized QR state construction."""

from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.fit.penalized_qr import build_penalized_qr_state_nonnegative
from nampy.gam.fit.linalg.stacked_qr import (
    STACKED_QR_RANK_TOLERANCE,
    penalty_sqrt_rows,
    pls_fit1_nonneg_w,
)


def _random_pls_instance(
    n: int, q: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, q))
    coef_true = rng.standard_normal(q)
    mu = X @ coef_true
    z = mu + 0.1 * rng.standard_normal(n)
    w = np.abs(rng.standard_normal(n)) + 0.5
    P_pen = np.diag(np.concatenate([[0.0], np.full(q - 1, 0.3 + rng.random(q - 1))]))
    E, Es = penalty_sqrt_rows(P_pen)
    return X, z, w, E, Es


def _rS_q_rows(E: np.ndarray, q: int) -> np.ndarray:
    """``mgcv`` packs each ``rS_i`` as ``q × rSncol[i]``; ``E`` from eigh is ``n_e × q`` → ``q × n_e``."""
    E = np.asarray(E, dtype=np.float64)
    n_e, qe = E.shape
    if qe != q:
        raise ValueError("E must have q columns.")
    return E.T.copy()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_penalized_qr_state_beta_matches_pls_fit1(seed: int) -> None:
    n, q = 80, 12
    X, z, w, E, Es = _random_pls_instance(n, q, seed=seed)
    wy = w * z
    rS = _rS_q_rows(E, q)

    coef_pls, _pen = pls_fit1_nonneg_w(
        X,
        z,
        w,
        wy,
        penalty_sqrt_E=E,
        penalty_rank_Es=Es,
        rank_tol=STACKED_QR_RANK_TOLERANCE,
    )
    state = build_penalized_qr_state_nonnegative(
        X,
        z,
        w,
        penalty_sqrt_E=E,
        penalty_rank_Es=Es,
        rS=rS,
        rank_tol=STACKED_QR_RANK_TOLERANCE,
        reml=True,
    )

    np.testing.assert_allclose(state.beta_full, coef_pls, rtol=0, atol=1e-9)
    fit_pls = X @ coef_pls
    fit_state = X @ state.beta_full
    np.testing.assert_allclose(fit_state, fit_pls, rtol=0, atol=1e-10)


@pytest.mark.parametrize("seed", [3])
def test_penalized_qr_state_ldet_matches_log_det_from_upper_R(seed: int) -> None:
    n, q = 60, 10
    X, z, w, E, Es = _random_pls_instance(n, q, seed=seed)
    state = build_penalized_qr_state_nonnegative(
        X,
        z,
        w,
        penalty_sqrt_E=E,
        penalty_rank_Es=Es,
        rS=_rS_q_rows(E, q),
        rank_tol=STACKED_QR_RANK_TOLERANCE,
        reml=True,
    )
    d = np.abs(np.diag(state.Rh))
    expect = float(2.0 * np.sum(np.log(np.maximum(d, np.finfo(np.float64).tiny))))
    np.testing.assert_allclose(state.ldet_XWX_plus_S, expect, rtol=0, atol=1e-10)


def test_pls_fit1_alias_is_rejected_for_coef_method() -> None:
    X, z, w, E, Es = _random_pls_instance(40, 6, seed=11)
    wy = w * z

    with pytest.raises(ValueError, match="Unknown coef_method"):
        pls_fit1_nonneg_w(
            X,
            z,
            w,
            wy,
            penalty_sqrt_E=E,
            penalty_rank_Es=Es,
            coef_method="pls_fit1",
        )

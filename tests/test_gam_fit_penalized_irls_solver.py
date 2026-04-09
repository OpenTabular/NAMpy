"""Structural tests for :mod:`nampy.gam.fit.solvers.penalized_irls` (mgcv ``gam.fit3`` analogue)."""

from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.families.exponential import GaussianIdentityFamily
from nampy.gam.fit.linalg.stacked_qr import (
    balanced_penalty_template_sqrt_for_rank,
    penalty_sqrt_rows,
    solve_gaussian_penalized_ls_stacked_qr,
)
from nampy.gam.fit.solvers.penalized_irls import (
    PenalizedIrlsControl,
    fit_penalized_irls,
)


class _PB:
    smoothing_index = 0
    matrix = np.eye(3)
    coef_slice = slice(0, 3)


def test_gaussian_identity_one_step_matches_direct_pls():
    """``strictly.additive`` short-circuit: one PLS step equals Gaussian PLS on working data."""
    rng = np.random.default_rng(2)
    n, q = 60, 4
    X = np.c_[np.ones(n), rng.standard_normal((n, q - 1))]
    beta = np.array([0.4, -0.15, 0.05, 0.2])
    y = X @ beta + rng.standard_normal(n) * 0.25
    lam = 0.35
    P = np.zeros((q, q), dtype=np.float64)
    P[1:, 1:] = lam * np.eye(q - 1)
    Sr, _ = penalty_sqrt_rows(P)
    Eb = balanced_penalty_template_sqrt_for_rank(
        [_PB()], fit_intercept=True, n_coef=q - 1
    )
    w = np.ones(n, dtype=np.float64)
    direct = solve_gaussian_penalized_ls_stacked_qr(
        X,
        y,
        w,
        P,
        penalty_blocks=[_PB()],
        fit_intercept=True,
        n_coef=q - 1,
    )
    out = fit_penalized_irls(
        X,
        y,
        np.log(np.array([lam])),
        Sr,
        Eb,
        P,
        GaussianIdentityFamily(),
        weights=w,
        offset=np.zeros(n),
        control=PenalizedIrlsControl(maxit=10, epsilon=1e-10),
    )
    assert out["converged"]
    assert out["iterations"] == 1
    np.testing.assert_allclose(out["coef"], direct["coef_full"], atol=1e-10, rtol=0.0)


def test_extended_family_raises():
    from nampy.gam.families.family_base import ExtendedFamily

    class DummyExt(ExtendedFamily):
        name = "dummy"
        link_name = "identity"

    X = np.eye(2)
    y = np.array([1.0, 2.0])
    P = np.eye(2) * 0.1
    Sr, Es = penalty_sqrt_rows(P)
    with pytest.raises(NotImplementedError, match="Extended and general family"):
        fit_penalized_irls(
            X,
            y,
            np.array([0.0]),
            Sr,
            Es,
            P,
            DummyExt(),
        )


def test_empty_columns():
    y = np.array([1.0, 2.0, 3.0])
    out = fit_penalized_irls(
        np.zeros((3, 0)),
        y,
        np.array([]),
        np.zeros((0, 0)),
        np.zeros((0, 0)),
        np.zeros((0, 0)),
        GaussianIdentityFamily(),
    )
    assert out["coef"].shape == (0,)
    assert out["converged"]


def test_step_halving_recomputes_mu_eta_and_variance():
    class TrackingFamily:
        name = "tracking"
        link_name = "identity"
        canonical_link = True

        def __init__(self):
            self.mu_eta_calls = 0
            self.variance_calls = 0

        def validate_y(self, y):
            return np.asarray(y, dtype=np.float64).ravel()

        def inverse_link(self, eta):
            return np.asarray(eta, dtype=np.float64)

        def link(self, mu):
            return np.asarray(mu, dtype=np.float64)

        def initialize_mu(self, y):
            return np.zeros_like(np.asarray(y, dtype=np.float64))

        def mu_eta(self, eta):
            self.mu_eta_calls += 1
            return np.ones_like(np.asarray(eta, dtype=np.float64))

        def variance(self, mu):
            self.variance_calls += 1
            return np.ones_like(np.asarray(mu, dtype=np.float64))

        def deviance(self, y, mu):
            y = np.asarray(y, dtype=np.float64)
            mu = np.asarray(mu, dtype=np.float64)
            return float(np.sum((y - mu) ** 2))

        def valid_mu(self, mu):
            return bool(np.all(np.abs(np.asarray(mu, dtype=np.float64)) < 1.0))

    family = TrackingFamily()
    X = np.ones((1, 1), dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)
    P = np.zeros((1, 1), dtype=np.float64)

    out = fit_penalized_irls(
        X,
        y,
        np.array([], dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
        P,
        family,
        control=PenalizedIrlsControl(maxit=10, epsilon=1e-12),
    )

    assert family.mu_eta_calls > 1
    assert family.variance_calls > 1
    assert out["warnings"]

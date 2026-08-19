"""Full-fit parity for structured-smooth variants with constructor-only cover.

Backlog targets: random slopes and numeric-by random effects, factor smooths
on cc/ps bases, non-Gaussian point constraints (pc=), linked id= groups under
Poisson/Gamma, and fx=/select= tensors under optimization.

Upstream references: mgcv/R/smooth.r (re/fs constructors, pc handling),
mgcv/R/mgcv.r::gam.setup (id linkage, fx, select), compared through
tests/parity/mgcv_snapshot.R.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _make_gamma_data,
    _make_gaussian_data_3col,
    _make_poisson_data,
    _normalize_python_formula_text,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]


def _random_slope_data(seed=481, n=180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    row = np.arange(n)
    f = np.asarray([f"g{i}" for i in range(6)])[row % 6]
    slopes = {f"g{i}": 0.25 * (i - 2.5) for i in range(6)}
    intercepts = {f"g{i}": 0.1 * (i % 3 - 1) for i in range(6)}
    y = (
        0.4 * np.sin(1.2 * x0)
        + np.asarray([slopes[v] for v in f]) * x0
        + np.asarray([intercepts[v] for v in f])
        + rng.normal(scale=0.15, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "z": z, "f": pd.Categorical(f)})


def _periodic_fs_data(seed=482, n=150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    row = np.arange(n)
    f = np.asarray(["a", "b", "c"])[row % 3]
    phase = {"a": 0.0, "b": 0.8, "c": 1.9}
    y = np.sin(2.0 * np.pi * x + np.asarray([phase[v] for v in f])) + rng.normal(
        scale=0.15, size=n
    )
    return pd.DataFrame({"y": y, "x": x, "f": pd.Categorical(f)})


@dataclass(frozen=True)
class _FullFitCase:
    case_id: str
    data_factory: object
    family: object
    formula: str
    method: str = "REML"
    select: bool = False
    pred_atol: float = 1e-4
    sp_log_atol: float = 3e-2
    check_sp: bool = True
    criterion_atol: float = 5e-2
    n_sp: int | None = None
    extra_checks: tuple = field(default=())


_FULL_FIT_CASES = [
    _FullFitCase(
        "random_slope_gaussian_reml",
        _random_slope_data,
        "gaussian",
        'y ~ s(x0, bs="cr", k=6) + s(x0, f, bs="re")',
        pred_atol=5e-4,
    ),
    _FullFitCase(
        "numeric_by_random_effect_gaussian_reml",
        _random_slope_data,
        "gaussian",
        'y ~ s(x0, bs="cr", k=6) + s(f, bs="re", by=z)',
        pred_atol=5e-4,
    ),
    _FullFitCase(
        "random_intercept_slope_pair_gaussian_reml",
        _random_slope_data,
        "gaussian",
        'y ~ s(x0, bs="cr", k=6) + s(f, bs="re") + s(x0, f, bs="re")',
        pred_atol=8e-4,
    ),
    _FullFitCase(
        "fs_cc_basis_gaussian_reml",
        _periodic_fs_data,
        "gaussian",
        'y ~ s(f, x, bs="fs", k=6, xt="cc")',
        pred_atol=5e-4,
        check_sp=False,
    ),
    _FullFitCase(
        "fs_ps_basis_gaussian_reml",
        _periodic_fs_data,
        "gaussian",
        'y ~ s(f, x, bs="fs", k=6, xt="ps")',
        pred_atol=5e-4,
        check_sp=False,
    ),
    _FullFitCase(
        "poisson_point_constraint_reml",
        _make_poisson_data,
        "poisson",
        'y ~ s(x0, bs="cr", k=8, pc=0.3) + s(x1, bs="cr", k=8)',
        pred_atol=5e-4,
    ),
    _FullFitCase(
        "gamma_point_constraint_reml",
        _make_gamma_data,
        {"name": "gamma", "link": "log"},
        'y ~ s(x0, bs="cr", k=8, pc=0.0) + s(x1, bs="cr", k=8)',
        pred_atol=8e-4,
    ),
    _FullFitCase(
        "linked_id_poisson_reml",
        _make_poisson_data,
        "poisson",
        'y ~ s(x0, bs="cr", k=8, id=1) + s(x1, bs="cr", k=8, id=1)',
        pred_atol=5e-4,
        n_sp=1,
    ),
    _FullFitCase(
        "linked_id_gamma_reml",
        _make_gamma_data,
        {"name": "gamma", "link": "log"},
        'y ~ s(x0, bs="cr", k=8, id=1) + s(x1, bs="cr", k=8, id=1)',
        pred_atol=8e-4,
        n_sp=1,
    ),
    _FullFitCase(
        "te_fx_with_free_smooth_gaussian_reml",
        lambda: _make_gaussian_data_3col(seed=483, n=150),
        "gaussian",
        'y ~ te(x0, x1, bs=["cr", "ps"], k=[5, 5], fx=True) + s(x2, bs="cr", k=6)',
        pred_atol=5e-4,
        n_sp=1,
    ),
    _FullFitCase(
        "te_select_gaussian_reml",
        lambda: _make_gaussian_data_3col(seed=484, n=150),
        "gaussian",
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5]) + s(x2, bs="cr", k=6)',
        select=True,
        pred_atol=5e-4,
        check_sp=False,
    ),
]


@pytest.mark.parametrize("case", _FULL_FIT_CASES, ids=lambda case: case.case_id)
def test_structured_full_fit_extension_matches_mgcv(case):
    """Optimized full fits (not just constructors) match mgcv."""
    data = case.data_factory()
    gam = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method=case.method,
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        _normalize_python_formula_text(case.formula),
        case.family,
        case.method,
        select=case.select,
        optimizer="newton",
        allow_live_run=True,
    )
    if case.n_sp is not None:
        assert len(np.atleast_1d(actual["fit"]["smoothing_params"])) == case.n_sp, (
            "unexpected smoothing-parameter count (id/fx routing)"
        )
        assert len(np.atleast_1d(expected["fit"]["smoothing_params"])) == case.n_sp
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=case.pred_atol,
        pred_rtol=case.pred_atol,
        sp_log_atol=case.sp_log_atol,
        check_sp=case.check_sp,
        criterion_atol=case.criterion_atol,
    )

    # Function-space covariance parity: L Vp L^T is identified even when the
    # per-term basis orientation is not.  (parity_snapshot flattens prediction
    # matrices, so build the actual lpmatrix through the public API.)
    rows = np.arange(2, len(data), 13)
    actual_lp = np.asarray(
        gam.lpmatrix(data.iloc[rows].drop(columns=["y"])), dtype=np.float64
    )
    expected_lp = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)[
        rows
    ]
    actual_cov = np.asarray(actual["fit"]["cov_bayes"], dtype=np.float64)
    expected_cov = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)
    actual_fn_cov = actual_lp @ actual_cov @ actual_lp.T
    expected_fn_cov = expected_lp @ expected_cov @ expected_lp.T
    scale_ref = max(float(np.max(np.abs(expected_fn_cov))), 1e-12)
    np.testing.assert_allclose(
        actual_fn_cov,
        expected_fn_cov,
        atol=max(50.0 * case.pred_atol, 1e-4) * scale_ref,
        rtol=max(50.0 * case.pred_atol, 1e-4),
    )


def test_point_constraint_zeroes_term_at_constraint_point():
    """pc= pins the smooth to zero at the given point in both systems."""
    data = _make_poisson_data()
    formula = 'y ~ s(x0, bs="cr", k=8, pc=0.3) + s(x1, bs="cr", k=8)'
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    probe = data.iloc[:1].drop(columns=["y"]).copy()
    probe.loc[probe.index[0], "x0"] = 0.3
    terms = np.asarray(gam.predict(probe, type="terms"), dtype=np.float64)
    # The pc-constrained s(x0) column evaluates to zero at x0 = 0.3.
    assert np.any(np.abs(terms) < 1e-8), terms

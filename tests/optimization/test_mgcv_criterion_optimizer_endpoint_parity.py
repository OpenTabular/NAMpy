"""Strict optimizer-endpoint parity for ML/GCV/UBRE criteria and sp regimes.

The strict newton/REML endpoint surfaces are owned elsewhere
(test_mgcv_outer_optimization_parity.py, test_mgcv_score_hist_trace_parity.py).
This file owns the criteria and sp regimes that previously had only loose or
routing coverage: ML/GCV/UBRE endpoints with covariances, mixed fixed/free
smoothing parameters outside Gaussian models, and simultaneous boundary sps.

Upstream references: mgcv/R/mgcv.r::gam / estimate.gam (method and optimizer
dispatch), mgcv/R/gam.fit3.r::gam.fit3 (criterion values), compared through
tests/parity/mgcv_snapshot.R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import (
    _make_binomial_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]

_TWO_CR_FORMULA = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
_MIXED_SP_FORMULA = 'y ~ s(x0, bs="cr", k=8, sp=0.5) + s(x1, bs="cr", k=8)'


def _curved_binomial_data(seed=425, n=240):
    """Binomial data with a curved x1 effect so both sp optima are interior."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.9 * np.sin(x0) + 0.5 * x1**2 - 0.4
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


@dataclass(frozen=True)
class _EndpointCase:
    case_id: str
    data_factory: object
    family: object
    formula: str
    nampy_method: str
    mgcv_method: str
    optimizer: str
    mgcv_optimizer: str
    log_sp_atol: float
    pred_tol: float
    cov_rtol: float
    # BFGS carries an approximate outer Hessian into Vc, so edf2 only matches
    # to the Hessian-approximation level there.
    edf2_atol: float = 1e-5
    edf2_rtol: float = 1e-6
    scale_rtol: float = 1e-7
    deviance_rtol: float = 1e-8
    edf_atol: float = 5e-6


_ENDPOINT_CASES = [
    _EndpointCase(
        "gaussian_ml_newton",
        _make_gaussian_data,
        "gaussian",
        _TWO_CR_FORMULA,
        "ML",
        "ML",
        "outer_newton",
        "newton",
        5e-6,
        1e-8,
        1e-6,
    ),
    _EndpointCase(
        "gaussian_gcv_newton",
        _make_gaussian_data,
        "gaussian",
        _TWO_CR_FORMULA,
        "GCV",
        "GCV.Cp",
        "outer_newton",
        "newton",
        1e-4,
        5e-6,
        1e-5,
        # The GCV optimum is flat: the ~5e-5 log-sp endpoint difference moves
        # the scale and deviance at the 1e-7 level while the criterion stays
        # matched to 1e-9.
        scale_rtol=1e-6,
        deviance_rtol=1e-6,
        edf_atol=1e-4,
    ),
    _EndpointCase(
        "poisson_ubre_newton",
        _make_poisson_data,
        "poisson",
        _TWO_CR_FORMULA,
        "UBRE",
        "GCV.Cp",
        "outer_newton",
        "newton",
        5e-6,
        1e-7,
        1e-6,
    ),
    _EndpointCase(
        "binomial_ubre_newton",
        _curved_binomial_data,
        "binomial",
        _TWO_CR_FORMULA,
        "UBRE",
        "GCV.Cp",
        "outer_newton",
        "newton",
        5e-5,
        2e-5,
        1e-5,
        deviance_rtol=1e-6,
        edf_atol=1e-4,
    ),
    _EndpointCase(
        "gamma_ml_newton",
        _make_gamma_data,
        "gamma",
        _TWO_CR_FORMULA,
        "ML",
        "ML",
        "outer_newton",
        "newton",
        5e-6,
        1e-7,
        1e-6,
    ),
    _EndpointCase(
        "gaussian_ml_bfgs",
        _make_gaussian_data,
        "gaussian",
        _TWO_CR_FORMULA,
        "ML",
        "ML",
        "bfgs",
        "bfgs",
        2e-4,
        1e-7,
        1e-5,
        edf2_atol=5e-3,
        edf2_rtol=2e-3,
    ),
    _EndpointCase(
        "poisson_ml_bfgs",
        _make_poisson_data,
        "poisson",
        _TWO_CR_FORMULA,
        "ML",
        "ML",
        "bfgs",
        "bfgs",
        2e-4,
        1e-6,
        1e-5,
        edf2_atol=5e-3,
        edf2_rtol=2e-3,
    ),
]


def _assert_strict_endpoint(actual, expected, case: _EndpointCase) -> None:
    a_fit = actual["fit"]
    e_fit = expected["fit"]
    a_pred = actual["predictions"]
    e_pred = expected["predictions"]

    np.testing.assert_allclose(
        np.asarray(a_fit["log_smoothing_params"], dtype=np.float64),
        np.asarray(e_fit["log_smoothing_params"], dtype=np.float64),
        atol=case.log_sp_atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["criterion_value"], dtype=np.float64),
        np.asarray(e_fit["criterion_value"], dtype=np.float64),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["scale"], dtype=np.float64),
        np.asarray(e_fit["scale"], dtype=np.float64),
        rtol=case.scale_rtol,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["deviance"], dtype=np.float64),
        np.asarray(e_fit["deviance"], dtype=np.float64),
        rtol=case.deviance_rtol,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_total"], dtype=np.float64),
        np.asarray(e_fit["edf_total"], dtype=np.float64),
        atol=case.edf_atol,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_by_term"], dtype=np.float64),
        np.asarray(e_fit["edf_by_term"], dtype=np.float64),
        atol=case.edf_atol,
        rtol=1e-7,
    )
    if a_fit.get("edf2", None) is not None and e_fit.get("edf2", None) is not None:
        np.testing.assert_allclose(
            np.asarray(a_fit["edf2"], dtype=np.float64),
            np.asarray(e_fit["edf2"], dtype=np.float64),
            atol=case.edf2_atol,
            rtol=case.edf2_rtol,
        )

    for surface in ("response", "link", "se_response", "se_link"):
        expected_values = e_pred.get(surface, None)
        if expected_values is None:
            continue
        actual_values = a_pred.get(surface, None)
        assert actual_values is not None, f"missing prediction surface {surface}"
        np.testing.assert_allclose(
            np.asarray(actual_values, dtype=np.float64),
            np.asarray(expected_values, dtype=np.float64),
            atol=case.pred_tol,
            rtol=case.pred_tol,
        )

    # cr-only formulas identify the coefficient basis uniquely, so covariances
    # are comparable elementwise.
    for cov_key in ("cov_bayes", "cov_freq"):
        expected_cov = e_fit.get(cov_key, None)
        if expected_cov is None:
            continue
        actual_cov = a_fit.get(cov_key, None)
        assert actual_cov is not None, f"missing covariance {cov_key}"
        expected_cov = np.asarray(expected_cov, dtype=np.float64)
        actual_cov = np.asarray(actual_cov, dtype=np.float64)
        assert actual_cov.shape == expected_cov.shape
        scale_ref = max(float(np.max(np.abs(expected_cov))), 1e-12)
        np.testing.assert_allclose(
            actual_cov,
            expected_cov,
            atol=case.cov_rtol * scale_ref,
            rtol=case.cov_rtol,
        )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=lambda case: case.case_id)
def test_criterion_optimizer_endpoint_quantities_match_mgcv(case):
    """ML/GCV/UBRE endpoints match mgcv in sp, criterion, EDF and covariance."""
    data = case.data_factory()
    gam = GAM(
        family=case.family,
        formula=case.formula,
        optimize_smoothing=True,
        smoothing_method=case.nampy_method,
        smoothing_optimizer=case.optimizer,
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.mgcv_method,
        optimizer=case.mgcv_optimizer,
        allow_live_run=True,
    )
    _assert_strict_endpoint(actual, expected, case)


@pytest.mark.parametrize(
    ("case_id", "data_factory", "family", "nampy_method", "mgcv_method"),
    [
        ("poisson_mixed_reml", _make_poisson_data, "poisson", "REML", "REML"),
        (
            "gamma_mixed_ml",
            _make_gamma_data,
            {"name": "gamma", "link": "log"},
            "ML",
            "ML",
        ),
        ("binomial_mixed_reml", _make_binomial_data, "binomial", "REML", "REML"),
    ],
)
def test_non_gaussian_mixed_fixed_free_sp_matches_mgcv(
    case_id, data_factory, family, nampy_method, mgcv_method
):
    """A formula-fixed sp beside a free sp reproduces mgcv's mixed optimum."""
    data = data_factory()
    gam = GAM(
        family=family,
        formula=_MIXED_SP_FORMULA,
        optimize_smoothing=True,
        smoothing_method=nampy_method,
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        _MIXED_SP_FORMULA,
        family,
        mgcv_method,
        optimizer="newton",
        allow_live_run=True,
    )

    actual_sp = np.atleast_1d(
        np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
    )
    # mgcv's fit$sp holds only the FREE sp for formula-fixed terms; nampy
    # reports the full vector with the fixed value in place.
    expected_sp = np.atleast_1d(
        np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    )
    assert actual_sp.shape == (2,)
    assert expected_sp.shape == (1,)
    np.testing.assert_allclose(actual_sp[0], 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.log(actual_sp[1]), np.log(expected_sp[0]), atol=5e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
        np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
        np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
        atol=5e-6,
        rtol=1e-7,
    )
    for surface in ("link", "se_link"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][surface], dtype=np.float64),
            np.asarray(expected["predictions"][surface], dtype=np.float64),
            atol=1e-7,
            rtol=1e-7,
        )
    expected_cov = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)
    actual_cov = np.asarray(actual["fit"]["cov_bayes"], dtype=np.float64)
    scale_ref = max(float(np.max(np.abs(expected_cov))), 1e-12)
    np.testing.assert_allclose(
        actual_cov, expected_cov, atol=1e-6 * scale_ref, rtol=1e-6
    )


def _pure_noise_data(seed=421, n=180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def test_multiple_simultaneous_boundary_sps_match_mgcv():
    """Noise-only smooths with select=True all reach the sp boundary.

    Without select the cr penalty null space keeps ~1 EDF per term even at
    the boundary; the double penalty shrinks the null space too, so all four
    sps ride to the boundary and each term's EDF collapses toward zero.
    """
    data = _pure_noise_data()
    gam = GAM(
        family="gaussian",
        formula=_TWO_CR_FORMULA,
        select=True,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        _TWO_CR_FORMULA,
        "gaussian",
        "REML",
        select=True,
        optimizer="newton",
        allow_live_run=True,
    )

    # At a boundary the exact log-sp endpoint is not uniquely identified;
    # the shared boundary state (both smooths penalized to ~zero EDF) and the
    # resulting fit are.
    actual_edf = np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64)
    expected_edf = np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64)
    # Strict parity of the shrunk per-term EDF, plus heavy shrinkage of both
    # noise smooths (nominal df is 7 per term).
    np.testing.assert_allclose(actual_edf, expected_edf, atol=2e-3, rtol=1e-3)
    assert np.all(actual_edf < 0.7), actual_edf
    assert np.all(expected_edf < 0.7), expected_edf
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
        np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
        np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
        atol=1e-4,
        rtol=1e-5,
    )
    for surface in ("response", "se_link"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][surface], dtype=np.float64),
            np.asarray(expected["predictions"][surface], dtype=np.float64),
            atol=1e-6,
            rtol=1e-6,
        )


_GCV_CP_CASES = [
    # (case_id, data_factory, family, formula, resolved_method, log_sp_atol,
    #  pred_tol)
    (
        "gaussian_gcv_cp",
        _make_gaussian_data,
        "gaussian",
        _TWO_CR_FORMULA,
        "gcv",
        1e-4,
        5e-6,
    ),
    (
        "poisson_gcv_cp",
        _make_poisson_data,
        "poisson",
        _TWO_CR_FORMULA,
        "aic",
        5e-6,
        1e-7,
    ),
    (
        "negbin_est_gcv_cp",
        lambda: _make_negbin_data(),
        {"name": "negbin", "theta": 1.8, "estimate_theta": True},
        'y ~ s(x0, bs="cr", k=8)',
        "reml",
        5e-6,
        1e-8,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "data_factory", "family", "formula", "resolved_method", "log_sp_atol", "pred_tol"),
    _GCV_CP_CASES,
    ids=[case[0] for case in _GCV_CP_CASES],
)
def test_gcv_cp_method_string_resolves_per_family_and_matches_mgcv(
    case_id, data_factory, family, formula, resolved_method, log_sp_atol, pred_tol
):
    """mgcv's default "GCV.Cp" method string resolves per family class.

    Upstream: extended families force REML (mgcv/R/mgcv.r:1891-1893); known
    scale selects UBRE/Cp and unknown scale selects GCV (mgcv.r:1952-1966).
    The resolved criterion identity is asserted as an intermediate object
    before comparing the optimized endpoint against mgcv run with
    method="GCV.Cp" itself.
    """
    del case_id
    data = data_factory()
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="gcv.cp",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)

    assert gam._optim_method == resolved_method

    actual = gam.parity_snapshot(X=data, include_covariances=False)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        family,
        "GCV.Cp",
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=log_sp_atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(np.asarray(actual["fit"]["criterion_value"], dtype=np.float64)),
        float(np.asarray(expected["fit"]["criterion_value"], dtype=np.float64)),
        atol=1e-8,
        rtol=1e-8,
    )
    for surface in ("response", "link"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][surface], dtype=np.float64),
            np.asarray(expected["predictions"][surface], dtype=np.float64),
            atol=pred_tol,
            rtol=pred_tol,
        )


def test_lbfgsb_ml_reml_auto_promotes_to_outer_newton():
    """smoothing_optimizer="lbfgsb" with an exact-Hessian REML path promotes.

    The driver upgrades lbfgsb to the mgcv Newton path whenever exact
    first/second derivatives exist (driver.py), so the endpoint and the trace
    shape must be identical to an explicit outer_newton fit — Newton trace
    rows carry a Hessian payload, which L-BFGS-B rows never do.
    """
    data = _make_gaussian_data()
    kwargs = {
        "family": "gaussian",
        "formula": _TWO_CR_FORMULA,
        "optimize_smoothing": True,
        "smoothing_method": "REML",
    }
    gam_lbfgsb = GAM(smoothing_optimizer="lbfgsb", **kwargs).fit(data=data)
    gam_newton = GAM(smoothing_optimizer="outer_newton", **kwargs).fit(data=data)

    np.testing.assert_array_equal(
        np.asarray(gam_lbfgsb.smoothing_params, dtype=np.float64),
        np.asarray(gam_newton.smoothing_params, dtype=np.float64),
    )
    trace = list(getattr(gam_lbfgsb, "_optim_trace", []) or [])
    assert trace, "promoted lbfgsb fit must carry an outer-newton trace"
    assert trace[-1].get("hessian") is not None
    np.testing.assert_array_equal(
        np.asarray(
            gam_lbfgsb.fit_result().coef_full, dtype=np.float64
        ),
        np.asarray(gam_newton.fit_result().coef_full, dtype=np.float64),
    )


def test_lbfgsb_gcv_runs_scipy_branch_and_matches_mgcv_endpoint():
    """GCV has no exact-Hessian path, so lbfgsb stays genuine L-BFGS-B.

    The trace rows carry no Hessian (the L-BFGS-B signature), and the
    optimized endpoint agrees with mgcv's Newton GCV optimum on the flat GCV
    surface: identical criterion value, log-sp within the flat-region spread.
    """
    data = _make_gaussian_data()
    gam = GAM(
        family="gaussian",
        formula=_TWO_CR_FORMULA,
        optimize_smoothing=True,
        smoothing_method="GCV",
        smoothing_optimizer="lbfgsb",
    ).fit(data=data)

    trace = list(getattr(gam, "_optim_trace", []) or [])
    assert trace, "lbfgsb GCV fit must record an optimizer trace"
    assert trace[-1].get("hessian") is None

    actual = gam.parity_snapshot(X=data, include_covariances=False)
    expected = _run_mgcv_snapshot(
        data,
        _TWO_CR_FORMULA,
        "gaussian",
        "GCV.Cp",
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=1e-3,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(np.asarray(actual["fit"]["criterion_value"], dtype=np.float64)),
        float(np.asarray(expected["fit"]["criterion_value"], dtype=np.float64)),
        atol=1e-10,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def test_nb_family_alias_estimates_theta_and_matches_mgcv():
    """family="nb" is mgcv::nb(): theta estimated from a unit initialization.

    The alias must produce the identical fit to the explicit estimated-theta
    negbin spec, and the optimized endpoint must match mgcv::nb() run under
    REML (the R harness token negbin_est:1 maps to nb(theta=-1)).
    """
    data = _make_negbin_data()
    formula = 'y ~ s(x0, bs="cr", k=8)'
    kwargs = {
        "formula": formula,
        "optimize_smoothing": True,
        "smoothing_method": "REML",
        "smoothing_optimizer": "outer_newton",
    }
    gam_alias = GAM(family="nb", **kwargs).fit(data=data)
    gam_explicit = GAM(
        family={"name": "negbin", "theta": 1.0, "estimate_theta": True}, **kwargs
    ).fit(data=data)

    assert bool(gam_alias.family.estimate_theta)
    np.testing.assert_array_equal(
        np.asarray(gam_alias.smoothing_params, dtype=np.float64),
        np.asarray(gam_explicit.smoothing_params, dtype=np.float64),
    )
    assert float(gam_alias.family.theta) == float(gam_explicit.family.theta)

    actual = gam_alias.parity_snapshot(X=data, include_covariances=False)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        {"name": "negbin", "theta": 1.0, "estimate_theta": True},
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=5e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(np.asarray(actual["fit"]["criterion_value"], dtype=np.float64)),
        float(np.asarray(expected["fit"]["criterion_value"], dtype=np.float64)),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-7,
        rtol=1e-7,
    )


def test_poisson_ubre_bfgs_endpoint_matches_mgcv():
    """UBRE previously only ever met outer_newton; cross it with BFGS."""
    data = _make_poisson_data()
    gam = GAM(
        family="poisson",
        formula=_TWO_CR_FORMULA,
        optimize_smoothing=True,
        smoothing_method="UBRE",
        smoothing_optimizer="bfgs",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=False)
    expected = _run_mgcv_snapshot(
        data,
        _TWO_CR_FORMULA,
        "poisson",
        "GCV.Cp",
        optimizer="bfgs",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(np.asarray(actual["fit"]["criterion_value"], dtype=np.float64)),
        float(np.asarray(expected["fit"]["criterion_value"], dtype=np.float64)),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=1e-9,
        rtol=1e-9,
    )


def test_extended_family_explicit_gcv_ubre_methods_coerce_to_reml():
    """Extended families silently reset GCV/UBRE requests to REML like mgcv.

    mgcv/R/mgcv.r:1891-1892: for extended families any method outside
    {REML, ML, NCV} — including an explicitly requested GCV.Cp — becomes
    REML. The coerced fit must be identical to a direct REML fit.
    """
    data = _make_negbin_data()
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 1.8}
    kwargs = {
        "family": family,
        "formula": formula,
        "optimize_smoothing": True,
        "smoothing_optimizer": "outer_newton",
    }
    gam_reml = GAM(smoothing_method="REML", **kwargs).fit(data=data)
    for requested in ("UBRE", "GCV"):
        gam = GAM(smoothing_method=requested, **kwargs).fit(data=data)
        assert gam._optim_method == "reml"
        np.testing.assert_array_equal(
            np.asarray(gam.smoothing_params, dtype=np.float64),
            np.asarray(gam_reml.smoothing_params, dtype=np.float64),
        )
        np.testing.assert_array_equal(
            np.asarray(gam.fit_result().coef_full, dtype=np.float64),
            np.asarray(gam_reml.fit_result().coef_full, dtype=np.float64),
        )


def test_negbin_estimated_theta_array_offset_weights_matches_mgcv():
    """The array (non-formula) negbin joint path gets a real mgcv comparison.

    Previously smoke-only (finite theta). The array construction with
    ``offset=`` and ``sample_weight=`` must reproduce mgcv's formula-offset
    weighted nb() fit: joint (log theta, log sp) endpoint, criterion, EDF,
    deviance, and offset-aware predictions.
    """
    data = _make_negbin_data(seed=910, n=220, theta=1.4).copy()
    data["off"] = 0.05 * np.sin(np.asarray(data["x0"], dtype=np.float64))
    data["w"] = np.exp(np.linspace(-0.4, 0.4, len(data)))
    family = {"name": "negbin", "theta": 1.8, "estimate_theta": True}
    offset = data["off"].to_numpy(dtype=np.float64)
    weights = data["w"].to_numpy(dtype=np.float64)
    X = data[["x0"]].to_numpy(dtype=np.float64)

    gam = GAM(
        family=family,
        basis="cr",
        k=8,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(X, data["y"].to_numpy(dtype=np.float64), offset=offset, sample_weight=weights)

    expected = _run_mgcv_snapshot(
        data,
        'y ~ s(x0, bs="cr", k=8) + offset(off)',
        family,
        "REML",
        weights_column="w",
        optimizer="newton",
        allow_live_run=True,
    )
    result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(
        np.log(np.atleast_1d(np.asarray(result.smoothing_params, dtype=np.float64))),
        np.atleast_1d(
            np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64)
        ),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(result.criterion_value),
        float(np.asarray(expected["fit"]["criterion_value"], dtype=np.float64)),
        atol=1e-8,
        rtol=1e-10,
    )
    # theta is refined by the EFS update inside PIRLS; the endpoint agrees to
    # the update tolerance rather than machine precision.
    np.testing.assert_allclose(
        float(gam.family.theta),
        float(np.asarray(expected["fit"]["family_theta"], dtype=np.float64)),
        atol=0.0,
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        float(result.deviance),
        float(np.asarray(expected["fit"]["deviance"], dtype=np.float64)),
        atol=1e-6,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        float(result.edf_total),
        float(np.asarray(expected["fit"]["edf_total"], dtype=np.float64)),
        atol=1e-6,
        rtol=0.0,
    )
    link, se_link = gam.predict(X, type="link", return_se=True, offset=offset)
    np.testing.assert_allclose(
        np.asarray(link, dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(se_link, dtype=np.float64),
        np.asarray(expected["predictions"]["se_link"], dtype=np.float64),
        atol=1e-8,
        rtol=1e-8,
    )

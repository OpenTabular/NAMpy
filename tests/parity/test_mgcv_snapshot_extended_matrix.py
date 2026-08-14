"""Extended mgcv parity scenarios that are useful but not part of the core matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests._mgcv_snapshot_parity_shared import (
    TestAdditionalScenarioParity as _SharedTestAdditionalScenarioParity,
)
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _assert_exact_mgcv_snapshot_parity,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_binomial_data,
    _make_fs_data_4levels,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_sz_data_3x3,
    _run_mgcv_snapshot,
)


def _make_transformed_formula_data(seed=531, n=120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.75, 1.75, size=n)
    z = rng.uniform(-1.25, 1.25, size=n)
    o = rng.uniform(0.2, 1.4, size=n)
    y = (
        1.2
        + 0.35 * x**2
        + 0.45 * np.sin(1.3 * z)
        + np.log1p(o)
        + rng.normal(scale=0.05, size=n)
    )
    return pd.DataFrame({"y": y, "x": x, "z": z, "o": o})


class TestAdditionalScenarioParity:
    """
    Additional end-to-end parity scenarios that extend the requested mgcv surface beyond
    the core snapshot matrix.
    """
    test_binomial_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_select_reml_matches_mgcv
    )
    test_poisson_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_poisson_select_reml_matches_mgcv
    )
    test_gaussian_re_select_reml_matches_mgcv_exactly = (
        _SharedTestAdditionalScenarioParity.test_gaussian_re_select_reml_matches_mgcv_exactly
    )
    test_gaussian_sz_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_sz_select_reml_matches_mgcv
    )
    test_weighted_poisson_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_weighted_poisson_fixed_sp_matches_mgcv
    )
    test_weighted_binomial_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_weighted_binomial_fixed_sp_matches_mgcv
    )

    test_gaussian_fs_ps_marginal_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_fs_ps_marginal_reml_matches_mgcv
    )
    test_gaussian_fs_ps_marginal_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_fs_ps_marginal_select_reml_matches_mgcv
    )
    test_negbin_theta_0p5_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_0p5_reml_matches_mgcv
    )
    test_negbin_theta_2p0_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_2p0_reml_matches_mgcv
    )
    test_negbin_theta_0p5_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_0p5_fixed_sp_matches_mgcv
    )
    test_negbin_theta_2p0_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_negbin_theta_2p0_fixed_sp_matches_mgcv
    )
    test_binomial_probit_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_probit_fixed_sp_matches_mgcv
    )
    test_binomial_probit_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_probit_reml_matches_mgcv
    )
    test_binomial_cloglog_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_cloglog_fixed_sp_matches_mgcv
    )
    test_binomial_cloglog_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_cloglog_reml_matches_mgcv
    )
    test_gamma_inverse_link_fixed_sp_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gamma_inverse_link_fixed_sp_matches_mgcv
    )
    test_gamma_inverse_link_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gamma_inverse_link_reml_matches_mgcv
    )

    def test_gaussian_transformed_formula_fixed_sp_snapshot_matches_mgcv_exactly(self):
        """
        Verify that gaussian transformed formula fixed sp snapshot matches mgcv exactly.
        """
        data = _make_transformed_formula_data()
        formula = (
            'I(y**2) ~ I(x**2) + s(I(z**2), bs="cr", k=6, sp=0.9) + offset(log(o + 1))'
        )

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(
            data,
            formula,
            "gaussian",
            "fixed",
        )

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["coef_full"], dtype=np.float64),
            np.asarray(expected["fit"]["coef_full"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["deviance"], dtype=np.float64),
            np.asarray(expected["fit"]["deviance"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gaussian_transformed_formula_reml_snapshot_matches_mgcv_exactly(self):
        """
        Verify that gaussian transformed formula REML snapshot matches mgcv exactly.
        """
        data = _make_transformed_formula_data(seed=533, n=140)
        formula = 'I(y**2) ~ I(x**2) + s(I(z**2), bs="cr", k=6) + offset(log(o + 1))'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(
            data,
            formula,
            "gaussian",
            "REML",
        )

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64),
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64),
            atol=5e-5,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
            np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=3e-4,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
            np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["deviance"], dtype=np.float64),
            np.asarray(expected["fit"]["deviance"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["coef_full"], dtype=np.float64),
            np.asarray(expected["fit"]["coef_full"], dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        for key in ("response", "link", "se_response", "se_link"):
            np.testing.assert_allclose(
                np.asarray(actual["predictions"][key], dtype=np.float64),
                np.asarray(expected["predictions"][key], dtype=np.float64),
                atol=1e-10,
                rtol=1e-10,
            )

    def test_gamma_identity_link_fixed_sp_matches_mgcv(self):
        """Verify that gamma identity link fixed sp matches mgcv."""
        data = _make_gamma_data(seed=362, n=220)
        family = {"name": "gamma", "link": "identity"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=3e-6,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    def test_gamma_identity_link_reml_prediction_matches_mgcv_fixed_sp(self):
        """Verify that gamma identity link REML prediction matches mgcv fixed sp."""
        data = _make_gamma_data(seed=363, n=220)
        family = {"name": "gamma", "link": "identity"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-6,
            rtol=0.0,
        )

    def test_gamma_identity_link_reml_matches_mgcv(self):
        """Verify that gamma identity link REML matches mgcv."""
        data = _make_gamma_data(seed=364, n=220)
        family = {"name": "gamma", "link": "identity"}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        expected = _run_mgcv_snapshot(data, formula, family, "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
        actual = gam.parity_snapshot(X=data, include_covariances=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=3e-6,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_criterion=False,
        )

    test_gaussian_te_cc_cc_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_cc_cc_fixed_matches_mgcv
    )
    test_gaussian_ti_cc_cc_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_ti_cc_cc_fixed_matches_mgcv
    )
    test_gaussian_te_ts_cr_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_ts_cr_fixed_matches_mgcv
    )
    test_gaussian_ti_ts_cr_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_ti_ts_cr_fixed_matches_mgcv
    )
    test_gaussian_te_tp_cr_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_tp_cr_fixed_matches_mgcv
    )
    def test_gaussian_te_select_reml_matches_mgcv(self):
        """Verify that gaussian te select REML matches mgcv."""
        data = _make_gaussian_data(seed=332, n=180)
        formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-7,
            pred_rtol=0.0,
            sp_log_atol=1e-5,
        )

    def test_gamma_select_reml_matches_mgcv(self):
        """Verify that gamma select REML matches mgcv."""
        data = _make_gamma_data(seed=302, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gamma", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gamma", "REML", select=True)

        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=1e-8,
            pred_rtol=1e-8,
            edf_atol=1e-8,
            criterion_atol=1e-10,
            criterion_rtol=1e-10,
            sp_atol=3e-6,
            sp_rtol=1e-8,
            log_sp_atol=1e-6,
        )

    def test_negbin_select_reml_matches_mgcv(self):
        """Verify that negative-binomial select REML matches mgcv."""
        data = _make_negbin_data(seed=303, n=240, theta=1.0)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        family = {"name": "negbin", "theta": 1.0}

        actual = _fit_nampy_snapshot(data, formula, family, "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, family, "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-4,
            pred_rtol=0.0,
            sp_log_atol=1e-10,
            check_sp=False,
            criterion_atol=1e-3,
        )

        np.testing.assert_allclose(
            np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
            np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )

    def test_gaussian_fs_numeric_by_fixed_sp_matches_mgcv(self):
        """Verify that gaussian fs numeric by fixed sp matches mgcv."""
        rng = np.random.default_rng(381)
        n = 120
        x = rng.uniform(-1.0, 1.0, size=n)
        z = rng.uniform(0.5, 1.5, size=n)
        f = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
        shifts = {"a": 0.35, "b": -0.25, "c": 0.15}
        y = z * (np.sin(1.4 * x) + np.array([shifts[str(v)] for v in f]))
        y = y + rng.normal(0.0, 0.05, size=n)
        data = pd.DataFrame({"y": y, "x": x, "z": z, "f": f})
        formula = 'y ~ s(f, x, bs="fs", by=z, k=5, xt="cr", sp=[0.9, 0.7, 0.5])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["link"], dtype=np.float64),
            np.asarray(expected["predictions"]["link"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
            np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["deviance"], dtype=np.float64),
            np.asarray(expected["fit"]["deviance"], dtype=np.float64),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_negbin_estimated_theta_reml_matches_mgcv(self):
        """Verify that negative-binomial estimated theta REML matches mgcv."""
        data = _make_negbin_data(seed=2024, n=240, theta=1.0)
        formula = 'y ~ s(x0, bs="cr", k=8)'
        family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-6,
            pred_rtol=0.0,
            sp_log_atol=1e-2,
            criterion_atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
            np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
            atol=5e-6,
            rtol=0.0,
        )


class TestFsSzMoreFactors:
    """FS/SZ parity with more factor levels than the minimal 3-level smoke tests."""

    def test_gaussian_fs_4levels_reml_matches_mgcv(self):
        """Verify that gaussian fs 4levels REML matches mgcv."""
        data = _make_fs_data_4levels()
        formula = 'y ~ s(f, x, bs="fs", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        assert (
            actual["parity"]["criterion_view"]["criterion_backend"] == "gaussian_exact"
        )
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-3,
            pred_rtol=6e-3,
            sp_log_atol=12.0,
            criterion_atol=5.0,
        )

    def test_gaussian_sz_3x3_reml_matches_mgcv(self):
        """Verify that gaussian sz 3x3 REML matches mgcv."""
        data = _make_sz_data_3x3()
        formula = 'y ~ s(f1, f2, x, bs="sz", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        assert (
            actual["parity"]["criterion_view"]["criterion_backend"] == "gaussian_exact"
        )
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-3,
            pred_rtol=6e-3,
            sp_log_atol=18.0,
            criterion_atol=5.0,
        )


def _make_positive_gaussian_link_data(seed=741, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    mean = 1.8 + 0.25 * np.sin(1.2 * x0)
    y = mean + rng.normal(scale=0.035, size=n)
    return pd.DataFrame({"y": np.maximum(y, 0.2), "x0": x0})


def _make_positive_count_link_data(seed=742, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.2, 1.2, size=n)
    mu = 5.0 + 0.45 * np.sin(1.1 * x0)
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x0": x0})


def _make_negbin_link_data(seed=743, n=220, theta=2.0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.2, 1.2, size=n)
    mu = 4.5 + 0.5 * np.cos(1.1 * x0)
    prob = theta / (theta + mu)
    y = rng.negative_binomial(theta, prob)
    return pd.DataFrame({"y": y, "x0": x0})


def _assert_fixed_link_snapshot_close(actual, expected, *, atol=1e-7, rtol=1e-7):
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
        np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["deviance"], dtype=np.float64),
        np.asarray(expected["fit"]["deviance"], dtype=np.float64),
        atol=atol,
        rtol=rtol,
    )
    for key in ("link", "response", "se_link", "se_response"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][key], dtype=np.float64),
            np.asarray(expected["predictions"][key], dtype=np.float64),
            atol=atol,
            rtol=rtol,
        )


def test_gaussian_log_and_inverse_links_fixed_sp_match_mgcv():
    """Verify non-default Gaussian links on stable fixed-smoothing surfaces."""
    cases = [
        (
            _make_positive_gaussian_link_data(seed=751),
            {"name": "gaussian", "link": "log"},
            'y ~ s(x0, bs="cr", k=7, sp=1.4)',
        ),
        (
            _make_positive_gaussian_link_data(seed=752),
            {"name": "gaussian", "link": "inverse"},
            'y ~ s(x0, bs="cr", k=7, sp=2.0)',
        ),
    ]
    for data, family, formula in cases:
        actual = _fit_nampy_snapshot(data, formula, family, "fixed")
        expected = _run_mgcv_snapshot(data, formula, family, "fixed")
        _assert_fixed_link_snapshot_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_poisson_identity_and_sqrt_links_fixed_sp_match_mgcv():
    """Verify non-default Poisson links on stable fixed-smoothing surfaces."""
    cases = [
        (
            _make_positive_count_link_data(seed=761),
            {"name": "poisson", "link": "identity"},
            'y ~ s(x0, bs="cr", k=7, sp=3.0)',
        ),
        (
            _make_positive_count_link_data(seed=762),
            {"name": "poisson", "link": "sqrt"},
            'y ~ s(x0, bs="cr", k=7, sp=2.0)',
        ),
    ]
    for data, family, formula in cases:
        actual = _fit_nampy_snapshot(data, formula, family, "fixed")
        expected = _run_mgcv_snapshot(data, formula, family, "fixed")
        _assert_fixed_link_snapshot_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_binomial_log_and_cauchit_links_fixed_sp_match_mgcv():
    """Verify less common Binomial links on stable fixed-smoothing surfaces."""
    cases = [
        (
            _make_binomial_data(seed=771, n=240),
            {"name": "binomial", "link": "log"},
            'y ~ s(x0, bs="cr", k=7, sp=2.5)',
        ),
        (
            _make_binomial_data(seed=772, n=240),
            {"name": "binomial", "link": "cauchit"},
            'y ~ s(x0, bs="cr", k=7, sp=1.5)',
        ),
    ]
    for data, family, formula in cases:
        actual = _fit_nampy_snapshot(data, formula, family, "fixed")
        expected = _run_mgcv_snapshot(data, formula, family, "fixed")
        _assert_fixed_link_snapshot_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_negbin_identity_and_sqrt_links_fixed_theta_fixed_sp_match_mgcv():
    """Verify fixed-theta NegativeBinomial non-default links."""
    cases = [
        (
            _make_negbin_link_data(seed=781, theta=2.0),
            {"name": "negbin", "theta": 2.0, "link": "identity"},
            'y ~ s(x0, bs="cr", k=7, sp=3.0)',
        ),
        (
            _make_negbin_link_data(seed=782, theta=2.0),
            {"name": "negbin", "theta": 2.0, "link": "sqrt"},
            'y ~ s(x0, bs="cr", k=7, sp=2.0)',
        ),
    ]
    for data, family, formula in cases:
        actual = _fit_nampy_snapshot(data, formula, family, "fixed")
        expected = _run_mgcv_snapshot(data, formula, family, "fixed")
        _assert_fixed_link_snapshot_close(actual, expected, atol=2e-6, rtol=2e-6)


def _assert_optimized_link_snapshot_close(actual, expected, *, atol=2e-5, rtol=2e-5):
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
        np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["criterion_value"], dtype=np.float64),
        np.asarray(expected["fit"]["criterion_value"], dtype=np.float64),
        atol=max(atol, 1e-4),
        rtol=rtol,
    )
    for key in ("link", "response"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][key], dtype=np.float64),
            np.asarray(expected["predictions"][key], dtype=np.float64),
            atol=atol,
            rtol=rtol,
        )


def test_gaussian_log_and_inverse_links_reml_and_ml_match_mgcv():
    """Verify optimized Gaussian non-default links against mgcv."""
    cases = [
        (_make_positive_gaussian_link_data(seed=791), {"name": "gaussian", "link": "log"}, "REML"),
        (_make_positive_gaussian_link_data(seed=792), {"name": "gaussian", "link": "inverse"}, "REML"),
        (_make_positive_gaussian_link_data(seed=793), {"name": "gaussian", "link": "log"}, "ML"),
        (_make_positive_gaussian_link_data(seed=794), {"name": "gaussian", "link": "inverse"}, "ML"),
    ]
    formula = 'y ~ s(x0, bs="cr", k=7)'
    for data, family, method in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_optimized_link_snapshot_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_poisson_identity_and_sqrt_links_reml_match_mgcv():
    """Verify optimized Poisson REML non-default links against mgcv."""
    cases = [
        (_make_positive_count_link_data(seed=801), {"name": "poisson", "link": "identity"}, "REML"),
        (_make_positive_count_link_data(seed=802), {"name": "poisson", "link": "sqrt"}, "REML"),
    ]
    formula = 'y ~ s(x0, bs="cr", k=7)'
    for data, family, method in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_optimized_link_snapshot_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_poisson_identity_and_sqrt_links_ml_match_mgcv():
    """Verify optimized Poisson ML non-default links against mgcv."""
    cases = [
        (_make_positive_count_link_data(seed=803), {"name": "poisson", "link": "identity"}, "ML"),
        (_make_positive_count_link_data(seed=804), {"name": "poisson", "link": "sqrt"}, "ML"),
    ]
    formula = 'y ~ s(x0, bs="cr", k=7)'
    for data, family, method in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_optimized_link_snapshot_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_binomial_all_links_reml_match_mgcv():
    """Verify optimized Binomial REML links across the public registry surface."""
    cases = [
        (_make_binomial_data(seed=811, n=240), {"name": "binomial", "link": "logit"}, "REML"),
        (_make_binomial_data(seed=812, n=240), {"name": "binomial", "link": "probit"}, "REML"),
        (_make_binomial_data(seed=813, n=240), {"name": "binomial", "link": "cloglog"}, "REML"),
        (_make_binomial_data(seed=814, n=240), {"name": "binomial", "link": "cauchit"}, "REML"),
        (_make_binomial_data(seed=815, n=240), {"name": "binomial", "link": "log"}, "REML"),
    ]
    formula = 'y ~ s(x0, bs="cr", k=7)'
    for data, family, method in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_optimized_link_snapshot_close(actual, expected, atol=8e-5, rtol=8e-5)


def test_binomial_all_links_ml_match_mgcv():
    """Verify optimized Binomial ML non-default links against mgcv."""
    cases = [
        (_make_binomial_data(seed=816, n=240), {"name": "binomial", "link": "probit"}, "ML"),
        (_make_binomial_data(seed=817, n=240), {"name": "binomial", "link": "cloglog"}, "ML"),
        (_make_binomial_data(seed=818, n=240), {"name": "binomial", "link": "cauchit"}, "ML"),
        (_make_binomial_data(seed=819, n=240), {"name": "binomial", "link": "log"}, "ML"),
    ]
    formula = 'y ~ s(x0, bs="cr", k=7)'
    for data, family, method in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_optimized_link_snapshot_close(actual, expected, atol=8e-5, rtol=8e-5)


def test_gamma_links_fixed_match_mgcv():
    """Verify Gamma identity/log/inverse links across fixed smoothing surfaces."""
    cases = [
        (_make_gamma_data(seed=821, n=220), {"name": "gamma", "link": "log"}, "fixed", 'y ~ s(x0, bs="cr", k=7, sp=1.2)'),
        (_make_gamma_data(seed=822, n=220), {"name": "gamma", "link": "identity"}, "fixed", 'y ~ s(x0, bs="cr", k=7, sp=2.0)'),
        (_make_gamma_data(seed=823, n=220), {"name": "gamma", "link": "inverse"}, "fixed", 'y ~ s(x0, bs="cr", k=7, sp=2.0)'),
    ]
    for data, family, method, formula in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_fixed_link_snapshot_close(actual, expected, atol=7e-5, rtol=7e-5)


def test_gamma_links_reml_and_ml_match_mgcv():
    """Verify Gamma identity/log/inverse optimized surfaces against mgcv."""
    cases = [
        (_make_gamma_data(seed=824, n=220), {"name": "gamma", "link": "log"}, "REML", 'y ~ s(x0, bs="cr", k=7)'),
        (_make_gamma_data(seed=825, n=220), {"name": "gamma", "link": "identity"}, "REML", 'y ~ s(x0, bs="cr", k=7)'),
        (_make_gamma_data(seed=826, n=220), {"name": "gamma", "link": "inverse"}, "REML", 'y ~ s(x0, bs="cr", k=7)'),
        (_make_gamma_data(seed=827, n=220), {"name": "gamma", "link": "log"}, "ML", 'y ~ s(x0, bs="cr", k=7)'),
        (_make_gamma_data(seed=828, n=220), {"name": "gamma", "link": "identity"}, "ML", 'y ~ s(x0, bs="cr", k=7)'),
        (_make_gamma_data(seed=829, n=220), {"name": "gamma", "link": "inverse"}, "ML", 'y ~ s(x0, bs="cr", k=7)'),
    ]
    for data, family, method, formula in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_optimized_link_snapshot_close(actual, expected, atol=7e-5, rtol=7e-5)


def test_negbin_fixed_theta_links_reml_and_ml_match_mgcv():
    """Verify fixed-theta NegativeBinomial links on optimized surfaces."""
    cases = [
        (_make_negbin_link_data(seed=831, theta=2.0), {"name": "negbin", "theta": 2.0, "link": "log"}, "REML"),
        (_make_negbin_link_data(seed=832, theta=2.0), {"name": "negbin", "theta": 2.0, "link": "identity"}, "REML"),
        (_make_negbin_link_data(seed=833, theta=2.0), {"name": "negbin", "theta": 2.0, "link": "sqrt"}, "REML"),
        (_make_negbin_link_data(seed=834, theta=2.0), {"name": "negbin", "theta": 2.0, "link": "log"}, "ML"),
        (_make_negbin_link_data(seed=835, theta=2.0), {"name": "negbin", "theta": 2.0, "link": "identity"}, "ML"),
        (_make_negbin_link_data(seed=836, theta=2.0), {"name": "negbin", "theta": 2.0, "link": "sqrt"}, "ML"),
    ]
    formula = 'y ~ s(x0, bs="cr", k=7)'
    for data, family, method in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        _assert_optimized_link_snapshot_close(actual, expected, atol=8e-5, rtol=8e-5)


def _coverage_gaulss_data(seed=841, n=140):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    x1 = rng.uniform(-1.2, 1.2, size=n)
    mu = 0.35 * np.sin(x0) + 0.2 * x1
    sigma = np.exp(-0.35 + 0.15 * np.cos(x1))
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _coverage_gammals_data(seed=843, n=140):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.2, 1.2, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    mu = np.exp(0.25 + 0.2 * np.sin(x0) - 0.1 * x1)
    shape = np.exp(0.4 + 0.15 * np.cos(x1))
    y = rng.gamma(shape=shape, scale=mu / shape, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def test_general_family_gaulss_and_gammals_tensor_multi_smooth_predictions_match_mgcv():
    """Verify general families with tensor and multiple smooths reach snapshot parity."""

    cases = [
        (
            _coverage_gaulss_data(seed=841, n=140),
            ['y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)', '~ s(x0, x1, bs="tp", k=10)'],
            "gaulss",
            "ML",
            2e-4,
        ),
        (
            _coverage_gaulss_data(seed=842, n=140),
            ['y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5])', '~ s(x1, bs="cr", k=6)'],
            "gaulss",
            "ML",
            2e-4,
        ),
        (
            _coverage_gammals_data(seed=843, n=140),
            ['y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)', '~ s(x0, bs="cr", k=6)'],
            "gammals",
            "ML",
            3e-4,
        ),
        (
            _coverage_gammals_data(seed=844, n=140),
            ['y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5])', '~ s(x1, bs="cr", k=6)'],
            "gammals",
            "ML",
            3e-4,
        ),
    ]
    for data, formula, family, method, atol in cases:
        actual = _fit_nampy_snapshot(data, formula, family, method)
        expected = _run_mgcv_snapshot(data, formula, family, method)
        for key in ("link", "response", "se_link", "se_response"):
            np.testing.assert_allclose(
                np.asarray(actual["predictions"][key], dtype=np.float64),
                np.asarray(expected["predictions"][key], dtype=np.float64),
                atol=atol,
                rtol=atol,
            )


def _snapshot_matrix_assert(actual, expected, *, atol=1e-5):
    for key in ("response", "link"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][key], dtype=np.float64),
            np.asarray(expected["predictions"][key], dtype=np.float64),
            atol=atol,
            rtol=atol,
        )
    for key in ("edf_total", "deviance"):
        np.testing.assert_allclose(
            np.asarray(actual["fit"][key], dtype=np.float64),
            np.asarray(expected["fit"][key], dtype=np.float64),
            atol=max(atol, 1e-4),
            rtol=atol,
        )

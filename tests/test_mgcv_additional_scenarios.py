"""Extended mgcv parity scenarios that are useful but not part of the core matrix."""

from __future__ import annotations

from _mgcv_snapshot_parity_shared import (
    TestAdditionalScenarioParity as _SharedTestAdditionalScenarioParity,
)
from mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_snapshot,
    _make_gaussian_data,
    _run_mgcv_snapshot,
)


class TestAdditionalScenarioParity:
    test_binomial_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_binomial_select_reml_matches_mgcv
    )
    test_poisson_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_poisson_select_reml_matches_mgcv
    )
    test_gaussian_re_select_reml_matches_mgcv_exactly = (
        _SharedTestAdditionalScenarioParity.test_gaussian_re_select_reml_matches_mgcv_exactly
    )
    test_gaussian_fs_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_fs_select_reml_matches_mgcv
    )
    test_gaussian_sz_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_sz_select_reml_matches_mgcv
    )
    test_gaussian_mrf_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_mrf_select_reml_matches_mgcv
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

    def test_gaussian_te_select_reml_matches_mgcv(self):
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
